"""Local HF target serving, preserving DLoop's per-document RoPE offsets.

Requests carry only tokens/masks/offsets. BF16 features use transient safetensors
on shared storage, never a dataset-sized hidden-state cache. The server uses
the same hidden-only HF engine as the pinned local-target trainer.
"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request
import argparse
import hashlib
import hmac
import json
import os
import time
import uuid

def request_json(url, token, payload=None, timeout=1800):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=body, headers={
        'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    # Local serving must not be redirected through a site proxy.
    from urllib.request import build_opener, ProxyHandler
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_payload(payload, config):
    import torch
    if (not isinstance(payload, dict) or not isinstance(payload.get('request_id'), str)
            or len(payload['request_id']) != 32
            or any(c not in '0123456789abcdef' for c in payload['request_id'])):
        raise ValueError('Invalid request identity')
    rows, masks = payload.get('input_ids'), payload.get('attention_mask')
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
        raise ValueError('Expected a nonempty token batch')
    batch, width = len(rows), len(rows[0])
    if not 0 < width <= 3072 or batch * width > 9216 or batch > 128:
        raise ValueError('Target input exceeds the frozen packed budget')
    if not isinstance(masks, list) or len(masks) != batch:
        raise ValueError('Invalid attention mask batch')
    for row, mask in zip(rows, masks):
        if len(row) != width or len(mask) != width:
            raise ValueError('Expected rectangular token/mask batches')
        if any(type(t) is not int or not 0 <= t < config.vocab_size for t in row):
            raise ValueError('Out-of-vocabulary token')
        if any(type(m) is not int or m not in (0, 1) for m in mask):
            raise ValueError('Expected binary masks')
        length = sum(mask)
        if length < 1 or mask != [1] * length + [0] * (width - length):
            raise ValueError('Expected nonempty, independently right-padded documents')
    offsets = payload.get('position_offsets')
    if offsets is not None and (not isinstance(offsets, list) or len(offsets) != batch
            or any(type(o) is not int or not 0 <= o <= 28672 for o in offsets)):
        raise ValueError('Invalid per-document RoPE offsets')
    layers = payload.get('capture_layer_ids')
    if (not isinstance(layers, list) or not layers or len(layers) > config.num_hidden_layers
            or len(layers) != len(set(layers))
            or any(type(i) is not int or not 0 <= i < config.num_hidden_layers for i in layers)):
        raise ValueError('Invalid capture layers')
    return (torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long),
            None if offsets is None else torch.tensor(offsets, dtype=torch.long))


def make_server(host, port, engine, storage, token, identity):
    """Serialized forwards per replica; distinct replicas run concurrently."""
    import torch
    from safetensors.torch import save_file
    storage = Path(storage).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    device = next(engine.model.parameters()).device

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, content):
            data = json.dumps(content).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            valid = hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)
            if not valid:
                self.reply(403, {'error': 'Unauthorized'})
            return valid

        def do_GET(self):
            if self.authorized():
                self.reply(200 if self.path == '/health' else 404,
                           dict(identity, ready=True) if self.path == '/health' else {})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/capture':
                self.reply(404, {})
                return
            path = None
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                ids, mask, offsets = validate_payload(payload, engine.model.config)
                path = storage / (payload['request_id'] + '.safetensors')
                if path.exists():
                    raise ValueError('Duplicate request identity')
                engine.set_capture_layers(payload['capture_layer_ids'])
                engine.return_last_hidden_states = bool(payload.get('return_last_hidden_states', False))
                started = time.perf_counter()
                with torch.inference_mode():
                    result = engine.generate_dflash_data(
                        ids.to(device), mask.to(device), torch.zeros_like(ids, device=device),
                        position_offsets=None if offsets is None else offsets.to(device))
                    if device.type in ('npu', 'cuda'):
                        getattr(torch, device.type).synchronize()
                    forwarded = time.perf_counter()
                    tensors = {'hidden_states': result.hidden_states.detach().cpu().contiguous()}
                    if result.last_hidden_states is not None:
                        tensors['last_hidden_states'] = result.last_hidden_states.detach().cpu().contiguous()
                copied = time.perf_counter()
                digest = payload_digest(payload)
                save_file(tensors, str(path), metadata={'request_sha256': digest})
                saved = time.perf_counter()
                self.reply(200, {'filename': path.name, 'request_sha256': digest,
                                'seconds': saved - started,
                                'target_forward_seconds': forwarded - started,
                                'target_to_cpu_seconds': copied - forwarded,
                                'feature_write_seconds': saved - copied})
            except Exception as exc:
                if path is not None:
                    path.unlink(missing_ok=True)
                print(json.dumps({'event': 'capture_error', 'error': str(exc)}), flush=True)
                self.reply(500, {'error': str(exc)})

    class Server(HTTPServer):
        request_queue_size = 16

    return Server((host, port), Handler)


class RemoteDLoopTarget:
    backend = 'hf_remote'

    def __init__(self, config_path, model_path, rank=0):
        self.service = json.loads(Path(config_path).read_text())
        if Path(model_path).resolve() != Path(self.service['model']).resolve():
            raise ValueError('Remote teacher model differs from the training target')
        self.url = self.service['endpoints'][rank % len(self.service['endpoints'])]
        self.storage = Path(self.service['storage']).resolve()
        self.capture_layer_ids = None
        self.return_last_hidden_states = False
        health = request_json(self.url + '/health', self.service['token'], timeout=30)
        if health.get('session') != self.service['session'] or health.get('model') != self.service['model']:
            raise ValueError('Remote teacher session/model mismatch')

    def set_capture_layers(self, layer_ids):
        self.capture_layer_ids = list(layer_ids)

    def generate_dflash_data(self, input_ids, attention_mask, loss_mask, position_offsets=None):
        import torch
        from safetensors import safe_open
        from specforge.inference.target_engine.dflash_target_model import DFlashTargetOutput
        if not self.capture_layer_ids:
            raise ValueError('Set capture layers before requesting target features')
        request_id = uuid.uuid4().hex
        payload = dict(request_id=request_id, input_ids=input_ids.cpu().tolist(),
                       attention_mask=attention_mask.long().cpu().tolist(),
                       position_offsets=None if position_offsets is None else position_offsets.cpu().tolist(),
                       capture_layer_ids=self.capture_layer_ids,
                       return_last_hidden_states=self.return_last_hidden_states)
        digest = payload_digest(payload)
        path = self.storage / (request_id + '.safetensors')
        requested = time.perf_counter()
        try:
            response = request_json(self.url + '/capture', self.service['token'], payload)
            received = time.perf_counter()
            if response.get('filename') != path.name or response.get('request_sha256') != digest:
                raise ValueError('Target response does not belong to this request')
            with safe_open(str(path), framework='pt', device='cpu') as saved:
                if saved.metadata().get('request_sha256') != digest:
                    raise ValueError('Target feature metadata mismatch')
                hidden = saved.get_tensor('hidden_states')
                last = saved.get_tensor('last_hidden_states') if self.return_last_hidden_states else None
            if hidden.shape[:2] != input_ids.shape or not torch.isfinite(hidden).all():
                raise ValueError('Invalid target features')
            # load_file/safe_open is memory-mapped; clone before deleting the file.
            hidden = hidden.clone()
            last = None if last is None else last.clone()
        finally:
            path.unlink(missing_ok=True)
        result = DFlashTargetOutput(hidden_states=hidden, input_ids=input_ids,
                                  attention_mask=attention_mask, loss_mask=loss_mask,
                                  last_hidden_states=last)
        result._pair_timing = dict(
            target_service_seconds=response['seconds'],
            queue_and_http_seconds=max(0., received - requested - response['seconds']),
            feature_read_seconds=time.perf_counter() - received,
            **{key: response[key] for key in ('target_forward_seconds', 'target_to_cpu_seconds',
                                              'feature_write_seconds') if key in response})
        return result


def install(config_path):
    import specforge.inference.target_engine.dflash_target_model as target
    import ascend_support.packed_target as packed
    original_target = target.get_dflash_target_model
    original_hidden_only = packed.enable_hidden_only_target

    def get_remote(pretrained_model_name_or_path, backend='sglang', **kwargs):
        if backend != 'hf':
            raise ValueError('Remote DLoop requires the HF target semantics')
        return RemoteDLoopTarget(config_path, pretrained_model_name_or_path, int(os.environ.get('RANK', '0')))

    target.get_dflash_target_model = get_remote
    packed.enable_hidden_only_target = lambda engine: (
        engine if isinstance(engine, RemoteDLoopTarget) else original_hidden_only(engine))
    return original_target, original_hidden_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-config', required=True)
    parser.add_argument('--replica', type=int, required=True)
    args = parser.parse_args()
    service = json.loads(Path(args.service_config).read_text())
    if not 0 <= args.replica < len(service['endpoints']):
        raise ValueError('Target replica is not present in this service configuration')
    import torch
    import torch_npu
    from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
    from ascend_support.packed_target import enable_hidden_only_target
    torch.npu.set_device(0)
    engine = HFDFlashTargetEngine.from_pretrained(
        service['model'], torch_dtype=torch.bfloat16, device='npu:0',
        attn_implementation='sdpa', local_files_only=True)
    engine = enable_hidden_only_target(engine)
    port = int(service['endpoints'][args.replica].rsplit(':', 1)[-1])
    server = make_server('127.0.0.1', port, engine, service['storage'], service['token'],
                         {'session': service['session'], 'model': service['model']})
    print(json.dumps({'event': 'target_ready', 'replica': args.replica, 'port': port}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
