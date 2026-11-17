import copy
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pair.common import write_json
from pair.remote_dloop import (make_server, RemoteDLoopTarget, validate_payload,
                               request_json, install)
from pair.dloop_8card import (plan, child_env, validate_checkpoint, WORLD,
                              train_and_evaluate)
from pair.run import dloop_arguments, launch_prefix
from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
from ascend_support.packed_target import enable_hidden_only_target


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('replicas,ranks', [(1, 7), (2, 6)])
def test_target_services_exact_local_parity_and_cleanup(tmp_path, dtype, replicas, ranks):
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512, attention_dropout=0.,
        _attn_implementation='sdpa')
    model = Qwen3ForCausalLM(cfg).to(dtype).eval()
    reference = enable_hidden_only_target(HFDFlashTargetEngine(model))
    reference.set_capture_layers([0, 1])
    reference.return_last_hidden_states = True
    ids = torch.tensor([[1, 2, 3, 4], [9, 8, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    loss = mask.float()
    offsets = torch.tensor([11, 53])
    expected = reference.generate_dflash_data(ids, mask, loss, offsets)
    servers, threads = [], []
    storage = tmp_path / 'exchange'
    service = dict(model=str(tmp_path / 'model'), session='test', token='test-token', storage=str(storage))
    try:
        for i in range(replicas):
            engine = enable_hidden_only_target(HFDFlashTargetEngine(copy.deepcopy(model)))
            server = make_server('127.0.0.1', 0, engine, storage, service['token'],
                                 {'session': service['session'], 'model': service['model']})
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        service['endpoints'] = [f'http://127.0.0.1:{s.server_port}' for s in servers]
        path = tmp_path / 'service.json'
        write_json(path, service)
        for rank in range(ranks):
            client = RemoteDLoopTarget(path, service['model'], rank)
            assert client.url == service['endpoints'][rank % replicas]
            client.set_capture_layers([0, 1])
            client.return_last_hidden_states = True
            actual = client.generate_dflash_data(ids, mask, loss, offsets)
            assert torch.equal(actual.hidden_states, expected.hidden_states)
            assert torch.equal(actual.last_hidden_states, expected.last_hidden_states)
            assert actual.hidden_states.dtype == dtype
            assert actual.input_ids is ids and actual.loss_mask is loss
            assert not list(storage.iterdir())
        with pytest.raises(Exception, match='403'):
            request_json(service['endpoints'][0] + '/health', 'wrong', timeout=5)
        bad = dict(request_id='a' * 32, input_ids=[[99]], attention_mask=[[1]],
                   position_offsets=[0], capture_layer_ids=[0])
        with pytest.raises(Exception, match='500'):
            request_json(service['endpoints'][0] + '/capture', service['token'], bad, timeout=5)
        assert not list(storage.iterdir())
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


@pytest.mark.parametrize('mutation', [
    {'input_ids': [[-1, 2]]},
    {'attention_mask': [[0, 1]]},
    {'position_offsets': [30000]},
    {'request_id': '../escape'},
    {'capture_layer_ids': [9]},
    {'input_ids': [[1] * 3073], 'attention_mask': [[1] * 3073]},
])
def test_invalid_target_requests_fail(mutation):
    payload = dict(request_id='a' * 32, input_ids=[[1, 2]], attention_mask=[[1, 1]],
                   position_offsets=[0], capture_layer_ids=[0])
    payload.update(mutation)
    with pytest.raises(ValueError):
        validate_payload(payload, SimpleNamespace(vocab_size=64, num_hidden_layers=3))


def test_seven_rank_schedule_covers_data_and_preserves_anchor_budget():
    lengths = np.array([3072] * 40 + [64, 128, 256, 1024, 1800, 2200] * 39)
    sampler, targets = plan(lengths)
    packs = sampler.global_packs()
    assert len(packs) == len(sampler) * WORLD
    assert set(i for pack in packs for i in pack) == set(range(len(lengths)))
    assert [p['scheduled_fractions'] for p in targets] == [[.25], [.5], [.75], [1.]]
    assert targets[-1]['step'] == len(sampler)
    assert targets[-1]['documents_consumed'] == sum(map(len, packs))
    for pack in packs:
        assert len(pack) * max(lengths[pack]) <= 9216
    s = dict(model='model', dataset='data', epochs=1, anchors_per_rank_batch=768,
             dloop_head_chunk=512, dloop_packing_index='index')
    command = dloop_arguments(s, Path('/run'), targets[0]['step'], False)
    assert command[command.index('--num-anchors') + 1] == '768'
    assert command[command.index('--num-epochs') + 1] == '1'
    assert command[command.index('--position-offset-max') + 1] == '28672'
    assert WORLD == 7 and '--nproc_per_node=7' in launch_prefix(29847, WORLD)


def test_remote_adapter_cannot_construct_a_local_teacher(tmp_path, monkeypatch):
    import pair.remote_dloop as remote
    import ascend_support.packed_target as packed
    import specforge.inference.target_engine.dflash_target_model as target
    monkeypatch.setattr(target, 'get_dflash_target_model', target.get_dflash_target_model)
    monkeypatch.setattr(packed, 'enable_hidden_only_target', packed.enable_hidden_only_target)
    called = []
    class FakeRemote:
        def __init__(self, *args):
            called.append(args)
    monkeypatch.setattr(remote, 'RemoteDLoopTarget', FakeRemote)
    monkeypatch.setenv('RANK', '5')
    install('service.json')
    engine = target.get_dflash_target_model('target', backend='hf', device='npu', torch_dtype=torch.bfloat16)
    assert called == [('service.json', 'target', 5)]
    assert packed.enable_hidden_only_target(engine) is engine


def test_evaluation_resume_uses_all_seven_rank_states(tmp_path, monkeypatch):
    import pair.dloop_8card as job
    import pair.evaluate as evaluate
    directory = tmp_path / 'dloop'
    directory.mkdir()
    settings = dict(model='model', dataset='data', epochs=1, anchors_per_rank_batch=768,
        dloop_head_chunk=512, dloop_packing_index='index', eval_limit=2,
        dloop_targets=[dict(step=3, epoch=1, actual_fraction=.5, documents_consumed=12),
                       dict(step=6, epoch=1, actual_fraction=1., documents_consumed=24)])
    calls = []
    def execute_fake(command, env, log):
        calls.append((command, env))
        assert '--nproc_per_node=7' in command
        assert env['ASCEND_RT_VISIBLE_DEVICES'] == '1,2,3,4,5,6,7'
        if 'pair.dloop_entry' in command:
            assert env['DLOOP_REMOTE_TEACHER_CONFIG'] == '/service.json'
            stop = int(command[command.index('--max-steps') + 1])
            checkpoint = directory / f'checkpoints/epoch_0_step_{stop}'
            checkpoint.mkdir(parents=True)
            write_json(checkpoint / 'COMPLETE.json', {'world_size': 7})
            for rank in range(7):
                (checkpoint / f'training_state_rank{rank}.pt').touch()
            (checkpoint.parent / 'latest_checkpoint.txt').write_text(str(checkpoint))
        else:
            assert 'DLOOP_REMOTE_TEACHER_CONFIG' not in env
    monkeypatch.setattr(job, 'execute', execute_fake)
    monkeypatch.setattr(evaluate, 'aggregate', lambda path, world, count: {'prompts': count})
    train_and_evaluate(tmp_path, settings, '/service.json', 29847)
    assert len(calls) == 4
    assert '--resume' not in calls[0][0] and '--resume' in calls[2][0]
    train_and_evaluate(tmp_path, settings, '/service.json', 29847)
    assert len(calls) == 4  # complete phases are skipped on restart
    checkpoint = directory / 'checkpoints/epoch_0_step_6'
    write_json(checkpoint / 'COMPLETE.json', {'world_size': 6})
    with pytest.raises(ValueError, match='six-rank optimizer states cannot resume'):
        validate_checkpoint(checkpoint, 6)
    write_json(checkpoint / 'COMPLETE.json', {'world_size': 7})
    (checkpoint / 'training_state_rank6.pt').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        validate_checkpoint(checkpoint, 6)


def test_prepare_reuses_data_but_replans_for_seven_ranks(tmp_path, monkeypatch):
    from datasets import Dataset
    from safetensors.torch import save_file
    from pair.common import sha
    import pair.dloop_8card as job
    from ascend_support.tokenized_data import SavedTokenDataset
    from ascend_support.packed_data import build_index
    package = tmp_path / 'package'
    package.mkdir()
    write_json(package / 'SOURCE_COMMITS.json', {'test': True})
    write_json(package / 'PACKAGE_SHA256.json', {'SOURCE_COMMITS.json': sha(package / 'SOURCE_COMMITS.json')})
    monkeypatch.setattr(job, 'ROOT', package)
    source = tmp_path / 'prepared_pair'
    init = source / 'initialization/dloop'
    init.mkdir(parents=True)
    target = tmp_path / 'target'
    write_json(target / 'config.json', dict(model_type='qwen3', hidden_size=32,
                                          vocab_size=64, num_hidden_layers=36))
    write_json(init / 'config.json', dict(model_type='qwen3', architectures=['DFlashDraftModel'],
        block_size=16, hidden_size=32, vocab_size=64, num_hidden_layers=5,
        use_sliding_window=False, attention_dropout=0., layer_types=['full_attention']*5,
        dflash_config={'target_layer_ids': [1, 9, 17, 25, 33], 'mask_token_id': 63}))
    save_file({'test_weight': torch.ones(2)}, str(init / 'model.safetensors'))
    data = source / 'training_data'
    Dataset.from_dict({'input_ids': [[1]*3072]*84, 'loss_mask': [[1.]*3072]*84}).save_to_disk(data)
    index = build_index(SavedTokenDataset(str(data), 3072, 64), source / 'cache')
    write_json(source / 'gsm8k.json', {'records': [{'input_ids': [1], 'source_index': 0}]})
    old = dict(model=str(target), dataset=str(data), dloop_packing_index=str(index),
               gsm8k_sha256=sha(source / 'gsm8k.json'), eval_limit=1, eval_max_new_tokens=10,
               loop_gate=-.5, stage_cap=16, anchors_per_rank_batch=1536, epochs=1)
    write_json(source / 'settings.json', old)
    write_json(source / 'PREPARED_SHA256.json', {n: sha(source / n) for n in ['settings.json', 'gsm8k.json']})
    before = (source / 'settings.json').read_bytes()
    out = tmp_path / 'eight_card'
    job.prepare(source, out)
    saved = job.validate_run(out)
    assert saved['epochs'] == 1 and saved['world_size_per_job'] == 7
    assert saved['teacher_devices'] == '0' and saved['train_devices'] == '1,2,3,4,5,6,7'
    assert saved['anchors_per_rank_batch'] == 768 and saved['anchors_per_sequence'] == 256
    assert saved['dataset'] == str(data) and not (out / 'training_data').exists()
    assert saved['dloop_steps_per_epoch'] == 4
    assert [p['step'] for p in saved['dloop_targets']] == [1, 2, 3, 4]
    assert (source / 'settings.json').read_bytes() == before
    assert (source / 'gsm8k.json').read_bytes() == (out / 'gsm8k.json').read_bytes()
    job.prepare(source, out)  # idempotent; no old paired-run edits


def test_launch_starts_exactly_one_target_on_card_zero(tmp_path, monkeypatch):
    import pair.dloop_8card as job
    import pair.remote_dloop as remote
    monkeypatch.setenv('TEACHER_EXCHANGE_DIR', str(tmp_path / 'exchange'))
    write_json(tmp_path / 'initialization/dloop/config.json',
               {'hidden_size': 32, 'dflash_config': {'target_layer_ids': [0, 1]}})
    launched = []
    def fake_popen(command, **kwargs):
        launched.append((command, kwargs['env']))
        return SimpleNamespace(poll=lambda: None)
    def fake_health(url, token, **kwargs):
        service = json.loads((tmp_path / 'teacher_service.json').read_text())
        assert token == service['token']
        assert url == service['endpoints'][0] + '/health'
        return {'session': service['session']}
    class FakeClient:
        def __init__(self, path, model, replica):
            assert replica == 0
            assert len(json.loads(Path(path).read_text())['endpoints']) == 1
        def set_capture_layers(self, layers):
            assert layers == [0, 1]
            self.capture_layer_ids = layers
        def generate_dflash_data(self, ids, mask, loss, offsets):
            assert offsets.tolist() == [0, 8192]
            return SimpleNamespace(hidden_states=torch.zeros(2, 4, 64, dtype=torch.bfloat16))
    monkeypatch.setattr(job.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(remote, 'request_json', fake_health)
    monkeypatch.setattr(remote, 'RemoteDLoopTarget', FakeClient)
    children, streams = [], []
    try:
        path, storage = job.start_targets(tmp_path, {'model': 'target', 'teacher_port': 8093}, children, streams)
        assert len(children) == len(launched) == 1
        assert launched[0][1]['ASCEND_RT_VISIBLE_DEVICES'] == '0'
        assert launched[0][0][-2:] == ['--replica', '0']
        service = json.loads(path.read_text())
        assert service['endpoints'] == ['http://127.0.0.1:8093']
        assert json.loads((tmp_path / 'teacher_contract.json').read_text())['replicas'] == 1
        assert not (tmp_path / 'target_1.log').exists()
    finally:
        for stream in streams:
            stream.close()


def test_actual_dloop_training_with_remote_teacher_save_resume(tmp_path, monkeypatch):
    """Real tiny trainer/HTTP/CE/backward; only distributed FSDP is stubbed."""
    import importlib.util
    import pair.remote_dloop as remote
    import ascend_support.packed_target as packed
    import specforge.inference.target_engine.dflash_target_model as target
    import specforge.core.dflash as core
    import specforge.modeling.draft.dflash as draft_module
    from pair.common import ROOT
    monkeypatch.setattr(target, 'get_dflash_target_model', target.get_dflash_target_model)
    monkeypatch.setattr(packed, 'enable_hidden_only_target', packed.enable_hidden_only_target)
    monkeypatch.setattr(draft_module.Qwen3DFlashAttention, 'forward', draft_module.Qwen3DFlashAttention.forward)
    monkeypatch.setattr(core, 'create_dflash_sdpa_mask', core.create_dflash_sdpa_mask)
    from pair.fast_attention import install as install_attention
    install_attention()
    original_client = remote.RemoteDLoopTarget
    original_enable = packed.enable_hidden_only_target
    servers, threads = [], []
    config = tmp_path / 'service.json'
    class LazyTestClient(original_client):
        def __init__(self, ignored, model_path, rank=0):
            if not servers:
                engine = original_enable(HFDFlashTargetEngine.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device='cpu', attn_implementation='sdpa'))
                metadata = dict(model=str(Path(model_path).resolve()), session='training', token='test-token',
                                storage=str(tmp_path / 'exchange'))
                server = make_server('127.0.0.1', 0, engine, metadata['storage'], metadata['token'],
                                     {'session': metadata['session'], 'model': metadata['model']})
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                servers.append(server)
                threads.append(thread)
                metadata['endpoints'] = [f'http://127.0.0.1:{server.server_port}']
                write_json(config, metadata)
            super().__init__(config, model_path, rank)
    monkeypatch.setattr(remote, 'RemoteDLoopTarget', LazyTestClient)
    install(config)
    tests = ROOT / 'vendor/dloop/tests'
    monkeypatch.syspath_prepend(str(tests))
    spec = importlib.util.spec_from_file_location('remote_native_smoke', tests / 'test_train_smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.test_actual_upstream_trainer_save_and_resume_cpu(tmp_path, monkeypatch, 'pretrained', True, True)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)
