"""DLoop alone: one target server, seven FSDP ranks, one extra epoch."""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time

from .common import ROOT, write_json, sha, quarter_plan
from .run import dloop_arguments, execute, launch_prefix

WORLD = 7
TRAIN_DEVICES = '1,2,3,4,5,6,7'
TARGET_DEVICES = ('0',)
MODE = 'dloop_remote_1plus7_v1'


def verify_files(directory, manifest):
    for name, digest in json.loads(Path(manifest).read_text()).items():
        if sha(Path(directory) / name) != digest:
            raise ValueError(f'Frozen input/code changed: {name}')


def plan(lengths, fraction=.25):
    from ascend_support.packed_data import TokenBudgetBatchSampler
    sampler = TokenBudgetBatchSampler(lengths, 9216, WORLD, seed=42,
                                      bucket_width=128, max_records=128)
    packs = sampler.global_packs()
    docs = [sum(map(len, packs[i:i+WORLD])) for i in range(0, len(packs), WORLD)]
    return sampler, [dict(point, epoch=1) for point in quarter_plan(docs, fraction)]


def validate_run(run):
    verify_files(ROOT, ROOT / 'PACKAGE_SHA256.json')
    verify_files(run, run / 'PREPARED_SHA256.json')
    if (run / 'PACKAGE_SHA256.json').read_bytes() != (ROOT / 'PACKAGE_SHA256.json').read_bytes():
        raise ValueError('This prepared run belongs to another code package; use a separate run directory')
    s = json.loads((run / 'settings.json').read_text())
    if (s.get('mode') != MODE or s['world_size_per_job'] != WORLD
            or s.get('teacher_devices') != ','.join(TARGET_DEVICES)
            or s.get('train_devices') != TRAIN_DEVICES
            or s['epochs'] != 1 or s['anchors_per_rank_batch'] != 768):
        raise ValueError('Expected one target, seven training ranks, one extra epoch, and 768 anchors per rank batch')
    if sha(run / 'gsm8k.json') != s['gsm8k_sha256']:
        raise ValueError('Frozen GSM8K prompts changed')
    return s


def prepare(source, run, port=8093):
    verify_files(ROOT, ROOT / 'PACKAGE_SHA256.json')
    verify_files(source, source / 'PREPARED_SHA256.json')
    old = json.loads((source / 'settings.json').read_text())
    if run == source or run.is_relative_to(source):
        raise ValueError('Use a separate DLOOP_RUN_DIR outside the original prepared run')
    if (run / 'settings.json').exists():
        s = validate_run(run)
        if s['prepared_source'] != str(source) or s['teacher_port'] != port:
            raise ValueError('Existing run uses different inputs or teacher port')
        print(json.dumps({'prepared': str(run), 'reused': True, 'eval_steps':
                          [p['step'] for p in s['dloop_targets']]}, indent=2), flush=True)
        return
    if run.exists() and any(run.iterdir()):
        raise ValueError('Output is not empty; select a new DLOOP_RUN_DIR')
    if not 1024 <= port <= 65535:
        raise ValueError('Teacher port must be in 1024..65535')
    from ascend_support.preflight import validate_configs
    from ascend_support.tokenized_data import SavedTokenDataset
    from ascend_support.packed_data import load_index
    draft = source / 'initialization/dloop'
    cfg = validate_configs(Path(old['model']), draft)
    if (cfg.get('layer_types') != ['full_attention'] * cfg['num_hidden_layers']
            or cfg.get('use_sliding_window') or cfg.get('attention_dropout', 0.) != 0.):
        raise ValueError('Prepared draft must have full attention and zero dropout')
    if sha(source / 'gsm8k.json') != old['gsm8k_sha256']:
        raise ValueError('Original prepared GSM8K changed')
    dataset = SavedTokenDataset(old['dataset'], 3072, 64)
    index = Path(old['dloop_packing_index'])
    lengths, replaced = load_index(dataset, index)
    if replaced:
        raise ValueError('Expected the common filtered data without replacement')
    sampler, targets = plan(lengths, .25)
    run.mkdir(parents=True, exist_ok=True)
    init = run / 'initialization/dloop'
    init.mkdir(parents=True)
    for src in draft.iterdir():
        if src.is_file():
            if src.suffix == '.safetensors':
                try:
                    os.link(src, init / src.name)
                except OSError:
                    shutil.copy2(src, init / src.name)
            else:
                shutil.copy2(src, init / src.name)
    shutil.copy2(source / 'gsm8k.json', run / 'gsm8k.json')
    # The existing Arrow dataset and its checked index are reused, not copied.
    s = dict(old, mode=MODE, prepared_source=str(source), epochs=1,
             world_size_per_job=WORLD, teacher_devices=','.join(TARGET_DEVICES), train_devices=TRAIN_DEVICES,
             teacher_backend='hf_remote', teacher_port=port,
             anchors_per_sequence=256, anchors_per_rank_batch=768, anchor_sequence_length=3072,
             token_budget_per_rank=9216, eval_fraction=.25, dloop_targets=targets,
             dloop_steps_per_epoch=len(sampler), dloop_packing=sampler.describe(),
             dloop_packing_index=str(index.resolve()), fast_dloop=True,
             attention_block_chunk=8, dloop_head_chunk=512)
    for key in ('mine_targets', 'mine_steps_per_epoch', 'mine_data_summary'):
        s.pop(key, None)
    write_json(run / 'settings.json', s)
    tracked = ['settings.json', 'gsm8k.json'] + [str(p.relative_to(run)) for p in init.glob('*.json')]
    write_json(run / 'PREPARED_SHA256.json', {name: sha(run / name) for name in tracked})
    for name in ('PACKAGE_SHA256.json', 'SOURCE_COMMITS.json'):
        shutil.copy2(ROOT / name, run / name)
    print(json.dumps({'prepared': str(run), 'method': 'dloop', 'training_rows': len(dataset),
        'target_npus': ','.join(TARGET_DEVICES), 'train_npus': TRAIN_DEVICES, 'extra_epochs': 1,
        'anchors_per_3072_token_unit': 256, 'anchors_per_rank_batch': 768,
        'token_budget_per_rank': 9216, 'steps_per_epoch': len(sampler),
        'gsm8k_eval_steps': [x['step'] for x in targets],
        'packing': sampler.describe()}, indent=2), flush=True)


def child_env(devices, service=None):
    env = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'ASCEND_DEVICE_ID', 'DLOOP_REMOTE_TEACHER_CONFIG'):
        env.pop(key, None)
    env.update(ASCEND_RT_VISIBLE_DEVICES=devices, PYTHONUNBUFFERED='1',
               FAST_DLOOP='1', ATTN_BLOCK_CHUNK='8')
    if service:
        env['DLOOP_REMOTE_TEACHER_CONFIG'] = str(service)
    return env


def checkpoint_step(path):
    return int(Path(path).name.rsplit('_step_', 1)[-1])


def validate_checkpoint(path, step):
    path = Path(path)
    if checkpoint_step(path) != step:
        raise ValueError('Checkpoint step differs from the evaluation milestone')
    if json.loads((path / 'COMPLETE.json').read_text())['world_size'] != WORLD:
        raise ValueError('Checkpoint is not from this seven-rank training job; six-rank optimizer states cannot resume here')
    if not all((path / f'training_state_rank{rank}.pt').is_file() for rank in range(WORLD)):
        raise ValueError('Incomplete per-rank optimizer/RNG states')


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def start_targets(run, settings, children, streams):
    from .remote_dloop import request_json
    session = secrets.token_hex(12)
    exchange_parent = Path(os.environ.get('TEACHER_EXCHANGE_DIR', str(run / 'teacher_exchange')))
    if 'TEACHER_EXCHANGE_DIR' not in os.environ and Path('/dev/shm').is_dir():
        if shutil.disk_usage('/dev/shm').free >= 4 * 1024**3:
            exchange_parent = Path('/dev/shm')
    storage = exchange_parent / f'dloop-{os.getuid()}-{session}'
    storage.mkdir(parents=True, mode=0o700)
    service = dict(model=settings['model'], session=session, token=secrets.token_hex(32),
                   storage=str(storage.resolve()),
                   endpoints=[f'http://127.0.0.1:{settings["teacher_port"] + i}'
                              for i in range(len(TARGET_DEVICES))])
    path = run / 'teacher_service.json'
    # Both the temporary file and the final service file contain the capability.
    previous_umask = os.umask(0o077)
    try:
        write_json(path, service)
    finally:
        os.umask(previous_umask)
    for replica, device in enumerate(TARGET_DEVICES):
        stream = (run / f'target_{replica}.log').open('a')
        streams.append(stream)
        command = [sys.executable, '-m', 'pair.remote_dloop', '--service-config', str(path),
                   '--replica', str(replica)]
        children.append(subprocess.Popen(command, env=child_env(device), stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True))
    print(f'Starting target server on NPU 0. Log: {run}/target_0.log', flush=True)
    deadline = time.monotonic() + int(os.environ.get('BOOT_TIMEOUT', '1800'))
    ready = set()
    last_notice = 0
    while len(ready) != len(TARGET_DEVICES):
        for i, child in enumerate(children):
            if child.poll() is not None:
                raise RuntimeError(f'Target {i} exited; inspect {run}/target_{i}.log')
            if i not in ready:
                try:
                    response = request_json(service['endpoints'][i] + '/health', service['token'], timeout=2)
                    if response.get('session') == session:
                        ready.add(i)
                except (OSError, ValueError):
                    pass
        if time.monotonic() > deadline:
            raise TimeoutError('Target startup timed out; inspect target_0.log')
        if time.monotonic() - last_notice > 30:
            print(f'Target servers ready: {len(ready)}/{len(TARGET_DEVICES)}', flush=True)
            last_notice = time.monotonic()
        if len(ready) < len(TARGET_DEVICES):
            time.sleep(2)
    # Exercise exact token/mask/offset transfer before allocating seven trainers.
    import torch
    from .remote_dloop import RemoteDLoopTarget
    cfg = json.loads((run / 'initialization/dloop/config.json').read_text())
    for replica in range(len(TARGET_DEVICES)):
        client = RemoteDLoopTarget(path, settings['model'], replica)
        client.set_capture_layers(cfg['dflash_config']['target_layer_ids'])
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        result = client.generate_dflash_data(ids, mask, mask, torch.tensor([0, 8192]))
        expected = (2, 4, cfg['hidden_size'] * len(client.capture_layer_ids))
        if tuple(result.hidden_states.shape) != expected or result.hidden_states.dtype != torch.bfloat16:
            raise ValueError('Live target contract check failed')
    write_json(run / 'teacher_contract.json', {'passed': True, 'replicas': len(TARGET_DEVICES),
               'tested_position_offsets': [0, 8192], 'features_shape': list(expected)})
    print('Target server passed token/mask/position-offset transfer checks.', flush=True)
    return path, storage


def train_and_evaluate(run, settings, service, port):
    directory = run / 'dloop'
    train_env = child_env(TRAIN_DEVICES, service)
    eval_env = child_env(TRAIN_DEVICES)
    evaluate = os.environ.get('DLOOP_EVALUATE', '0') == '1'
    for point in settings['dloop_targets']:
        step = point['step']
        phase = directory / f'phase-{step:07d}'
        phase.mkdir(exist_ok=True)
        if (phase / 'complete.json').exists():
            continue
        trained = phase / 'trained.json'
        if not trained.exists():
            latest = directory / 'checkpoints/latest_checkpoint.txt'
            resume = latest.read_text().strip() if latest.exists() else None
            recovered = checkpoint_step(resume) if resume else 0
            if recovered > step:
                raise ValueError('Latest checkpoint is beyond an unfinished evaluation')
            if resume:
                validate_checkpoint(resume, recovered)
            command = launch_prefix(port, WORLD) + ['-m', 'pair.dloop_entry'] + dloop_arguments(
                settings, run, step, bool(resume))
            write_json(phase / 'command.json', {'argv': command, 'train_devices': TRAIN_DEVICES,
                                               'target_devices': ','.join(TARGET_DEVICES), 'progress': point})
            if recovered < step:
                execute(command, train_env, phase / 'train.log')
            checkpoint = latest.read_text().strip()
            validate_checkpoint(checkpoint, step)
            write_json(trained, {'checkpoint': checkpoint, 'step': step})
        checkpoint = json.loads(trained.read_text())['checkpoint']
        validate_checkpoint(checkpoint, step)
        if not evaluate:
            print(f'Checkpoint {step} saved; scheduled evaluation disabled. Continuing training.', flush=True)
            continue
        evaluation = phase / 'gsm8k'
        evaluation.mkdir(exist_ok=True)
        print(f'Saved step {step}; pausing training for GSM8K on NPUs {TRAIN_DEVICES}.', flush=True)
        command = launch_prefix(port, WORLD) + ['-m', 'pair.evaluate', '--method', 'dloop',
            '--run-dir', str(run), '--checkpoint', checkpoint, '--output', str(evaluation)]
        execute(command, eval_env, phase / 'eval.log')
        from .evaluate import aggregate
        summary = aggregate(evaluation, WORLD, settings['eval_limit'])
        write_json(phase / 'complete.json', dict(method='dloop', checkpoint=checkpoint, **point, **summary))
        completed = [json.loads(p.read_text()) for p in sorted(directory.glob('phase-*/complete.json'))]
        write_json(directory / 'results.json', completed)
        with (directory / 'results.csv').open('w') as output:
            columns = ['step', 'epoch', 'actual_fraction', 'documents_consumed', 'prompts',
                       'acceptance_length_including_bonus', 'gsm8k_numeric_exact_match',
                       'accuracy_scored_prompts', 'accuracy_status', 'aggregate_tokens_per_second', 'checkpoint']
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(completed)
    if not evaluate:
        final_step = settings['dloop_targets'][-1]['step']
        trained = directory / f'phase-{final_step:07d}/trained.json'
        final = json.loads(trained.read_text())
        validate_checkpoint(final['checkpoint'], final_step)
        write_json(directory / 'training_complete.json', dict(final, evaluation='disabled'))


def run_training(run, port, check_only=False):
    settings = validate_run(run)
    from ascend_support.verify_sources import verify
    from ascend_support.preflight import require_free, validate_configs
    verify()
    validate_configs(Path(settings['model']), run / 'initialization/dloop')
    import torch_npu
    from torch.utils.tensorboard import SummaryWriter
    from pair.remote_dloop import RemoteDLoopTarget
    if check_only:
        print(json.dumps({'checked': 'package, inputs and imports; no NPU allocation',
            'target_npus': ','.join(TARGET_DEVICES), 'train_npus': TRAIN_DEVICES,
            'anchors_per_rank_batch': 768,
            'scheduled_evaluation_enabled': os.environ.get('DLOOP_EVALUATE', '0') == '1',
            'checkpoint_steps':
            [p['step'] for p in settings['dloop_targets']]}, indent=2))
        return
    directory = run / 'dloop'
    directory.mkdir(exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get('DLOOP_EVALUATE', '0') != '1':
            final_step = settings['dloop_targets'][-1]['step']
            trained = directory / f'phase-{final_step:07d}/trained.json'
            if trained.exists():
                final = json.loads(trained.read_text())
                validate_checkpoint(final['checkpoint'], final_step)
                print(f'Training already complete at step {final_step}: {final["checkpoint"]}', flush=True)
                return
        if all((directory / f'phase-{p["step"]:07d}/complete.json').exists()
               for p in settings['dloop_targets']):
            print(f'All training/evaluations already complete: {directory}/results.csv', flush=True)
            return
        require_free([str(i) for i in range(8)], subprocess.check_output(['npu-smi', 'info'], text=True))
        ports = [port] + [settings['teacher_port'] + i for i in range(len(TARGET_DEVICES))]
        if len(set(ports)) != len(ports):
            raise ValueError('Training and teacher ports must differ')
        for number in ports:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', number))
        children, streams = [], []
        storage = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            service, storage = start_targets(run, settings, children, streams)
            train_and_evaluate(run, settings, service, port)
        finally:
            for child in children:
                stop_child(child)
            for stream in streams:
                stream.close()
            signal.signal(signal.SIGTERM, previous)
            # A failed startup may have created its exchange directory already.
            if storage is None and (run / 'teacher_service.json').is_file():
                storage = Path(json.loads((run / 'teacher_service.json').read_text())['storage'])
            if storage is not None:
                for path in storage.glob('*.safetensors'):
                    path.unlink()
                try:
                    storage.rmdir()
                except OSError:
                    pass
        result = 'results.csv' if os.environ.get('DLOOP_EVALUATE', '0') == '1' else 'training_complete.json'
        print(f'DLoop extra epoch complete: {directory / result}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--teacher-port', type=int, default=8093)
    launch = sub.add_parser('run')
    launch.add_argument('--run-dir', required=True)
    launch.add_argument('--master-port', type=int, default=29847)
    launch.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    if args.action == 'prepare':
        prepare(Path(args.source).resolve(), run, args.teacher_port)
    else:
        run_training(run, args.master_port, args.check_only)


if __name__ == '__main__':
    main()
