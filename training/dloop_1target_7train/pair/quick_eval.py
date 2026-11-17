"""A standalone short evaluation of a complete seven-rank checkpoint."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import subprocess

from .common import write_json
from .dloop_8card import validate_run, validate_checkpoint, checkpoint_step, child_env, TRAIN_DEVICES, WORLD
from .run import execute, launch_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--parity-tokens', type=int, default=32)
    parser.add_argument('--master-port', type=int, default=29848)
    args = parser.parse_args()
    if min(args.limit, args.max_new_tokens, args.parity_tokens) < 1:
        parser.error('Evaluation limits must be positive')
    run = Path(args.run_dir).resolve()
    settings = validate_run(run)
    directory = run / 'dloop'
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = directory / 'checkpoints/latest_checkpoint.txt'
        checkpoint = Path(args.checkpoint or marker.read_text().strip()).resolve()
        step = checkpoint_step(checkpoint)
        validate_checkpoint(checkpoint, step)
        from ascend_support.preflight import require_free
        from ascend_support.verify_sources import verify
        verify()
        require_free(TRAIN_DEVICES.split(','), subprocess.check_output(['npu-smi', 'info'], text=True))
        records = json.loads((run / 'gsm8k.json').read_text())['records']
        count = min(args.limit, len(records))
        if not count:
            raise ValueError('No saved evaluation prompts')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = directory / f'quick-eval-step-{step:07d}-{stamp}-{os.getpid()}'
        output.mkdir()
        metadata = dict(checkpoint=str(checkpoint), step=step, prompts=count,
                        max_new_tokens=args.max_new_tokens, parity_tokens=args.parity_tokens,
                        scope='short diagnostic; not the full GSM8K evaluation',
                        loop_gate=settings['loop_gate'], stage_cap=settings['stage_cap'])
        write_json(output / 'evaluation_config.json', metadata)
        command = launch_prefix(args.master_port, WORLD) + ['-m', 'pair.evaluate',
            '--method', 'dloop', '--run-dir', str(run), '--checkpoint', str(checkpoint),
            '--output', str(output), '--limit', str(count),
            '--max-new-tokens', str(args.max_new_tokens), '--parity-tokens', str(args.parity_tokens)]
        execute(command, child_env(TRAIN_DEVICES), output / 'eval.log')
        from .evaluate import aggregate
        summary = aggregate(output, WORLD, count)
        summary.update(metadata)
        write_json(output / 'summary.json', summary)
        print(f'Short evaluation saved: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
