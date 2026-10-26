"""Clone a resolved vanilla DFlash recipe; record every changed setting."""
import argparse
import json
from pathlib import Path

from speculators.train.config import TrainConfig


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['ar', 'dflash'], default='ar')
    p.add_argument('--endpoint', default='http://127.0.0.1:8096/v1')
    p.add_argument('--target', help='Optional relocation of the same Qwen3-4B model')
    p.add_argument('--dataset', help='Optional relocation of the same prepared dataset')
    a = p.parse_args()
    base = TrainConfig.resolve(['--config', a.baseline_config])
    b = base.flatten()
    if b['speculator_type'] != 'dflash':
        raise ValueError('Use the vanilla DFlash run.yaml, not a prefix-selector training recipe')
    if b['from_pretrained'] or b.get('prefix_backbone_init'):
        raise ValueError('Baseline must describe scratch training; checkpoint warm-start is a different experiment')
    f = dict(b)
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError(f'Use a fresh output directory: {out}')
    f.update(speculator_type='dflash_ar' if a.mode == 'ar' else 'dflash',
             sample_from_anchor=a.mode == 'ar',
             save_path=str(out / 'checkpoints'), hidden_states_path=str(out / 'hidden_states'),
             log_dir=str(out / 'logs'), run_name=out.name,
             vllm_endpoint=a.endpoint, no_resume_from_checkpoint=True)
    if a.target:
        f['verifier_name_or_path'] = str(Path(a.target).resolve())
    if a.dataset:
        f['data_path'] = str(Path(a.dataset).resolve())
    if f['block_size'] != 16:
        raise ValueError('Expected the supplied block-size-16 baseline')
    cfg = TrainConfig.from_flat(f)
    resolved = cfg.flatten()
    out.mkdir(parents=True)
    (out / 'train_config.yaml').write_text(cfg.dump_yaml())
    roundtrip = TrainConfig.resolve(['--config', str(out / 'train_config.yaml')]).flatten()
    if roundtrip != resolved:
        raise RuntimeError('Config roundtrip changed values; do not start training')
    (out / 'changes.json').write_text(json.dumps(
        {k: {'before': b.get(k), 'after': v} for k, v in resolved.items() if b.get(k) != v}, indent=2) + '\n')
    (out / 'baseline.yaml').write_text(base.dump_yaml())
    (out / 'target_layers.txt').write_text(' '.join(map(str, f['target_layer_ids'])) + '\n')
    (out / 'target.txt').write_text(f['verifier_name_or_path'] + '\n')
    print(out / 'train_config.yaml')
    print('AR: 16 proposals. Baseline DFlash block16: 15 proposals + anchor.')


if __name__ == '__main__':
    main()
