"""Align unstarted runs: same batches, 512 anchors/3072 tokens, one extra epoch.

Only the prior full-attention package and this installed package are accepted.
Running/started experiments are never modified.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os

SUPPORTED_PREPARED_PACKAGE='7f2e701e1752631355fbbacde37eeb867bf41ed8dec9a912355a3924082e03c9'
ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def update(run,per_sequence=512,root=ROOT):
    run,root=Path(run).resolve(),Path(root).resolve()
    if per_sequence<1: raise ValueError('Anchors per sequence must be positive')
    if not (run/'settings.json').is_file(): raise ValueError('Complete preparation first')
    with ExitStack() as stack:
        for method in ('mine','dloop'):
            folder=run/method
            folder.mkdir(exist_ok=True)
            lock=stack.enter_context((folder/'run.lock').open('a'))
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'{method} is running; its configuration was not changed') from None
            if any(p.name!='run.lock' for p in folder.iterdir()):
                raise RuntimeError(f'{method} has already started or contains launch outputs; use a fresh experiment')
        current=root/'PACKAGE_SHA256.json'
        prior=run/'PACKAGE_SHA256.json'
        if digest(prior) not in {SUPPORTED_PREPARED_PACKAGE,digest(current)}:
            raise ValueError('Prepared run uses an unsupported package version; nothing was changed')
        for name,expected in json.loads(current.read_text()).items():
            if digest(root/name)!=expected: raise ValueError(f'Installed package was modified: {name}')
        prepared=json.loads((run/'PREPARED_SHA256.json').read_text())
        for name,expected in prepared.items():
            if digest(run/name)!=expected: raise ValueError(f'Prepared input was modified: {name}')
        settings=json.loads((run/'settings.json').read_text())
        config=json.loads((run/'mine_config.json').read_text())
        tokens=settings['token_budget_per_rank']
        if tokens!=9216 or config['max_sequence_length']!=tokens:
            raise ValueError('Expected the prepared 9216-token recipe')
        total=per_sequence*(tokens//3072)
        if total<settings['dloop_packing']['max_records_per_pack']:
            raise ValueError('This change would require rebuilding the DLoop packing plan')
        if config['max_anchors']!=settings['anchors_per_rank_batch']:
            raise ValueError('Existing method budgets disagree')
        if (total==settings['anchors_per_rank_batch'] and settings.get('anchors_per_sequence')==per_sequence
            and settings.get('packing_alignment')=='shared_length_buckets_v1'
            and digest(current)==digest(prior)):
            print(f'Already aligned: one additional epoch, {per_sequence} per 3072-token unit; {total} per rank/update',flush=True)
            return
        if settings['epochs']!=1:
            raise ValueError('This update is for the previously prepared one-epoch experiment')
        if (settings['dloop_packing'].get('bucket_width')!=128
            or settings['dloop_packing'].get('max_records_per_pack')!=128
            or settings['world_size_per_job']!=4 or config['train_data_ratio']!=1.0):
            raise ValueError('Unexpected source packing recipe')
        backup=run/'before_anchor_budget_update'
        if backup.exists():
            raise FileExistsError('A previous update backup exists; inspect it before making another change')
        backup.mkdir()
        names=['settings.json','mine_config.json','PREPARED_SHA256.json','PACKAGE_SHA256.json']
        for name in names: (backup/name).write_bytes((run/name).read_bytes())
        old=settings['anchors_per_rank_batch']
        config['max_anchors']=total
        config['epochs']=1
        config['warmup_steps']=int(settings['warmup_ratio']*settings['dloop_steps_per_epoch'])
        settings.update(anchors_per_rank_batch=total,anchors_per_sequence=per_sequence,anchor_sequence_length=3072,
                        packing_alignment='shared_length_buckets_v1',
                        mine_steps_per_epoch=[settings['dloop_steps_per_epoch']],
                        mine_targets=settings['dloop_targets'])
        write(run/'mine_config.json',config)
        write(run/'settings.json',settings)
        write(run/'PREPARED_SHA256.json',{name:digest(run/name) for name in prepared})
        temporary=prior.with_name(prior.name+'.tmp')
        temporary.write_bytes(current.read_bytes())
        temporary.replace(prior)
        record={'old_anchors_per_rank_batch':old,'anchors_per_sequence':per_sequence,
                'anchor_sequence_length':3072,'anchors_per_rank_batch':total,
                'epochs':1,'steps_per_epoch_both':settings['dloop_steps_per_epoch'],
                'eval_steps_both':[p['step'] for p in settings['dloop_targets']],
                'existing_data_and_dloop_schedule_reused':True,'prepared_run':str(run)}
        write(run/'anchor_budget_update.json',record)
        print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='/home/n84449292/a84384644/parallel_training_runs/dedup_base_pair')
    p.add_argument('--anchors-per-sequence',type=int,default=512)
    a=p.parse_args()
    update(a.run_dir,a.anchors_per_sequence)
