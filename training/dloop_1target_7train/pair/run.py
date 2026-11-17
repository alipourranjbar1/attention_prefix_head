"""Independent four-NPU job: train -> save -> GSM8K -> exact-state resume."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
from .common import ROOT, write_json, sha


def execute(command, env, log):
    print(shlex.join(command),flush=True)
    with Path(log).open('a') as stream:
        child=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,bufsize=1,start_new_session=True)
        try:
            for line in child.stdout:
                print(line,end='',flush=True)
                stream.write(line)
                stream.flush()
            status=child.wait()
            if status: raise subprocess.CalledProcessError(status,command)
        except BaseException:
            import signal
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise


def launch_prefix(port, world=4):
    return [sys.executable,'-m','torch.distributed.run','--nnodes=1',f'--nproc_per_node={world}',
            '--master_addr=127.0.0.1',f'--master_port={port}']


def dloop_arguments(settings, run, stop, resume):
    args=['--target-model-path',settings['model'],'--target-model-backend','hf',
          '--draft-init-path',str(run/'initialization/dloop'),
          '--train-data-path',settings['dataset'],'--saved-tokenized-data',
          '--chat-template','qwen','--max-length','3072','--output-dir',str(run/'dloop/checkpoints'),
          '--num-epochs',str(settings['epochs']),'--batch-size','1','--learning-rate','0.0006',
          '--warmup-ratio','0.04','--max-grad-norm','1.0','--seed','42','--attention-backend','sdpa',
          '--num-anchors',str(settings['anchors_per_rank_batch']),'--loss-type','dflash','--loss-decay-gamma','7','--dloop-max-stage','3',
          '--position-offset-max','28672','--log-interval','10','--save-interval',str(10**9),
          '--report-to','tensorboard','--dataloader-num-workers','4',
          '--activation-checkpointing','--head-loss-chunk-size',str(settings['dloop_head_chunk']),
          '--packed-token-budget','9216','--packing-index',settings['dloop_packing_index'],
          '--packing-bucket-width','128','--packing-max-records','128','--max-steps',str(stop)]
    if resume: args.append('--resume')
    return args


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method',choices=['mine','dloop'])
    p.add_argument('--run-dir',required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    run=Path(a.run_dir).resolve()
    for name,digest in json.loads((run/'PREPARED_SHA256.json').read_text()).items():
        if sha(run/name)!=digest: raise ValueError(f'Prepared configuration changed: {name}; create a new run')
    s=json.loads((run/'settings.json').read_text())
    if (s.get('packing_alignment')!='shared_length_buckets_v1'
        or s['mine_steps_per_epoch']!=[s['dloop_steps_per_epoch']]*s['epochs']
        or s['mine_targets']!=s['dloop_targets']):
        raise ValueError('The two methods must use the same prepared batch/evaluation schedule')
    if sha(run/'gsm8k.json')!=s['gsm8k_sha256']: raise ValueError('Frozen GSM8K prompts changed')
    manifest=json.loads((ROOT/'PACKAGE_SHA256.json').read_text())
    for name,digest in manifest.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Package code changed: {name}; prepare a separate experiment')
    if (run/'PACKAGE_SHA256.json').read_bytes()!=(ROOT/'PACKAGE_SHA256.json').read_bytes():
        raise ValueError('Prepared run uses a different code package')
    devices='0,1,2,3' if a.method=='mine' else '4,5,6,7'
    port=29703 if a.method=='mine' else 29747
    from ascend_support.preflight import require_free,validate_configs
    from ascend_support.verify_sources import verify
    verify()
    validate_configs(Path(s['model']),run/'initialization/dloop')
    import torch
    import torch_npu
    from four_anchor.train import TrainConfig
    from specforge.optimizer import BF16Optimizer
    from torch.utils.tensorboard import SummaryWriter
    if a.check_only:
        print(json.dumps({'method':a.method,'devices':devices,'check':'imports, pinned sources, input configs verified',
                          'anchors_per_rank_batch':s['anchors_per_rank_batch'],
                          'next_evaluations':s[a.method+'_targets']},indent=2))
        return
    directory=run/a.method
    directory.mkdir(exist_ok=True)
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy()
        env.update(ASCEND_RT_VISIBLE_DEVICES=devices,FAST_DLOOP='1' if s['fast_dloop'] else '0',
                   ATTN_BLOCK_CHUNK=str(s['attention_block_chunk']),PYTHONUNBUFFERED='1')
        env.pop('ASCEND_DEVICE_ID',None)
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
            env.pop(key,None)
        for point in s[a.method+'_targets']:
            step=point['step']
            phase=directory/f'phase-{step:07d}'
            phase.mkdir(exist_ok=True)
            if (phase/'complete.json').exists(): continue
            require_free(devices.split(','),subprocess.check_output(['npu-smi','info'],text=True))
            with socket.socket() as sock: sock.bind(('127.0.0.1',port))
            trained=phase/'trained.json'
            if not trained.exists():
                latest=directory/'checkpoints/latest_checkpoint.txt'
                resume=latest.read_text().strip() if latest.exists() else None
                recovered_step=(int(Path(resume).name.rsplit('-',1)[-1]) if a.method=='mine'
                                else int(Path(resume).name.rsplit('_step_',1)[-1])) if resume else 0
                if recovered_step>step:
                    raise ValueError('Latest checkpoint is past an unfinished evaluation; inspect the run before continuing')
                if a.method=='mine':
                    cfg=json.loads((run/'mine_config.json').read_text())
                    cfg.update(max_steps=step,resume=resume,
                               initialize_dflash=None if resume else str(run/'initialization/mine'))
                    write_json(phase/'train_config.json',cfg)
                    command=launch_prefix(port)+['-m','pair.mine_entry','train','--config',str(phase/'train_config.json')]
                else:
                    command=launch_prefix(port)+['-m','pair.dloop_entry']+dloop_arguments(s,run,step,bool(resume))
                write_json(phase/'command.json',{'argv':command,'physical_devices':devices,'progress':point})
                if recovered_step<step:
                    execute(command,env,phase/'train.log')
                checkpoint=Path(latest.read_text().strip())
                if a.method=='mine':
                    actual=int(checkpoint.name.rsplit('-',1)[-1])
                    if not (checkpoint/'trainer.pt').exists(): raise ValueError('Missing optimizer checkpoint')
                    if not all((checkpoint/f'rng_rank_{r}.pt').exists() for r in range(4)):
                        raise ValueError('Incomplete rank RNG states')
                else:
                    if not (checkpoint/'COMPLETE.json').exists(): raise ValueError('Incomplete DLoop checkpoint')
                    actual=int(checkpoint.name.split('_step_')[-1])
                if actual!=step: raise ValueError(f'Checkpoint step {actual} differs from phase target {step}')
                write_json(trained,{'checkpoint':str(checkpoint),'step':step})
            checkpoint=json.loads(trained.read_text())['checkpoint']
            evaluation=phase/'gsm8k'
            evaluation.mkdir(exist_ok=True)
            print(f'{a.method}: saved step {step}; evaluating GSM8K before resuming',flush=True)
            command=launch_prefix(port)+['-m','pair.evaluate','--method',a.method,'--run-dir',str(run),
                     '--checkpoint',checkpoint,'--output',str(evaluation)]
            execute(command,env,phase/'eval.log')
            from .evaluate import aggregate
            summary=aggregate(evaluation,4,s['eval_limit'])
            row={'method':a.method,'checkpoint':checkpoint,**point,**summary}
            write_json(phase/'complete.json',row)
            completed=[json.loads(x.read_text()) for x in sorted(directory.glob('phase-*/complete.json'))]
            write_json(directory/'results.json',completed)
            with (directory/'results.csv').open('w') as f:
                import csv
                columns=['step','epoch','actual_fraction','documents_consumed','prompts',
                         'acceptance_length_including_bonus','gsm8k_numeric_exact_match',
                         'accuracy_scored_prompts','accuracy_status','aggregate_tokens_per_second','checkpoint']
                w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
                w.writeheader();w.writerows(completed)
        print(f'{a.method}: training and all scheduled GSM8K evaluations completed. Results: {directory / "results.csv"}',flush=True)

if __name__=='__main__': main()
