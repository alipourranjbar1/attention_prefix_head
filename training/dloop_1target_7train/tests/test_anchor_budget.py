import copy
import fcntl
import json
from pathlib import Path
import pytest
import align_prepared_run as updater
from pair.common import write_json,sha
from pair.run import dloop_arguments


def prepared(tmp_path,monkeypatch):
    root=tmp_path/'code';run=tmp_path/'run'
    root.mkdir();run.mkdir()
    (root/'code.py').write_text('print("new launcher")\n')
    write_json(root/'PACKAGE_SHA256.json',{'code.py':sha(root/'code.py')})
    write_json(run/'PACKAGE_SHA256.json',{'code.py':'old-package-hash'})
    monkeypatch.setattr(updater,'SUPPORTED_PREPARED_PACKAGE',sha(run/'PACKAGE_SHA256.json'))
    settings={'model':'target','dataset':'prepared_data','epochs':1,'dloop_head_chunk':512,
              'dloop_packing_index':'index.npz','token_budget_per_rank':9216,
              'anchors_per_rank_batch':768,'dloop_packing':{'max_records_per_pack':128,'bucket_width':128},
              'world_size_per_job':4,'warmup_ratio':0.04,'dloop_steps_per_epoch':26847,'mine_steps_per_epoch':[24760],
              'mine_targets':[{'step':s,'epoch':1} for s in (6197,12392,18566,24760)],
              'dloop_targets':[{'step':s,'epoch':1} for s in (6694,13392,20140,26847)]}
    write_json(run/'settings.json',settings)
    write_json(run/'mine_config.json',{'max_sequence_length':9216,'max_anchors':768,'num_anchors':4,
                                      'train_data_ratio':1.0,'epochs':1,'warmup_steps':990})
    write_json(run/'gsm8k.json',{'records':[{'input_ids':[1,2,3]}]})
    write_json(run/'PREPARED_SHA256.json',{name:sha(run/name) for name in ('settings.json','mine_config.json','gsm8k.json')})
    return root,run,settings


def test_both_methods_get_1536_and_preserve_schedules(tmp_path,monkeypatch):
    root,run,old=prepared(tmp_path,monkeypatch)
    prompts=(run/'gsm8k.json').read_bytes()
    updater.update(run,512,root)
    new=json.loads((run/'settings.json').read_text())
    own=json.loads((run/'mine_config.json').read_text())
    assert new['anchors_per_sequence']==512 and new['anchors_per_rank_batch']==1536
    assert own['max_anchors']==1536 and own['num_anchors']==4
    args=dloop_arguments(new,run,6197,False)
    assert args[args.index('--num-anchors')+1]=='1536'
    assert new['mine_targets']==new['dloop_targets']==old['dloop_targets']
    assert new['mine_steps_per_epoch']==[new['dloop_steps_per_epoch']]==[26847]
    assert own['epochs']==1 and own['warmup_steps']==1073
    assert (run/'gsm8k.json').read_bytes()==prompts
    for name,expected in json.loads((run/'PREPARED_SHA256.json').read_text()).items():
        assert sha(run/name)==expected
    assert (run/'PACKAGE_SHA256.json').read_bytes()==(root/'PACKAGE_SHA256.json').read_bytes()
    assert json.loads((run/'before_anchor_budget_update/settings.json').read_text())['anchors_per_rank_batch']==768
    before=(run/'settings.json').read_bytes()
    updater.update(run,512,root)
    assert (run/'settings.json').read_bytes()==before


def test_running_job_is_not_modified(tmp_path,monkeypatch):
    root,run,_=prepared(tmp_path,monkeypatch)
    folder=run/'mine';folder.mkdir()
    before=(run/'settings.json').read_bytes()
    with (folder/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='is running'): updater.update(run,512,root)
    assert (run/'settings.json').read_bytes()==before


def test_started_job_is_not_modified(tmp_path,monkeypatch):
    root,run,_=prepared(tmp_path,monkeypatch)
    (run/'dloop/checkpoints').mkdir(parents=True)
    before=(run/'settings.json').read_bytes()
    with pytest.raises(RuntimeError,match='already started'): updater.update(run,512,root)
    assert (run/'settings.json').read_bytes()==before
