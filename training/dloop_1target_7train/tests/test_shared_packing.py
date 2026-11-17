from types import SimpleNamespace
import numpy as np
import pytest
from pair.shared_packing import epoch_batches
from pair.common import quarter_plan
from ascend_support.packed_data import TokenBudgetBatchSampler


@pytest.mark.parametrize('epoch',[0,1])
def test_identical_rank_batches_and_eval_steps(epoch):
    lengths=np.array([3072]*19+[64,127,255,513,1024,1800,2200]*23)
    cfg=SimpleNamespace(max_anchors=1536,max_sequence_length=9216,seed=42)
    own=[];native=[]
    for rank in range(4):
        mine=epoch_batches(cfg,lengths,'dataset',4,rank,epoch)
        dloop=TokenBudgetBatchSampler(lengths,9216,4,rank,seed=42,bucket_width=128,max_records=128)
        dloop.set_epoch(epoch)
        dloop=list(dloop)
        assert mine==dloop
        for pack in mine:
            assert lengths[pack].max()*len(pack)<=9216
        own.append(mine);native.append(dloop)
    mine_counts=[sum(len(p[step]) for p in own) for step in range(len(own[0]))]
    native_counts=[sum(len(p[step]) for p in native) for step in range(len(native[0]))]
    assert quarter_plan(mine_counts)==quarter_plan(native_counts)


def test_runtime_schedule_and_training_alias_both_use_shared_packing(monkeypatch):
    from four_anchor import packed_data,packed_train
    from pair.shared_packing import install
    monkeypatch.setattr(packed_data,'epoch_batches',packed_data.epoch_batches)
    monkeypatch.setattr(packed_train,'epoch_batches',packed_train.epoch_batches)
    install()
    assert packed_data.epoch_batches is epoch_batches
    assert packed_train.epoch_batches is epoch_batches
    cfg=SimpleNamespace(max_anchors=1536,max_sequence_length=9216,seed=42,epochs=1,gradient_accumulation=1)
    lengths=np.array([3072]*41)
    expected=len(TokenBudgetBatchSampler(lengths,9216,4,seed=42))
    assert packed_data.schedule(cfg,SimpleNamespace(_fingerprint='test'),lengths,4,verbose=False)==[expected]
