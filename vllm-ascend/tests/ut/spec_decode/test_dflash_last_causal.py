from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


def _proposer(eager=True):
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    names = [f"draft.layers.{i}.attn" for i in range(5)]
    proposer.model = MagicMock()
    proposer.model.get_draft_attn_causal.return_value = [False] * 4 + [True]
    proposer.model.get_draft_kv_cache_layer_names.return_value = names
    proposer.speculative_config = SimpleNamespace(enforce_eager=eager)
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
    )
    builder = MagicMock()
    group = MagicMock(layer_names=list(reversed(names)))
    group.get_metadata_builder.return_value = builder
    proposer.draft_attn_groups = [group]
    return proposer, names, builder


def test_causal_final_layer_does_not_mutate_shared_bidirectional_metadata():
    proposer, names, builder = _proposer()
    shared = SimpleNamespace(causal=False, attn_mask=None, slot_mapping=object())
    result = [{name: shared for name in names}]
    with patch.object(AscendEagleProposer, "build_draft_attn_metadata", return_value=(result, shared)):
        metadata, _ = proposer.build_draft_attn_metadata(None, 16, 16)
    assert proposer._dflash_attention_is_causal() is False
    assert all(metadata[0][name] is shared for name in names[:-1])
    assert shared.causal is False and shared.attn_mask is None
    final = metadata[0][names[-1]]
    assert final is not shared and final.causal is True
    assert final.slot_mapping is shared.slot_mapping
    assert final.attn_mask is builder.attn_mask_builder.get_attention_mask.return_value
    builder.attn_mask_builder.get_attention_mask.assert_called_once_with(True, builder.model_config)


def test_last_causal_rejects_unsupported_draft_graph():
    proposer, _, _ = _proposer(eager=False)
    with pytest.raises(ValueError, match="enforce_eager"):
        proposer._dflash_attention_is_causal()
