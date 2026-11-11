"""Actual target forward for a pending bonus; native scheduler state is unchanged.

Adapted from the earlier materializing vLLM adapter. Only confirmed prefix KV
and the bonus are visible; the normal scheduler later overwrites that slot.
"""
import torch
from vllm.config import CUDAGraphMode
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

@torch.inference_mode()
def materialize_bonus(self, token, position, common):
    """Run ONE target token in its allocated paged-cache slot.

    The scheduler processes this token again in its next normal forward.
    That deterministic overwrite is intentional; scheduler counters are not
    advanced behind its back. Rejected slots are excluded by seq_lens.
    """
    if len(self.attn_groups) != 1:
        raise ValueError("Only a single Qwen3 KV cache group is supported")
    table = self.input_batch.block_table[0]
    if position >= int(table.num_blocks_per_row[0]) * table.block_size:
        # custom_class does not allocate extra lookahead KV blocks. At a
        # page boundary, let the scheduler take one ordinary AR step.
        return None
    positions = torch.tensor([position], device=self.device, dtype=torch.long)
    ids = torch.tensor([token], device=self.device, dtype=torch.long)
    slot = (common.block_table_tensor[0, position // table.block_size].long()
            * table.block_size + position % table.block_size).reshape(1)
    q_cpu = torch.tensor([0, 1], dtype=torch.int32)
    lengths_cpu = torch.tensor([position + 1], dtype=torch.int32)
    cm = AscendCommonAttentionMetadata(
        query_start_loc=q_cpu.to(self.device), query_start_loc_cpu=q_cpu,
        seq_lens=lengths_cpu.to(self.device), _seq_lens_cpu=lengths_cpu,
        seq_lens_cpu=lengths_cpu, seq_lens_cpu_upper_bound=lengths_cpu,
        num_computed_tokens_cpu=torch.tensor([position], dtype=torch.int32),
        num_reqs=1, num_actual_tokens=1, max_query_len=1,
        max_seq_len=position + 1, block_table_tensor=common.block_table_tensor[:1],
        slot_mapping=slot, causal=True, is_prefilling=torch.tensor([False]),
        num_input_tokens=1, actual_seq_lengths_q=[1], positions=positions,
        attn_state=AscendAttentionState.DecodeOnly, decode_token_per_req=1,
    )
    metadata = {}
    for group in self.attn_groups[0]:
        builder = group.get_metadata_builder(0)
        built = builder.build(common_prefix_len=0, common_attn_metadata=cm)
        for name in group.layer_names:
            metadata[name] = built
    with set_ascend_forward_context(
        metadata, self.vllm_config, num_tokens=1, num_actual_tokens=1,
        aclgraph_runtime_mode=CUDAGraphMode.NONE, model_instance=self.model,
    ):
        _, aux = self.model(input_ids=ids, positions=positions)
    if len(aux) != len(self.drafter.layer_ids):
        raise RuntimeError("Bonus forward returned the wrong auxiliary layers")
    return ids, torch.cat(aux, dim=-1)
