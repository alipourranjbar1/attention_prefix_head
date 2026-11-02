"""In-process Ascend vLLM proposer for existing FourAnchorDFlash checkpoints.

The target, its paged KV cache, scheduler and greedy rejection sampler are vLLM.
The shared two-pass drafter uses the checkpoint's PyTorch DFlash equations.
This first adapter deliberately supports eager, synchronous, single-request,
single-NPU Qwen3 inference. It does not masquerade as vanilla one-anchor DFlash.
"""
import json

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

from .model import FourAnchorDFlash
from .remote_teacher import FrozenTargetInterface
from .vllm_support import (
    FeatureHistory, ServingSettings, fixed_window_proposal, retained_forward_tokens,
)


def _settings(config):
    values = getattr(config.model_config.hf_config, "four_anchor_serving", None)
    if not isinstance(values, dict):
        raise ValueError("hf_overrides.four_anchor_serving must specify a checkpoint")
    return ServingSettings(**values)


class FourAnchorProposer:
    """Loaded by FourAnchorNPUModelRunner through vLLM's custom-class route."""
    def __init__(self, vllm_config, device=None, runner=None):
        self.vllm_config, self.device, self.runner = vllm_config, device, runner
        self.settings = _settings(vllm_config)
        self.metadata = self.settings.metadata()
        self.layer_ids = self.metadata["draft_config"]["target_layer_ids"]
        if self.layer_ids != sorted(self.layer_ids):
            raise ValueError("vLLM captures auxiliary states in ascending layer order")
        self.model = None
        self.history = None

    def load_model(self, _vllm_target):
        # Only the frozen embedding/head are loaded here, not an HF transformer.
        target = FrozenTargetInterface(self.vllm_config.model_config.model,
                                       self.device, self.vllm_config.model_config.dtype)
        self.model = FourAnchorDFlash.load(self.settings.checkpoint, target).eval()
        self.model.requires_grad_(False)
        self.settings.apply(self.model)
        self.history = FeatureHistory(
            self.vllm_config.model_config.max_model_len,
            len(self.layer_ids) * self.model.hidden_size,
            self.device, target.get_input_embeddings().weight.dtype,
        )
        print(json.dumps({"event": "four_anchor_vllm_loaded",
                          "trained_proposals": self.metadata["draft_config"]["num_proposals"],
                          "serving_proposals": self.settings.num_proposals,
                          "conditioning_anchors": 4,
                          "query_positions": 4 + self.settings.num_proposals,
                          "maximum_committed_per_verification": self.settings.num_proposals + 1,
                          "target_backend": "vllm-ascend",
                          "draft_backend": "pytorch-sdpa", "mode": self.settings.mode}), flush=True)

    @torch.inference_mode()
    def dummy_run(self, **_kwargs):
        # Include the drafter's maximum-context activation peak in vLLM's
        # memory profiling; weights and persistent history are already loaded.
        n = self.history.ids.shape[1] - 2 * self.settings.num_proposals
        if n < 4:
            raise ValueError("max_model_len is too small for the requested proposal count")
        self.history.ids[:, :n].zero_()
        self.history.features[:, :n].zero_()
        fixed_window_proposal(self.model, self.history.ids[:, :n],
                              self.history.features[:, :n], self.settings.mode)

    def propose(self, *args, **kwargs):
        raise RuntimeError("Use four_anchor.vllm_ascend.FourAnchorWorker, not a stock worker")


class FourAnchorNPUModelRunner(NPUModelRunner):
    def _set_up_drafter(self):
        config = self.vllm_config
        settings = _settings(config)
        p, s = config.parallel_config, config.scheduler_config
        if config.speculative_config.method != "custom_class":
            raise ValueError("Four-anchor serving requires method=custom_class")
        if config.speculative_config.num_speculative_tokens != settings.num_proposals:
            raise ValueError("Speculative token count and four_anchor.num_proposals differ")
        if (p.tensor_parallel_size, p.pipeline_parallel_size, p.data_parallel_size) != (1, 1, 1):
            raise ValueError("Four-anchor vLLM currently supports TP=PP=DP=1")
        if s.max_num_seqs != 1 or s.enable_chunked_prefill or s.async_scheduling:
            raise ValueError("Use max_num_seqs=1, no chunked prefill, no async scheduling")
        if config.cache_config.enable_prefix_caching or not config.model_config.enforce_eager:
            raise ValueError("Use enforce_eager and disable prefix caching")
        if config.model_config.hf_text_config.model_type != "qwen3":
            raise ValueError("This adapter supports dense Qwen3 only")
        if config.model_config.quantization or config.lora_config or config.kv_transfer_config:
            raise ValueError("Quantization, LoRA and KV transfer are unsupported in this adapter")
        super()._set_up_drafter()
        self.use_aux_hidden_state_outputs = True

    def _get_drafter(self):
        return FourAnchorProposer(self.vllm_config, self.device, self)

    def _get_eagle3_aux_layers_from_config(self):
        # vLLM slot 0 is the embedding; decoder output i is capture slot i+1.
        return tuple(i + 1 for i in self.drafter.layer_ids)

    @torch.inference_mode()
    def _materialize_bonus(self, token, position, common):
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

    @torch.inference_mode()
    def propose_draft_token_ids(
        self, valid_sampled_token_ids, sampling_metadata, scheduler_output,
        spec_decode_metadata, spec_decode_common_attn_metadata, positions,
        num_scheduled_tokens, hidden_states, aux_hidden_states=None,
        sample_hidden_states=None, target_model_batch_desc=None,
    ):
        self._draft_probs = self._draft_prob_req_ids = None
        if not sampling_metadata.all_greedy:
            raise ValueError("Four-anchor vLLM currently requires temperature=0")
        if self.input_batch.num_reqs != 1 or not isinstance(valid_sampled_token_ids, list):
            raise RuntimeError("Expected synchronous single-request sampling")
        sampled = valid_sampled_token_ids[0]
        if not sampled:
            raise RuntimeError("Incomplete prefill is unsupported; disable chunked prefill")
        if aux_hidden_states is None or len(aux_hidden_states) != len(self.drafter.layer_ids):
            raise RuntimeError("Target auxiliary hidden states were not captured")
        num_draft = 0 if spec_decode_metadata is None else int(spec_decode_metadata.num_draft_tokens[0])
        keep = retained_forward_tokens(num_scheduled_tokens, num_draft, len(sampled))
        # One batched D2H read for absolute positions; no per-token .item loop.
        current_positions = self._get_positions(num_scheduled_tokens)[:keep].tolist()
        start = current_positions[0]
        if current_positions != list(range(start, start + keep)):
            raise RuntimeError("Target inputs are not a contiguous single-request prefix")
        request_id = self.input_batch.req_ids[0]
        history = self.drafter.history
        history.update(request_id, start, self.input_ids.gpu[:keep],
                       torch.cat([h[:keep] for h in aux_hidden_states], dim=-1))
        if num_draft:
            print(json.dumps({"event": "four_anchor_verification", "request_id": request_id,
                              "proposed": num_draft, "accepted_draft": len(sampled) - 1,
                              "committed_before_stop_truncation": len(sampled)}), flush=True)
        n, m = history.length, self.drafter.settings.num_proposals
        # Leave room for the second pass's M query positions, even though only
        # its within-window suffix is ultimately proposed to the verifier.
        if n + 2 * m >= self.max_model_len:
            return [[]]
        materialized = self._materialize_bonus(sampled[-1], n, spec_decode_common_attn_metadata)
        if materialized is None:
            return [[]]
        bonus_ids, bonus_features = materialized
        history.update(request_id, n, bonus_ids, bonus_features)
        ids, features = history.prefix()
        tokens, _boundary = fixed_window_proposal(
            self.drafter.model, ids, features, self.drafter.settings.mode,
        )
        return tokens.tolist()


class FourAnchorWorker(NPUWorker):
    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError("Set VLLM_USE_V2_MODEL_RUNNER=0 for the four-anchor adapter")
        # The pinned NPUWorker hardcodes its model-runner class. Reuse its device
        # setup and workspace API, selecting our subclass without monkeypatches.
        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        self.model_runner = FourAnchorNPUModelRunner(self.vllm_config, self.device)
