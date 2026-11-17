"""Native Ascend adapter for synchronous batched bonus20 inference."""
import json
import torch
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

from .batch_inference import BatchedBonusAddonDraft, VERSION
from .model import FourAnchorDFlash
from .remote_teacher import FrozenTargetInterface
from .vllm_support import ServingSettings, retained_forward_tokens


def settings(config):
    values = getattr(config.model_config.hf_config, "four_anchor_serving", None)
    if not isinstance(values, dict):
        raise ValueError("Missing four_anchor_serving settings")
    result = ServingSettings(**values)
    if result.anchor_policy != "bonus_addon" or result.draft_graph or result.profile_steps:
        raise ValueError("Batched adapter requires bonus_addon, eager drafting, profiling disabled")
    return result


class BatchedFourAnchorProposer:
    def __init__(self, vllm_config, device=None, runner=None):
        self.vllm_config, self.device, self.runner = vllm_config, device, runner
        self.settings = settings(vllm_config)
        self.metadata = self.settings.metadata()
        self.layer_ids = self.metadata["draft_config"]["target_layer_ids"]
        if self.layer_ids != sorted(self.layer_ids):
            raise ValueError("Capture layers must be sorted")
        self.model = None
        self.counters = {"extra_target_forwards": 0, "bonus_page_fallbacks": 0}
        self.batch_stats = {"draft_batches": 0, "draft_requests": 0, "max_draft_batch": 0}

    def load_model(self, _vllm_target):
        config = self.vllm_config
        target = FrozenTargetInterface(config.model_config.model, self.device, config.model_config.dtype)
        self.model = FourAnchorDFlash.load(self.settings.checkpoint, target).eval()
        self.model.requires_grad_(False)
        self.settings.apply(self.model)
        self.inference = BatchedBonusAddonDraft(self.model, config.model_config.max_model_len,
                                                config.scheduler_config.max_num_seqs)
        print(json.dumps({"event": "four_anchor_batched_loaded", "inference_version": VERSION,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "query_positions": {"round_one": 20, "round_two": 19},
            "verification_block_size": 16, "speculative_tokens": 15,
            "extra_target_forwards": 0, "draft_graph": False}), flush=True)

    @torch.inference_mode()
    def dummy_run(self, **_kwargs):
        """Reserve all request caches and profile the largest draft batch."""
        c = self.inference.cache
        n = c.capacity - 31
        names = [f"profile-{i}" for i in range(c.max_requests)]
        # Profile feature projection in the same bounded token chunks as real
        # scheduler batches; do not allocate 32 full raw target histories.
        width = self.model.fc.in_features
        for rid in names:
            features = self.model.embed_tokens.weight.new_zeros(n, width)
            c.update([(rid, 0, c.ids[0, :n], features)], names)
        self.inference.propose(names, [0] * len(names), self.settings.mode)
        c.clear()

    def propose(self, *args, **kwargs):
        raise RuntimeError("Use four_anchor.vllm_batch_ascend.BatchedFourAnchorWorker")


class BatchedFourAnchorRunner(NPUModelRunner):
    def _set_up_drafter(self):
        cfg = self.vllm_config
        spec, parallel, schedule = cfg.speculative_config, cfg.parallel_config, cfg.scheduler_config
        opts = settings(cfg)
        if spec.method != "custom_class" or spec.num_speculative_tokens != opts.num_proposals:
            raise ValueError("Require custom_class with 15 speculative tokens")
        if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size) != (1, 1, 1):
            raise ValueError("This adapter requires TP=PP=DP=1")
        if not 1 <= schedule.max_num_seqs <= 32 or schedule.async_scheduling:
            raise ValueError("Require 1..32 max_num_seqs and synchronous scheduling")
        if cfg.cache_config.enable_prefix_caching or not cfg.model_config.enforce_eager:
            raise ValueError("Require eager mode with prefix caching disabled")
        if cfg.model_config.hf_text_config.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        if cfg.model_config.quantization or cfg.lora_config or cfg.kv_transfer_config:
            raise ValueError("Quantization, LoRA and KV transfer are not supported")
        super()._set_up_drafter()
        self.use_aux_hidden_state_outputs = True

    def _get_drafter(self):
        return BatchedFourAnchorProposer(self.vllm_config, self.device, self)

    def _get_eagle3_aux_layers_from_config(self):
        return tuple(i + 1 for i in self.drafter.layer_ids)

    @torch.inference_mode()
    def propose_draft_token_ids(self, valid_sampled_token_ids, sampling_metadata, scheduler_output,
            spec_decode_metadata, spec_decode_common_attn_metadata, positions, num_scheduled_tokens,
            hidden_states, aux_hidden_states=None, sample_hidden_states=None, target_model_batch_desc=None):
        self._draft_probs = self._draft_prob_req_ids = None
        if not sampling_metadata.all_greedy:
            raise ValueError("Require temperature=0")
        reqs = list(self.input_batch.req_ids[:self.input_batch.num_reqs])
        if not isinstance(valid_sampled_token_ids, list) or len(valid_sampled_token_ids) != len(reqs):
            raise RuntimeError("Expected synchronous sampled-token lists in request order")
        if aux_hidden_states is None or len(aux_hidden_states) != len(self.drafter.layer_ids):
            raise RuntimeError("Missing target auxiliary features")
        common = spec_decode_common_attn_metadata
        starts, offsets = common.num_computed_tokens_cpu, common.query_start_loc_cpu
        if starts is None or offsets is None or starts.device.type != "cpu" or offsets.device.type != "cpu":
            raise RuntimeError("Pinned synchronous CPU query offsets/computed lengths are required")
        starts = starts[:len(reqs)].tolist()
        offsets = offsets[:len(reqs)+1].tolist()
        if offsets[0] != 0 or offsets[-1] != num_scheduled_tokens:
            raise RuntimeError("Unexpected packed target token boundaries")
        counts = [0] * len(reqs) if spec_decode_metadata is None else spec_decode_metadata.num_draft_tokens
        if len(counts) != len(reqs):
            raise RuntimeError("Draft metadata does not match request order")
        c = self.drafter.inference.cache
        c.release(getattr(scheduler_output, "finished_req_ids", ()) or ())
        c.release(getattr(scheduler_output, "preempted_req_ids", ()) or ())
        updates, selected, bonuses, row_indices = [], [], [], []
        for i, rid in enumerate(reqs):
            lo, hi = offsets[i:i+2]
            sampled, ndraft = valid_sampled_token_ids[i], int(counts[i])
            if hi <= lo:
                raise RuntimeError("Scheduled request has no target inputs")
            if sampled:
                keep = retained_forward_tokens(hi-lo, ndraft, len(sampled))
            else:
                if ndraft:
                    raise RuntimeError("Discarded sampled tokens with draft inputs are unsupported")
                keep = hi-lo  # unfinished chunked prefill; cache but do not draft
            features = torch.cat([h[lo:lo+keep] for h in aux_hidden_states], -1)
            updates.append((rid, int(starts[i]), self.input_ids.gpu[lo:lo+keep], features))
            n = int(starts[i]) + keep
            if sampled and n + 30 <= self.max_model_len:
                selected.append(rid)
                bonuses.append(sampled[-1])
                row_indices.append(i)
            if ndraft and self.drafter.settings.trace_steps:
                print(json.dumps({"event": "four_anchor_verification", "request_id": rid,
                    "proposed": ndraft, "accepted_draft": len(sampled)-1}), flush=True)
        c.update(updates, reqs)
        result = [[] for _ in reqs]
        if selected:
            tokens, _ = self.drafter.inference.propose(selected, bonuses, self.drafter.settings.mode)
            for row, values in zip(row_indices, tokens.tolist()):
                result[row] = values
            stats = self.drafter.batch_stats
            stats["draft_batches"] += 1
            stats["draft_requests"] += len(selected)
            stats["max_draft_batch"] = max(stats["max_draft_batch"], len(selected))
        return result


class BatchedFourAnchorWorker(NPUWorker):
    def four_anchor_counters(self):
        return dict(self.model_runner.drafter.counters)

    def four_anchor_batch_stats(self):
        return dict(self.model_runner.drafter.batch_stats)

    def four_anchor_reset_batch_stats(self):
        for key in self.model_runner.drafter.batch_stats:
            self.model_runner.drafter.batch_stats[key] = 0

    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError("Set VLLM_USE_V2_MODEL_RUNNER=0")
        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        self.model_runner = BatchedFourAnchorRunner(self.vllm_config, self.device)
