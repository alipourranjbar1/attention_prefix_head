"""In-process Ascend vLLM proposer for existing FourAnchorDFlash checkpoints.

The target, its paged KV cache, scheduler and greedy rejection sampler are vLLM.
The shared two-pass drafter uses the checkpoint's PyTorch DFlash equations.
This adapter supports eager, synchronous, single-request,
single-NPU Qwen3 inference. It does not masquerade as vanilla one-anchor DFlash.
"""
import json

import torch
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

from .inference import KnownBonusDraft
from .model import FourAnchorDFlash
from .remote_teacher import FrozenTargetInterface
from .vllm_support import (
    FeatureHistory, ServingSettings, retained_forward_tokens,
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
        self.inference = KnownBonusDraft(
            self.model, self.history.ids.shape[1], self.settings.num_proposals,
            profile_steps=self.settings.profile_steps, draft_graph=self.settings.draft_graph)
        print(json.dumps({"event": "four_anchor_vllm_loaded",
                          "inference_version": "known_bonus_cached_v1",
                          "known_bonus_slots": 1, "extra_target_forwards": 0,
                          "context_projection_cache": True, "draft_graph": self.settings.draft_graph,
                          "trained_proposals": self.metadata["draft_config"]["num_proposals"],
                          "serving_proposals": self.settings.num_proposals,
                          "conditioning_anchors": 4,
                          "query_positions": 4 + 1 + self.settings.num_proposals,
                          "maximum_committed_per_verification": self.settings.num_proposals + 1,
                          "target_backend": "vllm-ascend",
                          "draft_backend": "pytorch-sdpa", "mode": self.settings.mode}), flush=True)

    @torch.inference_mode()
    def dummy_run(self, **_kwargs):
        # Include the drafter's maximum-context activation peak in vLLM's
        # memory profiling; weights and persistent history are already loaded.
        n = self.history.ids.shape[1] - 2 * self.settings.num_proposals - 1
        if n < 4:
            raise ValueError("max_model_len is too small for the requested proposal count")
        self.history.ids[:, :n].zero_()
        self.history.features[:, :n].zero_()
        profiler = self.inference.profiler
        limit, profiler.limit = profiler.limit, 0
        try:
            self.inference.cache.update("profile", 0, self.history.features[:, :n])
            if self.inference.graphs is not None:
                self.inference.graphs.prepare(self.settings.mode)
            self.inference.propose(self.history.ids[:, :n], self.history.ids[:, :1], self.settings.mode)
        finally:
            profiler.limit = limit
            self.inference.cache.request_id, self.inference.cache.length = None, 0

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
        common = spec_decode_common_attn_metadata
        computed = getattr(common, "num_computed_tokens_cpu", None)
        if computed is not None and computed.device.type == "cpu":
            start = int(computed[0])
        else:
            # Compatibility fallback: one scalar transfer, not every position.
            start = int(self._get_positions(num_scheduled_tokens)[0].item())
        request_id = self.input_batch.req_ids[0]
        history, inference = self.drafter.history, self.drafter.inference
        with inference.profiler.region("context_cache_update"):
            accepted_features = torch.cat([h[:keep] for h in aux_hidden_states], dim=-1)
            history.update(request_id, start, self.input_ids.gpu[:keep], accepted_features)
            inference.cache.update(request_id, start, accepted_features[None])
        if num_draft and self.drafter.settings.trace_steps:
            print(json.dumps({"event": "four_anchor_verification", "request_id": request_id,
                              "proposed": num_draft, "accepted_draft": len(sampled) - 1,
                              "committed_before_stop_truncation": len(sampled)}), flush=True)
        n, m = history.length, self.drafter.settings.num_proposals
        if n + 2 * m + 1 > self.max_model_len:
            inference.profiler.finish()
            return [[]]
        # The correction/bonus stays pending in the normal vLLM scheduler. It
        # enters the drafter by token embedding only, and is processed exactly
        # once by the next regular target forward alongside the M proposals.
        bonus = torch.tensor([[sampled[-1]]], device=self.device, dtype=torch.long)
        tokens, _boundary = inference.propose(history.ids[:, :n], bonus, self.drafter.settings.mode)
        with inference.profiler.region("proposal_to_host"):
            result = tokens.tolist()  # required by the pinned custom-class ABI
        inference.profiler.finish()
        return result



class FourAnchorWorker(NPUWorker):
    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError("Set VLLM_USE_V2_MODEL_RUNNER=0 for the four-anchor adapter")
        # The pinned NPUWorker hardcodes its model-runner class. Reuse its device
        # setup and workspace API, selecting our subclass without monkeypatches.
        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        self.model_runner = FourAnchorNPUModelRunner(self.vllm_config, self.device)
