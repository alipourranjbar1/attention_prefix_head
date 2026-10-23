"""Ascend v1 DFlash prefix selector with optional acceptance-data collection."""

import os

import torch

from vllm.logger import init_logger
from vllm_ascend import envs
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


logger = init_logger("vllm.dflash_prefix")


def is_dflash_prefix_draft(speculative_config):
    return (
        speculative_config.method == "dflash"
        and "DFlashPrefixDraftModel"
        in (speculative_config.draft_model_config.architectures or [])
    )


class AscendDflashPrefixProposer(AscendDflashProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner=runner)

        spec = vllm_config.speculative_config

        if spec.draft_sample_method == "probabilistic":
            raise ValueError(
                "DFlashPrefix supports greedy proposals only. "
                "Set draft_sample_method=greedy."
            )

        cfg = self.draft_model_config.hf_config.dflash_config

        if self.num_speculative_tokens != cfg["prefix_block_size"] - 1:
            raise ValueError(
                "num_speculative_tokens must equal trained block_size - 1"
            )

        if self.dynamic_spec is not None:
            raise ValueError(
                "Disable dynamic_spec for the prefix-selector experiment"
            )

        self.prefix_top_k = int(cfg["prefix_top_k"])

        self._prefix_disable_selector = bool(
            cfg.get("prefix_disable_selector", False)
        )

        self._prefix_walk_backend = cfg.get(
            "prefix_walk_backend",
            "torch",
        )

        if self._prefix_walk_backend not in {"torch", "triton", "auto"}:
            raise ValueError(
                "prefix_walk_backend must be torch, triton, or auto"
            )

        if (
            cfg.get("prefix_selector_kind")
            in {"local_prefix_v2", "compiled_prefix_v3"}
            and self._prefix_walk_backend != "torch"
        ):
            raise ValueError(
                "Attention prefix selectors require "
                "prefix_walk_backend=torch"
            )

        self._prefix_fused_walk = None
        self._prefix_walk_checked = (
            self._prefix_walk_backend == "torch"
        )

        self._prefix_inference_mode = os.environ.get(
            "DFLASH_PREFIX_INFERENCE",
            "reference",
        )

        if self._prefix_inference_mode not in {
            "reference",
            "validate",
            "fast",
        }:
            raise ValueError(
                "DFLASH_PREFIX_INFERENCE must be reference, validate, or fast"
            )

        if (
            self._prefix_inference_mode != "reference"
            and (
                cfg.get("prefix_selector_kind")
                not in {"local_prefix_v2", "compiled_prefix_v3"}
                or self._prefix_walk_backend != "torch"
            )
        ):
            raise ValueError(
                "Optimized inference requires an attention prefix selector "
                "with the torch backend"
            )

        self._compiled_prefix = (
            cfg.get("prefix_selector_kind") == "compiled_prefix_v3"
        )

        self._prefix_inference_runtime = None
        self._prefix_in_dummy_run = False
        self._prefix_dummy_logged = False

        # Existing dispatch flag routes sampling through our selector.
        self.use_dflash2_selector = True

        self._prefix_anchor_indices = torch.arange(
            self.max_batch_size,
            device=device,
            dtype=torch.long,
        ) * (1 + self.num_speculative_tokens)

        # New manual collection implementation.
        self.acceptance_collector = None
        self.acceptance_pending_features = None
        self.acceptance_pending_tokens = None

        trace_directory = (
            envs.VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR
        )

        if trace_directory:
            self._initialize_acceptance_collector(
                vllm_config=vllm_config,
                trace_directory=trace_directory,
            )

    def _initialize_acceptance_collector(
        self,
        vllm_config,
        trace_directory,
    ):
        if not self._compiled_prefix:
            raise ValueError(
                "Acceptance collection requires compiled_prefix_v3."
            )

        if self._prefix_disable_selector:
            raise ValueError(
                "Acceptance collection requires the prefix selector "
                "to be enabled."
            )

        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config
        spec = vllm_config.speculative_config

        if scheduler_config.max_num_seqs != 1:
            raise ValueError(
                "Acceptance collection requires max_num_seqs=1."
            )

        if scheduler_config.async_scheduling:
            raise ValueError(
                "Acceptance collection requires synchronous scheduling."
            )

        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.data_parallel_size != 1
        ):
            raise ValueError(
                "Acceptance collection requires TP=PP=DP=1."
            )

        if not spec.enforce_eager:
            raise ValueError(
                "Acceptance collection requires an eager draft backbone. "
                "Set DRAFT_ENFORCE_EAGER=1."
            )

        from speculators.models.dflash_prefix.acceptance_collector import (
            AcceptanceCollector,
        )

        self.acceptance_collector = AcceptanceCollector(
            trace_directory=trace_directory,
            draft_checkpoint=self.draft_model_config.model,
            target_checkpoint=vllm_config.model_config.model,
            num_proposals=self.num_speculative_tokens,
        )

        logger.info(
            "Acceptance collection enabled: directory=%s, proposals=%d",
            self.acceptance_collector.directory,
            self.num_speculative_tokens,
        )

    def dummy_run(self, *args, **kwargs):
        # Startup warmup must not produce collection records or bind the
        # selector graph runtime to the startup stream.
        previous = self._prefix_in_dummy_run
        self._prefix_in_dummy_run = True

        try:
            return super().dummy_run(*args, **kwargs)
        finally:
            self._prefix_in_dummy_run = previous

    def _maybe_share_lm_head(self, model):
        if getattr(self.model, "draft_id_to_target_id", None) is not None:
            raise ValueError(
                "DFlashPrefix requires a full-vocabulary draft"
            )

        self.model.has_own_lm_head = False
        super()._maybe_share_lm_head(model)

    def compute_draft_token_ids(
        self,
        hidden_states,
        sampling_metadata=None,
    ):
        # Clear data from any previous proposal.
        self.acceptance_pending_features = None
        self.acceptance_pending_tokens = None

        collecting = (
            self.acceptance_collector is not None
            and not self._prefix_in_dummy_run
        )

        if collecting:
            if (
                sampling_metadata is None
                or not sampling_metadata.all_greedy
            ):
                raise ValueError(
                    "Acceptance collection requires target temperature=0."
                )

            if not sampling_metadata.no_penalties:
                raise ValueError(
                    "Use default repetition, frequency and presence "
                    "penalties during acceptance collection."
                )

            if (
                sampling_metadata.allowed_token_ids_mask is not None
                or sampling_metadata.bad_words_token_ids
            ):
                raise ValueError(
                    "Token allowlists and bad-word constraints are not "
                    "supported during acceptance collection."
                )

        if self._prefix_disable_selector:
            return (
                self.model.compute_unary_token_ids(hidden_states),
                None,
            )

        steps = self.num_speculative_tokens

        if hidden_states.shape[0] % steps:
            raise ValueError(
                "DFlashPrefix received an incompatible hidden-state layout"
            )

        batch_size = hidden_states.shape[0] // steps

        if batch_size > self.max_batch_size:
            raise ValueError(
                "DFlashPrefix batch exceeds the allocated anchor buffer"
            )

        if batch_size == 0:
            return self.input_ids[:0], None

        if collecting and batch_size != 1:
            raise ValueError(
                "Acceptance collection expects exactly one request."
            )

        hidden = hidden_states.reshape(
            batch_size,
            steps,
            -1,
        )

        candidates, unary = self.model.compute_candidates(
            hidden_states
        )

        candidates = candidates.reshape(
            batch_size,
            steps,
            self.prefix_top_k,
        )

        unary = unary.reshape(
            batch_size,
            steps,
            self.prefix_top_k,
        )

        anchors = self.input_ids[
            self._prefix_anchor_indices[:batch_size]
        ]

        head = self.model.model.prefix_head

        if self._prefix_in_dummy_run and (
            self._compiled_prefix
            or self._prefix_inference_mode != "reference"
        ):
            tables = head.prepare_tables(
                hidden,
                candidates,
                unary,
                anchors,
                embedding_weight=self.model.model.embed_tokens.weight,
            )

            selected = head.greedy_walk(tables)

            if not self._prefix_dummy_logged:
                logger.info(
                    "PREFIX_SELECTOR_DUMMY_EAGER "
                    "graph_capture=deferred_until_request; "
                    "startup tokens discarded; "
                    "production parity checks unchanged"
                )
                self._prefix_dummy_logged = True

            return selected.reshape(-1), None

        if (
            self._compiled_prefix
            or self._prefix_inference_mode != "reference"
        ):
            if self._prefix_inference_runtime is None:
                if self._compiled_prefix:
                    from speculators.models.dflash_prefix.compiled_inference import (
                        CompiledPrefixInferenceRuntime,
                    )

                    self._prefix_inference_runtime = (
                        CompiledPrefixInferenceRuntime(head, logger)
                    )
                else:
                    from speculators.models.dflash_prefix.inference_v2 import (
                        PrefixInferenceRuntime,
                    )

                    self._prefix_inference_runtime = (
                        PrefixInferenceRuntime(head, logger)
                    )

            if collecting:
                from speculators.models.dflash_prefix.acceptance_features import (
                    build_acceptance_features,
                )

                selected, final_scores = (
                    self._prefix_inference_runtime.select(
                        hidden,
                        candidates,
                        unary,
                        anchors,
                        embedding_weight=(
                            self.model.model.embed_tokens.weight
                        ),
                        return_scores=True,
                    )
                )

                features = build_acceptance_features(
                    hidden_states=hidden,
                    candidate_ids=candidates,
                    unary_scores=unary,
                    selected_token_ids=selected,
                    final_scores=final_scores,
                    validate=True,
                )

                # The model runner will associate these with the actual
                # submitted request before target verification.
                self.acceptance_pending_features = (
                    features.detach().clone()
                )
                self.acceptance_pending_tokens = (
                    selected.detach().clone()
                )

            else:
                selected = self._prefix_inference_runtime.select(
                    hidden,
                    candidates,
                    unary,
                    anchors,
                    embedding_weight=(
                        self.model.model.embed_tokens.weight
                    ),
                )

            return selected.reshape(-1), None

        # Existing reference path for non-compiled selectors.
        prepare = getattr(
            self.model,
            "prepare_prefix_tables",
            None,
        )

        if prepare is None:
            if getattr(head, "requires_token_embeddings", False):
                raise RuntimeError(
                    "Install the matching prefix-v2 vLLM adapter"
                )

            tables = head.prepare_tables(
                hidden,
                candidates,
                unary,
                anchors,
            )
        else:
            tables = prepare(
                hidden,
                candidates,
                unary,
                anchors,
            )

        selected = self._walk(head, tables)
        return selected.reshape(-1), None

    def _walk(self, head, tables):
        if not self._prefix_walk_checked:
            try:
                from vllm_ascend.ops.triton.spec_decode.dflash_prefix import (
                    greedy_select_prefix,
                )

                expected = head.greedy_walk(tables)
                actual = greedy_select_prefix(tables)

                if not torch.equal(expected, actual):
                    raise RuntimeError(
                        "Fused prefix walk disagrees with the torch reference"
                    )

                self._prefix_fused_walk = greedy_select_prefix
                self._prefix_walk_checked = True

                logger.info(
                    "DFlashPrefix: fused Triton walk passed "
                    "first-batch token parity"
                )

                return actual

            except Exception as exc:
                if self._prefix_walk_backend == "triton":
                    raise RuntimeError(
                        "Requested Triton prefix walk failed validation"
                    ) from exc

                logger.warning(
                    "DFlashPrefix: fused walk unavailable (%s); "
                    "using torch",
                    exc,
                )

                self._prefix_walk_checked = True

        if self._prefix_fused_walk is not None:
            return self._prefix_fused_walk(tables)

        return head.greedy_walk(tables)