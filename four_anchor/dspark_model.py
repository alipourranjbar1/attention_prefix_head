"""Four-anchor DSpark: shared two-pass backbone, Markov bias and confidence.

This is a new multi-round architecture, not a stock DSpark/vLLM checkpoint.
The backbone reuses the existing bonus_addon input isolation and detached
first-pass anchors. Both heads are the pinned upstream Speculators modules.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from speculators.models.dspark.model_definitions import MarkovHead, ConfidenceHead

from .model import DraftConfig, FourAnchorDFlash


@dataclass
class DSparkDraftConfig(DraftConfig):
    markov_rank: int = 256

    def __post_init__(self):
        super().__post_init__()
        if self.anchor_layout != "bonus_addon" or self.markov_rank < 1:
            raise ValueError("Multi-round DSpark requires bonus_addon and a positive Markov rank")


class FourAnchorDSpark(FourAnchorDFlash):
    checkpoint_architecture = "FourAnchorDSpark"
    config_class = DSparkDraftConfig

    def __init__(self, cfg, target):
        super().__init__(cfg, target)
        vocab = self.embed_tokens.num_embeddings
        self.markov_head = MarkovHead(verifier_vocab_size=vocab, draft_vocab_size=vocab,
                                      markov_rank=cfg.markov_rank, hidden_size=self.hidden_size,
                                      head_type="vanilla")
        self.confidence_head = ConfidenceHead(self.hidden_size + cfg.markov_rank)
        self.markov_head.to(device=self.embed_tokens.weight.device, dtype=self.embed_tokens.weight.dtype)
        self.confidence_head.to(device=self.embed_tokens.weight.device, dtype=self.embed_tokens.weight.dtype)

    def project(self, hidden, previous_ids):
        """One prediction per hidden state, conditioned on its previous token.

        No token at the predicted position is an input. During distillation
        previous_ids are teacher-forced; greedy_proposals uses its own tokens.
        """
        # Greedy proposals and loss evaluation share the same FP32 parameters
        # inside one BF16 autocast region. A cached cast created under no_grad
        # must not be reused by training, and checkpoint recomputation must
        # see the same cast graph as its original forward. Disable the cast
        # cache locally for these heads in both cases.
        device_type = hidden.device.type
        with torch.autocast(device_type, enabled=torch.is_autocast_enabled(device_type),
                            dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
            previous = self.markov_head.prev_embeddings(previous_ids)
            logits = self.lm_head(hidden) + self.markov_head.markov_w2(previous)
            confidence = self.confidence_head(torch.cat((hidden, previous.to(hidden.dtype)), -1))
        return logits, confidence

    @torch.no_grad()
    def greedy_proposals(self, hidden, previous_ids, *, need_token_probabilities=True):
        """Use a genuine sequential Markov chain, seeded by the known bonus
        (first round) or the final preceding anchor token (second round).
        Returns tokens, optional log greedy-token probabilities, log confidence.
        When confidence selects the boundary, vocabulary log-softmax is unused.
        Mixed-precision projection weights are cast once per proposal call,
        never cached across calls, optimizer updates, or FSDP resharding.
        """
        if hidden.ndim != 3 or previous_ids.shape != hidden.shape[:1]:
            raise ValueError("Expected [blocks, positions, hidden] and one seed token per block")
        device_type = hidden.device.type
        compute_dtype = (torch.get_autocast_dtype(device_type)
                         if torch.is_autocast_enabled(device_type) else hidden.dtype)
        def cast(value):
            return None if value is None else value.to(compute_dtype)
        lm_weight, lm_bias = cast(self.lm_head.weight), cast(self.lm_head.bias)
        markov_weight = cast(self.markov_head.markov_w2.weight)
        confidence_weight = cast(self.confidence_head.proj.weight)
        confidence_bias = cast(self.confidence_head.proj.bias)
        tokens, probability, confidence = [], [], []
        # Explicit casts avoid both repeated per-position weight conversions
        # and no-grad autocast-cache entries leaking into subsequent losses.
        with torch.autocast(device_type, enabled=False):
            # The frozen base logits do not depend on previous token IDs.
            # Batch this large projection; retain the sequential Markov chain.
            base_logits = (F.linear(hidden.to(compute_dtype), lm_weight, lm_bias)
                           if getattr(self, "proposal_base_batched", False) else None)
            for position in range(hidden.shape[1]):
                h = hidden[:, position].to(compute_dtype)
                previous = self.markov_head.prev_embeddings(previous_ids).to(compute_dtype)
                base = F.linear(h, lm_weight, lm_bias) if base_logits is None else base_logits[:, position]
                logits = base + F.linear(previous, markov_weight)
                c = F.linear(torch.cat((h, previous), -1), confidence_weight, confidence_bias).squeeze(-1)
                previous_ids = logits.argmax(-1)
                tokens.append(previous_ids)
                if need_token_probabilities:
                    probability.append(F.log_softmax(logits.float(), -1).amax(-1))
                confidence.append(F.logsigmoid(c.float()))
        return (torch.stack(tokens, 1),
                torch.stack(probability, 1) if need_token_probabilities else None,
                torch.stack(confidence, 1))
