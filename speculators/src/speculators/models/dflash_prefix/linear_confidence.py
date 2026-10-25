"""Opt-in acceptance prediction after the attention-prefix selector."""

import torch

from .acceptance_features import FEATURE_SCHEMA
from .acceptance_head import AcceptedLengthHead


class LinearConfidence:
    """Use the trained classifier's argmax as a draft-prefix length, including zero.

    Pending data is consumed at the CPU scheduler boundary, never in warmup.
    The legacy name 'linear confidence' refers to AcceptedLengthHead (an MLP).
    """

    def __init__(self, checkpoint, device, hidden_size, num_proposals,
                 max_batch_size=1, policy="argmax", budget_per_request=7):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = saved["architecture"]
        if (architecture["hidden_size"] != hidden_size
                or architecture["num_proposals"] != num_proposals):
            raise ValueError("Linear confidence checkpoint hidden size/proposal count mismatch")
        identity = saved.get("identity", {})
        if identity.get("schema") != FEATURE_SCHEMA:
            raise ValueError("Linear confidence checkpoint feature schema mismatch")
        if identity.get("decoding") != "greedy_default_penalties_v1":
            raise ValueError("Linear confidence requires a greedy acceptance checkpoint")
        self.head = AcceptedLengthHead(**architecture).float().to(device)
        self.head.load_state_dict(saved["state_dict"], strict=True)
        self.head.eval().requires_grad_(False)
        if policy not in {"argmax", "budget"}:
            raise ValueError("Linear confidence policy must be argmax or budget")
        if policy == "budget" and not 1 <= budget_per_request <= num_proposals:
            raise ValueError("Verification budget must be in 1..K")
        self.policy = policy
        self.budget_per_request = budget_per_request
        self.max_batch_size = max_batch_size
        self.host_buffer = None
        self.reset()

    def reset(self):
        self.pending = None
        self.pending_req_ids = None
        self.copy_queued = False

    def select_lengths(self, logits):
        if self.policy == "argmax":
            return logits.argmax(dim=-1)
        # The head predicts P(L=l). Tail sums give P(L >= j) directly;
        # unlike DSpark's conditional token confidences, do NOT cumprod them.
        probabilities = logits.float().softmax(dim=-1)
        survival = probabilities[:, 1:].flip(1).cumsum(1).flip(1)
        batch, proposals = survival.shape
        # DSpark-style shared budget, with at least one draft per request.
        lengths = torch.ones(batch, dtype=torch.long, device=logits.device)
        extras = batch * (self.budget_per_request - 1)
        if extras:
            indices = survival[:, 1:].reshape(-1).topk(extras, sorted=False).indices
            owners = torch.div(indices, proposals - 1, rounding_mode="floor")
            lengths.scatter_add_(0, owners, torch.ones_like(owners))
        return lengths

    def queue_copy(self, req_ids):
        """Called on the runner's token-copy stream BEFORE its completion event.

        The stream already waits for the producer; truncate is called only after
        the runner synchronizes that existing event. No extra NPU synchronization.
        """
        if self.pending is None:
            return
        req_ids = tuple(req_ids)
        if len(req_ids) != self.pending.shape[0] or len(set(req_ids)) != len(req_ids):
            raise RuntimeError("Linear confidence request rows are missing or duplicated")
        if self.copy_queued:
            raise RuntimeError("Linear confidence copy already queued")
        if self.host_buffer is None:
            self.host_buffer = torch.empty(
                (self.max_batch_size, self.head.num_proposals + 1), dtype=torch.long,
                device="cpu", pin_memory=self.pending.device.type != "cpu",
            )
        self.host_buffer[:len(req_ids)].copy_(self.pending, non_blocking=True)
        self.pending_req_ids = req_ids
        self.copy_queued = True

    @torch.inference_mode()
    def predict(self, features, selected):
        self.reset()
        logits = self.head(features)
        if (not 1 <= logits.shape[0] <= self.max_batch_size
                or selected.shape != features.shape[:2]):
            raise ValueError("Linear confidence batch or proposal shape mismatch")
        # Encode invalid logits on device; transfer once at the scheduler boundary.
        lengths = self.select_lengths(logits)
        lengths = torch.where(torch.isfinite(logits).all(dim=-1), lengths, -1)
        self.pending = torch.cat((lengths[:, None], selected), dim=1).detach()

    def truncate(self, req_ids, token_lists):
        pending_ids, copied = self.pending_req_ids, self.copy_queued
        self.reset()
        if len(req_ids) != len(token_lists) or len(set(req_ids)) != len(req_ids):
            raise RuntimeError("Invalid submitted request rows")
        if not any(token_lists):
            return token_lists
        if not copied or pending_ids is None:
            raise RuntimeError("Missing linear confidence prediction/copy for proposal")
        # The runner has already synchronized the shared draft-copy event.
        rows = self.host_buffer[:len(pending_ids)].tolist()
        by_request = dict(zip(pending_ids, rows))
        result = []
        for req_id, tokens in zip(req_ids, token_lists):
            if req_id not in by_request:
                raise RuntimeError("Unknown request in linear confidence submission")
            count, *selected = by_request[req_id]
            if count < 0 or count > len(selected):
                raise RuntimeError("Non-finite linear confidence logits or invalid length")
            if not tokens:
                result.append([])
                continue
            if tokens != selected:
                raise RuntimeError("Linear confidence prediction does not match submitted tokens")
            result.append(tokens[:count])
        return result
