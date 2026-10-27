# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark confidence observation and optional verifier-prefix selection."""

import json
import os
import uuid
from pathlib import Path

import torch


class DSparkConfidence:
    """Keep prediction rows associated with requests across the existing D2H event.

    No extra device synchronization is needed when consuming predictions. Trace
    output is diagnostic and must be disabled for throughput measurements.
    """

    def __init__(self, *, mode, scheduler, max_batch_size, num_proposals, trace_directory=None, eos_token_ids=()):
        if mode not in {"observe", "cut"}:
            raise ValueError("DSpark confidence mode must be observe or cut")
        self.mode = mode
        self.scheduler = scheduler
        self.max_batch_size = max_batch_size
        self.num_proposals = num_proposals
        self.eos = set(eos_token_ids or ())
        self.host_predictions = None
        self.host_tokens = None
        self.awaiting = {}
        self.trace = None
        self.round_id = 0
        self.reset()
        if trace_directory:
            directory = Path(trace_directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"rounds_{os.getpid()}_{uuid.uuid4().hex}.jsonl"
            self.trace = path.open("x", buffering=1)

    def reset(self):
        """Discard unsubmitted tensors, retaining submitted verification records."""
        self.pending = self.pending_tokens = self.copied_req_ids = None

    @torch.inference_mode()
    def predict(self, model, hidden, draft_with_anchor, sampling_metadata):
        self.reset()
        if (
            sampling_metadata is None
            or not sampling_metadata.all_greedy
            or not sampling_metadata.no_penalties
            or sampling_metadata.allowed_token_ids_mask is not None
            or sampling_metadata.bad_words_token_ids
        ):
            raise ValueError("DSpark confidence audit requires greedy decoding and default penalties/constraints")
        batch = draft_with_anchor.shape[0]
        if (
            draft_with_anchor.shape != (batch, self.num_proposals + 1)
            or hidden.shape[0] != batch * self.num_proposals
            or not 1 <= batch <= self.max_batch_size
        ):
            raise ValueError("Confidence must receive the sampled draft hidden rows, excluding anchor/padding")
        if getattr(getattr(model, "model", None), "confidence_head", None) is None:
            raise ValueError("DSpark checkpoint has no loaded confidence head")
        probabilities = self.scheduler._compute_dspark_token_probs(model, hidden, draft_with_anchor, batch)
        selected = self.scheduler._update_from_token_probs(probabilities)
        survival = torch.cumprod(probabilities.float(), dim=1)
        # P(L=0)=1-s[0], P(L=j)=s[j-1]-s[j], P(L=K)=s[K-1].
        distribution = torch.cat((1 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]), dim=1)
        modal_length = distribution.argmax(dim=1)
        finite = torch.isfinite(probabilities).all(dim=1)
        self.pending = torch.cat(
            (
                selected[:, None].float(),
                modal_length[:, None].float(),
                survival.sum(dim=1, keepdim=True),
                finite[:, None].float(),
                probabilities.float(),
            ),
            dim=1,
        ).detach()
        self.pending_tokens = draft_with_anchor[:, 1:].detach()

    def queue_copy(self, request_ids):
        """Enqueue on the runner's existing token-copy stream, before its event."""
        if self.pending is None:
            return
        request_ids = tuple(request_ids)
        if len(request_ids) != self.pending.shape[0] or len(set(request_ids)) != len(request_ids):
            raise RuntimeError("DSpark confidence prediction/request rows disagree")
        if self.copied_req_ids is not None:
            raise RuntimeError("DSpark confidence copy already queued")
        if self.host_predictions is None:
            pinned = self.pending.device.type != "cpu"
            self.host_predictions = torch.empty(
                (self.max_batch_size, self.num_proposals + 4), dtype=torch.float32, device="cpu", pin_memory=pinned
            )
            self.host_tokens = torch.empty(
                (self.max_batch_size, self.num_proposals), dtype=torch.int64, device="cpu", pin_memory=pinned
            )
        count = len(request_ids)
        self.host_predictions[:count].copy_(self.pending, non_blocking=True)
        self.host_tokens[:count].copy_(self.pending_tokens, non_blocking=True)
        self.copied_req_ids = request_ids

    def submit(self, request_ids, token_lists):
        """Called after the runner has synchronized the existing draft-copy event."""
        copied = self.copied_req_ids
        self.reset()
        if len(request_ids) != len(token_lists) or len(set(request_ids)) != len(request_ids):
            raise RuntimeError("Invalid DSpark submitted request rows")
        if not any(token_lists):
            return token_lists
        if copied is None:
            raise RuntimeError("Missing DSpark confidence prediction or host copy")
        predictions = self.host_predictions[: len(copied)].tolist()
        selected = self.host_tokens[: len(copied)].tolist()
        by_request = dict(zip(copied, zip(predictions, selected)))
        # Synchronous, unchunked requests without a new proposal may have ended
        # or been preempted. Their stale labels must not match a future block.
        if self.trace is not None:
            for rid in set(self.awaiting) - set(request_ids):
                self._skip(rid, "abandoned_or_preempted")
        result = []
        for rid, tokens in zip(request_ids, token_lists):
            if not tokens:
                result.append([])
                continue
            if rid not in by_request:
                raise RuntimeError("Unknown request in DSpark confidence submission")
            values, predicted_tokens = by_request[rid]
            keep, predicted, expected, finite, *probabilities = values
            if finite != 1 or not 1 <= keep <= self.num_proposals:
                raise RuntimeError("Nonfinite DSpark confidence or invalid verification budget")
            if tokens != predicted_tokens:
                raise RuntimeError("DSpark confidence tokens differ from submitted proposal")
            keep = int(keep)
            submitted = tokens[:keep] if self.mode == "cut" else list(tokens)
            result.append(submitted)
            if self.trace is not None:
                if rid in self.awaiting:
                    self._skip(rid, "replaced_without_verification")
                self.round_id += 1
                self.awaiting[rid] = dict(
                    request_id=rid,
                    round_id=self.round_id,
                    mode=self.mode,
                    proposed=self.num_proposals,
                    draft_tokens=list(tokens),
                    submitted_tokens=submitted,
                    predicted_accepted=int(predicted),
                    predicted_acceptance_length=1 + int(predicted),
                    expected_accepted=expected,
                    expected_acceptance_length=1 + expected,
                    selected_verify_tokens=keep,
                    conditional_probabilities=probabilities,
                )
        return result

    def _write(self, row):
        self.trace.write(json.dumps(row, allow_nan=False) + "\n")

    def _skip(self, rid, reason):
        row = self.awaiting.pop(rid)
        row.update(status="skipped", reason=reason)
        self._write(row)

    def verified(self, request_ids, scheduled, generated, discarded=(), finished=()):
        """Record the raw verifier output before EOS/output-budget trimming.

        Full acceptance of a shortened block is right-censored. If the bonus
        exposes the next mismatch or confirms the last maximum-block proposal,
        the boundary is known exactly. Otherwise the full label is censored.
        """
        if self.trace is None:
            return
        if len(request_ids) != len(generated):
            raise RuntimeError("DSpark verification request/output rows disagree")
        discarded = {int(i) for i in discarded}
        for index, rid in enumerate(request_ids):
            row = self.awaiting.get(rid)
            if row is None:
                continue
            tokens = list(scheduled.get(rid, []))
            if index in discarded or not tokens:
                self._skip(rid, "discarded_or_no_verification")
                continue
            submitted = row["submitted_tokens"]
            output = list(generated[index])
            if tokens != submitted[: len(tokens)] or len(tokens) > len(submitted):
                raise RuntimeError("Scheduled DSpark tokens differ from their confidence record")
            accepted = len(output) - 1
            if not 0 <= accepted <= len(tokens) or output[:accepted] != tokens[:accepted]:
                raise RuntimeError("Invalid DSpark accepted-prefix output")
            if accepted < len(tokens) and output[accepted] == tokens[accepted]:
                raise RuntimeError("Greedy rejection correction equals rejected draft token")
            if any(token in self.eos for token in output):
                self._skip(rid, "eos_round")
                continue
            self.awaiting.pop(rid)
            full_accepted = accepted if accepted < len(tokens) or len(tokens) == self.num_proposals else None
            if full_accepted is None:
                if output[accepted] != row["draft_tokens"][accepted]:
                    full_accepted = accepted  # Bonus exposes the next mismatch.
                elif accepted + 1 == self.num_proposals:
                    full_accepted = self.num_proposals  # Bonus confirms the last proposal.
            exact = full_accepted is not None
            row.update(
                status="verified",
                verified_tokens=len(tokens),
                accepted_after_cut=accepted,
                acceptance_length_after_cut=1 + accepted,
                actual_accepted_full=full_accepted,
                actual_acceptance_length_full=1 + full_accepted if exact else None,
                label_complete=exact,
                full_block_verified=len(tokens) == self.num_proposals,
                censored=not exact,
                accepted_full_lower_bound=full_accepted if exact else accepted + 1,
                scheduler_clipped=len(tokens) < len(submitted),
            )
            if exact:
                row["counterfactual_accepted_at_selected_length"] = min(full_accepted, row["selected_verify_tokens"])
            self._write(row)
        for rid in finished:
            if rid in self.awaiting:
                self._skip(rid, "finished_without_verification")
