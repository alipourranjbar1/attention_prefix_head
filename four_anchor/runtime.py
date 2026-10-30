"""Target cache management and exact greedy, two-draft decoding."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import FourAnchorDFlash, make_anchors, select_boundary


def resolve_device(name):
    if name in {"cuda", "npu"}:
        name += ":0"
    if name.startswith("npu"):
        import torch_npu  # noqa: F401; registers the device/backend
    device = torch.device(name)
    if device.type in {"cuda", "npu"}:
        getattr(torch, device.type).set_device(device)
    return device


def synchronize(device):
    if device.type in {"cuda", "npu"}:
        getattr(torch, device.type).synchronize(device)


def load_target(name, device="cpu", dtype="float32", revision=None, attention="sdpa"):
    device = resolve_device(device)
    target = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=getattr(torch, dtype),
        attn_implementation=attention, trust_remote_code=False,
    ).to(device).eval().requires_grad_(False)
    if target.config.model_type != "qwen3":
        raise ValueError("Only Qwen3 targets are supported by this reference backend")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    return target, tokenizer


@dataclass
class TargetState:
    ids: torch.Tensor
    features: torch.Tensor
    cache: object
    next_logits: torch.Tensor


class TargetRunner:
    """Capture selected actual decoder outputs, not a guessed HF tuple offset.

    The target is frozen. Hooks only store detached states during this runner's
    calls; callers may still use target.generate without retaining its states.
    """
    def __init__(self, target, layer_ids):
        self.target = target.eval().requires_grad_(False)
        self.layer_ids = list(layer_ids)
        self.captures = {}
        self.active = False
        self.calls = 0
        self.handles = []
        for i in self.layer_ids:
            def capture(_module, _inputs, output, i=i):
                if self.active:
                    self.captures[i] = (output[0] if isinstance(output, tuple) else output).detach()
            self.handles.append(target.model.layers[i].register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captures.clear()

    @torch.no_grad()
    def forward(self, ids, cache=None, all_logits=False):
        self.captures.clear()
        self.active = True
        self.calls += 1
        try:
            out = self.target(input_ids=ids, past_key_values=cache, use_cache=True,
                              logits_to_keep=0 if all_logits else 1, return_dict=True)
            features = torch.cat([self.captures[i] for i in self.layer_ids], dim=-1)
        finally:
            self.active = False
            self.captures.clear()
        return out, features

    def prefill(self, ids):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Expected a nonempty [1, length] prompt")
        out, features = self.forward(ids)
        return TargetState(ids, features, out.past_key_values, out.logits[:, -1])

    def append(self, state, ids):
        if ids.shape[1] == 0:
            return state
        out, features = self.forward(ids, state.cache)
        return TargetState(torch.cat([state.ids, ids], 1), torch.cat([state.features, features], 1),
                           out.past_key_values, out.logits[:, -1])

    def rollout(self, state, count, eos_ids=()):
        """Training teacher continuation, conditioned on the ACTUAL retained proposal.

        Mutates the supplied target KV cache. Stops loss after the first EOS.
        Does not expose these extra target states to the second-pass drafter.
        """
        tokens, valid = [], []
        alive = True
        for _ in range(count):
            token = state.next_logits.argmax(-1, keepdim=True)
            tokens.append(token)
            valid.append(alive)
            alive = alive and int(token.item()) not in eos_ids
            state = self.append(state, token)
        return torch.cat(tokens, 1), torch.tensor([valid], device=state.ids.device, dtype=torch.float32)

    def verify(self, state, candidates, materialize_correction=True):
        """Greedy prefix verification; crop all rejected KV entries before correction.

        Returns accepted length, correction/bonus, and accepted-prefix state.
        By default also materializes the correction/bonus in the returned state.
        The extra one-token target call is intentional: a target-chosen token has
        no target hidden state until processed. All four next anchors get real
        states rather than features computed from a rejected token.
        """
        n, m = state.ids.shape[1], candidates.shape[1]
        if m < 1:
            raise ValueError("Cannot verify an empty proposal")
        out, features = self.forward(candidates, state.cache, all_logits=True)
        expected = torch.cat([state.next_logits[:, None], out.logits[:, :-1]], 1).argmax(-1)
        mismatch = (candidates != expected)[0].nonzero(as_tuple=False)
        accepted = int(mismatch[0].item()) if mismatch.numel() else m
        correction = (expected[:, accepted:accepted + 1] if accepted < m
                      else out.logits[:, -1].argmax(-1, keepdim=True))
        out.past_key_values.crop(n + accepted)
        kept_ids = torch.cat([state.ids, candidates[:, :accepted]], 1)
        kept_features = torch.cat([state.features, features[:, :accepted]], 1)
        # next_logits is overwritten by append; this distribution corresponds
        # exactly to the accepted prefix, including the accepted=0 case.
        next_logits = state.next_logits if accepted == 0 else out.logits[:, accepted - 1]
        kept = TargetState(kept_ids, kept_features, out.past_key_values, next_logits)
        if materialize_correction:
            kept = self.append(kept, correction)
        return accepted, correction, kept


@dataclass
class Cycle:
    boundary: int | None
    proposed: int
    accepted_draft: int
    committed: int
    draft_calls: int
    anchor_positions: list[int]
    anchor_sources: list[int]


@dataclass
class Generation:
    token_ids: list[int]
    cycles: list[Cycle]
    seconds: float
    prefill_seconds: float
    target_calls: int

    def metrics(self):
        n = len(self.cycles)
        return {
            "output_tokens": len(self.token_ids), "cycles": n,
            "accepted_draft_per_cycle": sum(c.accepted_draft for c in self.cycles) / max(n, 1),
            "committed_per_cycle": sum(c.committed for c in self.cycles) / max(n, 1),
            "draft_calls": sum(c.draft_calls for c in self.cycles),
            "target_calls_including_prefill_and_repair": self.target_calls,
            "seconds_including_prefill": self.seconds, "prefill_seconds": self.prefill_seconds,
            "tokens_per_second": len(self.token_ids) / max(self.seconds, 1e-9),
            "decode_tokens_per_second": len(self.token_ids) / max(self.seconds - self.prefill_seconds, 1e-9),
        }


class Decoder:
    def __init__(self, runner: TargetRunner, drafter: FourAnchorDFlash | None):
        self.runner, self.drafter = runner, drafter

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=128, mode="two_pass", eos_ids=None):
        if mode not in {"two_pass", "one_pass", "ar"}:
            raise ValueError("mode must be two_pass, one_pass, or ar")
        if mode != "ar" and self.drafter is None:
            raise ValueError("Speculative generation requires a trained checkpoint")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if eos_ids is None:
            eos = self.runner.target.generation_config.eos_token_id
            eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
        if prompt_ids.shape[1] + max_new_tokens > self.runner.target.config.max_position_embeddings:
            raise ValueError("Requested sequence exceeds target max_position_embeddings")
        if max_new_tokens == 0:
            return Generation([], [], 0.0, 0.0, 0)
        if self.drafter is not None:
            self.drafter.eval()
        device = prompt_ids.device
        synchronize(device)
        started, calls_before = time.perf_counter(), self.runner.calls
        state = self.runner.prefill(prompt_ids)
        synchronize(device)
        prefill_seconds = time.perf_counter() - started
        result, cycles = [], []
        while len(result) < max_new_tokens:
            remaining = max_new_tokens - len(result)
            n = state.ids.shape[1]
            if mode == "ar":
                token = state.next_logits.argmax(-1, keepdim=True)
                result.append(int(token.item()))
                if result[-1] in eos_ids or len(result) == max_new_tokens:
                    break
                state = self.runner.append(state, token)
                continue
            cfg = self.drafter.cfg
            anchors = make_anchors(state.ids, state.features, cfg.num_anchors, self.drafter.hidden_size)
            first = self.drafter(state.features, anchors, n)
            boundary = None
            candidates = first.tokens
            if mode == "two_pass":
                boundary = select_boundary(first.logits)
                anchors = make_anchors(state.ids, state.features, cfg.num_anchors,
                                       self.drafter.hidden_size, first, boundary)
                second = self.drafter(state.features, anchors, n + boundary)
                # Fixed 16 new slots on BOTH passes: retained prefix + 16 fresh proposals.
                candidates = torch.cat([first.tokens[:, :boundary], second.tokens], dim=1)
            # Limit verification work to the requested output budget. Reserve no
            # mandatory bonus slot: the emitted extension is truncated below.
            candidates = candidates[:, :remaining]
            accepted, correction, state = self.runner.verify(state, candidates, materialize_correction=False)
            extension = torch.cat([candidates[:, :accepted], correction], 1)[0].tolist()[:remaining]
            eos_at = next((i for i, t in enumerate(extension) if t in eos_ids), None)
            if eos_at is not None:
                extension = extension[:eos_at + 1]
            cycles.append(Cycle(boundary, candidates.shape[1], min(accepted, len(extension)), len(extension),
                                2 if mode == "two_pass" else 1,
                                anchors.positions[0].tolist(), anchors.source[0].tolist()))
            result.extend(extension)
            if eos_at is not None or len(result) == max_new_tokens:
                break
            state = self.runner.append(state, correction)
        synchronize(device)
        return Generation(result, cycles, time.perf_counter() - started, prefill_seconds,
                          self.runner.calls - calls_before)
