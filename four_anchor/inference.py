"""Versioned cached drafting; target features are never fabricated or shifted.

KnownBonusDraft preserves the published 20-position embedding-only inference.
BonusInFourDraft is the superseded format-2 embedding-only 19-position variant.
TargetFourDraft is the corrected format-3 layout: four actual target-state
anchors, including a materialized bonus, plus fifteen masks in both rounds.
AcceptedPrefixDraft uses the same checkpoint and 19-position queries but starts
from the processed prefix, avoiding bonus materialization altogether.
"""
from contextlib import contextmanager
import json
import time

import torch

from .model import boundary_scores
from speculators.models.dflash.model_definitions import (
    ALL_ATTENTION_FUNCTIONS, _rotate_half, apply_rotary_pos_emb, eager_attention_forward,
)


def cached_query_layer(layer, hidden, context_k, context_v, mask, rope):
    """Original DFlash equations with preprojected context K/V.

    Kept here so installing inference does not require editing a user's
    training model.py or depending on a particular training optimization.
    """
    residual = hidden
    x = layer.input_layernorm(hidden)
    attention = layer.self_attn
    b, length = x.shape[:2]
    shape = (b, length, -1, attention.head_dim)
    q = attention.q_norm(attention.q_proj(x).view(shape)).transpose(1, 2)
    k = attention.k_norm(attention.k_proj(x).view(shape)).transpose(1, 2)
    v = attention.v_proj(x).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *rope)
    k, v = torch.cat((context_k, k), -2), torch.cat((context_v, v), -2)
    implementation = attention.config._attn_implementation
    fn = eager_attention_forward if implementation in {None, "eager"} else ALL_ATTENTION_FUNCTIONS[implementation]
    value = fn(attention, q, k, v, mask, dropout=0., scaling=attention.scaling,
               sliding_window=attention.sliding_window)[0]
    hidden = residual + attention.o_proj(value.reshape(b, length, -1))
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class DraftContextCache:
    """Only confirmed target inputs enter this per-request projected KV cache."""
    def __init__(self, model, capacity):
        self.model, self.capacity = model, capacity
        device, dtype = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
        self.projected = torch.zeros((1, capacity, model.hidden_size), device=device, dtype=dtype)
        self.memory = []
        for layer in model.layers:
            attn = layer.self_attn
            heads = attn.k_proj.out_features // attn.head_dim
            shape = (1, heads, capacity, attn.head_dim)
            self.memory.append((torch.zeros(shape, device=device, dtype=dtype),
                                torch.zeros(shape, device=device, dtype=dtype)))
        # Position-dependent/dynamic RoPE would need cache invalidation when
        # its frequencies change. This path is for the fixed-RoPE Qwen3 target.
        rope_type = getattr(model.rotary_emb, "rope_type", "default")
        if rope_type not in {"default", "linear", "yarn", "llama3"}:
            raise ValueError(f"Incremental draft cache does not support RoPE type {rope_type}")
        self.positions = torch.arange(capacity, device=device)
        self.cos, self.sin = model.rotary_emb(self.projected, self.positions[None])
        self.request_id, self.length, self.projected_tokens = None, 0, 0

    @torch.inference_mode()
    def update(self, request_id, start, features):
        if request_id != self.request_id or start == 0:
            self.request_id, self.length = request_id, 0
        count = features.shape[1]
        end = start + count
        if features.ndim != 3 or features.shape[0] != 1 or start < 0 or start > self.length:
            raise ValueError("Projected context update has a gap or invalid shape")
        if end > self.capacity:
            raise ValueError("Projected context exceeds capacity")
        ctx = self.model.hidden_norm(self.model.fc(features))
        self.projected[:, start:end].copy_(ctx)
        cos, sin = self.cos[:, None, start:end], self.sin[:, None, start:end]
        for layer, (key, value) in zip(self.model.layers, self.memory):
            attn = layer.self_attn
            shape = (1, count, -1, attn.head_dim)
            k = attn.k_norm(attn.k_proj(ctx).view(shape)).transpose(1, 2)
            key[:, :, start:end].copy_(k * cos + _rotate_half(k) * sin)
            value[:, :, start:end].copy_(attn.v_proj(ctx).view(shape).transpose(1, 2))
        self.length = end  # also discards an overwritten/rejected suffix
        self.projected_tokens += count


class DraftProfiler:
    """Optional synchronized wall timings; disabled for throughput benchmarks."""
    def __init__(self, device, steps=0):
        self.device, self.limit, self.step = torch.device(device), steps, 0
        self.values = {}

    def sync(self):
        if self.device.type in {"npu", "cuda"}:
            getattr(torch, self.device.type).synchronize(self.device)

    @contextmanager
    def region(self, name):
        active = self.step < self.limit
        if active:
            self.sync()
            before = time.perf_counter()
        yield
        if active:
            self.sync()
            self.values[name] = 1000 * (time.perf_counter() - before)

    def finish(self):
        if self.step < self.limit:
            print(json.dumps({"event": "four_anchor_profile", "step": self.step,
                              "synchronized_timings_ms": self.values,
                              "extra_target_forwards": getattr(self, "extra_target_forwards", 0)}), flush=True)
        self.values = {}
        self.extra_target_forwards = 0
        self.step += 1


class KnownBonusDraft:
    """Two tensor-only draft passes; no scalar readback for the boundary.

    Query width is A + M + 1 in BOTH passes. In pass one the first non-anchor
    slot is fixed to the bonus. In pass two all M+1 slots are masks; only the
    suffix within the original M future positions is selected for verification.
    """
    def __init__(self, model, capacity, proposals=15, profile_steps=0, draft_graph=False):
        if model.cfg.num_anchors != 4 or proposals + 1 > model.cfg.num_proposals:
            raise ValueError("Need four anchors and at least proposals+1 trained slots")
        self.model, self.proposals = model, proposals
        self.cache = DraftContextCache(model, capacity)
        device = model.embed_tokens.weight.device
        self.anchor_offsets = torch.arange(-4, 0, device=device)
        self.slot_offsets = torch.arange(proposals + 1, device=device)
        self.candidate_offsets = torch.arange(proposals, device=device)
        self.masks = torch.full((1, proposals + 1), model.cfg.mask_token_id,
                                device=device, dtype=torch.long)
        self.zero_ids = torch.zeros_like(self.masks)
        self.zero_hidden = torch.zeros((1, proposals + 1, model.hidden_size),
                                        device=device, dtype=model.embed_tokens.weight.dtype)
        self.profiler = DraftProfiler(device, profile_steps)
        self.start_buffer = torch.zeros((), device=device, dtype=torch.long)
        self.graphs = NpuDraftGraphs(self) if draft_graph else None

    def anchors(self, ids, n, start, prior_ids, prior_hidden):
        """Gather four past positions; target states only exist strictly before n."""
        pos = start.reshape(1, 1) + self.anchor_offsets[None]
        valid = pos >= 0
        is_target = valid & (pos < n)
        p = torch.minimum(pos.clamp_min(0), torch.as_tensor(n - 1, device=pos.device))
        q = (pos - n).clamp(min=0, max=self.proposals)
        anchor_ids = torch.where(is_target, ids.gather(1, p), prior_ids.gather(1, q))
        anchor_ids = torch.where(valid, anchor_ids, torch.zeros_like(anchor_ids))
        projected = self.cache.projected[:, :ids.shape[1]].gather(
            1, p[..., None].expand(-1, -1, self.model.hidden_size))
        draft = prior_hidden.gather(1, q[..., None].expand(-1, -1, self.model.hidden_size))
        draft = self.model.draft_anchor_norm(self.model.draft_anchor_proj(draft))
        feature = torch.where(is_target[..., None], projected, draft)
        feature = torch.where(valid[..., None], feature, torch.zeros_like(feature))
        return anchor_ids, pos.clamp_min(0), feature, valid

    def pass_hidden(self, ids, n, start, prior_ids, prior_hidden, slot_ids):
        anchors, anchor_pos, feature, valid = self.anchors(ids, n, start, prior_ids, prior_hidden)
        x = self.model.embed_tokens(torch.cat((anchors, slot_ids), dim=1))
        x = x + torch.cat((feature, torch.zeros_like(x[:, 4:])), dim=1)
        positions = torch.cat((anchor_pos, start.reshape(1, 1) + self.slot_offsets[None]), dim=1)
        cos = self.cache.cos[:, positions[0]]
        sin = self.cache.sin[:, positions[0]]
        span = ids.shape[1]
        key_valid = torch.cat(((self.cache.positions[:span] < n)[None],
                               valid, torch.ones_like(slot_ids, dtype=torch.bool)), dim=1)
        width = 4 + slot_ids.shape[1]
        mask = torch.zeros((1, 1, width, span + width),
                           device=x.device, dtype=x.dtype)
        mask.masked_fill_(~key_valid[:, None, None], torch.finfo(x.dtype).min)
        for layer, (k, v) in zip(self.model.layers, self.cache.memory):
            x = cached_query_layer(layer, x, k[:, :, :span], v[:, :, :span], mask, (cos, sin))
        return self.model.norm(x[:, getattr(self, "output_start", 4):])

    @torch.inference_mode()
    def propose(self, ids, bonus, mode="two_pass"):
        if mode not in {"one_pass", "two_pass"}:
            raise ValueError("Expected one_pass or two_pass")
        n = self.cache.length
        if n < 1 or ids.shape != (1, n):
            raise ValueError("Expected exactly the confirmed, processed target prefix")
        if n + getattr(self, "position_reserve", 2 * self.proposals + 1) > self.cache.capacity:
            raise ValueError("Not enough positions for the second draft pass")
        if self.graphs is not None:
            with self.profiler.region("draft_graph_replay"):
                return self.graphs.run(ids, bonus, mode)
        self.start_buffer.fill_(n)
        return self.tensor_propose(ids, bonus, self.start_buffer, mode)

    def tensor_propose(self, ids, bonus, start, mode):
        """Fixed-shape body. The context length/start remains a device scalar."""
        n = start
        # Token 9 is known in the 8-accepted example; its target state is not.
        slots = torch.cat((bonus.reshape(1, 1), self.masks[:, 1:]), dim=1)
        with self.profiler.region("draft_first"):
            h1 = self.pass_hidden(ids, n, start, self.zero_ids, self.zero_hidden, slots)
            # No vocabulary projection at the already-known bonus position.
            logits = self.model.lm_head(h1[:, 1:])
            first = logits.argmax(dim=-1)
        if mode == "one_pass":
            return first, None
        with self.profiler.region("boundary_and_anchors"):
            boundary = boundary_scores(logits).argmax(dim=-1)
            prior_ids = torch.cat((bonus.reshape(1, 1), first), dim=1)
            restart = start + 1 + boundary[0]
        with self.profiler.region("draft_second"):
            h2 = self.pass_hidden(ids, n, restart, prior_ids, h1, self.masks)
            # Fixed-sized gather avoids .item(), CPU loops, and redundant LM-head
            # work for the unused last slot. Selection never emits the bonus.
            second = self.model.lm_head(h2[:, :self.proposals]).argmax(dim=-1)
            index = self.candidate_offsets[None]
            replacement = second.gather(1, (index - boundary[:, None]).clamp_min(0))
            tokens = torch.where(index < boundary[:, None], first, replacement)
        return tokens, boundary


class BonusInFourDraft(KnownBonusDraft):
    """Format-2 checkpoint: [3 past anchors, bonus anchor, 15 masks].

The bonus's actual target state is unavailable. Round one adds no feature
to its embedding; round two uses its normalized first-pass draft state.
No target call, predecessor-state substitution, or shifted feature pairing.
"""
    output_start = 3

    def __init__(self, model, capacity, proposals=15, profile_steps=0, draft_graph=False):
        if (model.cfg.anchor_layout, model.cfg.num_anchors, model.cfg.num_proposals, proposals) != (
                "bonus_in_four", 4, 15, 15):
            raise ValueError("BonusInFourDraft requires a checkpoint trained for bonus_in_four / 15")
        self.model, self.proposals = model, proposals
        self.cache = DraftContextCache(model, capacity)
        device, dtype = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
        self.anchor_offsets = torch.arange(-4, 0, device=device)
        self.slot_offsets = torch.arange(15, device=device)
        self.candidate_offsets = self.slot_offsets
        self.masks = torch.full((1, 15), model.cfg.mask_token_id, device=device, dtype=torch.long)
        self.zero_ids = torch.zeros((1, 16), device=device, dtype=torch.long)
        self.zero_hidden = torch.zeros((1, 16, model.hidden_size), device=device, dtype=dtype)
        self.profiler = DraftProfiler(device, profile_steps)
        self.start_buffer = torch.zeros((), device=device, dtype=torch.long)
        self.graphs = NpuDraftGraphs(self) if draft_graph else None

    def tensor_propose(self, ids, bonus, start, mode):
        prior_ids = torch.cat((bonus.reshape(1, 1), self.zero_ids[:, 1:]), -1)
        with self.profiler.region("draft_first"):
            # Zero draft input produces exactly zero added feature (bias-free
            # linear + RMSNorm). The bonus is the fourth anchor, at position n.
            h1 = self.pass_hidden(ids, start, start + 1, prior_ids, self.zero_hidden, self.masks)
            logits = self.model.lm_head(h1[:, 1:])
            first = logits.argmax(-1)
        if mode == "one_pass":
            return first, None
        with self.profiler.region("boundary_and_anchors"):
            boundary = boundary_scores(logits).argmax(-1)
            prior_ids = torch.cat((bonus.reshape(1, 1), first), -1)
            restart = start + 1 + boundary[0]
        with self.profiler.region("draft_second"):
            h2 = self.pass_hidden(ids, start, restart, prior_ids, h1, self.masks)
            second = self.model.lm_head(h2[:, 1:]).argmax(-1)
            index = self.candidate_offsets[None]
            replacement = second.gather(1, (index - boundary[:, None]).clamp_min(0))
            tokens = torch.where(index < boundary[:, None], first, replacement)
        return tokens, boundary


class TargetFourDraft(KnownBonusDraft):
    """Four actual target-state anchors + 15 masks, in both passes.

The supplied processed prefix MUST already include the bonus and its actual
target state. Round two retains the original target/draft source-selection
rule. The model runner owns bonus materialization and counts that target call.
"""
    requires_materialized_bonus = True
    position_reserve = 30

    def __init__(self, model, capacity, proposals=15, profile_steps=0, draft_graph=False):
        if (model.cfg.anchor_layout, model.cfg.num_anchors, model.cfg.num_proposals, proposals) != (
                "target_four", 4, 15, 15):
            raise ValueError("TargetFourDraft requires a format-3 target_four / 15 checkpoint")
        self.model, self.proposals = model, proposals
        self.cache = DraftContextCache(model, capacity)
        device, dtype = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
        self.anchor_offsets = torch.arange(-4, 0, device=device)
        self.slot_offsets = torch.arange(15, device=device)
        self.candidate_offsets = self.slot_offsets
        self.masks = torch.full((1, 15), model.cfg.mask_token_id, device=device, dtype=torch.long)
        self.zero_ids = torch.zeros_like(self.masks)
        self.zero_hidden = torch.zeros((1, 15, model.hidden_size), device=device, dtype=dtype)
        self.zero_bonus = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.profiler = DraftProfiler(device, profile_steps)
        self.start_buffer = torch.zeros((), device=device, dtype=torch.long)
        self.graphs = NpuDraftGraphs(self) if draft_graph else None

    def propose(self, ids, bonus=None, mode="two_pass"):
        if bonus is not None:
            raise ValueError("The bonus must already be part of the processed target prefix")
        return super().propose(ids, self.zero_bonus, mode)

    def tensor_propose(self, ids, _unused_bonus, start, mode):
        with self.profiler.region("draft_first"):
            h1 = self.pass_hidden(ids, start, start, self.zero_ids, self.zero_hidden, self.masks)
            logits = self.model.lm_head(h1)
            first = logits.argmax(-1)
        if mode == "one_pass":
            return first, None
        with self.profiler.region("boundary_and_anchors"):
            boundary = boundary_scores(logits).argmax(-1)
            restart = start + boundary[0]
        with self.profiler.region("draft_second"):
            h2 = self.pass_hidden(ids, start, restart, first, h1, self.masks)
            second = self.model.lm_head(h2).argmax(-1)
            index = self.candidate_offsets[None]
            replacement = second.gather(1, (index - boundary[:, None]).clamp_min(0))
            tokens = torch.where(index < boundary[:, None], first, replacement)
        return tokens, boundary


class AcceptedPrefixDraft(TargetFourDraft):
    """Inference-only policy: four processed anchors, two 15-mask passes.

    The native verifier has already sampled a pending token at position n.
    It has not processed that token, so h_target[n] does not exist yet.
    Round one predicts n..n+14 from the four cached anchors ending at n-1.
    Its slot-zero token ID is then fixed to the known token (its feature stays
    the first-pass drafter state). Redrafting starts in n+1..n+14 and extends
    coverage through n+15. Return exactly n+1..n+15 to native vLLM, which
    schedules [pending token, 15 candidates] in its ONE verification forward.

    No target state is fabricated; neither the pending token nor rejected
    candidates are inserted into the processed-prefix feature cache.
    """
    requires_materialized_bonus = False

    def propose(self, ids, bonus=None, mode="two_pass"):
        if mode != "two_pass":
            raise ValueError("accepted_prefix requires two_pass to cover 15 candidates after the bonus")
        if bonus is None or bonus.shape != (1, 1):
            raise ValueError("accepted_prefix requires the pending bonus token ID, not its target state")
        return KnownBonusDraft.propose(self, ids, bonus, mode)

    def tensor_propose(self, ids, bonus, start, mode):
        if mode != "two_pass":
            raise ValueError("accepted_prefix requires two_pass")
        with self.profiler.region("draft_first"):
            h1 = self.pass_hidden(ids, start, start, self.zero_ids, self.zero_hidden, self.masks)
            # Slot zero is known; avoid an unused vocabulary projection there.
            logits = self.model.lm_head(h1[:, 1:])
            first = torch.cat((bonus.reshape(1, 1), logits.argmax(-1)), dim=1)
        with self.profiler.region("boundary_and_anchors"):
            # The known slot is retained even if its masked prediction would
            # disagree. Boundary selection only considers unknown positions.
            boundary = boundary_scores(logits).argmax(-1) + 1
            restart = start + boundary[0]
        with self.profiler.region("draft_second"):
            h2 = self.pass_hidden(ids, start, restart, first, h1, self.masks)
            second = self.model.lm_head(h2).argmax(-1)
            relative = self.candidate_offsets[None] + 1
            # Round one covers n..n+14, round two also covers n+15.
            # Clamp the unused first-pass gather at the final candidate.
            retained = first.gather(1, relative.clamp_max(self.proposals - 1))
            replacement = second.gather(1, (relative - boundary[:, None]).clamp_min(0))
            tokens = torch.where(relative < boundary[:, None], retained, replacement)
        return tokens, boundary


class BonusAddonDraft(TargetFourDraft):
    """20 queries in round one, 19 in round two, 15 verifier candidates.

    Round one: four processed token/target-feature anchors, the known bonus
    embedding (no target feature), then fifteen masks. Keep ALL twenty output
    states: early restarts need drafter outputs at original anchor positions.
    Round two: the four tokens immediately before the selected restart, with
    their corresponding FIRST-PASS DRAFTER states, then fifteen masks. There
    is no extra bonus slot in round two. Target prefix KV stays cached in both
    rounds; only the four query-anchor features switch entirely to drafter.
    """
    requires_materialized_bonus = False
    position_reserve = 30

    def __init__(self, model, capacity, proposals=15, profile_steps=0, draft_graph=False):
        super().__init__(model, capacity, proposals, profile_steps, draft_graph)
        self.first_slot_offsets = torch.arange(proposals + 1, device=self.masks.device)

    def propose(self, ids, bonus=None, mode="two_pass"):
        if bonus is None or bonus.shape != (1, 1):
            raise ValueError("bonus_addon requires the pending bonus token ID")
        return KnownBonusDraft.propose(self, ids, bonus, mode)

    def query_hidden(self, ids, n, anchor_ids, anchor_pos, feature, valid, slots, start):
        """Return normalized states for every query, including anchor slots."""
        x = self.model.embed_tokens(torch.cat((anchor_ids, slots), dim=1))
        x = x + torch.cat((feature, torch.zeros_like(x[:, 4:])), dim=1)
        offsets = self.first_slot_offsets if slots.shape[1] == 16 else self.slot_offsets
        positions = torch.cat((anchor_pos, start.reshape(1, 1) + offsets[None]), dim=1)
        rope = self.cache.cos[:, positions[0]], self.cache.sin[:, positions[0]]
        span, width = ids.shape[1], x.shape[1]
        key_valid = torch.cat(((self.cache.positions[:span] < n)[None], valid,
                               torch.ones_like(slots, dtype=torch.bool)), dim=1)
        mask = torch.zeros((1, 1, width, span + width), device=x.device, dtype=x.dtype)
        mask.masked_fill_(~key_valid[:, None, None], torch.finfo(x.dtype).min)
        for layer, (k, v) in zip(self.model.layers, self.cache.memory):
            x = cached_query_layer(layer, x, k[:, :, :span], v[:, :, :span], mask, rope)
        return self.model.norm(x)

    def tensor_propose(self, ids, bonus, start, mode):
        if mode not in {"one_pass", "two_pass"}:
            raise ValueError("Expected one_pass or two_pass")
        n = start
        with self.profiler.region("draft_first"):
            a_ids, a_pos, a_feature, a_valid = self.anchors(
                ids, n, start, self.zero_ids, self.zero_hidden)
            slots = torch.cat((bonus.reshape(1, 1), self.masks), dim=1)
            h1 = self.query_hidden(ids, n, a_ids, a_pos, a_feature, a_valid, slots, start)
            # The first 4 positions are past anchors; position 4 is known bonus.
            logits = self.model.lm_head(h1[:, 5:])
            first = logits.argmax(-1)
        if mode == "one_pass":
            return first, None
        with self.profiler.region("boundary_and_anchors"):
            # Boundary is a candidate index 0..14, never the known bonus.
            boundary = boundary_scores(logits).argmax(-1)
            restart = start + 1 + boundary[0]
            # Full first-pass window maps to positions n-4 .. n+15.
            # For b=0 choose indices 1..4 (3 past outputs + bonus output).
            # For b=14 choose indices 15..18; no future/rejected token is kept.
            select = boundary[:, None] + self.anchor_offsets[None] + 5
            prior_ids = torch.cat((a_ids, bonus.reshape(1, 1), first), dim=1)
            second_ids = prior_ids.gather(1, select)
            second_hidden = h1.gather(1, select[..., None].expand(-1, -1, self.model.hidden_size))
            second_pos = restart.reshape(1, 1) + self.anchor_offsets[None]
            second_valid = second_pos >= 0
            # All valid second-pass anchor features come from the drafter,
            # including original-prefix positions for early restart boundaries.
            second_feature = self.model.draft_anchor_norm(self.model.draft_anchor_proj(second_hidden))
            second_feature = torch.where(second_valid[..., None], second_feature,
                                         torch.zeros_like(second_feature))
        with self.profiler.region("draft_second"):
            h2 = self.query_hidden(ids, n, second_ids, second_pos.clamp_min(0),
                                   second_feature, second_valid, self.masks, restart)
            second = self.model.lm_head(h2[:, 4:]).argmax(-1)
            index = self.candidate_offsets[None]
            replacement = second.gather(1, (index - boundary[:, None]).clamp_min(0))
            tokens = torch.where(index < boundary[:, None], first, replacement)
        return tokens, boundary


class NpuDraftGraphs:
    """Optional bucketed ACLGraph for the drafter only; target stays eager.

    Captured in dummy_run so all persistent graph memory is counted before
    vLLM sizes its target KV cache. No lazy capture during timed generation.
    Context length, bonus, token IDs and cached K/V are refreshed before replay.
    """
    def __init__(self, draft):
        if draft.model.embed_tokens.weight.device.type != "npu":
            raise ValueError("DRAFT_GRAPH=1 requires an Ascend NPU")
        self.draft, self.entries = draft, {}
        capacity = draft.cache.capacity
        self.buckets = []
        width = 256
        while width < capacity:
            self.buckets.append(width)
            width *= 2
        self.buckets.append(capacity)

    @torch.inference_mode()
    def prepare(self, mode):
        api = torch.npu
        device = self.draft.model.embed_tokens.weight.device
        limit, self.draft.profiler.limit = self.draft.profiler.limit, 0
        try:
            for span in self.buckets:
                key = (span, mode)
                if key in self.entries:
                    continue
                ids = torch.zeros((1, span), device=device, dtype=torch.long)
                bonus = torch.zeros((1, 1), device=device, dtype=torch.long)
                length = torch.tensor(min(span, self.draft.cache.capacity - 2*self.draft.proposals - 1),
                                      device=device, dtype=torch.long)
                stream = api.Stream()
                stream.wait_stream(api.current_stream())
                with api.stream(stream):
                    for _ in range(2):
                        self.draft.tensor_propose(ids, bonus, length, mode)
                api.current_stream().wait_stream(stream)
                api.synchronize()
                graph = api.NPUGraph()
                try:
                    with api.graph(graph, stream=stream):
                        outputs = self.draft.tensor_propose(ids, bonus, length, mode)
                except RuntimeError as exc:
                    raise RuntimeError("Drafter ACLGraph capture failed. Run with DRAFT_GRAPH=0; "
                                       "the eager drafter remains available.") from exc
                api.current_stream().wait_stream(stream)
                self.entries[key] = (graph, ids, bonus, length, outputs)
            api.synchronize()
        finally:
            self.draft.profiler.limit = limit

    def run(self, ids, bonus, mode):
        n = ids.shape[1]
        span = next(b for b in self.buckets if b >= n)
        if (span, mode) not in self.entries:
            raise RuntimeError("Drafter graphs must be captured during memory profiling")
        graph, fixed_ids, fixed_bonus, length, outputs = self.entries[span, mode]
        fixed_ids[:, :n].copy_(ids)
        fixed_bonus.copy_(bonus)
        length.fill_(n)
        graph.replay()
        return outputs
