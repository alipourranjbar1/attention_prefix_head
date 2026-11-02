"""Inference-only known-bonus drafting and incremental context projection.

The frozen checkpoint is unchanged. A first pass has four past anchors, the
known target correction/bonus, and M masks. The bonus has an embedding, NEVER
a fabricated target hidden state. Pass two retains the trained query width.
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
                              "extra_target_forwards": 0}), flush=True)
        self.values = {}
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
        mask = torch.zeros((1, 1, 4 + self.proposals + 1, span + 4 + self.proposals + 1),
                           device=x.device, dtype=x.dtype)
        mask.masked_fill_(~key_valid[:, None, None], torch.finfo(x.dtype).min)
        for layer, (k, v) in zip(self.model.layers, self.cache.memory):
            x = cached_query_layer(layer, x, k[:, :, :span], v[:, :, :span], mask, (cos, sin))
        return self.model.norm(x[:, 4:])

    @torch.inference_mode()
    def propose(self, ids, bonus, mode="two_pass"):
        if mode not in {"one_pass", "two_pass"}:
            raise ValueError("Expected one_pass or two_pass")
        n = self.cache.length
        if n < 1 or ids.shape != (1, n):
            raise ValueError("Expected exactly the confirmed, processed target prefix")
        if n + 2 * self.proposals + 1 > self.cache.capacity:
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
                                       "the cached no-extra-target-forward path remains available.") from exc
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
