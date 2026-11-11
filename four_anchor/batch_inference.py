"""Batched bonus20 inference. No training or installed vLLM modifications."""
import torch

from .inference import cached_query_layer, _rotate_half
from .model import boundary_scores

VERSION = "bonus_addon20_draft4_batch_v1"


class RequestCache:
    """Fixed slot pool keyed by request ID, independent of scheduler row order.

    Store projected features and context KV only; raw target features need not
    remain resident. Rejected suffixes never enter the pool. Slots are cleared
    logically by lengths and masks, so reused storage cannot become context.
    """
    def __init__(self, model, capacity, max_requests):
        if not 1 <= max_requests <= 32 or capacity < 32:
            raise ValueError("Require 1..32 requests and context capacity >=32")
        self.model, self.capacity, self.max_requests = model, capacity, max_requests
        w = model.embed_tokens.weight
        self.ids = torch.zeros((max_requests, capacity), device=w.device, dtype=torch.long)
        self.projected = w.new_zeros(max_requests, capacity, model.hidden_size)
        self.memory = []
        for layer in model.layers:
            a = layer.self_attn
            shape = (max_requests, a.k_proj.out_features // a.head_dim, capacity, a.head_dim)
            self.memory.append((w.new_zeros(shape), w.new_zeros(shape)))
        rope_type = getattr(model.rotary_emb, "rope_type", "default")
        if rope_type not in {"default", "linear", "yarn", "llama3"}:
            raise ValueError(f"Unsupported changing RoPE: {rope_type}")
        self.positions = torch.arange(capacity, device=w.device)
        self.cos, self.sin = model.rotary_emb(self.projected[:1], self.positions[None])
        self.slots, self.lengths = {}, {}

    def release(self, request_ids):
        for rid in request_ids:
            self.slots.pop(rid, None)
            self.lengths.pop(rid, None)

    def clear(self):
        self.slots.clear()
        self.lengths.clear()

    @torch.inference_mode()
    def update(self, updates, active_ids):
        """updates = [(request_id, start, token_ids[N], target_features[N,F])]."""
        active = set(active_ids)
        if len(active) != len(active_ids) or len(active) > self.max_requests:
            raise ValueError("Duplicate requests or batch exceeds cache capacity")
        if len({u[0] for u in updates}) != len(updates):
            raise ValueError("Repeated update for a request in one target forward")
        rows, positions, features, tokens = [], [], [], []
        width = self.model.fc.in_features
        for rid, start, ids, feat in updates:
            count = ids.numel()
            if rid not in active or ids.ndim != 1 or feat.shape != (count, width) or count < 1:
                raise ValueError("Invalid request update or target feature shape")
            if start < 0 or start + count > self.capacity:
                raise ValueError("Context update outside capacity")
            if rid not in self.slots:
                if start != 0:
                    raise RuntimeError("Missing request prefix; disable prefix caching and use full recomputation after preemption")
                free = sorted(set(range(self.max_requests)) - set(self.slots.values()))
                if not free:
                    victim = next((r for r in self.slots if r not in active), None)
                    if victim is None:
                        raise RuntimeError("No free request cache slot")
                    free = [self.slots[victim]]
                    self.release([victim])
                self.slots[rid], self.lengths[rid] = free[0], 0
            if start == 0:
                self.lengths[rid] = 0
            if start > self.lengths[rid]:
                raise RuntimeError("Gap in request target features; refusing stale prefix reuse")
            slot = self.slots[rid]
            rows.extend([slot] * count)
            positions.extend(range(start, start + count))
            features.append(feat)
            tokens.append(ids)
            self.lengths[rid] = start + count
        if not updates:
            return
        device = self.ids.device
        rows = torch.tensor(rows, device=device)
        pos = torch.tensor(positions, device=device)
        # One projection per batch, not 32 independent drafter forwards.
        ctx = self.model.hidden_norm(self.model.fc(torch.cat(features, 0)))
        self.projected[rows, pos] = ctx
        # Native vLLM inputs are int32; the embedding/history IDs are int64.
        # Indexed assignment requires equal source/destination dtypes on NPU.
        self.ids[rows, pos] = torch.cat(tokens).to(self.ids)
        cos, sin = self.cos[0, pos, None], self.sin[0, pos, None]
        for layer, (key, value) in zip(self.model.layers, self.memory):
            a = layer.self_attn
            shape = (ctx.shape[0], -1, a.head_dim)
            k = a.k_norm(a.k_proj(ctx).view(shape))
            # Advanced indexing returns [tokens, kv_heads, head_dim].
            key[rows, :, pos] = k * cos + _rotate_half(k) * sin
            value[rows, :, pos] = a.v_proj(ctx).view(shape)


class BatchedBonusAddonDraft:
    """Round one Bx20; round two Bx19; exactly Bx15 verifier candidates."""
    requires_materialized_bonus = False

    def __init__(self, model, capacity, max_requests=32):
        if (model.cfg.anchor_layout, model.cfg.format_version, model.cfg.num_anchors, model.cfg.num_proposals) not in {
                ("target_four", 3, 4, 15), ("bonus_addon", 4, 4, 15)}:
            raise ValueError("Requires target_four v3 or bonus_addon v4, four anchors and 15 proposals")
        self.model = model
        self.cache = RequestCache(model, capacity, max_requests)
        self.offsets = torch.arange(-4, 0, device=self.cache.ids.device)
        self.candidates = torch.arange(15, device=self.cache.ids.device)

    def query_hidden(self, memory, lengths, anchor_ids, anchor_pos, feature, valid, slots, start):
        x = self.model.embed_tokens(torch.cat((anchor_ids, slots), 1))
        x += torch.cat((feature, torch.zeros_like(x[:, 4:])), 1)
        positions = torch.cat((anchor_pos, start[:, None] + torch.arange(slots.shape[1], device=x.device)), 1)
        rope = self.cache.cos[0, positions], self.cache.sin[0, positions]
        span = memory[0][0].shape[2]
        key_valid = torch.cat((self.cache.positions[None, :span] < lengths[:, None],
                               valid, torch.ones_like(slots, dtype=torch.bool)), 1)
        mask = x.new_zeros(x.shape[0], 1, x.shape[1], span + x.shape[1])
        mask.masked_fill_(~key_valid[:, None, None], torch.finfo(x.dtype).min)
        for layer, (k, v) in zip(self.model.layers, memory):
            x = cached_query_layer(layer, x, k, v, mask, rope)
        return self.model.norm(x)

    @torch.inference_mode()
    def propose(self, request_ids, bonuses, mode="two_pass"):
        if mode not in {"one_pass", "two_pass"} or not request_ids:
            raise ValueError("Expected a nonempty one_pass/two_pass batch")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Duplicate request IDs")
        c = self.cache
        sizes = [c.lengths[r] for r in request_ids]
        if any(n < 1 or n + 30 > c.capacity for n in sizes):
            raise ValueError("Not enough positions for both drafting rounds")
        device = c.ids.device
        rows = torch.tensor([c.slots[r] for r in request_ids], device=device)
        lengths = torch.tensor(sizes, device=device)
        bonus = torch.as_tensor(bonuses, device=device, dtype=torch.long).reshape(-1, 1)
        if bonus.shape[0] != len(request_ids):
            raise ValueError("One known bonus is required per request")
        pos = lengths[:, None] + self.offsets
        valid = pos >= 0
        pos = pos.clamp_min(0)
        a_ids = torch.where(valid, c.ids[rows[:, None], pos], 0)
        feature = c.projected[rows[:, None], pos] * valid[..., None]
        # Gather once, reused by both rounds. Each batch row has its own KV.
        span = max(sizes)
        memory = [(k[:, :, :span].index_select(0, rows), v[:, :, :span].index_select(0, rows))
                  for k, v in c.memory]
        masks = torch.full((len(request_ids), 15), self.model.cfg.mask_token_id, device=device, dtype=torch.long)
        h1 = self.query_hidden(memory, lengths, a_ids, pos, feature, valid,
                               torch.cat((bonus, masks), 1), lengths)
        logits = self.model.lm_head(h1[:, 5:])
        first = logits.argmax(-1)
        if mode == "one_pass":
            return first, None
        boundary = boundary_scores(logits).argmax(-1)
        restart = lengths + 1 + boundary
        select = boundary[:, None] + self.offsets + 5
        second_ids = torch.cat((a_ids, bonus, first), 1).gather(1, select)
        second_hidden = h1.gather(1, select[..., None].expand(-1, -1, self.model.hidden_size))
        second_pos = restart[:, None] + self.offsets
        valid = second_pos >= 0
        feature = self.model.draft_anchor_norm(self.model.draft_anchor_proj(second_hidden)) * valid[..., None]
        h2 = self.query_hidden(memory, lengths, second_ids, second_pos.clamp_min(0),
                               feature, valid, masks, restart)
        second = self.model.lm_head(h2[:, 4:]).argmax(-1)
        index = self.candidates[None].expand(len(request_ids), -1)
        replacement = second.gather(1, (index - boundary[:, None]).clamp_min(0))
        return torch.where(index < boundary[:, None], first, replacement), boundary
