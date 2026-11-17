"""Exact block-isolated SDPA for native DLoop; no cross-block masked work.

The upstream Q/K/V projections, RoPE, stage RNG, carriers, losses and optimizer
are retained. Only attention's representation of the same connectivity changes.
"""
from dataclasses import dataclass
import os
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class BlockMask:
    anchors: torch.Tensor
    keep: torch.Tensor
    context: int
    rows: int
    carriers: int


def make_mask(anchor_positions, block_keep_mask, S, block_size, device, n_carrier=0):
    return BlockMask(anchor_positions, block_keep_mask, S, block_size, n_carrier)


def block_sdpa(module, query, key, value, attention_mask, dropout=0.0,
               scaling=None, sliding_window=None, **kwargs):
    if not isinstance(attention_mask, BlockMask):
        raise ValueError("Block attention is training-only and requires its exact mask descriptor")
    if dropout or sliding_window is not None:
        raise ValueError("Only dropout-free full DFlash attention is supported")
    m = attention_mask
    b, h, qlen, d = query.shape
    n = m.anchors.shape[1]
    s, r, c = m.context, m.rows, m.carriers
    if qlen != n*r or key.shape[-2] != s+n*c+n*r:
        raise ValueError("Unexpected DLoop Q/K/V layout")
    hk = key.shape[1]
    if h % hk:
        raise ValueError("Invalid grouped-query heads")
    chunks = max(1, int(os.environ.get("ATTN_BLOCK_CHUNK", "8")))
    q = query.reshape(b, h, n, r, d)
    kc, vc = key[:, :, :s], value[:, :, :s]
    kl = torch.cat((key[:, :, s:s+n*c].reshape(b,hk,n,c,d),
                    key[:, :, s+n*c:].reshape(b,hk,n,r,d)), dim=3)
    vl = torch.cat((value[:, :, s:s+n*c].reshape(b,hk,n,c,d),
                    value[:, :, s+n*c:].reshape(b,hk,n,r,d)), dim=3)

    def attend(qpart, kc, vc, kp, vp, anchors, keep):
        count = qpart.shape[2]
        qq = qpart.permute(0,2,1,3,4).reshape(b*count,h,r,d)
        kk = torch.cat((kc[:,None].expand(-1,count,-1,-1,-1), kp.permute(0,2,1,3,4)), dim=3)
        vv = torch.cat((vc[:,None].expand(-1,count,-1,-1,-1), vp.permute(0,2,1,3,4)), dim=3)
        kk = kk.reshape(b*count,hk,s+c+r,d).repeat_interleave(h//hk, dim=1)
        vv = vv.reshape(b*count,hk,s+c+r,d).repeat_interleave(h//hk, dim=1)
        allowed = torch.cat((torch.arange(s,device=query.device)[None,None] < anchors[:,:,None],
                             torch.ones(b,count,c+r,device=query.device,dtype=torch.bool)), dim=2)
        allowed = (allowed & keep[:,:,None]).reshape(b*count,1,1,s+c+r)
        out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=allowed,
                                             dropout_p=0.0, scale=scaling)
        return out.reshape(b,count,h,r,d).permute(0,2,1,3,4)

    parts = []
    for start in range(0,n,chunks):
        end = min(start+chunks,n)
        args = (q[:,:,start:end],kc,vc,kl[:,:,start:end],vl[:,:,start:end],
                m.anchors[:,start:end],m.keep[:,start:end])
        # Bound gathered K/V activation memory without detaching carrier gradients.
        if torch.is_grad_enabled() and any(x.requires_grad for x in args[:5]):
            part = checkpoint(attend,*args,use_reentrant=False,preserve_rng_state=False)
        else:
            part = attend(*args)
        parts.append(part)
    out = torch.cat(parts,dim=2).reshape(b,h,qlen,d).transpose(1,2).contiguous()
    return out, None


def install():
    from specforge.core import dflash as core
    from specforge.modeling.draft import dflash as model
    original = model.Qwen3DFlashAttention.forward
    def dispatch(self, *args, **kwargs):
        if isinstance(kwargs.get("attention_mask"), BlockMask):
            previous = self.config._attn_implementation
            self.config._attn_implementation = "pair_block_sdpa"
            try:
                return original(self,*args,**kwargs)
            finally:
                self.config._attn_implementation = previous
        return original(self,*args,**kwargs)
    model.ALL_ATTENTION_FUNCTIONS.register("pair_block_sdpa", block_sdpa)
    model.Qwen3DFlashAttention.forward = dispatch
    core.create_dflash_sdpa_mask = make_mask

