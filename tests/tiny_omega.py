"""A tiny stand-in for VGGT-Omega's aggregator with the SAME interface the attention gate reads
(inter_frame_blocks of SelfAttentionBlock, inter_frame_attention_types, patch_token_start,
register_token), random weights on the CPU — the fidelity, score and hook tests run on it without
a card. The real blocks are the vendored ones (vendor/vggt-omega, never modified)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_FORK = Path(__file__).resolve().parents[1]
_OMEGA = _FORK.parent / "vggt-omega"
if str(_OMEGA) not in sys.path:
    sys.path.insert(0, str(_OMEGA))

from vggt_omega.models.layers import Mlp, SelfAttentionBlock      # noqa: E402


def tiny_block(dim=16, heads=2, seed=0) -> SelfAttentionBlock:
    """A SelfAttentionBlock as the aggregator builds its inter-frame blocks (qkv bias, qk norm,
    masked k bias) with random weights; the k-bias mask set to a finite pattern (the checkpoint's
    buffer is NaN until loaded)."""
    torch.manual_seed(seed)
    blk = SelfAttentionBlock(dim=dim, num_heads=heads, ffn_ratio=2.0, qkv_bias=True, proj_bias=True,
                             ffn_bias=True, ffn_layer=Mlp, init_values=1e-2, use_qk_norm=True, mask_k_bias=True)
    with torch.no_grad():
        mask = torch.ones(3 * dim)
        mask[dim:2 * dim] = 0.0                       # no bias on k, as the vendor's mask does
        blk.attn.qkv.bias_mask.copy_(mask)
        # LayerScale allocates its gamma with torch.empty and initialises it only in
        # reset_parameters (the real model gets it from the checkpoint): without this the tiny
        # block carried whatever memory held — NaN now and then, a different block per process
        blk.ls1.reset_parameters()
        blk.ls2.reset_parameters()
        for p in blk.parameters():
            p.add_(0.05 * torch.randn_like(p))
    blk.eval()
    return blk


class FakeAggregator(nn.Module):
    """Three inter-frame blocks (global, register, global); forward(tokens[B, S, N, C]) runs them
    the way Aggregator._run_inter_frame_attention_block does (no frame blocks, no RoPE)."""

    def __init__(self, dim=16, heads=2, n_reg=2):
        super().__init__()
        self.inter_frame_blocks = nn.ModuleList([tiny_block(dim, heads, seed=s) for s in (0, 1, 2)])
        self.inter_frame_attention_types = ["global", "register", "global"]
        self.register_token = nn.Parameter(torch.zeros(1, 2, n_reg, dim))
        self.patch_token_start = 1 + n_reg
        self.depth = 3

    def forward(self, tokens):
        # CPU tests: the math SDPA kernel (the fused CPU kernels of torch 2.11 returned NaN now and
        # then on these tiny shapes — a kernel matter, not the gate's; the card runs the real model)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            return self._forward_tokens(tokens)

    def _forward_tokens(self, tokens):
        B, S, N, C = tokens.shape
        for idx, kind in enumerate(self.inter_frame_attention_types):
            if kind == "global":
                x = tokens.reshape(B, S * N, C)
                x = self.inter_frame_blocks[idx](x, None)
                tokens = x.reshape(B, S, N, C)
            else:
                pts = self.patch_token_start
                cr = tokens[:, :, :pts].reshape(B, S * pts, C)
                cr = self.inter_frame_blocks[idx](cr, None)
                tokens = torch.cat([cr.reshape(B, S, pts, C), tokens[:, :, pts:]], dim=2)
        return tokens
