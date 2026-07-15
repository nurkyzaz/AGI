"""The four Phase-1 baselines, sharing a small CNN encoder/decoder.

All models map a few-shot Task to per-cell logits over the 10 ARC colors for the
query output. They differ only in where the *rule code* z comes from:

  raw           z = encode(full demo grids)         -- can exploit the barcode
  object_centric z = encode(border-stripped demos)  -- nuisance removed a priori
  cond_ib       z ~ N(mu,sigma) from demos, KL-penalized (compression pressure)
  oracle        z = MLP(one-hot true C)             -- upper bound, never sees U

The decoder is identical across models: it reads the query input and is
FiLM-conditioned on z to produce the output grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLORS = 10


@dataclass
class Batch:
    demo_in: torch.Tensor    # (B, k, H, W) long
    demo_out: torch.Tensor   # (B, k, H, W) long
    query_in: torch.Tensor   # (B, H, W) long
    query_out: torch.Tensor  # (B, H, W) long
    c_onehot: torch.Tensor   # (B, sum_card) float
    barcode: torch.Tensor    # (B,) long

    def to(self, device):
        return Batch(
            self.demo_in.to(device), self.demo_out.to(device),
            self.query_in.to(device), self.query_out.to(device),
            self.c_onehot.to(device), self.barcode.to(device),
        )


def _onehot(grid: torch.Tensor) -> torch.Tensor:
    """(..., H, W) long -> (..., N_COLORS, H, W) float."""
    return F.one_hot(grid.long(), N_COLORS).movedim(-1, -3).float()


def _strip_border(grid: torch.Tensor, depth: int = 1) -> torch.Tensor:
    g = grid.clone()
    g[..., :depth, :] = 0
    g[..., -depth:, :] = 0
    g[..., :, :depth] = 0
    g[..., :, -depth:] = 0
    return g


class _ConvStack(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DemoEncoder(nn.Module):
    """Encode k demo pairs -> rule code z (mean-pooled, permutation-invariant)."""

    def __init__(self, z_dim: int = 64, hidden: int = 48, variational: bool = False):
        super().__init__()
        self.variational = variational
        self.conv = _ConvStack(2 * N_COLORS, hidden)
        out_dim = 2 * z_dim if variational else z_dim
        self.head = nn.Linear(hidden, out_dim)
        self.z_dim = z_dim

    def forward(self, demo_in, demo_out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, k = demo_in.shape[:2]
        x = torch.cat([_onehot(demo_in), _onehot(demo_out)], dim=2)  # (B,k,2C,H,W)
        x = x.flatten(0, 1)                                          # (B*k,2C,H,W)
        feat = self.conv(x).mean(dim=(-1, -2))                       # (B*k, hidden)
        emb = self.head(feat).view(B, k, -1).mean(dim=1)            # (B, out_dim)
        if self.variational:
            mu, logvar = emb.chunk(2, dim=-1)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std) if self.training else mu
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
            return z, kl
        return emb, None


class Decoder(nn.Module):
    """Apply rule code z to the query input, produce output logits.

    A local conv stack cannot move pixels across the whole grid (needed for
    reflect / rotate / translate), so after FiLM-conditioning on z we add one
    global self-attention layer over the H*W cells (with learned positional
    embeddings). That gives the decoder a global receptive field at trivial cost
    on a 12x12 grid, so geometric transforms become learnable."""

    def __init__(self, z_dim: int = 64, hidden: int = 48, max_hw: int = 16 * 16):
        super().__init__()
        self.hidden = hidden
        self.conv = _ConvStack(N_COLORS, hidden)
        self.film = nn.Linear(z_dim, 2 * hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_hw, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                                 nn.Linear(2 * hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Conv2d(hidden, N_COLORS, 1)

    def forward(self, query_in, z):
        B, H, W = query_in.shape
        h = self.conv(_onehot(query_in))                  # (B, hidden, H, W)
        gamma, beta = self.film(z).chunk(2, dim=-1)
        h = F.relu(h * gamma[..., None, None] + beta[..., None, None])
        t = h.flatten(2).transpose(1, 2)                  # (B, H*W, hidden)
        t = t + self.pos[:, : H * W]
        a, _ = self.attn(t, t, t)
        t = self.norm(t + a)
        t = self.norm2(t + self.ffn(t))
        h = t.transpose(1, 2).reshape(B, self.hidden, H, W)
        return self.head(h)                                # (B, N_COLORS, H, W)


class BaselineModel(nn.Module):
    """Wraps an encoder + decoder; `kind` selects the baseline behavior."""

    def __init__(self, kind: str, c_onehot_dim: int, z_dim: int = 64, hidden: int = 48):
        super().__init__()
        assert kind in {"raw", "object_centric", "cond_ib", "oracle"}
        self.kind = kind
        self.z_dim = z_dim
        if kind == "oracle":
            self.encoder = nn.Sequential(
                nn.Linear(c_onehot_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim)
            )
        else:
            self.encoder = DemoEncoder(z_dim, hidden, variational=(kind == "cond_ib"))
        self.decoder = Decoder(z_dim, hidden)

    def encode(self, b: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.kind == "oracle":
            return self.encoder(b.c_onehot), None
        di, do = b.demo_in, b.demo_out
        if self.kind == "object_centric":
            di, do = _strip_border(di), _strip_border(do)
        return self.encoder(di, do)

    def forward(self, b: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z, kl = self.encode(b)
        qin = _strip_border(b.query_in) if self.kind in ("object_centric", "oracle") else b.query_in
        logits = self.decoder(qin, z)
        return logits, kl
