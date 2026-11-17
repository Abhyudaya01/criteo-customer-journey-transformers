"""Lu & Kannan (2026) heterogeneous multi-head Transformer, adapted to 12-hour windows.

One token per window: the activity series and calendar features (``b["x"]``), plus a bag of the
window's campaigns (sum of campaign embeddings weighted by log-impressions), plus a learned lag
embedding. Each head attends with its own full-width Q/K/V, and the heads are mixed with
customer-specific weights ``omega`` (the paper's heterogeneity):

* ``per_user``  (the paper): a learned weight vector per training customer; customers not in the
  training sample (all of them under the user-disjoint protocol) get the mean of the training
  customers' vectors;
* ``uniform``   (ablation): equal weights, i.e. no customer-specific weighting;
* ``amortized`` (ablation): weights predicted from the customer's pooled history, so unseen
  customers get their own weights too.

Conversion from the token at window t plus the static features; campaigns from a linear head
plus a learned multiple of the recency-weighted history term (``copy_history``).

Same architecture and defaults as notebooks/colab/training/Criteo_LuKannan_Colab.ipynb.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

WEIGHTINGS = ("per_user", "uniform", "amortized")


class MixtureHeadAttention(nn.Module):
    """Each head attends with its own full-width Q/K/V; heads are mixed with per-customer weights."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h, self.d = n_heads, d
        self.qkv = nn.Linear(d, 3 * n_heads * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad, omega):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, d).permute(2, 0, 3, 1, 4)     # each (B, H, L, d)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        att = att.masked_fill(pad[:, None, None, :], float("-inf"))
        z = self.drop(att.softmax(-1)) @ v                                        # (B, H, L, d)
        z = (omega[:, :, None, None].to(z.dtype) * z).sum(1)                      # mixture over heads
        return self.out(z)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MixtureHeadAttention(d, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad, omega):
        x = x + self.drop(self.att(self.n1(x), pad, omega))
        return x + self.drop(self.ff(self.n2(x)))


class LuKannanTransformer(nn.Module):
    def __init__(self, n_num: int, n_static: int, n_camp: int, n_users: int, n_heads: int = 4,
                 weighting: str = "per_user", lookback: int = 14, d: int = 64, layers: int = 2,
                 dropout: float = 0.1, copy_history: bool = True):
        super().__init__()
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        self.n_heads, self.weighting = n_heads, weighting
        self.num_proj = nn.Linear(n_num, d)
        self.camp_emb = nn.EmbeddingBag(n_camp, d, mode="sum")
        self.lag_emb = nn.Parameter(torch.zeros(lookback, d))
        self.blocks = nn.ModuleList([Block(d, n_heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        if weighting == "per_user":
            self.theta = nn.Embedding(n_users, n_heads)
            nn.init.zeros_(self.theta.weight)                                     # start from uniform weights
            self.register_buffer("seen", torch.zeros(n_users, dtype=torch.bool))
        elif weighting == "amortized":
            self.omega_net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_heads))
        self.conv_head = nn.Sequential(nn.Linear(d + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.camp_head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_camp))
        self.copy_scale = nn.Parameter(torch.tensor(1.0)) if copy_history else None

    def mark_seen(self, users) -> None:
        """Customers in the training sample (per_user only); the others get the mean weights at eval."""
        if self.weighting == "per_user":
            self.seen[torch.as_tensor(users, dtype=torch.long, device=self.seen.device)] = True

    def omega(self, tok, b):
        B = tok.shape[0]
        if self.weighting == "uniform" or self.n_heads == 1:
            return torch.full((B, self.n_heads), 1.0 / self.n_heads, device=tok.device)
        if self.weighting == "per_user":
            th = self.theta(b["u"])
            if not self.training:                                                 # unseen users -> population mean
                mean = self.theta.weight[self.seen].mean(0) if self.seen.any() else torch.zeros_like(th[0])
                th = torch.where(self.seen[b["u"]][:, None], th, mean)
            return th.softmax(-1)
        keep = (~b["pad"]).unsqueeze(-1).to(tok.dtype)                            # amortized: from history
        pooled = (tok * keep).sum(1) / keep.sum(1).clamp_min(1)
        return self.omega_net(pooled).softmax(-1)

    def forward(self, b: dict):
        B, L, _ = b["x"].shape
        bag = self.camp_emb(b["camp"], b["off"], per_sample_weights=b["wt"]).view(B, L, -1)
        tok = self.num_proj(b["x"]) + bag + self.lag_emb
        om = self.omega(tok, b)
        for blk in self.blocks:
            tok = blk(tok, b["pad"], om)
        h = self.norm(tok[:, -1])                                                 # representation at window t
        camp_logits = self.camp_head(h).float()
        if self.copy_scale is not None:                                           # pointer-style history term
            hist = torch.zeros_like(camp_logits)
            hist.index_put_((b["h_row"], b["camp"]), b["h_w"], accumulate=True)
            camp_logits = camp_logits + self.copy_scale * hist
        return self.conv_head(torch.cat([h, b["s"]], 1)).squeeze(-1), camp_logits, om
