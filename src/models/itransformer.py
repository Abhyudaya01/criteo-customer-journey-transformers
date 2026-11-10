"""iTransformer for next-window conversion and campaign prediction (Liu et al., ICLR 2024).

Inverted attention: one token per *variable* (the whole lookback series of that variable),
attention across variables rather than across time. Variables: the ``k`` most frequent training
campaigns (per-window log-impressions), one "other campaigns" token, and the 7 activity series of
``src/training/window_data.FEATS``. A learned variable-identity embedding is added because many
sparse campaign series are all zero.

Heads: conversion from mean+max pooled tokens plus the static features; campaigns from a pooled
linear head, plus a per-token score for each top campaign, plus a learned multiple of the
recency-weighted history term (``copy_history``), as in the other Transformers.

Same architecture and defaults as notebooks/colab/training/Criteo_iTransformer_Colab.ipynb.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ITransformer(nn.Module):
    def __init__(self, top_idx, n_static: int, n_camp: int, n_heads: int = 4, lookback: int = 14,
                 n_activity: int = 7, d: int = 64, layers: int = 2, dropout: float = 0.1,
                 copy_history: bool = True):
        super().__init__()
        self.k, self.n_heads, self.L, self.n_activity = len(top_idx), n_heads, lookback, n_activity
        tok = torch.full((n_camp,), self.k, dtype=torch.long)                 # campaign -> token ("other" = k)
        tok[torch.as_tensor(top_idx, dtype=torch.long)] = torch.arange(self.k)
        self.register_buffer("camp_tok", tok)
        self.register_buffer("top_idx", torch.as_tensor(top_idx, dtype=torch.long))
        self.n_var = self.k + 1 + n_activity
        self.var_proj = nn.Linear(lookback, d)                                # inverted embedding: series -> token
        self.var_emb = nn.Parameter(torch.randn(self.n_var, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.conv_head = nn.Sequential(nn.Linear(2 * d + n_static, 64), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(64, 1))
        self.camp_head = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(), nn.Linear(128, n_camp))
        self.tok_head = nn.Linear(d, 1)                                       # each top campaign scores itself
        self.copy_scale = nn.Parameter(torch.tensor(1.0)) if copy_history else None

    def campaign_series(self, b: dict, B: int) -> torch.Tensor:
        """(B, L, k+1): per-window log-impressions of each top campaign and of all others together."""
        lens = torch.diff(b["off"], append=torch.tensor([b["camp"].numel()], device=b["off"].device))
        flat_row = torch.repeat_interleave(torch.arange(B * self.L, device=lens.device), lens)
        series = torch.zeros(B * self.L, self.k + 1, device=lens.device)
        series.index_put_((flat_row, self.camp_tok[b["camp"]]), b["wt"], accumulate=True)
        return series.view(B, self.L, self.k + 1)

    def forward(self, b: dict):
        x = b["x"][..., :self.n_activity]                                     # (B, L, 7)
        B = x.shape[0]
        v = torch.cat([self.campaign_series(b, B), x], dim=-1)                # (B, L, V)
        z = self.var_proj(v.transpose(1, 2)) + self.var_emb                   # (B, V, d): one token per variable
        # Chunked: CUDA attention fails when sequences x heads exceeds 65,535 in one call.
        chunk = max(1, 32_768 // self.n_heads)
        z = self.norm(torch.cat([self.encoder(z[i:i + chunk]) for i in range(0, B, chunk)]))
        pooled = torch.cat([z.mean(1), z.max(1).values], dim=1)
        conv_logit = self.conv_head(torch.cat([pooled, b["s"]], 1)).squeeze(-1)
        camp_logits = self.camp_head(pooled).float()
        tok_scores = self.tok_head(z[:, :self.k]).squeeze(-1).float()         # (B, k)
        camp_logits = camp_logits.index_add(1, self.top_idx, tok_scores)
        if self.copy_scale is not None:
            hist = torch.zeros_like(camp_logits)
            hist.index_put_((b["h_row"], b["camp"]), b["h_w"], accumulate=True)
            camp_logits = camp_logits + self.copy_scale * hist
        return conv_logit, camp_logits, None
