"""PatchTST for next-window conversion and campaign prediction (Nie et al., ICLR 2023).

Channel independence: each of the 7 activity series is cut into patches of ``patch_len`` windows
(step ``stride``), and one shared Transformer encoder reads each channel's patch sequence. The
encoded patches of all channels are flattened; conversion from that plus the static features,
campaigns from a linear head plus a learned multiple of the recency-weighted history term
(``copy_history``). Calendar features are in the static features, not in the channels.

Same architecture and defaults as notebooks/colab/training/Criteo_PatchTST_Colab.ipynb.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, n_channels: int, n_static: int, n_camp: int, patch_len: int = 2, stride: int = 2,
                 n_heads: int = 4, lookback: int = 14, d: int = 64, layers: int = 2, dropout: float = 0.1,
                 copy_history: bool = True):
        super().__init__()
        if lookback < patch_len:
            raise ValueError(f"lookback {lookback} is shorter than patch_len {patch_len}")
        self.c, self.p, self.s, self.n_heads = n_channels, patch_len, stride, n_heads
        self.n_patches = (lookback - patch_len) // stride + 1
        self.patch_emb = nn.Linear(patch_len, d)
        self.pos_emb = nn.Parameter(torch.randn(self.n_patches, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        flat = n_channels * self.n_patches * d
        self.flat_drop = nn.Dropout(dropout)
        self.conv_head = nn.Sequential(nn.Linear(flat + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.camp_head = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, n_camp))
        self.copy_scale = nn.Parameter(torch.tensor(1.0)) if copy_history else None

    def forward(self, b: dict):
        x = b["x"][..., :self.c]                                   # (B, L, C) activity series, oldest first
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L)                   # channel independence
        patches = x.unfold(-1, self.p, self.s)                     # (B*C, N, P)
        z = self.patch_emb(patches) + self.pos_emb
        # Chunked: CUDA attention kernels fail when sequences x heads exceeds 65,535 in one call.
        chunk = max(1, 32_768 // self.n_heads)
        z = torch.cat([self.encoder(z[i:i + chunk]) for i in range(0, z.shape[0], chunk)])
        z = self.norm(z)
        flat = self.flat_drop(z.reshape(B, -1))                    # flatten head: C * N * d
        conv_logit = self.conv_head(torch.cat([flat, b["s"]], 1)).squeeze(-1)
        camp_logits = self.camp_head(flat).float()
        if self.copy_scale is not None:
            hist = torch.zeros_like(camp_logits)
            hist.index_put_((b["h_row"], b["camp"]), b["h_w"], accumulate=True)
            camp_logits = camp_logits + self.copy_scale * hist
        return conv_logit, camp_logits, None
