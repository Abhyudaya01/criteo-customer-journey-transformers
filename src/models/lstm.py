"""LSTM conversion baseline on the same window inputs as the Transformers.

One LSTM over the 7 activity series of the lookback windows (oldest first); conversion from the
last hidden state plus the static features. Conversion only (no campaign head): campaign
baselines are the popularity / recency / history rankings in the baselines notebook.

Same architecture as ``LSTMNet`` in notebooks/colab/training/Criteo_Baselines_Colab.ipynb. The
notebook trained it with Adam (lr 1e-3, no weight decay, no gradient clipping), batch 2048, fp32
and an 8-epoch cap; scripts/train_transformer.py uses the same settings except a 10-epoch cap
(patience 2), the same budget as the Transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, n_feat: int, n_static: int, hidden: int = 64):
        super().__init__()
        self.n_feat = n_feat
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, b: dict):
        _, (h, _) = self.lstm(b["x"][..., :self.n_feat])
        return self.head(torch.cat([h[-1], b["s"]], dim=1)).squeeze(-1), None, None
