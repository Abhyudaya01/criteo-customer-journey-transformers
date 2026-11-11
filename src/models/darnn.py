"""DARNN (Ren et al., 2018) for next-window conversion, under our chronological protocol.

Dual-attention RNN: an encoder LSTM reads impression features x_j; a decoder LSTM reads the
previous click z_{j-1} (plus the aligned encoder state) and predicts click z_j; impression-level
attention over encoder states and click-level attention over decoder states are combined with a
learned weight lambda, and the combined vector predicts conversion. See docs/darnn_protocol.md for
what is faithful and what is adapted.

Inputs are ``src.data.events.darnn_item`` batches (left-padded, oldest event first):
    campaign [B, L] int, context [B, L, 9] int, numeric [B, L, 5] float, click [B, L] float,
    mask [B, L] bool, n_events [B] int.

``last_click_step=True`` (default, our adaptation) runs the decoder one step past the last event
so it also consumes z_m, the click on the most recent event, which every other model sees. With
``False`` the decoder never consumes z_m, as in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_NUMERIC = 5
N_CONTEXT = 9
START, NO_CLICK, CLICK = 0, 1, 2  # decoder input tokens


def left_to_right_padded(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Move each row's last ``n`` items (left-padded) to the front (right-padded). Dim 1 is time."""
    B, L = x.shape[:2]
    idx = (torch.arange(L, device=x.device)[None, :] + (L - n)[:, None]) % L  # [B, L]
    idx = idx.view(B, L, *([1] * (x.dim() - 2))).expand_as(x)
    return torch.gather(x, 1, idx)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    w = torch.softmax(scores, dim=-1)
    return w * mask  # rows with no real items get all-zero weights


class Attention(nn.Module):
    """Additive attention: a_j = softmax(v' tanh(W h_j)) over real positions."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        a = masked_softmax(self.v(torch.tanh(self.proj(h))).squeeze(-1), mask)
        return torch.einsum("bl,bld->bd", a, h), a


class DARNN(nn.Module):
    def __init__(self, n_entities: int, context_sizes: list[int], d_model: int = 128,
                 d_campaign: int = 32, d_field: int = 8, d_numeric: int = 16, d_click: int = 16,
                 dropout: float = 0.1, last_click_step: bool = True):
        super().__init__()
        if len(context_sizes) != N_CONTEXT:
            raise ValueError(f"expected {N_CONTEXT} context vocab sizes, got {len(context_sizes)}")
        self.last_click_step = last_click_step
        self.campaign_emb = nn.Embedding(n_entities, d_campaign, padding_idx=0)
        self.context_emb = nn.ModuleList([nn.Embedding(v, d_field, padding_idx=0) for v in context_sizes])
        self.numeric = nn.Linear(N_NUMERIC, d_numeric)
        self.input_proj = nn.Sequential(
            nn.Linear(d_campaign + N_CONTEXT * d_field + d_numeric, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.click_emb = nn.Embedding(3, d_click)
        self.decoder = nn.LSTM(d_model + d_click, d_model, batch_first=True)
        self.click_head = nn.Linear(d_model, 1)
        self.att_impression = Attention(d_model)
        self.att_click = Attention(d_model)
        self.lam_logit = nn.Parameter(torch.zeros(1))  # lambda = sigmoid(.) starts at 0.5
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    @property
    def lam(self) -> torch.Tensor:
        return torch.sigmoid(self.lam_logit)

    def forward(self, b: dict) -> dict:
        n = b["n_events"].long().clamp(min=1)  # every position has >= 1 event; guard anyway
        B, L = b["campaign"].shape
        dev = b["campaign"].device
        # Right-pad so the LSTMs run from each user's first real event.
        campaign = left_to_right_padded(b["campaign"].long(), n)
        context = left_to_right_padded(b["context"].long(), n)
        numeric = left_to_right_padded(b["numeric"].float(), n)
        click = left_to_right_padded(b["click"].float(), n)
        steps = torch.arange(L, device=dev)[None, :]
        mask = steps < n[:, None]  # [B, L] real events

        x = torch.cat([self.campaign_emb(campaign),
                       *[emb(context[..., i]) for i, emb in enumerate(self.context_emb)],
                       self.numeric(numeric)], dim=-1)
        x = self.input_proj(x)
        packed = pack_padded_sequence(x, n.cpu(), batch_first=True, enforce_sorted=False)
        h_packed, (h_n, c_n) = self.encoder(packed)
        h, _ = pad_packed_sequence(h_packed, batch_first=True, total_length=L)  # [B, L, d]

        # Decoder step j reads h_j and the previous click z_{j-1} (START at j = 0) and predicts z_j.
        # With last_click_step, one more step (j = n) reads h_{n-1} again and z_{n-1}.
        tok = ((click > 0.5).long() + NO_CLICK) * mask  # NO_CLICK = 1, CLICK = 2, padding = 0
        prev = torch.cat([torch.full((B, 1), START, device=dev, dtype=torch.long), tok], dim=1)  # [B, L+1]
        h_ext = torch.cat([h, torch.zeros_like(h[:, :1])], dim=1)                                  # [B, L+1, d]
        last = h[torch.arange(B, device=dev), n - 1]                                               # [B, d]
        h_ext = h_ext.scatter(1, n.view(B, 1, 1).expand(B, 1, h.size(-1)), last[:, None, :])
        dec_len = n + 1 if self.last_click_step else n
        dec_in = torch.cat([h_ext, self.click_emb(prev)], dim=-1)
        d_packed = pack_padded_sequence(dec_in, dec_len.cpu(), batch_first=True, enforce_sorted=False)
        s_packed, _ = self.decoder(d_packed, (h_n, c_n))
        s, _ = pad_packed_sequence(s_packed, batch_first=True, total_length=L + 1)  # [B, L+1, d]

        click_logit = self.click_head(s[:, :L]).squeeze(-1)  # z_j from s_j, j < n
        steps_ext = torch.arange(L + 1, device=dev)[None, :]
        dec_mask = steps_ext < dec_len[:, None]

        c_imp, a_imp = self.att_impression(h, mask)
        c_clk, a_clk = self.att_click(s, dec_mask)
        lam = self.lam
        c = lam * c_imp + (1 - lam) * c_clk
        return {
            "conv_logit": self.head(c).squeeze(-1),
            "click_logit": click_logit, "click_target": click, "click_mask": mask,
            "att_impression": a_imp, "att_click": a_clk,
        }


def darnn_loss(out: dict, y: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, dict]:
    """L = L_conversion + alpha * L_click (click BCE averaged over real events)."""
    conv = F.binary_cross_entropy_with_logits(out["conv_logit"].float(), y.float())
    m = out["click_mask"]
    click = F.binary_cross_entropy_with_logits(out["click_logit"].float()[m], out["click_target"][m])
    return conv + alpha * click, {"conversion": float(conv.detach()), "click": float(click.detach())}
