"""Training and prediction loop shared by the window-based sequence models.

A model's ``forward(batch)`` returns ``(conversion_logit, campaign_logits, extra)``. Loss: binary
cross-entropy on conversion plus ``campaign_loss_weight`` x multi-label BCE on next-window
campaigns. Early stopping on validation-sample PR-AUC (sampled like training). Same procedure
as the Colab notebooks.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


@dataclass
class TrainConfig:
    epochs: int = 10
    patience: int = 2
    batch: int = 1024
    eval_batch: int = 16384
    lr: float = 1e-3
    weight_decay: float = 1e-4
    campaign_loss_weight: float = 0.1
    grad_clip: float = 1.0


def autocast(device: str, amp: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if (device == "cuda" and amp) else nullcontext()


def predict(model, data, idx: np.ndarray, cfg: TrainConfig, device: str, amp: bool, campaign: bool = False):
    """Sampling-corrected conversion probabilities (and campaign scores if ``campaign``)."""
    model.eval()
    conv, camp = np.empty(len(idx), dtype=np.float64), []
    with torch.no_grad():
        for s in range(0, len(idx), cfg.eval_batch):
            with autocast(device, amp):
                lc, lk, _ = model(data.make_batch(idx[s:s + cfg.eval_batch], device=device))
            conv[s:s + cfg.eval_batch] = torch.sigmoid(lc.float()).cpu().numpy()
            if campaign:
                camp.append(torch.sigmoid(lk.float()).cpu().numpy())
    return data.to_true_prob(conv), (np.concatenate(camp) if campaign else None)


def train(model, data, tr: np.ndarray, val_idx: np.ndarray, cfg: TrainConfig, seed: int,
          device: str, amp: bool, log=print):
    """Train in place; restores and returns the best state dict (by validation-sample PR-AUC)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    vs = data.val_sample(val_idx)
    rs = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        perm = tr[rs.permutation(len(tr))]
        t0, tot = time.time(), 0.0
        for s in range(0, len(perm), cfg.batch):
            b = data.make_batch(perm[s:s + cfg.batch], with_labels=True, device=device)
            with autocast(device, amp):
                lc, lk, _ = model(b)
            yc = torch.zeros(lk.shape, device=device)
            yc[b["yc_r"], b["yc_c"]] = 1.0
            loss = F.binary_cross_entropy_with_logits(lc.float(), b["y"]) + \
                cfg.campaign_loss_weight * F.binary_cross_entropy_with_logits(lk.float(), yc, reduction="sum") / len(yc)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item() * len(yc)
        p_vs, _ = predict(model, data, vs, cfg, device, amp)
        ap = average_precision_score(data.y[vs], p_vs) if data.y[vs].any() else 0.0
        log(f"    epoch {ep + 1}: loss {tot / max(len(perm), 1):.4f} | val-sample PR-AUC {ap:.4f} | "
            f"{time.time() - t0:.0f}s")
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return best_state
