#!/usr/bin/env python3
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
HOURS = 12
WINDOW_SECONDS = HOURS * 3600
MAX_LEN = 64
NEG_RATE = 0.01
VAL_SIZE = 200_000
BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
GRAD_ACCUM = 4
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
CHECKPOINT_EVERY = 500
CAT_COLS = [f"cat{i}" for i in range(1, 10)]
CAT_ID_COLS = [f"cat{i}_id" for i in range(1, 10)]
SAFE_RAW_COLS = ["timestamp", "uid", "campaign", "click", "cost", *CAT_COLS]


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_anchors(proc, cache_dir, neg_rate=NEG_RATE, val_size=VAL_SIZE):
    train_cache = cache_dir / f"train_anchors_neg_rate_{neg_rate:.4f}.parquet"
    val_cache = cache_dir / f"val_anchors_{val_size}.parquet"
    if train_cache.exists() and val_cache.exists():
        print("Loading cached anchors", flush=True)
        return pd.read_parquet(train_cache), pd.read_parquet(val_cache)

    shards = sorted((proc / "positions").glob("*.parquet"))
    if len(shards) != 64:
        print(f"WARNING: expected 64 position shards, found {len(shards)}", flush=True)

    total_train = 85_428_700
    total_train_pos = 158_834
    total_train_neg = total_train - total_train_pos
    neg_target = int(round(total_train_neg * neg_rate))
    neg_frac = min(1.0, (neg_target / total_train_neg) * 1.02)

    total_val = 20_421_186
    val_frac = min(1.0, (val_size / total_val) * 1.03)

    pos_parts, neg_parts, val_parts = [], [], []
    cols = ["uid", "window", "label_window", "y_conversion", "time_split"]

    for si, path in enumerate(shards):
        df = pd.read_parquet(path, columns=cols)
        split = df["time_split"].astype(str)

        tr = df.loc[split.eq("train")]
        p = tr.loc[tr["y_conversion"].eq(1)]
        n = tr.loc[tr["y_conversion"].eq(0)]
        if len(p):
            pos_parts.append(p)
        if len(n):
            take = int(math.ceil(len(n) * neg_frac))
            if take > 0:
                neg_parts.append(n.sample(n=min(take, len(n)), random_state=SEED + si))

        va = df.loc[split.eq("val")]
        if len(va):
            take = int(math.ceil(len(va) * val_frac))
            if take > 0:
                val_parts.append(va.sample(n=min(take, len(va)), random_state=10_000 + SEED + si))

        if (si + 1) % 8 == 0 or si + 1 == len(shards):
            print(f"anchor scan {si+1}/{len(shards)}", flush=True)

    train_pos = pd.concat(pos_parts, ignore_index=True)
    train_neg = pd.concat(neg_parts, ignore_index=True)
    if len(train_neg) >= neg_target:
        train_neg = train_neg.sample(n=neg_target, random_state=SEED)
    else:
        print(f"WARNING: only sampled {len(train_neg):,}/{neg_target:,} target negatives", flush=True)

    train = pd.concat([train_pos, train_neg], ignore_index=True)
    train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    val = pd.concat(val_parts, ignore_index=True)
    if len(val) > val_size:
        val = val.sample(n=val_size, random_state=SEED).reset_index(drop=True)

    for df in (train, val):
        df["anchor_timestamp"] = df["label_window"].astype(np.int64) * WINDOW_SECONDS

    print(f"Train anchors: {len(train):,}, positives={int(train.y_conversion.sum()):,}, rate={train.y_conversion.mean():.6f}", flush=True)
    print(f"Val anchors:   {len(val):,}, positives={int(val.y_conversion.sum()):,}, rate={val.y_conversion.mean():.6f}", flush=True)
    train.to_parquet(train_cache, index=False)
    val.to_parquet(val_cache, index=False)
    return train, val


def build_selected_events(raw_path, train, val, cache_dir):
    event_cache = cache_dir / "selected_events.parquet"
    meta_cache = cache_dir / "vocab_meta.json"
    if event_cache.exists() and meta_cache.exists():
        print("Loading cached selected events", flush=True)
        events = pd.read_parquet(event_cache)
        with open(meta_cache) as f:
            meta = json.load(f)
        return events, meta

    required_uids = np.unique(np.concatenate([
        train["uid"].to_numpy(np.int64),
        val["uid"].to_numpy(np.int64),
    ]))
    required_set = set(map(int, required_uids.tolist()))
    print(f"Required users: {len(required_uids):,}", flush=True)

    vocab_sets = {"campaign": set(), **{c: set() for c in CAT_COLS}}
    keep_parts = []
    train_cutoff = 43 * WINDOW_SECONDS  # latest training label window anchor

    reader = pd.read_csv(raw_path, sep="\t", usecols=SAFE_RAW_COLS, chunksize=1_000_000)
    for ci, chunk in enumerate(reader, start=1):
        train_rows = chunk.loc[chunk["timestamp"] < train_cutoff]
        for c in ["campaign", *CAT_COLS]:
            vals = train_rows[c].dropna().unique().tolist()
            vocab_sets[c].update(vals)

        keep = chunk["uid"].isin(required_set)
        part = chunk.loc[keep, SAFE_RAW_COLS].copy()
        if len(part):
            keep_parts.append(part)
        print(f"raw chunk {ci}: kept {len(part):,}", flush=True)

    events = pd.concat(keep_parts, ignore_index=True)
    del keep_parts
    print(f"Selected raw events: {len(events):,}", flush=True)

    campaign_values = sorted(vocab_sets["campaign"])
    campaign_map = {v: i + 1 for i, v in enumerate(campaign_values)}
    events["entity_id"] = events["campaign"].map(campaign_map).fillna(0).astype(np.int32)
    events["raw_action_id"] = np.where(events["click"].to_numpy() > 0, 2, 1).astype(np.int8)
    events["behavior_family"] = events["raw_action_id"].astype(np.int8)

    context_vocab_sizes = {}
    for c in CAT_COLS:
        values = sorted(vocab_sets[c])
        mapping = {v: i + 1 for i, v in enumerate(values)}
        events[f"{c}_id"] = events[c].map(mapping).fillna(0).astype(np.int32)
        context_vocab_sizes[c] = len(values) + 1

    events["log_cost"] = np.log1p(np.maximum(events["cost"].to_numpy(np.float64), 0.0) * 1000.0).astype(np.float32)
    events = events.sort_values(["uid", "timestamp"], kind="mergesort").reset_index(drop=True)

    prev_ts = events.groupby("uid", sort=False)["timestamp"].shift(1)
    delta_h = ((events["timestamp"] - prev_ts) / 3600.0).fillna(0.0).clip(lower=0)
    events["log_delta_hours"] = np.log1p(delta_h).astype(np.float32)

    prev_entity_ts = events.groupby(["uid", "entity_id"], sort=False)["timestamp"].shift(1)
    seen = prev_entity_ts.notna()
    gap_h = ((events["timestamp"] - prev_entity_ts) / 3600.0).fillna(0.0).clip(lower=0)
    events["entity_seen_before"] = seen.astype(np.float32)
    events["log_entity_gap_hours"] = np.log1p(gap_h).astype(np.float32)

    keep_cols = [
        "uid", "timestamp", "entity_id", "raw_action_id", "behavior_family",
        "log_cost", "log_delta_hours", "entity_seen_before", "log_entity_gap_hours",
        *CAT_ID_COLS,
    ]
    events = events[keep_cols]
    events.to_parquet(event_cache, index=False)

    meta = {
        "n_entities": len(campaign_values) + 1,
        "context_vocab_sizes": context_vocab_sizes,
    }
    with open(meta_cache, "w") as f:
        json.dump(meta, f, indent=2)
    print("Vocab meta:", meta, flush=True)
    return events, meta


class EventStore:
    def __init__(self, events):
        self.uid = events["uid"].to_numpy(np.int64, copy=True)
        self.timestamp = events["timestamp"].to_numpy(np.int64, copy=True)
        self.entity_id = events["entity_id"].to_numpy(np.int64, copy=True)
        self.raw_action_id = events["raw_action_id"].to_numpy(np.int64, copy=True)
        self.behavior_family = events["behavior_family"].to_numpy(np.int64, copy=True)
        self.log_cost = events["log_cost"].to_numpy(np.float32, copy=True)
        self.log_delta_hours = events["log_delta_hours"].to_numpy(np.float32, copy=True)
        self.entity_seen_before = events["entity_seen_before"].to_numpy(np.float32, copy=True)
        self.log_entity_gap_hours = events["log_entity_gap_hours"].to_numpy(np.float32, copy=True)
        self.context = events[CAT_ID_COLS].to_numpy(np.int64, copy=True)
        unique, starts, counts = np.unique(self.uid, return_index=True, return_counts=True)
        self.ranges = {int(u): (int(s), int(s+c)) for u, s, c in zip(unique, starts, counts)}
        print(f"Indexed {len(self.ranges):,} user histories", flush=True)


class JourneyDataset(Dataset):
    def __init__(self, anchors, store, max_len=MAX_LEN):
        self.store = store
        self.max_len = max_len
        ok = anchors["uid"].isin(store.ranges)
        if not ok.all():
            print(f"WARNING: dropping {(~ok).sum():,} anchors without event history", flush=True)
        a = anchors.loc[ok].reset_index(drop=True)
        self.uid = a["uid"].to_numpy(np.int64)
        self.anchor_ts = a["anchor_timestamp"].to_numpy(np.int64)
        self.y = a["y_conversion"].to_numpy(np.float32)

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = int(self.uid[idx])
        anchor = int(self.anchor_ts[idx])
        s, e = self.store.ranges[uid]
        ts_all = self.store.timestamp[s:e]
        end_local = np.searchsorted(ts_all, anchor, side="left")
        start_local = max(0, end_local - self.max_len)
        s2, e2 = s + start_local, s + end_local
        n = e2 - s2

        L = self.max_len
        entity = np.zeros(L, np.int64)
        action = np.zeros(L, np.int64)
        behavior = np.zeros(L, np.int64)
        context = np.zeros((L, 9), np.int64)
        numeric = np.zeros((L, 5), np.float32)
        timestamp = np.zeros(L, np.int64)
        mask = np.zeros(L, bool)

        if n > 0:
            off = L - n
            sl = slice(s2, e2)
            ts = self.store.timestamp[sl]
            entity[off:] = self.store.entity_id[sl]
            action[off:] = self.store.raw_action_id[sl]
            behavior[off:] = self.store.behavior_family[sl]
            context[off:] = self.store.context[sl]
            timestamp[off:] = ts
            age = np.log1p(np.maximum((anchor - ts) / 3600.0, 0.0)).astype(np.float32)
            numeric[off:, 0] = self.store.log_cost[sl]
            numeric[off:, 1] = age
            numeric[off:, 2] = self.store.log_delta_hours[sl]
            numeric[off:, 3] = self.store.entity_seen_before[sl]
            numeric[off:, 4] = self.store.log_entity_gap_hours[sl]
            mask[off:] = True

        return {
            "entity_id": torch.from_numpy(entity),
            "raw_action_id": torch.from_numpy(action),
            "behavior_family": torch.from_numpy(behavior),
            "context": torch.from_numpy(context),
            "numeric": torch.from_numpy(numeric),
            "timestamp": torch.from_numpy(timestamp),
            "mask": torch.from_numpy(mask),
            "y_conversion": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class FieldAwareContextEncoder(nn.Module):
    def __init__(self, vocab_sizes, d_field=16, d_out=32):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(v, d_field, padding_idx=0) for v in vocab_sizes])
        self.field_emb = nn.Embedding(len(vocab_sizes), d_field)
        self.score = nn.Sequential(nn.Linear(d_field, d_field), nn.Tanh(), nn.Linear(d_field, 1))
        self.out = nn.Linear(d_field, d_out)

    def forward(self, x):
        zs = []
        for f, emb in enumerate(self.emb):
            z = emb(x[:, :, f]) + self.field_emb.weight[f].view(1, 1, -1)
            zs.append(z)
        z = torch.stack(zs, dim=2)
        w = torch.softmax(self.score(z).squeeze(-1), dim=-1)
        return self.out((w.unsqueeze(-1) * z).sum(dim=2))


class BehaviorRouter(nn.Module):
    def __init__(self, d_model=128, n_experts=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        r = torch.softmax(self.router(x), dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=-2)
        return self.norm(x + (eo * r.unsqueeze(-1)).sum(dim=-2))


class RelativeTimeAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_buckets=32, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_buckets = n_buckets
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.time_bias = nn.Embedding(n_buckets, n_heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, timestamps, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        ti = timestamps.unsqueeze(2)
        tj = timestamps.unsqueeze(1)
        gap_h = (ti - tj).clamp(min=0).float() / 3600.0
        buckets = torch.floor(torch.log2(gap_h + 1.0)).long().clamp(max=self.n_buckets - 1)
        scores = scores + self.time_bias(buckets).permute(0, 3, 1, 2)

        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        allowed = causal[None, None, :, :] & mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, -1e4)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out) * mask.unsqueeze(-1)


class TimeAwareBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = RelativeTimeAttention(d_model, n_heads, dropout=dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, timestamps, mask):
        x = x + self.attn(self.n1(x), timestamps, mask)
        x = x + self.ff(self.n2(x))
        return x * mask.unsqueeze(-1)


class AttentionPool(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(s, dim=-1)
        return (w.unsqueeze(-1) * x).sum(dim=1)


class EntityMemory(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.repeat_bonus = nn.Parameter(torch.tensor(1.0))
        self.recency_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, numeric, mask):
        repeat = numeric[:, :, 3]
        gap = numeric[:, :, 4]
        s = self.score(h).squeeze(-1) + self.repeat_bonus * repeat - self.recency_weight.abs() * gap
        repeated = mask & (repeat > 0)
        has_repeat = repeated.any(dim=1)
        mmask = torch.where(has_repeat[:, None], repeated, mask)
        s = s.masked_fill(~mmask, -1e4)
        w = torch.softmax(s, dim=-1)
        return (w.unsqueeze(-1) * h).sum(dim=1)


class STAJT(nn.Module):
    def __init__(self, n_entities, context_sizes, d_model=128, n_heads=4, n_layers=2, n_experts=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.action_emb = nn.Embedding(3, 16, padding_idx=0)
        self.behavior_emb = nn.Embedding(5, 16, padding_idx=0)
        self.entity_emb = nn.Embedding(n_entities, 32, padding_idx=0)
        self.context_encoder = FieldAwareContextEncoder(context_sizes, 16, 32)
        self.numeric_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 16))
        self.time_mlp = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16))
        self.event_proj = nn.Sequential(nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.router = BehaviorRouter(d_model, n_experts, d_ff, dropout)
        self.blocks = nn.ModuleList([TimeAwareBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.pool = AttentionPool(d_model)
        self.memory = EntityMemory(d_model)
        self.behavior_proj = nn.Linear(16, d_model)
        self.context_proj = nn.Linear(32, d_model)
        self.time_proj = nn.Linear(16, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.modality_mixer = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    @staticmethod
    def masked_mean(x, mask):
        w = mask.float().unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def forward(self, b):
        mask = b["mask"]
        action = self.action_emb(b["raw_action_id"])
        behavior = self.behavior_emb(b["behavior_family"])
        entity = self.entity_emb(b["entity_id"])
        context = self.context_encoder(b["context"])
        nr = b["numeric"]
        numeric = self.numeric_mlp(torch.stack([nr[:, :, 0], nr[:, :, 3], nr[:, :, 4]], dim=-1))
        time_emb = self.time_mlp(nr[:, :, 1:3])
        x = self.event_proj(torch.cat([action, behavior, entity, context, numeric, time_emb], dim=-1)) * mask.unsqueeze(-1)
        x = self.router(x) * mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, b["timestamp"], mask)
        journey = self.pool(x, mask)
        memory = self.memory(x, nr, mask)
        bsum = self.behavior_proj(self.masked_mean(behavior, mask))
        csum = self.context_proj(self.masked_mean(context, mask))
        tsum = self.time_proj(self.masked_mean(time_emb, mask))
        toks = torch.stack([journey, memory, bsum, csum, tsum], dim=1)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = self.modality_mixer(torch.cat([cls, toks], dim=1))[:, 0]
        return self.head(z).squeeze(-1)


def gain_at_10(y, p):
    k = max(1, int(math.ceil(0.10 * len(y))))
    idx = np.argpartition(-p, k-1)[:k]
    total = y.sum()
    return float(y[idx].sum() / total) if total > 0 else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(b)
        ys.append(b["y_conversion"].cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    g10 = gain_at_10(y, p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "gain_at_10": g10,
        "lift_at_10": g10 / 0.10,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
    }


def save_ckpt(path, model, optimizer, scaler, epoch, batch_idx, history, meta, config):
    torch.save({
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "meta": meta,
        "config": config,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    seed_all()
    root = Path(__file__).resolve().parents[1]
    proc = root / "data/processed/criteo_w12h_full"
    raw = root / "data/raw/criteo_attribution_dataset/criteo_attribution_dataset.tsv.gz"
    cache_dir = root / "data/ujt_fair_chrono_cache"
    ckpt_dir = root / "checkpoints/ujt_fair_chrono"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run inside the PyTorch CUDA module environment.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("PyTorch", torch.__version__, flush=True)
    print("GPU", torch.cuda.get_device_name(0), flush=True)
    print("VRAM_GB", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), flush=True)

    train, val = sample_anchors(proc, cache_dir)
    events, meta = build_selected_events(raw, train, val, cache_dir)
    store = EventStore(events)
    del events

    # Causality sanity check.
    checked = 0
    for row in train.head(5000).itertuples(index=False):
        s, e = store.ranges[int(row.uid)]
        ts = store.timestamp[s:e]
        end_local = np.searchsorted(ts, int(row.anchor_timestamp), side="left")
        if end_local > 0:
            assert ts[end_local - 1] < int(row.anchor_timestamp)
        checked += 1
    print(f"Causality check passed on {checked} anchors", flush=True)

    train_ds = JourneyDataset(train, store, MAX_LEN)
    val_ds = JourneyDataset(val, store, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    context_sizes = [meta["context_vocab_sizes"][f"cat{i}"] for i in range(1, 10)]
    model = STAJT(meta["n_entities"], context_sizes).to(device)
    print("Parameters", f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    # One-batch GPU memory smoke test.
    smoke = next(iter(train_loader))
    smoke = {k: v.to(device, non_blocking=True) for k, v in smoke.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.amp.autocast(enabled=True):
        smoke_logits = model(smoke)
        smoke_loss = F.binary_cross_entropy_with_logits(smoke_logits, smoke["y_conversion"])
    smoke_loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Smoke loss={smoke_loss.item():.6f}, peak_allocated={peak:.2f} GB", flush=True)
    model.zero_grad(set_to_none=True)
    del smoke, smoke_logits, smoke_loss
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history = []
    start_epoch = 1
    best_pr = -1.0

    if args.resume:
        c = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(c["model_state_dict"])
        optimizer.load_state_dict(c["optimizer_state_dict"])
        scaler.load_state_dict(c["scaler_state_dict"])
        history = c.get("history", [])
        start_epoch = int(c.get("epoch", 1)) + (1 if c.get("batch_idx", None) is None else 0)
        if history:
            best_pr = max(float(r["pr_auc"]) for r in history if "pr_auc" in r)
        print(f"Resumed from {args.resume}; start_epoch={start_epoch}", flush=True)

    config = {
        "max_len": MAX_LEN,
        "neg_rate": NEG_RATE,
        "batch_size": BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    }

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for bi, b in enumerate(train_loader, start=1):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(b)
                loss = F.binary_cross_entropy_with_logits(logits, b["y_conversion"])
                scaled = loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {bi}")
            scaler.scale(scaled).backward()

            if bi % GRAD_ACCUM == 0 or bi == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            if bi % 100 == 0 or bi == 1:
                print(f"epoch={epoch:02d} batch={bi:05d}/{len(train_loader):05d} loss={loss.item():.5f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
            if bi % CHECKPOINT_EVERY == 0:
                save_ckpt(ckpt_dir / "ujt_criteo_inprogress.pt", model, optimizer, scaler, epoch, bi, history, meta, config)
                print("checkpoint: inprogress", flush=True)

        train_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "minutes": (time.time()-t0)/60}
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        save_ckpt(ckpt_dir / f"ujt_criteo_epoch_{epoch:02d}.pt", model, optimizer, scaler, epoch, None, history, meta, config)
        if metrics["pr_auc"] > best_pr:
            best_pr = metrics["pr_auc"]
            save_ckpt(ckpt_dir / "ujt_criteo_best.pt", model, optimizer, scaler, epoch, None, history, meta, config)
            print(f"NEW BEST PR-AUC {best_pr:.6f}", flush=True)
        pd.DataFrame(history).to_csv(ckpt_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
