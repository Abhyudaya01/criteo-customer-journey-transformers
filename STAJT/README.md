# STAJT: Sparse Time-Aware Journey Transformer

This directory contains the STAJT implementation and experiment artifacts for customer-journey conversion prediction on the Criteo Attribution dataset.

## Overview

STAJT models sparse, irregular customer journeys at the event level rather than treating each fixed 12-hour window as a single token. The architecture combines:

- field-aware contextual encoding,
- behavior-aware expert routing,
- relative-time self-attention,
- recency-aware campaign memory,
- attention pooling, and
- cross-modality fusion.

The current implementation uses up to 64 recent impression events before each prediction anchor and predicts next-window conversion probability.

## Training configurations

Two configurations are maintained for controlled evaluation:

- **Optimized STAJT:** all positive examples plus 10 sampled negatives per positive.
- **Team-aligned STAJT:** all positive examples plus 1% of available negatives, matching the common Tier-1 training protocol.

Checkpoint selection is performed using validation PR-AUC only; the held-out test set is not used for model selection.

## Current validation snapshot

On the full chronological validation population (20.42M prediction positions; 34.4K positives), optimized STAJT reached its best validation PR-AUC at epoch 4:

| Checkpoint | PR-AUC | ROC-AUC | Top-10% Gain | Top-10% Lift |
|---|---:|---:|---:|---:|
| Epoch 3 | 0.02467 | 0.88556 | 64.69% | 6.47x |
| **Epoch 4** | **0.02563** | 0.88616 | 64.84% | 6.48x |
| Epoch 5 | 0.02497 | **0.88656** | **65.07%** | **6.51x** |

Epoch 4 is selected because PR-AUC is the primary metric under the highly imbalanced conversion task.

## Reproducibility

Training and validation runs use persistent logs and epoch checkpoints. Full-population validation is kept separate from sampled training diagnostics. Large datasets and model checkpoints should remain outside normal Git tracking unless explicitly managed with an appropriate large-file workflow.

## Project scope

Planned extensions include component ablations, DARNN comparison, multiple random seeds, momentum-encoder contrastive pretraining, multi-scale temporal modeling, Mamba/attention hybrids, and cross-dataset evaluation on customer-behavior benchmarks.
