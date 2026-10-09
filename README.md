# Transformer-Based Customer Journey Prediction on the Criteo Attribution Dataset

DATA612 Group 3 project.

**Group members:** Shashank Ashoka, Rachel Huang, Abhyudaya Lohani

## Project objective

Given a customer's historical sequence of advertising touchpoints, predict:

1. which campaigns the customer will interact with in the next time window; and
2. whether the customer will convert.

The primary research direction adapts the heterogeneous multi-head self-attention Transformer from Lu & Kannan (2026) to the public Criteo Attribution Modeling for Bidding dataset.

## Planned models

### Baselines
- Hidden Markov Model (HMM)
- Long Short-Term Memory network (LSTM)

### Transformer models
- Lu & Kannan-style heterogeneous multi-head Transformer
- PatchTST
- iTransformer

### Exploratory extensions
- Self-supervised temporal contrastive learning inspired by TS2Vec / MoCo
- LoRA / parameter-efficient fine-tuning experiments
- DeepCausalMMM on synthetic MMM data

## Primary dataset

**Criteo Attribution Modeling for Bidding**
- ~30 days of advertising traffic
- ~16.5 million impressions
- ~45,000 conversions
- ~700 anonymized campaigns

The raw dataset should **not** be committed to GitHub. Download it separately and place it under `data/raw/`.

## Data pipeline

Planned workflow:

```text
Raw Criteo logs
      |
      v
sort by user + timestamp
      |
      v
construct customer histories
      |
      v
aggregate into 12-hour windows
(and compare 6-hour windows)
      |
      v
build campaign interaction features
+ future conversion labels
      |
      v
chronological train/validation/test split
      |
      +--> user-disjoint evaluation
```

## Evaluation

### Conversion prediction
- PR-AUC
- ROC-AUC
- Balanced accuracy
- F1
- Binary cross-entropy
- Top-decile conversion rate
- Lift
- Gain

### Campaign prediction
- Precision@K
- Recall@K
- F1
- Binary cross-entropy

### Verification / ablations
- HMM and LSTM comparisons
- attention-head count (1-4)
- customer-specific weighting
- 6-hour vs. 12-hour windows
- multiple random seeds
- attention visualizations

## Repository structure

```text
.
├── configs/              # experiment configuration files
├── data/
│   ├── raw/              # downloaded source data (gitignored)
│   ├── interim/          # intermediate artifacts (gitignored)
│   └── processed/        # model-ready tensors/tables (gitignored)
├── docs/                 # proposal and project documentation
├── notebooks/            # EDA/figures only
├── results/
│   ├── figures/
│   └── tables/
├── scripts/              # runnable entry-point scripts
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── .gitignore
├── CONTRIBUTING.md
├── environment.yml
└── README.md
```

## Suggested team ownership

- **Shashank:** Transformer architecture, tuning, Zaratan runs, report/presentation lead
- **Rachel:** data preparation, optional MMM extension, baseline support
- **Abhyudaya:** repository/reproducibility, HMM/LSTM baselines, evaluation harness

Everyone should review pull requests and contribute to documentation.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate criteo-journey
```

For GPU training on Zaratan, install the PyTorch build that matches the CUDA/module environment available on the cluster.

## Reproducibility rules

- Do not edit raw data.
- Use configuration files for experiment settings.
- Save random seeds in every experiment config.
- Save metrics and figures under `results/`.
- Keep notebooks for exploration/figures only; production preprocessing and training should live in `src/` or `scripts/`.
- Once the environment is stable on Zaratan, export an exact lock file for final reproducibility.

## Git workflow

Use one branch per task:

```bash
git checkout -b feature/criteo-preprocessing
git checkout -b feature/lstm-baseline
git checkout -b feature/heterogeneous-transformer
```

Open a pull request before merging substantial changes into `main`.

## Key references

See [`docs/references.md`](docs/references.md).

## Course note

Consider keeping the repository private while the course is active, depending on course policy, and making it public later if appropriate.
