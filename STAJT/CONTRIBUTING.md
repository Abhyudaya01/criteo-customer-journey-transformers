# Contributing

## Branch workflow

Create a branch for each task:

```bash
git checkout -b feature/<short-name>
```

Examples:

```bash
feature/criteo-preprocessing
feature/lstm-baseline
feature/heterogeneous-transformer
feature/itransformer
feature/evaluation
```

## Before merging

1. Pull the latest `main`.
2. Run tests or at minimum the relevant training/evaluation smoke test.
3. Update documentation/configs if behavior changed.
4. Open a pull request.
5. Have at least one teammate review substantial changes.

## Commit messages

Prefer concise, descriptive commits:

```text
Add chronological split helper
Implement LSTM baseline
Add PR-AUC and lift evaluation
Fix conversion label windowing
```

Avoid committing raw datasets, checkpoints, credentials, or local environment files.
