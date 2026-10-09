# Data

Do not commit the Criteo dataset or generated model-ready datasets to GitHub.

Expected workflow:

1. Download the Criteo Attribution Modeling for Bidding dataset from the official source.
2. Place the source file(s) under `data/raw/`.
3. Run preprocessing scripts to create intermediate files under `data/interim/`.
4. Save final model-ready datasets/tensors under `data/processed/`.

The `.gitignore` keeps the contents of these directories out of version control.
