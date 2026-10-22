#!/usr/bin/env bash
# Download the Criteo Attribution dataset from Hugging Face into data/raw/ (one ~623 MB file).
# Hub page: https://huggingface.co/datasets/criteo/criteo-attribution-dataset
# License: CC BY-NC-SA 4.0 (Criteo data terms) - do not commit the data to GitHub.
#
# Run on a Zaratan *login* node (compute nodes may have no internet), from the repo root:
#   conda activate criteo-journey && bash scripts/download_criteo.sh
#
# After this, prepare_criteo.py reads the local copy. Without it, the pipeline falls back to
#   pd.read_csv("hf://datasets/criteo/criteo-attribution-dataset/criteo_attribution_dataset.tsv.gz", sep="\t")
# which streams the file from the Hub on every run.
#
# If you get a 401/403: open the Hub page, accept the dataset terms if asked, then run
# `huggingface-cli login` (or `hf auth login`) with a read token and retry.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/raw/criteo_attribution_dataset"
mkdir -p "${DEST}"

python - "${DEST}" <<'EOF'
import sys
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="criteo/criteo-attribution-dataset",
    repo_type="dataset",
    filename="criteo_attribution_dataset.tsv.gz",
    local_dir=sys.argv[1],
)
print("Saved to", path)
EOF

FILE="${DEST}/criteo_attribution_dataset.tsv.gz"
ls -lh "${FILE}"
echo "First 3 lines (tabs shown as ^I):"
zcat "${FILE}" | head -n 3 | cat -A | cut -c1-300
echo
echo "Next: python scripts/prepare_criteo.py --check-only"
