#!/usr/bin/env bash
# Download the Criteo Attribution Modeling for Bidding dataset into data/raw/.
# Source: https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/
# License: CC BY-NC-SA 4.0 (Criteo data terms) - do not commit the data to GitHub.
#
# Usage (from the repo root, on a Zaratan login node is fine - it is a single download):
#   bash scripts/download_criteo.sh
set -euo pipefail

URL="http://go.criteo.net/criteo-research-attribution-dataset.zip"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
ZIP="${RAW_DIR}/criteo-research-attribution-dataset.zip"

mkdir -p "${RAW_DIR}"
if [[ ! -s "${ZIP}" ]]; then
  echo "Downloading ${URL} (~623 MB) ..."
  # -L follows the go.criteo.net redirect; -C - resumes a partial download.
  curl -L --fail -C - -o "${ZIP}" "${URL}"
fi

echo "Unzipping ..."
unzip -n "${ZIP}" -d "${RAW_DIR}"

FILE="$(find "${RAW_DIR}" -name 'criteo_attribution_dataset.tsv*' | head -n 1)"
if [[ -z "${FILE}" ]]; then
  echo "ERROR: criteo_attribution_dataset.tsv(.gz) not found after unzip. Contents:" >&2
  find "${RAW_DIR}" -maxdepth 3 >&2
  exit 1
fi
echo "Dataset file: ${FILE}"
ls -lh "${FILE}"
echo "First 3 lines (tabs shown as ^I):"
if [[ "${FILE}" == *.gz ]]; then zcat "${FILE}" | head -n 3 | cat -A | cut -c1-300; else head -n 3 "${FILE}" | cat -A | cut -c1-300; fi
echo
echo "Next: python scripts/prepare_criteo.py --check-only"
