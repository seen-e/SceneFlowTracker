#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-co-tracker}"
MANIFEST_PATH="${MANIFEST_PATH:-/mnt/data/chachaxu/save/abc_130k_v3/abc_130k_v3_train_all_views.json}"
VIEW_KEY="${VIEW_KEY:-observation.images.top}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
else
  echo "Cannot find conda. Activate ${CONDA_ENV} manually, then rerun this script." >&2
  exit 1
fi

python example/main.py \
  --manifest-path "${MANIFEST_PATH}" \
  --view-key "${VIEW_KEY}" \
  "$@"
