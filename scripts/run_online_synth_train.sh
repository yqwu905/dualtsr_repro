#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/online_synth_emmdit.yaml}"
PYTHON="${PYTHON:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RESUME="${RESUME:-auto}"
VAE_REPO="${VAE_REPO:-mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers}"
VAE_DIR="${VAE_DIR:-weights/dc-ae-f32c32-sana-1.0-diffusers}"

cd "$(dirname "$0")/.."

echo "+ ${PYTHON} scripts/download_fonts.py"
"${PYTHON}" scripts/download_fonts.py

echo "+ ${PYTHON} scripts/synthesize_pretrain_data.py --config ${CONFIG} --vocab-only"
"${PYTHON}" scripts/synthesize_pretrain_data.py --config "${CONFIG}" --vocab-only

if [[ ! -f "${VAE_DIR}/config.json" ]]; then
  echo "+ download ${VAE_REPO} -> ${VAE_DIR}"
  "${PYTHON}" - <<PY
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${VAE_REPO}",
    local_dir="${VAE_DIR}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
Path("${VAE_DIR}").mkdir(parents=True, exist_ok=True)
PY
else
  echo "VAE already present: ${VAE_DIR}"
fi

OVERRIDES=()
[[ -n "${OUTPUT_DIR:-}" ]] && OVERRIDES+=(--set "output_dir=${OUTPUT_DIR}")
[[ -n "${DEVICE:-}" ]] && OVERRIDES+=(--set "runtime.device=${DEVICE}")
[[ -n "${PRECISION:-}" ]] && OVERRIDES+=(--set "runtime.precision=${PRECISION}")
[[ -n "${MAX_STEPS:-}" ]] && OVERRIDES+=(--set "train.max_steps=${MAX_STEPS}")
[[ -n "${SAVE_EVERY:-}" ]] && OVERRIDES+=(--set "checkpoint.save_every=${SAVE_EVERY}")
[[ -n "${IMAGE_EVERY:-}" ]] && OVERRIDES+=(--set "logging.image_every=${IMAGE_EVERY}")
[[ -n "${BATCH_SIZE:-}" ]] && OVERRIDES+=(--set "loader.batch_size=${BATCH_SIZE}")
[[ -n "${GLOBAL_BATCH_SIZE:-}" ]] && OVERRIDES+=(--set "train.global_batch_size=${GLOBAL_BATCH_SIZE}")
[[ -n "${NUM_WORKERS:-}" ]] && OVERRIDES+=(--set "loader.num_workers=${NUM_WORKERS}")
[[ -n "${LR:-}" ]] && OVERRIDES+=(--set "optimizer.lr=${LR}")
[[ -n "${TEXT_TIMESTEPS:-}" ]] && OVERRIDES+=(--set "train.text_timesteps=${TEXT_TIMESTEPS}")
[[ -n "${RUN_NAME:-}" ]] && OVERRIDES+=(--set "output_dir=outputs/${RUN_NAME}")

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  echo "+ ${PYTHON} train.py --config ${CONFIG} --resume ${RESUME} ${OVERRIDES[*]}"
  exec "${PYTHON}" train.py --config "${CONFIG}" --resume "${RESUME}" "${OVERRIDES[@]}"
fi

echo "+ torchrun --nproc_per_node=${NPROC_PER_NODE} train.py --config ${CONFIG} --resume ${RESUME} ${OVERRIDES[*]}"
exec torchrun --nproc_per_node="${NPROC_PER_NODE}" train.py --config "${CONFIG}" --resume "${RESUME}" "${OVERRIDES[@]}"
