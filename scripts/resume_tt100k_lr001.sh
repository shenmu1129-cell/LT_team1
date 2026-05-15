#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
DATA_ROOT="${DATA_ROOT:-/home/sutongtong/LanTu_team1/TT100K-2016}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/sutongtong/wwt/code/LT_team1/outputs/tt100k_frcnn_ep50_bs4_lr003}"
RESUME="${RESUME:-${OUTPUT_DIR}/last.pth}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/tt100k_frcnn_ep50_bs4_lr003_resume_lr001.log}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${RESUME}" ]]; then
  echo "ERROR: resume checkpoint not found: ${RESUME}" >&2
  exit 1
fi

echo "GPU_ID=${GPU_ID}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESUME=${RESUME}"
echo "LOG_FILE=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup python -u train.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume "${RESUME}" \
  --resume-lr 0.001 \
  --epochs 100 \
  --batch-size 4 \
  --num-workers 8 \
  --lr-step-size 50 \
  --lr-gamma 0.1 \
  --weight-decay 0.0005 \
  --trainable-backbone-layers 5 \
  --min-size 1024 \
  --max-size 1600 \
  --eval-map-every 5 \
  --quick-eval-samples 100 \
  >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${OUTPUT_DIR}/resume_lr001.pid"
echo "Started resumed TT100K training with lr=0.001. PID=${PID}"
echo "Watch log: tail -f ${LOG_FILE}"
