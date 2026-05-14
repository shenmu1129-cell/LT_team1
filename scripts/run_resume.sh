#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-/home/sutongtong/LanTu_team1/TT100K-2016}"
EXP_NAME="${EXP_NAME:-tt100k_frcnn_ep160_bs16_lr008}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
RESUME="${RESUME:-${OUTPUT_DIR}/last.pth}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_resume.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${RESUME}" ]]; then
  echo "ERROR: resume checkpoint not found: ${RESUME}" >&2
  echo "Set RESUME=/path/to/last.pth or OUTPUT_DIR=/path/to/experiment." >&2
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
  --epochs 100 \
  --batch-size 16 \
  --num-workers 12 \
  --lr 0.008 \
  --lr-step-size 90 \
  --lr-gamma 0.1 \
  --weight-decay 0.0005 \
  --trainable-backbone-layers 5 \
  --min-size 833 \
  --max-size 1333 \
  --eval-map-every 5 \
  --quick-eval-samples 100 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${OUTPUT_DIR}/resume.pid"
echo "Started resumed training in background. PID=${PID}"
echo "Watch log: tail -f ${LOG_FILE}"
