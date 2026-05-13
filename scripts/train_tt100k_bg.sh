#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-/home/sutongtong/LanTu_team1/TT100K-2016}"
EXP_NAME="${EXP_NAME:-tt100k_frcnn_ep50_bs4_lr003}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "GPU_ID=${GPU_ID}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup python -u train.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 8 \
  --lr 0.003 \
  --lr-step-size 15 \
  --lr-gamma 0.1 \
  --weight-decay 0.0005 \
  --trainable-backbone-layers 5 \
  --min-size 1024 \
  --max-size 1600 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${OUTPUT_DIR}/train.pid"
echo "Started training in background. PID=${PID}"
echo "Watch log: tail -f ${LOG_FILE}"
