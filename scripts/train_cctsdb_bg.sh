#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-/home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021}"
EXP_NAME="${EXP_NAME:-cctsdb_frcnn_ep80_bs8_lr005}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "GPU_ID=${GPU_ID}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup python -u train.py \
  --config configs/cctsdb.yaml \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 80 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 0.005 \
  --lr-step-size 30 \
  --lr-gamma 0.1 \
  --weight-decay 0.0005 \
  --trainable-backbone-layers 5 \
  --min-size 1024 \
  --max-size 1600 \
  --eval-map-every 10 \
  --quick-eval-samples 100 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${OUTPUT_DIR}/train.pid"
echo "Started CCTSDB training in background. PID=${PID}"
echo "Watch log: tail -f ${LOG_FILE}"
