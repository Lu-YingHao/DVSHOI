#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
WORLD_SIZE="${WORLD_SIZE:-4}"
EPOCHS="${EPOCHS:-20}"
EVAL_LAST_EPOCHS="${EVAL_LAST_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/vcoco}"
DVS_ROOT="${DVS_ROOT:-$HOME/Data/vcoco-dvs}"
VCOCO_EVAL_ROOT="${VCOCO_EVAL_ROOT:-$PROJECT_ROOT/vcoco/v_coco}"
PRETRAINED="${PRETRAINED:-$PROJECT_ROOT/checkpoints/detr/detr-r50-vcoco.pth}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/upt-dvs-r50-vcoco-query_$run_stamp}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/train.log}"
mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"

echo "V-COCO: trainval -> test, DVS pair/action queries over ordered spatiotemporal tokens"
echo "Epochs: $EPOCHS; synchronous evaluation in the final $EVAL_LAST_EPOCHS epochs"
echo "Evaluation: IoU=0.5, point excluded (24 role-action classes)"
echo "Output directory: $OUT_DIR"
echo "Training log: $LOG_FILE"

"$PYTHON" -u main.py \
    --dataset vcoco \
    --partitions trainval test \
    --data-root "$DATA_ROOT" \
    --world-size "$WORLD_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --lr-head "${LR_HEAD:-1e-4}" \
    --lr-drop 10 \
    --use-dvs \
    --dvs-root "$DVS_ROOT" \
    --dvs-num-bins 8 \
    --dvs-variant "${DVS_VARIANT:-base}" \
    --dvs-query-dim 128 \
    --dvs-query-heads 4 \
    --dvs-query-grid 4 6 \
    --dvs-query-chunk-size 16 \
    --pretrained "$PRETRAINED" \
    --output-dir "$OUT_DIR" \
    --eval-last-epochs "$EVAL_LAST_EPOCHS" \
    --vcoco-eval-root "$VCOCO_EVAL_ROOT" \
    --vcoco-exclude-actions point \
    --distributed-timeout-minutes 180 \
    --port "${PORT:-3345}" \
    --print-interval "${PRINT_INTERVAL:-20}" \
    "$@" 2>&1 | tee "$LOG_FILE"
