#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RESUME="${RESUME:-$PROJECT_ROOT/checkpoints/upt-dvs-vcoco-bs4-12epochs-20261007_163525/ckpt_14904_12.pt}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/upt-dvs-vcoco-bs4-epochs13-16_$run_stamp}"

if [[ ! -f "$RESUME" ]]; then
    echo "Checkpoint not found: $RESUME" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "Resume V-COCO RGB+DVS from epoch 12; train epochs 13-16."
echo "Single GPU, batch size 4; restore AdamW and StepLR (current LR 1e-5)."
echo "Evaluate every resumed epoch; exclude point from AP."
echo "Checkpoint: $RESUME"
echo "Output directory: $OUT_DIR"

"$PYTHON" -u main.py \
    --dataset vcoco \
    --partitions trainval test \
    --data-root "${DATA_ROOT:-$PROJECT_ROOT/vcoco}" \
    --world-size 1 \
    --batch-size 4 \
    --num-workers "${NUM_WORKERS:-4}" \
    --epochs 16 \
    --lr-head 1e-4 \
    --lr-drop 10 \
    --gamma 0.2 \
    --use-dvs \
    --dvs-root "${DVS_ROOT:-$HOME/Data/vcoco-dvs}" \
    --dvs-num-bins 8 \
    --dvs-variant base \
    --pretrained "${PRETRAINED:-$PROJECT_ROOT/checkpoints/detr/detr-r50-vcoco.pth}" \
    --resume "$RESUME" \
    --resume-training \
    --output-dir "$OUT_DIR" \
    --eval-last-epochs 4 \
    --vcoco-eval-root "${VCOCO_EVAL_ROOT:-$PROJECT_ROOT/vcoco/v_coco}" \
    --vcoco-exclude-actions point \
    --distributed-timeout-minutes 180 \
    --port "${PORT:-3345}" \
    --print-interval 20 \
    "$@" 2>&1 | tee "$OUT_DIR/train.log"
