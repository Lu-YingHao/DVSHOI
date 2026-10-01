#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

mkdir -p logs checkpoints/upt-dvs-r50-hicodet-bs8
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/hicodet_dvs_bs8_${timestamp}.log"

echo "Training log: $PROJECT_ROOT/$log_file"

.venv/bin/python -u main.py \
    --world-size 4 \
    --batch-size 8 \
    --num-workers 4 \
    --lr-head 2e-4 \
    --use-dvs \
    --dvs-variant base \
    --pretrained checkpoints/detr/detr-r50-hicodet.pth \
    --output-dir checkpoints/upt-dvs-r50-hicodet-bs8 \
    --print-interval 1 \
    "$@" 2>&1 | tee "$log_file"
