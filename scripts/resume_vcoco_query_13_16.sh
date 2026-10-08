#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RESUME="${RESUME:-$PROJECT_ROOT/checkpoints/upt-dvs-r50-vcoco-query_20261008_114704/ckpt_14904_12.pt}"
export FINAL_EPOCH=16
run_stamp="$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/upt-dvs-vcoco-query-bs4-epochs13-16_$run_stamp}"

# Restore the query model's epoch-12 AdamW/StepLR state (LR=1e-5).
# Single GPU, batch=4; evaluate epochs 13, 14, 15, 16, excluding point.
exec bash "$PROJECT_ROOT/scripts/resume_vcoco_dvs_4epochs.sh" \
    --dvs-query-dim 128 \
    --dvs-query-heads 4 \
    --dvs-query-grid 4 6 \
    --dvs-query-chunk-size 16 \
    "$@"
