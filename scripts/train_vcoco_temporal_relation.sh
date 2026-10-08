#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export EPOCHS="${EPOCHS:-4}"
export EVAL_LAST_EPOCHS="${EVAL_LAST_EPOCHS:-4}"
export LR_HEAD="${LR_HEAD:-1e-5}"

# New experiments load model weights only. Optimizer/scheduler/counters restart.
BASELINE="${BASELINE:-$PROJECT_ROOT/checkpoints/upt-dvs-vcoco-query-bs4-epochs13-16_20261008_163427/ckpt_16146_13.pt}"
ADJACENT_CHANGES="${ADJACENT_CHANGES:-1}"
PRECOMP_RESIDUAL="${PRECOMP_RESIDUAL:-1}"
if [[ ! -f "$BASELINE" ]]; then
    echo "Set BASELINE to an existing baseline temporal-query checkpoint." >&2
    exit 1
fi
for branch_switch in "$ADJACENT_CHANGES" "$PRECOMP_RESIDUAL"; do
    if [[ "$branch_switch" != 0 && "$branch_switch" != 1 ]]; then
        echo "ADJACENT_CHANGES and PRECOMP_RESIDUAL must be 0 or 1." >&2
        exit 1
    fi
done

branch_args=()
if [[ "$ADJACENT_CHANGES" == 1 ]]; then
    branch_args+=(--dvs-adjacent-changes)
fi
if [[ "$PRECOMP_RESIDUAL" == 1 ]]; then
    branch_args+=(--dvs-precomp-residual)
fi
if [[ ${#branch_args[@]} -gt 0 ]]; then
    branch_args+=(--init-query-baseline)
fi
run_stamp="$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/upt-dvs-vcoco-changes${ADJACENT_CHANGES}-relation${PRECOMP_RESIDUAL}_$run_stamp}"

echo "New experiment: adjacent changes=$ADJACENT_CHANGES, pre-competitive residual=$PRECOMP_RESIDUAL"
echo "Initialize weights from $BASELINE; restart training at epoch 1, LR=$LR_HEAD."
exec bash "$PROJECT_ROOT/scripts/train_vcoco_dvs.sh" \
    --resume "$BASELINE" "${branch_args[@]}" "$@"
