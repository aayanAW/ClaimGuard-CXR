#!/usr/bin/env bash
# Phase 5 — IMG_correct evaluation, launched per-config as Phase 4 finals land.
#
# Usage: ./pivot_ab/scripts/launch_phase5_per_config.sh CONFIG SEED
#   CONFIG ∈ {sft_only, sft_faith, sft_dual, sft_full, sft_full_sfc}
#   SEED   ∈ {42, 1337, 9001}
#
# Mirrors the seed-suffix convention from train_chexagent.py:
#   seed 42 (default) → output_dir = pivot_ab_v2/<CONFIG>
#   seed 1337/9001    → output_dir = pivot_ab_v2/<CONFIG>_seed<SEED>

set -euo pipefail

CONFIG=${1:?"usage: $0 CONFIG SEED"}
SEED=${2:?"usage: $0 CONFIG SEED"}

if [ "$SEED" = "42" ]; then
    SUFFIX=""
    OUT_NAME="${CONFIG}"
else
    SUFFIX="_seed${SEED}"
    OUT_NAME="${CONFIG}_seed${SEED}"
fi

CKPT="/data/checkpoints/pivot_ab_v2/${CONFIG}${SUFFIX}/checkpoints/final"
OUT="/data/pivot_ab/img_correct_v2_${OUT_NAME}.jsonl"

echo "[phase5] launching IMG_correct for ${OUT_NAME}"
echo "[phase5]   checkpoint: ${CKPT}"
echo "[phase5]   output: ${OUT}"

cd "$(dirname "$0")/../.."
modal run --detach pivot_ab/modal_app.py::eval_img_correct \
    --checkpoint-dir "${CKPT}" \
    --eval-jsonl /data/groundbench_v5/all_v6/groundbench_v6_test.jsonl \
    --output-path "${OUT}" \
    --margin-tau 0.1 \
    --backbone google/medgemma-4b-it \
    --n-eval 500 \
    --max-total-tokens 768
