#!/usr/bin/env bash
# Phase 4 — Five primary fine-tunes × 3 seeds = 15 runs.
#
# Pre-conditions (verified 2026-05-05 post data-target fix + Opus pre-flight):
#   - Phase 0 + 0.5 + train smoke tests passed (2026-05-03)
#   - Phase 1 filters complete (dual_filter_weights.jsonl on volume)
#   - 2026-05-05 data-target fix applied: evidence_text as supervised target;
#     verifier-shape rows dropped; output_dir = pivot_ab_v2/.
#   - Opus pre-flight audit cleared after fixing #1 (eval OOM) and #2
#     (sc_idx vs raw-line-pos in weight remap).
#
# Cost: ~$15-20 per run × 15 runs = $225-300 (data halved post-fix).
# Wallclock: ~1.5h per run on H100, ~7-12h total at 3-concurrent.

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[Phase 4] full sweep — 5 configs × 3 seeds = 15 runs"
echo "[Phase 4] expected cost: ~\$225-300; wallclock ~7-12h at 3-concurrent"
echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

CONFIGS=("sft_only" "sft_faith" "sft_dual" "sft_full" "sft_full_sfc")
SEEDS=(42 1337 9001)

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "[launch] config=${cfg}.yaml seed=${seed}"
        modal run --detach pivot_ab/modal_app.py::run_sft \
            --config-path "${cfg}.yaml" \
            --seed "${seed}" &
    done
done
wait

echo "[Phase 4 complete] checkpoints under /data/checkpoints/pivot_ab_v2/*/checkpoints/final/"
