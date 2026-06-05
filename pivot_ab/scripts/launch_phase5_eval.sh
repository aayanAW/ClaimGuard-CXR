#!/usr/bin/env bash
# Phase 5 — IMG_correct evaluation on every Phase-4 checkpoint.
#
# Pre-conditions:
#   - Phase 4 complete (checkpoints under /data/checkpoints/pivot_ab_v2/*/checkpoints/final/)
#
# Note: the v2 checkpoint dir reflects the 2026-05-05 data-target fix
# (evidence_text as supervised target on OpenI rows; verifier-shaped rows
# dropped). The old /data/checkpoints/pivot_ab/* checkpoints are preserved
# but are not the right inputs for IMG_correct under the generator framing.
#
# Cost: ~$2-3 per checkpoint × ~5 configs × 3 seeds = $30-45.
# Wallclock: ~2-4 hours total.

set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIGS=("sft_only" "sft_faith" "sft_dual" "sft_full" "sft_full_sfc")
SEEDS=(42 1337 9001)

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ "$seed" = "42" ]; then
            ckpt="/data/checkpoints/pivot_ab_v2/${cfg}/checkpoints/final"
            out="/data/pivot_ab/img_correct_v2_${cfg}.jsonl"
        else
            ckpt="/data/checkpoints/pivot_ab_v2/${cfg}_seed${seed}/checkpoints/final"
            out="/data/pivot_ab/img_correct_v2_${cfg}_seed${seed}.jsonl"
        fi
        echo "[eval] ${cfg} seed=${seed} → ${out}"
        modal run --detach pivot_ab/modal_app.py::eval_img_correct \
            --checkpoint-dir "${ckpt}" \
            --eval-jsonl /data/groundbench_v5/all_v6/groundbench_v6_test.jsonl \
            --output-path "${out}" \
            --margin-tau 0.1 \
            --backbone google/medgemma-4b-it &
    done
done
wait

echo "[Phase 5 complete] IMG_correct results in /data/pivot_ab/img_correct_v2_*.jsonl"
