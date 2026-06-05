#!/usr/bin/env bash
# Phase 1 — Run image-only filter on the v6 train set, then combine with the
# pre-existing text-only HO filter weights to produce dual filter weights.
#
# Pre-conditions:
#   - GroundBench-v6 train JSONL exists at /data/groundbench_v5/all_v6/groundbench_v6_train.jsonl
#   - Text-only HO filter weights already exist at /data/groundbench_v5/ho_filter_weights_v6.jsonl
#     (produced by the existing v5 HO filter pipeline)
#   - Phase 0 + 0.5 smoke tests passed
#
# Cost: ~$3-5 (image-only filter ~$2-4 on H100; combine is CPU-only ~$0).
# Wallclock: ~30-60 min.

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[pre-flight] Phase 1 filters"
echo "[pre-flight] expected cost: ~\$3-5"
echo "[pre-flight] expected wallclock: ~30-60 min"
echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "[1.1] image-only filter (BiomedCLIP encoder + MLP head, 1 epoch)..."
modal run pivot_ab/modal_app.py::run_image_only_filter \
    --train-jsonl /data/groundbench_v5/all_v6/groundbench_v6_train.jsonl \
    --output-weights-path /data/pivot_ab/image_only_weights.jsonl \
    --n-epochs 1 \
    --batch-size 32 \
    --confidence-threshold 0.7 \
    --downweight 0.2

echo "[1.2] combining text-only + image-only filter weights..."
modal run pivot_ab/modal_app.py::combine_filters \
    --text-weights-path /data/groundbench_v5/ho_filter_weights_v6.jsonl \
    --image-weights-path /data/pivot_ab/image_only_weights.jsonl \
    --output-weights-path /data/pivot_ab/dual_filter_weights.jsonl \
    --aggregation min

echo "[Phase 1 complete] dual_filter_weights.jsonl ready at /data/pivot_ab/dual_filter_weights.jsonl"
