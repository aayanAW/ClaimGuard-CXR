#!/usr/bin/env bash
# Phase 0 gate — smoke test on Modal H100.
# Verifies CheXagent-2-3B downloads and LoRA wrapping works.
# Cost: ~$0.50 (one H100 minute for download + 5 minutes for load).
#
# Run:
#   bash pivot_ab/scripts/launch_phase0_smoke.sh
#
# Expected output:
#   {"status": "ok", "backbone": "StanfordAIMI/CheXagent-2-3b", "n_trainable_M": ..., "n_total_M": ...}

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[pre-flight] Phase 0 smoke test"
echo "[pre-flight] backbone: StanfordAIMI/CheXagent-2-3b"
echo "[pre-flight] expected cost: ~\$0.50"
echo "[pre-flight] expected duration: ~6 minutes"
echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

modal run pivot_ab/modal_app.py::smoke_test_load_chexagent
