#!/usr/bin/env bash
# Auto-launcher: poll Modal volume every 5 min for Phase 4 `final` checkpoints
# and launch eval_img_correct for any not yet evaluated.
# Tracks state in /tmp/phase5_launched.txt to avoid double-launching.
#
# Optimisation: single `modal volume ls --json` per cycle, then iterate locally,
# instead of 15 separate ls calls (which take >5 min on cold ls cache).
#
# Usage: nohup bash pivot_ab/scripts/auto_phase5_launcher.sh > /tmp/phase5_auto.log 2>&1 &

set -uo pipefail

export PATH=/Users/aayanalwani/miniforge3/bin:$PATH
cd "$(dirname "$0")/../.."

STATE_FILE=/tmp/phase5_launched.txt
touch "$STATE_FILE"

CONFIGS=(sft_only sft_faith sft_dual sft_full sft_full_sfc)
SEEDS=(42 1337 9001)

# Maximum runtime: 12 hours
END_TS=$(($(date +%s) + 12 * 3600))

while [ $(date +%s) -lt $END_TS ]; do
  echo "[$(date -u +%H:%M:%S)] poll cycle"

  # Single ls of the parent dir to find which configs have a `final` checkpoint
  ls_out=$(modal volume ls claimguard-v5-data /checkpoints/pivot_ab_v2/ 2>/dev/null)

  for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [ "$seed" = "42" ]; then
        suffix=""
        out_name="$cfg"
      else
        suffix="_seed$seed"
        out_name="${cfg}_seed${seed}"
      fi
      key="${cfg}${suffix}"

      # Skip if already launched
      if grep -qx "$key" "$STATE_FILE" 2>/dev/null; then
        continue
      fi

      # Quick check: does the config dir exist?
      if ! echo "$ls_out" | grep -q "${cfg}${suffix}$"; then
        continue
      fi

      # Detailed check: does it have a `final` checkpoint?
      if modal volume ls claimguard-v5-data "/checkpoints/pivot_ab_v2/${cfg}${suffix}/checkpoints/" 2>/dev/null | grep -qE "final$"; then
        ckpt="/data/checkpoints/pivot_ab_v2/${cfg}${suffix}/checkpoints/final"
        echo "[$(date -u +%H:%M:%S)] LAUNCH eval_img_correct for $key"
        modal run --detach pivot_ab/modal_app.py::eval_img_correct \
          --checkpoint-dir "$ckpt" \
          --eval-jsonl /data/groundbench_v5/all_v6/groundbench_v6_test.jsonl \
          --output-path "/data/pivot_ab/img_correct_v2_${out_name}.jsonl" \
          --margin-tau 0.1 \
          --backbone google/medgemma-4b-it \
          --n-eval 500 \
          --max-total-tokens 768 \
          > "/tmp/phase5_${out_name}.log" 2>&1 &
        echo "$key" >> "$STATE_FILE"
        sleep 1
      fi
    done
  done

  sleep 300
done

echo "[$(date -u +%H:%M:%S)] auto-launcher max runtime reached; exiting"
