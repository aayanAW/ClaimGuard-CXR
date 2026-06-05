#!/usr/bin/env bash
# Pull every Tier 3 output from the Modal volume to v5_final_results/ locally.
#
# Safe to run at any time. Files that don't exist yet are skipped. Re-running
# overwrites with the latest. No Modal spend — only volume reads, which are free.

set -e
cd "$(dirname "$0")/../.."

DEST=v5_final_results
mkdir -p "$DEST"
mkdir -p "$DEST/baselines"

echo "[fetch] downloading pipeline_status.json"
modal volume get claimguard-v5-data /groundbench_v5/pipeline_status.json "$DEST/pipeline_status.json" 2>/dev/null || true

echo "[fetch] downloading per-config diagnostics + conformal outputs"
for cfg in v5_0_base v5_1_ground v5_2_real v5_3_contrast v5_4_final \
           abl_no_ground abl_no_consist abl_no_contrast abl_no_uncert abl_no_hofilter \
           abl_hothresh_50 abl_hothresh_60 abl_hothresh_70 abl_hothresh_80 abl_hothresh_90 \
           abl_scale_25 abl_scale_50 abl_scale_100 \
           crosssite_openi_to_chestx_det10 crosssite_chestx_det10_to_openi; do
    modal volume get claimguard-v5-data "/checkpoints/claimguard_v5/$cfg/diagnostic.json" \
        "$DEST/${cfg}_diagnostic.json" 2>/dev/null >/dev/null || true
    modal volume get claimguard-v5-data "/checkpoints/claimguard_v5/$cfg/conformal.json" \
        "$DEST/${cfg}_conformal.json" 2>/dev/null >/dev/null || true
    if [ -f "$DEST/${cfg}_diagnostic.json" ]; then echo "  ✓ $cfg diagnostic"; fi
    if [ -f "$DEST/${cfg}_conformal.json" ]; then echo "  ✓ $cfg conformal"; fi
done

echo "[fetch] downloading cross-site diagnostic-on-other-site"
for pair in "openi_to_chestx_det10:chestx_det10" "chestx_det10_to_openi:openi"; do
    pair_name="${pair%%:*}"
    tested_on="${pair##*:}"
    modal volume get claimguard-v5-data \
        "/checkpoints/claimguard_v5/crosssite_${pair_name}/diagnostic_on_${tested_on}.json" \
        "$DEST/crosssite_${pair_name}_on_${tested_on}.json" 2>/dev/null >/dev/null || true
done

echo "[fetch] downloading baseline diagnostics"
modal volume get claimguard-v5-data /baselines/baseline_summary.json \
    "$DEST/baselines/baseline_summary.json" 2>/dev/null >/dev/null || true
for n in claude-3-5-sonnet biomedclip-zero-shot CheXagent-8b llava-med-v1.5-mistral-7b MedVLM gpt-4o gemini-1-5-pro; do
    modal volume get claimguard-v5-data "/baselines/baseline_${n}_diagnostic.json" \
        "$DEST/baselines/baseline_${n}_diagnostic.json" 2>/dev/null >/dev/null || true
    [ -f "$DEST/baselines/baseline_${n}_diagnostic.json" ] && echo "  ✓ baseline $n"
done

echo "[fetch] downloading orchestrator summaries"
for name in ablations_loss_drop_summary ablations_ho_threshold_summary ablations_scale_summary \
            crosssite_openi_to_chestx_det10_summary crosssite_chestx_det10_to_openi_summary \
            conformal_summary v5_results; do
    modal volume get claimguard-v5-data "/${name}.json" "$DEST/${name}.json" 2>/dev/null >/dev/null || true
    [ -f "$DEST/${name}.json" ] && echo "  ✓ ${name}.json"
done

echo "[fetch] done; files in $DEST/"
ls -la "$DEST" | head -30
