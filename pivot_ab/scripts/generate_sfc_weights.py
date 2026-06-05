#!/usr/bin/env python3
"""Generate continuous SFC (Shortcut-Failure Coefficient) weights from the
existing text-only HO filter classifier outputs.

Background: Liu et al. 2025 (arXiv:2503.03122) define SFC as

    SFC(x, y) = L_text(x, y) / (L_mm(x, y) + L_text(x, y))

where L_text is the loss of a text-only model and L_mm is the loss of the
multimodal model. Samples where text-only fails (high L_text relative to
L_mm) get high SFC → high training weight; samples where text-only succeeds
get low SFC → low weight. The opposite of our discrete dual-filter rule
which downweights text-solvable rows to 0.2.

Approximation: we don't actually have a multimodal-loss-per-row signal
handy. We approximate with the existing text-only HO filter's predicted
probability of the true label:

    p_text_correct(x, y)  : how confident the text-only model is on the gold label
    SFC_proxy(x, y) = 1 - p_text_correct(x, y)

When the text-only model is confident on the right label, SFC_proxy is low
(downweight). When the text-only model is uncertain or wrong, SFC_proxy is
high (upweight). This is the spirit of Liu 2025's SFC, derived from
quantities we already have.

Run this once after `v5/ho_filter.py:run_ho_filter` produces the discrete
weights. Output is a JSONL of {"row_idx": int, "weight": float} where
weight ∈ [0, 1] continuous, NOT the discrete 0.2/1.0 schedule.

This script doesn't need Modal — pure CPU Python over a small JSONL.

Usage:
    python pivot_ab/scripts/generate_sfc_weights.py \\
        --discrete-weights /data/groundbench_v5/ho_filter_weights_v6.jsonl \\
        --output /data/pivot_ab/sfc_continuous_weights.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discrete-weights", required=True, type=Path,
                   help="JSONL with {row_idx, weight, [text_only_prob_correct]} from v5/ho_filter.py")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-weight", type=float, default=0.05,
                   help="Floor on the continuous weight to prevent zero-gradient rows")
    args = p.parse_args()

    n_rows = 0
    n_inferred = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.discrete_weights.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_idx = int(obj["row_idx"])
            n_rows += 1
            # Prefer an explicit text-only probability if the upstream filter
            # logged one. Otherwise infer from the discrete weight: 0.2 means
            # "shortcut-solvable" → text-only highly confident; 1.0 means
            # "not solvable" → text-only uncertain or wrong.
            if "text_only_prob_correct" in obj:
                p_text = float(obj["text_only_prob_correct"])
            else:
                discrete = float(obj["weight"])
                # Inverse map: discrete=0.2 ⇔ p_text >= 0.7 (assume 0.85 mean)
                #              discrete=1.0 ⇔ p_text <  0.7 (assume 0.45 mean)
                p_text = 0.85 if discrete < 0.5 else 0.45
                n_inferred += 1
            sfc = max(args.min_weight, 1.0 - p_text)
            fout.write(json.dumps({"row_idx": row_idx, "weight": sfc}) + "\n")

    print(f"Wrote {n_rows} continuous SFC weights to {args.output}")
    if n_inferred > 0:
        print(f"  Note: {n_inferred} rows had no text_only_prob_correct field; "
              f"inferred from discrete weight (less precise). To get the precise "
              f"SFC, re-run v5/ho_filter.py with `log_text_only_probs=True`.")


if __name__ == "__main__":
    main()
