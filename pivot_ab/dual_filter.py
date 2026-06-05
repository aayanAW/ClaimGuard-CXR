"""Dual adversarial filter — combines text-only and image-only HO filters.

A training row is downweighted if EITHER the text-only filter
(`v5/ho_filter.py`) OR the image-only filter (`image_only_filter.py`) flags it
as solvable. The combined weight is the elementwise minimum of the two
per-row weights, which is the most conservative aggregation: any sample that
either modality alone can solve is treated as a shortcut and suppressed.

This is the genuine novelty axis vs Liu et al. 2025 (which only does
text-only filtering for multimodal reward models).

Inputs are two JSONLs of the form `{"row_idx": int, "weight": float}` written
by `v5/ho_filter.py:run_ho_filter` and
`pivot_ab/image_only_filter.py:run_image_only_filter`. The combined weights
are written to a third JSONL of the same schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_weights(path: Path) -> dict[int, float]:
    weights: dict[int, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            weights[int(obj["row_idx"])] = float(obj["weight"])
    return weights


def combine_dual_filters(
    *,
    text_weights_path: Path,
    image_weights_path: Path,
    output_weights_path: Path,
    aggregation: str = "min",
) -> dict:
    """Combine text-only and image-only filter weights into a single per-row weight.

    Args:
        text_weights_path: JSONL from `run_ho_filter` (text-only).
        image_weights_path: JSONL from `run_image_only_filter` (image-only).
        output_weights_path: destination for combined per-row weights.
        aggregation: how to combine the two weights:
            - "min" (default): w = min(w_text, w_image). Most conservative;
              downweights if EITHER side solves.
            - "product": w = w_text * w_image. Even more aggressive
              downweighting when both sides agree.
            - "mean": w = (w_text + w_image) / 2. Softer middle ground.

    Returns:
        Summary dict with combined statistics.
    """
    if aggregation not in {"min", "product", "mean"}:
        raise ValueError(f"unknown aggregation: {aggregation!r}")

    text_weights = _read_weights(text_weights_path)
    image_weights = _read_weights(image_weights_path)

    if set(text_weights.keys()) != set(image_weights.keys()):
        only_text = set(text_weights) - set(image_weights)
        only_image = set(image_weights) - set(text_weights)
        raise ValueError(
            "text and image weight files cover different row sets; "
            f"only_text={len(only_text)}, only_image={len(only_image)}. "
            "Both filters must be run on the same training JSONL."
        )

    combined: dict[int, float] = {}
    n_text_only_down = 0
    n_image_only_down = 0
    n_both_down = 0
    n_neither_down = 0
    for idx in sorted(text_weights):
        wt = text_weights[idx]
        wi = image_weights[idx]
        if aggregation == "min":
            w = min(wt, wi)
        elif aggregation == "product":
            w = wt * wi
        else:
            w = (wt + wi) / 2.0
        combined[idx] = w

        text_down = wt < 1.0
        image_down = wi < 1.0
        if text_down and image_down:
            n_both_down += 1
        elif text_down:
            n_text_only_down += 1
        elif image_down:
            n_image_only_down += 1
        else:
            n_neither_down += 1

    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with output_weights_path.open("w") as f:
        for idx in sorted(combined):
            f.write(json.dumps({"row_idx": idx, "weight": combined[idx]}) + "\n")

    n_total = len(combined)
    n_any_down = n_text_only_down + n_image_only_down + n_both_down
    summary = {
        "aggregation": aggregation,
        "n_rows": n_total,
        "n_text_only_downweighted": n_text_only_down,
        "n_image_only_downweighted": n_image_only_down,
        "n_both_downweighted": n_both_down,
        "n_neither_downweighted": n_neither_down,
        "n_any_downweighted": n_any_down,
        "fraction_text_only": n_text_only_down / max(1, n_total),
        "fraction_image_only": n_image_only_down / max(1, n_total),
        "fraction_both": n_both_down / max(1, n_total),
        "fraction_any_downweighted": n_any_down / max(1, n_total),
        "weights_path": str(output_weights_path),
    }
    # The fraction of rows where the two filters disagree on whether to
    # downweight — high disagreement is good evidence the dual filter adds signal.
    disagreement = n_text_only_down + n_image_only_down
    summary["disagreement_rate"] = disagreement / max(1, n_total)
    logger.info("dual filter summary: %s", summary)
    return summary
