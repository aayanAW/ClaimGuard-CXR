"""Unit tests for pivot_ab/dual_filter.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pivot_ab.dual_filter import combine_dual_filters


def _write_weights(path: Path, weights: dict[int, float]) -> None:
    with path.open("w") as f:
        for idx, w in sorted(weights.items()):
            f.write(json.dumps({"row_idx": idx, "weight": w}) + "\n")


def test_min_aggregation(tmp_path: Path) -> None:
    text_path = tmp_path / "text.jsonl"
    image_path = tmp_path / "image.jsonl"
    out_path = tmp_path / "out.jsonl"

    _write_weights(text_path, {0: 1.0, 1: 0.2, 2: 1.0, 3: 0.2})
    _write_weights(image_path, {0: 0.2, 1: 1.0, 2: 1.0, 3: 0.2})

    summary = combine_dual_filters(
        text_weights_path=text_path,
        image_weights_path=image_path,
        output_weights_path=out_path,
        aggregation="min",
    )
    assert summary["n_rows"] == 4
    assert summary["n_text_only_downweighted"] == 1
    assert summary["n_image_only_downweighted"] == 1
    assert summary["n_both_downweighted"] == 1
    assert summary["n_neither_downweighted"] == 1
    # min(0.2, 1.0)=0.2; min(1.0,0.2)=0.2; min(0.2,0.2)=0.2; min(1.0,1.0)=1.0
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    weights = {r["row_idx"]: r["weight"] for r in rows}
    assert weights[0] == pytest.approx(0.2)
    assert weights[1] == pytest.approx(0.2)
    assert weights[2] == pytest.approx(1.0)
    assert weights[3] == pytest.approx(0.2)


def test_product_aggregation(tmp_path: Path) -> None:
    text_path = tmp_path / "text.jsonl"
    image_path = tmp_path / "image.jsonl"
    out_path = tmp_path / "out.jsonl"

    _write_weights(text_path, {0: 0.5, 1: 1.0})
    _write_weights(image_path, {0: 0.5, 1: 1.0})

    combine_dual_filters(
        text_weights_path=text_path,
        image_weights_path=image_path,
        output_weights_path=out_path,
        aggregation="product",
    )
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    weights = {r["row_idx"]: r["weight"] for r in rows}
    assert weights[0] == pytest.approx(0.25)
    assert weights[1] == pytest.approx(1.0)


def test_mismatched_row_sets_raises(tmp_path: Path) -> None:
    text_path = tmp_path / "text.jsonl"
    image_path = tmp_path / "image.jsonl"
    out_path = tmp_path / "out.jsonl"

    _write_weights(text_path, {0: 1.0, 1: 0.2})
    _write_weights(image_path, {0: 1.0, 2: 0.2})

    with pytest.raises(ValueError, match="different row sets"):
        combine_dual_filters(
            text_weights_path=text_path,
            image_weights_path=image_path,
            output_weights_path=out_path,
        )


def test_unknown_aggregation_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        combine_dual_filters(
            text_weights_path=tmp_path / "a.jsonl",
            image_weights_path=tmp_path / "b.jsonl",
            output_weights_path=tmp_path / "c.jsonl",
            aggregation="invalid",
        )


def test_disagreement_rate(tmp_path: Path) -> None:
    text_path = tmp_path / "text.jsonl"
    image_path = tmp_path / "image.jsonl"
    out_path = tmp_path / "out.jsonl"

    # 5 rows: text-only (0), image-only (1), both (2), neither (3, 4)
    _write_weights(text_path, {0: 0.2, 1: 1.0, 2: 0.2, 3: 1.0, 4: 1.0})
    _write_weights(image_path, {0: 1.0, 1: 0.2, 2: 0.2, 3: 1.0, 4: 1.0})

    summary = combine_dual_filters(
        text_weights_path=text_path,
        image_weights_path=image_path,
        output_weights_path=out_path,
        aggregation="min",
    )
    # disagreement = (text-only + image-only) / total = (1 + 1) / 5 = 0.4
    assert summary["disagreement_rate"] == pytest.approx(0.4)
    assert summary["fraction_any_downweighted"] == pytest.approx(0.6)
