"""Unit tests for pivot_ab/data_loader.py.

The MedGemmaSFTDataset class requires a real MedGemma AutoProcessor to
exercise (it calls processor.apply_chat_template internally), which is
expensive and network-bound. Those tests are skipped here and exercised
end-to-end by the smoke_test_forward_pass Modal entrypoint instead.

The collate function and laterality regex are pure-Python and tested here.
"""

from __future__ import annotations

import torch
import pytest

from pivot_ab.data_loader import (
    CounterfactualBatch,
    counterfactual_collate,
    is_laterality_sensitive,
)


def test_is_laterality_sensitive():
    assert is_laterality_sensitive("Left lower lobe pneumothorax")
    assert is_laterality_sensitive("right pleural effusion")
    assert is_laterality_sensitive("bilateral effusions")
    assert not is_laterality_sensitive("Heart size is normal")
    assert not is_laterality_sensitive("")
    assert not is_laterality_sensitive("Lung opacity in posterior segment")


def _fake_row(input_len: int, target_len: int, weight: float = 1.0, laterality: bool = False) -> dict:
    """Build a synthetic dataset-row dict in the shape expected by counterfactual_collate."""
    total_len = input_len + target_len
    input_ids = torch.arange(1, total_len + 1, dtype=torch.long)
    attention_mask = torch.ones(total_len, dtype=torch.long)
    target_mask = torch.zeros(total_len, dtype=torch.long)
    target_mask[input_len:] = 1
    pixel_values = torch.zeros(3, 16, 16)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "row_weight": torch.tensor(weight, dtype=torch.float32),
        "flipped_eligible": torch.tensor(laterality, dtype=torch.bool),
        "row_idx": torch.tensor(0, dtype=torch.long),
    }


def test_collate_pads_variable_length_sequences():
    rows = [
        _fake_row(input_len=8, target_len=4, weight=1.0, laterality=False),
        _fake_row(input_len=12, target_len=6, weight=0.2, laterality=True),
    ]
    batch = counterfactual_collate(rows, enable_flipping=True, pad_token_id=0)
    # Total length should be max(12, 18) = 18
    assert batch.input_ids.shape == (2, 18)
    assert batch.attention_mask.shape == (2, 18)
    assert batch.target_mask.shape == (2, 18)
    # Row 0's input_ids beyond position 12 should be pad (0)
    assert batch.input_ids[0, 12:].sum() == 0
    # Row 1's last 6 positions should be target (mask=1)
    assert batch.target_mask[1, 12:].sum() == 6
    # Row weights preserved (float32 → exact comparison via approx)
    assert batch.row_weights[0].item() == pytest.approx(1.0)
    assert batch.row_weights[1].item() == pytest.approx(0.2)


def test_collate_produces_three_image_variants_on_laterality_batch():
    rows = [
        _fake_row(input_len=4, target_len=2, laterality=True),
        _fake_row(input_len=4, target_len=2, laterality=False),
    ]
    # Use non-trivial pixel_values so the per-channel mean differs from zero
    rows[0]["pixel_values"] = torch.full((3, 16, 16), 0.5)
    rows[1]["pixel_values"] = torch.full((3, 16, 16), 0.5)
    batch = counterfactual_collate(rows, enable_flipping=True)
    assert batch.pixel_values_full.shape == (2, 3, 16, 16)
    assert batch.pixel_values_masked.shape == (2, 3, 16, 16)
    # Per-channel mean of 0.5 fills should equal 0.5 everywhere
    assert torch.allclose(batch.pixel_values_masked, torch.full_like(batch.pixel_values_masked, 0.5))
    assert batch.pixel_values_flipped is not None
    # Row 1 (non-laterality) should equal the neutral image (per-channel mean)
    assert torch.allclose(batch.pixel_values_flipped[1], torch.full_like(batch.pixel_values_flipped[1], 0.5))


def test_collate_no_flipping_when_disabled():
    rows = [_fake_row(input_len=4, target_len=2, laterality=True)]
    batch = counterfactual_collate(rows, enable_flipping=False)
    assert batch.pixel_values_flipped is None


def test_collate_no_flipping_when_no_laterality_rows():
    rows = [_fake_row(input_len=4, target_len=2, laterality=False)]
    batch = counterfactual_collate(rows, enable_flipping=True)
    assert batch.pixel_values_flipped is None


def test_collate_returns_labels_equal_to_input_ids():
    rows = [_fake_row(input_len=4, target_len=2)]
    batch = counterfactual_collate(rows)
    assert torch.equal(batch.labels, batch.input_ids)


# --- Dataset filter tests (no processor required) -------------------------

def _write_jsonl(tmp_path, rows):
    p = tmp_path / "train.jsonl"
    import json as _json
    with p.open("w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")
    return p


def test_dataset_filter_drops_verifier_only_rows(tmp_path):
    """Generator training requires real multi-token targets. Rows whose only
    text is the verdict label (no evidence_text, no reference_report) must be
    filtered out at __init__ time.
    """
    from pivot_ab.data_loader import MedGemmaSFTDataset

    rows = [
        # Keep: real evidence_text → multi-token target
        {"gt_label": "SUPPORTED", "evidence_text": "The lungs are clear.", "claim_text": "no effusion", "image_path": "a.png"},
        # Keep: reference_report present even though evidence is empty
        {"gt_label": "CONTRADICTED", "evidence_text": "", "reference_report": "Findings: ...", "claim_text": "x", "image_path": "b.png"},
        # Drop: only gt_label as text
        {"gt_label": "SUPPORTED", "evidence_text": "", "claim_text": "y", "image_path": "c.png"},
        # Drop: gt_label not S/C
        {"gt_label": "NO_GT", "evidence_text": "Some text", "claim_text": "z", "image_path": "d.png"},
        # Drop: evidence_text only whitespace
        {"gt_label": "SUPPORTED", "evidence_text": "   ", "claim_text": "w", "image_path": "e.png"},
    ]
    path = _write_jsonl(tmp_path, rows)
    ds = MedGemmaSFTDataset(jsonl_path=path, weights_jsonl=None, processor=None)
    assert len(ds.rows) == 2
    assert ds.rows[0]["claim_text"] == "no effusion"
    assert ds.rows[1]["claim_text"] == "x"


def test_dataset_remaps_filter_weights_to_new_indices(tmp_path):
    """Filter weights file uses the upstream filter's row_idx convention:
    position in the S/C-filtered list (NOT raw-line position in the JSONL).
    The dataset must remap that S/C-position onto its own new-dataset-idx
    after dropping rows lacking text targets, so row_weight lookup hits
    correctly during training.

    This test deliberately interleaves NO_GT rows between S/C rows so the
    raw-line-pos and S/C-pos differ — matching how the real
    groundbench_v6_train.jsonl is laid out (verified 2026-05-05).
    """
    from pivot_ab.data_loader import MedGemmaSFTDataset

    # Layout:                                                     raw_line, sc_idx, fate
    rows = [
        {"gt_label": "SUPPORTED", "evidence_text": "kept0", "claim_text": "a", "image_path": "1.png"},   # 0, 0, NEW=0
        {"gt_label": "NO_GT",     "evidence_text": "skip",  "claim_text": "b", "image_path": "2.png"},   # 1, -, skip
        {"gt_label": "SUPPORTED", "evidence_text": "",      "claim_text": "c", "image_path": "3.png"},   # 2, 1, DROPPED (no text)
        {"gt_label": "NO_GT",     "evidence_text": "skip",  "claim_text": "d", "image_path": "4.png"},   # 3, -, skip
        {"gt_label": "SUPPORTED", "evidence_text": "kept2", "claim_text": "e", "image_path": "5.png"},   # 4, 2, NEW=1
        {"gt_label": "CONTRADICTED","evidence_text": "kept3","claim_text": "f", "image_path": "6.png"},  # 5, 3, NEW=2
    ]
    path = _write_jsonl(tmp_path, rows)

    # Weights file uses the S/C-filtered convention (the upstream filter
    # builds its own S/C-only list before writing row_idx values).
    weights_path = tmp_path / "w.jsonl"
    import json as _json
    with weights_path.open("w") as f:
        for sc_idx, w in [(0, 0.9), (1, 0.5), (2, 0.7), (3, 1.0)]:
            f.write(_json.dumps({"row_idx": sc_idx, "weight": w}) + "\n")

    ds = MedGemmaSFTDataset(jsonl_path=path, weights_jsonl=weights_path, processor=None)
    assert len(ds.rows) == 3
    # sc_idx=0 → new=0 (weight 0.9), sc_idx=1 dropped, sc_idx=2 → new=1 (0.7), sc_idx=3 → new=2 (1.0)
    assert ds.weights == {0: 0.9, 1: 0.7, 2: 1.0}
    # The dropped row's weight (sc_idx=1, weight=0.5) must NOT mis-apply.
    assert 0.5 not in ds.weights.values()


def test_dataset_does_not_use_raw_line_index_for_weights(tmp_path):
    """Regression: ensure weights are keyed by S/C-position, not raw-line-position.

    If the implementation accidentally counted raw lines (the bug Opus caught
    on 2026-05-05), the weight at row_idx=2 would be applied to the row at
    raw-line 2 (which is a SUPPORTED-with-empty-evidence → dropped), and the
    weight at row_idx=4 would be applied to raw-line 4 (the third surviving
    row). This test pins down the correct S/C-position convention.
    """
    from pivot_ab.data_loader import MedGemmaSFTDataset

    rows = [
        {"gt_label": "NO_GT",     "evidence_text": "x",     "claim_text": "n0", "image_path": "0.png"},  # raw=0, sc=-, skip
        {"gt_label": "SUPPORTED", "evidence_text": "first", "claim_text": "s0", "image_path": "1.png"},  # raw=1, sc=0, NEW=0
        {"gt_label": "NO_GT",     "evidence_text": "x",     "claim_text": "n1", "image_path": "2.png"},  # raw=2, sc=-, skip
        {"gt_label": "SUPPORTED", "evidence_text": "second","claim_text": "s1", "image_path": "3.png"},  # raw=3, sc=1, NEW=1
    ]
    path = _write_jsonl(tmp_path, rows)

    weights_path = tmp_path / "w.jsonl"
    import json as _json
    with weights_path.open("w") as f:
        # If the dataset uses S/C-position (correct), sc_idx=0 should map to new=0.
        # If it uses raw-line-pos (buggy), row_idx=0 would map to raw-line-0
        # which is NO_GT (skipped), and the surviving rows would get no weight.
        f.write(_json.dumps({"row_idx": 0, "weight": 0.42}) + "\n")
        f.write(_json.dumps({"row_idx": 1, "weight": 0.99}) + "\n")

    ds = MedGemmaSFTDataset(jsonl_path=path, weights_jsonl=weights_path, processor=None)
    assert len(ds.rows) == 2
    assert ds.weights == {0: 0.42, 1: 0.99}, (
        f"Got {ds.weights} — if this is empty, the dataset is using raw-line-pos "
        f"instead of S/C-position for the weights index. See "
        f"PIVOT_AB_EXECUTION_LOG.md 2026-05-05 Opus pre-flight finding #2."
    )
