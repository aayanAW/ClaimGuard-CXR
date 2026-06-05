"""Unit tests for pivot_ab/img_correct.py."""

from __future__ import annotations

import torch
import pytest

from pivot_ab.img_correct import img_correct_verifier


def test_perfect_grounding_no_inversion():
    """All N rows: full=correct, masked=incorrect, with substantial prob drop."""
    N = 10
    targets = torch.zeros(N, dtype=torch.long)
    pred_full = torch.zeros(N, dtype=torch.long)  # all correct
    pred_masked = torch.ones(N, dtype=torch.long)  # all incorrect
    prob_full_correct = torch.full((N,), 0.9)
    prob_masked_correct = torch.full((N,), 0.1)  # large drop

    res = img_correct_verifier(
        pred_full=pred_full,
        pred_masked=pred_masked,
        prob_full_correct=prob_full_correct,
        prob_masked_correct=prob_masked_correct,
        targets=targets,
        margin_tau=0.1,
    )
    assert res.img_pp == pytest.approx(100.0)  # 100% - 0% = 100pp
    assert res.img_correct_pp == pytest.approx(100.0)  # all 10 cases qualify
    assert res.induced_inversion_pp == pytest.approx(0.0)


def test_pure_induced_inversion_caught():
    """Full=correct, masked=incorrect, but the prob drop is below margin_tau —
    looks like grounding under naive IMG, but IMG_correct correctly excludes it."""
    N = 10
    targets = torch.zeros(N, dtype=torch.long)
    pred_full = torch.zeros(N, dtype=torch.long)
    pred_masked = torch.ones(N, dtype=torch.long)
    prob_full_correct = torch.full((N,), 0.55)
    prob_masked_correct = torch.full((N,), 0.50)  # tiny drop

    res = img_correct_verifier(
        pred_full=pred_full,
        pred_masked=pred_masked,
        prob_full_correct=prob_full_correct,
        prob_masked_correct=prob_masked_correct,
        targets=targets,
        margin_tau=0.1,  # require 10pp drop
    )
    assert res.img_pp == pytest.approx(100.0)  # naive IMG sees 100pp
    assert res.img_correct_pp == pytest.approx(0.0)  # IMG_correct sees 0
    assert res.induced_inversion_pp == pytest.approx(100.0)  # all induced inversion


def test_mixed_grounding_and_inversion():
    """5 rows of genuine grounding + 5 rows of induced inversion."""
    N = 10
    targets = torch.zeros(N, dtype=torch.long)
    pred_full = torch.zeros(N, dtype=torch.long)
    pred_masked = torch.ones(N, dtype=torch.long)  # all wrong on masked
    # First 5: large prob drop (grounding); next 5: tiny drop (inversion)
    prob_full_correct = torch.tensor([0.9] * 5 + [0.55] * 5)
    prob_masked_correct = torch.tensor([0.1] * 5 + [0.50] * 5)

    res = img_correct_verifier(
        pred_full=pred_full,
        pred_masked=pred_masked,
        prob_full_correct=prob_full_correct,
        prob_masked_correct=prob_masked_correct,
        targets=targets,
        margin_tau=0.1,
    )
    assert res.img_pp == pytest.approx(100.0)
    assert res.img_correct_pp == pytest.approx(50.0)  # 5/10 with sufficient drop
    assert res.induced_inversion_pp == pytest.approx(50.0)


def test_no_change_zero_img():
    """Both correct → IMG = 0, IMG_correct = 0."""
    N = 8
    targets = torch.zeros(N, dtype=torch.long)
    pred_full = torch.zeros(N, dtype=torch.long)
    pred_masked = torch.zeros(N, dtype=torch.long)
    prob_full_correct = torch.full((N,), 0.8)
    prob_masked_correct = torch.full((N,), 0.7)

    res = img_correct_verifier(
        pred_full=pred_full,
        pred_masked=pred_masked,
        prob_full_correct=prob_full_correct,
        prob_masked_correct=prob_masked_correct,
        targets=targets,
        margin_tau=0.1,
    )
    assert res.img_pp == pytest.approx(0.0)
    assert res.img_correct_pp == pytest.approx(0.0)
    assert res.induced_inversion_pp == pytest.approx(0.0)


def test_shape_mismatch_raises():
    targets = torch.zeros(5, dtype=torch.long)
    pred_full = torch.zeros(5, dtype=torch.long)
    pred_masked = torch.zeros(4, dtype=torch.long)  # wrong size
    with pytest.raises(ValueError, match="shape"):
        img_correct_verifier(
            pred_full=pred_full,
            pred_masked=pred_masked,
            prob_full_correct=torch.zeros(5),
            prob_masked_correct=torch.zeros(5),
            targets=targets,
        )


def test_generator_path_requires_caller_to_pre_shift_for_causal_lm():
    """Document the eval-pipeline shift requirement.

    `img_correct_generator` does NOT apply the causal-LM shift (logit at
    position t predicts position t+1). Callers using a causal LM (e.g., the
    Modal eval entrypoint) MUST pre-shift their tensors:

        log_p_shifted = log_p[:, :-1, :]
        target_ids_shifted = target_ids[:, 1:]
        target_mask_shifted = target_mask[:, 1:]

    Without the shift, argmax accuracy is ~0 because the position-t logit's
    argmax doesn't match the position-t target token. The 2026-05-03 eval
    pipeline missed this and reported 0% accuracy on a trained checkpoint
    until fixed.

    This test verifies the documented contract: when caller has already
    aligned (target[t] == argmax of log_p[t]), accuracy is non-zero.
    """
    import torch
    from pivot_ab.img_correct import img_correct_generator
    V = 8
    target_id = 3
    # Already-shifted alignment: log_p at position t peaks at target_id, and
    # target_ids[t] == target_id at every position.
    log_p_full = torch.full((1, 4, V), -10.0)
    log_p_full[..., target_id] = 10.0
    log_p_full = torch.log_softmax(log_p_full, dim=-1)
    target_ids = torch.full((1, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(1, 4, dtype=torch.long)

    res = img_correct_generator(
        log_p_full=log_p_full,
        log_p_masked=log_p_full,  # alt = full → IMG = 0
        target_ids=target_ids,
        target_mask=target_mask,
        margin_tau=0.1,
    )
    # When shifted properly and full=alt, accuracy_full=100% and IMG=0.
    assert res.accuracy_full == pytest.approx(100.0)
    assert res.img_pp == pytest.approx(0.0)


def test_replicates_claimguard_v53_overstatement():
    """Synthetic stand-in for the v5.3-contrast configuration: high IMG, but
    a substantial fraction is induced inversion when the consistency loss
    penalises p_masked aggressively without rewarding p_full."""
    N = 1000
    targets = torch.zeros(N, dtype=torch.long)
    pred_full = torch.zeros(N, dtype=torch.long)  # all correct on full
    pred_masked = torch.ones(N, dtype=torch.long)  # all flipped on masked
    # Mix: 30% genuine grounding (large drop), 70% induced inversion (small drop)
    n_grounding = 300
    n_inversion = 700
    prob_full_correct = torch.cat(
        [torch.full((n_grounding,), 0.92), torch.full((n_inversion,), 0.55)]
    )
    prob_masked_correct = torch.cat(
        [torch.full((n_grounding,), 0.05), torch.full((n_inversion,), 0.48)]
    )

    res = img_correct_verifier(
        pred_full=pred_full,
        pred_masked=pred_masked,
        prob_full_correct=prob_full_correct,
        prob_masked_correct=prob_masked_correct,
        targets=targets,
        margin_tau=0.1,
    )
    assert res.img_pp == pytest.approx(100.0)
    assert res.img_correct_pp == pytest.approx(30.0)
    assert res.induced_inversion_pp == pytest.approx(70.0)
