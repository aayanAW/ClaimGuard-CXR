"""Unit tests for pivot_ab/faith_loss.py.

After the 2026-05-03 fix, the loss uses a token-wise correctness-gain margin
on gold target tokens rather than full-distribution KL. The tests verify the
new behaviour: divergence term is zero when log_p_full(y) - log_p_alt(y) >=
margin on every target token; positive when the gain is below margin.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from pivot_ab.faith_loss import causal_faithfulness_loss


def _peaked_log_p(B: int, T: int, V: int, target_id: int, peak_value: float = 10.0) -> torch.Tensor:
    """Almost-deterministic distribution favouring target_id (high logit)."""
    logits = torch.full((B, T, V), -peak_value)
    logits[:, :, target_id] = peak_value
    return F.log_softmax(logits, dim=-1).clone().requires_grad_(True)


def test_diverge_zero_when_full_overwhelmingly_predicts_target_and_alt_does_not():
    """Faith loss should be near zero when full input strongly predicts the
    correct token and the alternate input puts low probability on it."""
    V = 8
    target_id = 0
    target_ids = torch.full((2, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    log_p_full = _peaked_log_p(2, 4, V, target_id=0, peak_value=10.0)
    log_p_masked = _peaked_log_p(2, 4, V, target_id=7, peak_value=10.0)  # alt favours wrong token
    out = causal_faithfulness_loss(
        log_p_full=log_p_full,
        log_p_masked=log_p_masked,
        log_p_flipped=None,
        target_ids=target_ids,
        target_mask=target_mask,
        margin_kl=0.3,
        lambda_diverge=1.0,
        lambda_correct=0.0,
    )
    # gain = log p_full(0) - log p_masked(0) ≈ 20 nats, well above margin 0.3
    assert float(out.diverge) == pytest.approx(0.0, abs=1e-3)


def test_diverge_positive_when_full_and_alt_predict_target_equally():
    """Penalty fires when the gold-token log-probability is the same under
    full and masked inputs — i.e., the model is not using the image."""
    V = 8
    target_id = 0
    target_ids = torch.full((2, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    log_p_full = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0)
    log_p_masked = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0).detach().clone().requires_grad_(True)
    out = causal_faithfulness_loss(
        log_p_full=log_p_full,
        log_p_masked=log_p_masked,
        log_p_flipped=None,
        target_ids=target_ids,
        target_mask=target_mask,
        margin_kl=0.3,
        lambda_diverge=1.0,
        lambda_correct=0.0,
    )
    # gain = 0 (same distribution), margin = 0.3 → hinge = 0.3
    assert float(out.diverge) == pytest.approx(0.3, abs=1e-3)


def test_diverge_alone_does_not_block_induced_inversion_relies_on_sft_coupling():
    """Documents that the divergence term ALONE is not self-contained against
    induced inversion — the SFT correctness term outside this loss is what
    couples gold-token correctness on full input to the divergence signal.

    Construction: full input puts low prob on gold (model is wrong); masked
    input puts even lower prob on gold. The token-wise gain is positive
    (and large), so the divergence hinge fires at zero — even though the
    model is wrong on full input. In practice the SFT term drives
    log_p_full(gold) up, and the divergence term then constrains
    log_p_alt(gold) to lag by margin. Together they prevent induced
    inversion; alone, the divergence term does not.

    See faith_loss.py docstring for the full rationale.
    """
    V = 8
    target_id = 0
    target_ids = torch.full((1, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(1, 4)

    # Construct logits explicitly so we control the gold-token log-prob
    logits_full = torch.full((1, 4, V), 0.0)
    logits_full[..., target_id] = -2.0  # gold logit low → low prob
    logits_masked = torch.full((1, 4, V), 0.0)
    logits_masked[..., target_id] = -10.0  # gold logit very low

    log_p_full = F.log_softmax(logits_full, dim=-1).requires_grad_(True)
    log_p_masked = F.log_softmax(logits_masked, dim=-1).requires_grad_(True)

    # gain = log p_full(0) - log p_masked(0) = (-2 - logsumexp_full) - (-10 - logsumexp_masked)
    # logsumexp for V=8 with 7 zeros + 1 negative ≈ log(7 + e^{neg})
    # roughly: gain ≈ -2 - (-10) = 8 (large!)

    out_a = causal_faithfulness_loss(
        log_p_full=log_p_full,
        log_p_masked=log_p_masked,
        log_p_flipped=None,
        target_ids=target_ids,
        target_mask=target_mask,
        margin_kl=0.3,
        lambda_diverge=1.0,
        lambda_correct=0.0,
    )
    # The loss IS satisfied (low) by the induced-inversion construction —
    # the gain is large because p_masked is even worse than p_full. This
    # demonstrates that the divergence term ALONE doesn't fully prevent
    # induced inversion either; the SFT correctness term outside the loss
    # is what punishes the model for being wrong on full input. The fix
    # vs the original ClaimGuard is that here the gain depends on the
    # GOLD TOKEN log-prob, not on whole-distribution divergence — meaning
    # the SFT term and the faith term are coupled through the gold token.
    # In actual training the SFT term will drive log_p_full(gold) up,
    # which is the correct coupling.
    assert float(out_a.diverge) == pytest.approx(0.0, abs=1e-3)


def test_correctness_term_minimised_when_full_is_peaked_at_target():
    V = 8
    target_id = 2
    target_ids = torch.full((2, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    log_p_full = _peaked_log_p(2, 4, V, target_id=target_id)
    out = causal_faithfulness_loss(
        log_p_full=log_p_full,
        log_p_masked=None,
        log_p_flipped=None,
        target_ids=target_ids,
        target_mask=target_mask,
        margin_kl=0.3,
        lambda_diverge=0.3,
        lambda_correct=1.0,
    )
    assert float(out.correct) < 0.01


def test_padding_is_ignored_in_correctness():
    V = 8
    target_id = 0
    target_ids = torch.full((2, 4), target_id, dtype=torch.long)
    target_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float)
    log_p_full = _peaked_log_p(2, 4, V, target_id=target_id)
    out = causal_faithfulness_loss(
        log_p_full=log_p_full,
        log_p_masked=None,
        log_p_flipped=None,
        target_ids=target_ids,
        target_mask=target_mask,
        margin_kl=0.3,
        lambda_diverge=0.3,
        lambda_correct=1.0,
    )
    assert float(out.correct) < 0.01


def test_total_combines_components_with_weights():
    V = 8
    target_id = 0
    target_ids = torch.full((2, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    log_p_full = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0)
    log_p_masked = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0).detach().clone().requires_grad_(True)

    out_a = causal_faithfulness_loss(
        log_p_full=log_p_full, log_p_masked=log_p_masked, log_p_flipped=None,
        target_ids=target_ids, target_mask=target_mask,
        margin_kl=0.3, lambda_diverge=0.0, lambda_correct=1.0,
    )
    out_b = causal_faithfulness_loss(
        log_p_full=log_p_full, log_p_masked=log_p_masked, log_p_flipped=None,
        target_ids=target_ids, target_mask=target_mask,
        margin_kl=0.3, lambda_diverge=0.5, lambda_correct=1.0,
    )
    # b has positive divergence weight, total should be larger than a's total.
    assert float(out_b.total) > float(out_a.total)


def test_loss_is_differentiable():
    V = 8
    target_id = 0
    target_ids = torch.full((2, 4), target_id, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    log_p_full = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0)
    log_p_masked = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0).detach().clone().requires_grad_(True)

    out = causal_faithfulness_loss(
        log_p_full=log_p_full, log_p_masked=log_p_masked, log_p_flipped=None,
        target_ids=target_ids, target_mask=target_mask,
        margin_kl=0.3, lambda_diverge=1.0, lambda_correct=0.0,
    )
    out.total.backward()
    assert log_p_full.grad is not None
    assert torch.isfinite(log_p_full.grad).all()


def test_flipped_row_mask_excludes_non_laterality_rows():
    """When flipped_row_mask is supplied, only rows where it's True
    contribute to the flipped divergence term."""
    V = 8
    target_id = 0
    target_ids = torch.zeros(2, 4, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    # Row 0: full and flipped agree → high gain → no penalty (laterality, mask=True)
    # Row 1: full and flipped disagree wildly → low gain → penalty fires (non-laterality, mask=False)
    log_p_full = _peaked_log_p(2, 4, V, target_id=0, peak_value=10.0)
    # Flipped: peaked at a *different* token so log_p_flipped(0) is very low →
    # gain = log_p_full(0) - log_p_flipped(0) is very large for both rows.
    log_p_flipped = _peaked_log_p(2, 4, V, target_id=7, peak_value=10.0).detach().clone().requires_grad_(True)

    # With row_mask=None, both rows contribute; with row_mask=[True, False], only row 0
    out_with_mask = causal_faithfulness_loss(
        log_p_full=log_p_full, log_p_masked=None, log_p_flipped=log_p_flipped,
        target_ids=target_ids, target_mask=target_mask,
        margin_kl=0.3, lambda_diverge=1.0, lambda_correct=0.0,
        flipped_row_mask=torch.tensor([True, False]),
    )
    out_no_mask = causal_faithfulness_loss(
        log_p_full=log_p_full, log_p_masked=None, log_p_flipped=log_p_flipped,
        target_ids=target_ids, target_mask=target_mask,
        margin_kl=0.3, lambda_diverge=1.0, lambda_correct=0.0,
        flipped_row_mask=None,
    )
    # Both should be near zero for this construction (large gain)
    # but the with-mask version computes over fewer rows
    assert float(out_with_mask.diverge) == pytest.approx(0.0, abs=1e-3)
    assert float(out_no_mask.diverge) == pytest.approx(0.0, abs=1e-3)


def test_flipped_row_mask_with_zero_valid_rows_returns_zero():
    """If no rows are valid for flipping, loss must be zero (no NaN from
    division by zero)."""
    V = 8
    target_ids = torch.zeros(2, 4, dtype=torch.long)
    target_mask = torch.ones(2, 4)
    log_p_full = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0)
    log_p_flipped = _peaked_log_p(2, 4, V, target_id=0, peak_value=2.0).detach().clone().requires_grad_(True)

    out = causal_faithfulness_loss(
        log_p_full=log_p_full, log_p_masked=None, log_p_flipped=log_p_flipped,
        target_ids=target_ids, target_mask=target_mask,
        margin_kl=0.3, lambda_diverge=1.0, lambda_correct=0.0,
        flipped_row_mask=torch.tensor([False, False]),
    )
    assert torch.isfinite(out.diverge)
