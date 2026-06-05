"""IMG_correct: a methodologically refined image-masking diagnostic.

The naive image-masking gap (IMG) is

    IMG(f) = acc(f, D) - acc(f, D_{img=0})

This metric is insensitive to a critical failure mode that the consistency
loss `ReLU(margin - (p_full(y) - p_masked(y)))` actively trains for: the
model can lower its masked-input correctness to ZERO by always flipping its
prediction under masking, even when the flipped prediction is no closer to
the truth than chance. This *induced inversion* is observationally identical
to faithful grounding loss under the IMG metric, but mechanistically distinct.

IMG_correct addresses this by counting only the cases where the model is

  (a) correct on the full input,
  (b) incorrect on the masked input, AND
  (c) the masked-input correct-class probability is LOWER than the full-input
      correct-class probability by at least a margin tau.

The third condition rules out "the model was already uncertain" cases. The
gap between IMG and IMG_correct quantifies how much of any reported
faithfulness improvement is induced inversion vs faithful grounding.

For verifiers (binary classification), this is a straightforward per-row
check. For generators (sequence outputs), we apply the same logic per token
on the reference target sequence and average.

Reusable on prior work too: report IMG_correct on the v5.0–v5.3 + v6.0-retrain
ClaimGuard configurations to quantify how much of their reported IMG of
69.24pp was induced inversion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class IMGCorrectResult:
    n_examples: int
    img_pp: float
    img_correct_pp: float
    induced_inversion_pp: float
    accuracy_full: float
    accuracy_masked: float
    fraction_correct_to_correct: float
    fraction_correct_to_incorrect_with_margin: float
    fraction_correct_to_incorrect_no_margin: float
    fraction_incorrect_to_correct: float
    fraction_incorrect_to_incorrect: float
    margin_tau: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_jsonl_row(self) -> str:
        return json.dumps(self.to_dict())


def img_correct_verifier(
    *,
    pred_full: torch.Tensor,
    pred_masked: torch.Tensor,
    prob_full_correct: torch.Tensor,
    prob_masked_correct: torch.Tensor,
    targets: torch.Tensor,
    margin_tau: float = 0.1,
) -> IMGCorrectResult:
    """IMG_correct for a binary or multi-class verifier.

    Args:
        pred_full: predicted class under full input, shape (N,).
        pred_masked: predicted class under image-masked input, shape (N,).
        prob_full_correct: model's predicted probability of the TRUE class
            under full input, shape (N,) in [0, 1].
        prob_masked_correct: same under image-masked input, shape (N,) in [0, 1].
        targets: true class ids, shape (N,).
        margin_tau: minimum drop in correct-class probability under masking
            that counts as "model relied on the image."

    Returns:
        IMGCorrectResult with per-cell counts, IMG, and IMG_correct in pp.
    """
    if not (
        pred_full.shape == pred_masked.shape == prob_full_correct.shape == prob_masked_correct.shape == targets.shape
    ):
        raise ValueError("all tensors must have shape (N,)")

    n = pred_full.shape[0]
    correct_full = (pred_full == targets)
    correct_masked = (pred_masked == targets)
    prob_drop = prob_full_correct - prob_masked_correct
    enough_drop = prob_drop >= margin_tau

    n_cc = int((correct_full & correct_masked).sum())
    n_ci_no_margin = int((correct_full & ~correct_masked).sum())
    n_ci_with_margin = int((correct_full & ~correct_masked & enough_drop).sum())
    n_ic = int((~correct_full & correct_masked).sum())
    n_ii = int((~correct_full & ~correct_masked).sum())

    acc_full = float(correct_full.float().mean()) * 100.0
    acc_masked = float(correct_masked.float().mean()) * 100.0
    img_pp = acc_full - acc_masked
    # IMG_correct counts only the (correct → incorrect with sufficient prob drop)
    # cases as "image-grounded transitions"; subtracts the induced-inversion
    # contribution.
    img_correct_pp = (n_ci_with_margin / max(1, n)) * 100.0
    induced_inversion_pp = ((n_ci_no_margin - n_ci_with_margin) / max(1, n)) * 100.0

    return IMGCorrectResult(
        n_examples=n,
        img_pp=img_pp,
        img_correct_pp=img_correct_pp,
        induced_inversion_pp=induced_inversion_pp,
        accuracy_full=acc_full,
        accuracy_masked=acc_masked,
        fraction_correct_to_correct=n_cc / max(1, n),
        fraction_correct_to_incorrect_with_margin=n_ci_with_margin / max(1, n),
        fraction_correct_to_incorrect_no_margin=n_ci_no_margin / max(1, n),
        fraction_incorrect_to_correct=n_ic / max(1, n),
        fraction_incorrect_to_incorrect=n_ii / max(1, n),
        margin_tau=margin_tau,
    )


def img_correct_generator(
    *,
    log_p_full: torch.Tensor,
    log_p_masked: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    margin_tau: float = 0.1,
) -> IMGCorrectResult:
    """IMG_correct for an autoregressive generator.

    For each target token position we compare:
        prob_full(target) and prob_masked(target)
        argmax_full and argmax_masked

    A token position contributes to IMG_correct when:
        argmax_full == target AND argmax_masked != target AND
        prob_full(target) - prob_masked(target) >= margin_tau.

    Token positions where the full-input model was already wrong, or where
    the masked-input model is also right, do not count.

    Args:
        log_p_full: log-probabilities under full input, shape (B, T, V).
        log_p_masked: log-probabilities under image-masked input, shape (B, T, V).
        target_ids: target token ids, shape (B, T).
        target_mask: 1 on target positions, 0 on padding, shape (B, T).
        margin_tau: minimum probability drop to count.

    Returns:
        IMGCorrectResult aggregated over all valid target token positions.
    """
    if log_p_full.shape != log_p_masked.shape:
        raise ValueError("log_p shapes must match")

    pred_full = log_p_full.argmax(dim=-1)  # (B, T)
    pred_masked = log_p_masked.argmax(dim=-1)
    prob_full_target = log_p_full.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).exp()
    prob_masked_target = log_p_masked.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).exp()

    valid = target_mask.bool()
    pred_full_v = pred_full[valid]
    pred_masked_v = pred_masked[valid]
    prob_full_v = prob_full_target[valid]
    prob_masked_v = prob_masked_target[valid]
    targets_v = target_ids[valid]

    return img_correct_verifier(
        pred_full=pred_full_v,
        pred_masked=pred_masked_v,
        prob_full_correct=prob_full_v,
        prob_masked_correct=prob_masked_v,
        targets=targets_v,
        margin_tau=margin_tau,
    )


def write_results(
    results: dict[str, IMGCorrectResult],
    output_path: Path,
) -> None:
    """Write a dict of {config_name: result} to a JSONL where each row has the
    config name plus the IMGCorrectResult fields."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for name, res in results.items():
            row = {"config": name, **res.to_dict()}
            f.write(json.dumps(row) + "\n")
    logger.info("wrote IMG_correct results: %d configs to %s", len(results), output_path)
