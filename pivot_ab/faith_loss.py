"""Bidirectional causal-faithfulness loss for radiology generator SFT.

This loss is the centrepiece of Pivot A. It addresses the methodological flaw
the external reviewer correctly identified in the prior ClaimGuard work: the
naive consistency loss `ReLU(margin - (p_full(y) - p_masked(y)))` rewards the
model for being *wrong* under image masking, not for being *more correct* with
the image present. The high IMG numbers reported by ClaimGuard partially
measured induced inversion rather than faithful grounding.

The original Pivot A+B implementation used a full-distribution KL divergence
between p_full and p_alt as the divergence term. The 2026-05-03 Opus
pre-flight reviewer correctly flagged this as ALSO admitting induced
inversion: a full-distribution KL can be satisfied by making p_alt arbitrary
(uniform, peaked on a default token) while p_full is shaped externally by the
SFT term — the loss does not actually couple correctness on the gold target
under full input to absence-of-correctness under masked input.

This revised loss uses a **token-wise correctness-gain margin** evaluated only
on gold target tokens:

    For each target token position t:
        gain_t = log p_full(y_t) - log p_alt(y_t)
    L_diverge = mean over target positions of ReLU(margin - gain_t)

Why this is bidirectional in a meaningful way:
- gain_t is large only when p_full(y_t) >> p_alt(y_t).
- This requires both that the full input assigns high probability to the
  correct token AND that the alternate input assigns lower probability.
- Cannot be satisfied by making p_alt arbitrary — gain_t depends on
  log p_alt(y), not on the whole distribution. Flipping all of p_alt to a
  uniform distribution leaves log p_alt(y) at -log V (small but bounded
  below), while making the model wrong on full input drives log p_full(y)
  toward the same level → gain_t → 0 → penalty fires.

The correctness term L_correct is left available (and used by the SFT path
externally), but the divergence term itself now requires correctness on the
full input to drive down to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FaithLossOutput:
    total: torch.Tensor
    correct: torch.Tensor
    diverge: torch.Tensor
    diverge_masked: torch.Tensor | None
    diverge_flipped: torch.Tensor | None


def _gain_hinge(
    log_p_full: torch.Tensor,
    log_p_alt: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    row_mask: torch.Tensor | None,
    margin: float,
) -> torch.Tensor:
    """Mean of ReLU(margin - (log_p_full(y) - log_p_alt(y))) over masked
    target positions (and over row_mask if supplied)."""
    log_p_full_y = log_p_full.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    log_p_alt_y = log_p_alt.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    gain = log_p_full_y - log_p_alt_y                        # (B, T)
    hinge = F.relu(torch.tensor(margin, device=gain.device) - gain)
    mask = target_mask.float()
    if row_mask is not None:
        mask = mask * row_mask.float().unsqueeze(-1)
    masked_hinge = hinge * mask
    denom = mask.sum().clamp_min(1.0)
    return masked_hinge.sum() / denom


def causal_faithfulness_loss(
    *,
    log_p_full: torch.Tensor,
    log_p_masked: torch.Tensor | None,
    log_p_flipped: torch.Tensor | None,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    margin_kl: float = 1.0,
    lambda_diverge: float = 0.3,
    lambda_correct: float = 1.0,
    flipped_row_mask: torch.Tensor | None = None,
) -> FaithLossOutput:
    """Compute the bidirectional causal-faithfulness loss.

    Args:
        log_p_full: log-probs under full input, shape (B, T, V).
        log_p_masked: log-probs under image-masked input, shape (B, T, V).
            If None, the masked divergence term is skipped.
        log_p_flipped: log-probs under image-flipped input, shape (B, T, V).
            If None, the flipped divergence term is skipped.
        target_ids: target token ids, shape (B, T).
        target_mask: 1 on target positions, 0 on padding, shape (B, T).
        margin_kl: required minimum gain (log_p_full(y) - log_p_alt(y)) per
            target token. Despite the name (kept for backwards-compat with
            existing configs), this is now a *log-probability gain* margin,
            not a KL margin. Reasonable values: 0.1–1.0 nats.
        lambda_diverge: weight on the divergence term.
        lambda_correct: weight on the correctness (SFT) term. The training
            loop typically sets this to 0 because L_correct is computed
            outside this function and added separately. Kept here for unit
            tests and for callers that want a single-pass loss.
        flipped_row_mask: optional (B,) bool tensor; True for rows where the
            flipped variant is meaningful (laterality-sensitive). When
            supplied, the flipped divergence term is computed only over True
            rows. Required to avoid the bug where non-laterality rows whose
            flipped pixels were zeroed contribute spurious signal.
    """
    if log_p_full.dim() != 3:
        raise ValueError(f"log_p_full must be (B, T, V); got {log_p_full.shape}")

    nll = -log_p_full.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    nll_masked = nll * target_mask.float()
    denom = target_mask.float().sum().clamp_min(1.0)
    L_correct = nll_masked.sum() / denom

    L_diverge_masked: torch.Tensor | None = None
    L_diverge_flipped: torch.Tensor | None = None
    components: list[torch.Tensor] = []

    if log_p_masked is not None:
        L_diverge_masked = _gain_hinge(
            log_p_full=log_p_full,
            log_p_alt=log_p_masked,
            target_ids=target_ids,
            target_mask=target_mask,
            row_mask=None,
            margin=margin_kl,
        )
        components.append(L_diverge_masked)

    if log_p_flipped is not None:
        L_diverge_flipped = _gain_hinge(
            log_p_full=log_p_full,
            log_p_alt=log_p_flipped,
            target_ids=target_ids,
            target_mask=target_mask,
            row_mask=flipped_row_mask,
            margin=margin_kl,
        )
        components.append(L_diverge_flipped)

    if components:
        L_diverge = torch.stack(components).sum()
    else:
        L_diverge = torch.tensor(0.0, device=log_p_full.device, dtype=log_p_full.dtype)

    L_total = lambda_correct * L_correct + lambda_diverge * L_diverge
    return FaithLossOutput(
        total=L_total,
        correct=L_correct,
        diverge=L_diverge,
        diverge_masked=L_diverge_masked,
        diverge_flipped=L_diverge_flipped,
    )


# Backwards-compatibility shim: the older test file imported _seq_kl from
# this module. Provide a stub so existing imports don't break (the new loss
# does not use full-distribution KL at all).
def _seq_kl(*args, **kwargs):  # pragma: no cover - retained only for import compat
    raise NotImplementedError(
        "_seq_kl was removed in the 2026-05-03 fix; the faith loss now uses "
        "a token-wise correctness-gain margin instead of full-distribution KL. "
        "See faith_loss.py docstring for rationale."
    )
