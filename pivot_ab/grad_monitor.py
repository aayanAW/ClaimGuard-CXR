"""Gradient-norm + attention monitor for evidence-blindness emergence.

This is the secondary contribution of Pivot A+B (the Pivot-B mechanistic-
interpretability piece bundled into the Pivot-A model paper).

Hypothesis: evidence-blindness emerges during multimodal SFT when the gradient
of the loss with respect to image-token embeddings decays below a threshold
relative to text-token gradients. If the model can solve the loss with text
inputs alone, image-token gradients carry no signal, the cross-modal attention
layers lose their bind to the image, and the model becomes effectively
unimodal.

The monitor logs three quantities at every K training steps:

  R(t) = g_img(t) / (g_img(t) + g_txt(t))            in [0, 1]
       — fraction of total token-embedding gradient norm coming from image
         tokens. R near 0 = image tokens get no learning signal.

  r_cm(t) = effective rank of the cross-modal attention matrix
       — when image tokens are ignored, attention rank toward image columns
         collapses.

  alpha_img(t) = sum of attention weights on image tokens at the verdict /
                 final layer, divided by total attention
       — direct attribution measure.

A logistic regressor fit on (R, r_cm, alpha_img) trajectories predicts whether
post-training IMG > 5pp. The Phase-6 gate is AUROC ≥ 0.75 on held-out
architectures.

The monitor is implemented via PyTorch forward + backward hooks. It adds
negligible overhead because gradient norms are already computed during backprop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MonitorTrace:
    step: list[int] = field(default_factory=list)
    g_img_norm: list[float] = field(default_factory=list)
    g_txt_norm: list[float] = field(default_factory=list)
    R: list[float] = field(default_factory=list)
    cross_modal_rank: list[float] = field(default_factory=list)
    alpha_img: list[float] = field(default_factory=list)

    def to_jsonl(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i in range(len(self.step)):
                f.write(
                    json.dumps(
                        {
                            "step": self.step[i],
                            "g_img_norm": self.g_img_norm[i],
                            "g_txt_norm": self.g_txt_norm[i],
                            "R": self.R[i],
                            "cross_modal_rank": self.cross_modal_rank[i],
                            "alpha_img": self.alpha_img[i],
                        }
                    )
                    + "\n"
                )


class GradMonitor:
    """Hooks into a multimodal model and records gradient + attention statistics.

    Usage:
        monitor = GradMonitor(
            model=model,
            image_token_indices_fn=lambda batch: batch["image_token_mask"],
            cross_modal_attention_layer=model.fusion.layers[-1].cross_attn,
            log_every=50,
        )
        for step, batch in enumerate(loader):
            loss = model(**batch).loss
            loss.backward()
            monitor.step(step)  # captures one observation
            optimizer.step()
            optimizer.zero_grad()
        monitor.trace.to_jsonl(out_path)
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        image_token_indices_fn: Callable[[dict[str, Any]], torch.Tensor],
        cross_modal_attention_layer: nn.Module | None = None,
        log_every: int = 50,
        rank_singular_threshold: float = 1e-3,
    ):
        """
        Args:
            model: the multimodal model being trained.
            image_token_indices_fn: callable(batch) -> bool tensor of shape
                (B, T) with True at positions corresponding to image tokens
                in the fused sequence. Required so we can split gradient norms
                between image-token and text-token positions.
            cross_modal_attention_layer: if provided, must be an nn.Module
                exposing a forward hook that yields attention probabilities
                with shape (B, H, T_q, T_k). The monitor reads attn from a
                hook attached on this module.
            log_every: capture an observation every N steps.
            rank_singular_threshold: relative singular value cutoff for
                effective-rank computation.
        """
        self.model = model
        self.image_token_indices_fn = image_token_indices_fn
        self.cm_layer = cross_modal_attention_layer
        self.log_every = log_every
        self.rank_threshold = rank_singular_threshold
        self.trace = MonitorTrace()

        self._last_batch: dict[str, Any] | None = None
        self._last_attn: torch.Tensor | None = None
        self._embedding_grad: torch.Tensor | None = None
        self._embedding_layer = self._find_embedding_layer(model)

        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

    @staticmethod
    def _find_embedding_layer(model: nn.Module) -> nn.Module:
        """Heuristic: pick the first nn.Embedding submodule."""
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                return module
        raise RuntimeError("could not locate embedding layer in model")

    def _install_hooks(self) -> None:
        # Capture the input-embedding gradient via a backward hook on the
        # embedding output. This is the cleanest way to get per-token
        # gradients without modifying model code.
        def _emb_hook(module, grad_input, grad_output):
            # grad_output is a tuple; the first element has shape (B, T, D).
            if grad_output and grad_output[0] is not None:
                self._embedding_grad = grad_output[0].detach()

        h_emb = self._embedding_layer.register_full_backward_hook(_emb_hook)
        self._hook_handles.append(h_emb)

        if self.cm_layer is not None:
            def _attn_hook(module, inputs, outputs):
                # Heuristic: assume the attention module returns either
                # (output, attn_weights) or just output. We try to extract
                # attention via a registered output structure.
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    attn = outputs[1]
                    if isinstance(attn, torch.Tensor):
                        self._last_attn = attn.detach()

            h_attn = self.cm_layer.register_forward_hook(_attn_hook)
            self._hook_handles.append(h_attn)

    def attach_batch(self, batch: dict[str, Any]) -> None:
        """Call before the forward pass to make the batch available to the monitor."""
        self._last_batch = batch

    def step(self, step_id: int) -> None:
        """Capture one observation if `step_id % log_every == 0`."""
        if step_id % self.log_every != 0:
            return
        if self._embedding_grad is None or self._last_batch is None:
            logger.debug("monitor skipped step %d: missing grad or batch", step_id)
            return
        try:
            image_mask = self.image_token_indices_fn(self._last_batch).to(
                self._embedding_grad.device
            )
        except Exception as e:
            logger.warning("monitor could not derive image_token mask at step %d: %s", step_id, e)
            return

        # Per-token L2 norms of embedding gradients
        # _embedding_grad shape: (B, T, D)
        per_token_norm = self._embedding_grad.norm(dim=-1)  # (B, T)
        if image_mask.shape != per_token_norm.shape:
            logger.warning(
                "monitor mask/grad shape mismatch at step %d: %s vs %s",
                step_id,
                image_mask.shape,
                per_token_norm.shape,
            )
            return
        text_mask = ~image_mask

        g_img = (per_token_norm * image_mask.float()).sum().item()
        g_txt = (per_token_norm * text_mask.float()).sum().item()
        denom = g_img + g_txt
        R = g_img / denom if denom > 0 else 0.0

        # Effective cross-modal rank
        cm_rank = 0.0
        alpha_img = 0.0
        if self._last_attn is not None:
            attn = self._last_attn  # (B, H, T_q, T_k)
            # Average over batch and heads, take attention to image tokens
            attn_avg = attn.mean(dim=(0, 1))  # (T_q, T_k)
            try:
                s = torch.linalg.svdvals(attn_avg)
                if s.numel() > 0:
                    s_max = s[0].clamp_min(1e-12)
                    cm_rank = float((s / s_max > self.rank_threshold).sum())
            except Exception as e:
                logger.debug("svdvals failed at step %d: %s", step_id, e)

            # Image-attention attribution: sum of attention to image-token columns
            try:
                img_cols = image_mask[0]  # use first sample
                if img_cols.shape[0] == attn_avg.shape[1]:
                    img_attn_mass = attn_avg[:, img_cols].sum().item()
                    total_mass = attn_avg.sum().item()
                    alpha_img = img_attn_mass / max(total_mass, 1e-12)
            except Exception as e:
                logger.debug("alpha_img failed at step %d: %s", step_id, e)

        self.trace.step.append(step_id)
        self.trace.g_img_norm.append(g_img)
        self.trace.g_txt_norm.append(g_txt)
        self.trace.R.append(R)
        self.trace.cross_modal_rank.append(cm_rank)
        self.trace.alpha_img.append(alpha_img)

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []


def fit_blindness_predictor(
    *,
    traces: list[MonitorTrace],
    img_labels: list[bool],
    feature_window: tuple[int, int] = (-100, -1),
) -> dict:
    """Fit a logistic regressor that predicts post-training IMG > 5pp from
    training-time monitor trajectories.

    Args:
        traces: one MonitorTrace per training run.
        img_labels: same length as `traces`; True if post-training IMG > 5pp.
        feature_window: slice of the trace (last N observations) to use as
            features. Default uses the last 100 observations.

    Returns:
        Summary dict with predictor coefficients, AUROC, and intercept.
    """
    import numpy as np

    if len(traces) != len(img_labels):
        raise ValueError("traces and img_labels must be same length")
    if len(traces) < 4:
        raise ValueError(f"need at least 4 traces to fit predictor; got {len(traces)}")

    X_rows = []
    for t in traces:
        s, e = feature_window
        R = np.array(t.R[s:e]) if len(t.R) >= abs(s) else np.array(t.R)
        cm = np.array(t.cross_modal_rank[s:e]) if len(t.cross_modal_rank) >= abs(s) else np.array(t.cross_modal_rank)
        a = np.array(t.alpha_img[s:e]) if len(t.alpha_img) >= abs(s) else np.array(t.alpha_img)
        if R.size == 0:
            R = np.array([0.0])
        if cm.size == 0:
            cm = np.array([0.0])
        if a.size == 0:
            a = np.array([0.0])
        feats = np.array(
            [
                R.mean(),
                R.min(),
                R.std() if R.size > 1 else 0.0,
                cm.mean(),
                cm.min(),
                a.mean(),
                a.min(),
            ]
        )
        X_rows.append(feats)
    X = np.vstack(X_rows)
    y = np.array(img_labels, dtype=int)

    # Standardize and fit logistic regression manually (avoids sklearn dep).
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    Xs = (X - mu) / sd

    # Weighted least squares logistic regression via IRLS.
    w = np.zeros(Xs.shape[1])
    b = 0.0
    for _ in range(100):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xs.T @ (p - y) / len(y) + 1e-3 * w
        grad_b = float((p - y).mean())
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b

    # Compute training AUROC for diagnostic.
    z = Xs @ w + b
    order = np.argsort(z)
    y_sorted = y[order]
    pos = y_sorted.sum()
    neg = len(y_sorted) - pos
    if pos == 0 or neg == 0:
        auroc = float("nan")
    else:
        # Mann-Whitney U statistic / (pos*neg)
        cum_neg = 0
        u = 0
        for yy in y_sorted:
            if yy == 0:
                cum_neg += 1
            else:
                u += cum_neg
        auroc = u / (pos * neg)

    return {
        "feature_names": [
            "R_mean",
            "R_min",
            "R_std",
            "cm_rank_mean",
            "cm_rank_min",
            "alpha_img_mean",
            "alpha_img_min",
        ],
        "weights": w.tolist(),
        "intercept": float(b),
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "n_samples": int(len(y)),
        "training_auroc": float(auroc),
    }
