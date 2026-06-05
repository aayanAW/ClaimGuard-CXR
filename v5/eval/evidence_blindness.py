"""Evidence-blindness diagnostic: IMG, ESG, and IPG metrics.

Implements the three counterfactual-based metrics described in
``ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md`` Section 7. Each metric compares a
verifier's accuracy under a baseline condition to its accuracy under a
counterfactual intervention that disrupts a specific input stream. A small
gap means the verifier is not using that input stream.

Usage (from Python):

    from v5.eval.evidence_blindness import run_diagnostic
    report = run_diagnostic(
        model_ckpt="/data/checkpoints/claimguard_v5/v5_0_base/best.pt",
        val_jsonl="/data/groundbench_v5/all/groundbench_v5_val.jsonl",
        image_root="/data",
        n_shuffle_seeds=3,
    )
    # report contains IMG, ESG, IPG, per-category breakdowns, and the binary
    # evidence-blind verdict under the 5pp default threshold.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Summary returned by ``run_diagnostic``."""
    n_test: int
    acc_full: float
    acc_image_zeroed: float
    acc_evidence_shuffled: float
    acc_laterality_flipped: float | None  # None if laterality subset empty
    img_gap_pp: float
    esg_gap_pp: float
    ipg_gap_pp: float | None
    per_category: dict[str, dict[str, float]]
    evidence_blind: bool
    threshold_pp: float

    def to_dict(self) -> dict:
        return {
            "n_test": self.n_test,
            "acc_full": self.acc_full,
            "acc_image_zeroed": self.acc_image_zeroed,
            "acc_evidence_shuffled": self.acc_evidence_shuffled,
            "acc_laterality_flipped": self.acc_laterality_flipped,
            "img_gap_pp": self.img_gap_pp,
            "esg_gap_pp": self.esg_gap_pp,
            "ipg_gap_pp": self.ipg_gap_pp,
            "per_category": self.per_category,
            "evidence_blind": self.evidence_blind,
            "threshold_pp": self.threshold_pp,
        }


def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return float((preds == labels).float().mean().item())


def _run_forward_all(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    *,
    zero_image: bool = False,
    flip_image: bool = False,
    shuffle_evidence: bool = False,
    shuffle_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run the model on every batch and return stacked (preds, labels, categories)."""
    model.eval()
    preds_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    cats_all: list[str] = []
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    with torch.no_grad():
        for batch in dataloader:
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            cats = batch.get("finding_category", [""] * ii.size(0))

            if zero_image:
                pv = torch.zeros_like(pv)
            if flip_image:
                pv = torch.flip(pv, dims=[-1])
            if shuffle_evidence:
                # Derange within batch, rejecting identity with a single shift
                n = ii.size(0)
                if n >= 2:
                    perm = list(range(n))
                    (rng or random).shuffle(perm)
                    if any(perm[i] == i for i in range(n)):
                        perm = [(p + 1) % n for p in perm]
                    ii = ii[perm]
                    am = am[perm]

            out = model(pv, ii, am)
            preds = out["verdict_logits"].argmax(dim=-1)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
            if isinstance(cats, (list, tuple)):
                cats_all.extend([str(c) for c in cats])
            else:
                cats_all.extend([""] * ii.size(0))
    return torch.cat(preds_all), torch.cat(labels_all), cats_all


def _is_laterality_claim(claim_text: str) -> bool:
    """Heuristic identifier for laterality-turning claims."""
    lc = claim_text.lower()
    return any(tok in lc for tok in (" left ", " right ", "bilateral", "left-", "right-"))


def _loaded_dataset(
    val_jsonl: Path,
    image_root: Path,
    tokenizer: Any,
    cfg: Any,
) -> Any:
    from v5.train import GroundBenchDataset

    return GroundBenchDataset(val_jsonl, image_root, tokenizer, cfg)


def run_diagnostic(
    *,
    model_ckpt: Path,
    val_jsonl: Path,
    image_root: Path,
    batch_size: int = 32,
    n_shuffle_seeds: int = 3,
    threshold_pp: float = 5.0,
    device: torch.device | str = "cuda",
) -> DiagnosticResult:
    """Compute IMG, ESG, IPG on a trained verifier.

    Args:
        model_ckpt: Path to a saved model state dict produced by ``train_v5``.
        val_jsonl: Path to the evaluation JSONL (typically the ``val`` split).
        image_root: Root directory for image file resolution.
        batch_size: Batch size for inference.
        n_shuffle_seeds: Number of independent derangement seeds for ESG.
        threshold_pp: Percentage-point threshold below which a verifier is
            classified evidence-blind on either IMG or ESG.
        device: Inference device.

    Returns:
        DiagnosticResult summarizing the three metrics and the binary verdict.
    """
    from v5.model import V5Config, build_v5_tokenizer, build_v5_model
    from v5.train import V5TrainConfig

    device = torch.device(device) if not isinstance(device, torch.device) else device

    model_cfg = V5Config()
    tokenizer = build_v5_tokenizer(model_cfg)
    model = build_v5_model(model_cfg).to(device)
    state = torch.load(model_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)

    train_cfg = V5TrainConfig(
        train_jsonl=val_jsonl,
        val_jsonl=val_jsonl,
        out_dir=Path("/tmp/eb_diag"),
        image_root=image_root,
    )
    ds = _loaded_dataset(val_jsonl, image_root, tokenizer, train_cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Standard pass
    preds_full, labels, cats = _run_forward_all(model, loader, device)
    acc_full = _accuracy(preds_full, labels)

    # Image-masking gap
    preds_zero, _, _ = _run_forward_all(model, loader, device, zero_image=True)
    acc_zero = _accuracy(preds_zero, labels)

    # Evidence-shuffling gap: average over seeds
    shuf_accs: list[float] = []
    for seed in range(n_shuffle_seeds):
        preds_shuf, _, _ = _run_forward_all(
            model, loader, device, shuffle_evidence=True, shuffle_seed=17 + seed
        )
        shuf_accs.append(_accuracy(preds_shuf, labels))
    acc_shuf = float(sum(shuf_accs) / len(shuf_accs)) if shuf_accs else acc_full

    # Image-perturbation gap on laterality subset
    lat_indices: list[int] = []
    for i, row in enumerate(ds.rows):
        if _is_laterality_claim(row.get("claim_text", "")):
            lat_indices.append(i)

    if lat_indices:
        lat_subset = torch.utils.data.Subset(ds, lat_indices)
        lat_loader = DataLoader(lat_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        preds_lat, labels_lat, _ = _run_forward_all(model, lat_loader, device)
        preds_flip, _, _ = _run_forward_all(model, lat_loader, device, flip_image=True)
        acc_lat = _accuracy(preds_lat, labels_lat)
        acc_flip = _accuracy(preds_flip, labels_lat)
        ipg = (acc_lat - acc_flip) * 100.0
    else:
        acc_flip = None
        ipg = None

    img_gap = (acc_full - acc_zero) * 100.0
    esg_gap = (acc_full - acc_shuf) * 100.0

    # Per-category breakdown (simple approach: re-compute gaps per category)
    per_category: dict[str, dict[str, float]] = {}
    # Build category index
    cat_index: dict[str, list[int]] = {}
    for i, row in enumerate(ds.rows):
        c = row.get("claim_struct", {}).get("finding_family") or row.get("source_site", "unknown")
        cat_index.setdefault(str(c), []).append(i)
    for cat, idxs in cat_index.items():
        if len(idxs) < 10:
            continue  # skip tiny categories
        labels_c = labels[idxs]
        preds_full_c = preds_full[idxs]
        preds_zero_c = preds_zero[idxs]
        acc_full_c = _accuracy(preds_full_c, labels_c)
        acc_zero_c = _accuracy(preds_zero_c, labels_c)
        per_category[cat] = {
            "n": len(idxs),
            "acc_full": acc_full_c,
            "acc_image_zeroed": acc_zero_c,
            "img_gap_pp": (acc_full_c - acc_zero_c) * 100.0,
        }

    return DiagnosticResult(
        n_test=len(ds),
        acc_full=acc_full,
        acc_image_zeroed=acc_zero,
        acc_evidence_shuffled=acc_shuf,
        acc_laterality_flipped=acc_flip,
        img_gap_pp=img_gap,
        esg_gap_pp=esg_gap,
        ipg_gap_pp=ipg,
        per_category=per_category,
        evidence_blind=(img_gap < threshold_pp) or (esg_gap < threshold_pp),
        threshold_pp=threshold_pp,
    )


def diagnostic_to_json(result: DiagnosticResult, out_path: Path) -> None:
    """Write a DiagnosticResult to JSON for downstream table generation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2))


def _run_forward_with_probs(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    *,
    zero_image: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like ``_run_forward_all`` but also returns full softmax probabilities.

    Returns:
        preds: (N,) argmax class
        probs: (N, num_classes) softmax probabilities
        labels: (N,) gold labels
    """
    model.eval()
    preds_all: list[torch.Tensor] = []
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if zero_image:
                pv = torch.zeros_like(pv)

            out = model(pv, ii, am)
            logits = out["verdict_logits"]
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            preds_all.append(preds.cpu())
            probs_all.append(probs.cpu())
            labels_all.append(labels.cpu())
    return torch.cat(preds_all), torch.cat(probs_all), torch.cat(labels_all)


def run_img_correct_diagnostic(
    *,
    model_ckpt: Path,
    val_jsonl: Path,
    image_root: Path,
    batch_size: int = 32,
    margin_tau: float = 0.1,
    device: torch.device | str = "cuda",
):
    """Compute IMG_correct on a v5 verifier checkpoint.

    Calls the verifier-side diagnostic from ``pivot_ab.img_correct``: counts
    only (correct_full ∧ incorrect_masked ∧ Δp_correct ≥ τ) cases as
    image-grounded, subtracting induced inversion. Returns the
    ``IMGCorrectResult`` dataclass.

    Args:
        model_ckpt: Path to a saved model state dict produced by ``train_v5``.
        val_jsonl: Path to the evaluation JSONL.
        image_root: Root directory for image file resolution.
        batch_size: Inference batch size.
        margin_tau: Min drop in correct-class probability to count as
            image-grounded transition (default 0.1).
        device: Inference device.
    """
    from v5.model import V5Config, build_v5_tokenizer, build_v5_model
    from v5.train import V5TrainConfig
    from pivot_ab.img_correct import img_correct_verifier

    device = torch.device(device) if not isinstance(device, torch.device) else device

    model_cfg = V5Config()
    tokenizer = build_v5_tokenizer(model_cfg)
    model = build_v5_model(model_cfg).to(device)
    state = torch.load(model_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)

    train_cfg = V5TrainConfig(
        train_jsonl=val_jsonl,
        val_jsonl=val_jsonl,
        out_dir=Path("/tmp/eb_imgcorrect"),
        image_root=image_root,
    )
    ds = _loaded_dataset(val_jsonl, image_root, tokenizer, train_cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    preds_full, probs_full, labels = _run_forward_with_probs(model, loader, device)
    preds_masked, probs_masked, _ = _run_forward_with_probs(model, loader, device, zero_image=True)

    # Probability of the TRUE class under each input variant
    prob_full_correct = probs_full.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    prob_masked_correct = probs_masked.gather(1, labels.unsqueeze(-1)).squeeze(-1)

    return img_correct_verifier(
        pred_full=preds_full,
        pred_masked=preds_masked,
        prob_full_correct=prob_full_correct,
        prob_masked_correct=prob_masked_correct,
        targets=labels,
        margin_tau=margin_tau,
    )
