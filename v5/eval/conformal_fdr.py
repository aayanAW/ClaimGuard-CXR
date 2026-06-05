"""Conformal FDR evaluation for v5 checkpoints.

Implements inverted conformal Benjamini-Hochberg (calibrated against the
CONTRADICTED class) on the support-score output of a trained v5 verifier.
The inversion is the same procedure used in the v3/v4 results and described
in `ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md` §8.3. For comparison we also
report StratCP per-pathology thresholds and forward cfBH (which collapses
on small calibration sets).

Usage from Modal:

    from v5.eval.conformal_fdr import run_conformal
    report = run_conformal(
        model_ckpt="/data/checkpoints/claimguard_v5/v5_3_contrast/best.pt",
        cal_jsonl="/data/groundbench_v5/all/groundbench_v5_cal.jsonl",
        test_jsonl="/data/groundbench_v5/all/groundbench_v5_test.jsonl",
        image_root="/data",
        alphas=(0.05, 0.10, 0.15, 0.20),
        out_path="/data/checkpoints/claimguard_v5/v5_3_contrast/conformal.json",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inverted cfBH
# ---------------------------------------------------------------------------


def _inverted_cfbh_pvalues(
    contradicted_cal_scores: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    """Conformal p-values using calibration on the CONTRADICTED class.

    A low p-value means the test claim's support score is much higher than
    typical CONTRADICTED calibration scores — i.e., the claim looks strongly
    supported.
    """
    n_cal = len(contradicted_cal_scores)
    # For each test score s, p = (1 + |{c : s_c >= s}|) / (1 + n_cal)
    sorted_cal = np.sort(contradicted_cal_scores)
    # count of cal scores >= each test score via searchsorted
    ge_counts = n_cal - np.searchsorted(sorted_cal, test_scores, side="left")
    p = (1 + ge_counts) / (1 + n_cal)
    return p


def _bh_select(p: np.ndarray, alpha: float) -> np.ndarray:
    """Standard BH at level alpha. Returns a boolean mask of accepted tests."""
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = p_sorted <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    k = int(np.max(np.where(below)[0]))
    cutoff = p_sorted[k]
    accepted = p <= cutoff
    return accepted


# ---------------------------------------------------------------------------
# StratCP comparison (pathology-stratified)
# ---------------------------------------------------------------------------


def _stratcp_per_stratum(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    cal_strata: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_strata: np.ndarray,
    alpha: float,
    min_stratum_size: int = 20,
) -> dict:
    """Pathology-stratified miscoverage thresholds a la StratCP.

    Returns a dict with per-stratum FDR / n_rejected / power, and a global roll-up.
    """
    unique_strata = sorted(set(cal_strata.tolist()))
    per_stratum: dict[str, dict] = {}
    mask = np.zeros(len(test_scores), dtype=bool)
    for s in unique_strata:
        cal_mask = cal_strata == s
        test_mask = test_strata == s
        if cal_mask.sum() < min_stratum_size or test_mask.sum() == 0:
            continue
        cal_s = cal_scores[cal_mask]
        # Threshold at the alpha-quantile of calibration scores
        thresh = np.quantile(cal_s, 1 - alpha)
        stratum_mask = (test_scores >= thresh) & test_mask
        mask |= stratum_mask
        n_rej = int(stratum_mask.sum())
        if n_rej > 0:
            fp = int(((test_labels == 0) & stratum_mask).sum())  # false rejections (CONTRADICTED among rejected)
            tp = int(((test_labels == 1) & stratum_mask).sum())
            fdr = fp / n_rej
            contradicted = int((test_labels[test_mask] == 1).sum())
            power = tp / max(1, contradicted)
        else:
            fdr = 0.0
            power = 0.0
        per_stratum[str(s)] = {
            "n_cal": int(cal_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_rejected": n_rej,
            "fdr": float(fdr),
            "power": float(power),
        }
    # Global roll-up
    n_rej = int(mask.sum())
    if n_rej > 0:
        fp = int(((test_labels == 0) & mask).sum())
        tp = int(((test_labels == 1) & mask).sum())
        global_fdr = fp / n_rej
        global_power = tp / max(1, int((test_labels == 1).sum()))
    else:
        global_fdr = 0.0
        global_power = 0.0
    return {"per_stratum": per_stratum, "global": {"fdr": global_fdr, "power": global_power, "n_rejected": n_rej}}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ConformalReport:
    n_cal: int
    n_test: int
    n_cal_contradicted: int
    alphas: list[float]
    inverted_cfbh: dict
    stratcp: dict
    forward_cfbh: dict


def _compute_support_scores(
    model_ckpt: Path,
    jsonl_path: Path,
    image_root: Path,
    *,
    batch_size: int = 32,
    device: torch.device | str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the trained v5 model on a JSONL and return (support_scores, labels, strata)."""
    from v5.model import V5Config, build_v5_tokenizer, build_v5_model
    from v5.train import GroundBenchDataset, V5TrainConfig

    device = torch.device(device) if not isinstance(device, torch.device) else device
    cfg = V5Config()
    tokenizer = build_v5_tokenizer(cfg)
    model = build_v5_model(cfg).to(device)
    state = torch.load(model_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model.eval()

    tcfg = V5TrainConfig(
        train_jsonl=jsonl_path,
        val_jsonl=jsonl_path,
        out_dir=Path("/tmp/conformal_eval"),
        image_root=image_root,
    )
    ds = GroundBenchDataset(jsonl_path, image_root, tokenizer, tcfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    scores: list[float] = []
    labels: list[int] = []
    strata: list[str] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            out = model(pv, ii, am)
            s = out["support_score"].detach().cpu().numpy()
            scores.extend(s.tolist())
            labels.extend(batch["labels"].tolist())
            idx_base = i * batch_size
            for j in range(ii.size(0)):
                row_idx = idx_base + j
                if row_idx < len(ds.rows):
                    row = ds.rows[row_idx]
                    stratum = row.get("claim_struct", {}).get("finding_family") or row.get("source_site", "unknown")
                    strata.append(str(stratum))
                else:
                    strata.append("unknown")
    return (np.asarray(scores, dtype="float64"),
            np.asarray(labels, dtype="int64"),
            np.asarray(strata, dtype="object"))


def run_conformal(
    *,
    model_ckpt: Path | str,
    cal_jsonl: Path | str,
    test_jsonl: Path | str,
    image_root: Path | str,
    alphas: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    out_path: Path | str | None = None,
    device: torch.device | str = "cuda",
) -> ConformalReport:
    """Execute the conformal FDR comparison on a trained v5 checkpoint.

    Returns a ConformalReport including inverted cfBH, StratCP, and forward cfBH
    results across the requested alpha levels.
    """
    model_ckpt = Path(model_ckpt)
    cal_jsonl = Path(cal_jsonl)
    test_jsonl = Path(test_jsonl)
    image_root = Path(image_root)

    logger.info("scoring calibration split %s", cal_jsonl)
    cal_s, cal_y, cal_strata = _compute_support_scores(
        model_ckpt, cal_jsonl, image_root, device=device
    )
    logger.info("scoring test split %s", test_jsonl)
    test_s, test_y, test_strata = _compute_support_scores(
        model_ckpt, test_jsonl, image_root, device=device
    )

    # Labels: SUPPORTED=0, CONTRADICTED=1. In conformal selection we
    # want to ACCEPT SUPPORTED claims. FDR = fraction of accepted that are
    # actually CONTRADICTED.
    contradicted_mask = cal_y == 1
    contradicted_cal_scores = cal_s[contradicted_mask]

    inverted = {}
    p_inv = _inverted_cfbh_pvalues(contradicted_cal_scores, test_s)
    for a in alphas:
        accepted = _bh_select(p_inv, a)
        n_green = int(accepted.sum())
        if n_green > 0:
            fp = int(((test_y == 1) & accepted).sum())
            fdr = fp / n_green
        else:
            fdr = 0.0
        n_supp = int((test_y == 0).sum())
        power = int(((test_y == 0) & accepted).sum()) / max(1, n_supp)
        inverted[f"alpha_{a:g}"] = {
            "n_green": n_green,
            "fdr": float(fdr),
            "power": float(power),
            "target_alpha": a,
            "controlled": fdr <= a,
        }

    # StratCP
    strat = {}
    for a in alphas:
        strat[f"alpha_{a:g}"] = _stratcp_per_stratum(
            cal_s, cal_y, cal_strata, test_s, test_y, test_strata, alpha=a
        )

    # Forward cfBH: calibrate on SUPPORTED class instead.
    supported_cal = cal_s[cal_y == 0]
    forward = {}
    for a in alphas:
        # Forward uses low-score calibration; we threshold by the (alpha)-quantile
        # of SUPPORTED cal scores from below. Test scores below that are accepted.
        if len(supported_cal) == 0:
            forward[f"alpha_{a:g}"] = {"n_green": 0, "fdr": 0.0, "power": 0.0, "note": "no supported cal"}
            continue
        thresh = np.quantile(supported_cal, a)
        accepted = test_s <= thresh
        n_green = int(accepted.sum())
        if n_green > 0:
            fp = int(((test_y == 1) & accepted).sum())
            fdr = fp / n_green
        else:
            fdr = 0.0
        n_supp = int((test_y == 0).sum())
        power = int(((test_y == 0) & accepted).sum()) / max(1, n_supp)
        forward[f"alpha_{a:g}"] = {
            "n_green": n_green,
            "fdr": float(fdr),
            "power": float(power),
            "target_alpha": a,
        }

    report = ConformalReport(
        n_cal=len(cal_s),
        n_test=len(test_s),
        n_cal_contradicted=int(contradicted_mask.sum()),
        alphas=list(alphas),
        inverted_cfbh=inverted,
        stratcp=strat,
        forward_cfbh=forward,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import dataclasses
        out_path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str))

    return report
