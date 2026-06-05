"""Collect v5 training + diagnostic results into a single table.

Reads:
    /data/checkpoints/claimguard_v5/<config>/diagnostic.json
    /data/groundbench_v5/pipeline_status.json

Produces:
    /data/v5_results.json — canonical final artifact
    /data/v5_results.csv  — table-friendly view

Usage from a Modal function:

    from v5.eval.collect_results import collect_all
    collect_all(checkpoints_root="/data/checkpoints/claimguard_v5")
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIGS_IN_ORDER = [
    "v5_0_base",
    "v5_1_ground",
    "v5_2_real",
    "v5_3_contrast",
    "v5_4_final",
]


def collect_all(
    checkpoints_root: Path | str = "/data/checkpoints/claimguard_v5",
    status_path: Path | str = "/data/groundbench_v5/pipeline_status.json",
    out_json: Path | str = "/data/v5_results.json",
    out_csv: Path | str = "/data/v5_results.csv",
) -> dict:
    """Assemble the final results table.

    Walks each config directory for ``diagnostic.json``, parses the pipeline
    status for training stats, and writes a combined JSON + CSV.
    """
    checkpoints_root = Path(checkpoints_root)
    status_path = Path(status_path)
    out_json = Path(out_json)
    out_csv = Path(out_csv)

    status: dict = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except Exception as exc:
            logger.warning("could not parse %s: %s", status_path, exc)

    training_results = (status.get("step5_training") or {})
    rows: list[dict] = []

    for cfg in CONFIGS_IN_ORDER:
        cfg_dir = checkpoints_root / cfg
        diag_path = cfg_dir / "diagnostic.json"
        diag: dict = {}
        if diag_path.exists():
            try:
                diag = json.loads(diag_path.read_text())
            except Exception as exc:
                logger.warning("could not parse %s: %s", diag_path, exc)

        train_info = training_results.get(cfg, {})
        if not isinstance(train_info, dict):
            train_info = {}

        last_stats = {}
        if isinstance(train_info.get("stats"), list) and train_info["stats"]:
            last_stats = train_info["stats"][-1]

        row = {
            "config": cfg,
            "epochs": train_info.get("epochs"),
            "train_loss_final": last_stats.get("train_loss"),
            "val_acc_final": last_stats.get("val_acc"),
            "val_image_masked_gap_train": last_stats.get("image_masked_gap"),
            "test_n": diag.get("n_test"),
            "test_acc_full": diag.get("acc_full"),
            "test_acc_image_zeroed": diag.get("acc_image_zeroed"),
            "test_acc_evidence_shuffled": diag.get("acc_evidence_shuffled"),
            "test_acc_laterality_flipped": diag.get("acc_laterality_flipped"),
            "img_gap_pp": diag.get("img_gap_pp"),
            "esg_gap_pp": diag.get("esg_gap_pp"),
            "ipg_gap_pp": diag.get("ipg_gap_pp"),
            "evidence_blind": diag.get("evidence_blind"),
            "training_error": train_info.get("error"),
            "diagnostic_error": diag.get("error") if isinstance(diag, dict) else None,
        }
        rows.append(row)

    artifact = {
        "configs_evaluated": [r["config"] for r in rows],
        "rows": rows,
        "source_status_path": str(status_path),
        "source_checkpoints_root": str(checkpoints_root),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(artifact, indent=2, default=str))

    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in r.items()})

    return artifact


def format_summary_table(artifact: dict) -> str:
    """Produce a compact text table for console display."""
    lines = [
        "config              | epochs | val_acc | test_acc | IMG   | ESG   | IPG   | blind?",
        "-" * 90,
    ]
    for r in artifact.get("rows", []):
        def _fmt(v, fmt):
            if v is None:
                return "   -   "
            if isinstance(v, bool):
                return " yes " if v else "  no "
            try:
                return fmt.format(v)
            except (TypeError, ValueError):
                return f"{v}"
        lines.append(
            f"{r['config']:<20}| {_fmt(r['epochs'],'{:>4d}  ')} | "
            f"{_fmt(r['val_acc_final'],'{:>5.3f} ')} | "
            f"{_fmt(r['test_acc_full'],'{:>5.3f}  ')} | "
            f"{_fmt(r['img_gap_pp'],'{:>5.2f} ')} | "
            f"{_fmt(r['esg_gap_pp'],'{:>5.2f} ')} | "
            f"{_fmt(r['ipg_gap_pp'],'{:>5.2f} ')} | "
            f"{_fmt(r['evidence_blind'],'{}')}"
        )
    return "\n".join(lines)
