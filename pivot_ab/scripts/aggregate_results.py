#!/usr/bin/env python3
"""Aggregate Phase 5 (IMG_correct on Pivot A+B fine-tunes) and Phase 8
(IMG_correct retroactive on prior v5/v6 verifiers) results into Markdown
tables for the manuscript Results section.

Run after Phase 4-5-8 complete:
    cd /Users/aayanalwani/VeriFact/verifact
    python pivot_ab/scripts/aggregate_results.py

Reads from local copies of the Modal volume result JSONLs (download with
`modal volume get` first):
    /tmp/v5_results.jsonl
    /tmp/v6_results.jsonl
    /tmp/v5_ablations.jsonl
    /tmp/img_correct_v2_*.jsonl  (Phase 5 per-config Pivot A+B results)

Writes:
    /tmp/aggregated_results.md  (paste-ready Markdown for manuscript)
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skip malformed row in {path.name}: {e}")
    return rows


def _format_pp(x: float) -> str:
    return f"{x:.2f}"


def _seed_aggregate(rows: list[dict], key: str) -> tuple[float, float]:
    """Return (mean, std) of the named field over `rows`."""
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def aggregate_phase5_results():
    """Phase 5: Pivot A+B fine-tunes evaluated under IMG_correct.
    Expected files (one per config × seed): /tmp/img_correct_v2_<config>[_seed<S>].jsonl
    """
    print("\n## §5.1 Phase 5 — IMG_correct on Pivot A+B fine-tunes\n")
    configs = ["sft_only", "sft_faith", "sft_dual", "sft_full", "sft_full_sfc"]
    seeds = [42, 1337, 9001]

    rows_per_config: dict[str, list[dict]] = {c: [] for c in configs}
    for cfg in configs:
        for seed in seeds:
            if seed == 42:
                p = Path(f"/tmp/img_correct_v2_{cfg}.jsonl")
            else:
                p = Path(f"/tmp/img_correct_v2_{cfg}_seed{seed}.jsonl")
            rows = _read_jsonl(p)
            if rows:
                rows_per_config[cfg].extend(rows)

    print("| Config | n seeds | IMG_correct (pp) mean±std | IMG (pp) | Induced inv. (pp) | Acc full (%) |")
    print("|---|---:|---:|---:|---:|---:|")
    for cfg in configs:
        rows = rows_per_config[cfg]
        if not rows:
            print(f"| {cfg} | 0 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |")
            continue
        ic_m, ic_s = _seed_aggregate(rows, "img_correct_pp")
        i_m, _ = _seed_aggregate(rows, "img_pp")
        ii_m, _ = _seed_aggregate(rows, "induced_inversion_pp")
        a_m, _ = _seed_aggregate(rows, "accuracy_full")
        bold = "**" if cfg == "sft_full" else ""
        print(f"| {bold}{cfg}{bold} | {len(rows)} | {bold}{_format_pp(ic_m)} ± {_format_pp(ic_s)}{bold} | {_format_pp(i_m)} | {_format_pp(ii_m)} | {_format_pp(a_m)} |")


def aggregate_phase8_results():
    """Phase 8: retroactive IMG_correct on prior v5 + v6 verifier checkpoints."""
    print("\n## §5.4 Phase 8 — Retroactive IMG_correct on prior verifiers\n")

    main_v5 = ["v5_0_base", "v5_1_ground", "v5_2_real", "v5_3_contrast", "v5_4_final"]
    main_v6 = ["v6_0_3site", "v6_0_loo_no_openi", "v6_0_loo_no_padchest", "v6_0_loo_no_chestx"]

    v5 = _read_jsonl(Path("/tmp/v5_results.jsonl"))
    v6 = _read_jsonl(Path("/tmp/v6_results.jsonl"))
    v5_abl = _read_jsonl(Path("/tmp/v5_ablations.jsonl"))

    by_config = {r["config"]: r for r in v5 + v6 + v5_abl if "config" in r}

    print("**Headline configs:**\n")
    print("| Config | n_examples | IMG_correct (pp) | IMG (pp) | Induced inv. (pp) | Inv./IMG | Acc full (%) | Acc masked (%) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cfg in main_v5 + main_v6:
        r = by_config.get(cfg)
        if not r:
            print(f"| {cfg} | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |")
            continue
        ic = r["img_correct_pp"]
        i = r["img_pp"]
        ii = r["induced_inversion_pp"]
        af = r["accuracy_full"]
        am = r["accuracy_masked"]
        ratio = f"{ii/i*100:.1f}%" if i > 0 else "N/A"
        bold = "**" if cfg == "v5_3_contrast" else ""
        print(f"| {bold}{cfg}{bold} | {r['n_examples']} | {_format_pp(ic)} | {_format_pp(i)} | {_format_pp(ii)} | {ratio} | {_format_pp(af)} | {_format_pp(am)} |")

    print("\n**v5 ablations:**\n")
    if v5_abl:
        print("| Config | IMG_correct (pp) | IMG (pp) | Induced inv. (pp) | Acc full (%) |")
        print("|---|---:|---:|---:|---:|")
        for r in v5_abl:
            print(f"| {r['config']} | {_format_pp(r['img_correct_pp'])} | {_format_pp(r['img_pp'])} | {_format_pp(r['induced_inversion_pp'])} | {_format_pp(r['accuracy_full'])} |")
    else:
        print("[PENDING — ablation evals still running]")


def headline_summary():
    """Print a 1-paragraph headline summary suitable for the manuscript abstract."""
    print("\n## Headline summary for §1 / abstract\n")

    v5 = _read_jsonl(Path("/tmp/v5_results.jsonl"))
    v5_3 = next((r for r in v5 if r.get("config") == "v5_3_contrast"), None)

    if v5_3:
        ic = v5_3["img_correct_pp"]
        i = v5_3["img_pp"]
        ii = v5_3["induced_inversion_pp"]
        ratio_str = f"{ii/i*100:.1f}%" if i > 0 else "N/A"
        print(f"- v5_3_contrast retroactive audit: IMG = {_format_pp(i)}pp, IMG_correct = {_format_pp(ic)}pp, induced inversion = {_format_pp(ii)}pp ({ratio_str} of IMG).")
    else:
        print("- v5_3_contrast retroactive audit: [PENDING].")

    print(f"- Phase 5 results: [PENDING; see §5 table once Phase 4 fine-tunes complete and Phase 5 IMG_correct evals run].")


if __name__ == "__main__":
    print(f"# Pivot A+B aggregated results")
    print(f"\n_Generated by `pivot_ab/scripts/aggregate_results.py` from local /tmp/*.jsonl files._\n")
    aggregate_phase5_results()
    aggregate_phase8_results()
    headline_summary()
