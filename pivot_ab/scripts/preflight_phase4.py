#!/usr/bin/env python3
"""Phase 4 pre-flight audit per CLAUDE.md.

Spawns Opus-4.7 reviewer agents that read the training code, configs, and
filter outputs and return a launch / no-launch verdict before the user
spends $200+ on the full sweep.

Run: python pivot_ab/scripts/preflight_phase4.py

Cost: ~$0.50 in Anthropic API tokens.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

PRE_FLIGHT_PROMPT = """\
You are an Opus-4.7 pre-flight reviewer for a Modal H100 fine-tuning job.

Project context: Pivot A+B for ClaimGuard-CXR — LoRA fine-tune of
CheXagent-2-3B with a bidirectional causal-faithfulness loss + dual adversarial
filter (text-only AND image-only). About to spawn 5 configs × 3 seeds = 15
runs at ~$15-25 each. Total budget at risk: ~$300.

Your task: read the inputs below and return a verdict in {{ready-to-launch,
launch-with-fixes, do-not-launch}} plus a list of issues. Look especially for:

1. Loss-function bugs that would either NaN or train induced inversion (the
   methodological flaw the project is specifically trying to avoid).
2. Off-by-one in token alignment between log_p and target_ids.
3. LoRA configuration that doesn't actually train anything (e.g., wrong
   target_modules for CheXagent-2 architecture).
4. Data-loader issues that would silently drop rows.
5. Filter weights JSONL not aligned with training row indices.
6. Backward hook in grad_monitor.py that would interfere with normal backprop.
7. Mixed-precision (bf16) issues with the faithfulness loss.

Files to read:
- {plan_path}
- {train_path}
- {loss_path}
- {monitor_path}
- {data_loader_path}
- {dual_filter_path}
- {config_full_path}

Output format:
{{
  "verdict": "ready-to-launch" | "launch-with-fixes" | "do-not-launch",
  "blocking_issues": [...],
  "non_blocking_warnings": [...],
  "summary": "..."
}}
"""


def main() -> None:
    inputs = {
        "plan_path": REPO_ROOT / "PIVOT_AB_FAITHFUL_GENERATOR_PLAN.md",
        "train_path": REPO_ROOT / "pivot_ab" / "train_chexagent.py",
        "loss_path": REPO_ROOT / "pivot_ab" / "faith_loss.py",
        "monitor_path": REPO_ROOT / "pivot_ab" / "grad_monitor.py",
        "data_loader_path": REPO_ROOT / "pivot_ab" / "data_loader.py",
        "dual_filter_path": REPO_ROOT / "pivot_ab" / "dual_filter.py",
        "config_full_path": REPO_ROOT / "pivot_ab" / "configs" / "sft_full.yaml",
    }
    for k, p in inputs.items():
        if not p.exists():
            print(f"[fatal] missing input: {k} = {p}")
            sys.exit(2)

    prompt = PRE_FLIGHT_PROMPT.format(**{k: str(v) for k, v in inputs.items()})

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "[error] ANTHROPIC_API_KEY not set. Either set it and rerun, or run\n"
            "the audit manually by sending the prompt below to any LLM with vision.\n\n"
            "----- BEGIN PROMPT -----"
        )
        print(prompt)
        print("----- END PROMPT -----")
        sys.exit(1)

    try:
        from anthropic import Anthropic
    except ImportError:
        print("[error] anthropic package not installed. Run: pip install 'anthropic>=0.60.0'")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    files_blob = []
    for k, p in inputs.items():
        try:
            content = p.read_text()
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
            continue
        files_blob.append(f"## {k} ({p.name})\n\n```{p.suffix.lstrip('.')}\n{content}\n```\n")

    full_prompt = prompt + "\n\n" + "\n".join(files_blob)

    print(f"[preflight] spawning Opus-4.7 reviewer over {len(inputs)} files...")
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        messages=[{"role": "user", "content": full_prompt}],
    )

    out = response.content[0].text if response.content else ""
    print(out)

    # Try to parse a verdict
    import re
    m = re.search(r'"verdict"\s*:\s*"([^"]+)"', out)
    verdict = m.group(1) if m else "unknown"
    print(f"\n[preflight] verdict: {verdict}")
    if verdict == "do-not-launch":
        sys.exit(3)
    elif verdict == "launch-with-fixes":
        sys.exit(2)
    elif verdict == "ready-to-launch":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
