"""One-liner spawner for the full v5 training ladder + diagnostic eval.

Invoked via:
    python3 v5/modal/spawn_full.py

Uses modal.Function.from_name().spawn() which returns instantly (<2s), so
laptop sleep or local client disconnect after spawn cannot interrupt the
pipeline. The orchestrator must already be deployed (via modal deploy).
"""

from __future__ import annotations

import modal


def spawn_full() -> str:
    configs = "v5_0_base,v5_1_ground,v5_2_real,v5_3_contrast,v5_4_final"
    fc = modal.Function.from_name(
        "claimguard-v5-orchestrator", "orchestrate"
    ).spawn(configs_csv=configs, smoke=False)
    return fc.object_id


if __name__ == "__main__":
    fc_id = spawn_full()
    print(f"Spawned full ladder: {fc_id}")
    print("Dashboard: https://modal.com/apps/alwaniaayan6/main/deployed/claimguard-v5-orchestrator")
