"""Regenerate Figure 2 (baseline landscape) for the AI4Science workshop paper.

Uses the verified numbers from Table 2 of paper.tex:
  BiomedCLIP (zero-shot) : acc 0.538, IMG 8.0,  ESG 0.0
  MedGemma-4B-IT         : acc 0.758, IMG 14.0, ESG 2.0
  MAIRA-2 (as verifier)  : acc 0.498, IMG 32.2, ESG 6.6     (fixed this session)
  Ours (v6.0-retrain)    : acc 0.911, IMG 70.21, ESG 20.01

Drops Claude 3.5 Sonnet (infrastructure-failed, 0/500 calls) and RadFlag
(different detector class; IMG/ESG do not apply, per paper footnote).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

entries = [
    ("BiomedCLIP",           8.0,   0.0),
    ("MedGemma-4B-IT",       14.0,  2.0),
    ("MAIRA-2",              32.2,  6.6),
    ("ClaimGuard-CXR (ours)", 70.21, 20.01),
]

names = [e[0] for e in entries]
imgs  = [e[1] for e in entries]
esgs  = [e[2] for e in entries]
x = np.arange(len(names))
w = 0.36

fig, ax = plt.subplots(figsize=(6.5, 3.3))
ax.bar(x - w / 2, imgs, width=w, color="#DD8452", label="IMG (pp)")
ax.bar(x + w / 2, esgs, width=w, color="#55A868", label="ESG (pp)")
ax.axhline(5, color="#888", linestyle=":", linewidth=0.8,
           label="Evidence-blind threshold (5pp)")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("Gap (pp)")
ax.set_title("Baseline landscape on real-RRG claims")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "F6_baseline_landscape.pdf")
fig.savefig(OUT / "F6_baseline_landscape.png", dpi=200)
plt.close(fig)
print(f"wrote {OUT / 'F6_baseline_landscape.pdf'}")
