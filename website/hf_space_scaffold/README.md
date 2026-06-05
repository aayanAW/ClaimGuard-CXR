---
title: ClaimGuard-CXR
emoji: 🩻
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: true
license: mit
short_description: FDR-controlled hallucination detection for CXR reports
hardware: zero-a10g
suggested_hardware: zero-a10g
tags:
  - medical
  - radiology
  - chest-x-ray
  - hallucination-detection
  - conformal-prediction
  - fdr-control
  - roberta
---

# ClaimGuard-CXR

**Claim-level hallucination detection for AI-generated chest X-ray radiology reports, with provable false-discovery-rate control.**

This Space runs the live verifier behind the [ClaimGuard-CXR marketing site](https://alwaniaayan6-png.github.io/ClaimGuard-CXR). Paste a radiology report, drag the FDR target slider, and watch each atomic claim get flagged GREEN (verified) / YELLOW (review) / RED (likely hallucinated) in real time — with a formal mathematical guarantee on the error rate among GREEN claims.

## How it works

1. **Decompose** — rule-based extractor splits the report into sentence-level atomic claims
2. **Verify** — RoBERTa-large binary cross-encoder scores each claim against retrieved evidence
3. **Calibrate** — inverted conformal Benjamini-Hochberg procedure computes a formal p-value
4. **Triage** — BH at level α decides which claims are flagged GREEN; FDR among GREEN ≤ α

## Key results (CheXpert Plus)

| Metric | Value |
|---|---|
| Accuracy | **98.31%** |
| Macro F1 | 98.08% |
| Contradiction recall | 96.38% |
| AUROC | 99.52% |
| FDR @ α=0.05 | **1.30%** |
| Power @ α=0.05 | **98.06%** |
| Margin over best baseline (CheXagent-8b VLM) | **+31 pp** |

## Cross-dataset transfer

Trained on CheXpert Plus (Stanford), evaluated zero-shot on OpenI (Indiana University). Accuracy drops ~13 pp under distribution shift but **FDR stays controlled at every α** without any retraining.

## Disclaimer

Research prototype. **Not for clinical use.** All example reports are synthetic.

## Author

Aayan Alwani — Laughney Lab, Weill Cornell Medicine. Targeting NeurIPS 2026.

- Paper site: https://alwaniaayan6-png.github.io/ClaimGuard-CXR
- GitHub: https://github.com/alwaniaayan6-png/ClaimGuard-CXR
