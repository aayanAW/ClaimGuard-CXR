# ClaimGuard-CXR

**Claim-level hallucination detection and calibrated triage for radiology report generation, with conformal false-discovery-rate control.**

[![Status](https://img.shields.io/badge/status-active%20research-orange)](#project-status)
[![Python](https://img.shields.io/badge/python-3.13-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Data](https://img.shields.io/badge/data-public%20only-brightgreen)](#data-policy)

---

## Overview

Vision–language models that draft radiology reports hallucinate at rates of roughly
8–15%: they fabricate findings, omit critical observations, flip laterality, or
contradict the image. A single unfaithful claim such as _"no pneumothorax"_ can
delay treatment.

**ClaimGuard-CXR** attacks this from two complementary directions:

1. **Generator-side faithfulness (current focus, "Pivot A+B").** A LoRA fine-tune
   of a radiology VLM backbone (CheXagent-2-3B) trained with a composite objective
   that combines supervised fine-tuning with a causal-faithfulness term and a
   **dual-adversarial hallucination/omission filter** over both text and image. An
   inline **gradient-norm monitor** predicts post-training _image-blindness_ (when
   the model stops attending to pixels), and an `IMG_correct` methodological patch
   retroactively audits how much of a measured grounding gain is faithful grounding
   versus induced inversion.

2. **Claim-level triage with formal error control.** A conformal **cfBH**
   (Benjamini–Hochberg-style) procedure assigns per-claim triage labels —
   _accept / review / reject_ — with **pathology-stratified false-discovery-rate
   control** and exchangeability diagnostics (one-claim-per-patient subsampling,
   intra-patient ICC). This component carries over from the project's earlier
   verifier-centric design and lives in the evaluation harness.

The system is a **drafting-and-triage aid for human review**, not an autonomous
reporting system. The error notion it controls is the false discovery rate among
claims labeled high-confidence, under the standard calibration/test exchangeability
assumption.

## Project status

> **Active, unpublished research.** Target venues: NeurIPS / ICLR / ICML main track
> (quality-driven, not deadline-driven). Numbers in the manuscripts and `results/`
> are work-in-progress and are framed honestly, including negative results
> (e.g. honest 0-match PadChest-GR validation, hypothesis-only shortcut analysis).
> Do not treat any metric here as a final published claim.

The repository carries layered design history. The **authoritative** current design
lives in the Pivot A+B documents and `ARCHITECTURE.md`; earlier `ARCHITECTURE_V*`
specs and `CLAIMGUARD_PROPOSAL.md` are retained as historical record and are
explicitly marked superseded where relevant.

## Data policy

**Public data only.** Training and evaluation are restricted to openly available
chest-radiograph corpora — **OpenI / IU X-ray, ChestX-Det10, PadChest-GR, ReXVal**
— with **no PhysioNet-credentialed datasets** (e.g. MIMIC-CXR, CheXpert Plus images
are not redistributed here). This keeps the pipeline reproducible without
credentialing barriers. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full
trust model and provenance gate.

## Repository structure

| Path                                         | Contents                                                      |
| -------------------------------------------- | ------------------------------------------------------------- |
| `pivot_ab/`                                  | Current generator-side causal-faithfulness method (Pivot A+B) |
| `models/`                                    | Model definitions and LoRA / fine-tuning code                 |
| `data/`                                      | Dataset loaders, label tooling, masks (no raw images tracked) |
| `evaluation/`                                | Conformal cfBH triage, FDR control, baselines, fairness/DCA   |
| `inference/`                                 | Inference and verification entrypoints                        |
| `configs/`                                   | Experiment configuration files                                |
| `scripts/`                                   | Data regeneration and orchestration utilities                 |
| `tests/`                                     | Test suite / gate scripts                                     |
| `v5/`, `v5_final_results/`                   | v5 image-grounded pipeline and frozen results                 |
| `claimguard_nmi/`, `ai4science/`             | Manuscript / submission builds                                |
| `paper/`, `mlhc_build/`, `mock_paper_build/` | LaTeX paper builds                                            |
| `figures/`                                   | Generated figures and plots                                   |
| `results/`                                   | Experiment outputs                                            |
| `demo/`                                      | Demonstration assets                                          |
| `website/`                                   | Project landing page + Hugging Face Space scaffold            |
| `runpod/`                                    | Cloud GPU runner configs                                      |

Key documents: [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) · [`decisions.md`](decisions.md) ·
manuscripts (`MANUSCRIPT_*.md`).

## Setup

```bash
git clone https://github.com/aayanAW/ClaimGuard-CXR.git
cd ClaimGuard-CXR

# Python 3.13 (miniforge recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU experiments run on [Modal](https://modal.com) (cloud GPU); CPU-only baselines
and evaluation run locally. See `configs/` and `scripts/` for entrypoints.

## Reproducibility

Every paper result is reproducible from public data via the steps in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Experiment runs are tracked by
git SHA + config hash + artifact URI.

## Citation

A formal citation will be added on publication. For now:

```bibtex
@misc{alwani2026claimguardcxr,
  author = {Alwani, Aayan},
  title  = {ClaimGuard-CXR: Claim-Level Hallucination Detection and
            Conformal FDR Control for Radiology Report Generation},
  year   = {2026},
  howpublished = {\url{https://github.com/aayanAW/ClaimGuard-CXR}}
}
```

## License

Released under the [MIT License](LICENSE).
