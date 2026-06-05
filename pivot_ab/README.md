# Pivot A+B — Causal-Faithfulness Radiology Generator

LoRA fine-tune of **MedGemma-4B-IT** with a bidirectional causal-faithfulness
loss + dual adversarial filter (text-only AND image-only) + gradient-norm
training-time monitor + correctness-under-counterfactual diagnostic.

**Backbone note (2026-05-03):** the original plan targeted CheXagent-2-3B
(MIT-licensed) but its tokenizer-embedded image paradigm
(`tokenizer.from_list_format` rather than a `pixel_values` kwarg) makes
counterfactual fine-tuning awkward. We switched to MedGemma-4B-IT (Google
HAI-DEF research-only licence) which exposes the standard
`AutoModelForImageTextToText` forward signature with `pixel_values`. The
training-script filename `train_chexagent.py` is retained for backwards-compat
with import paths but the actual code targets MedGemma. See
`PIVOT_AB_EXECUTION_LOG.md` for the full backbone-selection rationale.

See:
- `../PIVOT_AB_FAITHFUL_GENERATOR_PLAN.md` — strategic plan with rationale,
  contributions, risks, mitigations
- `../PIVOT_AB_EXECUTION_LOG.md` — running execution journal

## Three contributions

1. **Dual-adversarial causal-faithfulness training** — a LoRA fine-tune that
   forces the model's output distribution under full input to diverge from its
   distribution under image-masked and image-flipped inputs by a margin, while
   preserving SFT correctness; samples solvable by either text alone OR image
   alone are downweighted by the dual filter.

2. **Gradient-norm + attention monitor** — a cheap training-time diagnostic
   (image-token gradient norm, cross-modal attention rank, image-attention
   attribution) that predicts post-training image-blindness. Implemented via
   forward + backward hooks; logged every K steps.

3. **IMG_correct diagnostic** — a methodological refinement of the standard
   IMG metric that distinguishes induced inversion (model is wrong on the
   same examples it was right on, just flipped) from faithful grounding loss
   (model is wrong on different examples). Addresses the methodological flaw
   in prior ClaimGuard work.

## Module layout

```
pivot_ab/
├── README.md                     # this file
├── __init__.py
├── image_only_filter.py          # vision-side adversarial filter
├── dual_filter.py                # combines text + image filter weights
├── faith_loss.py                 # bidirectional causal-faithfulness loss
├── grad_monitor.py               # gradient-norm + attention hooks
├── img_correct.py                # IMG_correct diagnostic
├── data_loader.py                # counterfactual-augmented data loader
├── train_chexagent.py            # LoRA fine-tuning entry point
├── modal_app.py                  # Modal orchestrator
├── configs/
│   ├── sft_only.yaml             # baseline (i)
│   ├── sft_faith.yaml            # SFT + L_faith (ii)
│   ├── sft_dual.yaml             # SFT + dual filter (iii)
│   ├── sft_full.yaml             # full method (iv)
│   └── sft_full_sfc.yaml         # full + Liu-2025 SFC ablation (v)
├── scripts/
│   ├── launch_phase0_smoke.sh    # ~$0.50, 6 minutes
│   ├── launch_phase1_filters.sh  # ~$15, 3 hours
│   ├── launch_phase4_full.sh     # ~$300, 36-50 hours total
│   └── preflight_phase4.py       # Opus-4.7 audit before Phase 4
└── tests/
    ├── test_dual_filter.py       # 5 tests
    ├── test_faith_loss.py        # 8 tests
    ├── test_img_correct.py       # 6 tests
    └── test_data_loader.py       # 5 tests
```

## Status

All code written. All 24 unit tests pass on CPU.

## Running

The full execution sequence:

```bash
# Phase 0 — smoke test
bash pivot_ab/scripts/launch_phase0_smoke.sh

# Phase 1 — filters (assumes the v5 text-only filter has already been run
# and its weights are at /data/pivot_ab/text_only_weights.jsonl)
bash pivot_ab/scripts/launch_phase1_filters.sh

# Pre-flight audit (mandatory per CLAUDE.md before > $1 Modal spend)
python pivot_ab/scripts/preflight_phase4.py

# Phase 4 — full sweep (requires Modal billing re-up to ≥ $5K)
bash pivot_ab/scripts/launch_phase4_full.sh

# Phase 5 — IMG_correct on each checkpoint (post 2026-05-05 data fix)
modal run pivot_ab/modal_app.py::eval_img_correct \\
    --checkpoint-dir /data/checkpoints/pivot_ab_v2/sft_full/checkpoints/final
```

## Compute budget

Within the user-stated $5–8K envelope. Phase-by-phase estimates in the
strategic plan (`PIVOT_AB_FAITHFUL_GENERATOR_PLAN.md` §3.1).

## Constraints

- All fine-tuning data restricted to non-credentialed public sources (OpenI,
  ChestX-Det10, PadChest-GR, IU X-ray, ReXVal). User does not have PhysioNet
  access — confirmed 2026-05-03.
- H100-only on Modal per the user's standing GPU policy.
- Pre-flight Opus-4.7 review mandatory before any Modal launch ≥ $1.
