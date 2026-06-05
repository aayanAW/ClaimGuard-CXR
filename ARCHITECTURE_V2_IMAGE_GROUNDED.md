# ClaimGuard-CXR v2 (Path B): Image-Grounded Claim Verification

**Doc version:** 2.0 — Path B pivot
**Target venue:** Nature Machine Intelligence (primary), Nature Communications / npj Digital Medicine (fallback), Medical Image Analysis (floor)
**Constraints:** no PhysioNet credentialing, no recruited radiologists, public datasets only, laptop-sleep-durable execution
**Supersedes:** `ARCHITECTURE.md` v2.1 (text-only NeurIPS 2026 scope)
**Author of record:** Aayan Alwani (student); senior author Ashley Laughney (required before submission)

---

## 0. Why Path B

The v2.1 ARCHITECTURE.md honestly admits the vision branch is unused and evaluation is text-only over synthetic perturbations. For Nature-family publication, this is insufficient:

- Hypothesis-only baselines reach near-verifier accuracy on synthetic data → the model isn't using evidence.
- No radiologist-labeled ground truth → reviewers reject clinical claims.
- Cross-dataset FDR on a single OOD dataset (OpenI) is not cross-institutional generalization.

Path B's answer: anchor ground truth in **radiologist-drawn image annotations from public datasets** (MS-CXR phrase grounding, PadChest bounding boxes, RSNA Pneumonia boxes, CheXmask segmentation). The verifier must satisfy two agreement axes: (i) claim label agrees with evidence, (ii) the verifier's internal region attention agrees with radiologist annotations. Training uses both supervisions. Evaluation uses both.

This reframes the paper from "synthetic hallucination detector" to "**image-grounded claim verifier whose every support/contradict decision is checkable against a radiologist's drawn region.**"

---

## 1. Contributions (three clean claims)

1. **Image-grounded claim verifier** (model) — a multimodal architecture that fuses a CXR image (frozen BiomedCLIP / RAD-DINO-MAIRA-2 encoder) with claim + evidence text (RoBERTa-large) via cross-modal attention, producing (a) a verdict, (b) a calibrated score, and (c) a per-patch attention map auditable against radiologist annotations.

2. **Cross-site image-grounded benchmark** (data) — a public, radiologist-anchored evaluation suite combining MS-CXR, PadChest-localized, RSNA Pneumonia, CheXmask, CheXpert Plus, OpenI, and BRAX. ~50k (claim, image, evidence, radiologist-anchor) tuples across 5 institutions, released under inherited dataset licenses. All ground truth traces to a radiologist-drawn image annotation.

3. **Conformal FDR under institutional shift** (methods) — four conformal variants (inverted cfBH, weighted cfBH with density-ratio estimation, doubly-robust cfBH, StratCP baseline) evaluated across 5 sites with empirical coverage audit, effective-sample-size diagnostics, and a formal treatment of when each guarantee holds.

Optional bolt-on contribution:
4. **Provenance gate for AI-generated evidence** (engineering) — retained from v3; empirically validated on scaled dual-run experiment with 3 VLMs × 4 temperatures × 1000 images.

---

## 2. Scope — IN / OUT

**IN**
- Image-grounded verifier training on CheXpert Plus (existing local copy).
- Radiologist-anchored evaluation on MS-CXR, PadChest, RSNA, CheXmask, BRAX, OpenI, CheXpert Plus test.
- Conformal FDR with 4 variants + per-site empirical audit.
- Silver-standard LLM-ensemble labels on generated VLM reports for real-hallucination side-experiment (supplementary, not primary).
- Provenance gate (Task 9) scaled to 3 VLMs × 4 temperatures.
- Fairness / subgroup analysis where demographics are public.
- Baseline tuning for MAIRA-2, CheXagent, MedGemma, Llama-3.2-Vision, GPT-4o.
- Laptop-sleep-durable execution infrastructure (Modal detached + remote Claude cron).

**OUT**
- Any PhysioNet dataset (MIMIC-CXR, VinDr-CXR, ReXVal, ReXErr, RadGraph-XL).
- Any recruited-radiologist adjudication.
- Any prospective clinical deployment or reader study with live clinicians.
- Report generation (we consume VLM outputs but do not train a new generator).

**DEFERRED to future work**
- MIMIC-CXR evaluation (pending PhysioNet credentialing through Laughney; non-blocking for submission).
- Reader study with real radiologists (required only if pursuing Nature Medicine later).
- Prospective deployment at Weill Cornell.

---

## 3. Datasets (all non-credentialed, public)

| Dataset | Access | Role | Radiologist involvement | Has images? | Has reports? | Has image annotations? |
|---|---|---|---|---|---|---|
| **CheXpert Plus** | Stanford registration (have) | Training + test | 8 rads on test set; labels via CheXpert labeler elsewhere | ✓ | ✓ | Partial (CheXlocalize subset) |
| **MS-CXR** | HuggingFace public | **Grounding supervision + eval** | **Radiologist-verified phrase-image grounding** (1,162 pairs) | ✓ | ✗ (phrases only) | ✓ bboxes |
| **CheXmask** | CC-BY (public) | Anatomical consistency check | Radiologist-verified segmentation over existing public CXRs | — (masks over other datasets) | — | ✓ segmentations |
| **PadChest** | BIMCV email registration | Eval + grounding supervision | 27% radiologist-direct; localized subset has bboxes | ✓ | ✓ (Spanish; use English-translated subset) | ✓ subset bboxes |
| **RSNA Pneumonia Detection** | Kaggle (public) | Pneumonia-specific eval + grounding | Radiologist bounding boxes for pneumonia | ✓ | ✗ (labels only) | ✓ bboxes |
| **OpenI (Indiana)** | Direct download | Cross-institution eval | Radiologist-authored reports | ✓ | ✓ | ✗ |
| **BRAX** | IEEE DataPort reg | Cross-institution eval (Brazil) | Radiologist-authored reports | ✓ | ✓ | ✗ |
| **NIH ChestX-ray14** | Direct download | Supplementary training signal | Labels via NLP (weak) | ✓ | ✗ | ✗ (only 880-image subset) |
| **SIIM-ACR Pneumothorax** | Kaggle | Pneumothorax-specific eval | Radiologist segmentation | ✓ | ✗ | ✓ masks |

**Claim sources** (where claims come from for the test sets that don't have reports):
- CheXpert Plus, OpenI, PadChest, BRAX: extract claims from existing radiologist-authored reports via the validated LLM claim extractor (§10).
- MS-CXR: phrases are already claim-like; use directly.
- RSNA / SIIM / NIH: no reports — we generate claims from a tuned VLM (CheXagent-8b or MAIRA-2) as inputs, then use bounding boxes as image-anchor ground truth.

---

## 4. System overview

```
                              INPUTS
         ┌───────────────────┬───────────────────┬───────────────┐
         │     CXR image     │   claim text      │ evidence text │
         │ (any source, RGB  │ (atomic sentence  │ (retrieved    │
         │   224x224)        │   from report)    │  passage OR   │
         │                   │                   │   oracle)     │
         └────────┬──────────┴────────┬──────────┴───────┬───────┘
                  │                   │                  │
                  v                   v                  v
        ┌──────────────────┐  ┌───────────────────────────────┐
        │ BiomedCLIP ViT   │  │ RoBERTa-large tokenize+encode  │
        │  (frozen except  │  │   "[CLS] claim [SEP] evidence" │
        │   last 2 blocks) │  │                                │
        │                  │  │                                │
        │ → patch tokens   │  │ → CLS (1024)                   │
        │   (196 x 768)    │  │ → seq tokens                   │
        │ → image CLS (768)│  │                                │
        └────────┬─────────┘  └────────┬──────────────────────┘
                 │                     │
                 │    ┌────────────────┘
                 │    │
                 v    v
         ┌──────────────────────────────┐
         │  CROSS-MODAL FUSION          │
         │  - project image patches to  │
         │    1024 via Linear(768,1024) │
         │  - text-to-image cross-attn: │
         │    text_cls queries attend   │
         │    over projected patches    │
         │  - output: image_grounded    │
         │    context (1024) + attention│
         │    map (14x14)               │
         └────────┬─────────────────────┘
                  │ attention map A in R^14x14
                  │
                  v
         ┌───────────────────────────────────────┐
         │  FUSED REPRESENTATION (3072-d)         │
         │  = [text_cls, image_cls_proj,          │
         │     image_grounded_context]            │
         └────────┬──────────────────────────────┘
                  │
        ┌─────────┼──────────┬────────────────────┐
        v         v          v                    v
   ┌────────┐ ┌────────┐ ┌─────────┐       ┌──────────────┐
   │verdict │ │ score  │ │contrast │       │ grounding    │
   │ head   │ │ head   │ │ proj    │       │ head (A)     │
   │ → 2-cls│ │ → σ ∈  │ │ → 128   │       │ → 14x14 map  │
   │ softmax│ │  [0,1] │ │   dim   │       │ supervised   │
   │        │ │        │ │         │       │ by radiolog. │
   │        │ │        │ │         │       │ bboxes when  │
   │        │ │        │ │         │       │ available    │
   └────────┘ └────────┘ └─────────┘       └──────────────┘
        │
        v
   ┌────────────────────────────────────────────────┐
   │           DOWNSTREAM PIPELINE                   │
   │  score → temperature scaling                    │
   │  → conformal variant (inv/weighted/DR cfBH)     │
   │  → provenance gate                              │
   │  → final verdict ∈                              │
   │  {supported_trusted,                            │
   │   supported_uncertified,                        │
   │   contradicted,                                 │
   │   uncertain}                                    │
   └────────────────────────────────────────────────┘
```

---

## 5. Model (detailed)

### 5.1 Image encoder

**Primary choice: BiomedCLIP** (Zhang et al. 2023, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`).
- Trained on 15M biomedical image-text pairs from PubMed Central.
- Does not include MIMIC-CXR.
- Contamination with CheXpert test images is minimal (PMC figures are redrawn/cropped).
- ViT-B/16, 224×224 input, 196 patch tokens + 1 CLS.
- Pin `revision=<specific SHA>` in HF download.

**Secondary / ablation: RAD-DINO-MAIRA-2** (`microsoft/rad-dino-maira-2`).
- Radiology-specific self-supervised.
- Trained on public CXR datasets including MIMIC-CXR, CheXpert, PadChest, BRAX, NIH.
- We DO NOT evaluate on MIMIC-CXR, so this is OK; but contamination with CheXpert/PadChest training splits is a concern when evaluating on CheXpert Plus test.
- Reported as ablation only to show the primary result isn't dependent on a radiology-specific backbone.

**Ablation (tertiary): DINOv2 (`facebook/dinov2-base`)** — natural-image baseline to show radiology-specific encoders help.

All three are frozen except the final 2 transformer blocks, which are fine-tuned with a small learning rate (1e-5).

### 5.2 Text encoder

RoBERTa-large, unchanged from v1/v3.
- Input: `[CLS] claim [SEP] evidence [SEP]`
- Max length 256 tokens.
- First 8 layers frozen during fine-tuning (same as v3 recipe).
- Version-pinned revision.

### 5.3 Cross-modal fusion

```python
class CrossModalFusion(nn.Module):
    def __init__(self, text_dim=1024, image_dim=768, num_heads=8):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(text_dim)

    def forward(self, text_cls, image_patches):
        # text_cls: (B, 1024), image_patches: (B, 196, 768)
        image_proj = self.image_proj(image_patches)           # (B, 196, 1024)
        q = text_cls.unsqueeze(1)                             # (B, 1, 1024)
        attn_out, attn_weights = self.cross_attn(
            q, image_proj, image_proj, need_weights=True,
            average_attn_weights=False,
        )
        # attn_weights: (B, num_heads, 1, 196) -> mean over heads -> (B, 196)
        attn_map = attn_weights.mean(dim=1).squeeze(1)        # (B, 196)
        grounded = self.layer_norm(attn_out.squeeze(1))       # (B, 1024)
        return grounded, attn_map
```

The 196-token attention map reshapes to a 14×14 heatmap aligned with the 224×224 input via 16-pixel patches.

### 5.4 Fused representation + heads

```python
fused = torch.cat([text_cls, image_cls_proj, grounded_context], dim=-1)  # (B, 3072)
# image_cls_proj = Linear(768, 1024)(image_cls)

verdict_logits = verdict_head(fused)        # (B, 2)
score = sigmoid(score_head(fused))          # (B,)
contrastive_emb = contrastive_proj(fused)   # (B, 128)
grounding_map = attn_map.view(B, 14, 14)    # (B, 14, 14)  — separate head
```

Head MLPs are `Linear(3072, 256) → ReLU → Dropout(0.1) → Linear(256, out_dim)`.

### 5.5 Loss function

```
L_total = L_verdict + λ_score · L_score + λ_contrast · L_contrast
        + λ_ground · L_grounding · 1[has_annotation]
        + λ_consistency · L_consistency · 1[counterfactual_pair]
        + λ_evidence · L_evidence_ablation
```

- `L_verdict`: cross-entropy on 2-class verdict, **no label smoothing** (empirical finding from v1; label smoothing prevents calibration).
- `L_score`: BCE on sigmoid score with verdict label.
- `L_contrast`: InfoNCE between (claim, supporting-evidence) pairs vs (claim, contradicting-evidence) pairs within batch.
- `L_grounding`: BCE between attention map and radiologist bbox mask (downsampled to 14×14). Applies only when a bbox is available (MS-CXR, PadChest-loc, RSNA, SIIM subsets).
- `L_consistency`: symmetric KL between scores on claim vs claim-counterfactual (from the existing 2880-pair corpus, reused).
- `L_evidence_ablation`: hypothesis-only regularizer. For a sampled fraction of batch examples, blank the evidence and add a margin loss: the model's score with evidence should exceed score-without-evidence by at least ε. **This is the direct fix for the v4 HO-gap-negative problem.**

Weights (initial, tuned by Hydra+Optuna):
- `λ_score = 0.3`
- `λ_contrast = 0.1`
- `λ_ground = 0.5` (only when active)
- `λ_consistency = 0.2`
- `λ_evidence = 0.3`

### 5.6 Backward-compatible inference

For deployment where no image is available (legacy text-only inference), the model accepts `image=None` and replaces image features with the learned `image_null_embedding` parameter (a 768-d learned vector). This preserves v3-compatible behavior while allowing image grounding when available.

---

## 6. Training pipeline

### 6.1 Training data

**Text-only claims with images (primary, ~30k examples):**
- Reuse existing `/data/verifier_training_data_v3.json` (30k claims, 12-type perturbation taxonomy).
- Augment each row with the source CXR image from CheXpert Plus.
- Stratify patient-disjoint: 38,835 training / 9,708 calibration / 16,182 test (existing split).

**Grounding-supervised subset (~5k examples):**
- MS-CXR 1,162 phrase-image pairs.
- PadChest-localized ~3k bboxes.
- RSNA Pneumonia ~6k bboxes (use subset balanced with other classes).
- SIIM-ACR Pneumothorax ~2k masks.
- Generate positive claim from annotated finding; generate hard-negative claims via the 12-type taxonomy; `L_grounding` supervised against radiologist bbox.

**Counterfactual pairs (~8.6k):**
- Reuse existing `/data/counterfactual_preference_pairs_v3.json` (2880 claims × 3 variants).
- These drive `L_consistency`.

**Real-hallucination training mix (optional, Plan B):**
- Run MAIRA-2, CheXagent-8b, MedGemma, LLaVA-Rad, Llama-3.2-Vision on CheXpert Plus training images.
- Extract claims from generated reports.
- Label with 3-LLM ensemble (GPT-4o + Claude 3.7 + Llama-3.1-405B), consensus rule.
- Validate labeler against MS-CXR radiologist grounding on a holdout.
- Mix 70% real / 30% synthetic perturbation if Plan A HO-gap fix doesn't reach target.

### 6.2 HO-gap fix plan (A / B / C)

Success criterion (all plans): HO gap ≥ 10 pp AND MS-CXR grounding IoU ≥ 0.4.

**Plan A — Evidence-ablation regularizer + adversarial filter.**
- Add `L_evidence_ablation` to loss (§5.5).
- Run the HO baseline on the training set once per epoch; drop examples the HO baseline solves in ≥ 50% of 3 seeds.
- Retrain. 5 seeds.

**Plan B — Real-hallucination-dominant training.**
- Replace 70% of training data with real-VLM-generated hallucinations labeled by the LLM ensemble.
- Retrain with same losses.

**Plan C — Architectural evidence-gating.**
- Replace concatenation-based fusion with explicit evidence-gated attention: verdict head receives only the fused representation conditional on a learned evidence-present gate; when evidence is masked, gate closes → verdict defaults to "cannot certify."
- Retrain.

**Plan D (negative-result pivot)** if all three fail: reframe paper as an honest demonstration that HO shortcut is fundamental to synthetic training, release benchmark + protocol. Still publishable in NMI / NeurIPS D&B.

### 6.3 Training hyperparameters

- Optimizer: AdamW (lr=2e-5 for unfrozen text layers, 1e-5 for unfrozen image blocks, weight_decay=0.01).
- Effective batch size 32 (8 per GPU × 4 grad-accum; H100 single-GPU).
- Epochs: 3, early stop on val-loss patience=1.
- Mixed precision bf16.
- Gradient clipping max_norm=1.0.
- 3 seeds per reported configuration.

### 6.4 Modal training job

App name: `claimguard-pathb-train`.
Image: existing `transformers==4.40.0` image with added `open_clip_torch`, `timm`, and pinned `torch==2.3.0`.
GPU: H100, detached launch via `Function.from_name(...).spawn(...)`.
Expected wall time per run: ~90 min. Full sweep (hyperparameter × seeds): ~25 H100-hours.

---

## 7. Evaluation pipeline

### 7.1 Per-site test sets

| Site | Claim source | Claims (target size) | Image anchor | Primary metric |
|---|---|---|---|---|
| CheXpert Plus test | Radiologist reports | 15k (existing) | CheXlocalize subset | Accuracy, AUROC, FDR |
| MS-CXR | Direct phrases | 1,162 | Radiologist bboxes | Grounding IoU, claim accuracy |
| PadChest test | Radiologist reports | 5k | Localized subset bboxes | Accuracy, grounding IoU (subset) |
| BRAX test | Radiologist reports | 5k | — | Accuracy, calibration ECE |
| OpenI | Radiologist reports | 1,784 | — | Accuracy, FDR (existing) |
| RSNA | Generated VLM claims | 2k | Radiologist bboxes | Claim accuracy, grounding IoU |
| SIIM-ACR | Generated VLM claims | 1k | Radiologist masks | Grounding IoU |
| CheXmask audit | Generated VLM claims | 2k | Segmentation | Anatomical consistency rate |

### 7.2 Metrics

Per-site headline metrics (all with 95% bootstrap CI, B=1000):
- Accuracy (binary and per-class).
- Macro F1.
- Contradiction recall.
- AUROC (with DeLong test for pairwise).
- Expected calibration error (ECE).
- Empirical FDR at α ∈ {0.01, 0.05, 0.10, 0.20}.
- Power (fraction of truly-safe claims flagged safe).

Grounding metrics (MS-CXR, PadChest-loc, RSNA, SIIM):
- Attention-map IoU with radiologist bbox at threshold 0.3.
- Pointing-game accuracy (max-attention pixel inside bbox).
- AUC-PR on per-pixel attention vs bbox mask.

Fairness (where demographics available):
- Accuracy, FDR, ECE stratified by sex, age-quartile, race (CheXpert Plus), scanner manufacturer, report length decile.

### 7.3 Eval harness

Unified CLI:
```bash
claimguard-eval \
  --site {chexpert_plus,ms_cxr,padchest,brax,openi,rsna,siim,chexmask} \
  --checkpoint <path> \
  --retrieval {oracle,dense,bm25,hybrid} \
  --conformal {none,inverted,weighted,doubly_robust,stratcp} \
  --provenance-gate {on,off} \
  --output <dir>
```

Each run writes:
- `per_claim.jsonl` — one row per claim with all scores, predictions, conformal p-value, attention map.
- `summary.json` — headline metrics + CIs.
- `config.json` — pinned hashes (dataset SHA, checkpoint SHA, code SHA, transformers SHA).
- `mlflow_run_id` — MLflow run UUID for drill-down.

---

## 8. Conformal machinery (four variants)

### 8.1 Inverted cfBH (retained from v3)

Score `s_j = softmax(verdict_logits)[:, 0]`. Calibrate on contradicted calibration claims. p-value via reverse-rank. Apply global BH at level α.

### 8.2 Weighted cfBH (NEW)

Tibshirani et al. 2019 weighted conformal. Estimate density ratio `w(x) = p_test(x) / p_cal(x)` via a classifier trained to distinguish calibration from test features (logits from the verifier's fused representation + demographic stratum). Weight conformal p-values accordingly.

Diagnostic: effective sample size `ESS = (Σ w_i)² / Σ w_i²`. If `ESS < 200`, fall back to un-weighted and document.

### 8.3 Doubly-robust cfBH (NEW)

Yang & Kuchibhotla 2024 style: estimate both the score distribution and the density ratio; combine via a doubly-robust estimator that is valid if either is correct.

### 8.4 StratCP (baseline, retained)

Zitnik lab medRxiv 2026 reimplementation. Validated earlier at ±2 pp of nominal coverage on synthetic strata.

### 8.5 Empirical audit

For each (site, variant, α):
- Report target α vs empirical FDR with 1000-replicate bootstrap CI.
- Report power.
- Report ESS for weighted variants.
- If empirical FDR > α + CI, flag "guarantee violated" in the table honestly.

---

## 9. Provenance gate (retained, scaled)

Retained architecture from v3: 5 trust tiers, gate function downgrades same-model / unknown evidence before cfBH decision is consumed. See `inference/provenance.py` (unchanged).

Scaled dual-run experiment:
- 3 VLMs (CheXagent-8b, MAIRA-2, MedGemma-4B) × 4 temperatures (0.3, 0.7, 1.0, 1.2) × 1000 images.
- Pairwise: same-model (A vs A') and cross-model (A vs B) at matched temperatures.
- Report downgrade rate, verifier-score divergence as a function of sampling entropy, and a new cross-model diagnostic: does the verifier agree more when A and B are both trained on MIMIC-CXR (shared pretraining) vs A trained-on-MIMIC and B trained-on-CheXpert? If so, shared-pretraining-induced agreement is a new failure mode to document.

---

## 10. Claim extraction

The LLM claim extractor from v3 is retained but must be **validated before use**:
- Evaluation set: RadGraph-XL is PhysioNet-gated, so use a 200-report subset of CheXpert Plus + OpenI, self-annotated or LLM-ensemble-annotated for atomic claim boundaries.
- Precision target: ≥ 0.95. If below, swap for a supervised extractor trained on MS-CXR phrases (already atomic).
- Locked at a specific version SHA before downstream claim generation; any change in extractor invalidates all downstream data.

---

## 11. Baselines

All baselines are tuned (3+ prompt variants, 0/4/16-shot each, CoT on/off), and report the best-on-dev configuration:

- **Rule-based** (keyword negation detector) — reused.
- **Hypothesis-only** (evidence blanked, same verifier architecture) — reused; this is the HO baseline for the HO-gap metric.
- **Zero-shot Llama-3.1-70B / Llama-3.2-Vision / GPT-4o / Claude 3.7 Sonnet** — prompted as medical fact-checker; Claude is dropped for the v3 paper because it was used in silver labeling (circularity). GPT-4o carries.
- **CheXagent-8b (image + text)** — using validated prompt format, tuned.
- **MAIRA-2** — Microsoft (check license; if redistributable model weights available).
- **MedGemma-4B** — Google.
- **LLaVA-Med** — fallback if MedGemma unavailable.

**Contamination audit:** for each (baseline, site), flag whether the baseline's pretraining data overlaps with the site's test set. Headline Table 1 reports only *clean* pairs. Supplementary includes contaminated pairs with explicit annotation.

---

## 12. Laptop-sleep-durable execution

This is a hard requirement. The system must make forward progress while Aayan's laptop is asleep.

### 12.1 Three-tier durability

**Tier 1 — Modal detached execution.**
All long-running work (training, evaluation, large API batches, dataset downloads > 1GB) runs on Modal via the `Function.from_name(...).spawn(...)` pattern (see `scripts/launch_task3c_detached.py`). Local Python exits after spawning; Modal's cloud runs the job. Laptop can sleep, disconnect, close — the job runs.

**Tier 2 — Modal chain scripts.**
When phase N's output is phase N+1's input, phase N's Modal function ends with a `spawn()` call that launches phase N+1 directly. No local orchestration between phases. Implementation: a `PipelineOrchestrator` class on Modal that reads a YAML pipeline spec and transitions between stages based on artifact-existence checks on the shared volume.

**Tier 3 — Remote Claude cron agent.**
For decisions that require agent judgment (retry failed runs, advance to next phase after reviewing artifacts, respond to reviewer-agent findings), a Claude remote agent is scheduled via `CronCreate` to fire every 6 hours. This agent runs in Anthropic's cloud (not on the laptop), reads pipeline state from the Modal volume, and makes advance/retry/escalate decisions. It writes its decisions to a log; on laptop wake, the user sees progress.

### 12.2 Pipeline state model

All pipeline state lives on the Modal volume `claimguard-data` under `/pipeline/`:

```
/pipeline/
├── state.json              # current phase, last checkpoint, next action
├── manifest.json           # artifact inventory with SHAs
├── events.jsonl            # append-only event log (phase started, failed, completed)
├── next_actions.json       # queue of pending Modal function calls
└── locks/                  # advisory locks to prevent double-execution
```

State transitions are atomic writes to `state.json` via a `VolumeFS.atomic_replace()` helper.

### 12.3 Failure handling

- Modal retries transient infra errors (OOM, preemption) automatically up to 3x.
- Persistent failures are logged to `events.jsonl` with stack traces and escalated to the cron agent, which can retry with modified hyperparameters or escalate to the user (email via a cheap API).
- A kill-switch flag `/pipeline/kill.flag` on the volume halts all cron advances when set. Safety mechanism if Aayan wants to stop.

### 12.4 Budget enforcement

A `BudgetGuard` Modal function runs every hour:
- Reads current Modal cost from the Modal billing API.
- If spend exceeds per-phase cap × 110%, sets `/pipeline/kill.flag`.
- Emails Aayan.

---

## 13. Reproducibility infrastructure

### 13.1 Code

- Single Git repo `claimguard-nmi/` (rename from `verifact/`).
- Every experiment config is a Hydra YAML; runs are launched as `python -m claimguard.run <config>`.
- Pin `torch==2.3.0`, `transformers==4.40.0` for trainer, `transformers==4.50.0` for MedGemma (separate images), `anthropic==0.40.0`, `captum==0.7.0`, `open_clip_torch==2.24.0`, `timm==0.9.16`.
- Every HF model load pins `revision=<sha>`.
- GitHub Actions CI: unit tests + 100-example smoke eval on every PR.

### 13.2 Data

- DVC tracking every input dataset + every generated artifact.
- Content-addressed storage on the Modal volume: SHA256 of dataset files → `/data/<sha>/...`.
- Dataset manifests (`/pipeline/manifest.json`) track version, source URL, download date, SHA256.

### 13.3 Experiments

- MLflow tracking server (self-hosted on Modal or Weights & Biases).
- Every training run logs: code SHA, data SHA, config YAML, all metrics, loss curves, checkpoint paths.
- Tables in the paper are generated from MLflow queries — no loose numbers.

### 13.4 Release

- **Model weights:** Hugging Face (`aayan-alwani/claimguard-cxr-pathb-v1`, one per seed).
- **Benchmark data:** Zenodo (DOI-assigned) under inherited dataset licenses. Two tiers:
  - `ClaimGuard-Bench-Open` (OpenI, BRAX, NIH, RSNA, SIIM, MS-CXR, PadChest, CheXmask subsets that are fully open).
  - `ClaimGuard-Bench-Restricted` — empty for Path B (MIMIC-CXR deferred).
- **Eval harness:** GitHub release with semver tag + Docker image (`docker.io/aayanalwani/claimguard-eval:1.0.0`).
- **Model card** (Mitchell et al. 2019).
- **Data card** (Gebru et al. 2021).
- **Third-party reproduction:** send the full harness + data + weights to one independent lab for end-to-end reproduction before submission.

---

## 14. Preregistration

Filed at OSF before collecting the benchmark or running final evals:

- Primary endpoints: (a) MS-CXR grounding IoU, (b) HO gap, (c) per-site FDR at α=0.05.
- Sample sizes: pre-committed for each site.
- Analysis plan: frozen BH procedure, frozen conformal variant selection rules, frozen baseline-tuning protocol.
- Abort criteria: per §6.2 (Plan A → B → C → D).
- All primary tests Bonferroni-corrected across baseline comparisons.

---

## 15. Authorship and governance

- **Senior / corresponding author:** Ashley Laughney (PI, Weill Cornell).
- **First author:** Aayan Alwani.
- **Biostatistics co-author:** Weill Cornell Biostatistics Core consult → named co-author for conformal + FDR claims.
- **Radiology consultant (optional, strongly recommended):** one Weill Cornell thoracic radiologist for lightweight review and edge-case adjudication; named co-author.
- **IRB:** Weill Cornell exempt/non-human-subjects determination for retrospective analysis of public de-identified datasets. Laughney files; reference letter in manuscript.
- **Conflict-of-interest + funding disclosures** prepared at submission.

Submission is **blocked** until Laughney + biostatistician are on the author list. Radiology consultant is strongly preferred but not strictly blocking.

---

## 16. Directory layout (Path B)

```
verifact/ (rename → claimguard-nmi/)
├── ARCHITECTURE_V2_IMAGE_GROUNDED.md       # this doc
├── CLAIMGUARD_PROPOSAL.md                   # proposal, updated to Path B
├── MANUSCRIPT.md                            # paper draft
├── decisions.md                             # D1..DN log
├── progress.md                              # phase tracker
│
├── claimguard/                              # the pip-installable package
│   ├── __init__.py
│   ├── config/                              # Hydra YAMLs
│   │   ├── model/biomedclip.yaml
│   │   ├── model/raddino.yaml
│   │   ├── data/ms_cxr.yaml
│   │   ├── data/padchest.yaml
│   │   ├── conformal/inverted.yaml
│   │   └── conformal/weighted.yaml
│   ├── models/
│   │   ├── verifier.py                      # NEW canonical VerifierModel (image-grounded)
│   │   ├── image_encoder.py                 # NEW BiomedCLIP / RAD-DINO / DINOv2 wrappers
│   │   ├── fusion.py                        # NEW CrossModalFusion
│   │   └── heads.py                         # verdict/score/contrast/grounding
│   ├── data/
│   │   ├── adapters/                        # NEW per-dataset loaders
│   │   │   ├── chexpert_plus.py
│   │   │   ├── ms_cxr.py
│   │   │   ├── padchest.py
│   │   │   ├── rsna.py
│   │   │   ├── siim.py
│   │   │   ├── openi.py
│   │   │   ├── brax.py
│   │   │   ├── chexmask.py
│   │   │   └── nih_cxr14.py
│   │   ├── augmentation/                    # REUSED
│   │   ├── extraction/                      # claim extractor validation
│   │   └── splits.py                        # patient-disjoint split utils
│   ├── training/
│   │   ├── losses.py                        # all L_* components
│   │   ├── trainer.py                       # multi-loss trainer
│   │   └── regularizers.py                  # evidence ablation, adversarial filter
│   ├── eval/
│   │   ├── harness.py                       # unified CLI backend
│   │   ├── metrics.py                       # reused + extended
│   │   ├── grounding.py                     # IoU, pointing-game, AUC-PR
│   │   ├── conformal.py                     # inverted/weighted/DR/stratcp
│   │   └── baselines/
│   ├── pipeline/
│   │   ├── orchestrator.py                  # Tier-2 Modal chain controller
│   │   ├── state.py                         # atomic state updates
│   │   ├── budget.py                        # BudgetGuard
│   │   └── cron_agent.py                    # Tier-3 Claude cron prompt + logic
│   └── utils/
│
├── scripts/
│   ├── launchers/                           # all use `Function.from_name().spawn()`
│   │   ├── launch_download_site.py
│   │   ├── launch_training.py
│   │   ├── launch_eval.py
│   │   ├── launch_provenance_scaled.py
│   │   └── launch_pipeline.py               # end-to-end chain
│   ├── modal_apps/                          # @app.function definitions
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── dataset_downloader.py
│   │   ├── budget_guard.py
│   │   └── orchestrator.py
│   └── tools/
│       ├── download_<dataset>.py            # one per dataset
│       ├── validate_claim_extractor.py
│       ├── build_retrieval_index.py
│       └── compile_tables.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/                               # run on every CI
│
├── docker/
│   ├── train.Dockerfile
│   ├── eval.Dockerfile
│   └── medgemma.Dockerfile                  # separate transformers pin
│
├── docs/
│   ├── data_card.md
│   ├── model_card.md
│   └── osf_preregistration.md
│
└── .github/workflows/
    ├── ci.yml
    └── smoke_eval.yml
```

The existing `verifact/` tree is preserved as a reference — we don't delete it, we build alongside and migrate.

---

## 17. Feasibility gates

Before executing any GPU work, these must all pass:

| Gate | Check | How validated |
|---|---|---|
| F1 | BiomedCLIP + RAD-DINO + DINOv2 all downloadable from HF without auth | `huggingface-cli download` dry run |
| F2 | MS-CXR accessible without DUA | Download test of 5 samples |
| F3 | CheXmask accessible and usable | Download manifest + verify mask alignment to public CXRs |
| F4 | PadChest English-translated subset available | BIMCV registration OR HF mirror check |
| F5 | RSNA + SIIM on Kaggle accessible with API key | `kaggle datasets download` test |
| F6 | BRAX accessible via IEEE DataPort | Registration + sample download |
| F7 | OpenI direct download still works | URL test |
| F8 | NIH CXR14 direct download works | URL test |
| F9 | Existing `VerifierModel` class extends cleanly to image-grounded | Forward-pass shape test in a Python REPL |
| F10 | `scripts/launch_task3c_detached.py` spawn pattern still works for new trainers | Spawn a `print("hello")` test Modal function |
| F11 | `CronCreate` remote agent can read/write Modal volume | Write-test via a scheduled no-op |
| F12 | Modal budget cap raised to $2,500 | Confirmation from Aayan's Modal dashboard |

F12 is the only gate requiring user action. Others are automated.

---

## 18. Phase-by-phase execution

Phases are numbered; each has an abort gate. Phases ≥3 are fully laptop-sleep-durable.

**P0 — Scaffolding & feasibility (local, fast).**
- Verify F1–F11 automatically.
- Scaffold `claimguard/` package skeleton, configs, CI.
- Set up MLflow, DVC, pre-commit hooks.
- Commit.

**P1 — Dataset download & validation (Modal detached).**
- One Modal function per dataset; all fire-and-forget spawns.
- Each validates integrity (SHA256, sample parse, manifest emission).
- CronCreate agent polls completion, advances to P2 when all present.

**P2 — Claim extractor validation (Modal).**
- Extract claims on held-out subset, self-score, validate ≥ 0.95 precision.
- If fail: swap extractor, retry.

**P3 — Cross-modal verifier training, Plan A (Modal, H100 detached, 3 seeds).**
- Spawn 3 training runs in parallel.
- Log to MLflow.
- Compute HO gap + MS-CXR grounding IoU on completion.
- Gate: HO ≥ 10 pp AND IoU ≥ 0.4 → P5. Else → P4.

**P4 — Plan B / C iteration (Modal, conditional).**
- If Plan A failed gate: kick off Plan B (real-hallucination training mix). If that fails: Plan C (architectural evidence gate). If all three fail: Plan D (negative-result reframe).

**P5 — Full multi-site evaluation (Modal detached, parallel per site).**
- 8 sites × 1 checkpoint × 4 conformal variants × 4 α levels.
- All run in parallel Modal jobs.
- Merge into final results tables.

**P6 — Scaled provenance gate experiment (Modal, ~$100).**
- 3 VLMs × 4 temperatures × 1000 images.
- Full cross-model + same-model matrix.

**P7 — Statistical analysis + fairness + CIs (local or Modal; cheap).**
- Bootstrap CIs for all tables.
- DeLong, McNemar on pairwise.
- Subgroup stratification.

**P8 — Manuscript drafting.**
- Claude drafts sections; Laughney + biostats review.

**P9 — Submission prep.**
- Final figures, supplementary, formatting.
- OSF preregistration locked.
- Independent reproducer run scheduled.

**P10 — Submit to NMI.**

---

## 19. Risk register (top 10, plan-specific)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Plan A HO fix underperforms | Medium | High | Plans B → C → D pre-specified |
| R2 | BiomedCLIP + RoBERTa fusion doesn't beat text-only | Medium | High | RAD-DINO ablation + retrieval-augmented fallback |
| R3 | PadChest BIMCV registration denied | Low | Medium | Drop PadChest; still have 6 other sites |
| R4 | Kaggle API rate-limits on RSNA/SIIM | Low | Low | Pre-stage via one-time download |
| R5 | Modal budget overrun | Medium | High | BudgetGuard + kill-switch |
| R6 | Laughney rejects the scope pivot | Low–Medium | Critical | Early meeting to align; this doc is the pitch |
| R7 | Grounding supervision is too noisy | Medium | Medium | Downweight `λ_ground` and report grounding as secondary metric |
| R8 | Reviewer insists on MIMIC-CXR | High | Medium | Attempt PhysioNet credentialing in parallel as add-on; submit without if credentialing delays |
| R9 | BiomedCLIP has hidden contamination with CheXpert test | Medium | Medium | Report RAD-DINO and DINOv2 ablations; disclose in limitations |
| R10 | Cron agent mis-advances pipeline (runs wrong next step) | Low | High | Strict state-machine checks + advisory locks + kill-switch |

---

## 20. Budget (revised)

| Line | Estimate |
|---|---|
| Training (3 plans × 3 seeds × ~1.5h H100) | $150 |
| Multi-site eval (8 sites × 4 baselines × ~30 min) | $200 |
| Dataset downloads (storage, transfer) | $30 |
| Scaled provenance experiment | $150 |
| Real-hallucination mining (if Plan B triggers) | $250 |
| LLM-ensemble labeling (tiered: GPT-4o-mini triage + GPT-4o/Claude for disagreements) | $400 |
| Baseline API calls (GPT-4o, Claude 3.7, Llama-3.1-405B) | $300 |
| Contingency | $500 |
| **Total** | **~$1,980** |

Request Modal/API budget lift to $2,500.

---

## 21. Decision log (to be appended)

Path B decisions will be numbered PB1, PB2, ... in `decisions.md`, keeping the existing D1–D30 intact.

- **PB1** — Pivot from text-only NeurIPS 2026 submission to image-grounded Nature MI target. Rationale: negative HO-gap on v4 v2 makes text-only story unsupportable at Nature-family venues; image-grounded evaluation via public radiologist annotations circumvents both the HO problem and the radiologist-recruitment constraint.

- **PB2** — Primary vision encoder is BiomedCLIP, not RAD-DINO. Rationale: RAD-DINO includes MIMIC-CXR training which would be a contamination problem if we later add MIMIC-CXR evaluation via credentialing; BiomedCLIP is broader-biomedical and cleaner. RAD-DINO retained as ablation.

- **PB3** — All long-running work runs via Modal `Function.from_name().spawn()`. Local orchestration is banned for work > 5 min. Rationale: laptop-sleep durability is a hard project requirement.

---

**End of ARCHITECTURE_V2_IMAGE_GROUNDED.md.**
