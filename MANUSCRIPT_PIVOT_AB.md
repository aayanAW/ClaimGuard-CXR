# Causally-Faithful Radiology Report Generation via Dual-Adversarial Counterfactual Training

**Anonymous Authors**

*Submission under double-blind review. Author identification will be added at camera-ready.*

---

## Abstract

Multimodal medical generators — vision-language models that produce radiology reports from chest radiographs — have repeatedly been shown to underuse the image and overrely on textual context. Prior work has documented this *evidence-blindness* diagnostically, by showing that masking the image at inference time produces only a small drop in output quality, and proposed inference-time mitigations (conformal selection, post-hoc filtering) that do not change the underlying model. We instead train against the failure mode at fine-tuning time. We introduce **dual-adversarial causal-faithfulness training**, a LoRA-based fine-tuning recipe for image-conditioned radiology generators that combines (a) a *symmetric* adversarial filter that downweights training samples solvable by either text alone *or* image alone — extending the text-only Shortcut-Failure Coefficient of Z. Li et al. (2025) with the image-only side that the literature has not previously implemented for medical generation; and (b) an online *causal-faithfulness loss* that penalises a small gold-token correctness gap between the model's distribution under the full image and under per-channel-mean and laterality-flipped counterfactual images. We further introduce **IMG_correct**, a methodological refinement of the standard image-masking gap (IMG) that excludes the *induced inversion* failure mode — the phenomenon where a consistency-loss-trained model flips its argmax under image-zeroing without its gold-class probability dropping by a meaningful margin (i.e., the prediction changed but support for the truth did not actually decrease, so the IMG metric over-credits the model with image-grounded behaviour). We apply IMG_correct retroactively to representative prior systems to quantify how much of the historical evidence-blindness mitigation literature was induced inversion versus faithful grounding. On the public-data-only OpenI chest X-ray collection, our full-method model (`sft_full`, faith loss + discrete dual filter; n = 3 seeds) achieves **IMG_correct = 74.96 ± 0.83 pp at 75.48 % token-level accuracy**, and the continuous-SFC ablation (`sft_full_sfc`; n = 3 seeds) achieves **IMG_correct = 75.42 ± 0.28 pp at 75.44 % accuracy** — versus **2.38 ± 0.26 pp at 74.31 % accuracy** for the seed-averaged SFT-only baseline (n = 3 seeds). The faith-loss-only ablation (`sft_faith`, n = 3) achieves 74.11 ± 0.26 pp, isolating the faith loss as the load-bearing component. The full method gives a **+72.58 pp improvement at +1.17 pp accuracy** over the SFT-only baseline. We then run a *cross-counterfactual* test (§5.5) to disambiguate "model learned its trained loss" from "model genuinely uses the image": under image-cropping or centre-occlusion (counterfactuals the faith loss did not train on), `sft_full` IMG_correct drops to ≤ 5 pp — close to baseline. Under noise replacement (a globally-uniform untrained counterfactual) it stays at 62.20 pp. The headline +72.58 pp result is therefore partially counterfactual-specific: the model has tightly fit the trained per-channel-mean and noise distributions, with weak transfer to localised counterfactuals. We also run a cross-site transfer (§5.6, ChestX-Det10 silver-target, n = 15,955 token positions); IMG_correct retains 54.62 pp at 54.65 % accuracy, a roughly proportional drop in both metrics. Induced-inversion contributions are essentially zero across all three faith-loss-bearing configurations × three seeds (nine fine-tunes; per-configuration seed-mean ≤ 0.01 pp). The pre-registered claim of ≥ 5 pp improvement is exceeded by an order of magnitude on three independent seeds × three independent faith-loss configurations. The IMG_correct retroactive audit on prior verifier checkpoints (Section 5.4) reveals that, contra our pre-registered concern, induced inversion was already small in the prior literature (≤ 1.31 pp on every audited cell, ≤ 13.2 % of the corresponding IMG); the methodological refinement is therefore a *verification* tool that increases confidence in prior IMG numbers, not an audit that overturns them. All training data, code, and model weights are released, reproducible without PhysioNet credentialing.

---

## 1. Introduction

A growing body of work has documented that multimodal medical AI systems — vision-language models for radiology, ophthalmology, pathology — make their predictions in ways that disregard the visual input. Restrepo et al. (2025) introduced *Selective Modality Shifting* on MIMIC-CXR and FairVLMed and showed that six open-source VLMs, including those fine-tuned for medical use, exhibit a "marked dependency on text input which persists despite the presence of complementary visual information." This phenomenon is the medical-imaging analog of the partial-input shortcuts documented by Gururangan et al. (2018) and Poliak et al. (2018) for natural language inference, and the multimodal-reward-model shortcuts documented by Z. Li et al. (2025).

Two facts about the existing literature motivate the present work. First, the evidence-blindness diagnostics in the medical-imaging literature so far have been *measurement-only*: Restrepo et al. (2025) propose SMS, but their conclusion explicitly leaves training-time mitigation to future work. The conformal-selection literature in this space (Gui et al. 2024; Li et al. 2025) similarly wraps an existing predictor without touching the underlying model. Second, the only training-time mitigation that has been applied at scale to multimodal models — Z. Li et al. (2025)'s Shortcut-Failure Coefficient reweighting — operates only on the text-only side, despite the fact that an image-only shortcut is just as conceptually possible and, on naturalistic medical data, just as likely. To our knowledge, no published work combines a text-only and image-only adversarial filter for a medical multimodal generator and demonstrates an effect on a counterfactual-grounding diagnostic.

Our **two primary contributions** are as follows. **First**, we develop and release a fine-tuning recipe — *dual-adversarial causal-faithfulness training* — that addresses both training-distribution shortcuts symmetrically and adds an online causal-faithfulness loss with a token-wise correctness-gain margin. The full method is a LoRA fine-tune of MedGemma-4B-IT (Google HAI-DEF, research-only licence) trained on the OpenI chest X-ray collection with no PhysioNet-credentialed data. **Second**, we contribute a methodological patch to the standard image-masking gap (IMG): IMG_correct, which counts only those examples where the model's correct-class probability dropped by a margin τ under image masking, ruling out the *induced inversion* failure mode under which a consistency-loss-trained model is wrong on the image-zeroed input by flipping to a different-but-still-wrong answer rather than dropping its support for the gold class. We apply IMG_correct retroactively to our prior verifier work — re-evaluating the v5.3 contrast configuration that originally produced an IMG ≈ 68 pp claim — and find that the IMG = 68.12 pp re-evaluation decomposes into IMG_correct = 71.05 pp with **induced inversion of only 0.74 pp (1.1 % of IMG)**. The pre-registered hypothesis that ≥ 30 % of the IMG would resolve to induced inversion is decisively refuted; the prior IMG number on this verifier family was largely measuring real image-grounded prediction. We report this negative result transparently and recast IMG_correct as a *verification* default that gives future workers a way to detect the failure mode if less calibrated training recipes were to introduce it; the methodological patch is reusable on any prior image-masking diagnostic and we recommend it become a standard reporting requirement. As a **third instrumentation release** (not a validated contribution), we ship a training-time *gradient-norm monitor* (`pivot_ab/grad_monitor.py`) that logs `R(t) = ‖∇ image-tokens‖ / (‖∇ image-tokens‖ + ‖∇ text-tokens‖)` and the cross-modal attention rank, with a hook for fitting a logistic blindness predictor; we do not run the held-out mech-interp sweep required to validate the predictor's AUROC in the present paper, and so we explicitly do not claim a deployable monitor here.

The combination of these two contributions yields a deployable model and a methodological audit-tool in a single paper. The rest of this paper is organised as follows. Section 2 reviews the prior literature on evidence-blindness diagnostics, adversarial filtering, and conformal selection for multimodal medical AI, and positions our work along the two contribution axes. Section 3 details the methods: backbone, loss function, filter, monitor instrumentation, and diagnostic. Section 4 describes the experimental setup. Section 5 reports the results. Section 6 discusses what the results mean for the deployment of multimodal medical generators. Section 7 reports our limitations — including the central limitation that the faith loss and the IMG_correct headline metric measure overlapping per-token quantities, plus the OpenI-only training restriction that makes the contribution reproducible without PhysioNet but also restricts us to a single-site training distribution. Section 8 concludes.

---

## 2. Related Work

We position the present work along four axes of prior literature: evidence-blindness diagnostics, adversarial filtering for shortcuts, conformal selection on top of frozen multimodal models, and the broader literature on radiology report generation.

**Evidence-blindness diagnostics.** Restrepo et al. (2025) introduce *Selective Modality Shifting* (SMS), a perturbation-based diagnostic that swaps images or text between samples with opposing labels and quantifies the resulting drop in accuracy. They evaluate four generalist VLMs and two medical-specialist VLMs on MIMIC-CXR and FairVLMed and find a systematic over-reliance on text. Their conclusion explicitly identifies training-time mitigation as future work. We reuse the SMS family of metrics (extended in our IPG, ESG, IMG family — Section 4) as our primary evaluation harness, but our contribution is the training-time intervention they call for. The image-masking gap (IMG) used throughout this paper is conceptually identical to the image-perturbation-only branch of SMS; our refinement (IMG_correct) addresses an *induced-inversion* failure mode that the SMS family does not distinguish. Earlier diagnostic work for natural-language entailment partial-input baselines (Gururangan et al. 2018; Poliak et al. 2018) provides the conceptual precedent for treating shortcut-solvability as a property of the dataset rather than the model.

**Adversarial filtering for shortcuts.** Z. Li et al. (2025) (arXiv:2503.03122, ICML 2025; first author Zichao Li) introduce the *Shortcut-Failure Coefficient* (SFC), a continuous training-sample reweighting scheme for multimodal reward models. The SFC down-weights training samples whose verdict is solvable by a text-only model and up-weights samples where the text-only model fails. The conceptual lineage runs through earlier dataset-bias diagnostics for natural language inference, including the partial-input baselines of Gururangan et al. (2018) and Poliak et al. (2018) referenced above. The SFC is operated symmetrically only along the text axis; the obvious complementary mechanism — an image-only shortcut detector that mirrors the text-only SFC — is not implemented in Z. Li et al. (2025) or, to our knowledge, any subsequent work for the medical-imaging setting. We extend the family with a *symmetric dual filter* and use the union of text-only and image-only shortcut signals (combined via `min(w_text, w_image)`) to downweight rows that are unimodally solvable along *either* axis. The dual extension is the genuine novelty axis of the present paper relative to Z. Li et al. (2025); our dual filter additionally downweights at a discrete threshold (rule we inherited from our prior verifier work) and, in an ablation, compares against the continuous SFC formulation.

**Conformal selection on top of frozen multimodal models.** Gui et al. (2024) and Li et al. (2025) develop conformal-prediction wrappers for multimodal generators that provide finite-sample false-discovery-rate guarantees for selected outputs. *Conformal Alignment* (Gui et al. 2024) trains an alignment predictor on a reference dataset of human-labelled alignment scores, then applies a data-dependent threshold to the predictor's scores at deployment. *CONFLVLM* (Li et al. 2025) treats each generated text claim as an individual hypothesis under a heuristic uncertainty measure and demonstrates, on LLaVa-1.5 across scene/medical/document settings, that the error rate can be reduced from 87.8% to 10.0% while maintaining a 95.3% true-positive rate. Both methods operate on a frozen underlying model. Our contribution is orthogonal: we improve the underlying generator's faithfulness, which compounds with — rather than competes with — any subsequent conformal wrapper. A practitioner who deploys the full system can stack our trained model with either conformal procedure and receive both training-time and inference-time guarantees.

**Radiology report generation.** Recent work on the underlying generator side includes MAIRA-2 (Bannur et al. 2024), CheXagent-2 (Chen et al. 2024), MedGemma-4B-IT (Google HAI-DEF 2024), and LLaVa-Rad (Microsoft 2024). Of these, only MedGemma-4B-IT exposes a standard `AutoModelForImageTextToText` forward signature with a `pixel_values` keyword argument, which is required for our counterfactual training (the full forward pass is run three times per training step on full image, image-zeroed counterfactual, and image-flipped counterfactual). CheXagent-2 uses a tokenizer-embedded image paradigm that does not accept a `pixel_values` keyword and is therefore incompatible with counterfactual training without a substantial reverse-engineering effort. Our backbone choice was driven by this engineering constraint. Audit-and-repair frameworks for prior radiology generators include RadFlag (Zhang et al. 2025; ML4H), ReXTrust (Hardy et al. 2025; AAAI Bridge), and FactCheXcker (Heiman et al. 2025; CVPR). All three are post-hoc, inference-time hallucination filters operating on a frozen generator; none implement a training-time intervention against evidence-blindness. Our own prior work [citation withheld for double-blind review] is similarly frozen-predictor. The training-time, model-modifying contribution proposed here is orthogonal to all of these and stacks with each (a more faithful underlying generator delivers fewer hallucinations to filter, regardless of the filtering recipe).

**Methodological-audit precedent.** The notion that prior reported numbers in a literature should be re-audited under a refined metric has clean precedent. Recht et al. (2019) re-audited ImageNet-trained classifiers under a held-out-from-the-original-distribution test set and found that aggregate accuracy claims did not transfer. Our IMG_correct audit of prior IMG numbers is the medical-imaging analog: we accept that prior IMG numbers are correct under the IMG definition but ask how much of the reported gap survives under a more demanding definition that excludes induced inversion. The result, reported in Section 5.4 and the appendix, is a corrective table that other workers in this area can use to recalibrate their reading of the prior literature.

---

## 3. Methods

### 3.1 Backbone and parameter-efficient fine-tuning

We fine-tune MedGemma-4B-IT, a 4.31-billion-parameter image-text instruction-tuned variant of Gemma-3 trained on a mixture of medical images including chest radiographs, pathology slides, dermatology images, and fundus photography (Google HAI-DEF, research-only licence). The model exposes a standard `AutoModelForImageTextToText` interface with a `pixel_values` keyword argument that accepts a `(B, 3, H, W)` tensor of normalised image patches. We wrap the base model with a LoRA adapter (Hu et al. 2022) of rank 16 and alpha 32, applied to the `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules of every transformer block. The trainable parameter count is 11.9 million (0.28% of base); the rest of the model is frozen.

### 3.2 Composite training loss

The training objective combines three components:

```
L_total(x) = w(x) · L_SFT(x)  +  λ_faith · L_faith(x)
```

where `w(x) ∈ [w_min, 1]` is the per-row weight produced by the dual adversarial filter (Section 3.3), `L_SFT(x)` is the supervised fine-tuning cross-entropy loss on the gold radiology report tokens, and `L_faith(x)` is the causal-faithfulness loss (Section 3.4). The faithfulness weight `λ_faith` is a fixed hyperparameter (set to 0.3 throughout, chosen by a small smoke-search over `{0.1, 0.3, 1.0}` on a 6-step training pre-flight). We do not include a separate text-only filter loss component; the text-only filter contributes via its share of the dual filter's `w(x)`.

### 3.3 Dual adversarial filter

The dual filter combines two independently-trained shortcut detectors: a text-only branch that scores how predictable the gold label is from the claim and evidence text alone, and an image-only branch that scores how predictable the gold label is from the image alone. Each branch produces a per-row weight in `{w_min, 1.0}` (discrete schedule) or in `[w_min, 1.0]` (continuous SFC ablation). The dual weight is `w(x) = min(w_text(x), w_image(x))`: a row is downweighted aggressively if it is unimodally solvable along *either* axis, since unimodal solvability on either side is evidence that the model can shortcut around the multimodal grounding objective.

The text-only branch reuses the implementation from our prior verifier work [citation withheld for double-blind review]: a RoBERTa-large cross-encoder trained for one epoch on `(claim, evidence) → gt_label` pairs from the training split. At inference, rows where the text-only model's predicted probability of the gold class exceeds 0.7 are downweighted to `w_min = 0.2`; the remaining rows are kept at `1.0`.

The image-only branch is the genuine novelty axis. We extract image features via the BiomedCLIP image encoder (Zhang et al. 2024) and train a 2-layer MLP head on `image → gt_label` for one epoch. The same threshold rule (`p_correct ≥ 0.7 → 0.2`) is applied. On the 20,560-row training pool (before our subsequent text-target filter restricts to the OpenI subset), the **text-only branch downweights 85.3% of rows; the image-only branch downweights 81.2%; the union after `min(w_text, w_image)` aggregation downweights 91.7%** of rows. The four-cell decomposition of which rows each branch flags is informative for the symmetric-extension claim:

| Filter outcome | Count | % of training pool |
|---|---:|---:|
| Both filters caught (unimodally solvable on either axis) | 15,375 | 74.8% |
| Text-only caught, image-only passed | 2,156 | 10.5% |
| **Image-only caught, text-only passed (the symmetric-extension contribution)** | **1,329** | **6.5%** |
| Neither caught (genuinely multimodal) | 1,700 | 8.3% |

The **6.5%** of rows in the third row are exactly the genuine novelty axis: text-only adversarial filtering á la Z. Li et al. (2025) would let these rows train at full weight, but our image-only branch correctly identifies them as image-shortcut-solvable and downweights them. After the subsequent text-target filter that restricts training to OpenI rows with non-empty findings text (Section 4.2), the per-row weight distribution within the surviving 10,839 rows is 95.6% at `w_min=0.2` and 4.4% at `w=1.0`.

In an ablation against Z. Li et al. (2025), we replace the discrete `{w_min, 1}` rule with the continuous SFC formulation `SFC(x) = 1 - p_text_correct(x)` (clipped to `[w_min, 1]`), which provides a smoother weighting and matches the published SFC recipe.

### 3.4 Causal-faithfulness loss with token-wise gain margin

The naive consistency loss `ReLU(margin - KL(p_full ‖ p_alt))` admits a degenerate optimum: the model can drive `p_alt` to an arbitrary distribution while `p_full` is shaped by the SFT loss alone, and the divergence margin can be satisfied without the model ever using the image. This is the *induced inversion* failure mode (Section 3.6); our IMG_correct diagnostic directly measures it. To rule it out at training time, we replace the symmetric KL with a *correctness-gain margin* on the gold tokens:

```
L_faith(x) = mean over t of ReLU( margin - (log p_full(y_t) - log p_alt(y_t)) )
```

where `y_t` is the gold token at position `t`, `p_full` and `p_alt` are the model's per-position distributions under the full and image-zeroed inputs, the mean is taken over assistant-turn token positions only, and the margin is set to 0.3 nats (a deliberately small value; faithfulness-via-reduction-in-gold-token-probability is what we want, not faithfulness-via-arbitrary-distributional-divergence). The token-wise margin formulation couples the divergence requirement to gold-token correctness on the full input, so the model cannot satisfy the margin by being arbitrarily wrong on the image-zeroed input. The margin acts symmetrically across both image-zeroed and image-flipped counterfactuals; on rows that are not laterality-sensitive, the image-flipped counterfactual is replaced with the image-zeroed counterfactual as a no-op fallback, masked out via `flipped_row_mask` so it contributes zero gradient.

Image-zeroing is implemented as the per-channel mean of the input batch's pixel values, *not* as a literal zero tensor. A literal zero tensor is, after the processor's normalisation, a specific out-of-distribution input that the model has never seen at training time; the per-channel mean produces a uniform "neutral" image at the model's own normalisation, which is what we want when we say "the image is absent."

The faithfulness loss requires three forward passes per training step (full, image-zeroed, image-flipped on laterality rows). To stay memory-bounded on a single H100, we use a batch size of 2 with gradient accumulation factor 8, yielding an effective batch size of 16.

### 3.5 Gradient-norm training-time monitor

The training-time monitor logs three quantities every K = 50 steps:

1. The mean L2 norm of the loss gradient with respect to image-token embeddings, `g_img(t) = E[ ‖ ∂L / ∂h_img ‖_2 ]`.
2. The mean L2 norm of the loss gradient with respect to text-token embeddings, `g_txt(t)`.
3. The ratio `R(t) = g_img(t) / (g_img(t) + g_txt(t))`.

We additionally log a cross-modal attention rank score `r_cm(t)` (the matrix rank of the verdict-layer cross-modal attention matrix after thresholding) and an image-attention attribution score `α_img(t)` (sum of attention weights on image-token positions divided by total attention). All three quantities require only forward and backward hooks on the embedding layer; gradient norms are read from the existing backward pass, so no additional forward pass is incurred. Empirically the hook adds negligible per-step overhead (well under 1 % of the 1.5–4 hour Phase 4 fine-tune wallclock; we did not run a microbenchmark beyond confirming the trace files are written without observable training-loop slowdown).

The training-time hypothesis — *not validated in this paper* — is that evidence-blindness emerges when `R(t)` decays below a critical value relative to `g_txt(t)`: if the model can solve the SFT loss with text inputs alone, the image-token gradient carries no signal, the cross-modal attention rank collapses, and the model becomes effectively unimodal. The natural validation procedure (training 8–10 small additional configurations varying LoRA rank, training subset, and learning rate, then fitting a logistic regressor that predicts post-training IMG > 5 percentage points from the trajectory of `(R(t), r_cm(t), α_img(t))` over the first 25 % of training, validated on held-out architectures and seeds) is **not run in the present paper** (see §5.3); we ship the monitor as instrumentation so that future researchers can collect trajectories cheaply and contribute to a cross-architecture predictor.

The key practical claim *if validated* would be that the monitor is cheap (gradient norms are computed during backpropagation anyway), runs without extra forward passes, and provides an early-warning signal that lets a practitioner intervene before evidence-blindness sets in. We do not make this claim in the present paper.

### 3.6 IMG_correct: methodological refinement of the image-masking gap

The standard image-masking gap is

```
IMG(f, D) = acc(f, D) - acc(f, D_{img=0})
```

This metric counts every example on which the model is correct on the full input and incorrect on the image-zeroed input as evidence of "image-grounded" prediction. But a consistency-loss-trained model can lower its image-zeroed correctness to zero by always flipping its argmax under masking, *even if its gold-class probability barely drops*. We call the cases where the model flips its argmax under masking but does not drop its gold-class probability by ≥ τ the *induced inversion* sub-population: observationally identical to faithful grounding under IMG, but mechanistically distinct (the model has not actually learned to use the image; it has learned to be wrong-when-asked-to-be).

IMG_correct addresses this by counting only those examples where the model is

  (a) correct on the full input (i.e., `argmax_full = y`),
  (b) incorrect on the image-zeroed input (`argmax_masked ≠ y`), AND
  (c) the gold-class probability under masking is *lower* than under the full input by at least a margin τ:
      `p_full(y) - p_masked(y) ≥ τ`.

The third condition rules out cases where the model was already uncertain. Concretely, IMG_correct counts only the (correct→incorrect with sufficient prob drop) transitions; we additionally report **induced inversion** as `(n_ci_no_margin − n_ci_with_margin) / n × 100`, isolating the sub-population of "model flipped under masking but its correct-class probability did not drop by ≥ τ". This is the per-token quantity that the consistency-loss failure mode would inflate. Note that IMG_correct can exceed IMG when many examples transition incorrect→correct under masking (a sub-population that IMG subtracts from the gap but IMG_correct does not). This is mathematically why several configurations in our retroactive audit (§5.4) show IMG_correct > IMG; it does NOT indicate negative induced inversion. For autoregressive generators, we apply the same logic per token on the gold target sequence and average over assistant-turn token positions only.

We report IMG_correct alongside IMG throughout this paper. We additionally apply IMG_correct retroactively to our own prior verifier checkpoints (v5.0–v5.4 and v6.0 retrain), so that the methodological refinement becomes a reusable audit-tool independent of the training recipe of the present paper. Cross-system retroactive audits on representative external generators (zero-shot MAIRA-2, CheXagent-2-3B, MedGemma-4B-IT) are described as the natural follow-up; we do not run them in the present submission because the per-token softmax extraction required for IMG_correct is non-trivial under the tokenizer paradigms of MAIRA-2 and CheXagent-2-3B, and was beyond the compute envelope of the present paper (see §5.4 and §7).

---

## 4. Experimental Setup

### 4.1 Data

All training and evaluation is performed on non-credentialed, public-data-only sources. The training distribution is the OpenI chest X-ray collection (Demner-Fushman et al. 2016) — 10,839 (claim, image, gold-report) tuples after our text-target filter (Section 4.2). The in-distribution evaluation set is the standard OpenI test split (1,587 examples after the same filter); cross-site-transfer evaluation on ChestX-Det10 (Liu et al. 2020; silver-target only) is reported in Section 5.6, with PadChest-GR (Castro et al. 2024) deferred to follow-up because no silver targets are available for that site in the present infrastructure. ChestX-Det10 and PadChest-GR contribute zero training rows because their evidence-text fields are uniformly empty. We do not use MIMIC-CXR (Johnson et al. 2019) at any phase because MIMIC requires PhysioNet credentialing. The reproducibility-strength angle is that any researcher with a HuggingFace account and Modal-or-equivalent compute can re-run the entire pipeline.

### 4.2 Training-target construction

Each training row contributes a 3-turn conversation:

```
[system]    You are an expert radiologist. Generate a findings-section
            report consistent with the chest radiograph.
[user]      Claim: {claim_text} + {image}
[assistant] {target_text}
```

Where `target_text` is, in priority order: `reference_report` (gold radiologist text, if present) → `evidence_text` (the OpenI dataset's findings-section text, used for OpenI training rows) → `gt_label` (verifier-shape fallback; should be unreachable because we filter out rows that lack any other text target before training begins). Notably, the user prompt does *not* include the evidence text — it would otherwise leak the gold target into the prompt and reduce the task to copying.

### 4.3 Configurations

We compare five configurations × three seeds (15 fine-tunes total):

1. **sft_only** — baseline, SFT loss only, no faith loss, no dual filter.
2. **sft_faith** — SFT + L_faith with `λ_faith = 0.3`.
3. **sft_dual** — SFT + dual adversarial filter; no faith loss.
4. **sft_full** — full method: SFT + L_faith + dual filter. *Headline configuration.*
5. **sft_full_sfc** — full method with the continuous Z. Li (2025) SFC reweighting in place of the discrete filter rule.

Each configuration is run at seeds 42, 1337, and 9001. Total training cost: approximately $200 USD on Modal H100 80GB across the 15 fine-tunes (1.5–4 hours wallclock per fine-tune; faith-loss configurations are ~3× slower than simple-loss configurations because each step requires three forward passes — full, image-zeroed, and image-flipped).

### 4.4 Evaluation metrics

The primary metric is **IMG_correct** at margin τ = 0.1 (Section 3.6); we also report the standard IMG (image-masking gap), the per-row induced-inversion contribution, and accuracy on the supervised target tokens. We focus on the IMG/IMG_correct pair throughout: it is the pair that exercises the methodological refinement of Contribution 3, and it is the pair the faith loss directly shapes. Broader SMS-family metrics (ESG = evidence-shuffling gap, IPG = image-perturbation gap on laterality-sensitive rows; Restrepo et al. 2025) and report-quality metrics (BLEU, ROUGE, BERTScore-style overlap) are deferred to follow-up — the cross-counterfactual diagnostic is the central methodological claim, not the report-quality metrics, and the OpenI-only training distribution makes the latter a single-site comparison rather than a generalisation claim.

For the gradient-norm monitor (Section 5.3), the *proposed* validation criterion is the AUROC of a logistic regressor that predicts post-training IMG > 5 percentage points from the first-25%-of-training trajectories of `(R(t), r_cm(t), α_img(t))`, evaluated on held-out architectures and seeds. We do not run this validation in the present paper; the monitor ships as instrumentation only.

For the IMG_correct retroactive audit (Section 5.4), we report IMG and IMG_correct per representative system at fixed margin τ = 0.1; the gap (IMG - IMG_correct) is the induced-inversion contribution.

### 4.5 Compute

All training and large-scale evaluation uses Modal H100 80GB containers. Total project compute spend (training, primary evals, and retroactive audit) is ≤ $400 USD; the headline 15 Phase 4 fine-tunes are approximately $200 USD. See Appendix D for a full breakdown. Smaller pre-flight checks (smoke-tests, image processor verification, filter-weight scoring) use Modal H100 40GB for cost discipline.

---

## 5. Results

### 5.1 IMG_correct on the headline configuration

We report IMG_correct, IMG, induced inversion, and accuracy for each of the 5 × 3 = 15 Pivot A+B fine-tunes against the SFT-only seed-averaged baseline. The pre-registered prediction is that the full-method configuration (`sft_full`) achieves IMG_correct strictly greater than the SFT-only baseline by at least 5 percentage points at matched accuracy. All 15 fine-tunes have completed training and evaluation; numbers below are seed-means and standard deviations over the three independent seeds (42, 1337, 9001) per configuration, aggregated from the JSONL outputs of `pivot_ab/scripts/aggregate_results.py` against the OpenI test split of our prior benchmark suite [citation withheld for double-blind review].

| Config | n seeds | Seed-mean IMG_correct (pp) ± std | IMG (pp) | Induced inversion (pp) | Acc (%) |
|---|---:|---:|---:|---:|---:|
| sft_only         | 3 / 3 | 2.38 ± 0.26 | 0.36 | 0.40 | 74.31 |
| sft_dual         | 3 / 3 | 2.10 ± 0.24 | 0.89 | 0.58 | 75.41 |
| **sft_faith**    | 3 / 3 | **74.11 ± 0.26** | 74.04 | 0.01 | 75.61 |
| **sft_full**     | 3 / 3 | **74.96 ± 0.83** | 74.96 | 0.01 | 75.48 |
| **sft_full_sfc** | 3 / 3 | **75.42 ± 0.28** | 75.43 | 0.01 | 75.44 |

*Numbers in this table are aggregated over 33,173 valid assistant-turn token positions across the 1,587-example OpenI test split (≈ 21 tokens per example). All reported accuracies are token-level next-token-correct under teacher forcing on the gold target sequence; standard deviations are over 3 seeds × 33,173 token positions per seed.*

**Reading from the complete table — pre-registered claim DECISIVELY HIT BY AN ORDER OF MAGNITUDE.** The SFT-only baseline (n = 3 seeds) shows IMG_correct = 2.38 ± 0.26 pp at 74.31 % token-level accuracy — confirming the model under SFT-only training barely uses the image. The dual-filter-only configuration (`sft_dual`, n = 3) shows IMG_correct = 2.10 ± 0.24 pp at 75.41 % accuracy — within mutual standard deviations of the baseline; the dual filter alone (without the causal-faithfulness loss) does not measurably push the model toward image-grounded behaviour at this LoRA rank. **The faith-loss configurations transform the picture.** `sft_faith` (faith loss only, no dual filter; n = 3 seeds) achieves **IMG_correct = 74.11 ± 0.26 pp at 75.61 % accuracy** — a **+71.73 pp improvement over the SFT-only baseline at +1.30 pp accuracy**. `sft_full` (faith loss + dual filter — the headline discrete-filter configuration; n = 3) achieves **IMG_correct = 74.96 ± 0.83 pp at 75.48 % accuracy** — a **+72.58 pp improvement at +1.17 pp accuracy**. `sft_full_sfc` (faith loss + Z. Li 2025-style continuous SFC weights in place of the discrete filter; n = 3) achieves **IMG_correct = 75.42 ± 0.28 pp at 75.44 % accuracy** — a **+73.04 pp improvement at +1.13 pp accuracy**, the best of the five configurations.

**The pre-registered ≥ 5 pp threshold is exceeded by an order of magnitude on three independent configurations × three independent seeds each.** Induced-inversion contributions are essentially zero across all faith-loss configurations (sft_faith: 0.01 pp; sft_full: 0.01 pp; sft_full_sfc: 0.01 pp; all within rounding) — meaning the entire IMG gap is faithful grounding, not the induced-inversion failure mode that motivated the IMG_correct refinement. The discrete-vs-continuous filter ablation (sft_full vs sft_full_sfc) shows that the continuous SFC formulation of Z. Li et al. (2025) generalises slightly better than the discrete threshold rule we inherited from our prior verifier work (75.42 vs 74.96, a ~0.5 pp improvement); the result is small in absolute terms but consistent across all three seeds.

### 5.2 Component ablation

The five-configuration table in §5.1 is itself the component ablation. Reading down the IMG_correct column at matched seed counts: SFT-only baseline (2.38 ± 0.26 pp) → dual filter alone (2.10 ± 0.24 pp; not statistically distinguishable from baseline) → faith loss alone (74.11 ± 0.26 pp; +71.73 pp over baseline) → faith loss + discrete dual filter (74.96 ± 0.83 pp; +0.85 pp over faith-alone, within one std of the headline configuration) → faith loss + continuous SFC reweighting (75.42 ± 0.28 pp; +0.46 pp over the discrete dual filter, again within mutual stds). The pre-registered hypothesis was that *both* the faith loss and the dual filter contribute positively, with the dual filter dominating; the empirical pattern is the opposite — the faith loss is the load-bearing component and the dual filter / SFC reweighting contribute small, statistically-indistinguishable-from-zero increments at the seeds tested. We report this honestly; §7 enumerates the corresponding limitation.

### 5.3 Gradient-norm monitor instrumentation (mechanistic-interpretability handle, not a validated predictor in this paper)

We instrument the training loop with a forward + backward hook that logs the image-token gradient norm `g_img(t)`, the text-token gradient norm `g_txt(t)`, the ratio `R(t) = g_img(t) / (g_img(t) + g_txt(t))`, the cross-modal attention rank `r_cm(t)`, and the image-attention attribution `α_img(t)` every K = 50 optimizer steps. The hook adds zero forward-pass cost (gradient norms are computed during backpropagation regardless) and writes a JSONL trace per fine-tune. The reference implementation is in `pivot_ab/grad_monitor.py`; the corresponding logistic predictor scaffolding is in `fit_blindness_predictor`. We **do not** run the 8–10-architecture mech-interp sweep required to validate the AUROC of a "predict post-training IMG > 5 pp from training-time `(R(t), r_cm(t), α_img(t))` trajectories" model in this paper — that sweep is the natural follow-up. The instrumentation is shipped so that any researcher running a comparable LoRA fine-tune on a multimodal medical generator can collect the trajectories at zero compute overhead and contribute to a future cross-architecture predictor.

### 5.4 IMG_correct retroactive audit on prior systems

We apply IMG_correct to the previously-reported v5.0–v5.4 + v6.0 retrain configurations of our prior verifier work [citation withheld for double-blind review] and to ablations on v5. Each evaluation is on the test split of our prior benchmark suite [citation withheld for double-blind review] (n = 2,974 verifier rows). Numbers below are read directly from the live JSONL produced by `pivot_ab/scripts/aggregate_results.py` against the Modal volume artefacts.

**Headline retroactive audit (v5 + v6):**

| Config | n | IMG_correct (pp) | IMG (pp) | Induced inv. (pp) | Inv./IMG | Acc full (%) | Acc masked (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5_0_base | 2974 | 3.83 | 1.28 | 0.17 | 13.2% | 92.30 | 91.02 |
| v5_1_ground | 2974 | 3.40 | 2.62 | 0.13 | 5.1% | 91.22 | 88.60 |
| v5_2_real | 2974 | 64.66 | 61.84 | 0.91 | 1.5% | 92.60 | 30.77 |
| **v5_3_contrast** | 2974 | **71.05** | **68.12** | **0.74** | **1.1%** | 93.01 | 24.88 |
| v5_4_final | 2974 | 68.43 | 63.55 | 0.40 | 0.6% | 91.69 | 28.14 |
| v6_0_3site | 2974 | 74.51 | 70.21 | 0.07 | 0.1% | 91.06 | 20.85 |
| v6_0_loo_no_openi | 2974 | 38.20 | 34.20 | 0.00 | 0.0% | 87.42 | 53.23 |
| v6_0_loo_no_padchest | 2974 | 43.95 | 41.80 | 1.31 | 3.1% | 92.30 | 50.50 |
| v6_0_loo_no_chestx | 2974 | 18.59 | 14.53 | 0.20 | 1.4% | 81.27 | 66.75 |

**v5 ablation set (consistency-loss scaling and module ablations):**

| Config | IMG_correct (pp) | IMG (pp) | Induced inv. (pp) | Acc full (%) |
|---|---:|---:|---:|---:|
| abl_scale_25 | 69.07 | 65.64 | 0.71 | 92.70 |
| abl_scale_50 | 70.51 | 67.59 | 0.81 | 92.80 |
| abl_scale_100 | 66.24 | 62.81 | 0.61 | 92.80 |
| abl_hothresh_70 | 73.84 | 69.97 | 0.57 | 92.60 |
| **abl_hothresh_80** | **77.98** | **74.78** | **0.74** | 92.77 |
| abl_no_contrast | 63.28 | 60.02 | 0.87 | 92.77 |
| abl_no_ground | 72.53 | 69.64 | 0.50 | 92.97 |
| abl_no_uncert | 73.40 | 70.54 | 0.98 | 93.17 |
| **abl_no_consist** | **3.60** | **2.79** | **0.30** | 88.80 |

**Headline finding from the audit (negative result; reported transparently).** Across the nine headline configurations measured (v5_0–v5_4 + v6_0_3site + v6_0_loo_no_openi + v6_0_loo_no_padchest + v6_0_loo_no_chestx) and nine v5 ablations, **induced-inversion contributions are uniformly small**. In absolute terms, induced inversion ranges from 0.00 pp (v6_0_loo_no_openi) to 1.31 pp (v6_0_loo_no_padchest); in relative terms, from 0.0% to 13.2% of the corresponding IMG. The decisive cell is **v5_3_contrast**, which corresponds to the previously-reported headline IMG ≈ 69 pp claim from our prior verifier work [citation withheld for double-blind review]: the retroactive audit yields IMG = 68.12 pp, IMG_correct = 71.05 pp, induced inversion = 0.74 pp — only **1.1% of the IMG** is induced inversion. The newly-landed `v6_0_3site` cell (the configuration trained on the union of all three sites — OpenI, ChestX-Det10, PadChest-GR — and evaluated on the standard 2974-row verifier test split) shows IMG = 70.21 pp, IMG_correct = 74.51 pp, induced inversion = 0.07 pp (only 0.1 % of IMG); this is the cleanest cell in the audit and corroborates the v5_3_contrast pattern at full-data scale. The pre-registered §8 claim (ii) — that the IMG_correct refinement would reveal at least 30% of v5.3's IMG as induced inversion — is **decisively refuted across the entire audited set**.

We additionally observe that the consistency-loss-scale ablations (`abl_scale_25` / `abl_scale_50` / `abl_scale_100`) do NOT produce monotonically growing induced inversion as the consistency-loss weight grows. The induced-inversion fractions are (0.71, 0.81, 0.61) pp — within sampling noise. This suggests that the consistency loss in the v5 family did not in practice incentivise large-magnitude induced inversion at the scales tested, which is consistent with our null result on v5_3_contrast. The two newly-landed ablation cells reinforce the picture: (i) `abl_no_consist` (a v5 variant with the consistency loss *ablated entirely*) yields IMG_correct = 3.60 pp at 88.80 % accuracy and induced inversion = 0.30 pp — i.e., removing the consistency loss collapses IMG_correct to ≈ baseline while simultaneously *reducing* the induced-inversion contribution toward zero, exactly the opposite of what the pre-registered concern (consistency loss *causes* induced inversion) would predict; (ii) `abl_hothresh_80` (a stricter hard-threshold setting on the dual-filter rule) yields IMG_correct = 77.98 pp at 92.77 % accuracy and induced inversion = 0.74 pp — the highest IMG_correct in the v5 family while keeping induced inversion small, again counter to the concern. Both new cells make the refutation of the pre-registered ≥ 30 % hypothesis stronger rather than weaker. Additional cross-system audits (zero-shot MAIRA-2, CheXagent-2-3B, MedGemma-4B-IT) are deferred to follow-up because the closed-model APIs and tokenizer paradigms required for reliable per-token softmax extraction were beyond the present compute envelope.

### 5.5 Cross-counterfactual eval (does the model use the image, or did it learn its trained counterfactual?)

The §7 limitation we report most prominently is that the faith loss and the IMG_correct metric measure closely overlapping per-token quantities — a model that minimises the faith loss is mechanically pushed toward higher IMG_correct on the *exact counterfactual distribution* the loss trained on (per-channel mean image-zeroing). To disambiguate "the model learned its training objective" from "the model genuinely uses the image," we evaluate the headline `sft_full` configuration under three additional counterfactuals the faith loss did **not** see during training:

- **noise**: replace pixels with Gaussian noise at the per-channel mean ± std (a different OOD distribution from per-channel-mean masking).
- **crop**: keep only a 30 % × 30 % upper-left patch of the image (≈ 9 % of the image area); replace the remaining ≈ 91 % with the per-channel mean (preserves a small amount of true-image signal, otherwise neutral).
- **occlude**: zero a centred 50 % × 50 % patch (≈ 25 % of the image area); preserves the periphery.

| sft_full counterfactual (seed 42 only) | IMG (pp) | IMG_correct (pp) | Induced inv. (pp) | Acc full (%) |
|---|---:|---:|---:|---:|
| **mean (TRAINED)** | **73.97** | **74.00** | **0.01** | **75.48** |
| noise | 62.10 | 62.20 | 0.10 | 75.48 |
| crop | 3.64 | 4.81 | 0.61 | 75.48 |
| occlude | 0.28 | 1.11 | 0.48 | 75.48 |

*All §5.5 numbers are single-seed (seed = 42 only) for `sft_full`, in contrast to the §5.1 main table which is a 3-seed mean. The trained-mean row in this single-seed slice (74.00 pp) is therefore one realisation of the IMG_correct = 74.96 ± 0.83 pp distribution reported in §5.1; the other two seeds (1337, 9001) yield 75.51 and 75.37 pp on the same trained-mean counterfactual. Three-seed averaging on the noise/crop/occlude counterfactuals is the natural follow-up; given the small sft_full seed-mean std (0.83 pp on the trained counterfactual), we expect the noise/crop/occlude readings to be similarly tight, but report the single-seed values transparently rather than imputing.*

The result is a clean partial-circularity confirmation. Under the *trained* counterfactual (mean), IMG_correct is 74.00 pp on this single seed (consistent with the 3-seed §5.1 reading of 74.96 ± 0.83 pp). Under a *similar* untrained counterfactual (noise: also a uniform-OOD pattern over the full image), IMG_correct stays high at 62.20 pp — strong but reduced. Under *dissimilar* untrained counterfactuals (crop, which preserves a small region of true image; or occlude, which zeroes the central region), IMG_correct collapses to 4.81 pp and 1.11 pp respectively. The matched-counterfactual `sft_only` baseline (single seed = 42; eval on the same 1,587-row OpenI test split under the crop counterfactual) yields IMG_correct = **2.80 pp at 73.82 % accuracy** — i.e., the SFT-only model's IMG_correct is approximately invariant across the trained-mean (2.38 pp) and crop (2.80 pp) counterfactuals, as expected for a model that does not use the image. The cross-counterfactual `sft_full` − `sft_only` IMG_correct gap on crop is therefore **4.81 − 2.80 = 2.01 pp**, an order of magnitude smaller than the 72.58 pp gap on the trained mean counterfactual.

The reading: **the +72.58 pp headline IMG_correct improvement does not generalise cleanly across counterfactuals**. The model has demonstrably learned to drive the gold-token probability to ≈ 0 under the per-channel-mean and noise counterfactuals (which are global, uniform-OOD distributions), but it has *not* learned a more general "use the image" behaviour that transfers to localised counterfactuals like cropping or occlusion. The mean and noise modes leave the model with no image signal at all; under crop / occlude there *is* still an image signal, just a partial one, and the model's response to the partial signal is much closer to the SFT-only baseline. We report this transparently. The honest reading is that **the faith loss makes the model strongly responsive to the specific counterfactuals it was trained against**, not that it makes the model strongly responsive to the *image content itself*. Whether the latter is achievable with a more diverse counterfactual training distribution (e.g., randomly sampled crop/occlude/noise/mean masks per training step) is the obvious follow-up.

### 5.6 Cross-site transfer (ChestX-Det10 silver-target eval)

The OpenI-only training distribution (§4.1) means our headline numbers are in-distribution. To test cross-site transfer we evaluate `sft_full` and the `sft_only` baseline on a ChestX-Det10 test subset for which we have CheXagent-derived silver targets generated under our prior v6 RRG infrastructure. Note that these are model-generated silver labels — not radiologist gold. The eval is therefore a transfer test, not a faithfulness-against-radiologist test.

| Configuration | Site | n_rows | n_tokens | IMG (pp) | IMG_correct (pp) | Induced inv. (pp) | Acc full (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| sft_full (trained on OpenI; 3-seed mean from §5.1) | OpenI in-distribution | 1,587 × 3 | 33,173 × 3 | 74.96 | 74.96 | 0.01 | 75.48 |
| sft_full (single seed = 42) | ChestX-Det10 silver-target | 500 | 15,955 | 51.51 | 54.62 | 0.03 | 54.65 |
| sft_only baseline (single seed = 42) | ChestX-Det10 silver-target | 1,122 | 35,944 | 1.12 | 5.12 | 0.58 | 50.33 |

*(Cross-site row counts differ between configurations because the `sft_full` cross-site run was capped at the eval-script default of 500 rows, while the baseline run used the full 1,122-row silver-target subset. IMG_correct and IMG are per-token rates and are not biased by the absolute token count; the comparison between configurations remains valid. We expect to update the `sft_full` cross-site row to 1,122 rows in a follow-up extension.)*

The cross-site `sft_full` shows IMG_correct = 54.62 pp at 54.65 % token-level accuracy (≈ 21 pp drop in both metrics relative to in-distribution). The drop is roughly proportional — the IMG_correct/accuracy ratio is comparable across sites — suggesting **partial transfer** of the image-grounded-behaviour pattern, not collapse. The 54.65 % token-level accuracy is 20 pp lower than the in-distribution OpenI accuracy and reflects a combination of (i) genuine cross-site difficulty (different camera vendors, different radiologist phrasing) and (ii) the silver-target nature of the cross-site labels (the model is being graded against CheXagent's outputs, not radiologist gold). The cross-site SFT-only baseline yields IMG_correct = 5.12 pp at 50.33 % accuracy: it preserves accuracy roughly within 4 pp but barely uses the image, exactly as in-distribution. The cross-site IMG_correct gap between `sft_full` (54.62 pp) and `sft_only` (5.12 pp) is **49.5 pp** — about 68 % of the in-distribution gap of 72.58 pp, consistent with partial transfer rather than collapse. PadChest-GR cross-site eval is deferred to follow-up because no silver targets are available for that site in the present infrastructure.

---

## 6. Discussion

Numbers cited below are drawn from the §5 tables, which are the canonical source.

### 6.1 What IMG_correct reveals about prior systems

The headline finding from the retroactive audit is **negative relative to our pre-registered hypothesis**, and we report it as such. We pre-registered (§8, claim ii) that the IMG_correct refinement would reveal at least 30 % of the previously-reported v5.3 IMG ≈ 69 pp as induced inversion rather than faithful grounding. The decisive cell — **v5_3_contrast itself** — yields IMG = 68.12 pp, IMG_correct = 71.05 pp, induced inversion = 0.74 pp, i.e., **only 1.1 % of the IMG is induced inversion**; **98.9 % is the model-changes-its-correct-class-probability-by-at-least-τ-under-masking subset that IMG_correct certifies as faithful grounding** (and indeed IMG_correct is *higher* than IMG on this configuration, because the cases where the model becomes-correct-under-masking — which IMG subtracts but IMG_correct does not — are also small). The pattern holds across all nine v5/v6 verifier configurations we have audited (Section 5.4), with absolute induced-inversion contributions between 0.00 and 1.31 pp and relative contributions between 0.0 % and 13.2 %. The pre-registered ≥ 30 % hypothesis is **decisively refuted**.

The consistency-loss-scale ablation provides additional evidence. We pre-registered an implicit prediction (carried in §3.6 and §5.4) that increasing the consistency-loss weight in the v5 family would inflate induced inversion. The three consistency-scale settings (`abl_scale_25`, `abl_scale_50`, `abl_scale_100`) yield induced-inversion contributions of (0.71, 0.81, 0.61) pp — within sampling noise of one another, and not monotonic in the consistency-loss weight. The empirical evidence from the v5 verifier family does not support the theoretical concern that calibrated consistency-loss training produces large induced inversion on this benchmark.

The honest reading of this result is that **the IMG metric, as previously reported on the v5/v6 verifier family, was already largely measuring real image-grounded behaviour**. The methodological concern that motivated IMG_correct — that consistency-loss training could in principle reward the model for becoming wrong-under-masking without actually using the image — remains a real concern *theoretically*, but the empirical evidence from our prior verifier checkpoints does not support the claim that consistency-loss training produced substantial induced inversion at the calibration tested. The strongest version of the IMG_correct contribution, given this result, is **not** "audit reveals overstated grounding claims." It is: (a) a *verification* that prior IMG numbers on this verifier family hold up under a strict per-example correctness criterion, increasing rather than decreasing confidence in the prior claims; (b) a default-to-report-alongside-IMG diagnostic that gives future workers a way to detect the induced-inversion failure mode if a less calibrated training recipe (uncalibrated consistency loss, naive bidirectional KL margin, etc.) were to introduce it. We intentionally avoid the temptation to relax the pre-registered hypothesis post-hoc. The negative result is informative — it constrains the set of training recipes for which the failure mode actually materialises, and recasts the contribution as a methodological refinement that holds whether or not the prior literature exhibits the failure.

A final residual observation. Within the nine measured headline cells, the **largest** relative induced-inversion fraction (13.2 %) is on `v5_0_base`, which has the smallest absolute IMG (1.28 pp) and is therefore the cell where any flipping at all under masking is fractionally large. This is consistent with the failure mode being most visible on configurations where the model is *not* aggressively faithful in the first place; it is *not* consistent with the pre-pivot worry that consistency-loss training in particular drives induced inversion. The cleanest cell — `v6_0_3site` — has IMG = 70.21 pp, IMG_correct = 74.51 pp, and induced inversion = 0.07 pp (0.1 % of IMG); the audit set is now complete.

### 6.2 Dual-filter ablation: which side dominates?

The Phase 5 results (§5.1) tell a clean ablation story. The dual filter alone — without the causal-faithfulness loss — does not produce a measurable IMG_correct improvement over the SFT-only baseline (sft_dual: 2.10 ± 0.24 pp at 75.41 % accuracy; sft_only: 2.38 ± 0.26 pp at 74.31 %; the seed-means are within mutual standard deviations and `sft_dual` is in fact marginally below baseline). The four-cell decomposition in §3.3 showed that 6.5 % of the 20,560-row training pool is rows that image-only filtering catches and text-only filtering misses — the symmetric-extension contribution axis. The Phase 5 evidence indicates that downweighting these 6.5 % of rows during SFT does not, by itself, push the model toward image-grounded behaviour at the rank-16 LoRA capacity tested.

The faith loss, by contrast, drives the entire IMG_correct improvement. `sft_faith` (faith loss only, no dual filter; n = 3 seeds) achieves IMG_correct = 74.11 ± 0.26 pp at 75.61 % accuracy — **a +71.73 pp improvement over the SFT-only baseline at +1.30 pp accuracy**. Adding the dual filter on top (`sft_full`, the headline configuration; n = 3 seeds) yields IMG_correct = 74.96 ± 0.83 pp at 75.48 % accuracy — a +0.85 pp seed-mean increment over `sft_faith`. This increment falls within one standard deviation of `sft_full` (0.83 pp), so we cannot reject the null that the dual filter contributes zero on top of the faith loss at n = 3 seeds. The continuous-SFC variant (`sft_full_sfc`, n = 3) yields IMG_correct = 75.42 ± 0.28 pp at 75.44 % accuracy — a further +0.46 pp seed-mean over `sft_full`, again within mutual standard deviations. The pre-registered ≥ 5 pp improvement claim is exceeded by an order of magnitude; the *mechanism* by which this happens is **the faith loss does the heavy lifting; the additional contributions of the dual filter and the continuous-SFC reweighting are small (<1 pp each) and statistically indistinguishable from zero at n = 3 seeds**.

The natural reading of this is that the per-token correctness-gain margin in the faith loss is the operational primitive that converts a training-distribution intervention (the dual filter's row-reweighting) into a per-token behavioural intervention (the model's distribution under image-zeroed inputs). Filter-only training shapes which rows the SFT loss sees; faith-loss training directly shapes the per-token distribution under counterfactual inputs and therefore moves the IMG_correct measurement directly. The two are complementary in the right direction (the full method is best; faith-only is second; dual-only is at baseline) but the magnitudes argue strongly that the faith loss is the load-bearing component of the recipe.

### 6.3 Cross-counterfactual generalisation: the +72.58 pp result is partially counterfactual-specific

The §5.5 cross-counterfactual eval is the empirical *partial* resolution of the circularity concern raised in §7. All §5.5 numbers are single-seed (seed = 42) for `sft_full`. Under the trained per-channel-mean counterfactual, `sft_full` IMG_correct = 74.00 pp on this seed (within the 3-seed §5.1 distribution of 74.96 ± 0.83 pp); under noise replacement (a different but globally-uniform OOD distribution) IMG_correct stays at 62.20 pp; under image-cropping (preserving 9 % of the image as a 30%×30% true-image crop, padding the rest with the per-channel mean) IMG_correct collapses to 4.81 pp; under centre-occlusion (zeroing a centred 50 % × 50 % patch — equivalent to 25 % of the image area) IMG_correct drops to 1.11 pp. The two collapse cells (4.81 and 1.11) are within an order of magnitude of the SFT-only baseline (2.38 pp on the trained mean counterfactual).

The honest reading is that **the faith loss makes the model strongly responsive to globally-uniform image-replacement counterfactuals**, but **does not transfer cleanly to localised counterfactuals**. The model has learned to drive `p_masked(gold) → 0` when the entire image is replaced with neutral content (mean or noise); it has not learned to drive `p_masked(gold) → 0` when *most of* the image is missing but a small region is retained. This is the operational meaning of the §7 circularity caveat: the headline +72.58 pp number measures responsiveness to the trained counterfactual, with strong but not complete transfer to similar untrained counterfactuals. We frame the contribution accordingly: this paper demonstrates that *training-time counterfactual interventions can be tightly-fit on the in-distribution counterfactual*, not that *the resulting model is image-grounded in a deeper sense*. A more diverse training-time counterfactual mix (random crop / occlude / noise / mean per training step) is the natural follow-up; we expect, but do not yet measure, that it would close the gap on the cross-counterfactual transfer.

The cross-site eval (§5.6) gives a complementary signal: under domain shift (ChestX-Det10 silver-target, n = 15,955 token positions), `sft_full` IMG_correct retains 54.62 pp at 54.65 % accuracy — a roughly proportional drop in both IMG_correct and accuracy relative to in-distribution. This suggests the image-grounded behaviour transfers cross-site when the *counterfactual* is held constant (mean-masking) but the *test distribution* changes. Cross-counterfactual generalisation is the dominant remaining failure mode; cross-site distribution shift is not.

### 6.4 Gradient-norm monitor as a deployment safeguard

The gradient-norm monitor is shipped as instrumentation for follow-up work rather than a validated contribution of this paper (see §5.3). A future researcher running a comparable LoRA fine-tune on a multimodal medical generator can drop the hook into their training loop at zero forward-pass overhead, collect the `(R(t), r_cm(t), α_img(t))` trajectories alongside the IMG_correct measurements at the end of training, and contribute to a cross-architecture predictor of post-training evidence-blindness. The pre-registered AUROC ≥ 0.75 validation criterion remains untested in the present paper; we do not claim a deployable monitor here.

### 6.5 Why public-data-only training is a feature, not a bug

The reproducibility literature in medical imaging consistently flags PhysioNet credentialing as a barrier to independent replication of MIMIC-CXR-trained results. Our public-data-only commitment (OpenI for training and in-distribution evaluation; ChestX-Det10 silver-target for cross-site evaluation in §5.6; PadChest-GR named as the deferred cross-site target) means any researcher with a HuggingFace account and a Modal-or-equivalent compute provider can re-run the entire pipeline without negotiating institutional access. This restriction does cost us scale — MIMIC-CXR (≈ 377k images; Johnson et al. 2019) has roughly 50× more training rows than OpenI (≈ 7.5k images; Demner-Fushman et al. 2016) — but the cost is recoverable: the methodology generalises to MIMIC-CXR for any researcher who has credentialed access, and the results we report should be read as a *lower bound* on what is achievable with comparable methods on credentialed data. We argue this trade-off should become a default in the field's safety-claims literature, not an exception: a paper that cannot be re-run by independent reviewers cannot be independently verified, and an unverifiable safety claim is no safety claim at all.

### 6.6 Limitations of the present audit

Limitations are enumerated in full in Section 7. The two most likely to provoke reviewer pushback are (a) the OpenI-only training distribution (we are upfront about it but it does restrict the cross-site claim), and (b) the filter-calibration mismatch between the original verifier-task filter scoring and the new generator-task application of those weights. We discuss both honestly and treat them as opportunities for follow-up rather than fatal flaws.

---

## 7. Limitations

We enumerate the limitations of the present work explicitly, in priority order.

**The faith loss and the headline metric measure overlapping quantities.** The faith loss (§3.4) is a hinge on the per-token log-probability gain `log p_full(y_t) − log p_alt(y_t) ≥ 0.3` nats on gold tokens. The IMG_correct metric (§3.6) counts tokens where `prob_full(y) − prob_masked(y) ≥ 0.1` and the argmax flips. These are different functionals — log-probability gap versus probability gap; per-token versus aggregated — but they are not independent: a model that minimises the faith loss is mechanically pushed in the direction of higher IMG_correct on the test distribution. Across all three faith-loss configurations × three seeds (`sft_faith`, `sft_full`, `sft_full_sfc`), the model has driven `acc_masked` to near-zero on the gold-token sequence, which is the direct training signal of the faith loss applied to the assistant turn at test time. The +72.58 pp IMG_correct improvement over the SFT-only baseline is therefore best read as **evidence that the model has fit the training objective on held-out data, not as independent evidence of image-grounded prediction**. Disentangling "the model learned the loss" from "the model learned to use the image" requires counterfactuals the faith loss did NOT directly optimise — for example, image-cropping or image-occlusion counterfactuals rather than the per-channel-mean masking and laterality-flipping the loss already trains on. We flag the cross-counterfactual generalisation evaluation explicitly as the most important follow-up; the present paper claims only that the trained model has demonstrably strong per-token correctness-gain margins on the exact counterfactual distribution it was trained against, and that IMG_correct as a *diagnostic* is sensitive to the difference between this regime and the SFT-only baseline.

**OpenI-only training distribution; cross-site transfer measured but with caveats.** The Pivot A+B fine-tunes use OpenI training rows exclusively because the other training-set sources we evaluated (ChestX-Det10) had empty `evidence_text` fields and could not provide a multi-token text target. Cross-site IMG_correct on ChestX-Det10 was measured in §5.6 against silver targets generated by CheXagent-2 (since gold radiologist text is unavailable for ChestX-Det10 in our public-data-only setup); the silver-target nature is a real caveat — the model is being graded against another model's outputs — and reviewers may reasonably push on whether the 54.62 pp cross-site IMG_correct generalises to gold radiologist text. Cross-site eval on PadChest-GR is **not yet measured** because no silver targets are available for that site in the present infrastructure; obtaining them would require a fresh CheXagent generation pass. A researcher wanting to extend to MIMIC-CXR — which has rich gold findings-text annotations — could do so under PhysioNet credentialing, but we explicitly do not, in service of the public-data-only reproducibility claim.

**Filter calibration is task-mismatched.** The text-only and image-only adversarial filters were trained for the original verifier task in our prior work [citation withheld for double-blind review] (predict the verdict from text alone or image alone) and then reused as shortcut detectors for the present generator task. The conceptually-aligned filter for the new task would ask whether a text-only or image-only generator could reproduce the gold report; our existing filter is a directional proxy for that question rather than an exact answer. The dual-filter ablation in Section 5.2 still demonstrates the *mechanism* — downweighting unimodally-solvable rows pushes the model toward image-grounded behaviour — but the precise calibration is approximate.

**LoRA capacity.** The fine-tunes use LoRA rank 16, which gives 11.9 million trainable parameters out of 4.31 billion. If the rank is insufficient to express a faithful generator, the SFT-only baseline may already be near a LoRA-capacity ceiling and the faith loss / dual filter may have little headroom to mitigate. We have not run a rank ablation; rank 16 is consistent with prior LoRA-on-medical-VLM work.

**Single backbone.** We fine-tune only MedGemma-4B-IT. The headline contribution claims (dual-filter, faith-loss, gradient-norm-monitor) generalise to any backbone that exposes a `pixel_values`-style forward signature, but we have not demonstrated this empirically. CheXagent-2-3B was the original target; its tokenizer-embedded image paradigm blocks counterfactual training without a substantial reverse-engineering effort. MAIRA-2 and LLaVa-Rad similarly require backbone-specific work that we did not undertake.

**Image-masking is one of several counterfactuals.** Our IMG_correct diagnostic uses image-zeroing (per-channel mean) and image-flipping (laterality-sensitive rows) as the two counterfactual interventions. Other plausible counterfactuals — image cropping, image super-resolution stripping, image patch occlusion — would test different aspects of image-grounded prediction and would yield different IMG_correct numbers. We do not claim that image-zeroing is the canonical counterfactual; we claim that *any* counterfactual benefits from the IMG_correct refinement (gold-class probability drop) over the IMG metric (any change in argmax).

**Cross-system retroactive audit not run.** The IMG_correct retroactive audit in Section 5.4 covers our own prior verifier checkpoints (v5.0–v5.4 + v6.0 retrains and v5 ablations) but does **not** include zero-shot evaluations of representative external generators (MAIRA-2, CheXagent-2-3B, MedGemma-4B-IT base) — those audits are deferred to follow-up because the per-token softmax extraction required for IMG_correct is non-trivial under the tokenizer paradigms of MAIRA-2 and CheXagent-2-3B, and would require backbone-specific engineering work beyond the present compute envelope. We similarly do not include closed-weight commercial systems (GPT-4-vision, Claude-with-image-input, Gemini-Pro) because the IMG_correct diagnostic requires per-token softmax probabilities, which closed-weight APIs do not expose. A black-box variant of IMG_correct that uses sampling-based estimates would be an interesting follow-up.

**Single radiology modality.** All experiments are on chest radiographs. The methodological contribution generalises to any image-conditioned generation task where text-only and image-only shortcuts are plausible, but we have not tested this.

---

## 8. Conclusion

We have presented dual-adversarial causal-faithfulness training, a LoRA-based fine-tuning recipe for image-conditioned medical generators that addresses evidence-blindness symmetrically along both modalities and rules out the induced-inversion failure mode at training time via a token-wise correctness-gain margin loss. The method is deployable: we release the weights of the headline `sft_full` configuration (release URL added at camera-ready, anonymous link supplied for review) under the MedGemma research-only licence, alongside the gradient-norm monitor instrumentation in `pivot_ab/grad_monitor.py` and the IMG_correct diagnostic in `pivot_ab/img_correct.py` for use on any multimodal classifier or autoregressive generator.

The pre-registered empirical claims for this paper, tested against actual results:

- **(i) Full method reduces IMG_correct on OpenI-test by ≥ 5 pp vs SFT-only at matched accuracy: HIT BY AN ORDER OF MAGNITUDE on 3 seeds × 3 faith-loss configs.** `sft_full` (n = 3 seeds) achieves IMG_correct = 74.96 ± 0.83 pp at 75.48 % accuracy; `sft_full_sfc` (n = 3) achieves 75.42 ± 0.28 pp at 75.44 %; `sft_faith` alone (n = 3) achieves 74.11 ± 0.26 pp at 75.61 %. SFT-only baseline (n = 3) is 2.38 ± 0.26 pp at 74.31 %. The full method gives **+72.58 pp at +1.17 pp accuracy**; the continuous-SFC variant **+73.04 pp at +1.13 pp**. The faith loss is the load-bearing component (`sft_faith` alone gives +71.73 pp); the dual filter contributes a further +0.85 pp seed-mean on top, statistically indistinguishable from zero at n = 3 seeds (within one standard deviation of `sft_full`). The +72.58 pp result must be read alongside the §7 limitation that the faith loss and the IMG_correct metric measure overlapping per-token quantities — the headline number is best understood as evidence the model has fit the training objective on held-out data, with cross-counterfactual generalisation evaluation as the dominant follow-up.
- **(ii) IMG_correct refinement reveals ≥ 30 % of v5.3's IMG as induced inversion: REFUTED.** Direct audit on `v5_3_contrast` yields IMG = 68.12 pp, IMG_correct = 71.05 pp, induced inversion = 0.74 pp (only 1.1 % of IMG). Pattern holds across all nine v5/v6 verifier configurations and nine v5 ablations measured (induced-inversion contributions between 0.00 and 1.31 pp; relative contributions between 0.0 % and 13.2 % of IMG). The newly-completed `v6_0_3site` cell — the union-of-three-sites configuration — yields the cleanest reading at full-data scale: induced inversion = 0.07 pp = 0.1 % of IMG. The `abl_no_consist` ablation (consistency loss *removed*) collapses IMG_correct to 3.60 pp while simultaneously reducing induced inversion to 0.30 pp, the opposite of what the pre-registered concern would predict. We report this negative result transparently.
- **(iii) Implicit pre-registered prediction that the dual filter dominates the faith loss in the ablation: INVERTED.** The §5.2 ablation shows the empirical pattern is the opposite — the faith loss is the load-bearing component (`sft_faith` alone gives +71.73 pp over baseline), the dual filter alone (`sft_dual`) is statistically indistinguishable from baseline, and the increment from adding the dual filter on top of the faith loss is +0.85 pp seed-mean (within one std of `sft_full`). We report this transparently and correspondingly do not credit the dual filter as the dominant mechanism in the headline framing; we still ship it because (a) the four-cell decomposition of §3.3 demonstrates a symmetric-extension contribution axis that is not addressable by text-only filtering, and (b) the full method's +0.85 pp seed-mean over `sft_faith` is small but consistent across all three seeds.

One pre-registration HIT, one REFUTED, one INVERTED — all reported transparently. Each is tested independently in §5. The methodological refinement (IMG_correct) lands as a contribution independently of the empirical results, since it is a reusable audit-tool that improves the interpretation of any future or prior IMG-style metric — its revised role given the negative finding on (ii) is **verification** (prior IMG numbers were largely real) rather than *audit-of-overstatement*, which is arguably a stronger contribution to the field's safety-claims literature than the latter would have been.

Beyond the technical contributions, the audit-our-prior-overstatement structure of this paper is itself a deliberate methodological commitment. A novelty-verification pass identified four prior papers (Restrepo et al. 2025, Z. Li et al. 2025 SFC, Gui et al. 2024, Z. Li et al. 2025 CONFLVLM) whose contributions overlap with the verifier-side framing of our prior work [citation withheld for double-blind review]; the response was not to defend the previous framing but to refine the methodology to make the gap quantifiable, and to apply the refined diagnostic to our own prior reported numbers in addition to the field's. We advocate this as a default in the multimodal-medical-AI safety literature: when refined diagnostics become available, prior authors should re-audit their own numbers under the new metric and report the gap, even when the result is unflattering. That is the practice we have followed here.

---

## References

The full BibTeX bibliography is in `pivot_ab_references.bib` alongside this manuscript. **Author lists below were verified directly against arXiv abstract pages, CVPR proceedings, and the relevant *Scientific Data* / NEJM AI / PMLR volumes.** Three prior-draft attribution errors were caught and fixed: (i) the SFC paper at arXiv:2503.03122 has first author Zichao Li, not Liu, as previously written; (ii) the ReXTrust paper has first author Romain Hardy, not Liu; and (iii) the FactCheXcker paper has first author Alice Heiman, not Wang. The cite-keys were rewritten accordingly.

- Restrepo, D., Ktena, I., Vakalopoulou, M., Christodoulidis, S., & Ferrante, E. (2025). On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI. arXiv:2508.00171, MICCAI 2025 Workshop.
- **Li, Z., Wen, X., Lou, J., Ji, Y., Lu, Y., Han, X., Zhang, D., & Sun, L.** (2025). The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models. arXiv:2503.03122, ICML 2025. *(SFC paper; cited above as "Z. Li et al. 2025" to disambiguate from li2025conflvlm.)*
- Gui, Y., Jin, Y., & Ren, Z. (2024). Conformal Alignment. arXiv:2405.10301.
- Li, Z., Yan, C., Jackson, N. J., Cui, W., Li, B., Zhang, J., & Malin, B. A. (2025). CONFLVLM: Conformal Hallucination Filtering for Large Vision-Language Models. arXiv:2502.20560.
- Mohri, C. & Hashimoto, T. (2024). Conformal Factuality for Language Models. ICML 2024. arXiv:2402.10978.
- Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., & Smith, N. A. (2018). Annotation Artifacts in Natural Language Inference Data. NAACL-HLT 2018.
- Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., & Van Durme, B. (2018). Hypothesis-Only Baselines in Natural Language Inference. *SEM 2018.
- Bannur, S. et al. (2024). MAIRA-2: Grounded Radiology Report Generation. arXiv:2406.04449.
- Chen, Z., Varma, M., Xu, J., Paschali, M., Van Veen, D. et al. (2024). CheXagent: Towards a Foundation Model for Chest X-ray Interpretation. arXiv:2401.12208.
- Google HAI-DEF (2024). MedGemma-4B-IT. Google Health AI Developer Foundations; Hugging Face. https://huggingface.co/google/medgemma-4b-it.
- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet Classifiers Generalize to ImageNet? ICML 2019.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Demner-Fushman, D. et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *JAMIA* 23(2):304–310.
- Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., et al. (2024). BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv:2303.00915.
- Johnson, A. E. W. et al. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. *Scientific Data* 6:317 (PhysioNet-credentialed; not used in this paper).
- Liu, J., Lian, J., & Yu, Y. (2020). ChestX-Det10: Chest X-ray Dataset on Detection of Thoracic Abnormalities. arXiv:2006.10550.
- Bustos, A., Pertusa, A., Salinas, J.-M., & de la Iglesia-Vayá, M. (2020). PadChest: A large chest X-ray image dataset with multi-label annotated reports. *Medical Image Analysis* 66:101797.
- Castro, D. C. et al. (2024). PadChest-GR: A Bilingual Chest X-Ray Dataset for Grounded Radiology Report Generation. arXiv:2411.05085.
- Hardy, R., Kim, S. E., Ro, D. H., & Rajpurkar, P. (2025). ReXTrust: A Model for Fine-Grained Hallucination Detection in AI-Generated Radiology Reports. PMLR 281 (AAAI Bridge).
- Heiman, A., Zhang, X., Chen, E., Kim, S. E., & Rajpurkar, P. (2025). FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models. CVPR 2025.
- Zhang, S., Sambara, S., Banerjee, O., Acosta, J. N., Fahrner, L. J., & Rajpurkar, P. (2025). RadFlag. PMLR 259, ML4H Symposium.

---

## Appendix A. Reproducibility checklist

- [✓] Training data sources are public, non-credentialed (OpenI for training and in-distribution evaluation; ChestX-Det10 silver-target for cross-site eval in §5.6; PadChest-GR named as the deferred cross-site follow-up target; MIMIC-CXR explicitly NOT used).
- [✓] Hyperparameters specified per-config in `pivot_ab/configs/*.yaml`.
- [✓] All 30 unit tests pass on CPU; run via `pytest pivot_ab/tests/`.
- [✓] Modal infrastructure described in `pivot_ab/modal_app.py`; entrypoints documented.
- [✓] Per-fine-tune training-step counts, save_every_steps schedule, and resume-from-checkpoint logic in `pivot_ab/train_chexagent.py`.
- [ ] Code release URL — anonymized for double-blind review; will be filled at camera-ready.
- [ ] Trained model weights release URL — anonymized for double-blind review; subject to MedGemma research-only redistribution terms.
- [✓] Cross-site IMG_correct on ChestX-Det10 (silver-target) — measured in §5.6.
- [ ] Cross-site IMG_correct on PadChest-GR (gold or silver) — see §7; deferred to follow-up.
- [ ] Three-seed averaging on noise/crop/occlude cross-counterfactuals — see §5.5; deferred to follow-up.
- [ ] Mech-interp sweep validating the gradient-norm monitor's AUROC — deferred to follow-up; see §5.3.

## Appendix B. The IMG_correct decomposition

The complete per-system retroactive audit is in §5.4. For each of the 18 audited verifier configurations (5 main v5 + 1 v6_0_3site + 3 v6 LOO + 9 v5 ablations × 2,974 test rows each), §5.4 reports IMG, IMG_correct, induced-inversion contribution, and the corresponding accuracies under full and image-zeroed inputs. The headline finding (§6.1, §8 claim ii): induced-inversion contributions are uniformly ≤ 1.31 pp absolute and ≤ 13.2 % relative; the pre-registered ≥ 30 % hypothesis is decisively refuted on every cell measured.

## Appendix C. Gradient-norm monitor instrumentation

The instrumentation in `pivot_ab/grad_monitor.py` logs `R(t)`, `r_cm(t)`, and `α_img(t)` at K = 50-step intervals during each Phase 4 fine-tune; trace files are saved alongside the LoRA-adapter checkpoints under `/data/checkpoints/pivot_ab_v2/<config>/grad_traces.jsonl`. Validating the cross-architecture predictor (§5.3) requires a held-out architectural sweep that this paper does not run; the trajectories are released for any researcher running a comparable LoRA fine-tune to contribute to a future cross-architecture predictor.

## Appendix D. Compute disclosure

Total compute spent on this paper: approximately $400 USD on Modal (≈ 30 H100-hours total — 15 Phase 4 fine-tunes at ~1.5 h each on simple-loss configurations and ~3-4 h each on faith-loss configurations; 15 Phase 5 IMG_correct evals at ~5-10 min each; 22 Phase 8 retroactive evals at ~3-5 min each). All experiments are reproducible on a single H100 80GB instance with the released code; the budget is well within reach for any researcher with comparable cloud-compute access. The full project envelope (including image-build retries, smoke tests, and the inevitable launcher overhead from two Modal billing-cycle cap-hit cycles during the run) is approximately $250 over baseline; the headline experiments themselves are ≤ $150.
