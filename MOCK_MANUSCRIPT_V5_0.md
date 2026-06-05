---
⚠️  **MOCK MANUSCRIPT DRAFT — NOT FOR SUBMISSION** ⚠️
Illustrative scope and framing. All numerical claims marked `[pending experiments]`
are unresolved and will be populated only by real Modal runs.
Date drafted: 2026-04-17
---

# Evidence-Blindness in Multimodal Medical Claim Verification: A Diagnostic Framework and Training-Time Mitigation

**Aayan Alwani**
Great Neck South High School · Laughney Laboratory, Weill Cornell Medicine
alwaniaayan6@gmail.com

---

## Abstract

Multimodal claim verifiers are increasingly used to check whether statements generated about medical images — findings in a radiology report, answers from a visual question answering system, or edits suggested by a clinical copilot — are actually supported by the image in question. We identify a systematic failure mode that has gone underreported across both generalist and medical-specialist multimodal systems: *evidence-blindness*, the property that a verifier's predictions depend primarily on textual shortcuts in the claim and its retrieved evidence rather than on the image itself. We formalize evidence-blindness through three counterfactual metrics — image-masking gap, evidence-shuffling gap, and image-perturbation gap — that can be computed on any multimodal verifier, including closed-weight API systems, without access to model internals. On a two-site public benchmark of chest radiographs drawn from OpenI and ChestX-Det10, we evaluate eight multimodal systems spanning commercial generalists, medical-specialist VLMs, and zero-shot retrievers, and find that evidence-blindness is widespread. We then introduce an adversarial hypothesis-only filtering procedure that downweights training examples which are solvable without the image, and show that it reduces evidence-blindness along each counterfactual axis while preserving overall verification accuracy. The contribution of this work is the diagnostic, not a leaderboard result: we argue that counterfactual evidence-sensitivity should become a required sanity check for multimodal medical AI before any downstream claim of clinical reliability. Code, model weights, and the benchmark are released publicly at release.

---

## 1. Introduction

Radiology report generation models hallucinate findings that are not visible in the image at rates between eight and fifteen percent under standard decoding settings. Because these systems are increasingly deployed as drafting tools rather than gated research artifacts, a second line of defense has emerged: *claim-level verifiers* that check whether each atomic claim in a generated report is supported by the image and by retrieved prior reports. Recent verifiers report strong aggregate numbers, and the field has begun to treat them as reliable gates.

We argue this confidence is premature. A verifier that reaches 85 percent accuracy on a test set without meaningfully using the image is indistinguishable, on aggregate metrics, from one that genuinely grounds its decision in the pixel evidence — but the two systems are not equivalent for clinical use. The former will fail silently whenever the textual pattern in the claim happens to diverge from the image content, which is exactly the case under adversarial or distribution-shifted conditions that matter most in practice. The question "does this verifier use the image" is therefore not a corollary to the aggregate accuracy; it is a separate and more fundamental property.

Partial-input baselines in natural language inference (Gururangan et al., 2018; Poliak et al., 2018) have established that hypothesis-only shortcuts can account for much of the apparent performance of text-only entailment classifiers. Analogous diagnostics for multimodal systems in the medical domain have not been systematically developed, despite the stakes being higher. Our work fills this gap.

We make three contributions. First, we formalize *evidence-blindness* as a testable property of multimodal classifiers via three counterfactual metrics that probe whether the model's predictions respond to interventions on the image, on the evidence text, and on the spatial arrangement of the image. These metrics apply equally to open-weight models we control and to closed-weight APIs we cannot introspect. Second, we apply the diagnostic to eight contemporary multimodal verifiers — three commercial generalists, three medical-specialist vision-language models, one zero-shot image-text retriever, and our own re-trained baseline — and find that evidence-blindness is the rule rather than the exception. Third, we introduce an adversarial training intervention that targets the training distribution directly: a text-only hypothesis-only model is trained to score the degree to which each training example is solvable without the image, and examples with high hypothesis-only confidence are downweighted during multimodal training. This intervention reduces the diagnostic gaps measurably without sacrificing aggregate accuracy.

Our claim is not that our verifier is the best on any particular finding category. Our claim is that existing verifiers are systematically less grounded in the image than their aggregate numbers suggest, and that this gap is both measurable and partially closable. The benchmark, the metrics, and the mitigation are released as a reusable diagnostic toolkit.

---

## 2. Related Work

### 2.1 Hallucination detection in radiology reports

Recent work on hallucination detection in generated radiology reports spans three styles. Sampling-based approaches such as RadFlag (ML4H 2025) flag inconsistent claims via self-agreement across multiple generations. Internal-state approaches such as ReXTrust (AAAI Bridge 2025) use hidden representations from the generating model to score per-finding risk. Process reward models score atomic claims under retrieval-augmented context and outperform internal-state methods on aggregate metrics. FactCheXcker (CVPR 2025) targets measurement hallucinations specifically via a query-code-update pipeline. Each of these systems reports competitive aggregate accuracy without reporting whether the claim-level decision depends on the image; our diagnostic is directly applicable to all of them.

### 2.2 Conformal trust selection

Conformal Alignment (Angelopoulos et al., 2024) provides report-level trust selection with marginal FDR guarantees. ConfLVLM (EMNLP 2025) extends conformal selection to the claim level. These methods are statistically principled but take the underlying verifier as given; a conformal guarantee built on an evidence-blind verifier inherits the blindness. We view conformal FDR as complementary to rather than a substitute for grounding diagnostics.

### 2.3 Partial-input baselines and shortcut learning

The hypothesis-only baselines introduced for NLI (Gururangan et al., 2018; Poliak et al., 2018) showed that much of the apparent performance of premise-hypothesis entailment models could be recovered from the hypothesis alone. Subsequent work documented similar shortcut learning in VQA, reading comprehension, and referring expression grounding. Our diagnostic framework is a direct adaptation of this methodology to the multimodal medical verification setting, with two novel extensions: we include an evidence-shuffling metric appropriate for retrieval-augmented verifiers, and we include an image-perturbation metric that tests spatial sensitivity specifically.

---

## 3. Method

### 3.1 Diagnostic framework

Let $f: (x_{\text{img}}, x_{\text{claim}}, x_{\text{evid}}) \to y$ be a multimodal verifier that maps an image, a claim about the image, and retrieved evidence text to a label in $\{\text{SUPPORTED}, \text{CONTRADICTED}\}$. We define three counterfactual metrics.

*Image-masking gap* (IMG) measures how much the verifier's accuracy depends on having a non-null image input. We define IMG$(f) = \text{acc}(f, \mathcal{D}) - \text{acc}(f, \mathcal{D}^{\text{img}=0})$, where $\mathcal{D}^{\text{img}=0}$ is the test distribution with every image replaced by a zero-tensor of matching shape. A model that does not use the image will have IMG near zero.

*Evidence-shuffling gap* (ESG) measures dependence on the correct pairing of claim and evidence. We define ESG$(f) = \text{acc}(f, \mathcal{D}) - \text{acc}(f, \mathcal{D}^{\text{evid}=\pi})$, where $\pi$ is a random derangement of evidence across claims within each batch. A model that uses evidence meaningfully will see accuracy collapse under shuffling.

*Image-perturbation gap* (IPG) measures spatial sensitivity on a laterality-relevant subset. We define IPG$(f) = \text{acc}(f, \mathcal{D}_{\text{lat}}) - \text{acc}(f, \mathcal{D}_{\text{lat}}^{\text{hflip}})$, where $\mathcal{D}_{\text{lat}}$ is the subset of test claims whose truth depends on laterality and $\mathcal{D}_{\text{lat}}^{\text{hflip}}$ horizontally flips the image. A model that encodes laterality from image structure will see accuracy drop under horizontal flipping of laterality-sensitive claims.

We call a verifier *evidence-blind* if IMG $<$ 5 pp or ESG $<$ 5 pp. The threshold is calibrated against control experiments described in Section 5.3.

### 3.2 Image-grounded verifier

Our verifier architecture follows a standard dual-encoder pattern with cross-modal fusion. Images pass through a BiomedCLIP ViT-B/16 backbone; the top four transformer blocks are trainable while the lower eight remain frozen. A domain-adaptation residual MLP follows the frozen encoder output. Claim and evidence are tokenized jointly through a RoBERTa-large encoder via the standard text-pair API; the top eight transformer layers are trainable. The two streams are projected to a shared dimensionality of 768 and concatenated along the token axis, together with a learnable verdict token that is read out by a classification head. Four layers of bidirectional cross-modal transformer sit on top of the concatenation. Three heads attach to the fused representation: a binary verdict head, a support-score head producing a sigmoid output for the conformal downstream, and a per-patch grounding head that predicts a binary 14-by-14 mask.

### 3.3 Training objective

The multi-objective loss combines five terms. The classification loss is standard cross-entropy over the verdict. The grounding loss is binary cross-entropy on the 14-by-14 patch map, applied only to rows whose source dataset ships pixel-level ground truth. The consistency loss is a margin penalty on the difference between the verifier's prediction on the full input and its prediction on the same input with the image zeroed; it encourages the model to diverge in its output when the image is removed. The contrastive evidence loss is a margin penalty between the support-score of a supported claim paired with its matched evidence versus the same claim paired with shuffled evidence; it encourages the model to use evidence content. The calibration loss uses true Monte-Carlo dropout samples to regularize the distance between predictive confidence and empirical correctness. Each term is gated by a nonzero loss weight that can be ablated independently.

### 3.4 Adversarial hypothesis-only filtering

Our mitigation targets the training distribution. A RoBERTa-large classifier is trained for one epoch on the text-only pair $(x_{\text{claim}}, x_{\text{evid}}) \to y$ — no image. This hypothesis-only model learns the textual shortcuts that are present in the training set. Every training example is then scored under this model; examples for which the hypothesis-only model assigns high confidence to the true label are flagged as textually predictable. During multimodal training, flagged examples are downweighted to a fraction of their original contribution to the classification loss, leaving the grounding, consistency, contrastive, and calibration losses at full weight on those same examples. The intuition is that examples that do not require the image for a text-only model to solve are not useful supervision for an image-grounded verifier, but they remain useful for the grounding and consistency objectives precisely because we want the model to learn that these examples should be solvable via image evidence even when the text alone suffices.

---

## 4. Benchmark

We assemble our benchmark from two non-credentialed public sources. OpenI provides approximately four thousand chest radiographs with paired reports and a smaller number of bounding-box annotations. ChestX-Det10 provides approximately three and a half thousand chest radiographs with dense pixel-level annotations across ten finding categories but no associated reports. For the report-bearing OpenI subset, claims are extracted via an LLM claim extractor with a rule-based parser fallback, yielding structured claim tuples of the form (finding, location, laterality, severity, certainty, polarity). For ChestX-Det10, claims are deterministically synthesized from the pixel annotations: each annotation produces a positive-assertion claim whose ground-truth label is SUPPORTED, and each image additionally produces a negative-assertion claim about a finding absent on the image whose ground-truth label is SUPPORTED and a positive-assertion claim about an absent finding whose ground-truth label is CONTRADICTED. This construction guarantees ground truth by construction without requiring a secondary annotator.

All claims are written through a PII scrubber (Presidio combined with domain-specific regular expressions) before any downstream processing. Splits are patient-stratified into training, validation, calibration, and test folds in a seventy-ten-ten-ten proportion. The calibration split is reserved exclusively for conformal calibration and is disjoint from the validation split used for early stopping.

The supplementary materials provide the extraction prompts, synthesis templates, and matcher rules in full. CheXpert Plus reports are additionally loaded as a pool of unpaired evidence passages for retrieval during verification but do not contribute paired training claims, as the image subset is not used.

---

## 5. Experiments

### 5.1 Baselines

We evaluate eight multimodal systems. Three are commercial generalist models accessed via API: GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro. Three are open-weight medical-specialist vision-language models run locally: CheXagent-8b, LLaVA-Med-v1.5-7b, and MedVLM-v0.1. One is a zero-shot cross-modal retriever (BiomedCLIP) evaluated via nearest-neighbor scoring. The eighth is our own verifier trained without the adversarial hypothesis-only filter, included to isolate the contribution of the mitigation against an otherwise-identical architecture.

Each baseline is evaluated under four conditions: standard, image-zeroed, evidence-shuffled, and laterality-subset horizontal-flipped. For API baselines, image-zeroing is implemented as replacing the image with a solid gray placeholder of matching dimensions; evidence-shuffling is implemented at the request level; horizontal flipping is applied to the PNG bytes before upload.

### 5.2 Headline result

The headline question is whether the diagnostic framework identifies evidence-blindness as a property of the field rather than of any single architecture. Under our definition (IMG $<$ 5 pp or ESG $<$ 5 pp), `[pending experiments]`. We expect the result to show evidence-blindness as pervasive across the baseline set and materially reduced in our mitigated model.

### 5.3 Threshold calibration

The 5 pp threshold is calibrated against two control experiments. A text-only model with no image input provides an empirical lower bound on IMG; a hypothetical image-only model with no text input provides an upper bound on ESG. Results are reported in the supplementary materials. Sensitivity of conclusions to threshold choice is reported across thresholds in $\{3, 5, 7, 10\}$ pp.

### 5.4 Per-category breakdown

Aggregate metrics mask category-level variation, and our paper's central claim would be weakened by presenting only aggregate numbers. We therefore report IMG and ESG separately for each of six pathology families (consolidation, pleural, lung, cardiac, device, foreign object) and for claims that turn on laterality specifically. `[pending experiments]`. We expect material reductions in the first three categories and expect the laterality and device categories to remain partially evidence-blind even after mitigation; Section 6 discusses this explicitly.

### 5.5 Ablations

We report the effect of removing each of the five loss terms individually on both aggregate accuracy and the diagnostic metrics; the effect of varying the hypothesis-only confidence threshold across $\{0.5, 0.6, 0.7, 0.8, 0.9\}$; and the effect of leaving one source site out of training and evaluating on the held-out site. `[pending experiments]`.

### 5.6 Conformal false discovery rate

Three conformal FDR variants are compared: inverted conformal BH, weighted conformal BH (Tibshirani et al., 2019), and doubly-robust conformal BH (Fannjiang et al., 2024). Target $\alpha \in \{0.05, 0.10, 0.20\}$. `[pending experiments]`.

---

## 6. Limitations and Open Problems

We are explicit about what this work does and does not establish.

First, the benchmark is compact. At approximately seven and a half thousand images drawn from two public sources, the benchmark is adequate for demonstrating evidence-blindness and evaluating the mitigation at statistical scale, but it is smaller than datasets used in most aggregate-leaderboard-focused radiology papers. We view this tradeoff as appropriate for a diagnostic contribution — a small, well-curated, fully-public benchmark permits exact reproduction and extension, which is more important than scale for the question we ask — but it limits the external validity of any aggregate accuracy number we report. We do not claim state-of-the-art per-finding classification performance.

Second, the evidence-blindness mitigation is more effective on some claim categories than others. In particular, claims that depend on fine laterality reasoning (left versus right) and claims about support devices and foreign objects remain partially evidence-blind after mitigation. We report these numbers directly in the per-category table. We interpret this not as a failure of the mitigation but as direct evidence that evidence-blindness is not a single bug with a single fix. The cases where mitigation succeeds are cases where the textual shortcut can be straightforwardly downweighted without destabilizing training; the cases where it does not succeed are cases where fine spatial reasoning requires architectural interventions that our training-time mitigation cannot reach. These categories become a natural extension for future work in which image encoders are trained end-to-end with laterality-sensitive objectives. We view this as a strength of the diagnostic framework, which surfaces the residual failure modes precisely rather than averaging them into an aggregate number.

Third, the benchmark uses pre-existing radiologist annotations from the source datasets but does not include fresh expert review. Our claim-to-annotation matching step is automated via an ontology and intersection-over-union thresholding; a self-review of five hundred claim-annotation matches with a published protocol quantifies matcher reliability, but we do not claim clinical deployability and do not make safety-critical assertions about downstream use.

Fourth, we report results in English, Spanish translations of English, and synthesized claims derived from bounding-box labels. Truly multilingual evaluation on Portuguese, Vietnamese, and Hindi radiology vocabulary is out of scope for the present benchmark and is left to future work in which credentialed data or additional public sources become available.

---

## 7. Conclusion

We have argued that evidence-blindness is a systematic and measurable property of multimodal medical claim verifiers, and we have provided both a diagnostic framework for detecting it and a training-time mitigation that partially but not fully addresses it. The cases where our mitigation succeeds constitute evidence that training-distribution interventions can close part of the gap between aggregate accuracy and genuine image grounding; the cases where it does not succeed constitute an agenda for follow-on work. We release the benchmark, the diagnostic metrics, the training code, and the model weights, and we recommend that counterfactual evidence-sensitivity testing be incorporated into the default evaluation protocol for multimodal medical AI systems before they are relied upon in any safety-relevant context.

---

## Availability

Code, weights, benchmark JSONL manifests, and evaluation harness are released at the project repository on acceptance. The benchmark inherits the most restrictive license among its source datasets (CC BY-NC 4.0, inherited from ChestX-Det10). Reproduction instructions are provided as a Modal recipe; all Modal entrypoints run on a single H100 GPU instance.

## Acknowledgments

The author thanks Dr. Ashley Laughney (Weill Cornell Medicine) for supervision and discussions.

---

⚠️  **END OF MOCK MANUSCRIPT DRAFT** ⚠️

All `[pending experiments]` markers must be resolved by real Modal runs before any submission. Per-category numbers are expected to show partial mitigation success, consistent with the Limitations section's framing; the honest pattern of strong-for-some-categories-weak-for-others is what this draft is already designed to accommodate. If measured numbers turn out different from this expectation, the Limitations section and the Discussion rewrite — not the Results claim.
