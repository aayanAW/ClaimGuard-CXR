# ClaimGuard-CXR v6.0 — Poster / Presentation Figure Script

A coherent narrative connecting all 15 generated figure prompts plus the one existing matplotlib figure I found usable (`fig7_cross_dataset.png`). Read top-to-bottom — each row is 1–3 sentences of spoken/written script plus the figure it anchors.

Copy this into a Google Doc; each "→ Figure X" is where the image goes.

---

## Act 1 — The problem (hook for viewers)

**1. → F1 (hero illustration)**
A chest-X-ray claim verifier can reach 90.9% accuracy without ever looking at the image. We call this *evidence-blindness*: the model is solving the task via text shortcuts, not image content. That's dangerous when the verifier is a safety layer over generative medical AI.

**2. → F2 (NLI → CXR analogy)**
This failure mode is not new. In 2018, hypothesis-only baselines showed that NLI classifiers solved premise-hypothesis benchmarks with the premise removed. The same shortcut pattern appears here — 24 years later, different domain, identical diagnosis.

## Act 2 — Landscape & stakes

**3. → F4 (research timeline, 2018–2026)**
The field has built increasingly powerful CXR report generators (MAIRA-2, CheXagent-2, MedGemma) and hallucination detectors (RadFlag, ReXTrust, GREEN, VERT). Conformal trust-selection wraps them. None of these works checks whether the underlying verifier uses the image.

**4. → F3 (hallucination taxonomy)**
The hallucinations we want to catch come in four flavors: false findings, missing findings, anatomic/laterality errors, and severity errors. Each type has a different signature and a different clinical cost.

## Act 3 — Our benchmark

**5. → F10 (GroundBench-v6 composition)**
We assemble ClaimGuard-GroundBench-v6 from three non-credentialed public sources — 40,585 (image, claim) pairs across OpenI, ChestX-Det10, and PadChest-GR. Bilingual (EN/ES), US + Spain. PadChest-GR provides per-sentence radiologist bounding boxes that serve as the external validation target.

## Act 4 — How we label without a radiologist

**6. → F5 (cost-quality frontier of labelers)**
We cannot recruit radiologists but still need claim-level ground truth. The labeler landscape spans from cheap-and-unreliable (CheXbert rule-based) to expensive-and-gold (board-certified MD). Our three-grader ensemble sits on the Pareto frontier of what's accessible to a small lab.

**7. → F9 (three-grader silver-labeling ensemble)**
GREEN (MIMIC-trained, high-τ), RadFact (MIMIC-free), and VERT (MIMIC-free) vote on each claim. We report Krippendorff α per grader-pair to decouple MIMIC-pretraining correlation. A claim is accepted only when at least 2 of 3 agree.

## Act 5 — Method

**8. → F6 (pipeline architecture)**
ClaimGuard-CXR v6.0 = BiomedCLIP ViT-B/16 (image) + RoBERTa-large (text) + 4-layer cross-modal fusion + three heads (verdict, support-score, grounding mask) + inverted conformal cfBH on top. 480M total params, 122M trainable.

**9. → F7 (composite loss + ablation)**
Five training loss terms: classification, grounding, consistency, contrastive evidence, uncertainty. The consistency loss directly penalizes the image-masking gap; the adversarial hypothesis-only filter (HO) downweights text-solvable training examples.

**10. → F8 (HO filter procedure)**
The HO filter works in three stages: train a text-only classifier, score every training row, downweight rows the text-only model can solve. This removes shortcut-solvable claims from the main training signal.

## Act 6 — The headline result

**11. → F11 (IMG ladder: v5.0 → v6.0-3site)**
The cross-entropy baseline is evidence-blind (IMG = 2.03pp). Adding grounding loss alone doesn't help (IMG = 2.17pp). Adding the consistency loss + HO filter is the decisive intervention (IMG = 62.9pp → 69.24pp → **75.25pp on v6.0-3site**). Aggregate accuracy climbs from 90.9% to 95.4%. The mitigation works.

**12. → F12 (load-bearing ablation)**
But not all five losses matter. Drop-one ablation identifies exactly two load-bearing components: the consistency loss and the HO filter. Removing either drops IMG back to near the evidence-blind threshold. The grounding, contrastive, and uncertainty losses are neutral-to-small.

## Act 7 — Honest limitations

**13. → F13 (cross-site 3-way LOO heatmap)**
The mitigation is training-distribution-specific. In 3-way leave-one-site-out evaluation, IMG on the held-out site collapses to 0–10 pp. The evidence-blindness fix does not transfer — it's a feature of the training distribution, not a learned general capability.

**14. → fig7_cross_dataset.png (existing, FDR control cross-site)**
What DOES transfer is the conformal FDR guarantee itself: observed FDR tracks target α across in-domain (CheXpert Plus) and out-of-domain (OpenI) evaluation. Power drops on OOD, but the false-discovery control stays honest.

## Act 8 — Deployment-shaped artifacts

**15. → F14 (real-hallucination precision-recall)**
On 2,707 claims extracted from real RRG outputs (MAIRA-2, CheXagent-2, MedGemma-4B on 500 OpenI images) and silver-labeled by our ensemble, our detector Pareto-dominates five public baselines for hallucination precision/recall. Two of the four public detectors that completed show evidence-blindness on at least one axis.

**16. → F15 (conformal FDR selection + support-score sharpness)**
Our v5.3/v6.0 support-score distributions are sharp enough for inverted cfBH to produce a meaningful selection set (n_green = 985, FDR = 0.9% at α = 0.10). v5.0–v5.2 fail this test — their scores are too concentrated. The final triage output is a green/yellow/red label per claim with a formal FDR guarantee when the diagnostic passes.

---

## One-paragraph TL;DR (for poster sidebar)

ClaimGuard-CXR v6.0 delivers three artifacts to the NeurIPS 2026 E&D track: (1) a diagnostic framework (IMG/ESG/IPG) that measures whether a multimodal medical verifier actually uses its image, (2) a training-time mitigation that closes the gap 30× in-distribution, and (3) a 3-site public benchmark with multi-grader silver labels validated against PadChest-GR's radiologist bounding boxes. The honest limitation is that the mitigation doesn't fully transfer across sites — training-distribution intervention is a cheap lever where it applies and an architectural question where it doesn't.

---

## Poster layout suggestion (48" × 36" landscape, 3-column)

```
┌─────────────────────────────────────────────────────────────────┐
│ TITLE · Aayan Alwani · Laughney Lab · Weill Cornell · QR code  │
├─────────────┬──────────────────────────┬────────────────────────┤
│ 1. F1 hero  │ 4. F10 data composition  │ 11. F11 IMG ladder     │
│             │                          │    (the money shot)    │
│ 2. F2 NLI   ├──────────────────────────┼────────────────────────┤
│    analogy  │ 7. F9 silver ensemble    │ 12. F12 ablation       │
│             │    (methods)             │    (load-bearing)      │
├─────────────┼──────────────────────────┼────────────────────────┤
│ 3. F4       │ 8. F6 pipeline arch      │ 13. F13 LOO heatmap    │
│    timeline │    (methods)             │    (honest limit)      │
│             ├──────────────────────────┼────────────────────────┤
│             │ 10. F8 HO filter         │ 14. fig7_cross_dataset │
│             │                          │    (FDR transfers)     │
├─────────────┴──────────────────────────┼────────────────────────┤
│ TL;DR · Takeaways · References                                  │
└─────────────────────────────────────────────────────────────────┘
```

Cut from the end if space is tight: F4 → F8 → F3 → F5 → F2 → F10. F1 + F11 + F12 + F13 + F6 + fig7 is the minimum viable poster.

## Narrative rhythm cues (for spoken talks / poster walks)

- **Figures 1–4**: frame the problem. Keep attention on "medical AI has a shortcut-learning pathology."
- **Figures 5–7**: set up your contribution by showing the benchmark and labeling protocol. Viewers want to know "is your ground truth credible?"
- **Figures 8–10**: the method. Don't spend more than 20 seconds per figure here — it's the dense part.
- **Figures 11–12**: the payoff. Slow down. Let the 30× IMG improvement land.
- **Figures 13–14**: honesty. This is where a strong paper separates from a weak one. Admit the transfer gap, show the FDR control survives it.
- **Figures 15–16**: close the loop. The conformal triage output is the final artifact a clinician would see.

## Written sentence cap per figure (for poster caption boxes)

Keep each figure's caption under **35 words**. Example for F11:

> *"Image-masking gap (IMG) across training configurations. The cross-entropy baseline (v5.0) is evidence-blind; adding consistency loss + adversarial hypothesis-only filter (v5.2+) closes the gap 30×; v6.0's 3-site training pushes IMG to 75.25 pp."*
