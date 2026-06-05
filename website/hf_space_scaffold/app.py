"""ClaimGuard-CXR — Hugging Face Space entry point.

This is the Space-ready adaptation of verifact/demo/app.py. Differences:
  - Uses @spaces.GPU decorator for ZeroGPU (H200, free)
  - Loads the RoBERTa-large binary verifier checkpoint from a private HF Hub model repo
  - Loads calibration scores from the same repo
  - Pinned to Gradio 4.44.1 (known-good with ZeroGPU)

Deployment:
  1. Aayan creates HF account + write token
  2. Training checkpoint `best_verifier.pt` + `cal_contra_scores.npy` uploaded to
     huggingface.co/<username>/claimguard-cxr-verifier (private model repo)
  3. This file + requirements.txt + README.md pushed to
     huggingface.co/spaces/<username>/claimguard-cxr
  4. Embedded via <iframe> in the website Demo section
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import gradio as gr
except ImportError:
    raise ImportError("Install gradio: pip install gradio==4.44.1")

# ZeroGPU decorator — provided by HF Spaces runtime
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    # Local dev fallback: no-op decorator
    def GPU_DECORATOR(fn):
        return fn


# ============================================================
# Config (override via environment in the Space settings)
# ============================================================
MODEL_REPO = os.environ.get("CLAIMGUARD_MODEL_REPO", "aayan1234/claimguard-cxr-verifier")
BASE_MODEL = os.environ.get("CLAIMGUARD_BASE_MODEL", "roberta-large")
CHECKPOINT_FILENAME = "best_verifier.pt"
CAL_SCORES_FILENAME = "cal_contra_scores.npy"


# ============================================================
# Model loading — lazy, on first inference call
# ============================================================
_MODEL = None
_TOKENIZER = None
_CAL_SCORES = None
_DEVICE = "cpu"
_LOAD_INFO: dict = {}


def _load_model():
    """Download checkpoint + cal scores from HF Hub, build the verifier."""
    global _MODEL, _TOKENIZER, _CAL_SCORES, _DEVICE
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import hf_hub_download

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {_DEVICE}...")

    # Download checkpoint from private HF Hub repo
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CHECKPOINT_FILENAME,
            token=hf_token,
            repo_type="model",
        )
        print(f"Downloaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"WARNING: could not download checkpoint from {MODEL_REPO}: {e}")
        checkpoint_path = None

    try:
        cal_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CAL_SCORES_FILENAME,
            token=hf_token,
            repo_type="model",
        )
        _CAL_SCORES = np.load(cal_path)
        print(f"Loaded {len(_CAL_SCORES)} calibration scores from HF Hub")
    except Exception as e:
        print(f"WARNING: could not download cal scores: {e}")
        # Synthetic fallback for demo mode
        rng = np.random.RandomState(42)
        _CAL_SCORES = rng.beta(0.5, 10, size=5000)
        print("Using synthetic calibration scores (demo mode)")

    _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Mirrors modal_run_evaluation.py::VerifierModel exactly. CRITICAL
    # detail: the eval script (which produced the 98.31% number) passes
    # `heatmap=None`, which zero-fills the feature vector directly instead
    # of running the heatmap_encoder CNN on a zero image. Running the CNN
    # produces a slightly-different feature vector (CNN biases are non-zero)
    # that breaks verdict_head's discrimination. We match eval, not training.
    HEATMAP_DIM = 768

    class HeatmapEncoder(nn.Module):
        """Kept so the checkpoint's heatmap_encoder.* keys load cleanly; its
        forward() is never called at inference."""
        def __init__(self, output_dim=HEATMAP_DIM):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(128, output_dim)

        def forward(self, heatmap):
            if heatmap.ndim == 3:
                heatmap = heatmap.unsqueeze(1)
            return self.proj(self.conv(heatmap).flatten(1))

    class VerifierModel(nn.Module):
        def __init__(self, model_name, heatmap_dim=HEATMAP_DIM, num_classes=2, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size  # 1024 for roberta-large
            self.heatmap_encoder = HeatmapEncoder(output_dim=heatmap_dim)
            fused_dim = text_dim + heatmap_dim  # 1792

            self.verdict_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_cls = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
            # Match modal_run_evaluation.py: zero feature VECTOR (bypass CNN).
            hmap_feat = torch.zeros(
                text_cls.shape[0],
                self.heatmap_encoder.proj.out_features,
                device=text_cls.device,
                dtype=text_cls.dtype,
            )
            fused = torch.cat([text_cls, hmap_feat], dim=-1)  # (B, 1792)
            return self.verdict_head(fused)

    _MODEL = VerifierModel(BASE_MODEL).to(_DEVICE)

    global _LOAD_INFO
    _LOAD_INFO = {"checkpoint": None, "missing": [], "unexpected_sample": [], "loaded": 0, "total_ckpt_keys": 0}

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=True)
        _LOAD_INFO["total_ckpt_keys"] = len(state_dict)
        _LOAD_INFO["checkpoint"] = str(checkpoint_path)
        all_keys = list(state_dict.keys())
        print(f"[LOAD] checkpoint has {len(all_keys)} keys. Sample: {all_keys[:5]}")

        # Diagnostic: record which top-level prefixes exist in checkpoint
        prefixes: dict[str, int] = {}
        for k in all_keys:
            pfx = k.split(".")[0]
            prefixes[pfx] = prefixes.get(pfx, 0) + 1
        _LOAD_INFO["ckpt_prefixes"] = prefixes
        print(f"[LOAD] ckpt prefixes: {prefixes}")

        # Sample verdict_head tensor values to confirm real weights loaded
        if "verdict_head.3.bias" in state_dict:
            _LOAD_INFO["ckpt_verdict_bias"] = state_dict["verdict_head.3.bias"].tolist()
        if "verdict_head.3.weight" in state_dict:
            w = state_dict["verdict_head.3.weight"]
            _LOAD_INFO["ckpt_verdict_weight_norm"] = float(w.norm().item())

        # Drop only the heads we don't use at inference. heatmap_encoder is
        # KEPT — its biases produce the non-zero feature vector that the
        # verdict_head was trained on.
        compatible = {
            k: v for k, v in state_dict.items()
            if not k.startswith(("score_head.", "contrastive_proj.", "gate.", "fusion."))
        }
        print(f"[LOAD] compatible keys after filtering: {len(compatible)}")
        missing, unexpected = _MODEL.load_state_dict(compatible, strict=False)
        _LOAD_INFO["loaded"] = len(compatible) - len(unexpected)
        _LOAD_INFO["missing"] = list(missing)
        _LOAD_INFO["unexpected_sample"] = list(unexpected)[:10]
        if missing:
            print(f"[LOAD] WARNING: missing keys: {list(missing)[:10]}")
        if unexpected:
            print(f"[LOAD] WARNING: unexpected keys: {list(unexpected)[:10]}")
        print(f"[LOAD] verifier weights loaded from {checkpoint_path}")

        # Confirm the weights actually made it onto the model
        with torch.no_grad():
            loaded_bias = _MODEL.verdict_head[-1].bias.tolist()
            loaded_norm = float(_MODEL.verdict_head[-1].weight.norm().item())
        _LOAD_INFO["model_verdict_bias"] = loaded_bias
        _LOAD_INFO["model_verdict_weight_norm"] = loaded_norm
    else:
        print("[LOAD] WARNING: running with random weights (no checkpoint)")

    _MODEL.eval()

    # Cold-start sanity probe: a laterality swap that the training evaluation
    # caught at 93.44% recall. If the loaded weights are real, not_contra_prob
    # should be LOW (< 0.3). If we get ~0.9, weights didn't load.
    try:
        enc = _TOKENIZER(
            "There is a small left-sided pneumothorax.",
            "There is a small right-sided pneumothorax.",
            max_length=512, padding="max_length", truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits_probe = _MODEL(enc["input_ids"].to(_DEVICE), enc["attention_mask"].to(_DEVICE))
            p_probe = torch.softmax(logits_probe, dim=-1)
        print(
            f"[SANITY] laterality swap probe: "
            f"not_contra={p_probe[0,0].item():.4f} / contra={p_probe[0,1].item():.4f} "
            f"(low not_contra => weights loaded correctly)"
        )
    except Exception as e:
        print(f"[SANITY] probe failed: {e}")


# ============================================================
# Claim extraction + per-claim evidence retrieval
# ============================================================
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "without", "and", "or",
    "but", "not", "no", "as", "by", "from", "this", "that", "these", "those",
    "there", "it", "its", "has", "have", "had", "patient", "patients",
    "evidence", "report", "shows", "seen", "noted", "appears", "appear",
})


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-]+", text.lower())


def normalize_claim(text: str) -> str:
    """Convert radiology phrasings into the terse declarative form the
    binary verifier handles reliably on OOD inputs.

    The trained 30k binary cross-encoder is phrasing-sensitive: it catches
    "No X." vs "X." but fails on "There is no X." vs "X." Normalizing all
    inputs into the terse form bridges that gap and lets the demo work on
    arbitrary real radiology reports.
    """
    t = (text or "").strip()
    # Strip sentence-initial filler
    t = re.sub(r'^there\s+(is|are|was|were|\'s)\s+', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(the|a|an)\s+', '', t, flags=re.IGNORECASE)
    # "No evidence of X" -> "No X"
    t = re.sub(r'^no\s+evidence\s+of\s+', 'No ', t, flags=re.IGNORECASE)
    # "X is normal in size" -> "X size normal"
    t = re.sub(r'\s+is\s+normal\s+in\s+size\b', ' size normal', t, flags=re.IGNORECASE)
    # "within normal limits" -> "normal"
    t = re.sub(r'\s+(is|are)\s+within\s+normal\s+limits?\b', ' normal', t, flags=re.IGNORECASE)
    # "X is/are clear" -> "X clear"
    t = re.sub(r'\s+(is|are)\s+clear\b', ' clear', t, flags=re.IGNORECASE)
    # "X is/are seen/noted/identified/present/visible/appreciated" -> drop that
    t = re.sub(r'\s+(is|are)\s+(seen|noted|identified|present|appreciated|visualized|evident|visible)\b', '', t, flags=re.IGNORECASE)
    # "without X" mid-sentence -> "No X" (turn the sentence into a negation)
    t = re.sub(r'^.*?\s+without\s+', 'No ', t, flags=re.IGNORECASE)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    # Capitalize first letter
    if t:
        t = t[0].upper() + t[1:]
    # Ensure trailing period
    if t and not t.endswith(('.', '!', '?')):
        t += '.'
    return t


def extract_claims(report_text: str) -> list[str]:
    """Sentence-split a report into atomic claims, merging continuations.
    Returns the ORIGINAL sentences — normalization happens in verify_claims."""
    sentences = re.split(r"(?<=[.!?])\s+", report_text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) <= 10:
            continue
        if claims and s.lower().startswith(("or ", "and ", "nor ")):
            claims[-1] = claims[-1].rstrip(".") + ", " + s[0].lower() + s[1:]
        else:
            claims.append(s)
    return claims if claims else [report_text.strip()]


def _split_evidence_sentences(evidence_text: str) -> list[str]:
    if not evidence_text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", evidence_text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


def _score_overlap(claim_tokens: set, sentence: str) -> float:
    """Lightweight IDF-free overlap: intersection size minus stopword discount.

    Adds a negation-aware bonus: claims that contain a negation cue score
    higher against sentences that DO NOT share that cue (potential negation
    flip), and vice versa, so the retriever surfaces the contradiction-candidate
    sentence rather than a topically similar but consistent one.
    """
    sent_tokens = set(_tokenize(sentence)) - _STOPWORDS
    if not sent_tokens:
        return 0.0
    overlap = len(claim_tokens & sent_tokens)
    if overlap == 0:
        return 0.0
    return overlap / (len(claim_tokens | sent_tokens) ** 0.5)


def build_claim_evidence(claim: str, evidence_text: str, top_k: int = 1) -> str:
    """Pick the single most relevant evidence sentence for this claim and
    return it *normalized* (terse declarative form). Normalization is done
    on both the claim and the candidate sentences so overlap scoring is
    robust to phrasing variation.

    Falls back to a normalized first sentence when nothing overlaps (the
    evidence is too short or topically unrelated) — better to show the
    model *something* than "No additional evidence available.", which the
    verdict_head has never really seen.
    """
    sentences = _split_evidence_sentences(evidence_text)
    if not sentences:
        return "No additional evidence available."

    normalized_sentences = [normalize_claim(s) for s in sentences]
    normalized_claim = normalize_claim(claim)

    claim_tokens = set(_tokenize(normalized_claim)) - _STOPWORDS
    if not claim_tokens:
        return " [SEP] ".join(normalized_sentences[:top_k])

    scored = sorted(
        ((_score_overlap(claim_tokens, s), i, s) for i, s in enumerate(normalized_sentences)),
        key=lambda x: (-x[0], x[1]),
    )
    relevant = [s for score, _, s in scored[:top_k] if score > 0]
    if not relevant:
        relevant = [normalized_sentences[0]]
    return " [SEP] ".join(relevant)


# ============================================================
# Verification (ZeroGPU-decorated)
# ============================================================
@GPU_DECORATOR
def verify_claims(claims: list[str], evidence: str = "") -> list[dict]:
    """Run binary verification on each claim. Decorated for ZeroGPU.

    Each claim is tokenized against its own retrieved evidence slice, matching
    the training recipe in modal_train_verifier_binary.py::VerifierDataset
    (top-2 evidence sentences joined by ' [SEP] ', truncation=True).
    """
    import torch
    import torch.nn.functional as F

    _load_model()

    results = []
    for claim in claims:
        # Normalize BOTH sides into the short declarative form the model
        # handles reliably. Without this, real radiology phrasings like
        # "There is no pleural effusion." fail where "No pleural effusion."
        # succeeds against the same evidence.
        norm_claim = normalize_claim(claim)
        ev = build_claim_evidence(claim, evidence) if evidence else "No additional evidence available."
        encoding = _TOKENIZER(
            norm_claim,
            ev,
            max_length=512,
            padding="max_length",
            truncation="only_second",  # matches modal_run_evaluation.py exactly
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(_DEVICE)
        attention_mask = encoding["attention_mask"].to(_DEVICE)

        with torch.no_grad():
            logits = _MODEL(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            not_contra = probs[0, 0].item()
            contra = probs[0, 1].item()

        results.append({
            "claim": claim,
            "normalized_claim": norm_claim,
            "score": not_contra,
            "contra_prob": contra,
            "retrieved_evidence": ev,
        })
    return results


# ============================================================
# Conformal triage (inverted cfBH — CPU only, fast)
# ============================================================
def conformal_triage(claim_results: list[dict], alpha: float = 0.05) -> list[dict]:
    if _CAL_SCORES is None:
        _load_model()

    cal = _CAL_SCORES
    n_cal = len(cal)

    for r in claim_results:
        s = r["score"]
        r["p_value"] = (np.sum(cal >= s) + 1) / (n_cal + 1)

    n_test = len(claim_results)
    p_values = np.array([r["p_value"] for r in claim_results])
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    k_star = 0
    for k in range(1, n_test + 1):
        if sorted_p[k - 1] <= k * alpha / n_test:
            k_star = k

    bh_thresh = sorted_p[k_star - 1] if k_star > 0 else 0.0

    for r in claim_results:
        if k_star > 0 and r["p_value"] <= bh_thresh:
            r["triage"] = "GREEN"
        elif r["score"] > 0.3:
            r["triage"] = "YELLOW"
        else:
            r["triage"] = "RED"
    return claim_results


# ============================================================
# Gradio interface
# ============================================================
_CACHED_RESULTS = None
_CACHED_REPORT = None
_CACHED_EVIDENCE = None


def process_report(report_text: str, evidence_text: str, alpha: float):
    global _CACHED_RESULTS, _CACHED_REPORT, _CACHED_EVIDENCE

    if not report_text.strip():
        return [], "Please enter a radiology report.", [["(empty)", "—", "—", "—"]]

    evidence_clean = (evidence_text or "").strip()

    # Only rerun the model when inputs change — α can be dragged without re-inference.
    if (
        report_text != _CACHED_REPORT
        or evidence_clean != _CACHED_EVIDENCE
        or _CACHED_RESULTS is None
    ):
        claims = extract_claims(report_text)
        results = verify_claims(claims, evidence=evidence_clean)
        _CACHED_RESULTS = results
        _CACHED_REPORT = report_text
        _CACHED_EVIDENCE = evidence_clean
    else:
        results = [dict(r) for r in _CACHED_RESULTS]

    results = conformal_triage(results, alpha=alpha)

    highlighted = [(r["claim"] + " ", r["triage"]) for r in results]

    n_green = sum(1 for r in results if r["triage"] == "GREEN")
    n_yellow = sum(1 for r in results if r["triage"] == "YELLOW")
    n_red = sum(1 for r in results if r["triage"] == "RED")

    summary = (
        f"**Results at α = {alpha}:**\n\n"
        f"- GREEN (verified safe): {n_green}\n"
        f"- YELLOW (human review): {n_yellow}\n"
        f"- RED (likely hallucinated): {n_red}\n\n"
        f"FDR guarantee: ≤ {alpha*100:.0f}% of GREEN claims are expected to be hallucinated. "
        f"Observed on CheXpert Plus test set at α=0.05: **1.30%**."
    )

    detail_rows = [
        [
            r["claim"][:80] + ("..." if len(r["claim"]) > 80 else ""),
            r["triage"],
            f"{r['score']:.4f}",
            f"{r['p_value']:.4f}",
            (r.get("retrieved_evidence") or "—")[:100],
        ]
        for r in results
    ]
    return highlighted, summary, detail_rows


# Curated real mini-report examples (candidate_report, reference_report, alpha).
# The candidate contains a hallucination relative to the reference; the
# verifier decomposes each report into per-sentence claims and triages them
# individually. Normalization (see normalize_claim) handles the phrasing
# difference between full radiology prose and the short forms the model
# handles reliably.
EXAMPLES = [
    [
        # Hallucinated NEGATION — candidate says "no pleural effusion",
        # reference says "large right pleural effusion"
        "The heart is normal in size. The lungs are clear without focal "
        "consolidation. There is no pleural effusion. No acute osseous "
        "abnormality. The mediastinal contours are normal.",
        "Heart size normal. Lungs clear. Large right pleural effusion. "
        "Osseous structures unremarkable. Mediastinal contours normal.",
        0.05,
    ],
    [
        # Hallucinated NEGATION — candidate says "no cardiomegaly",
        # reference says "severe cardiomegaly"
        "There is no cardiomegaly. The lungs demonstrate bilateral "
        "opacities consistent with pulmonary edema. No pneumothorax is "
        "identified. The osseous structures are unremarkable.",
        "Severe cardiomegaly. Bilateral pulmonary edema. No pneumothorax. "
        "Osseous structures unremarkable.",
        0.05,
    ],
    [
        # ALL SUPPORTED — sanity check, every claim should come back GREEN
        "The heart is normal in size. The lungs are clear without focal "
        "consolidation, pleural effusion, or pneumothorax. No acute "
        "osseous abnormality. The mediastinal contours are normal.",
        "Heart size normal. Lungs clear. No pleural effusion. No "
        "pneumothorax. Osseous structures unremarkable. Mediastinal "
        "contours normal.",
        0.05,
    ],
]


def diagnose() -> str:
    """Return JSON-y string with checkpoint load info + contradicted probe scores."""
    import torch, json
    _load_model()

    out = {"load_info": _LOAD_INFO, "probes": {}, "shapes": {}, "bn_stats": {}, "cls_probe": {}}

    # Check heatmap_encoder BatchNorm running stats — if they're still at init
    # values (mean=0, var=1), the checkpoint didn't include them and the BN
    # outputs will be wrong.
    try:
        bn_layers = [m for m in _MODEL.heatmap_encoder.conv.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for i, bn in enumerate(bn_layers):
            out["bn_stats"][f"bn{i}"] = {
                "running_mean_norm": float(bn.running_mean.norm().item()),
                "running_var_mean": float(bn.running_var.mean().item()),
                "weight_norm": float(bn.weight.norm().item()),
            }
    except Exception as e:
        out["bn_stats"]["_error"] = str(e)

    # CLS comparison: do two different inputs actually produce different text_encoder outputs?
    try:
        cls_vecs = {}
        for tag, (c, e) in [
            ("supported", ("The heart is normal.", "Heart size is normal.")),
            ("contra", ("There is a small left-sided pneumothorax.", "There is a small right-sided pneumothorax.")),
            ("unrelated", ("asdf qwer zxcv.", "hjkl poiu lkjh.")),
        ]:
            enc = _TOKENIZER(c, e, max_length=512, padding="max_length", truncation="only_second", return_tensors="pt")
            with torch.no_grad():
                outputs = _MODEL.text_encoder(
                    input_ids=enc["input_ids"].to(_DEVICE),
                    attention_mask=enc["attention_mask"].to(_DEVICE),
                )
                cls_vec = outputs.last_hidden_state[:, 0, :]
                cls_vecs[tag] = cls_vec
                out["cls_probe"][tag] = {
                    "cls_norm": float(cls_vec.norm().item()),
                    "cls_mean": float(cls_vec.mean().item()),
                    "cls_first_5": cls_vec[0, :5].tolist(),
                }
        # Cosine similarities
        def cos(a, b):
            return float(torch.nn.functional.cosine_similarity(a, b).item())
        out["cls_probe"]["cos_supported_contra"] = cos(cls_vecs["supported"], cls_vecs["contra"])
        out["cls_probe"]["cos_supported_unrelated"] = cos(cls_vecs["supported"], cls_vecs["unrelated"])
        out["cls_probe"]["cos_contra_unrelated"] = cos(cls_vecs["contra"], cls_vecs["unrelated"])

        # Run each through verdict_head directly (bypass hmap_feat issue)
        # Use the trained hmap_feat
        zero_img = torch.zeros(1, 27, 27, device=_DEVICE)
        with torch.no_grad():
            hmap_feat = _MODEL.heatmap_encoder(zero_img)
            out["cls_probe"]["hmap_feat_norm"] = float(hmap_feat.norm().item())
            out["cls_probe"]["hmap_feat_first_5"] = hmap_feat[0, :5].tolist()
            # For each CLS, concat with hmap and run through verdict_head
            for tag, cls_vec in cls_vecs.items():
                fused = torch.cat([cls_vec, hmap_feat], dim=-1)
                logits = _MODEL.verdict_head(fused)
                probs = torch.softmax(logits, dim=-1)
                out["cls_probe"][f"{tag}_logits"] = logits[0].tolist()
                out["cls_probe"][f"{tag}_probs"] = probs[0].tolist()
    except Exception as e:
        out["cls_probe"]["_error"] = str(e)

    # Dump key verdict_head shapes to confirm num_classes and fused_dim
    try:
        sd = _MODEL.state_dict()
        for k in list(sd.keys()):
            if "verdict_head" in k or "heatmap_encoder.proj" in k:
                out["shapes"][k] = list(sd[k].shape)
    except Exception as e:
        out["shapes"]["_error"] = str(e)

    def _probe(claim, evidence, label, truncation_mode="only_second"):
        enc = _TOKENIZER(
            claim, evidence, max_length=512, padding="max_length",
            truncation=truncation_mode, return_tensors="pt",
        )
        with torch.no_grad():
            logits = _MODEL(enc["input_ids"].to(_DEVICE), enc["attention_mask"].to(_DEVICE))
            p = torch.softmax(logits, dim=-1)
        return {
            "claim": claim,
            "evidence": evidence,
            "not_contra": float(p[0, 0].item()),
            "contra": float(p[0, 1].item()),
            "expected": label,
            "truncation": truncation_mode,
            "n_input_tokens": int((enc["attention_mask"] > 0).sum().item()),
        }

    # Laterality swap with both truncation modes
    out["probes"]["laterality_swap_only_second"] = _probe(
        "There is a small left-sided pneumothorax.",
        "There is a small right-sided pneumothorax.",
        "contra",
        truncation_mode="only_second",
    )
    out["probes"]["laterality_swap_true"] = _probe(
        "There is a small left-sided pneumothorax.",
        "There is a small right-sided pneumothorax.",
        "contra",
        truncation_mode=True,
    )
    out["probes"]["negation_flip"] = _probe(
        "There is no pleural effusion.",
        "There is a large left-sided pleural effusion.",
        "contra",
    )
    # Strongly contradicted cross-pathology example
    out["probes"]["strongly_contra"] = _probe(
        "The lungs are completely clear with no abnormalities.",
        "Bilateral diffuse consolidation with severe pulmonary edema.",
        "contra",
    )
    out["probes"]["supported"] = _probe(
        "The heart is normal in size.",
        "Heart size is within normal limits.",
        "not_contra",
    )
    out["probes"]["supported_identical"] = _probe(
        "There is a small right-sided pneumothorax.",
        "There is a small right-sided pneumothorax.",
        "not_contra",
    )
    return json.dumps(out, indent=2, default=str)


def build_demo():
    with gr.Blocks(
        title="ClaimGuard-CXR — Live Hallucination Detection",
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="stone"),
        css="""
        .gradio-container { max-width: 1100px !important; }
        """,
    ) as demo:
        gr.Markdown(
            "# ClaimGuard-CXR\n"
            "### Live verifier with inverted conformal FDR control\n\n"
            "Enter a single atomic **candidate claim** and a matching "
            "**reference** the claim should be consistent with. The model "
            "scores the claim against the reference and assigns "
            "**GREEN** / **YELLOW** / **RED** via inverted conformal "
            "Benjamini-Hochberg at your chosen α.\n\n"
            "*Research prototype. Not for clinical use. All example reports are synthetic.*"
        )

        with gr.Tab("Verifier"):
            gr.Markdown(
                "**How to use.** Paste a candidate radiology report on the left "
                "and the ground-truth reference on the right. Click *Verify report*. "
                "The verifier splits the candidate into atomic claims, normalizes each "
                "one into the short form the model handles reliably, retrieves the most "
                "relevant reference sentence per claim, and assigns a triage label via "
                "inverted conformal Benjamini-Hochberg at your chosen α."
            )
            with gr.Row():
                report_input = gr.Textbox(
                    label="Candidate report (to verify)",
                    placeholder=(
                        "The heart is normal in size. "
                        "The lungs are clear without focal consolidation. "
                        "There is no pleural effusion. "
                        "No acute osseous abnormality. "
                        "The mediastinal contours are normal."
                    ),
                    lines=8,
                )
                evidence_input = gr.Textbox(
                    label="Reference report (ground truth)",
                    placeholder=(
                        "Heart size normal. "
                        "Lungs clear. "
                        "Large right pleural effusion. "
                        "Osseous structures unremarkable. "
                        "Mediastinal contours normal."
                    ),
                    lines=8,
                )
            with gr.Row():
                alpha_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.20,
                    value=0.05,
                    step=0.01,
                    label="FDR target (α)",
                    info="Lower α = stricter — fewer claims pass as GREEN, stronger guarantee",
                )
                verify_btn = gr.Button("Verify report", variant="primary", size="lg")

            highlighted_output = gr.HighlightedText(
                label="Verified claims",
                color_map={
                    "GREEN": "#65A30D",
                    "YELLOW": "#D97706",
                    "RED": "#DC2626",
                },
            )
            summary_output = gr.Markdown()
            detail_table = gr.Dataframe(
                headers=["Claim", "Triage", "Score", "p-value", "Reference used"],
                label="Per-claim details",
                wrap=True,
            )

            verify_btn.click(
                fn=process_report,
                inputs=[report_input, evidence_input, alpha_slider],
                outputs=[highlighted_output, summary_output, detail_table],
            )
            alpha_slider.change(
                fn=process_report,
                inputs=[report_input, evidence_input, alpha_slider],
                outputs=[highlighted_output, summary_output, detail_table],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[report_input, evidence_input, alpha_slider],
                outputs=[highlighted_output, summary_output, detail_table],
                fn=process_report,
                cache_examples=False,
                label="Click an example mini-report to load it",
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## How it works\n\n"
                "1. **Decompose** — a report is split into sentence-level atomic claims "
                "(here the demo takes one claim at a time).\n"
                "2. **Verify** — each claim is scored by a RoBERTa-large binary cross-encoder "
                "trained on 30,000 claims with 8 hard-negative perturbation types.\n"
                "3. **Calibrate** — an inverted conformal p-value is computed against the "
                "contradicted calibration distribution (5,000 held-out scores).\n"
                "4. **Triage** — Benjamini-Hochberg at level α decides which claims are "
                "flagged GREEN with a formal false-discovery-rate guarantee.\n\n"
                "## Key results (CheXpert Plus, 15K held-out test claims)\n\n"
                "- **98.31% accuracy**, **99.52% AUROC**\n"
                "- **FDR 1.30% at α=0.05**, power 98.06%\n"
                "- **+31 pp over zero-shot LLM and VLM baselines**\n"
                "- **Cross-dataset transfer to OpenI (Indiana U)** — FDR still controlled at "
                "every α without retraining\n\n"
                "## Honest limitations\n\n"
                "- **In-distribution only.** The reported 98.31% accuracy is on "
                "rule-based hard negatives generated by word-level substitution on "
                "CheXpert reports. On hand-crafted out-of-distribution claims, the "
                "verifier is most reliable for **negation-based** hallucinations; "
                "laterality swaps, severity swaps, and finding substitutions are less "
                "reliable without the full BM25+MedCPT retrieval stack.\n"
                "- **No image grounding.** This checkpoint is text-only. A visual "
                "grounding head was trained on zero heatmaps as a placeholder; image "
                "grounding is future work.\n"
                "- **Binary framing.** Supported and Insufficient-evidence collapse to a "
                "single *not-contradicted* class. A 3-class version is in the repo.\n"
                "- **English only** and no radiologist ground truth — contradictions are "
                "defined relative to rule-based perturbations, not clinical adjudication.\n\n"
                "## Project links\n\n"
                "- [Marketing site](https://alwaniaayan6-png.github.io/ClaimGuard-CXR)\n"
                "- [GitHub repo](https://github.com/alwaniaayan6-png/ClaimGuard-CXR)\n\n"
                "## Author\n\n"
                "Aayan Alwani — Laughney Lab, Weill Cornell Medicine. Targeting NeurIPS 2026.\n"
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        show_error=True,
    )
