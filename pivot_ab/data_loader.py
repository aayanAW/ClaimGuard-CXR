"""Counterfactual-augmented data loader for Pivot A+B SFT on MedGemma-4B-IT.

For each training row, returns the original (image, prompt, target) packaged
through MedGemma's chat-template processor. The collator stacks rows into a
batch and produces the full / masked / flipped pixel_values variants needed
by the bidirectional causal-faithfulness loss. Image-flipping is only applied
to laterality-sensitive rows; on others the flipped tensor is the same as
the masked tensor (both effectively unused by the loss).

The dataset returns a `target_mask` (1 on assistant-turn token positions,
0 on system + user turns) so the SFT loss and the faithfulness loss are
computed only over the supervised target tokens.

Designed to plug into a custom training loop in `train.py`. See
`PIVOT_AB_FAITHFUL_GENERATOR_PLAN.md` for the strategic context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Same regex used in v5/eval/evidence_blindness.py for IPG laterality detection.
LATERALITY_REGEX = re.compile(r"\b(left|right|bilateral)\b", re.IGNORECASE)


def is_laterality_sensitive(text: str) -> bool:
    """Return True if the claim text references laterality."""
    return bool(LATERALITY_REGEX.search(text or ""))


@dataclass
class CounterfactualBatch:
    """Bundles full/masked/flipped variants of a batch."""

    pixel_values_full: torch.Tensor          # (B, ...) — processor-shaped
    pixel_values_masked: torch.Tensor        # per-channel mean of full (neutral image, post-2026-05-03 fix)
    pixel_values_flipped: torch.Tensor | None  # full.flip(dims=[-1]); None if no laterality rows. Non-laterality rows get the neutral image as a no-op fallback
    flipped_mask: torch.Tensor               # (B,) bool — True if row is laterality-sensitive
    input_ids: torch.Tensor                  # (B, T)
    attention_mask: torch.Tensor             # (B, T)
    labels: torch.Tensor                     # (B, T) — copy of input_ids; loss masked via target_mask
    target_mask: torch.Tensor                # (B, T) — 1 on assistant-turn target tokens
    row_weights: torch.Tensor                # (B,)


def _resolve_image_path(image_path: str) -> str:
    """Try a few common path roots for an image_path that may be relative.

    Modal volume convention: file paths in JSONLs may be either absolute
    (already prefixed with /data) or relative to the volume root. Try both.
    """
    candidates = [image_path, f"/data/{image_path.lstrip('/')}", f"/data/{image_path}"]
    for c in candidates:
        if Path(c).exists():
            return c
    return image_path


class MedGemmaSFTDataset(Dataset):
    """Generator-side SFT dataset using MedGemma's chat template.

    Each example builds a 3-turn conversation:
        [system] You are an expert radiologist. Generate a findings-section
                 report consistent with the chest radiograph.
        [user]   Claim: {claim}
                 Evidence: {evidence}
                 + image
        [assistant] {reference_report}

    The processor's apply_chat_template returns input_ids (with image-token
    spans), attention_mask, and pixel_values. We compute target_mask by
    diff'ing the with-assistant and without-assistant token lengths.

    The dataset materialises one row at a time. The collate function
    (`counterfactual_collate`) handles padding and the counterfactual
    pixel_values variants.
    """

    SYSTEM_PROMPT = (
        "You are an expert radiologist. Generate a findings-section report "
        "consistent with the chest radiograph."
    )

    def __init__(
        self,
        *,
        jsonl_path: Path,
        weights_jsonl: Path | None,
        processor: Any,
        max_total_tokens: int = 768,
    ):
        """
        Args:
            jsonl_path: training JSONL with rows containing
                {image_path, claim_text, evidence_text, gt_label,
                 reference_report (or claim_text-derived target)}.
            weights_jsonl: optional JSONL of {row_idx, weight} from
                `combine_dual_filters`. If None, all weights default to 1.0.
            processor: MedGemma's AutoProcessor instance.
        """
        # Pivot A+B is a *generator* paper: training requires a real text target
        # (multi-token report), not a verdict label. We therefore filter to rows
        # where either `reference_report` or `evidence_text` provides a non-empty
        # supervised target. Rows whose only supervisable text is the verdict
        # label (gt_label) are skipped — training on a 1-2-token target would
        # collapse the model to a verifier and break the IMG_correct diagnostic.
        # See PIVOT_AB_EXECUTION_LOG.md "Training-target / framing misalignment"
        # 2026-05-05 finding for full rationale.
        #
        # Indexing convention. The upstream filter scripts (`v5/ho_filter.py`
        # and `pivot_ab/image_only_filter.py`) build their own list of S/C-only
        # rows BEFORE writing weights, so a weight row {"row_idx": k, "weight": w}
        # means "the k-th S/C row in the JSONL," NOT the k-th raw line. The
        # JSONL contains many non-S/C rows (e.g. NO_GT) so raw-line-pos and
        # S/C-pos diverge from line 1 onwards. The `sc_idx` counter below
        # matches the upstream filter convention so weight lookup is correct.
        self.rows: list[dict] = []
        self._sc_to_new_idx: dict[int, int] = {}
        sc_idx = -1  # increments only on S/C-eligible rows (matches upstream filter convention)
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("gt_label") not in {"SUPPORTED", "CONTRADICTED"}:
                    continue
                sc_idx += 1
                ev = (r.get("evidence_text") or "").strip()
                ref = (r.get("reference_report") or "").strip()
                if not ev and not ref:
                    # No real text target available — verifier-shaped only;
                    # skip in generator training. sc_idx still advances so the
                    # mapping to the upstream filter's row_idx remains correct.
                    continue
                new_idx = len(self.rows)
                self._sc_to_new_idx[sc_idx] = new_idx
                self.rows.append(r)

        # Filter weights are stored against the upstream filter's S/C-position
        # row_idx. Remap to the new dataset indices so the training loop's
        # `row_weight` lookup hits correctly.
        self.weights: dict[int, float] = {}
        if weights_jsonl is not None:
            with weights_jsonl.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    sc = int(obj["row_idx"])
                    new = self._sc_to_new_idx.get(sc)
                    if new is None:
                        continue
                    self.weights[new] = float(obj["weight"])

        self.processor = processor
        self.max_total_tokens = max_total_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def _build_messages(self, row: dict, image: Image.Image, *, with_assistant: bool) -> list[dict]:
        # Generator framing: the user message provides the claim only (NOT the
        # evidence_text — evidence_text becomes the supervised target below, and
        # leaking it into the prompt would defeat the generation objective).
        prompt = f"Claim: {row.get('claim_text', '')}"
        # Target priority: reference_report (gold) > evidence_text (gold radiology
        # text from OpenI) > gt_label (verifier fallback; should be unreachable
        # because the dataset filter in __init__ rejects rows lacking text).
        target = (
            row.get("reference_report")
            or row.get("evidence_text")
            or row.get("gt_label", "")
        )
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        if with_assistant:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return msgs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = _resolve_image_path(row["image_path"])
        with Image.open(image_path) as im:
            img = im.convert("RGB")

        # Encode with assistant turn (training input + supervision target)
        msgs_full = self._build_messages(row, img, with_assistant=True)
        enc_full = self.processor.apply_chat_template(
            msgs_full,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Encode without assistant turn (just system + user) to get prompt length
        msgs_prompt = self._build_messages(row, img, with_assistant=False)
        enc_prompt = self.processor.apply_chat_template(
            msgs_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = enc_prompt["input_ids"].shape[-1]
        total_len = enc_full["input_ids"].shape[-1]

        # Truncate if the row produced more tokens than max_total_tokens.
        # This caps memory/wall-clock spikes from unusually long target reports.
        # Truncation drops tokens from the END of the assistant turn, which is
        # the right policy for SFT: the early target tokens still get
        # supervision, just not the tail.
        if total_len > self.max_total_tokens:
            total_len = self.max_total_tokens

        # target_mask: 1 on assistant tokens (positions [prompt_len, total_len))
        target_mask = torch.zeros(total_len, dtype=torch.long)
        if total_len > prompt_len:
            target_mask[prompt_len:] = 1

        input_ids = enc_full["input_ids"].squeeze(0)[:total_len]
        attention_mask = enc_full["attention_mask"].squeeze(0)[:total_len]

        weight = self.weights.get(idx, 1.0)
        flipped_eligible = is_laterality_sensitive(row.get("claim_text", ""))

        return {
            "pixel_values": enc_full["pixel_values"].squeeze(0),  # (C, H, W) or (N, C, H, W)
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "row_weight": torch.tensor(weight, dtype=torch.float32),
            "flipped_eligible": torch.tensor(flipped_eligible, dtype=torch.bool),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }


def _pad_to(tensor: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
    """Right-pad a 1-D tensor to target_len with pad_value."""
    cur_len = tensor.shape[0]
    if cur_len >= target_len:
        return tensor[:target_len]
    pad = torch.full((target_len - cur_len,), pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def _make_neutral_image_tensor(pixel_values: torch.Tensor) -> torch.Tensor:
    """Create the model's "neutral" image tensor — a uniform per-channel mean
    in pixel-value space *after* the processor's normalisation. We compute
    the per-channel mean of the input batch's own pixel_values; this is a
    self-consistent neutral that does not require knowing the processor's
    exact normalisation constants. The result has the same shape as
    pixel_values and represents "image is uniform" rather than "image is
    OOD all-zeros".

    Falls back to zeros_like if the input is degenerate (uniform across
    spatial dims already).
    """
    spatial_dims = tuple(range(2, pixel_values.dim()))
    if not spatial_dims:
        return torch.zeros_like(pixel_values)
    # mean over spatial dims, keep batch + channel dims
    means = pixel_values.mean(dim=spatial_dims, keepdim=True)
    return means.expand_as(pixel_values).clone()


def counterfactual_collate(
    batch: list[dict[str, Any]],
    *,
    enable_flipping: bool = True,
    pad_token_id: int = 0,
) -> CounterfactualBatch:
    """Collate variable-length examples into a CounterfactualBatch.

    Padding policy:
        - input_ids and labels padded with `pad_token_id` (defaults 0)
        - attention_mask and target_mask padded with 0
        - pixel_values stacked directly (assumes a fixed image-tensor shape
          per row, which is true for MedGemma's processor since it always
          resizes to a fixed grid)
        - pixel_values_masked uses the per-image per-channel mean (a uniform
          "gray" image at the model's own normalisation), NOT zeros_like —
          zeros post-normalisation are an OOD adversarial input rather than
          "image absent". See faith_loss.py docstring for the methodological
          rationale.
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.stack([_pad_to(b["input_ids"], max_len, pad_token_id) for b in batch])
    attention_mask = torch.stack([_pad_to(b["attention_mask"], max_len, 0) for b in batch])
    target_mask = torch.stack([_pad_to(b["target_mask"], max_len, 0) for b in batch])
    labels = input_ids.clone()  # loss masked via target_mask, not labels
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    row_weights = torch.stack([b["row_weight"] for b in batch])
    flipped_eligible = torch.stack([b["flipped_eligible"] for b in batch])

    pixel_values_masked = _make_neutral_image_tensor(pixel_values)

    pixel_values_flipped: torch.Tensor | None = None
    if enable_flipping and flipped_eligible.any():
        pixel_values_flipped = pixel_values.flip(dims=[-1])
        # On rows where flipping doesn't apply, replace with the neutral image
        # so the forward pass produces non-spurious activations; the faith
        # loss masks these rows out via flipped_row_mask anyway, but the
        # neutral image is a safer fallback than zeros.
        non_flip = (~flipped_eligible).view(-1, *([1] * (pixel_values_flipped.dim() - 1)))
        neutral = _make_neutral_image_tensor(pixel_values)
        pixel_values_flipped = torch.where(non_flip, neutral, pixel_values_flipped)

    return CounterfactualBatch(
        pixel_values_full=pixel_values,
        pixel_values_masked=pixel_values_masked,
        pixel_values_flipped=pixel_values_flipped,
        flipped_mask=flipped_eligible,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        target_mask=target_mask,
        row_weights=row_weights,
    )


# ---- Backwards-compatibility alias for the existing tests ----------------
# The legacy test_data_loader.py references CounterfactualSFTDataset; re-export
# the new MedGemma class under that name so existing tests still import.
CounterfactualSFTDataset = MedGemmaSFTDataset
