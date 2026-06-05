"""Image-only adversarial filter (vision-side counterpart to v5/ho_filter.py).

The image-only filter is the symmetric pair of the existing text-only HO filter.
It identifies training rows where the verdict is solvable from the image alone
(no claim text, no evidence text). Rows flagged by either the text-only filter
OR this image-only filter are downweighted in SFT, preventing the model from
relying on either modality as a shortcut.

This is the genuine novelty axis of Pivot A+B vs Liu et al. 2025, who only
implement the text-only side.

Pipeline:
1. Train a vision-only classifier (BiomedCLIP image encoder + 2-layer MLP head)
   on (image, label) pairs. The classifier never sees the claim or evidence.
2. Score every training row. Rows where the classifier predicts the true label
   above `confidence_threshold` are flagged as image-shortcut-solvable.
3. Write per-row weights to a JSONL: flagged rows get `downweight` (default 0.2),
   the rest stay at 1.0.

The dual_filter.py module then combines this with the text-only filter via
`weight = min(text_weight, image_weight)` (most conservative — downweight if
EITHER side solves the row).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class _ImageOnlyDataset(Dataset):
    """Image-only dataset: image tensors → label.

    Reads the same GroundBench JSONL as the text-only filter but uses only the
    image_path field; claim_text and evidence_text are ignored.
    """

    def __init__(
        self,
        jsonl_path: Path,
        image_loader: Any,
        image_size: int = 224,
    ):
        self.rows: list[dict] = []
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}:
                    self.rows.append(r)
        self.image_loader = image_loader
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = self.image_loader(row["image_path"], size=self.image_size)
        label = 1 if row["gt_label"] == "CONTRADICTED" else 0
        return {
            "image": image,
            "labels": torch.tensor(label, dtype=torch.long),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }


class _ImageOnlyClassifier(nn.Module):
    """Vision encoder + 2-layer MLP head."""

    def __init__(self, vision_encoder: nn.Module, hidden_dim: int, n_classes: int = 2):
        super().__init__()
        self.encoder = vision_encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Encoder returns either pooled (B, D) or token sequence (B, T, D)
        feats = self.encoder(image)
        if feats.ndim == 3:
            # Use CLS token if a sequence is returned
            feats = feats[:, 0]
        return self.head(feats)


def run_image_only_filter(
    *,
    train_jsonl: Path,
    output_weights_path: Path,
    image_loader: Any,
    vision_encoder: nn.Module,
    encoder_dim: int,
    device: torch.device | str = "cuda",
    image_size: int = 224,
    confidence_threshold: float = 0.7,
    downweight: float = 0.2,
    n_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 5e-5,
    seed: int = 17,
    freeze_encoder: bool = True,
) -> dict:
    """Train an image-only baseline on (image -> label) and write per-row weights.

    Args:
        train_jsonl: GroundBench training JSONL.
        output_weights_path: destination for per-row {row_idx, weight} JSONL.
        image_loader: callable(path, size) -> torch.Tensor of shape (3, H, W).
            Should match preprocessing used by `vision_encoder`.
        vision_encoder: torch module that accepts (B, 3, H, W) and returns either
            pooled (B, encoder_dim) or token-sequence (B, T, encoder_dim).
        encoder_dim: feature dimension emitted by `vision_encoder`.
        device: "cuda" or "cpu" or torch.device.
        image_size: image resolution to feed the encoder.
        confidence_threshold: rows where the vision baseline assigns the TRUE
            label a probability above this threshold are considered
            image-shortcut-solvable.
        downweight: training weight applied to flagged rows.
        n_epochs: training epochs.
        batch_size: batch size.
        lr: learning rate for the head (encoder uses lr/10 if unfrozen).
        seed: random seed.
        freeze_encoder: if True, only the MLP head is trained. Default True
            because the encoder is a strong pretrained backbone (BiomedCLIP)
            and fine-tuning it on a 1-epoch shortcut probe risks overfitting.

    Returns:
        Summary dict with counts and distributions.
    """
    torch.manual_seed(seed)
    device = torch.device(device) if not isinstance(device, torch.device) else device

    ds = _ImageOnlyDataset(train_jsonl, image_loader, image_size=image_size)
    if len(ds) == 0:
        raise RuntimeError(f"no resolved-GT rows in {train_jsonl}")
    loader_train = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_score = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    if freeze_encoder:
        for p in vision_encoder.parameters():
            p.requires_grad = False
        vision_encoder.eval()

    model = _ImageOnlyClassifier(vision_encoder, encoder_dim).to(device)
    head_params = list(model.head.parameters())
    if freeze_encoder:
        optimizer = AdamW(head_params, lr=lr)
    else:
        optimizer = AdamW(
            [
                {"params": head_params, "lr": lr},
                {"params": vision_encoder.parameters(), "lr": lr / 10},
            ]
        )

    # Train
    for epoch in range(n_epochs):
        model.train()
        if freeze_encoder:
            vision_encoder.eval()  # keep encoder in eval mode for frozen BN
        running = 0.0
        n_batches = 0
        for batch in loader_train:
            img = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(img)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1
        logger.info(
            "image-only filter epoch %d mean_loss=%.4f",
            epoch,
            running / max(1, n_batches),
        )

    # Score
    model.eval()
    weights = [1.0] * len(ds)
    n_downweighted = 0
    n_supported_down = 0
    n_contradicted_down = 0
    with torch.no_grad():
        for batch in loader_score:
            img = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            idx = batch["row_idx"]
            logits = model(img)
            probs = F.softmax(logits, dim=-1)
            true_label_prob = probs.gather(1, y.unsqueeze(-1)).squeeze(-1)
            for j in range(img.size(0)):
                if float(true_label_prob[j]) > confidence_threshold:
                    weights[int(idx[j])] = downweight
                    n_downweighted += 1
                    if int(y[j]) == 0:
                        n_supported_down += 1
                    else:
                        n_contradicted_down += 1

    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with output_weights_path.open("w") as f:
        for i, w in enumerate(weights):
            f.write(json.dumps({"row_idx": i, "weight": w}) + "\n")

    summary = {
        "filter_kind": "image_only",
        "n_rows": len(ds),
        "n_downweighted": n_downweighted,
        "fraction_downweighted": n_downweighted / max(1, len(ds)),
        "n_supported_downweighted": n_supported_down,
        "n_contradicted_downweighted": n_contradicted_down,
        "confidence_threshold": confidence_threshold,
        "downweight": downweight,
        "weights_path": str(output_weights_path),
    }
    logger.info("image-only filter summary: %s", summary)
    return summary
