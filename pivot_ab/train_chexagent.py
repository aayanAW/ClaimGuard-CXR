"""LoRA fine-tuning entry point for MedGemma-4B-IT with the Pivot A+B objective.

(Filename retained for backwards-compat with import paths; backbone switched
from CheXagent-2-3B to MedGemma-4B-IT after the 2026-05-03 smoke test
revealed CheXagent's tokenizer-embedded image paradigm blocks
counterfactual training. See PIVOT_AB_EXECUTION_LOG.md.)

Composite per-step objective:

    L = mean over batch of:
            row_weight * NLL_full(target | full input)            # SFT, dual-filter weighted
        + lambda_faith * (
              KL-hinge( p_full || p_masked )                      # bidirectional
            + KL-hinge( p_full || p_flipped ) on laterality rows  # faithfulness
          )

Each training step runs 2 forward passes (3 on laterality batches): full,
image-masked, image-flipped. Image-flipped is conditional on at least one
row in the batch being laterality-sensitive; otherwise we save the third pass.

Gradient hooks from `grad_monitor.py` are installed before the first step
and log every K steps. Final monitor trace is saved as JSONL.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data_loader import CounterfactualBatch, MedGemmaSFTDataset, counterfactual_collate
from .faith_loss import FaithLossOutput, causal_faithfulness_loss
from .grad_monitor import GradMonitor

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Backbone (MedGemma-4B-IT; standard AutoModelForImageTextToText interface)
    backbone: str = "google/medgemma-4b-it"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # Data
    train_jsonl: str = ""
    val_jsonl: str = ""
    weights_jsonl: str = ""  # dual filter weights; "" means uniform 1.0
    # Optimization
    n_epochs: int = 2
    batch_size: int = 2
    grad_accum_steps: int = 8
    lr_lora: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    # Faithfulness
    enable_faith_loss: bool = True
    enable_dual_filter: bool = True
    enable_flipping: bool = True
    lambda_faith: float = 0.3
    margin_kl: float = 1.0
    # Monitor
    monitor_log_every: int = 50
    # Misc
    seed: int = 42
    bf16: bool = True
    output_dir: str = ""
    log_every: int = 20
    save_every_steps: int = 500
    max_steps: int = 0  # if > 0, stop after this many optimizer steps (smoke tests)


def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _forward(model, *, pixel_values, input_ids, attention_mask):
    """Forward pass returning per-token log-probs (B, T, V)."""
    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    log_probs = F.log_softmax(out.logits, dim=-1)
    return log_probs, out.logits


def _shifted_nll(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Causal-LM-style shifted next-token NLL, masked by target_mask.

    Returns a per-row mean NLL over the supervised positions, shape (B,).
    """
    # Shift: predict labels[t] from logits[t-1]
    log_probs_shift = log_probs[:, :-1, :]
    labels_shift = labels[:, 1:]
    mask_shift = target_mask[:, 1:].float()

    nll_per_token = -log_probs_shift.gather(2, labels_shift.unsqueeze(-1)).squeeze(-1)
    nll_per_token = nll_per_token * mask_shift
    nll_per_row = nll_per_token.sum(dim=-1) / mask_shift.sum(dim=-1).clamp_min(1.0)
    return nll_per_row


def _shifted_log_p_for_faith(
    log_probs_full: torch.Tensor,
    log_probs_masked: torch.Tensor,
    log_probs_flipped: torch.Tensor | None,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
):
    """Shift the log-prob tensors and the labels/mask so they align as causal-LM
    next-token predictions, then forward to causal_faithfulness_loss."""
    log_probs_full_s = log_probs_full[:, :-1, :]
    log_probs_masked_s = log_probs_masked[:, :-1, :]
    log_probs_flipped_s = log_probs_flipped[:, :-1, :] if log_probs_flipped is not None else None
    labels_s = labels[:, 1:]
    mask_s = target_mask[:, 1:].float()
    return log_probs_full_s, log_probs_masked_s, log_probs_flipped_s, labels_s, mask_s


def _find_latest_checkpoint(out_dir: Path) -> tuple[Path, int] | None:
    """Find the highest-step checkpoint under out_dir/checkpoints, returning
    (path, step). Returns (final_path, -1) if a 'final' checkpoint exists
    (signifying the run is already done). Returns None if no checkpoints."""
    ckpt_dir = out_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    final = ckpt_dir / "final"
    if final.exists():
        return final, -1
    candidates = []
    for p in ckpt_dir.iterdir():
        if p.name.startswith("step_"):
            try:
                step = int(p.name[5:])
                candidates.append((p, step))
            except ValueError:
                continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])


def train(cfg: TrainConfig) -> dict:
    """Main entry point. Returns a summary dict.

    Resume support: if `out_dir/checkpoints/` contains intermediate or final
    checkpoints, training resumes from the latest one. Useful for Modal
    preemption recovery — repeated launches will pick up where they left
    off rather than restart from scratch.
    """
    _set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "train_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Check for prior checkpoints to resume from
    resume_info = _find_latest_checkpoint(out_dir)
    resume_step = 0
    resume_from_path: Path | None = None
    if resume_info is not None:
        path, step = resume_info
        if step == -1:
            logger.info("Run already completed (final checkpoint exists at %s); exiting.", path)
            return {
                "status": "already_done",
                "n_train_examples": -1,
                "global_steps": -1,
                "n_epochs": cfg.n_epochs,
                "output_dir": str(out_dir),
                "resumed_from": str(path),
            }
        logger.info("Resuming from checkpoint %s (step %d)", path, step)
        resume_step = step
        resume_from_path = path

    logger.info("Loading backbone %s", cfg.backbone)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.backbone)
    base_model = AutoModelForImageTextToText.from_pretrained(
        cfg.backbone,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )

    from peft import LoraConfig, get_peft_model, PeftModel

    if resume_from_path is not None:
        logger.info("Loading LoRA adapters from %s", resume_from_path)
        model = PeftModel.from_pretrained(base_model, str(resume_from_path), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA-wrapped: %.2fM trainable / %.1fM total", n_trainable / 1e6, n_total / 1e6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weights_jsonl = Path(cfg.weights_jsonl) if cfg.weights_jsonl else None
    train_ds = MedGemmaSFTDataset(
        jsonl_path=Path(cfg.train_jsonl),
        weights_jsonl=weights_jsonl,
        processor=processor,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"empty training dataset at {cfg.train_jsonl}")

    val_ds = None
    if cfg.val_jsonl:
        val_ds = MedGemmaSFTDataset(
            jsonl_path=Path(cfg.val_jsonl),
            weights_jsonl=None,
            processor=processor,
        )

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: counterfactual_collate(
            b, enable_flipping=cfg.enable_flipping, pad_token_id=pad_id
        ),
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_lora,
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, (len(train_loader) // cfg.grad_accum_steps) * cfg.n_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, cfg.warmup_steps, total_steps),
    )

    monitor: GradMonitor | None = None
    # Resolve the actual image-soft-token id from the processor. MedGemma's
    # AutoProcessor uses a special token (typically <image_soft_token>) at
    # the position(s) where image tokens are spliced into input_ids. The
    # monitor must mask these EXACT positions, not a heuristic prefix.
    image_token_id = None
    for cand in ("<image_soft_token>", "<image>", "<|image|>", "<image_placeholder>"):
        try:
            tid = processor.tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid != processor.tokenizer.unk_token_id:
                image_token_id = tid
                logger.info("Resolved image token %r → id=%d for monitor mask", cand, tid)
                break
        except Exception:
            continue

    if image_token_id is None:
        logger.warning(
            "Could not resolve an image-soft-token id from the processor; "
            "gradient monitor will be DISABLED to avoid reporting bogus "
            "image-side gradients."
        )
    else:
        try:
            def _image_mask_fn(batch_dict, _img_id=image_token_id):
                ids = batch_dict["input_ids"]
                return ids == _img_id

            monitor = GradMonitor(
                model=model,
                image_token_indices_fn=_image_mask_fn,
                cross_modal_attention_layer=None,
                log_every=cfg.monitor_log_every,
            )
        except RuntimeError as e:
            logger.warning("could not install grad monitor: %s", e)

    global_step = resume_step
    metrics_log: list[dict] = []
    if resume_step > 0:
        # Skip ahead in the LR schedule to match the resumed step.
        for _ in range(resume_step):
            scheduler.step()
        logger.info("LR schedule advanced to resumed step %d", resume_step)
    for epoch in range(cfg.n_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = _move_to_device(batch, device)
            if monitor is not None:
                monitor.attach_batch(_batch_to_dict(batch))

            log_p_full, _ = _forward(
                model,
                pixel_values=batch.pixel_values_full,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )

            # SFT loss (per-row, weighted by dual-filter sample weights)
            nll_per_row = _shifted_nll(log_p_full, batch.labels, batch.target_mask)
            sft_loss = (nll_per_row * batch.row_weights).mean()

            log_p_masked = None
            log_p_flipped = None
            if cfg.enable_faith_loss:
                log_p_masked, _ = _forward(
                    model,
                    pixel_values=batch.pixel_values_masked,
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )
                if (
                    batch.pixel_values_flipped is not None
                    and batch.flipped_mask.any()
                    and cfg.enable_flipping
                ):
                    log_p_flipped, _ = _forward(
                        model,
                        pixel_values=batch.pixel_values_flipped,
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                    )
                lp_full_s, lp_masked_s, lp_flipped_s, labels_s, mask_s = _shifted_log_p_for_faith(
                    log_p_full, log_p_masked, log_p_flipped, batch.labels, batch.target_mask
                )
                faith = causal_faithfulness_loss(
                    log_p_full=lp_full_s,
                    log_p_masked=lp_masked_s,
                    log_p_flipped=lp_flipped_s,
                    target_ids=labels_s,
                    target_mask=mask_s,
                    margin_kl=cfg.margin_kl,
                    lambda_diverge=1.0,
                    lambda_correct=0.0,  # correctness already in sft_loss
                    flipped_row_mask=batch.flipped_mask,  # mask out non-laterality rows
                )
                total_loss = sft_loss + cfg.lambda_faith * faith.diverge
            else:
                faith = FaithLossOutput(
                    total=sft_loss,
                    correct=sft_loss,
                    diverge=torch.tensor(0.0, device=device),
                    diverge_masked=None,
                    diverge_flipped=None,
                )
                total_loss = sft_loss

            (total_loss / cfg.grad_accum_steps).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                if monitor is not None:
                    monitor.step(global_step)
                if cfg.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    metrics = {
                        "epoch": epoch,
                        "step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "sft_loss": float(sft_loss.detach()),
                        "faith_diverge": float(faith.diverge.detach()),
                        "total_loss": float(total_loss.detach()),
                        "row_weight_mean": float(batch.row_weights.mean()),
                    }
                    metrics_log.append(metrics)
                    logger.info("step %d %s", global_step, metrics)

                if global_step % cfg.save_every_steps == 0:
                    _save_checkpoint(model, out_dir, global_step)

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    logger.info("Reached max_steps=%d; stopping early.", cfg.max_steps)
                    break
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    _save_checkpoint(model, out_dir, global_step, final=True)

    if monitor is not None:
        monitor.close()
        monitor.trace.to_jsonl(out_dir / "monitor_trace.jsonl")

    with (out_dir / "metrics.jsonl").open("w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    return {
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds) if val_ds is not None else 0,
        "global_steps": global_step,
        "n_epochs": cfg.n_epochs,
        "n_trainable_M": n_trainable / 1e6,
        "output_dir": str(out_dir),
    }


def _move_to_device(batch: CounterfactualBatch, device: torch.device) -> CounterfactualBatch:
    return CounterfactualBatch(
        pixel_values_full=batch.pixel_values_full.to(device, non_blocking=True),
        pixel_values_masked=batch.pixel_values_masked.to(device, non_blocking=True),
        pixel_values_flipped=batch.pixel_values_flipped.to(device, non_blocking=True)
        if batch.pixel_values_flipped is not None
        else None,
        flipped_mask=batch.flipped_mask.to(device, non_blocking=True),
        input_ids=batch.input_ids.to(device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
        target_mask=batch.target_mask.to(device, non_blocking=True),
        row_weights=batch.row_weights.to(device, non_blocking=True),
    )


def _batch_to_dict(batch: CounterfactualBatch) -> dict[str, Any]:
    return {
        "pixel_values": batch.pixel_values_full,
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
    }


def _save_checkpoint(model, out_dir: Path, step: int, final: bool = False) -> None:
    name = "final" if final else f"step_{step}"
    path = out_dir / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    # Drop legacy/unused keys (filename retained for backwards-compat)
    for legacy_key in ("lr_head", "max_text_tokens", "max_target_tokens", "image_size", "prompt_template"):
        raw.pop(legacy_key, None)
    cfg = TrainConfig(**raw)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = train(cfg)
    print(json.dumps(summary, indent=2))
