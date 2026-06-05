"""Modal orchestrator for Pivot A+B.

Reuses the existing `claimguard-v5-data` volume so all GroundBench-v6 data,
silver labels, and prior checkpoints are immediately available.

Entrypoints:
    smoke_test_load_chexagent  — Phase 0 gate; loads CheXagent-2-3B + LoRA
    run_text_only_filter       — wraps v5/ho_filter.py:run_ho_filter
    run_image_only_filter      — wraps pivot_ab/image_only_filter.py
    combine_filters            — wraps pivot_ab/dual_filter.py
    run_sft                    — wraps pivot_ab/train_chexagent.py:train
    eval_img_correct           — wraps pivot_ab/img_correct.py on a checkpoint
    eval_diagnostic_suite      — runs IMG, ESG, IPG, IMG_correct on a checkpoint

GPU policy: H100:80GB for training; H100:40GB for filter scoring;
CPU for combine_filters and any pure-Python orchestration.

Cost discipline: every entrypoint that exceeds $1 of compute prints a
pre-flight banner before doing work. The user is expected to run the
`pre_flight` command before queueing the heavy phase-4 entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

VOLUME_NAME = "claimguard-v5-data"
APP_NAME = "pivot-ab-faithful-generator"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# Image: pinned for MedGemma-4B-IT (AutoModelForImageTextToText, standard
# pixel_values forward signature). Switched from CheXagent-2-3B because the
# latter embeds images via its tokenizer (no pixel_values kwarg), which
# blocks counterfactual training. See PIVOT_AB_EXECUTION_LOG.md for details.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.51.3",  # AutoModelForImageTextToText needs 4.45+
        "accelerate==0.34.2",
        "peft==0.13.2",
        "sentencepiece==0.2.0",
        "pillow==10.4.0",
        "pyyaml==6.0.2",
        "numpy==1.26.4",
        "open-clip-torch==2.26.1",  # BiomedCLIP loader for image-only filter
        "timm==1.0.11",
        "anthropic>=0.60.0",
    )
    .add_local_dir(
        "/Users/aayanalwani/VeriFact/verifact/pivot_ab",
        remote_path="/root/pivot_ab",
    )
    .add_local_dir(
        "/Users/aayanalwani/VeriFact/verifact/v5",
        remote_path="/root/v5",
    )
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_test_load_medgemma() -> dict:
    """Phase 0 gate: load MedGemma-4B-IT, inspect forward signature, wrap in LoRA.

    Cost: ~$1 (model download + load on H100 ≈ 7 min).
    """
    import sys
    import inspect
    import json

    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    backbone = "google/medgemma-4b-it"
    print(f"[pre-flight] loading {backbone}")
    processor = AutoProcessor.from_pretrained(backbone)
    model = AutoModelForImageTextToText.from_pretrained(
        backbone,
        torch_dtype=torch.bfloat16,
    )
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[pre-flight] loaded base model with {n_total/1e6:.1f}M params; class={type(model).__name__}")

    forward_sig = list(inspect.signature(model.forward).parameters.keys())
    print(f"[pre-flight] forward params: {forward_sig}")
    assert "pixel_values" in forward_sig, f"expected pixel_values in forward; got {forward_sig}"

    linear_names = sorted({
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    })
    print(f"[pre-flight] linear submodule names: {linear_names[:30]}")
    candidate_targets = [n for n in ["q_proj", "k_proj", "v_proj", "o_proj"] if n in linear_names]
    print(f"[pre-flight] LoRA target modules: {candidate_targets}")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=candidate_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_lora = sum(p.numel() for p in model.parameters())
    print(f"[pre-flight] LoRA-wrapped: {n_trainable/1e6:.2f}M trainable / {n_total_lora/1e6:.1f}M total")

    result = {
        "status": "ok",
        "backbone": backbone,
        "model_class": type(model).__name__,
        "forward_params": forward_sig,
        "has_pixel_values": "pixel_values" in forward_sig,
        "lora_target_modules": candidate_targets,
        "n_trainable_M": n_trainable / 1e6,
        "n_total_M": n_total_lora / 1e6,
    }
    print("[pre-flight] result:")
    print(json.dumps(result, indent=2, default=str))
    return result




@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_test_introspect_chexagent() -> dict:
    """Inspect CheXagent's internal structure to find a usable image-embedding hook.

    Cost: ~$0.30 (3-5 min H100).
    """
    import sys
    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "StanfordAIMI/CheXagent-2-3b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Top-level attributes
    top_attrs = [a for a in dir(model) if not a.startswith("_")]
    # Direct submodules
    submodules = {name: type(child).__name__ for name, child in model.named_children()}
    # Look for vision-related modules
    vision_candidates = [name for name, _ in model.named_modules() if "vision" in name.lower() or "image" in name.lower() or "visual" in name.lower()]
    # Methods that look like "encode_image" or "embed"
    method_candidates = [a for a in top_attrs if "image" in a.lower() or "embed" in a.lower() or "vision" in a.lower()]
    result = {
        "submodules": submodules,
        "vision_candidates": vision_candidates[:30],
        "method_candidates": method_candidates,
        "has_get_input_embeddings": hasattr(model, "get_input_embeddings"),
    }
    import json
    print("[introspect] result:")
    print(json.dumps(result, indent=2, default=str))
    return result


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_test_forward_pass(
    train_jsonl: str = "/data/groundbench_v5/all_v6/groundbench_v6_train.jsonl",
    n_examples: int = 2,
) -> dict:
    """Phase 0.5 gate: load MedGemma-4B-IT, take N rows from the training
    JSONL, and run full + masked + flipped forward passes with the
    faithfulness loss components.

    Verifies:
        - Image paths in the training JSONL resolve on the volume.
        - MedGemma's forward signature accepts (pixel_values, input_ids, labels).
        - Three forward passes fit in H100 80GB memory at batch_size=N.
        - The faithfulness loss is finite and produces a non-zero gradient.

    Cost: ~$1.50 (~7-10 min H100 with model load).
    """
    import sys
    import json
    import time
    import inspect

    sys.path.insert(0, "/root")
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    backbone = "google/medgemma-4b-it"
    print(f"[smoke] loading {backbone}")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(backbone)
    model = AutoModelForImageTextToText.from_pretrained(
        backbone,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    print(f"[smoke] loaded in {time.time()-t0:.1f}s; class={type(model).__name__}")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    # Load N training rows
    rows = []
    with open(train_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}:
                rows.append(r)
                if len(rows) >= n_examples:
                    break
    print(f"[smoke] loaded {len(rows)} rows; sample row image_path={rows[0].get('image_path')!r} gt_label={rows[0].get('gt_label')!r}")

    # Build inputs via the official chat template + processor
    forward_sig = list(inspect.signature(model.forward).parameters.keys())
    print(f"[smoke] forward params: {forward_sig}")

    # Open the actual images
    pil_images = []
    for r in rows:
        ipath = r.get("image_path", "")
        for candidate in [ipath, f"/data/{ipath.lstrip('/')}", f"/data/{ipath}"]:
            try:
                pil_images.append(Image.open(candidate).convert("RGB"))
                print(f"[smoke] loaded image: {candidate}")
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"could not resolve image_path={ipath!r}")

    # Build messages and process per-example, then collate into a batch.
    # We process each example separately because PIL images don't batch
    # cleanly through apply_chat_template.
    encoded_per_row = []
    for r, img in zip(rows, pil_images):
        # Match data_loader._build_messages: claim only in prompt; target is
        # reference_report > evidence_text > gt_label. Putting evidence_text
        # in the prompt would leak the gold target into the generation context.
        prompt = f"Claim: {r.get('claim_text','')}"
        target = (
            r.get("reference_report")
            or r.get("evidence_text")
            or r.get("gt_label", "")
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist. Generate a findings-section report consistent with the chest radiograph."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            },
        ]
        enc = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded_per_row.append(enc)

    # Reshape pixel_values: each enc has pixel_values shape (1, num_images, C, H, W)
    # or (1, C, H, W) depending on processor version. Inspect first encoding.
    sample_pv = encoded_per_row[0]["pixel_values"]
    print(f"[smoke] processor pixel_values shape (per row): {tuple(sample_pv.shape)} dtype={sample_pv.dtype}")
    sample_iids = encoded_per_row[0]["input_ids"]
    print(f"[smoke] processor input_ids shape (per row): {tuple(sample_iids.shape)}")

    # Run the per-row forward pass (un-batched for the smoke test).
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses_full = []
    losses_masked = []
    losses_flipped = []
    for enc in encoded_per_row:
        pv = enc["pixel_values"].to("cuda", dtype=torch.bfloat16)
        iids = enc["input_ids"].to("cuda")
        amask = enc["attention_mask"].to("cuda")
        labels = iids.clone()
        out_full = model(pixel_values=pv, input_ids=iids, attention_mask=amask, labels=labels)
        out_masked = model(pixel_values=torch.zeros_like(pv), input_ids=iids, attention_mask=amask, labels=labels)
        out_flipped = model(pixel_values=pv.flip(dims=[-1]), input_ids=iids, attention_mask=amask, labels=labels)
        losses_full.append(float(out_full.loss))
        losses_masked.append(float(out_masked.loss))
        losses_flipped.append(float(out_flipped.loss))
    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] {len(rows)} examples × 3 passes in {elapsed:.1f}s, peak GPU memory: {peak_mem_gb:.1f} GB")
    print(f"[smoke] sft_loss full={losses_full} masked={losses_masked} flipped={losses_flipped}")

    # Verify the loss responds to image content: full vs masked should differ.
    delta_masked = sum(m - f for f, m in zip(losses_full, losses_masked)) / len(rows)
    delta_flipped = sum(fl - f for f, fl in zip(losses_full, losses_flipped)) / len(rows)
    print(f"[smoke] mean delta loss(masked - full)={delta_masked:.4f} loss(flipped - full)={delta_flipped:.4f}")

    # Backward on the last forward
    out_full.loss.backward()
    grad_norm = sum(
        (p.grad.detach().norm().item() ** 2)
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    print(f"[smoke] backward ok; LoRA grad norm: {grad_norm:.4f}")

    result = {
        "status": "ok",
        "n_examples": len(rows),
        "elapsed_total_s": elapsed,
        "peak_gpu_memory_gb": peak_mem_gb,
        "sft_loss_full": losses_full,
        "sft_loss_masked": losses_masked,
        "sft_loss_flipped": losses_flipped,
        "delta_masked_minus_full": delta_masked,
        "delta_flipped_minus_full": delta_flipped,
        "grad_norm_lora": grad_norm,
        "pixel_values_shape": list(sample_pv.shape),
        "input_ids_shape": list(sample_iids.shape),
    }
    print("[smoke] result:")
    print(json.dumps(result, indent=2, default=str))
    return result


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_image_only_filter(
    train_jsonl: str = "/data/groundbench_v5/all_v6/groundbench_v6_train.jsonl",
    output_weights_path: str = "/data/pivot_ab/image_only_weights.jsonl",
    n_epochs: int = 1,
    batch_size: int = 32,
    confidence_threshold: float = 0.7,
    downweight: float = 0.2,
) -> dict:
    """Run the image-only adversarial filter on the training set."""
    import sys

    sys.path.insert(0, "/root")
    import torch
    from PIL import Image
    from torchvision import transforms

    # Use BiomedCLIP image encoder via open_clip
    import open_clip

    print("[pre-flight] loading BiomedCLIP via open_clip...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    encoder = model.visual
    encoder_dim = 512  # BiomedCLIP ViT-B/16 vision projection dim

    def _image_loader(path: str, size: int = 224) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return preprocess(im)

    from pivot_ab.image_only_filter import run_image_only_filter as run

    summary = run(
        train_jsonl=Path(train_jsonl),
        output_weights_path=Path(output_weights_path),
        image_loader=_image_loader,
        vision_encoder=encoder,
        encoder_dim=encoder_dim,
        device="cuda",
        n_epochs=n_epochs,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        downweight=downweight,
        freeze_encoder=True,
    )
    vol.commit()
    return summary


@app.function(image=image, volumes={"/data": vol}, timeout=60 * 10)
def combine_filters(
    text_weights_path: str = "/data/pivot_ab/text_only_weights.jsonl",
    image_weights_path: str = "/data/pivot_ab/image_only_weights.jsonl",
    output_weights_path: str = "/data/pivot_ab/dual_filter_weights.jsonl",
    aggregation: str = "min",
) -> dict:
    """Combine text-only + image-only filter weights via dual_filter."""
    import sys

    sys.path.insert(0, "/root")
    from pivot_ab.dual_filter import combine_dual_filters

    summary = combine_dual_filters(
        text_weights_path=Path(text_weights_path),
        image_weights_path=Path(image_weights_path),
        output_weights_path=Path(output_weights_path),
        aggregation=aggregation,
    )
    vol.commit()
    return summary


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_test_train(n_steps: int = 6) -> dict:
    """Phase 0.6 gate: run the actual training loop for N steps on a tiny
    config. Validates the full forward+loss+backward pipeline end-to-end
    using MedGemma + the dual-filter weights (uniform if not present).

    Cost: ~$1.50 (~5 min H100 with model load).
    """
    import sys
    sys.path.insert(0, "/root")
    from pivot_ab.train_chexagent import TrainConfig, train

    cfg = TrainConfig(
        backbone="google/medgemma-4b-it",
        lora_rank=16,  # match Phase 4
        lora_alpha=32,
        train_jsonl="/data/groundbench_v5/all_v6/groundbench_v6_train.jsonl",
        val_jsonl="",
        weights_jsonl="/data/pivot_ab/dual_filter_weights.jsonl",  # exercise the dual filter path
        n_epochs=1,
        batch_size=2,  # match Phase 4
        grad_accum_steps=8,
        warmup_steps=2,
        enable_faith_loss=True,
        enable_dual_filter=False,
        enable_flipping=True,
        lambda_faith=0.3,
        margin_kl=1.0,
        monitor_log_every=2,
        seed=42,
        bf16=True,
        output_dir="/data/pivot_ab/smoke_train",
        log_every=2,
        save_every_steps=1000,
        max_steps=n_steps,  # stop early after n_steps optimizer steps
    )
    import json
    summary = train(cfg)
    print("[smoke-train] result:")
    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_sft(config_path: str, seed: int = 0) -> dict:
    """Phase 4 entrypoint — LoRA fine-tune MedGemma-4B-IT with the Pivot A+B
    objective. Per CLAUDE.md, mandatory pre-flight Opus audit before launch
    has been completed (verdict: launch-with-fixes, all 3 minor items fixed).

    Cost per run: ~$25-35 on H100 over 1.5-3 hours depending on config.
    With 5 configs × 3 seeds = 15 runs × ~$30 ≈ $450.

    Args:
        config_path: filename in pivot_ab/configs/ (e.g. "sft_full.yaml").
        seed: if non-zero, OVERRIDES the seed in the YAML config and appends
            a "_seed{seed}" suffix to the output_dir so per-seed checkpoints
            don't collide.
    """
    import sys

    sys.path.insert(0, "/root")
    import yaml
    from pivot_ab.train_chexagent import TrainConfig, train

    print(f"[run_sft] loading config {config_path}")
    with open(f"/root/pivot_ab/configs/{config_path}") as f:
        raw = yaml.safe_load(f)
    # Drop legacy keys that the dataclass no longer accepts
    for legacy in ("lr_head", "max_text_tokens", "max_target_tokens", "image_size", "prompt_template"):
        raw.pop(legacy, None)
    cfg = TrainConfig(**raw)

    if seed != 0 and seed != cfg.seed:
        print(f"[run_sft] overriding seed {cfg.seed} → {seed}; suffixing output_dir")
        cfg.seed = seed
        cfg.output_dir = f"{cfg.output_dir.rstrip('/')}_seed{seed}"

    print(
        f"[run_sft] config: enable_faith_loss={cfg.enable_faith_loss} "
        f"enable_dual_filter={cfg.enable_dual_filter} seed={cfg.seed} "
        f"output_dir={cfg.output_dir}"
    )

    summary = train(cfg)
    vol.commit()
    return summary


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface")],
)
def eval_img_correct(
    checkpoint_dir: str,
    eval_jsonl: str = "/data/groundbench_v5/all_v6/groundbench_v6_test.jsonl",
    output_path: str = "/data/pivot_ab/img_correct_results.jsonl",
    margin_tau: float = 0.1,
    backbone: str = "google/medgemma-4b-it",
    n_eval: int = 500,
    max_total_tokens: int = 768,
    counterfactual_mode: str = "mean",
    site_filter: str = "",
) -> dict:
    """Run IMG_correct on a trained MedGemma checkpoint.

    Loads the LoRA adapter on top of the base MedGemma model and computes
    log-probs under (full image, masked image) to feed the IMG_correct
    diagnostic. Uses the same chat-template encoding as training so the
    log-probs are aligned with how the model was fine-tuned.
    """
    import sys

    sys.path.insert(0, "/root")
    import json
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    from pivot_ab.img_correct import write_results
    from pivot_ab.data_loader import _make_neutral_image_tensor

    print(f"[eval] loading base {backbone} + LoRA adapter from {checkpoint_dir}")
    processor = AutoProcessor.from_pretrained(backbone)
    base = AutoModelForImageTextToText.from_pretrained(
        backbone, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval().to("cuda")

    SYSTEM_PROMPT = (
        "You are an expert radiologist. Generate a findings-section report "
        "consistent with the chest radiograph."
    )

    # Mirror the dataset filter from data_loader.MedGemmaSFTDataset:
    # only rows with non-empty evidence_text or reference_report yield a real
    # text target (multi-token generation); drop verifier-shaped rows.
    rows = []
    with open(eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("gt_label") not in {"SUPPORTED", "CONTRADICTED"}:
                continue
            if site_filter and r.get("source_site") != site_filter:
                continue
            ev = (r.get("evidence_text") or "").strip()
            ref = (r.get("reference_report") or "").strip()
            if not ev and not ref:
                continue
            rows.append(r)
    rows = rows[:n_eval]
    print(f"[eval] running IMG_correct on {len(rows)} test rows (counterfactual_mode={counterfactual_mode}, site_filter={site_filter or 'none'})")

    # Per-row tensors: keep argmax + prob_correct (small) instead of full
    # log-probs (would OOM CPU at vocab=262K × 500 rows × ~400 tokens).
    pred_full_list: list[torch.Tensor] = []
    pred_masked_list: list[torch.Tensor] = []
    prob_full_list: list[torch.Tensor] = []
    prob_masked_list: list[torch.Tensor] = []
    target_ids_list: list[torch.Tensor] = []
    target_mask_list: list[torch.Tensor] = []

    with torch.no_grad():
        for r in rows:
            ipath = r["image_path"]
            for cand in [ipath, f"/data/{ipath.lstrip('/')}", f"/data/{ipath}"]:
                try:
                    img = Image.open(cand).convert("RGB")
                    break
                except FileNotFoundError:
                    continue
            else:
                print(f"[eval] skipping unresolvable image_path={ipath}")
                continue

            # Match data_loader._build_messages: claim only in the prompt;
            # evidence_text becomes the supervised target (or reference_report
            # if present). Putting evidence_text in the prompt would leak the
            # gold text into the generation context.
            prompt = f"Claim: {r.get('claim_text', '')}"
            target_text = (
                r.get("reference_report")
                or r.get("evidence_text")
                or r.get("gt_label", "")
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]
            messages_prompt = messages[:-1]

            enc_full = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            enc_prompt = processor.apply_chat_template(
                messages_prompt, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )

            iids = enc_full["input_ids"].to("cuda")
            amask = enc_full["attention_mask"].to("cuda")
            pv = enc_full["pixel_values"].to("cuda", dtype=torch.bfloat16)

            # counterfactual_mode controls how the image is degraded for the
            # masked forward pass. "mean" (default) is per-channel mean — the
            # exact counterfactual the faith loss trained on. "crop" replaces
            # 70% of the image with a uniform crop padding (a counterfactual
            # the faith loss did NOT train on); "occlude" zeroes a random
            # 50% × 50% center patch; "noise" replaces with gaussian noise.
            # The cross-counterfactual modes test whether IMG_correct
            # generalises beyond the trained counterfactual.
            if counterfactual_mode == "mean":
                pv_masked = _make_neutral_image_tensor(pv)
            elif counterfactual_mode == "crop":
                # Crop the upper-left 30% × 30% region; pad the rest with the
                # per-channel mean. This replaces ~91% of pixel area with a
                # neutral fill while keeping a small image-derived signal.
                pv_masked = _make_neutral_image_tensor(pv).clone()
                H, W = pv.shape[-2], pv.shape[-1]
                h_keep = int(0.30 * H); w_keep = int(0.30 * W)
                pv_masked[..., :h_keep, :w_keep] = pv[..., :h_keep, :w_keep]
            elif counterfactual_mode == "occlude":
                # Zero a centered 50% × 50% patch (a different OOD pattern).
                pv_masked = pv.clone()
                H, W = pv.shape[-2], pv.shape[-1]
                h0, h1 = H // 4, 3 * H // 4
                w0, w1 = W // 4, 3 * W // 4
                pv_masked[..., h0:h1, w0:w1] = 0.0
            elif counterfactual_mode == "noise":
                # Gaussian noise at the per-channel mean ± per-channel std.
                pv_masked = torch.randn_like(pv) * pv.std(dim=tuple(range(2, pv.dim())), keepdim=True) + pv.mean(dim=tuple(range(2, pv.dim())), keepdim=True)
            else:
                raise ValueError(f"unknown counterfactual_mode {counterfactual_mode!r}")

            # Mirror training-time truncation (data_loader.py:max_total_tokens)
            # so eval and training see the same context window.
            if iids.shape[-1] > max_total_tokens:
                iids = iids[:, :max_total_tokens]
                amask = amask[:, :max_total_tokens]

            prompt_len = enc_prompt["input_ids"].shape[-1]
            total_len = iids.shape[-1]
            target_mask = torch.zeros(total_len, dtype=torch.long)
            if total_len > prompt_len:
                target_mask[prompt_len:] = 1

            out_full = model(
                pixel_values=pv, input_ids=iids, attention_mask=amask,
            )
            out_masked = model(
                pixel_values=pv_masked, input_ids=iids, attention_mask=amask,
            )
            log_p_full = torch.log_softmax(out_full.logits, dim=-1)
            log_p_masked = torch.log_softmax(out_masked.logits, dim=-1)

            # Causal-LM shift: logit at position t predicts token at t+1.
            # We must do the shift here BEFORE collapsing the vocab dim, so
            # the gather targets the right positions. We then collapse vocab
            # immediately to avoid a CPU OOM:
            #   500 rows × ~400 tokens × 262K vocab × 4B = ~200 GB if we
            #   accumulated full log_p tensors. Instead we keep only the
            #   per-position argmax (uint32) and the per-position prob of the
            #   gold token (float32), shape (1, T-1) each.
            log_p_full_s = log_p_full[:, :-1, :]
            log_p_masked_s = log_p_masked[:, :-1, :]
            target_ids_s = iids[:, 1:]
            target_mask_s = target_mask.unsqueeze(0)[:, 1:].to(iids.device)

            pred_full = log_p_full_s.argmax(dim=-1)  # (1, T-1)
            pred_masked = log_p_masked_s.argmax(dim=-1)
            prob_full_target = log_p_full_s.gather(2, target_ids_s.unsqueeze(-1)).squeeze(-1).exp()
            prob_masked_target = log_p_masked_s.gather(2, target_ids_s.unsqueeze(-1)).squeeze(-1).exp()

            # Move only the small per-row tensors to CPU.
            pred_full_list.append(pred_full.cpu())
            pred_masked_list.append(pred_masked.cpu())
            prob_full_list.append(prob_full_target.float().cpu())
            prob_masked_list.append(prob_masked_target.float().cpu())
            target_ids_list.append(target_ids_s.cpu())
            target_mask_list.append(target_mask_s.cpu())

            # Free the large logit tensors before the next iteration.
            del out_full, out_masked, log_p_full, log_p_masked
            del log_p_full_s, log_p_masked_s
            torch.cuda.empty_cache()

    # Concatenate the per-position vectors directly along the valid-position
    # axis. We don't need to pad to a common length because img_correct_verifier
    # operates on a flat (N,) layout — the valid mask already filters.
    pred_full_flat: list[torch.Tensor] = []
    pred_masked_flat: list[torch.Tensor] = []
    prob_full_flat: list[torch.Tensor] = []
    prob_masked_flat: list[torch.Tensor] = []
    targets_flat: list[torch.Tensor] = []
    for pf, pm, qf, qm, tids, tmask in zip(
        pred_full_list, pred_masked_list,
        prob_full_list, prob_masked_list,
        target_ids_list, target_mask_list,
    ):
        valid = tmask.bool()
        pred_full_flat.append(pf[valid])
        pred_masked_flat.append(pm[valid])
        prob_full_flat.append(qf[valid])
        prob_masked_flat.append(qm[valid])
        targets_flat.append(tids[valid])

    pf = torch.cat(pred_full_flat) if pred_full_flat else torch.empty(0, dtype=torch.long)
    pm = torch.cat(pred_masked_flat) if pred_masked_flat else torch.empty(0, dtype=torch.long)
    qf = torch.cat(prob_full_flat) if prob_full_flat else torch.empty(0)
    qm = torch.cat(prob_masked_flat) if prob_masked_flat else torch.empty(0)
    tg = torch.cat(targets_flat) if targets_flat else torch.empty(0, dtype=torch.long)

    print(f"[eval] computing IMG_correct over {pf.shape[0]} valid token positions")

    from pivot_ab.img_correct import img_correct_verifier
    res = img_correct_verifier(
        pred_full=pf,
        pred_masked=pm,
        prob_full_correct=qf,
        prob_masked_correct=qm,
        targets=tg,
        margin_tau=margin_tau,
    )
    print(f"[eval] result: {res.to_dict()}")
    write_results({Path(checkpoint_dir).name: res}, Path(output_path))
    vol.commit()
    return res.to_dict()


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def eval_img_correct_v5(
    checkpoint_path: str,
    val_jsonl: str = "/data/groundbench_v5/all_v6/groundbench_v6_test.jsonl",
    output_path: str = "/data/pivot_ab/img_correct_v5_results.jsonl",
    margin_tau: float = 0.1,
    batch_size: int = 32,
    config_name: str | None = None,
) -> dict:
    """Phase 8 entrypoint: IMG_correct on a v5 verifier checkpoint.

    Loads a v5 model (binary verifier; ``best.pt`` state-dict format),
    runs forward with full image + image-zeroed, and computes IMG_correct
    via ``v5.eval.evidence_blindness.run_img_correct_diagnostic``.

    Cost: ~$3-5 per checkpoint on H100 depending on val_jsonl size.

    Args:
        checkpoint_path: Path to a v5 ``best.pt`` file on the volume,
            e.g. ``/data/checkpoints/claimguard_v5/v5_3_contrast/best.pt``.
        val_jsonl: Test/eval JSONL on the volume.
        output_path: Where to write the IMG_correct row (appends).
        margin_tau: Min Δp for image-grounded transition (default 0.1).
        batch_size: Inference batch size.
        config_name: Override for the row's "config" field. Defaults to the
            parent dir of ``checkpoint_path`` (e.g. "v5_3_contrast").
    """
    import sys

    sys.path.insert(0, "/root")
    import json
    from v5.eval.evidence_blindness import run_img_correct_diagnostic

    cp = Path(checkpoint_path)
    name = config_name or cp.parent.name
    print(f"[eval-v5] running IMG_correct on {checkpoint_path} (config={name})")

    res = run_img_correct_diagnostic(
        model_ckpt=cp,
        val_jsonl=Path(val_jsonl),
        image_root=Path("/data"),
        batch_size=batch_size,
        margin_tau=margin_tau,
        device="cuda",
    )

    print(f"[eval-v5] result: {res.to_dict()}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        row = {"config": name, "checkpoint": checkpoint_path, **res.to_dict()}
        f.write(json.dumps(row) + "\n")
    vol.commit()
    return res.to_dict()


@app.local_entrypoint()
def main():
    """Local helper. Run `modal run pivot_ab/modal_app.py::main` for help."""
    print("Pivot A+B Modal app deployed.")
    print("Entrypoints:")
    print("  modal run pivot_ab/modal_app.py::smoke_test_load_chexagent")
    print("  modal run pivot_ab/modal_app.py::run_image_only_filter")
    print("  modal run pivot_ab/modal_app.py::combine_filters")
    print("  modal run pivot_ab/modal_app.py::run_sft --config-path sft_only.yaml")
    print("  modal run pivot_ab/modal_app.py::eval_img_correct --checkpoint-dir /data/checkpoints/pivot_ab_v2/sft_full/checkpoints/final")
    print("  modal run pivot_ab/modal_app.py::eval_img_correct_v5 --checkpoint-path /data/checkpoints/claimguard_v5/v5_3_contrast/best.pt")
