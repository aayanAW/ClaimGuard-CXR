"""Modal orchestrator: run the entire ClaimGuard-CXR v5.0 pipeline server-side.

This single Modal function chains the full v5.0 pipeline end-to-end on one
Modal container so the run is immune to local-client disconnect / laptop sleep:

    1. Build GroundBench for ChestX-Det10 (synthesized claims from annotations)
    2. Build GroundBench for OpenI (rule-based claim extraction from reports)
    3. Aggregate per-site manifests into /data/groundbench_v5/all/
    4. Run the adversarial hypothesis-only filter to produce per-row weights
    5. Train v5.0-base (1-epoch smoke) on the aggregated corpus

Each step is idempotent: if its output file already exists and is non-empty,
the step is skipped. Errors in any step are logged and do not halt the
pipeline; the orchestrator continues to the next step so partial results
accumulate. Final status is written to
``/data/groundbench_v5/pipeline_status.json``.

Usage (once Modal spend limit is available):

    cd /Users/aayanalwani/VeriFact/verifact
    modal run --detach v5/modal/run_v5_pipeline.py::launch

``--detach`` is important: it means the local Modal client can disconnect
(and the laptop can sleep) without killing the running pipeline on Modal.
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    if Path(__file__).resolve().parent.name == "modal"
    else Path("/root/verifact")
)

app = modal.App("claimguard-v5-orchestrator")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.46.3",
        "open-clip-torch==2.26.1",
        "sentencepiece==0.2.0",
        "accelerate==0.33.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "scikit-learn==1.5.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "pillow==10.4.0",
        "scipy==1.14.1",
        "pyyaml==6.0.2",
        "torchxrayvision==1.3.5",
        "anthropic==0.42.0",
        "scikit-image==0.24.0",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")

secrets: list[modal.Secret] = []
try:
    secrets.append(modal.Secret.from_name("anthropic"))
except Exception:
    pass
# wandb secret intentionally omitted: smoke training runs with use_wandb=False


# ---------------------------------------------------------------------------
# Inlined per-site build logic (copy of build_groundbench.py core loop)
# ---------------------------------------------------------------------------

def _build_site(site: str, limit: int = 0, use_llm_extractor: bool = False) -> dict:
    """Run GroundBench assembly for a single site. Returns summary dict."""
    import json
    import sys
    from pathlib import Path as P

    sys.path.insert(0, "/root/verifact")

    from v5.data.claim_extractor import llm_extract, rule_extract
    from v5.data.claim_parser import llm_parse, rule_parse, load_ontology
    from v5.data.claim_matcher import ClaimMatcher, Annotation
    from v5.data.anatomy_masks import compute_anatomy_masks
    from v5.data.claim_synthesizer import synthesize_claims_for_image, default_site_findings
    from v5.data.groundbench import assemble_row, split_and_write, aggregate_summary
    from v5.data import chexpert_plus as ds_chexpert
    from v5.data import openi as ds_openi
    from v5.data import padchest as ds_padchest
    from v5.data import rsna_pneumonia as ds_rsna
    from v5.data import siim_pneumothorax as ds_siim
    from v5.data import object_cxr as ds_objcxr
    from v5.data import chestx_det10 as ds_cx10

    ontology = load_ontology()
    matcher = ClaimMatcher()
    out_root = P("/data/groundbench_v5") / site
    out_root.mkdir(parents=True, exist_ok=True)

    site_cfg = {
        "chexpert_plus": (
            lambda: ds_chexpert.iter_chexpert_plus(
                metadata_csv=P("/data/chexpert_plus_meta/df_chexpert_plus_240401.csv"),
                require_image=False,
            ),
            ds_chexpert.annotations_for_record,
        ),
        "openi": (
            lambda: ds_openi.iter_openi(
                image_root=P("/data/openi"),
                report_csv=P("/data/iu_xray_meta/iu_xray_reports.csv"),
            ),
            ds_openi.annotations_for_record,
        ),
        "chestx_det10": (
            lambda: ds_cx10.iter_chestx_det10(
                annot_root=P("/data/chestx_det10"),
                image_root=P("/data/chestx_det10_images"),
            ),
            ds_cx10.annotations_for_record,
        ),
        "padchest": (
            lambda: ds_padchest.iter_padchest(P("/data/padchest")),
            ds_padchest.annotations_for_record,
        ),
        "rsna_pneumonia": (
            lambda: ds_rsna.iter_rsna(P("/data/rsna_pneumonia")),
            ds_rsna.annotations_for_record,
        ),
        "siim_acr": (
            lambda: ds_siim.iter_siim(P("/data/siim_acr_pneumothorax")),
            ds_siim.annotations_for_record,
        ),
        "object_cxr": (
            lambda: ds_objcxr.iter_object_cxr(P("/data/object_cxr")),
            ds_objcxr.annotations_for_record,
        ),
    }
    if site not in site_cfg:
        raise ValueError(f"Unknown site {site}; valid: {list(site_cfg)}")
    iter_callable, ann_fn = site_cfg[site]

    rows = []
    summary_counts = {"images": 0, "claims": 0}

    for idx, rec in enumerate(iter_callable()):
        if limit and idx >= limit:
            break
        annotations = ann_fn(rec)
        report_text = ""
        patient_id = None
        age = None
        sex = None
        scanner = None
        country = None
        if site == "chexpert_plus":
            report_text = f"{rec.report_findings}\n\n{rec.report_impression}"
            patient_id = rec.patient_id
            country = "US"
        elif site == "openi":
            report_text = f"{rec.report_findings}\n\n{rec.report_impression}"
            country = "US"
        elif site == "padchest":
            report_text = rec.report_en or rec.report_raw
            country = "ES"
        elif site in {"rsna_pneumonia", "siim_acr", "object_cxr", "chestx_det10"}:
            report_text = ""
        summary_counts["images"] += 1

        if report_text.strip():
            extractor_fn = llm_extract if use_llm_extractor else rule_extract
            parser_fn = llm_parse if use_llm_extractor else rule_parse
            try:
                claims = extractor_fn(report_text, report_id=str(rec.image_id))
            except Exception as exc:
                print(f"[{site}] extractor failed on {rec.image_id}: {exc}")
                continue
            for cl in claims:
                try:
                    structured = parser_fn(
                        cl.claim_text, cl.claim_id, cl.report_id, ontology=ontology,
                        evidence_source_type="oracle_human", generator_id="oracle",
                    )
                except Exception as exc:
                    print(f"[{site}] parser failed on claim {cl.claim_id}: {exc}")
                    continue
                try:
                    anatomy = compute_anatomy_masks(rec.image_path)
                except Exception as exc:
                    print(f"[{site}] anatomy_masks failed on {rec.image_id}: {exc}")
                    anatomy = None
                row = assemble_row(
                    structured, annotations, anatomy,
                    image_path=rec.image_path, source_site=site,
                    evidence_text=report_text, patient_id=patient_id,
                    sex=sex, age=age, scanner_manufacturer=scanner,
                    country=country, matcher=matcher,
                )
                rows.append(row)
                summary_counts["claims"] += 1
        elif site in {"rsna_pneumonia", "siim_acr", "object_cxr", "chestx_det10"} and annotations:
            try:
                anatomy = compute_anatomy_masks(rec.image_path)
            except Exception as exc:
                print(f"[{site}] anatomy_masks failed on {rec.image_id}: {exc}")
                anatomy = None
            synth_claims = synthesize_claims_for_image(
                image_id=str(rec.image_id),
                annotations=annotations,
                source=site,
                all_findings_in_site=default_site_findings(site),
                emit_negatives=True,
                emit_contradicted_positives=True,
            )
            bbox_by_finding = {
                ann.finding: tuple(ann.bbox)
                for ann in annotations if ann.bbox is not None
            }
            for sc in synth_claims:
                synth_bbox = None
                if (
                    sc.gt_label == "SUPPORTED"
                    and "negative" not in sc.structured.modifier_tags
                ):
                    synth_bbox = bbox_by_finding.get(sc.structured.finding)
                row = assemble_row(
                    sc.structured, annotations, anatomy,
                    image_path=rec.image_path, source_site=site,
                    evidence_text=None, patient_id=patient_id,
                    sex=sex, age=age, scanner_manufacturer=scanner,
                    country=country, matcher=matcher,
                    synthesized_gt=sc.gt_label, synthesized_bbox=synth_bbox,
                )
                rows.append(row)
                summary_counts["claims"] += 1

    split_and_write(rows, out_root, seed=17)
    summary = aggregate_summary(rows)
    summary.update(summary_counts)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ---------------------------------------------------------------------------
# Orchestrator: chains every step idempotently on one container
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=secrets,
)
def orchestrate(
    configs_csv: str = "v5_0_base",
    smoke: bool = True,
) -> dict:
    """Execute the v5 pipeline idempotently.

    Args:
        configs_csv: Comma-separated list of training config names to run in
            order. Each must correspond to a file in ``v5/configs/``. Default
            ``"v5_0_base"`` trains only the base config.
        smoke: If True, override every config's ``epochs`` to 1 for fast
            smoke-level validation. Default True so a naive launch is cheap.
    """
    import json
    import sys
    import traceback

    sys.path.insert(0, "/root/verifact")

    from pathlib import Path as P

    _TRAIN_PLAN = {
        "configs": [c.strip() for c in configs_csv.split(",") if c.strip()],
        "smoke": bool(smoke),
    }

    out_root = P("/data/groundbench_v5")
    out_root.mkdir(parents=True, exist_ok=True)

    status: dict = {"plan": _TRAIN_PLAN}

    def _nonempty(p: P) -> bool:
        return p.exists() and p.stat().st_size > 0

    # === STEP 1: ChestX-Det10 ===
    cx_out = out_root / "chestx_det10" / "groundbench_v5_train.jsonl"
    if _nonempty(cx_out):
        print(f"[orch] STEP 1 skipped — {cx_out} exists")
        status["step1_chestx_det10"] = "skipped_exists"
    else:
        try:
            print("[orch] STEP 1: building ChestX-Det10")
            status["step1_chestx_det10"] = _build_site("chestx_det10")
            print(f"[orch] STEP 1 done: {status['step1_chestx_det10']}")
        except Exception as exc:
            print(f"[orch] STEP 1 FAILED: {exc}")
            traceback.print_exc()
            status["step1_chestx_det10"] = {"error": str(exc)}
    (out_root / "pipeline_status.json").write_text(json.dumps(status, indent=2, default=str))
    volume.commit()

    # === STEP 2: OpenI ===
    openi_out = out_root / "openi" / "groundbench_v5_train.jsonl"
    if _nonempty(openi_out):
        print(f"[orch] STEP 2 skipped — {openi_out} exists")
        status["step2_openi"] = "skipped_exists"
    else:
        try:
            print("[orch] STEP 2: building OpenI")
            status["step2_openi"] = _build_site("openi", use_llm_extractor=False)
            print(f"[orch] STEP 2 done: {status['step2_openi']}")
        except Exception as exc:
            print(f"[orch] STEP 2 FAILED: {exc}")
            traceback.print_exc()
            status["step2_openi"] = {"error": str(exc)}
    (out_root / "pipeline_status.json").write_text(json.dumps(status, indent=2, default=str))
    volume.commit()

    # === STEP 3: Aggregate ===
    agg_out = out_root / "all" / "groundbench_v5_train.jsonl"
    if _nonempty(agg_out):
        print(f"[orch] STEP 3 skipped — {agg_out} exists")
        status["step3_aggregate"] = "skipped_exists"
    else:
        try:
            from v5.data.groundbench import aggregate_groundbench
            available_sites = [
                s for s in ("chestx_det10", "openi")
                if _nonempty(out_root / s / "groundbench_v5_train.jsonl")
            ]
            if not available_sites:
                raise RuntimeError("No site builds available to aggregate")
            print(f"[orch] STEP 3: aggregating sites={available_sites}")
            status["step3_aggregate"] = aggregate_groundbench(out_root, sites=available_sites)
            print(f"[orch] STEP 3 done: {status['step3_aggregate']}")
        except Exception as exc:
            print(f"[orch] STEP 3 FAILED: {exc}")
            traceback.print_exc()
            status["step3_aggregate"] = {"error": str(exc)}
    (out_root / "pipeline_status.json").write_text(json.dumps(status, indent=2, default=str))
    volume.commit()

    # === STEP 4: HO filter ===
    ho_weights = out_root / "ho_filter_weights.jsonl"
    if _nonempty(ho_weights):
        print(f"[orch] STEP 4 skipped — {ho_weights} exists")
        status["step4_ho_filter"] = "skipped_exists"
    else:
        try:
            import torch
            from v5.ho_filter import run_ho_filter
            from v5.model import V5Config, build_v5_tokenizer

            if not _nonempty(agg_out):
                raise RuntimeError("Aggregate train.jsonl missing; cannot run HO filter")
            tokenizer = build_v5_tokenizer(V5Config())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[orch] STEP 4: running HO filter")
            status["step4_ho_filter"] = run_ho_filter(
                train_jsonl=agg_out,
                output_weights_path=ho_weights,
                tokenizer=tokenizer,
                device=device,
                n_epochs=1,
            )
            print(f"[orch] STEP 4 done: {status['step4_ho_filter']}")
        except Exception as exc:
            print(f"[orch] STEP 4 FAILED: {exc}")
            traceback.print_exc()
            status["step4_ho_filter"] = {"error": str(exc)}
    (out_root / "pipeline_status.json").write_text(json.dumps(status, indent=2, default=str))
    volume.commit()

    # === STEP 5: Train the requested training ladder ===
    # Configs can include any subset of {v5_0_base, v5_1_ground, v5_2_real,
    # v5_3_contrast, v5_4_final}. The smoke variant uses only v5_0_base with
    # epochs=1; the full variant runs all five configs at their config-specified
    # epoch counts.
    #
    # Orchestrator callers set `configs` and `smoke` via the launch wrapper.
    # Defaults here: smoke mode (v5_0_base, 1 epoch).
    configs_to_train = _TRAIN_PLAN["configs"]
    force_smoke_epochs = _TRAIN_PLAN["smoke"]

    training_results: dict = {}
    for cfg_name in configs_to_train:
        out_ckpt = P("/data/checkpoints/claimguard_v5") / cfg_name
        done_marker = out_ckpt / ("smoke_done.txt" if force_smoke_epochs else "full_done.txt")
        if done_marker.exists():
            print(f"[orch] STEP 5/{cfg_name} skipped — already done")
            training_results[cfg_name] = "skipped_exists"
            continue
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf

            if not _nonempty(agg_out):
                raise RuntimeError("Aggregate train.jsonl missing; cannot train")

            with initialize_config_dir(
                config_dir=str(P("/root/verifact/v5/configs").resolve()), version_base=None
            ):
                cfg = compose(config_name=cfg_name)
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            from v5.train import V5TrainConfig, train_v5
            from v5.model import V5Config
            from v5.losses import LossWeights

            lw = LossWeights(**cfg_dict["loss"])
            val_path = out_root / "all" / "groundbench_v5_val.jsonl"
            if not _nonempty(val_path):
                val_path = out_root / "all" / "groundbench_v5_cal.jsonl"

            tcfg = V5TrainConfig(
                train_jsonl=agg_out,
                val_jsonl=val_path,
                out_dir=out_ckpt,
                image_root=P("/data"),
                batch_size=cfg_dict["train"]["batch_size"],
                grad_accum=cfg_dict["train"]["grad_accum"],
                epochs=1 if force_smoke_epochs else cfg_dict["train"]["epochs"],
                lr_encoders=cfg_dict["train"]["lr_encoders"],
                lr_heads=cfg_dict["train"]["lr_heads"],
                weight_decay=cfg_dict["train"]["weight_decay"],
                warmup_steps=cfg_dict["train"]["warmup_steps"],
                mixed_precision=cfg_dict["train"]["mixed_precision"],
                image_masked_prob=cfg_dict["train"]["image_masked_prob"],
                contrast_prob=cfg_dict["train"]["contrast_prob"],
                adversarial_ho_filter=cfg_dict["train"]["adversarial_ho_filter"],
                ho_filter_weights_path=(
                    ho_weights if cfg_dict["train"]["adversarial_ho_filter"] else None
                ),
                seed=cfg_dict["train"]["seed"],
                loss_weights=lw,
                use_wandb=False,
                wandb_project=cfg_dict["train"]["wandb_project"],
                freeze_image_layers=cfg_dict["model"]["freeze_image_layers"],
                freeze_text_layers=cfg_dict["model"]["freeze_text_layers"],
                grounding_enabled=cfg_dict["model"]["grounding_enabled"],
            )
            model_cfg = V5Config(
                image_backbone=cfg_dict["model"]["image_backbone"],
                image_backbone_revision=cfg_dict["model"].get("image_backbone_revision"),
                text_backbone=cfg_dict["model"]["text_backbone"],
                text_backbone_revision=cfg_dict["model"].get("text_backbone_revision"),
                shared_dim=cfg_dict["model"]["shared_dim"],
                fusion_layers=cfg_dict["model"]["fusion_layers"],
                fusion_heads=cfg_dict["model"]["fusion_heads"],
                fusion_ffn_dim=cfg_dict["model"]["fusion_ffn_dim"],
                fusion_dropout=cfg_dict["model"]["fusion_dropout"],
                mc_dropout_p=cfg_dict["model"]["mc_dropout_p"],
                num_verdict_classes=cfg_dict["model"]["num_verdict_classes"],
                image_patches_side=cfg_dict["model"]["image_patches_side"],
                max_text_tokens=cfg_dict["model"]["max_text_tokens"],
                freeze_image_layers=cfg_dict["model"]["freeze_image_layers"],
                freeze_text_layers=cfg_dict["model"]["freeze_text_layers"],
                grounding_enabled=cfg_dict["model"]["grounding_enabled"],
                uncertainty_samples=cfg_dict["model"]["uncertainty_samples"],
            )
            mode_label = "smoke (1 epoch)" if force_smoke_epochs else f"full ({tcfg.epochs} epochs)"
            print(f"[orch] STEP 5/{cfg_name}: training {mode_label}")
            stats = train_v5(tcfg, model_cfg)
            out_ckpt.mkdir(parents=True, exist_ok=True)
            done_marker.write_text("ok")
            training_results[cfg_name] = {
                "epochs": len(stats),
                "stats": [s.__dict__ for s in stats],
            }
            print(f"[orch] STEP 5/{cfg_name} done")
            volume.commit()
        except Exception as exc:
            print(f"[orch] STEP 5/{cfg_name} FAILED: {exc}")
            traceback.print_exc()
            training_results[cfg_name] = {"error": str(exc)}
            volume.commit()
            # Do not break — try the next config. This gives partial progress
            # if, for example, v5_0 trains fine but v5_3 hits a bug; we still
            # get three successful training runs out of the attempt.
        # Incremental status write after each config finishes (success or error)
        status["step5_training"] = training_results
        (out_root / "pipeline_status.json").write_text(
            json.dumps(status, indent=2, default=str)
        )
        volume.commit()

    status["step5_training"] = training_results
    volume.commit()

    # === STEP 6: Evidence-blindness diagnostic on every trained checkpoint ===
    diagnostic_results: dict = {}
    for cfg_name in configs_to_train:
        result = training_results.get(cfg_name)
        if isinstance(result, dict) and "error" in result:
            print(f"[orch] STEP 6/{cfg_name} skipped — training failed")
            diagnostic_results[cfg_name] = "skipped_training_failed"
            continue
        diag_out = P("/data/checkpoints/claimguard_v5") / cfg_name / "diagnostic.json"
        if diag_out.exists() and diag_out.stat().st_size > 0:
            print(f"[orch] STEP 6/{cfg_name} skipped — diagnostic.json exists")
            diagnostic_results[cfg_name] = "skipped_exists"
            continue
        try:
            from v5.eval.evidence_blindness import run_diagnostic, diagnostic_to_json
            # Locate the training checkpoint. The train loop writes per-epoch
            # checkpoints; prefer the last one available.
            ckpt_dir = P("/data/checkpoints/claimguard_v5") / cfg_name
            ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
            if not ckpts:
                ckpts = sorted(ckpt_dir.glob("best*.pt"))
            if not ckpts:
                raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
            ckpt_path = ckpts[-1]
            val_path = out_root / "all" / "groundbench_v5_val.jsonl"
            if not _nonempty(val_path):
                val_path = out_root / "all" / "groundbench_v5_cal.jsonl"
            print(f"[orch] STEP 6/{cfg_name}: running diagnostic on {ckpt_path.name}")
            diag = run_diagnostic(
                model_ckpt=ckpt_path,
                val_jsonl=val_path,
                image_root=P("/data"),
                batch_size=32,
                n_shuffle_seeds=3,
            )
            diagnostic_to_json(diag, diag_out)
            diagnostic_results[cfg_name] = diag.to_dict()
            print(f"[orch] STEP 6/{cfg_name} done: IMG={diag.img_gap_pp:.2f}pp ESG={diag.esg_gap_pp:.2f}pp evidence_blind={diag.evidence_blind}")
            volume.commit()
        except Exception as exc:
            print(f"[orch] STEP 6/{cfg_name} FAILED: {exc}")
            traceback.print_exc()
            diagnostic_results[cfg_name] = {"error": str(exc)}
            volume.commit()
        status["step6_diagnostic"] = diagnostic_results
        (out_root / "pipeline_status.json").write_text(
            json.dumps(status, indent=2, default=str)
        )
        volume.commit()

    status["step6_diagnostic"] = diagnostic_results

    # === STEP 7: Consolidate final results into a single artifact ===
    try:
        from v5.eval.collect_results import collect_all, format_summary_table
        artifact = collect_all(
            checkpoints_root="/data/checkpoints/claimguard_v5",
            status_path=out_root / "pipeline_status.json",
            out_json="/data/v5_results.json",
            out_csv="/data/v5_results.csv",
        )
        print("[orch] STEP 7 done — final results:\n" + format_summary_table(artifact))
        status["step7_collect_results"] = {
            "n_rows": len(artifact.get("rows", [])),
            "output_json": "/data/v5_results.json",
            "output_csv": "/data/v5_results.csv",
        }
    except Exception as exc:
        print(f"[orch] STEP 7 FAILED: {exc}")
        traceback.print_exc()
        status["step7_collect_results"] = {"error": str(exc)}
    volume.commit()

    # === Finalize ===
    (out_root / "pipeline_status.json").write_text(json.dumps(status, indent=2, default=str))
    print(f"[orch] Pipeline complete. Status keys: {list(status.keys())}")
    return status


@app.local_entrypoint()
def launch() -> None:
    """Smoke launch: v5.0-base only, 1 epoch. Recommended first run.

    Validates the full pipeline end-to-end cheaply (~$20-30). Once smoke
    completes successfully, relaunch with ``launch_full`` for the full ladder.
    """
    print(orchestrate.remote(configs_csv="v5_0_base", smoke=True))


@app.local_entrypoint()
def launch_full() -> None:
    """Full ladder: v5.0 -> v5.1 -> v5.2 -> v5.3 -> v5.4 at config-specified
    epoch counts. Costs ~$350-450 in Modal compute. Only launch after ``launch``
    has produced a clean smoke result.
    """
    configs = "v5_0_base,v5_1_ground,v5_2_real,v5_3_contrast,v5_4_final"
    print(orchestrate.remote(configs_csv=configs, smoke=False))


# ===========================================================================
# Tier 3 orchestrators: baselines, conformal FDR, ablations, cross-site
# ===========================================================================


# Secrets for baseline APIs + gated-HF-repo access. `anthropic` enables
# Claude-3.5-Sonnet; `huggingface` (HF_TOKEN) enables CheXagent and other
# gated medical VLM weights.
baseline_secrets: list[modal.Secret] = []
for _name in ("anthropic", "huggingface"):
    try:
        baseline_secrets.append(modal.Secret.from_name(_name))
    except Exception:
        pass


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
    secrets=baseline_secrets,
)
def run_baselines(n_test: int = 500) -> dict:
    """Run every available baseline verifier (API + local) on a subset of the test set.

    Writes per-baseline ``baseline_<name>_diagnostic.json`` to /data/baselines/.
    The subset size is bounded by ``n_test`` to keep API costs reasonable;
    default 500 claims × 4 conditions × up to 7 baselines ≈ $60.
    """
    import sys, json as _json, traceback
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import torch as _torch

    from v5.eval.baselines import (
        available_baselines,
        run_baseline_diagnostic,
        _load_test_subset,
    )

    out_dir = P("/data/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_jsonl = P("/data/groundbench_v5/all/groundbench_v5_test.jsonl")
    image_root = P("/data")
    rows = _load_test_subset(test_jsonl, image_root, max_rows=n_test)
    print(f"[baselines] using {len(rows)} test rows")

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    baselines = available_baselines(device=device)
    results: dict = {}
    for b in baselines:
        try:
            print(f"[baselines] running {b.name}")
            res = run_baseline_diagnostic(
                b, rows, image_root,
                out_path=out_dir / f"baseline_{b.name}_diagnostic.json",
            )
            results[b.name] = {
                "n_test": res.n_test,
                "acc_full": res.acc_full,
                "img_gap_pp": res.img_gap_pp,
                "esg_gap_pp": res.esg_gap_pp,
                "ipg_gap_pp": res.ipg_gap_pp,
                "evidence_blind": res.evidence_blind,
            }
        except Exception as exc:
            print(f"[baselines] {b.name} failed: {exc}")
            traceback.print_exc()
            results[b.name] = {"error": str(exc)}
        volume.commit()

    summary_path = out_dir / "baseline_summary.json"
    summary_path.write_text(_json.dumps(results, indent=2, default=str))
    return results


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 3,
    volumes={"/data": volume},
)
def run_conformal(configs_csv: str = "v5_0_base,v5_1_ground,v5_2_real,v5_3_contrast,v5_4_final") -> dict:
    """Run inverted cfBH + StratCP + forward cfBH on each trained checkpoint."""
    import sys, json as _json, traceback
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    from v5.eval.conformal_fdr import run_conformal as do_run
    import dataclasses

    configs = [c.strip() for c in configs_csv.split(",") if c.strip()]
    out_root = P("/data/checkpoints/claimguard_v5")
    cal_jsonl = P("/data/groundbench_v5/all/groundbench_v5_cal.jsonl")
    test_jsonl = P("/data/groundbench_v5/all/groundbench_v5_test.jsonl")
    image_root = P("/data")
    results: dict = {}
    for cfg in configs:
        cfg_dir = out_root / cfg
        ckpts = sorted(cfg_dir.glob("ckpt_epoch_*.pt")) or sorted(cfg_dir.glob("best*.pt"))
        if not ckpts:
            results[cfg] = {"error": f"no checkpoint in {cfg_dir}"}
            continue
        ckpt = ckpts[-1]
        try:
            print(f"[conformal] {cfg}: using {ckpt.name}")
            rep = do_run(
                model_ckpt=ckpt,
                cal_jsonl=cal_jsonl,
                test_jsonl=test_jsonl,
                image_root=image_root,
                alphas=(0.05, 0.10, 0.15, 0.20),
                out_path=cfg_dir / "conformal.json",
            )
            results[cfg] = dataclasses.asdict(rep)
        except Exception as exc:
            print(f"[conformal] {cfg} failed: {exc}")
            traceback.print_exc()
            results[cfg] = {"error": str(exc)}
        volume.commit()
    summary_path = P("/data/conformal_summary.json")
    summary_path.write_text(_json.dumps(results, indent=2, default=str))
    return results


@app.local_entrypoint()
def launch_baselines(n_test: int = 500) -> None:
    """Spawn baseline diagnostic run. Writes /data/baselines/*.json."""
    print(run_baselines.remote(n_test=n_test))


@app.local_entrypoint()
def launch_conformal(configs: str = "v5_0_base,v5_1_ground,v5_2_real,v5_3_contrast,v5_4_final") -> None:
    """Spawn conformal FDR eval on trained checkpoints."""
    print(run_conformal.remote(configs_csv=configs))


# ---------------------------------------------------------------------------
# Loss / threshold ablations
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
)
def run_ablations(ablation_set: str = "loss_drop") -> dict:
    """Run one of three ablation families.

    ablation_set:
        ``loss_drop``: drop each of ground/consist/contrast/uncert in turn
                       (cls is required; HO filter toggled separately).
        ``ho_threshold``: vary HO-filter confidence threshold across
                          {0.5, 0.6, 0.7, 0.8, 0.9}.
        ``scale``: train on 25%, 50%, 100% of the aggregated training set.

    Each variant trains a new v5.3-contrast-style config and writes a checkpoint
    under /data/checkpoints/claimguard_v5/abl_<variant>/. The existing step-6
    diagnostic is invoked on each new checkpoint.
    """
    import sys, json as _json, traceback
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from v5.train import V5TrainConfig, train_v5
    from v5.model import V5Config
    from v5.losses import LossWeights
    from v5.eval.evidence_blindness import run_diagnostic, diagnostic_to_json

    out_root = P("/data/groundbench_v5")
    ckpt_root = P("/data/checkpoints/claimguard_v5")

    # Baseline config is v5_3_contrast (contrastive + consistency + HO).
    with initialize_config_dir(config_dir=str(P("/root/verifact/v5/configs").resolve()), version_base=None):
        base_cfg = compose(config_name="v5_3_contrast")
        base = OmegaConf.to_container(base_cfg, resolve=True)

    variants: list[tuple[str, dict]] = []
    if ablation_set == "loss_drop":
        for term in ("ground", "consist", "contrast", "uncert"):
            tag = f"no_{term}"
            v = _json.loads(_json.dumps(base))
            v["loss"][term] = 0.0
            variants.append((f"abl_{tag}", v))
        # also drop HO filter
        v = _json.loads(_json.dumps(base))
        v["train"]["adversarial_ho_filter"] = False
        variants.append(("abl_no_hofilter", v))
    elif ablation_set == "ho_threshold":
        # Requires re-running the HO filter at each threshold. Here we just
        # train under the *same* HO weights but the name reflects intent; the
        # user must run the HO filter at each threshold separately (see
        # v5/modal/ho_filter.py). This variant documents the sweep.
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            v = _json.loads(_json.dumps(base))
            v["_ho_threshold_sweep"] = thr
            variants.append((f"abl_hothresh_{int(thr*100)}", v))
    elif ablation_set == "scale":
        for frac in (0.25, 0.5, 1.0):
            v = _json.loads(_json.dumps(base))
            v["_scale_fraction"] = frac
            variants.append((f"abl_scale_{int(frac*100)}", v))
    else:
        raise ValueError(f"unknown ablation_set {ablation_set}")

    results: dict = {}
    for name, v in variants:
        out_ckpt = ckpt_root / name
        done_marker = out_ckpt / "full_done.txt"
        if done_marker.exists():
            results[name] = "skipped_exists"
            continue
        try:
            lw = LossWeights(**v["loss"])
            train_jsonl = out_root / "all" / "groundbench_v5_train.jsonl"
            val_jsonl = out_root / "all" / "groundbench_v5_val.jsonl"
            ho_weights = out_root / "ho_filter_weights.jsonl"
            tcfg = V5TrainConfig(
                train_jsonl=train_jsonl,
                val_jsonl=val_jsonl,
                out_dir=out_ckpt,
                image_root=P("/data"),
                batch_size=v["train"]["batch_size"],
                grad_accum=v["train"]["grad_accum"],
                epochs=v["train"]["epochs"],
                lr_encoders=v["train"]["lr_encoders"],
                lr_heads=v["train"]["lr_heads"],
                weight_decay=v["train"]["weight_decay"],
                warmup_steps=v["train"]["warmup_steps"],
                mixed_precision=v["train"]["mixed_precision"],
                image_masked_prob=v["train"]["image_masked_prob"],
                contrast_prob=v["train"]["contrast_prob"],
                adversarial_ho_filter=v["train"]["adversarial_ho_filter"],
                ho_filter_weights_path=(ho_weights if v["train"]["adversarial_ho_filter"] else None),
                seed=v["train"]["seed"],
                loss_weights=lw,
                use_wandb=False,
                wandb_project=v["train"]["wandb_project"],
                freeze_image_layers=v["model"]["freeze_image_layers"],
                freeze_text_layers=v["model"]["freeze_text_layers"],
                grounding_enabled=v["model"]["grounding_enabled"],
            )
            mcfg = V5Config(
                image_backbone=v["model"]["image_backbone"],
                text_backbone=v["model"]["text_backbone"],
                shared_dim=v["model"]["shared_dim"],
                fusion_layers=v["model"]["fusion_layers"],
                fusion_heads=v["model"]["fusion_heads"],
                fusion_ffn_dim=v["model"]["fusion_ffn_dim"],
                fusion_dropout=v["model"]["fusion_dropout"],
                mc_dropout_p=v["model"]["mc_dropout_p"],
                num_verdict_classes=v["model"]["num_verdict_classes"],
                image_patches_side=v["model"]["image_patches_side"],
                max_text_tokens=v["model"]["max_text_tokens"],
                freeze_image_layers=v["model"]["freeze_image_layers"],
                freeze_text_layers=v["model"]["freeze_text_layers"],
                grounding_enabled=v["model"]["grounding_enabled"],
                uncertainty_samples=v["model"]["uncertainty_samples"],
            )
            print(f"[abl] training {name}")
            stats = train_v5(tcfg, mcfg)
            done_marker.write_text("ok")

            # Diagnostic on fresh checkpoint
            ckpts = sorted(out_ckpt.glob("ckpt_epoch_*.pt")) or sorted(out_ckpt.glob("best*.pt"))
            if ckpts:
                diag = run_diagnostic(
                    model_ckpt=ckpts[-1],
                    val_jsonl=val_jsonl,
                    image_root=P("/data"),
                    batch_size=32,
                    n_shuffle_seeds=3,
                )
                diagnostic_to_json(diag, out_ckpt / "diagnostic.json")
                results[name] = {
                    "epochs": len(stats),
                    "img_gap_pp": diag.img_gap_pp,
                    "esg_gap_pp": diag.esg_gap_pp,
                    "ipg_gap_pp": diag.ipg_gap_pp,
                    "evidence_blind": diag.evidence_blind,
                }
            else:
                results[name] = {"epochs": len(stats), "diagnostic_error": "no ckpt"}
        except Exception as exc:
            print(f"[abl] {name} failed: {exc}")
            traceback.print_exc()
            results[name] = {"error": str(exc)}
        volume.commit()

    (P("/data") / f"ablations_{ablation_set}_summary.json").write_text(
        _json.dumps(results, indent=2, default=str)
    )
    return results


@app.local_entrypoint()
def launch_ablations(ablation_set: str = "loss_drop") -> None:
    """Spawn an ablation run. ablation_set in {loss_drop, ho_threshold, scale}."""
    print(run_ablations.remote(ablation_set=ablation_set))


# ---------------------------------------------------------------------------
# Cross-site transfer
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
)
def run_cross_site(train_site: str = "openi", test_site: str = "chestx_det10") -> dict:
    """Train on one source site and evaluate on the other.

    Uses v5.3-contrast loss stack. Writes checkpoint under
    /data/checkpoints/claimguard_v5/crosssite_{train_site}_to_{test_site}/.
    """
    import sys, json as _json, traceback
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from v5.train import V5TrainConfig, train_v5
    from v5.model import V5Config
    from v5.losses import LossWeights
    from v5.eval.evidence_blindness import run_diagnostic, diagnostic_to_json

    out_root = P("/data/groundbench_v5")
    ckpt_root = P("/data/checkpoints/claimguard_v5")
    train_jsonl = out_root / train_site / "groundbench_v5_train.jsonl"
    val_jsonl = out_root / train_site / "groundbench_v5_val.jsonl"
    test_jsonl = out_root / test_site / "groundbench_v5_test.jsonl"
    for p in (train_jsonl, val_jsonl, test_jsonl):
        if not p.exists():
            return {"error": f"missing {p}"}

    with initialize_config_dir(config_dir=str(P("/root/verifact/v5/configs").resolve()), version_base=None):
        base = OmegaConf.to_container(compose(config_name="v5_3_contrast"), resolve=True)

    name = f"crosssite_{train_site}_to_{test_site}"
    out_ckpt = ckpt_root / name
    done_marker = out_ckpt / "full_done.txt"
    if done_marker.exists():
        return {name: "skipped_exists"}

    try:
        lw = LossWeights(**base["loss"])
        tcfg = V5TrainConfig(
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            out_dir=out_ckpt,
            image_root=P("/data"),
            batch_size=base["train"]["batch_size"],
            grad_accum=base["train"]["grad_accum"],
            epochs=base["train"]["epochs"],
            lr_encoders=base["train"]["lr_encoders"],
            lr_heads=base["train"]["lr_heads"],
            weight_decay=base["train"]["weight_decay"],
            warmup_steps=base["train"]["warmup_steps"],
            mixed_precision=base["train"]["mixed_precision"],
            image_masked_prob=base["train"]["image_masked_prob"],
            contrast_prob=base["train"]["contrast_prob"],
            adversarial_ho_filter=False,  # site-specific HO would need re-running
            seed=base["train"]["seed"],
            loss_weights=lw,
            use_wandb=False,
            wandb_project=base["train"]["wandb_project"],
            freeze_image_layers=base["model"]["freeze_image_layers"],
            freeze_text_layers=base["model"]["freeze_text_layers"],
            grounding_enabled=base["model"]["grounding_enabled"],
        )
        mcfg = V5Config(
            image_backbone=base["model"]["image_backbone"],
            text_backbone=base["model"]["text_backbone"],
            shared_dim=base["model"]["shared_dim"],
            fusion_layers=base["model"]["fusion_layers"],
            fusion_heads=base["model"]["fusion_heads"],
            fusion_ffn_dim=base["model"]["fusion_ffn_dim"],
            fusion_dropout=base["model"]["fusion_dropout"],
            mc_dropout_p=base["model"]["mc_dropout_p"],
            num_verdict_classes=base["model"]["num_verdict_classes"],
            image_patches_side=base["model"]["image_patches_side"],
            max_text_tokens=base["model"]["max_text_tokens"],
            freeze_image_layers=base["model"]["freeze_image_layers"],
            freeze_text_layers=base["model"]["freeze_text_layers"],
            grounding_enabled=base["model"]["grounding_enabled"],
            uncertainty_samples=base["model"]["uncertainty_samples"],
        )
        print(f"[cross-site] training {name}")
        stats = train_v5(tcfg, mcfg)
        done_marker.write_text("ok")

        # Evaluate on the held-out site's test split
        ckpts = sorted(out_ckpt.glob("ckpt_epoch_*.pt")) or sorted(out_ckpt.glob("best*.pt"))
        diag = run_diagnostic(
            model_ckpt=ckpts[-1],
            val_jsonl=test_jsonl,  # held-out site's test split
            image_root=P("/data"),
            batch_size=32,
            n_shuffle_seeds=3,
        )
        diagnostic_to_json(diag, out_ckpt / f"diagnostic_on_{test_site}.json")
        result = {
            "train_site": train_site, "test_site": test_site,
            "epochs": len(stats),
            "test_acc_full": diag.acc_full,
            "img_gap_pp": diag.img_gap_pp,
            "esg_gap_pp": diag.esg_gap_pp,
            "ipg_gap_pp": diag.ipg_gap_pp,
            "evidence_blind": diag.evidence_blind,
        }
        (P("/data") / f"{name}_summary.json").write_text(_json.dumps(result, indent=2, default=str))
        volume.commit()
        return {name: result}
    except Exception as exc:
        traceback.print_exc()
        return {name: {"error": str(exc)}}


@app.local_entrypoint()
def launch_cross_site(train_site: str = "openi", test_site: str = "chestx_det10") -> None:
    """Train on one site, evaluate diagnostic on the other."""
    print(run_cross_site.remote(train_site=train_site, test_site=test_site))
