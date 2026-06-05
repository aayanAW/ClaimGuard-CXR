"""OpenI (Indiana University) loader.

3,996 frontal/lateral CXRs with full radiologist-written reports. No bounding
boxes, but a CheXbert-format CSV ships per-image 14-class pathology labels
(No Finding + 13 CheXpert classes). The loader turns those labels into
structured annotations usable by the claim matcher so OpenI claims can be
resolved to SUPPORTED / CONTRADICTED even without pixel grounding.

Access: NLM — fully public, no credentialing.

On-disk layout (Laughney lab):
  image_root/            e.g. ~/data/openi/
    CXR{uid}_IM-*.png    one or more views per study
  report_csv             e.g. ~/data/claimguard/iu-xray/iu_xray_reports.csv
    columns: uid, findings, impression, indication, problems, mesh
  chexbert_csv           e.g. /data/iu_xray_meta/iu_xray_chexpert_format.csv
    columns: patient_id, study_id, + 14 CheXpert-class labels (0 / 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation

# Default paths matching the Laughney-lab layout.
DEFAULT_IMAGE_ROOT = Path.home() / "data" / "openi"
DEFAULT_REPORT_CSV = Path.home() / "data" / "claimguard" / "iu-xray" / "iu_xray_reports.csv"
DEFAULT_CHEXBERT_CSV = Path("/data/iu_xray_meta/iu_xray_chexpert_format.csv")

# Map CheXbert column header -> v5 ontology finding name.
_CHEXBERT_COLS_TO_FINDING = {
    "No Finding": "no_finding",
    "Enlarged Cardiomediastinum": "cardiomegaly",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "opacity",
    "Lung Lesion": "lung_lesion",
    "Edema": "edema",
    "Consolidation": "consolidation",
    "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis",
    "Pneumothorax": "pneumothorax",
    "Pleural Effusion": "effusion",
    "Pleural Other": "pleural_other",
    "Fracture": "fracture",
    "Support Devices": "support_device",
}

_CHEXBERT_CACHE: dict[str, dict[str, dict[str, int]]] = {}


def _load_chexbert_labels(csv_path: Path) -> dict[str, dict[str, int]]:
    """Load the OpenI CheXbert-format CSV; returns uid -> {finding: 0/1}.

    Labels come out of the CSV as 0 / 1 / -1 / NaN. We keep 0/1 and drop
    the rest as UNKNOWN.
    """
    key = str(csv_path)
    if key in _CHEXBERT_CACHE:
        return _CHEXBERT_CACHE[key]
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    out: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        uid = str(row.get("study_id", row.get("patient_id", ""))).strip()
        if not uid:
            continue
        per_img: dict[str, int] = {}
        for col, finding in _CHEXBERT_COLS_TO_FINDING.items():
            if col not in row:
                continue
            v = row[col]
            try:
                vi = int(v)
            except (ValueError, TypeError):
                continue
            if vi in (0, 1):
                per_img[finding] = vi
        out[uid] = per_img
    _CHEXBERT_CACHE[key] = out
    return out


@dataclass
class OpenIRecord:
    image_id: str
    image_path: Path
    report_findings: str
    report_impression: str
    mesh_terms: list[str] = field(default_factory=list)
    # CheXbert-style per-image labels; finding name -> 0 (absent) or 1 (present).
    chexbert_labels: dict[str, int] = field(default_factory=dict)


def iter_openi(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    report_csv: Path = DEFAULT_REPORT_CSV,
    chexbert_csv: Path | None = None,
) -> Iterator[OpenIRecord]:
    """Yield one OpenIRecord per (study × frontal view) pair that has an image on disk.

    If ``chexbert_csv`` is supplied (defaults to ``DEFAULT_CHEXBERT_CSV`` when
    it exists), the per-image CheXbert pathology labels are attached to each
    record and later surfaced as structured annotations by
    :func:`annotations_for_record`.
    """
    import pandas as pd

    if chexbert_csv is None:
        chexbert_csv = DEFAULT_CHEXBERT_CSV
    chex_labels: dict[str, dict[str, int]] = {}
    if chexbert_csv and Path(chexbert_csv).exists():
        chex_labels = _load_chexbert_labels(Path(chexbert_csv))

    df = pd.read_csv(report_csv, low_memory=False)
    for _, row in df.iterrows():
        uid = str(row.get("uid", "")).strip()
        if not uid:
            continue

        candidates = sorted(image_root.glob(f"CXR{uid}_IM-*.png"))
        frontal = [p for p in candidates if p.stem.endswith("1001")]
        img_path = frontal[0] if frontal else (candidates[0] if candidates else None)
        if img_path is None:
            continue

        findings = str(row.get("findings", "") or "").strip()
        impression = str(row.get("impression", "") or "").strip()
        mesh_raw = str(row.get("mesh", row.get("problems", "")) or "")
        mesh_terms = [t.strip() for t in mesh_raw.split(";") if t.strip()]

        # CheXbert labels keyed by uid (study_id column in the CSV)
        labels_for_img = chex_labels.get(uid, {})

        yield OpenIRecord(
            image_id=f"openi_{uid}",
            image_path=img_path,
            report_findings=findings,
            report_impression=impression,
            mesh_terms=mesh_terms,
            chexbert_labels=labels_for_img,
        )


def annotations_for_record(rec: OpenIRecord) -> list[Annotation]:
    """Emit structured annotations for OpenI using the CheXbert labels.

    Present findings (label == 1) become plain annotations; absent findings
    (label == 0) become structured negatives. Both kinds are attached to the
    image id but carry no bounding box, since OpenI does not provide pixel
    grounding. The claim matcher will fall through to name-level matching for
    these and resolve most text-extracted claims into SUPPORTED / CONTRADICTED.
    """
    out: list[Annotation] = []
    for finding, present in rec.chexbert_labels.items():
        if finding == "no_finding":
            # A "No Finding" = 1 means no abnormalities; we do NOT emit a
            # present annotation for it (would confuse the matcher). Instead
            # it is handled implicitly: if no other finding is present, the
            # image is considered normal. Emit structured negatives for the
            # 13 CheXpert classes so "no pneumothorax" type claims resolve.
            if present == 1:
                for other in set(_CHEXBERT_COLS_TO_FINDING.values()):
                    if other == "no_finding":
                        continue
                    if other in rec.chexbert_labels:
                        continue
                    out.append(
                        Annotation(
                            image_id=rec.image_id,
                            finding=other,
                            laterality="unknown",
                            bbox=None,
                            source="openi_chexbert_nofinding",
                            is_structured_negative=True,
                        )
                    )
            continue

        out.append(
            Annotation(
                image_id=rec.image_id,
                finding=finding,
                laterality="unknown",
                bbox=None,
                source="openi_chexbert",
                is_structured_negative=(present == 0),
            )
        )
    return out
