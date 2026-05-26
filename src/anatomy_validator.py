"""Anatomy-level validation of generated DICOM volumes.

`src/validate.py` only checks DICOM structure. This module runs
TotalSegmentator on each output volume and checks that the synthesized
anatomy is plausible:

  Hard checks (any failure -> reject volume):
    * sacrum, both femoral heads, both hip bones present
      ("present" = mask >= MIN_VOXELS connected voxels)

  Soft checks (failures logged, don't auto-reject):
    * bilateral symmetry: min(hip_left, hip_right) / max(...) >= SYMMETRY_THRESHOLD
    * pelvic inlet width within +/- INLET_Z_THRESHOLD sigma of the real
      training distribution (read from cache/region_labels.csv)

Output: synthetic_dataset/anatomy_report.json with per-patient verdicts.

TotalSegmentator runs on GPU if available, CPU otherwise. First call
downloads ~2 GB of model weights into the user cache. Per-volume runtime
~10-30 s on a GTX 1080 Ti.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


REQUIRED_STRUCTURES = ["sacrum", "femur_left", "femur_right", "hip_left", "hip_right"]
SOFT_STRUCTURES = ["urinary_bladder", "vertebrae_L5"]
MIN_VOXELS = 1000               # below this, structure considered missing
SYMMETRY_THRESHOLD = 0.85       # min(L,R) / max(L,R) below this is asymmetric
INLET_Z_THRESHOLD = 2.0         # |z-score| above this is anatomically suspect


# ----- TotalSegmentator wrapper -------------------------------------------

def _run_total_segmentator(dicom_dir: Path, out_dir: Path,
                           roi_subset: list[str]) -> None:
    """Invoke TotalSegmentator on a DICOM folder. Raises if not installed."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as e:
        raise RuntimeError(
            "totalsegmentator not installed. Run:\n"
            "  pip install totalsegmentator\n"
            "(downloads ~2 GB of model weights on first use)"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        input=str(dicom_dir),
        output=str(out_dir),
        roi_subset=roi_subset,
        fast=True,                    # 3 mm resolution model, ~5x faster
        ml=False,                     # write per-structure files, not multilabel
        quiet=True,
    )


# ----- Per-volume validation ----------------------------------------------

@dataclass
class AnatomyReport:
    patient_id: str
    ok: bool
    issues: list[str]
    structures: dict          # {name: {present: bool, voxels: int}}
    symmetry: Optional[float] = None
    inlet_mm: Optional[float] = None
    inlet_z: Optional[float] = None


def _voxel_count(mask_path: Path) -> int:
    import nibabel as nib
    if not mask_path.exists():
        return 0
    return int(np.asanyarray(nib.load(mask_path).dataobj).astype(bool).sum())


def _pelvic_inlet_mm_from_hips(mask_dir: Path) -> Optional[float]:
    """Approximate pelvic inlet width = max axial extent of (hip_left | hip_right)."""
    import nibabel as nib

    l_path = mask_dir / "hip_left.nii.gz"
    r_path = mask_dir / "hip_right.nii.gz"
    if not l_path.exists() or not r_path.exists():
        return None
    l_img = nib.load(l_path)
    r_img = nib.load(r_path)
    l = np.asanyarray(l_img.dataobj).astype(bool)
    r = np.asanyarray(r_img.dataobj).astype(bool)
    combined = l | r
    spacing = l_img.header.get_zooms()[:3]      # (sx, sy, sz) per nib canonical
    # Find slice with max X extent (in voxels), convert to mm.
    extents_mm = []
    for z in range(combined.shape[2]):
        slc = combined[:, :, z]
        if slc.any():
            xs = np.where(slc.any(axis=1))[0]
            extents_mm.append((xs.max() - xs.min() + 1) * float(spacing[0]))
    return float(max(extents_mm)) if extents_mm else None


def validate_volume_anatomy(dicom_dir: str | Path,
                             ref_inlet_mean: float,
                             ref_inlet_std: float,
                             patient_id: str = "") -> AnatomyReport:
    """Run TotalSegmentator + structural checks on one synthetic DICOM volume."""
    dicom_dir = Path(dicom_dir)
    patient_id = patient_id or dicom_dir.parent.name
    issues: list[str] = []
    structures: dict = {}

    with tempfile.TemporaryDirectory(prefix="ts_") as tmp:
        mask_dir = Path(tmp) / "masks"
        _run_total_segmentator(dicom_dir, mask_dir,
                               REQUIRED_STRUCTURES + SOFT_STRUCTURES)

        for s in REQUIRED_STRUCTURES:
            n = _voxel_count(mask_dir / f"{s}.nii.gz")
            present = n >= MIN_VOXELS
            structures[s] = {"present": bool(present), "voxels": n}
            if not present:
                issues.append(f"missing required: {s} ({n} voxels)")

        for s in SOFT_STRUCTURES:
            n = _voxel_count(mask_dir / f"{s}.nii.gz")
            structures[s] = {"present": bool(n >= MIN_VOXELS), "voxels": n}

        # Bilateral hip symmetry
        l = structures.get("hip_left", {}).get("voxels", 0)
        r = structures.get("hip_right", {}).get("voxels", 0)
        symmetry = None
        if l > 0 and r > 0:
            symmetry = float(min(l, r) / max(l, r))
            if symmetry < SYMMETRY_THRESHOLD:
                issues.append(f"asymmetric hips (sym={symmetry:.2f} < {SYMMETRY_THRESHOLD})")

        # Pelvic inlet width
        inlet_mm = _pelvic_inlet_mm_from_hips(mask_dir)
        inlet_z = None
        if inlet_mm is not None and ref_inlet_std > 1e-3:
            inlet_z = (inlet_mm - ref_inlet_mean) / ref_inlet_std
            if abs(inlet_z) > INLET_Z_THRESHOLD:
                issues.append(
                    f"pelvic inlet {inlet_mm:.0f}mm is {inlet_z:+.1f} sigma "
                    f"from real (mean {ref_inlet_mean:.0f} +- {ref_inlet_std:.0f})"
                )

    # Hard fail = any "missing required" issue. Soft fails (symmetry, inlet
    # z-score) are recorded but don't flip ok=False -- the volume is still
    # delivered but flagged in the report.
    hard_fail = any(i.startswith("missing required") for i in issues)
    return AnatomyReport(
        patient_id=patient_id,
        ok=not hard_fail,
        issues=issues,
        structures=structures,
        symmetry=round(symmetry, 3) if symmetry is not None else None,
        inlet_mm=round(inlet_mm, 1) if inlet_mm is not None else None,
        inlet_z=round(inlet_z, 2) if inlet_z is not None else None,
    )


# ----- Dataset-level driver -----------------------------------------------

def reference_inlet_stats(labels_csv: str | Path) -> tuple[float, float]:
    """Read (mean, std) of pelvic_inlet_mm from the training labels CSV."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    vals = df["pelvic_inlet_mm"].to_numpy()
    return float(vals.mean()), float(vals.std())


def validate_dataset_anatomy(out_root: str | Path, labels_csv: str | Path,
                              progress_cb: Optional[Callable[[str, bool, list[str]], None]] = None
                              ) -> dict:
    """Run anatomy validation on every patient under out_root.

    Writes out_root/anatomy_report.json. Returns the summary dict.
    """
    out_root = Path(out_root)
    ref_mean, ref_std = reference_inlet_stats(labels_csv)
    patient_dirs = sorted(p for p in out_root.iterdir()
                          if p.is_dir() and (p / "DICOM").is_dir())
    if not patient_dirs:
        return {"n_patients": 0, "n_ok": 0, "reports": []}

    reports = []
    n_ok = 0
    for pdir in patient_dirs:
        try:
            r = validate_volume_anatomy(pdir / "DICOM", ref_mean, ref_std, patient_id=pdir.name)
            r_dict = r.__dict__
        except Exception as e:
            r_dict = {"patient_id": pdir.name, "ok": False,
                      "issues": [f"validator error: {type(e).__name__}: {e}"],
                      "structures": {}, "symmetry": None,
                      "inlet_mm": None, "inlet_z": None}
        reports.append(r_dict)
        if r_dict["ok"]:
            n_ok += 1
        if progress_cb:
            progress_cb(pdir.name, r_dict["ok"], r_dict["issues"])

    summary = {
        "n_patients": len(patient_dirs),
        "n_ok": n_ok,
        "reference_inlet_mm_mean": round(ref_mean, 1),
        "reference_inlet_mm_std": round(ref_std, 1),
        "thresholds": {
            "min_voxels": MIN_VOXELS,
            "symmetry_threshold": SYMMETRY_THRESHOLD,
            "inlet_z_threshold": INLET_Z_THRESHOLD,
        },
        "reports": reports,
    }
    (out_root / "anatomy_report.json").write_text(json.dumps(summary, indent=2))
    return summary
