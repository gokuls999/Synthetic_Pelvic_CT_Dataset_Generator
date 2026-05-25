"""Sanity-check a generated DICOM study before opening in 3D Slicer.

Verifies: shared StudyInstanceUID/SeriesInstanceUID, strictly increasing
ImagePositionPatient[Z], consistent rows/cols, HU range plausible, modality CT.
"""

from __future__ import annotations

import json
from pathlib import Path


def validate_dicom_series(series_dir: str | Path) -> dict:
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(p for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")
    issues: list[str] = []
    if len(files) < 2:
        issues.append(f"Too few slices: {len(files)}")
        return {"ok": False, "n_files": len(files), "issues": issues}

    dsets = [pydicom.dcmread(str(f), stop_before_pixels=False) for f in files]
    study_uids = {str(d.StudyInstanceUID) for d in dsets}
    series_uids = {str(d.SeriesInstanceUID) for d in dsets}
    if len(study_uids) != 1:
        issues.append(f"StudyInstanceUID inconsistent: {len(study_uids)} distinct")
    if len(series_uids) != 1:
        issues.append(f"SeriesInstanceUID inconsistent: {len(series_uids)} distinct")

    modalities = {str(d.Modality) for d in dsets}
    if modalities != {"CT"}:
        issues.append(f"Non-CT modality present: {modalities}")

    rows = {int(d.Rows) for d in dsets}
    cols = {int(d.Columns) for d in dsets}
    if len(rows) != 1 or len(cols) != 1:
        issues.append(f"Rows/Cols inconsistent: rows={rows}, cols={cols}")

    zs = [float(d.ImagePositionPatient[2]) for d in dsets]
    diffs = [b - a for a, b in zip(zs, zs[1:])]
    if not all(d > 0 for d in diffs):
        issues.append("ImagePositionPatient[Z] is not strictly increasing")

    # HU range sanity
    import numpy as np
    sample = dsets[len(dsets) // 2]
    px = sample.pixel_array.astype("int32") * int(getattr(sample, "RescaleSlope", 1)) \
         + int(getattr(sample, "RescaleIntercept", 0))
    hu_min, hu_max = int(px.min()), int(px.max())
    if hu_min < -2000 or hu_max > 5000:
        issues.append(f"HU range out of bounds: [{hu_min}, {hu_max}]")

    return {
        "ok": len(issues) == 0,
        "n_files": len(files),
        "study_uid": study_uids.pop() if len(study_uids) == 1 else None,
        "series_uid": series_uids.pop() if len(series_uids) == 1 else None,
        "rows": next(iter(rows)) if len(rows) == 1 else None,
        "cols": next(iter(cols)) if len(cols) == 1 else None,
        "z_first": zs[0], "z_last": zs[-1],
        "hu_range_mid_slice": [hu_min, hu_max],
        "issues": issues,
    }


def validate_dataset(out_root: str | Path) -> dict:
    from .progress import pbar

    out_root = Path(out_root)
    patient_dirs = sorted(p for p in out_root.iterdir() if (p / "DICOM").is_dir())
    reports = []
    n_ok = 0
    bar = pbar(patient_dirs, desc="Validating", unit="pt")
    for pdir in bar:
        rpt = validate_dicom_series(pdir / "DICOM")
        rpt["patient_id"] = pdir.name
        reports.append(rpt)
        if rpt["ok"]:
            n_ok += 1
        bar.set_postfix(ok=n_ok, fail=len(reports) - n_ok)
    bar.close()
    summary = {"n_patients": len(patient_dirs), "n_ok": n_ok, "reports": reports}
    (out_root / "validation_report.json").write_text(json.dumps(summary, indent=2))
    return summary
