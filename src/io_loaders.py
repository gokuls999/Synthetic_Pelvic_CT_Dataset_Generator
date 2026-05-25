"""Load CT volumes (DICOM series or NIfTI) into a common (volume_HU, spacing) form.

A volume is returned as `np.ndarray` of shape (Z, Y, X) in Hounsfield Units (HU),
with `spacing` as a 3-tuple (sz, sy, sx) in millimetres, axial-first orientation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Volume:
    array: np.ndarray             # (Z, Y, X) HU, float32
    spacing: tuple[float, float, float]   # (sz, sy, sx) mm
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source: str = ""              # human-readable origin path
    patient_id: str = ""
    series_id: str = ""


# ----- DICOM ---------------------------------------------------------------

def load_dicom_series(series_dir: str | os.PathLike) -> Volume:
    """Read a folder of .dcm slices into a Volume sorted by ImagePositionPatient[2]."""
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(p for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")
    if not files:
        raise FileNotFoundError(f"No .dcm files under {series_dir}")

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        slices.append(ds)

    if not slices:
        raise RuntimeError(f"No readable pixel slices in {series_dir}")

    # Sort by z position; fall back to InstanceNumber.
    def _z(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=_z)

    first = slices[0]
    rows = int(first.Rows)
    cols = int(first.Columns)

    arr = np.zeros((len(slices), rows, cols), dtype=np.float32)
    for i, ds in enumerate(slices):
        px = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr[i] = px * slope + intercept

    # Spacing
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    sy, sx = float(pixel_spacing[0]), float(pixel_spacing[1])
    # Slice thickness: prefer derived from positions if available
    if len(slices) > 1:
        z0 = _z(slices[0]); z1 = _z(slices[1])
        sz = abs(z1 - z0) or float(getattr(first, "SliceThickness", 1.0) or 1.0)
    else:
        sz = float(getattr(first, "SliceThickness", 1.0) or 1.0)

    origin = tuple(float(v) for v in getattr(first, "ImagePositionPatient", (0.0, 0.0, 0.0)))

    pid = str(getattr(first, "PatientID", series_dir.parent.name))
    sid = str(getattr(first, "SeriesInstanceUID", series_dir.name))

    return Volume(
        array=arr,
        spacing=(sz, sy, sx),
        origin=origin,
        source=str(series_dir),
        patient_id=pid,
        series_id=sid,
    )


# ----- NIfTI ---------------------------------------------------------------

def load_nifti(path: str | os.PathLike) -> Volume:
    """Read a .nii / .nii.gz volume into HU, oriented so axis 0 is axial slices."""
    import nibabel as nib

    path = Path(path)
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)        # RAS+ orientation
    data = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    # nibabel canonical layout is (X, Y, Z) with Z as axial. Transpose to (Z, Y, X).
    data = np.transpose(data, (2, 1, 0))
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    return Volume(
        array=data,
        spacing=(sz, sy, sx),
        origin=(0.0, 0.0, 0.0),
        source=str(path),
        patient_id=path.stem.replace(".nii", ""),
        series_id=path.stem.replace(".nii", ""),
    )


# ----- Dataset walkers -----------------------------------------------------
#
# These walk the project-specific `output/<dataset>/...` layouts documented in
# DATASET_HANDOFF.md and yield (dataset_name, volume_loader_callable, identity).
# The loader is deferred so the cache builder can skip already-cached series
# without paying the read cost.

def _iter_dicom_series_dirs(root: Path, depth: int) -> Iterator[Path]:
    """Yield directories at `depth` levels below `root` that contain .dcm files."""
    def recurse(p: Path, d: int):
        if d == 0:
            if any(f.suffix.lower() == ".dcm" for f in p.iterdir() if f.is_file()):
                yield p
            return
        for child in p.iterdir():
            if child.is_dir():
                yield from recurse(child, d - 1)
    if not root.exists():
        return
    yield from recurse(root, depth)


def walk_rsna(data_root: Path) -> Iterator[tuple[str, Path, str]]:
    """RSNA: pelvic_cts/train_images/{pid}/{series}/*.dcm  -> depth 2 from train_images."""
    base = data_root / "rsna-2023-abdominal-trauma-detection" / "pelvic_cts" / "train_images"
    for d in _iter_dicom_series_dirs(base, depth=2):
        pid = d.parent.name
        sid = d.name
        yield "rsna", d, f"{pid}__{sid}"


def walk_ct_colonography(data_root: Path) -> Iterator[tuple[str, Path, str]]:
    """CT-Colonography: female_cts/{PatientUID}/{StudyUID}/{SeriesUID}/*.dcm -> depth 3."""
    base = data_root / "ct_colonography" / "female_cts"
    for d in _iter_dicom_series_dirs(base, depth=3):
        sid = d.name
        pid = d.parent.parent.name
        yield "ct_colonography", d, f"{pid}__{sid}"


def walk_tcga(data_root: Path, name: str) -> Iterator[tuple[str, Path, str]]:
    """TCGA-UCEC/OV/CESC: female_cts/<patient>/... -- find every dir that has .dcm files."""
    base = data_root / name / "female_cts"
    if not base.exists():
        return
    for series_dir in _all_dicom_dirs(base):
        # Identify patient: first level below base.
        try:
            rel = series_dir.relative_to(base)
            pid = rel.parts[0] if rel.parts else series_dir.name
        except ValueError:
            pid = series_dir.parent.name
        sid = series_dir.name
        yield name, series_dir, f"{pid}__{sid}"


def _all_dicom_dirs(root: Path) -> Iterator[Path]:
    for cur, _dirs, files in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in files):
            yield Path(cur)


def walk_ctpelvic1k(data_root: Path) -> Iterator[tuple[str, Path, str]]:
    """CTPelvic1K: female_cts/*.nii.gz."""
    base = data_root / "ctpelvic1k" / "female_cts"
    if not base.exists():
        return
    for p in sorted(base.glob("*.nii*")):
        if p.is_file():
            yield "ctpelvic1k", p, p.stem.replace(".nii", "")


DATASET_WALKERS = {
    "rsna": walk_rsna,
    "ct_colonography": walk_ct_colonography,
    "ctpelvic1k": walk_ctpelvic1k,
    "tcga_ucec": lambda r: walk_tcga(r, "tcga_ucec"),
    "tcga_ov": lambda r: walk_tcga(r, "tcga_ov"),
    "tcga_cesc": lambda r: walk_tcga(r, "tcga_cesc"),
}


def walk_all(data_root: str | os.PathLike,
             enable: dict[str, bool] | None = None,
             max_per_dataset: int | None = None,
             ) -> Iterator[tuple[str, Path, str]]:
    """Walk every enabled dataset under `data_root` and yield (name, path, uid)."""
    data_root = Path(data_root)
    enable = enable or {k: True for k in DATASET_WALKERS}
    for name, walker in DATASET_WALKERS.items():
        if not enable.get(name, True):
            continue
        count = 0
        for tup in walker(data_root):
            yield tup
            count += 1
            if max_per_dataset is not None and count >= max_per_dataset:
                break


def load_any(path: Path) -> Volume:
    """Dispatch loader based on path type."""
    if path.is_dir():
        return load_dicom_series(path)
    return load_nifti(path)
