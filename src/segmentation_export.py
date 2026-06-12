"""Write pelvic organ + muscle masks as a 3D Slicer .seg.nrrd segmentation file.

Each patient gets one   DICOM/segmentation.seg.nrrd
that Slicer loads via:
  File → Add Data → segmentation.seg.nrrd  (check "Show Options" → Segmentation)

Structures included:
  Bones   — hip L/R, sacrum, femur L/R
  Organs  — urinary bladder, rectum (from colon inferior third)
  Muscles — gluteus maximus/medius/minimus L/R, iliopsoas L/R
  Pelvic floor — estimated levator ani ROI (from sacrum + hip geometry)

The levator ani itself is NOT segmented by TotalSegmentator fast task,
so we generate an estimated Z-slab mask from the sacrum/hip anchors.
It is labeled "Pelvic Floor (levator ani region)" and colored magenta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, gaussian_filter, zoom


# ── Segment colour / label catalogue ──────────────────────────────────────
# label 0 = background (never painted)
SEGMENT_DEFS = [
    # Bones — warm ivory
    {"key": "hip_left",              "label": 1,  "color": (0.87, 0.77, 0.56), "name": "Hip Left"},
    {"key": "hip_right",             "label": 2,  "color": (0.87, 0.77, 0.56), "name": "Hip Right"},
    {"key": "sacrum",                "label": 3,  "color": (0.93, 0.83, 0.62), "name": "Sacrum"},
    {"key": "femur_left",            "label": 4,  "color": (0.82, 0.72, 0.52), "name": "Femur Left"},
    {"key": "femur_right",           "label": 5,  "color": (0.82, 0.72, 0.52), "name": "Femur Right"},
    # Pelvic organs
    {"key": "urinary_bladder",       "label": 6,  "color": (0.22, 0.55, 1.00), "name": "Urinary Bladder"},
    {"key": "rectum",                "label": 7,  "color": (0.78, 0.38, 0.18), "name": "Rectum"},
    # Gluteal muscles — orange family
    {"key": "gluteus_maximus_left",  "label": 8,  "color": (1.00, 0.45, 0.20), "name": "Gluteus Maximus L"},
    {"key": "gluteus_maximus_right", "label": 9,  "color": (1.00, 0.45, 0.20), "name": "Gluteus Maximus R"},
    {"key": "gluteus_medius_left",   "label": 10, "color": (1.00, 0.68, 0.18), "name": "Gluteus Medius L"},
    {"key": "gluteus_medius_right",  "label": 11, "color": (1.00, 0.68, 0.18), "name": "Gluteus Medius R"},
    {"key": "gluteus_minimus_left",  "label": 12, "color": (1.00, 0.88, 0.10), "name": "Gluteus Minimus L"},
    {"key": "gluteus_minimus_right", "label": 13, "color": (1.00, 0.88, 0.10), "name": "Gluteus Minimus R"},
    # Iliopsoas — green
    {"key": "iliopsoas_left",        "label": 14, "color": (0.45, 0.85, 0.28), "name": "Iliopsoas L"},
    {"key": "iliopsoas_right",       "label": 15, "color": (0.45, 0.85, 0.28), "name": "Iliopsoas R"},
    # Estimated pelvic floor (levator ani region) — magenta
    {"key": "_pelvic_floor_est",     "label": 16, "color": (0.92, 0.18, 0.55), "name": "Pelvic Floor (levator ani region)"},
]


# ── Estimated levator ani ROI ──────────────────────────────────────────────

def _estimate_pelvic_floor_mask(masks: dict[str, np.ndarray],
                                 zhw: tuple[int, int, int]) -> np.ndarray:
    """Build a binary mask covering the estimated levator ani Z-slab.

    Derived from sacrum bottom (coccyx level) and hip bone bounds.
    Intentionally conservative: only the soft-tissue floor zone between
    the hip bones, NOT the bones themselves.
    """
    Z, H, W = zhw
    floor_z    = Z * 0.72
    center_x   = W * 0.50
    hip_half_x = W * 0.20

    sacrum = masks.get("sacrum")
    hl     = masks.get("hip_left")
    hr     = masks.get("hip_right")

    if sacrum is not None and sacrum.any():
        sac_z_max = int(np.where(sacrum.any(axis=(1, 2)))[0].max())
        floor_z   = min(float(sac_z_max) + 2.0, float(Z - 1))

    if hl is not None and hr is not None and (hl.any() or hr.any()):
        combined = hl.astype(bool) | hr.astype(bool)
        x_idx    = np.where(combined.any(axis=(0, 1)))[0]
        if len(x_idx) >= 2:
            center_x   = float((x_idx[0] + x_idx[-1]) / 2.0)
            hip_half_x = float((x_idx[-1] - x_idx[0]) / 4.2)

    mask = np.zeros((Z, H, W), dtype=bool)
    z0 = max(0,   int(floor_z - 5))
    z1 = min(Z,   int(floor_z + 12))
    y0 = int(H * 0.28);  y1 = int(H * 0.78)
    x0 = int(center_x - hip_half_x * 1.5)
    x1 = int(center_x + hip_half_x * 1.5)
    x0 = max(0, x0);     x1 = min(W, x1)
    mask[z0:z1, y0:y1, x0:x1] = True

    # Exclude bone voxels so only soft tissue is coloured
    for key in ("sacrum", "hip_left", "hip_right"):
        bone = masks.get(key)
        if bone is not None and bone.any():
            mask &= ~binary_dilation(bone > 0.5, iterations=3)

    return mask


# ── Mask resampler ─────────────────────────────────────────────────────────

def _resample_mask(mask: np.ndarray, z_scale: float,
                   Z_out: int, H: int, W: int) -> np.ndarray:
    """Resample a (Z_in, H, W) binary mask to (Z_out, H, W).

    Uses nearest-neighbour along Z (order=0) to preserve binary values.
    """
    if abs(z_scale - 1.0) > 0.02:
        resampled = zoom(mask.astype(np.float32),
                         (z_scale, 1.0, 1.0), order=0) > 0.5
    else:
        resampled = mask.astype(bool)

    Z_r = resampled.shape[0]
    if Z_r >= Z_out:
        return resampled[:Z_out]
    # Pad with False at the caudal end
    pad = np.zeros((Z_out - Z_r, H, W), dtype=bool)
    return np.concatenate([resampled, pad], axis=0)


# ── Main public function ───────────────────────────────────────────────────

def write_seg_nrrd(out_path: Path,
                   masks_crop: dict[str, np.ndarray],
                   spacing_crop: tuple[float, float, float],
                   spacing_out: tuple[float, float, float],
                   Z_out: int,
                   dicom_dir: Path | None = None) -> None:
    """Write a 3D Slicer .seg.nrrd segmentation file.

    Parameters
    ----------
    out_path      : destination file (e.g. DICOM/segmentation.seg.nrrd)
    masks_crop    : organ masks aligned to pelvic crop, shape (Z_crop, H, W)
    spacing_crop  : (sz, sy, sx) mm — spacing of the pelvic crop volume
    spacing_out   : (sz_out, sy, sx) mm — spacing of the output DICOM
    Z_out         : number of slices in the output DICOM
    dicom_dir     : if provided, reads origin + direction from DICOM series
    """
    # Infer H, W from any mask
    H, W = 256, 256
    for m in masks_crop.values():
        if m.ndim == 3:
            _, H, W = m.shape
            break

    z_scale = spacing_crop[0] / max(spacing_out[0], 1e-6)

    # ── Build label volume ─────────────────────────────────────────────────
    label_vol = np.zeros((Z_out, H, W), dtype=np.uint8)

    for seg in SEGMENT_DEFS:
        key = seg["key"]
        lv  = seg["label"]

        if key == "_pelvic_floor_est":
            raw_mask = _estimate_pelvic_floor_mask(masks_crop, masks_crop[
                next(iter(masks_crop))].shape)
        else:
            raw_mask = masks_crop.get(key)

        if raw_mask is None or not raw_mask.any():
            continue

        resampled = _resample_mask(raw_mask, z_scale, Z_out, H, W)
        # Paint where label_vol is still background (priority = first defined wins)
        label_vol[resampled & (label_vol == 0)] = lv

    # ── Read DICOM origin / direction if available ─────────────────────────
    origin    = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)
    sitk_spacing = (float(spacing_out[2]),   # x
                    float(spacing_out[1]),   # y
                    float(spacing_out[0]))   # z

    if dicom_dir is not None and dicom_dir.is_dir():
        try:
            reader = sitk.ImageSeriesReader()
            fnames  = reader.GetGDCMSeriesFileNames(str(dicom_dir))
            if fnames:
                reader.SetFileNames(fnames)
                dcm = reader.Execute()
                origin    = dcm.GetOrigin()
                direction = dcm.GetDirection()
                sitk_spacing = dcm.GetSpacing()
        except Exception:
            pass   # fall back to default origin

    # ── Build SimpleITK image ──────────────────────────────────────────────
    # SimpleITK uses (W, H, Z) axis order — transpose label_vol
    img = sitk.GetImageFromArray(label_vol)          # (Z, H, W) → numpy default
    img.SetSpacing(sitk_spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)

    # ── Slicer segmentation metadata ──────────────────────────────────────
    img.SetMetaData("Segmentation_ContainedRepresentationNames",
                    "Binary labelmap|Closed surface")

    for i, seg in enumerate(SEGMENT_DEFS):
        r, g, b = seg["color"]
        lv      = seg["label"]
        name    = seg["name"]

        # Compute voxel extent for this label
        vox = (label_vol == lv)
        if vox.any():
            z_idx = np.where(vox.any(axis=(1, 2)))[0]
            y_idx = np.where(vox.any(axis=(0, 2)))[0]
            x_idx = np.where(vox.any(axis=(0, 1)))[0]
            extent = (f"{int(x_idx[0])} {int(x_idx[-1])} "
                      f"{int(y_idx[0])} {int(y_idx[-1])} "
                      f"{int(z_idx[0])} {int(z_idx[-1])}")
        else:
            extent = "0 0 0 0 0 0"

        img.SetMetaData(f"Segment{i}_Color",               f"{r:.4f} {g:.4f} {b:.4f}")
        img.SetMetaData(f"Segment{i}_ColorAutoGenerated",  "1")
        img.SetMetaData(f"Segment{i}_Extent",              extent)
        img.SetMetaData(f"Segment{i}_ID",                  f"Segment_{i}")
        img.SetMetaData(f"Segment{i}_LabelValue",          str(lv))
        img.SetMetaData(f"Segment{i}_Layer",               "0")
        img.SetMetaData(f"Segment{i}_Name",                name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
