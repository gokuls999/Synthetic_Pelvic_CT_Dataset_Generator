"""TotalSegmentator wrapper for cached pelvic volumes.

For a cache .npz file we:
  1. Un-window the slices back to HU
  2. Save as a temp NIfTI
  3. Run TotalSegmentator (fast=True, 3 mm model -- 5-10 min/volume on CPU)
  4. Load per-structure masks back as (Z, H, W) boolean arrays aligned with
     the original cached volume
  5. Cache the masks to disk under cache/masks/<dataset>/<uid>/<organ>.npy so
     subsequent runs are instant (~50 ms)

Used by the Phase-1 PFD pipeline to anchor deformation fields to the ACTUAL
organ positions (cystocele = displace the bladder mask centroid; rectocele =
displace the lower-colon centroid; uterine prolapse = displace the midpoint
between bladder and rectum since TotalSegmentator does not segment the uterus).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# TotalSegmentator structure names we ask for. The standard model (fast=True)
# does not have a uterus class -- we synthesize the uterus location from the
# bladder + rectum centroids in pfd_deformation.build_pattern_from_masks.
PFD_ROI_SUBSET = [
    "urinary_bladder",
    "colon",           # rectum = inferior portion of colon
    "sacrum",          # for posterior reference
    "hip_left",        # for PCL anchor (pubic symphysis approx)
    "hip_right",
]

# Extended subset for segmentation overlay generation (includes pelvic muscles).
# TotalSegmentator total task (fast=True) includes all of these.
# First run per source CT takes 5-10 min; subsequent runs are instant (cached).
SEG_SUBSET = PFD_ROI_SUBSET + [
    "femur_left",
    "femur_right",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "iliopsoas_left",
    "iliopsoas_right",
]


@dataclass
class MaskStats:
    """Per-organ summary: centroid + bounding-box extent + voxel count."""
    name: str
    voxels: int
    center: tuple[float, float, float]    # (z, y, x) in voxels
    extent: tuple[int, int, int]          # (dz, dy, dx) in voxels
    bbox_min: tuple[int, int, int]
    bbox_max: tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "name": self.name, "voxels": int(self.voxels),
            "center": list(self.center), "extent": list(self.extent),
            "bbox_min": list(self.bbox_min), "bbox_max": list(self.bbox_max),
        }


# -------------------------------------------------------------------------
# Cache npz <-> NIfTI conversion
# -------------------------------------------------------------------------

def _load_cache_volume(npz_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return ([Z,H,W] float32 in [-1,1], (sz, sy, sx) mm spacing)."""
    with np.load(npz_path) as npz:
        slices = np.asarray(npz["slices"]).astype(np.float32)
        try:
            sp = tuple(float(x) for x in np.asarray(npz["spacing"]).tolist())
            if len(sp) != 3:
                sp = (1.5, 1.0, 1.0)
        except Exception:
            sp = (1.5, 1.0, 1.0)
    return slices, sp


def _cache_volume_to_nifti(npz_path: Path, hu_min: float, hu_max: float,
                           out_nifti: Path,
                           stretch_for_ts: bool = True) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Cache npz -> HU int16 -> NIfTI saved at out_nifti.
    Returns the [Z,H,W] HU volume and (sz, sy, sx) spacing.

    stretch_for_ts: the training cache stores slices windowed to a NARROW
    HU range (e.g. [-200, 500]) which optimizes bone+soft-tissue contrast
    for the generator but flattens out the air/bone extremes that TS uses
    to find anatomy. When stretch_for_ts=True we re-map the cache's
    normalized [-1, 1] range to a WIDE synthetic [-1000, +1500] HU range
    before writing the NIfTI. This restores the contrast TS expects (air
    at -1000, bone bright) at the cost of inventing exact HU values --
    fine because we only use the masks for spatial anchoring, never to
    report HU.
    """
    import nibabel as nib
    from .preprocessing import unwindow

    slices, spacing = _load_cache_volume(npz_path)
    if stretch_for_ts:
        # Linear remap [-1, 1] -> [HU_LO_FOR_TS, HU_HI_FOR_TS]
        HU_LO_FOR_TS, HU_HI_FOR_TS = -1000.0, 1500.0
        norm01 = (slices.astype(np.float32) + 1.0) * 0.5
        hu = norm01 * (HU_HI_FOR_TS - HU_LO_FOR_TS) + HU_LO_FOR_TS
    else:
        hu = unwindow(slices, hu_min, hu_max)
    hu_int = np.clip(hu, -1024, 3071).astype(np.int16)

    # NIfTI conventional storage is [X, Y, Z]. Our slices are [Z, H, W] = [Z, Y, X].
    # Transpose to [X, Y, Z] so the affine units match.
    data_xyz = np.transpose(hu_int, (2, 1, 0))    # (W, H, Z)
    sz, sy, sx = spacing
    affine = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
    nib.save(nib.Nifti1Image(data_xyz, affine), str(out_nifti))
    return hu_int, spacing


def _load_mask_nifti(path: Path, ref_shape_zhw: tuple[int, int, int]) -> np.ndarray:
    """Load a TS mask NIfTI written in [X, Y, Z] space, return [Z, H, W] bool."""
    import nibabel as nib
    img = nib.load(str(path))
    data_xyz = np.asanyarray(img.dataobj).astype(np.uint8)
    mask = np.transpose(data_xyz, (2, 1, 0)).astype(bool)
    if mask.shape != ref_shape_zhw:
        # TotalSegmentator sometimes resamples; warn but keep going.
        # Caller can decide whether to fall back to fractional anchors.
        raise RuntimeError(
            f"mask shape {mask.shape} != reference {ref_shape_zhw} for {path.name}"
        )
    return mask


# -------------------------------------------------------------------------
# TotalSegmentator runner
# -------------------------------------------------------------------------

def _run_ts(nifti_in: Path, out_dir: Path,
            roi_subset: list[str], fast: bool = True) -> None:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as e:
        raise RuntimeError(
            "totalsegmentator not installed. Run: pip install totalsegmentator"
        ) from e
    out_dir.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        input=str(nifti_in), output=str(out_dir),
        roi_subset=roi_subset, fast=fast, ml=False, quiet=True,
    )


# -------------------------------------------------------------------------
# Mask stats
# -------------------------------------------------------------------------

def mask_stats(mask: np.ndarray, name: str) -> Optional[MaskStats]:
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    center = coords.mean(axis=0).astype(np.float64)
    bb_min = coords.min(axis=0).astype(np.int64)
    bb_max = coords.max(axis=0).astype(np.int64)
    extent = (bb_max - bb_min + 1).astype(np.int64)
    return MaskStats(
        name=name, voxels=int(coords.shape[0]),
        center=tuple(map(float, center)),
        extent=tuple(map(int, extent)),
        bbox_min=tuple(map(int, bb_min)),
        bbox_max=tuple(map(int, bb_max)),
    )


def pelvic_z_range_from_masks(masks: dict[str, np.ndarray],
                              margin_voxels: int = 6,
                              fallback: tuple[int, int] | None = None
                              ) -> tuple[int, int]:
    """Return (z_top, z_bot) defining the pelvic-only Z extent from TS masks.

    Uses the union of sacrum + hip_left + hip_right as the bone scaffold of
    the pelvis. The pelvic Z range is the min..max Z over those voxels,
    plus a small margin.

    This corrects the pre-existing preprocessing bug where pelvic_z_range()
    in src/preprocessing.py kept the whole spine region (thorax + abdomen +
    pelvis) for whole-trunk source CTs, leaving every cache volume
    contaminated with abdominal anatomy.

    Returns `fallback` (or (0, Z) if not supplied) if no pelvic bone mask
    has any voxels.
    """
    bone = None
    for name in ("sacrum", "hip_left", "hip_right"):
        m = masks.get(name)
        if m is None or not m.any():
            continue
        bone = m if bone is None else (bone | m)
    if bone is None or not bone.any():
        if fallback is not None:
            return fallback
        Z = 0
        for m in masks.values():
            Z = max(Z, m.shape[0])
        return (0, Z)

    coords_z = np.where(bone.any(axis=(1, 2)))[0]
    z_top = int(coords_z.min()) - margin_voxels
    z_bot = int(coords_z.max()) + 1 + margin_voxels
    z_top = max(0, z_top)
    z_bot = min(bone.shape[0], z_bot)
    if z_bot - z_top < 8:           # sanity: must be at least 8 slices
        if fallback is not None:
            return fallback
        return (0, bone.shape[0])
    return (z_top, z_bot)


def rectum_subregion(colon_mask: np.ndarray, frac: float = 0.66) -> np.ndarray:
    """Rectum = inferior portion of the colon. In our cached pelvic volumes
    'inferior' = larger Z (DICOM caudal). Take the lowest ~1/3 of colon voxels
    by Z and return that subregion as a fresh mask."""
    if not colon_mask.any():
        return colon_mask.copy()
    coords = np.argwhere(colon_mask)
    z_min, z_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    threshold = z_min + int(round((z_max - z_min) * frac))
    sub = np.zeros_like(colon_mask)
    sub[threshold:] = colon_mask[threshold:]
    # Guard: if too few voxels (volume crops cut the colon high), fall back to half.
    if sub.sum() < 50:
        threshold = z_min + int(round((z_max - z_min) * 0.5))
        sub = np.zeros_like(colon_mask)
        sub[threshold:] = colon_mask[threshold:]
    return sub


# -------------------------------------------------------------------------
# Public: segment_volume (with on-disk mask cache)
# -------------------------------------------------------------------------

def _mask_cache_dir(cache_root: Path, dataset: str, uid: str) -> Path:
    return cache_root / "masks" / dataset / uid


def cached_masks_exist(cache_root: Path, dataset: str, uid: str,
                       roi_subset: list[str] = PFD_ROI_SUBSET) -> bool:
    d = _mask_cache_dir(cache_root, dataset, uid)
    return all((d / f"{name}.npy").exists() for name in roi_subset)


def load_cached_masks(cache_root: Path, dataset: str, uid: str,
                      roi_subset: list[str] = PFD_ROI_SUBSET) -> dict[str, np.ndarray]:
    d = _mask_cache_dir(cache_root, dataset, uid)
    return {name: np.load(d / f"{name}.npy") for name in roi_subset}


def segment_original_volume(source_path: Path, dataset: str, uid: str,
                            cfg: dict, cache_root: Path,
                            roi_subset: list[str] = PFD_ROI_SUBSET,
                            on_progress=None) -> tuple[dict[str, np.ndarray], dict[str, MaskStats]]:
    """Load the ORIGINAL DICOM at full HU range, run TotalSegmentator on the
    full (uncropped) volume, then crop+resample the masks the same way
    preprocessing.preprocess_volume cropped+resampled the cache volume.

    Why this exists: TotalSegmentator's training distribution is whole-body
    or whole-trunk CT. Our cache .npz files are pelvic-only crops in a narrow
    HU window ([-200, 500]) -- TS produces empty masks on them. Loading the
    original DICOM gives the spatial+contrast context TS needs to fire.

    The same crop+resample chain is then applied to the masks so they land
    on the exact (Z, H, W) grid of the cache volume that the deformation
    code consumes.

    First call per volume is slow (~5-15 min on CPU). Masks are persisted
    under <cache_root>/masks/<dataset>/<uid>/ so subsequent calls are
    instant (~50 ms).
    """
    import nibabel as nib
    from scipy.ndimage import zoom
    from .io_loaders import load_any
    from .preprocessing import pelvic_z_range, pelvic_xy_bbox

    out_dir = _mask_cache_dir(cache_root, dataset, uid)

    if cached_masks_exist(cache_root, dataset, uid, roi_subset):
        if on_progress:
            on_progress("loading masks from disk cache")
        masks = load_cached_masks(cache_root, dataset, uid, roi_subset)
    else:
        if on_progress:
            on_progress("loading original DICOM")
        vol = load_any(source_path)
        arr = vol.array.astype(np.float32)
        sz, sy, sx = vol.spacing

        with tempfile.TemporaryDirectory() as tdir:
            tdir = Path(tdir)
            nin = tdir / f"{uid}.nii.gz"
            ts_out = tdir / "ts_out"

            data_xyz = np.transpose(
                np.clip(arr, -1024, 3071).astype(np.int16), (2, 1, 0)
            )
            affine = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
            nib.save(nib.Nifti1Image(data_xyz, affine), str(nin))

            if on_progress:
                on_progress(f"TotalSegmentator on full volume ({arr.shape[0]} slices)")
            _run_ts(nin, ts_out, roi_subset=roi_subset, fast=True)

            original_masks: dict[str, np.ndarray] = {}
            for name in roi_subset:
                mp = ts_out / f"{name}.nii.gz"
                if not mp.exists():
                    original_masks[name] = np.zeros(arr.shape, dtype=bool)
                    continue
                m_xyz = np.asanyarray(nib.load(str(mp)).dataobj).astype(bool)
                original_masks[name] = np.transpose(m_xyz, (2, 1, 0))

        if on_progress:
            on_progress("cropping + resampling masks to cache coordinates")
        pp = cfg["preprocess"]
        z0, z1 = pelvic_z_range(arr, min_span_mm=60.0, sz_mm=sz)
        cropped_arr = arr[z0:z1]
        margin_px = max(1, int(pp["margin_mm"] / max(sx, sy)))
        y0, y1, x0, x1 = pelvic_xy_bbox(cropped_arr, margin_px=margin_px)
        z_stride = int(pp.get("z_stride", 1))
        slice_size = int(pp["slice_size"])

        masks: dict[str, np.ndarray] = {}
        for name, m_full in original_masks.items():
            m_crop = m_full[z0:z1, y0:y1, x0:x1]
            if z_stride > 1:
                m_crop = m_crop[::z_stride]
            h_now, w_now = m_crop.shape[1], m_crop.shape[2]
            if h_now != w_now:
                side = max(h_now, w_now)
                pad_h_before = (side - h_now) // 2
                pad_w_before = (side - w_now) // 2
                m_crop = np.pad(
                    m_crop,
                    ((0, 0),
                     (pad_h_before, side - h_now - pad_h_before),
                     (pad_w_before, side - w_now - pad_w_before)),
                    mode="constant", constant_values=False,
                )
            h2 = m_crop.shape[1]
            zoom_fac = (1.0, slice_size / h2, slice_size / h2)
            m_resampled = zoom(m_crop.astype(np.uint8), zoom_fac,
                               order=0, prefilter=False).astype(bool)
            masks[name] = m_resampled

        out_dir.mkdir(parents=True, exist_ok=True)
        for name, m in masks.items():
            np.save(out_dir / f"{name}.npy", m)
        if on_progress:
            on_progress(f"cached masks -> {out_dir}")

    stats: dict[str, MaskStats] = {}
    for name, m in masks.items():
        st = mask_stats(m, name)
        if st is not None:
            stats[name] = st
    (out_dir / "stats.json").write_text(json.dumps(
        {k: v.to_dict() for k, v in stats.items()}, indent=2
    ))
    return masks, stats


def segment_volume(npz_path: Path, dataset: str, uid: str,
                   hu_min: float, hu_max: float,
                   cache_root: Path,
                   roi_subset: list[str] = PFD_ROI_SUBSET,
                   on_progress=None) -> tuple[dict[str, np.ndarray], dict[str, MaskStats]]:
    """Run TotalSegmentator on a cache volume (or load from disk cache).

    Returns:
        masks  : {organ_name: [Z,H,W] bool}
        stats  : {organ_name: MaskStats}

    The first call for a given volume is slow (~5-10 min on CPU with fast=True).
    Subsequent calls are instant (~50 ms) -- masks are persisted under
    <cache_root>/masks/<dataset>/<uid>/<organ>.npy.
    """
    out_dir = _mask_cache_dir(cache_root, dataset, uid)
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = cached_masks_exist(cache_root, dataset, uid, roi_subset)
    if cached:
        if on_progress:
            on_progress("loading masks from cache")
        masks = load_cached_masks(cache_root, dataset, uid, roi_subset)
    else:
        if on_progress:
            on_progress("running TotalSegmentator (fast=True)")
        slices, _ = _load_cache_volume(npz_path)
        Z, H, W = slices.shape
        with tempfile.TemporaryDirectory() as tdir:
            tdir = Path(tdir)
            nifti_in = tdir / f"{uid}.nii.gz"
            ts_out = tdir / "ts_out"
            _cache_volume_to_nifti(npz_path, hu_min, hu_max, nifti_in)
            _run_ts(nifti_in, ts_out, roi_subset=roi_subset, fast=True)

            masks = {}
            for name in roi_subset:
                mask_path = ts_out / f"{name}.nii.gz"
                if not mask_path.exists():
                    masks[name] = np.zeros((Z, H, W), dtype=bool)
                    continue
                try:
                    masks[name] = _load_mask_nifti(mask_path, (Z, H, W))
                except RuntimeError:
                    # Shape mismatch; record empty so caller falls back.
                    masks[name] = np.zeros((Z, H, W), dtype=bool)

        # Persist to disk cache
        for name, m in masks.items():
            np.save(out_dir / f"{name}.npy", m)
        if on_progress:
            on_progress(f"cached masks -> {out_dir}")

    # Compute stats
    stats: dict[str, MaskStats] = {}
    for name, m in masks.items():
        st = mask_stats(m, name)
        if st is not None:
            stats[name] = st

    # Also persist stats summary as JSON for debugging
    summary_path = out_dir / "stats.json"
    summary_path.write_text(json.dumps(
        {k: v.to_dict() for k, v in stats.items()}, indent=2
    ))
    return masks, stats
