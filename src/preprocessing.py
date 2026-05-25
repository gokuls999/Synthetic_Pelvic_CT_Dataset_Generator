"""Volume preprocessing: HU window -> pelvic crop -> axial resample -> cache.

For each input volume we emit a single .npz under cache_dir/<dataset>/<uid>.npz
with:
    slices:   float16, (Z', slice_size, slice_size), in [-1, 1]
    spacing:  (sz, sy_resampled, sx_resampled) mm
    morphometry: small dict with pelvic_inlet_mm, iliac_flare, sacral_tilt (proxies)
    source:   original path
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import zoom

from .io_loaders import Volume, load_any, walk_all


# ----- HU windowing --------------------------------------------------------

def window_hu(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU to [hu_min, hu_max] then map to [-1, 1] (float32)."""
    v = np.clip(volume, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min)     # 0..1
    return (v * 2.0 - 1.0).astype(np.float32)


def unwindow(arr_norm: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Invert window_hu -- used by the DICOM builder to put pixels back in HU."""
    v = (arr_norm + 1.0) / 2.0
    return v * (hu_max - hu_min) + hu_min


# ----- Pelvic cropping -----------------------------------------------------

def bone_mask(slice_hu: np.ndarray, hu_thresh: float = 200.0) -> np.ndarray:
    """Cheap bone segmentation: HU above threshold = bone."""
    return slice_hu >= hu_thresh


def pelvic_z_range(volume_hu: np.ndarray, min_span_mm: float = 60.0,
                   sz_mm: float = 1.0) -> tuple[int, int]:
    """Find Z indices that contain pelvic bone (iliac wings, sacrum, femoral heads).

    Heuristic: per-slice bone area; pelvic slices have bone area well above
    background spine-only slices. We pick a contiguous region with bone area
    >= 50% of the volume-wide max area.
    """
    areas = np.array([bone_mask(s).sum() for s in volume_hu], dtype=np.float32)
    if areas.max() == 0:
        return 0, len(volume_hu)
    thresh = 0.5 * areas.max()
    keep = areas >= thresh
    if not keep.any():
        return 0, len(volume_hu)
    idx = np.where(keep)[0]
    z0, z1 = int(idx[0]), int(idx[-1]) + 1
    # Enforce minimum span (in mm).
    span_mm = (z1 - z0) * sz_mm
    if span_mm < min_span_mm:
        # Fall back to whole volume.
        return 0, len(volume_hu)
    return z0, z1


def pelvic_xy_bbox(volume_hu: np.ndarray, margin_px: int = 16) -> tuple[int, int, int, int]:
    """Tight XY bounding box around bone across all selected slices."""
    bone_any = np.zeros(volume_hu.shape[1:], dtype=bool)
    for s in volume_hu:
        bone_any |= bone_mask(s)
    if not bone_any.any():
        h, w = volume_hu.shape[1:]
        return 0, h, 0, w
    ys, xs = np.where(bone_any)
    y0, y1 = max(int(ys.min()) - margin_px, 0), min(int(ys.max()) + margin_px + 1, volume_hu.shape[1])
    x0, x1 = max(int(xs.min()) - margin_px, 0), min(int(xs.max()) + margin_px + 1, volume_hu.shape[2])
    return y0, y1, x0, x1


# ----- Morphometry (used for both crop and pseudo-labels) ------------------

def compute_morphometry(volume_hu: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """Cheap pelvic morphometric proxies.

    All measured on the bone mask of the middle 60% of selected slices.
    Returns mm-scale features so they're comparable across resolutions.
    """
    sz, sy, sx = spacing
    Z = volume_hu.shape[0]
    if Z == 0:
        return {"pelvic_inlet_mm": 0.0, "iliac_flare": 0.0, "sacral_tilt": 0.0,
                "pelvic_width_mm": 0.0, "pelvic_depth_mm": 0.0}

    z_lo, z_hi = int(0.2 * Z), int(0.8 * Z)
    sub = volume_hu[z_lo:z_hi]

    widths_mm = []
    depths_mm = []
    for s in sub:
        bm = bone_mask(s)
        if not bm.any():
            continue
        ys, xs = np.where(bm)
        widths_mm.append((xs.max() - xs.min()) * sx)
        depths_mm.append((ys.max() - ys.min()) * sy)
    if not widths_mm:
        return {"pelvic_inlet_mm": 0.0, "iliac_flare": 0.0, "sacral_tilt": 0.0,
                "pelvic_width_mm": 0.0, "pelvic_depth_mm": 0.0}

    widths_mm = np.array(widths_mm)
    depths_mm = np.array(depths_mm)

    # Pelvic inlet ≈ the widest bone slice in the cranial half (iliac wings).
    half = len(widths_mm) // 2 or 1
    pelvic_inlet = float(widths_mm[:half].max())
    # Iliac flare proxy = max-width / min-width across the sub-volume.
    iliac_flare = float(widths_mm.max() / (widths_mm.min() + 1e-6))
    # Sacral tilt proxy = depth growth from top->bottom (positive = curves anteriorly).
    sacral_tilt = float((depths_mm[-half:].mean() - depths_mm[:half].mean()))

    return {
        "pelvic_inlet_mm": pelvic_inlet,
        "pelvic_width_mm": float(widths_mm.mean()),
        "pelvic_depth_mm": float(depths_mm.mean()),
        "iliac_flare": iliac_flare,
        "sacral_tilt": sacral_tilt,
    }


# ----- Resampling ---------------------------------------------------------

def resample_axial(volume_hu: np.ndarray, out_size: int) -> np.ndarray:
    """Bilinear-resample each axial slice to (out_size, out_size). Keep Z dim."""
    z, h, w = volume_hu.shape
    zoom_h = out_size / h
    zoom_w = out_size / w
    return zoom(volume_hu, (1.0, zoom_h, zoom_w), order=1, prefilter=False).astype(np.float32)


# ----- The main per-volume pipeline ---------------------------------------

def preprocess_volume(vol: Volume, cfg: dict) -> tuple[np.ndarray, tuple[float, float, float], dict] | None:
    """Run the full preprocessing chain. Returns (slices, spacing, morphometry) or None if skipped."""
    pp = cfg["preprocess"]
    arr = vol.array
    sz, sy, sx = vol.spacing

    # Z-range: keep pelvic slices.
    z0, z1 = pelvic_z_range(arr, min_span_mm=60.0, sz_mm=sz)
    arr = arr[z0:z1]
    if arr.shape[0] < pp["min_slices_per_series"]:
        return None

    # XY crop: bone bbox + margin -> square.
    margin_px = max(1, int(pp["margin_mm"] / max(sx, sy)))
    y0, y1, x0, x1 = pelvic_xy_bbox(arr, margin_px=margin_px)
    arr = arr[:, y0:y1, x0:x1]

    # Optional Z stride for speed.
    if pp["z_stride"] > 1:
        arr = arr[::pp["z_stride"]]

    # Pad to square so resize keeps aspect ratio.
    h, w = arr.shape[1:]
    if h != w:
        side = max(h, w)
        pad_h = (side - h) // 2
        pad_w = (side - w) // 2
        arr = np.pad(
            arr,
            ((0, 0), (pad_h, side - h - pad_h), (pad_w, side - w - pad_w)),
            mode="constant",
            constant_values=-1024.0,
        )

    morphometry = compute_morphometry(arr, (sz, sy, sx))

    arr = resample_axial(arr, pp["slice_size"])
    arr = window_hu(arr, pp["hu_min"], pp["hu_max"])

    # Convert to float16 to halve cache size; training will upcast.
    return arr.astype(np.float16), (sz * pp["z_stride"], sy, sx), morphometry


# ----- Cache building -----------------------------------------------------

def build_cache(cfg: dict, progress_cb=None) -> dict:
    """Walk every enabled dataset, preprocess, write per-volume .npz files.

    Returns a summary dict with counts and the manifest path.
    """
    from .progress import pbar, Spinner

    paths = cfg["paths"]
    pp = cfg["preprocess"]
    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    enable = pp.get("enable", {})
    max_per = pp.get("max_volumes_per_dataset", None)

    # First pass: enumerate all (dataset, path, uid) tuples so we have a total
    # for the progress bar. Scanning is fast (just directory traversal).
    with Spinner("Scanning input datasets"):
        tasks = list(walk_all(paths["data_root"], enable=enable, max_per_dataset=max_per))

    manifest = []
    n_ok = 0
    n_skip = 0
    n_err = 0

    bar = pbar(tasks, desc="Build cache", unit="vol")
    for ds_name, src_path, uid in bar:
        bar.set_postfix_str(f"{ds_name}/{uid[:24]}")
        out_dir = cache_dir / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_safe(uid)}.npz"
        if out_path.exists():
            n_ok += 1
            manifest.append({"dataset": ds_name, "uid": uid, "cache_path": str(out_path),
                             "source": str(src_path), "status": "cached"})
            if progress_cb: progress_cb(ds_name, uid, "cached")
            bar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, status="cached")
            continue
        try:
            vol = load_any(src_path)
            result = preprocess_volume(vol, cfg)
            if result is None:
                n_skip += 1
                if progress_cb: progress_cb(ds_name, uid, "skipped")
                bar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, status="skip")
                continue
            slices, spacing, morph = result
            np.savez_compressed(
                out_path,
                slices=slices,
                spacing=np.array(spacing, dtype=np.float32),
                morphometry=np.array(json.dumps(morph)),
                source=np.array(str(src_path)),
            )
            n_ok += 1
            manifest.append({"dataset": ds_name, "uid": uid, "cache_path": str(out_path),
                             "source": str(src_path), "status": "ok",
                             "n_slices": int(slices.shape[0]),
                             **{f"morph_{k}": v for k, v in morph.items()}})
            if progress_cb: progress_cb(ds_name, uid, "ok")
            bar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, status="ok")
        except Exception as e:
            n_err += 1
            manifest.append({"dataset": ds_name, "uid": uid, "cache_path": "",
                             "source": str(src_path), "status": f"error:{type(e).__name__}:{e}"})
            if progress_cb: progress_cb(ds_name, uid, f"error:{e}")
            bar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, status=f"err:{type(e).__name__}")

    bar.close()
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"ok": n_ok, "skipped": n_skip, "errors": n_err, "manifest": str(manifest_path)}


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:128]
