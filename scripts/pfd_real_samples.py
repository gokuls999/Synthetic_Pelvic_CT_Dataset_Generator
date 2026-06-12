# -*- coding: utf-8 -*-
"""Generate 3-4 sample patients using REAL CT source data + PFD deformation.

Approach (replaces CVAE generation entirely):
  1. Load original NIfTI source CT with BONE-INCLUSIVE HU window (-1000 to 1500)
  2. Preprocess through same pipeline (pelvic crop, 256×256) but with wide window
  3. Use cached TotalSegmentator masks (already aligned to this shape)
  4. Apply PFD deformation to pelvic floor organs only
  5. Denormalize → proper bone HU (700-1500 HU in DICOM)

Result: real CT anatomy + PFD deformation = proper 3D rendering in Slicer.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from _common import add_repo_to_path, load_config
add_repo_to_path()

import copy, json, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.io_loaders import load_nifti, load_any
from src.preprocessing import preprocess_volume, unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata
from src.pfd_segmentation import (
    segment_original_volume, rectum_subregion, mask_stats,
    pelvic_z_range_from_masks, PFD_ROI_SUBSET,
)
from src.pfd_deformation import (
    build_pattern_graded,
    apply_confined_deformation,
    apply_levator_ani_deformation,
)

# ── Config ───────────────────────────────────────────────────────────────────
cfg       = load_config("configs/default.yaml")
cache_root= Path(cfg["paths"]["cache_dir"])

# Wide HU window — preserves bone (700-1500 HU) instead of clipping at 500 HU
HU_MIN_WIDE = -1000.0
HU_MAX_WIDE =  1500.0

cfg_wide = copy.deepcopy(cfg)
cfg_wide["preprocess"]["hu_min"] = HU_MIN_WIDE
cfg_wide["preprocess"]["hu_max"] = HU_MAX_WIDE

OUT_ROOT  = Path("synthetic_dataset/PFD_Real_Samples")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LO_U, HI_U = -150.0, 400.0   # soft-tissue display window for COMPARISON

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()


# ── Pick 4 sample patients ───────────────────────────────────────────────────
# Select from existing dataset: variety of patterns, grades, populations
SAMPLES = []
old_root = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")
want = [
    ("plain", "combined_pfd",    1),
    ("plain", "cystocele",       2),
    ("hilly", "rectocele",       3),
    ("hilly", "combined_pfd",    2),
]
for pop, pat, grd in want:
    pop_dir = "Plain_175" if pop == "plain" else "Hilly_175"
    for pdir in (old_root / pop_dir).iterdir():
        if not pdir.is_dir(): continue
        mf = pdir / "metadata.json"
        if not mf.exists(): continue
        m = json.loads(mf.read_text())
        if (m.get("population") == pop and
                m.get("pfd_pattern") == pat and
                m.get("pfd_grade") == grd):
            src = Path(m.get("source_path", "")) if m.get("source_path") else None
            # find src_path from npz
            npz = cache_root / m["source_dataset"] / f"{m['source_uid']}.npz"
            if npz.exists():
                with np.load(npz, allow_pickle=True) as n:
                    sp = Path(str(n["source"]))
                if sp.exists():
                    SAMPLES.append({
                        "population": pop,
                        "pattern":    pat,
                        "grade":      grd,
                        "dataset":    m["source_dataset"],
                        "uid":        m["source_uid"],
                        "src_path":   sp,
                    })
                    break

print(f"Selected {len(SAMPLES)} samples:")
for s in SAMPLES:
    print(f"  {s['population'].upper():6s} {s['pattern']:20s} G{s['grade']}  {s['src_path'].name}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _win(arr_norm):
    hu = (arr_norm.astype(np.float32) + 1.0) / 2.0 * (HU_MAX_WIDE - HU_MIN_WIDE) + HU_MIN_WIDE
    return ((np.clip(hu, LO_U, HI_U) - LO_U) / (HI_U - LO_U) * 255).astype(np.uint8)

def _heatmap(diff_u8):
    d = diff_u8.astype(np.float32) / 255.0
    r = np.clip(4*d - 1.5, 0, 1)
    g = np.clip(np.where(d < 0.5, 4*d - 0.5, -4*d + 3.5), 0, 1)
    b = np.clip(np.where(d < 0.25, 4*d, -4*d + 2.5), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def _save_comparison(vol_orig, vol_edited, out_path, grade, pattern, population):
    Z = vol_orig.shape[0]
    diffs = np.array([float(np.mean(np.abs(
        vol_edited[z].astype(np.float32) - vol_orig[z].astype(np.float32))))
        for z in range(Z)])
    min_gap = max(4, Z // 6)
    best = []
    for idx in np.argsort(diffs)[::-1]:
        if all(abs(int(idx) - b) >= min_gap for b in best):
            best.append(int(idx))
        if len(best) == 3: break
    best.sort()

    rows = []
    for z in best:
        o = _win(vol_orig[z]); e = _win(vol_edited[z])
        diff = np.clip(np.abs(e.astype(np.int32) - o.astype(np.int32)) * 2, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([
            np.stack([o,o,o], axis=-1),
            np.stack([e,e,e], axis=-1),
            _heatmap(diff)
        ], axis=1))

    canvas = np.concatenate(rows, axis=0)
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    W_col = rows[0].shape[1] // 3
    for col, (lbl, clr) in enumerate(zip(
            ["ORIGINAL", "EDITED (PFD)", "DIFFERENCE"],
            [(200,200,200), (180,255,180), (255,180,180)])):
        draw.text((col * W_col + 5, 4), lbl, fill=clr, font=font)
    info = f"Grade {grade}  |  {pattern.replace('_',' ').upper()}  |  {population.upper()}"
    draw.text((5, canvas.shape[0] - 18), info, fill=(180,180,255), font=font)
    img.save(out_path)


_PATTERN_ORGANS = {
    "combined_pfd":    ["urinary_bladder","rectum","uterus"],
    "cystocele":       ["urinary_bladder"],
    "rectocele":       ["rectum"],
    "uterine_prolapse":["urinary_bladder","uterus"],
}

# ── Process each sample ───────────────────────────────────────────────────────
for i, s in enumerate(SAMPLES):
    pid  = f"REAL_{i+1:02d}_{s['population'].upper()}_{s['pattern'].upper()}_G{s['grade']}"
    pdir = OUT_ROOT / pid
    pdir.mkdir(exist_ok=True)
    # Skip if already completed
    if (pdir / "COMPARISON.png").exists() and list((pdir / "DICOM").glob("*.dcm")):
        print(f"\n[{i+1}/{len(SAMPLES)}] {pid}  -- already done, skipping")
        continue
    print(f"\n[{i+1}/{len(SAMPLES)}] {pid}")

    # ── Load original CT (NIfTI or DICOM series) with wide HU window ─────
    print("  Loading original CT...")
    vol_obj = load_any(s["src_path"])
    result  = preprocess_volume(vol_obj, cfg_wide)
    if result is None:
        print("  SKIP: preprocess returned None"); continue
    vol_norm, spacing, _ = result
    vol_norm = vol_norm.astype(np.float32)
    print(f"  Shape: {vol_norm.shape}  spacing: {spacing}")

    # ── Load cached TotalSeg masks ─────────────────────────────────────────
    print("  Loading cached masks...")
    masks, _ = segment_original_volume(
        source_path=s["src_path"], dataset=s["dataset"], uid=s["uid"],
        cfg=cfg, cache_root=cache_root, roi_subset=PFD_ROI_SUBSET)

    # Align mask Z to vol_norm Z (preprocess may crop differently)
    for k in list(masks.keys()):
        if masks[k].shape[0] != vol_norm.shape[0]:
            from scipy.ndimage import zoom as ndi_zoom
            ratio = vol_norm.shape[0] / masks[k].shape[0]
            masks[k] = ndi_zoom(masks[k].astype(np.float32),
                                 (ratio, 1.0, 1.0), order=0).astype(bool)

    # ── Pelvic floor Z crop ────────────────────────────────────────────────
    Z_orig = vol_norm.shape[0]
    z_top, z_bot = pelvic_z_range_from_masks(masks, margin_voxels=6, fallback=(0, Z_orig))
    vol_norm = vol_norm[z_top:z_bot]
    masks    = {k: m[z_top:z_bot] for k, m in masks.items()}
    print(f"  Pelvic crop: z={z_top}-{z_bot}  ({vol_norm.shape[0]} slices)")

    # ── X-center ──────────────────────────────────────────────────────────
    W = vol_norm.shape[2]
    _hl = mask_stats(masks.get("hip_left"),  "hip_left")  if "hip_left"  in masks else None
    _hr = mask_stats(masks.get("hip_right"), "hip_right") if "hip_right" in masks else None
    offset_x = 0
    if _hl and _hr:
        midline_x = (_hl.center[2] + _hr.center[2]) / 2.0
        offset_x  = max(-32, min(32, int(round(W / 2.0 - midline_x))))
        if abs(offset_x) >= 2:
            vol_norm = np.roll(vol_norm, offset_x, axis=2)
            masks    = {k: np.roll(m, offset_x, axis=2) for k, m in masks.items()}

    # ── Organ stats ────────────────────────────────────────────────────────
    if "colon" in masks and masks["colon"].any():
        masks["rectum"] = rectum_subregion(masks["colon"], frac=0.66)
    stats = {n: st for n, m in masks.items()
             if (st := mask_stats(m, n)) is not None}

    needed = {"combined_pfd":  ["urinary_bladder","rectum"],
              "cystocele":     ["urinary_bladder"],
              "rectocele":     ["rectum"],
              "uterine_prolapse":["urinary_bladder","rectum"]}[s["pattern"]]
    missing = [n for n in needed if n not in stats or stats[n].voxels < 200]
    if missing:
        print(f"  SKIP: missing {missing}"); continue

    vol_orig_for_compare = vol_norm.copy()

    # ── PFD deformation ────────────────────────────────────────────────────
    print(f"  Applying {s['pattern']} G{s['grade']} deformation...")
    p_obj  = build_pattern_graded(s["pattern"], stats, grade=s["grade"],
                                   population=s["population"])
    t_orgs = _PATTERN_ORGANS.get(s["pattern"], ["urinary_bladder","rectum"])

    vol_edited = apply_confined_deformation(
        volume=vol_norm, blobs=p_obj.blobs, organ_masks=masks,
        target_organs=t_orgs, order=3, cval=-1.0,
        dilation_voxels=8, blur_sigma=4.0)
    vol_edited = apply_levator_ani_deformation(
        volume=vol_edited, masks=masks, grade=s["grade"],
        pixel_spacing_mm=float(spacing[1]),
        slice_spacing_mm=float(spacing[0]), blur_sigma=4.0)

    # ── Back to HU (WIDE window → proper bone!) ───────────────────────────
    vol_hu = np.clip(
        unwindow(vol_edited, HU_MIN_WIDE, HU_MAX_WIDE), -1024.0, 3071.0
    ).astype(np.int16)
    print(f"  HU range: {vol_hu.min()} to {vol_hu.max()}")

    # ── Save DICOM ─────────────────────────────────────────────────────────
    print("  Writing DICOM...")
    vol_obj2 = SyntheticVolume(
        pixels_hu=vol_hu, spacing=spacing,
        region=s["population"],
        region_id=0 if s["population"]=="plain" else 1,
        patient_id=pid, seed=i)
    write_dicom_series(vol_obj2, pdir / "DICOM", cfg)
    write_png_jpg_metadata(vol_obj2, pdir / "PNG", pdir / "JPG",
                            pdir / "metadata.json", cfg)

    # ── Save COMPARISON ────────────────────────────────────────────────────
    print("  Saving COMPARISON.png...")
    _save_comparison(vol_orig_for_compare, vol_edited,
                      pdir / "COMPARISON.png",
                      s["grade"], s["pattern"], s["population"])

    # ── metadata ──────────────────────────────────────────────────────────
    meta = {
        "patient_id": pid, "population": s["population"],
        "pfd_pattern": s["pattern"], "pfd_grade": s["grade"],
        "source_dataset": s["dataset"], "source_uid": s["uid"],
        "hu_window": [HU_MIN_WIDE, HU_MAX_WIDE],
        "approach": "real_ct_wide_hu",
    }
    (pdir / "metadata_real.json").write_text(json.dumps(meta, indent=2))
    print(f"  Done -> {pdir.name}")

print(f"\nAll samples saved to: {OUT_ROOT}")
print("Load DICOM/ folder in 3D Slicer to verify bone rendering.")
