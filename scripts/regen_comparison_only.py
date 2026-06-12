"""Regenerate COMPARISON.png for all 350 patients correctly.

Re-runs only the minimal pipeline per patient:
  load npz → z-crop → x-center → augment → deform → comparison strip
Uses cached TotalSegmentator masks (fast, no GPU re-run).
Skips DICOM / PDF / segmentation export.
"""
from _common import add_repo_to_path, load_config
add_repo_to_path()

import json, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.pfd_segmentation import (
    segment_original_volume, rectum_subregion, mask_stats,
    pelvic_z_range_from_masks, PFD_ROI_SUBSET,
)
from src.pfd_deformation import (
    build_pattern_graded, apply_confined_deformation,
    apply_levator_ani_deformation,
)

cfg      = load_config("configs/default.yaml")
HU_MIN   = float(cfg["preprocess"]["hu_min"])
HU_MAX   = float(cfg["preprocess"]["hu_max"])
LO_U, HI_U = -150.0, 400.0        # soft-tissue display window

OUT_ROOT = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")

_PATTERN_ORGANS = {
    "combined_pfd":     ["urinary_bladder", "rectum", "uterus"],
    "cystocele":        ["urinary_bladder"],
    "rectocele":        ["rectum"],
    "uterine_prolapse": ["urinary_bladder", "uterus"],
}

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()


def _win(arr_norm):
    """Normalized [-1,1] → HU → soft-tissue window → uint8."""
    hu = (arr_norm.astype(np.float32) + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN
    return ((np.clip(hu, LO_U, HI_U) - LO_U) / (HI_U - LO_U) * 255).astype(np.uint8)


def _heatmap(diff_u8):
    """Grayscale diff → RGB heatmap (black→blue→cyan→green→yellow→red)."""
    d = diff_u8.astype(np.float32) / 255.0
    r = np.clip(4 * d - 1.5, 0, 1)
    g = np.clip(np.where(d < 0.5, 4 * d - 0.5, -4 * d + 3.5), 0, 1)
    b = np.clip(np.where(d < 0.25, 4 * d, -4 * d + 2.5), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _augment(vol, patient_num):
    rng   = np.random.default_rng(patient_num * 999983 + 7)
    vol   = vol.astype(np.float32)
    sigma = float(rng.uniform(0.003, 0.010))
    vol  += rng.standard_normal(vol.shape).astype(np.float32) * sigma
    vol  += float(rng.uniform(-0.015, 0.015))
    return np.clip(vol, -1.0, 1.0).astype(np.float32)


def _save_comparison(vol_orig, vol_edited, pdir, grade, pattern, population):
    Z = vol_orig.shape[0]

    # Compute per-slice mean absolute diff to find the best 3 slices
    diffs = np.array([
        float(np.mean(np.abs(vol_edited[z].astype(np.float32)
                             - vol_orig[z].astype(np.float32))))
        for z in range(Z)
    ])
    # Pick 3 globally highest-diff slices, each at least Z//6 apart
    min_gap = max(4, Z // 6)
    order   = np.argsort(diffs)[::-1]
    best    = []
    for idx in order:
        if all(abs(int(idx) - b) >= min_gap for b in best):
            best.append(int(idx))
        if len(best) == 3:
            break
    best.sort()  # top → bottom for natural anatomical reading order

    rows = []
    for z in best:
        orig_u8 = _win(vol_orig[z])
        edit_u8 = _win(vol_edited[z])
        diff_raw = np.clip(
            np.abs(edit_u8.astype(np.int32) - orig_u8.astype(np.int32)) * 2,
            0, 255).astype(np.uint8)
        diff_rgb = _heatmap(diff_raw)
        orig_rgb = np.stack([orig_u8, orig_u8, orig_u8], axis=-1)
        edit_rgb = np.stack([edit_u8, edit_u8, edit_u8], axis=-1)
        rows.append(np.concatenate([orig_rgb, edit_rgb, diff_rgb], axis=1))

    canvas = np.concatenate(rows, axis=0)
    img    = Image.fromarray(canvas, mode="RGB")
    draw   = ImageDraw.Draw(img)
    W_col  = rows[0].shape[1] // 3
    labels = ["ORIGINAL", "EDITED (PFD)", "DIFFERENCE"]
    colors = [(200, 200, 200), (180, 255, 180), (255, 180, 180)]
    for col, (label, color) in enumerate(zip(labels, colors)):
        draw.text((col * W_col + 5, 4), label, fill=color, font=font)

    pat_str = pattern.replace("_", " ").upper()
    info    = f"Grade {grade}  |  {pat_str}  |  {population.upper()}"
    draw.text((5, canvas.shape[0] - 18), info, fill=(180, 180, 255), font=font)
    img.save(pdir / "COMPARISON.png")


total = ok = skipped = failed = 0

for pop_dir in ["Plain_175", "Hilly_175"]:
    for pdir in sorted((OUT_ROOT / pop_dir).iterdir()):
        if not pdir.is_dir():
            continue
        total += 1

        mf = pdir / "metadata.json"
        if not mf.exists():
            skipped += 1; continue
        meta     = json.loads(mf.read_text())
        orig_pid = meta.get("patient_id", pdir.name)      # original patient_id (pre-rename)
        m        = re.search(r"PFD_(\d+)_", orig_pid)
        pat_num  = int(m.group(1)) if m else 1

        pattern    = meta["pfd_pattern"]
        grade      = meta["pfd_grade"]
        population = meta["population"]
        dataset    = meta["source_dataset"]
        uid        = meta["source_uid"]
        crop_z     = meta.get("pelvic_crop_z", [])

        cache_root = Path(cfg["paths"]["cache_dir"])
        npz_path   = cache_root / dataset / f"{uid}.npz"
        if not npz_path.exists():
            skipped += 1; continue

        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                vol_norm = np.asarray(npz["slices"]).astype(np.float32)
                src_path = Path(str(npz["source"]))
                spacing  = tuple(float(x) for x in np.asarray(npz["spacing"]).tolist())
            Z_orig = vol_norm.shape[0]

            # ── Z-crop ──────────────────────────────────────────────────
            if crop_z and len(crop_z) >= 2:
                z_top = max(0, int(crop_z[0]))
                z_bot = min(Z_orig, int(crop_z[1]))
            else:
                masks_tmp, _ = segment_original_volume(
                    source_path=src_path, dataset=dataset, uid=uid,
                    cfg=cfg, cache_root=cache_root, roi_subset=PFD_ROI_SUBSET)
                z_top, z_bot = pelvic_z_range_from_masks(
                    masks_tmp, margin_voxels=6, fallback=(0, Z_orig))
            vol_norm = vol_norm[z_top:z_bot]

            # ── Load TotalSeg masks (from cache — fast) ──────────────────
            masks, _ = segment_original_volume(
                source_path=src_path, dataset=dataset, uid=uid,
                cfg=cfg, cache_root=cache_root, roi_subset=PFD_ROI_SUBSET)
            masks = {k: m[z_top:z_bot] for k, m in masks.items()}

            # ── X-center ─────────────────────────────────────────────────
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

            # ── Augment + save original for compare ──────────────────────
            vol_orig_for_compare = vol_norm.copy()
            vol_norm = _augment(vol_norm, pat_num)

            # ── Organ stats ───────────────────────────────────────────────
            if "colon" in masks and masks["colon"].any():
                masks["rectum"] = rectum_subregion(masks["colon"], frac=0.66)
            stats = {n: s for n, m in masks.items()
                     if (s := mask_stats(m, n)) is not None}

            needed  = {"combined_pfd": ["urinary_bladder","rectum"],
                       "cystocele":    ["urinary_bladder"],
                       "rectocele":    ["rectum"],
                       "uterine_prolapse": ["urinary_bladder","rectum"]}[pattern]
            missing = [n for n in needed if n not in stats or stats[n].voxels < 200]
            if missing:
                skipped += 1; continue

            # ── Deformation ───────────────────────────────────────────────
            p_obj          = build_pattern_graded(pattern, stats, grade=grade,
                                                   population=population)
            target_organs  = _PATTERN_ORGANS.get(pattern, ["urinary_bladder","rectum"])
            vol_edited     = apply_confined_deformation(
                volume=vol_norm, blobs=p_obj.blobs, organ_masks=masks,
                target_organs=target_organs, order=3, cval=-1.0,
                dilation_voxels=8, blur_sigma=4.0)
            vol_edited     = apply_levator_ani_deformation(
                volume=vol_edited, masks=masks, grade=grade,
                pixel_spacing_mm=float(spacing[1]),
                slice_spacing_mm=float(spacing[0]), blur_sigma=4.0)

            # ── Save comparison ────────────────────────────────────────────
            _save_comparison(vol_orig_for_compare, vol_edited,
                             pdir, grade, pattern, population)
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{total} done...")

        except Exception as e:
            failed += 1
            print(f"  FAIL {pdir.name}: {e}")

print(f"\nDone.  ok={ok}  skipped={skipped}  failed={failed}  total={total}")
