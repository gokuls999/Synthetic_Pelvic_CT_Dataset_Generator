"""Regenerate COMPARISON.png for all 350 patients — properly aligned + colored diff.

Fixes:
  1. X-aligns the original by finding the hip bone centroid (same as pipeline does)
     so bones are pixel-identical between original and edited.
  2. Colored heatmap for difference (blue→green→red = low→mid→high change).
  3. Picks the 3 slices with the highest deformation signal, not arbitrary Z/4 positions.
  4. Single clean layout: ORIGINAL | EDITED (PFD) | DIFFERENCE (colored).
"""
from _common import add_repo_to_path, load_config
add_repo_to_path()

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cfg    = load_config("configs/default.yaml")
HU_MIN = float(cfg["preprocess"]["hu_min"])
HU_MAX = float(cfg["preprocess"]["hu_max"])

# Soft-tissue display window
LO_U, HI_U = -150.0, 400.0

OUT_ROOT = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")


def _norm_to_hu(arr):
    return (arr.astype(np.float32) + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


def _win(hu_arr):
    """HU → uint8 soft-tissue window."""
    return ((np.clip(hu_arr, LO_U, HI_U) - LO_U)
            / (HI_U - LO_U) * 255).astype(np.uint8)


def _heatmap(diff_u8):
    """Grayscale diff → RGB heatmap (black→blue→cyan→green→yellow→red→white)."""
    d = diff_u8.astype(np.float32) / 255.0
    r = np.clip(4 * d - 1.5, 0, 1)
    g = np.clip(np.where(d < 0.5, 4 * d - 0.5, -4 * d + 3.5), 0, 1)
    b = np.clip(np.where(d < 0.25, 4 * d, -4 * d + 2.5), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return rgb


def _x_align_offset(vol_norm, z_mid):
    """Estimate x-centering offset from hip bone centroid at z_mid slice."""
    hu   = _norm_to_hu(vol_norm[z_mid])
    bone = hu > 250
    H, W = bone.shape
    # Use lateral thirds only (exclude spine at center)
    lat  = bone.copy()
    lat[:, W // 3: 2 * W // 3] = False
    x_idx = np.where(lat.any(axis=0))[0]
    if len(x_idx) < 4:
        return 0
    centroid_x = float((x_idx[0] + x_idx[-1]) / 2.0)
    offset = int(round(W / 2.0 - centroid_x))
    return max(-40, min(40, offset))


try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

total = fixed = skipped = 0

for pop_dir in ["Plain_175", "Hilly_175"]:
    for pdir in sorted((OUT_ROOT / pop_dir).iterdir()):
        if not pdir.is_dir():
            continue
        total += 1

        mf = pdir / "metadata.json"
        if not mf.exists():
            skipped += 1; continue
        meta = json.loads(mf.read_text())

        crop_z = meta.get("pelvic_crop_z")
        if not crop_z or len(crop_z) < 2:
            skipped += 1; continue

        cache_root = Path(cfg["paths"]["cache_dir"])
        npz_path   = cache_root / meta["source_dataset"] / f"{meta['source_uid']}.npz"
        if not npz_path.exists():
            skipped += 1; continue

        png_dir   = pdir / "PNG"
        png_files = sorted(png_dir.glob("*.png")) if png_dir.exists() else []
        if len(png_files) < 4:
            skipped += 1; continue

        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                vol_src = np.asarray(npz["slices"]).astype(np.float32)
        except Exception:
            skipped += 1; continue

        z_top = max(0, int(crop_z[0]))
        z_bot = min(vol_src.shape[0], int(crop_z[1]))
        vol_orig = vol_src[z_top:z_bot]          # normalized, pelvic crop
        Z_orig, H, W = vol_orig.shape

        # ── Align original to edited via hip bone centroid ─────────────────
        offset_x = _x_align_offset(vol_orig, Z_orig // 2)
        if offset_x != 0:
            vol_orig = np.roll(vol_orig, offset_x, axis=2)

        n_edit = len(png_files)

        # ── Compute per-slice difference to find best 3 slices ─────────────
        diffs = []
        for si in range(n_edit):
            edited = np.array(Image.open(png_files[si]).convert("L"), dtype=np.float32)
            z_src  = max(0, min(Z_orig - 1, int(si / n_edit * Z_orig)))
            orig   = _win(_norm_to_hu(vol_orig[z_src])).astype(np.float32)
            if orig.shape != edited.shape:
                orig = np.array(Image.fromarray(orig.astype(np.uint8)).resize(
                    (edited.shape[1], edited.shape[0]), Image.BILINEAR), dtype=np.float32)
            diffs.append(float(np.mean(np.abs(edited - orig))))

        # Pick 3 slices: spread across top-third, mid, bottom-third of volume
        # but weighted toward high-difference regions
        diffs_arr = np.array(diffs)
        thirds = np.array_split(np.arange(n_edit), 3)
        best_slices = [int(t[np.argmax(diffs_arr[t])]) for t in thirds]

        # ── Build comparison rows ──────────────────────────────────────────
        rows = []
        for si in best_slices:
            edited = np.array(Image.open(png_files[si]).convert("L"))
            Hb, Wb  = edited.shape
            z_src   = max(0, min(Z_orig - 1, int(si / n_edit * Z_orig)))
            orig_hu = _norm_to_hu(vol_orig[z_src])
            orig    = _win(orig_hu)
            if orig.shape != (Hb, Wb):
                orig = np.array(Image.fromarray(orig).resize((Wb, Hb), Image.BILINEAR))

            diff_raw = np.clip(
                np.abs(edited.astype(np.int32) - orig.astype(np.int32)),
                0, 255).astype(np.uint8)
            diff_rgb = _heatmap(diff_raw)

            # Convert orig and edited to RGB for stacking
            orig_rgb   = np.stack([orig,   orig,   orig],   axis=-1)
            edited_rgb = np.stack([edited, edited, edited], axis=-1)

            rows.append(np.concatenate([orig_rgb, edited_rgb, diff_rgb], axis=1))

        canvas = np.concatenate(rows, axis=0)
        img    = Image.fromarray(canvas, mode="RGB")
        draw   = ImageDraw.Draw(img)
        W_col  = rows[0].shape[1] // 3

        labels = ["ORIGINAL", "EDITED (PFD)", "DIFFERENCE"]
        colors = [(200, 200, 200), (180, 255, 180), (255, 180, 180)]
        for col, (label, color) in enumerate(zip(labels, colors)):
            draw.text((col * W_col + 5, 4), label, fill=color, font=font)

        # Add patient info bottom-left
        grade   = meta.get("pfd_grade", "?")
        pattern = meta.get("pfd_pattern", "?").replace("_", " ").upper()
        pop     = meta.get("population", "?").upper()
        info    = f"Grade {grade}  |  {pattern}  |  {pop}"
        draw.text((5, canvas.shape[0] - 18), info, fill=(180, 180, 255), font=font)

        img.save(pdir / "COMPARISON.png")
        fixed += 1
        if fixed % 25 == 0:
            print(f"  {fixed}/{total} done...")

print(f"\nDone.  fixed={fixed}  skipped={skipped}  total={total}")
