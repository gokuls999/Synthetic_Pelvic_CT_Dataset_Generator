# -*- coding: utf-8 -*-
"""pfd_real_v1.py — 350-patient real CT PFD dataset.

175 Plain (gynecoid) + 175 Hilly (android)
4 patterns x 3 grades, balanced across patients
Wide HU window (-1000 to 1500) — proper bone in 3D Slicer
Resumable: skips completed patient folders
Output: synthetic_dataset/PFD_Real_Dataset_350/
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse, copy, json, random, traceback, shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom

_ap = argparse.ArgumentParser()
_ap.add_argument("--limit", type=int, default=None, help="Stop after generating this many new patients")
_args = _ap.parse_args()
LIMIT = _args.limit
from PIL import Image, ImageDraw, ImageFont

from src.io_loaders import load_any
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

# ── Config ────────────────────────────────────────────────────────────────────
cfg        = load_config("configs/default.yaml")
cache_root = Path(cfg["paths"]["cache_dir"])

HU_MIN = -1000.0
HU_MAX =  1500.0
cfg_wide = copy.deepcopy(cfg)
cfg_wide["preprocess"]["hu_min"] = HU_MIN
cfg_wide["preprocess"]["hu_max"] = HU_MAX

OUT_ROOT  = Path("synthetic_dataset/PFD_Real_Dataset_350")
PLAIN_DIR = OUT_ROOT / "Plain_175"
HILLY_DIR = OUT_ROOT / "Hilly_175"
for d in [PLAIN_DIR, HILLY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_PLAIN = 175
N_HILLY = 175
LO_U, HI_U = -150.0, 400.0   # soft-tissue display window for COMPARISON

PATTERNS = ["combined_pfd", "cystocele", "rectocele", "uterine_prolapse"]
GRADES   = [1, 2, 3]

NEEDED = {
    "combined_pfd":     ["urinary_bladder", "rectum"],
    "cystocele":        ["urinary_bladder"],
    "rectocele":        ["rectum"],
    "uterine_prolapse": ["urinary_bladder", "rectum"],
}
DEFORM_ORGANS = {
    "combined_pfd":     ["urinary_bladder", "rectum", "uterus"],
    "cystocele":        ["urinary_bladder"],
    "rectocele":        ["rectum"],
    "uterine_prolapse": ["urinary_bladder", "uterus"],
}

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _win(arr_norm):
    hu = (arr_norm.astype(np.float32) + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN
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
            np.stack([o, o, o], axis=-1),
            np.stack([e, e, e], axis=-1),
            _heatmap(diff),
        ], axis=1))

    canvas = np.concatenate(rows, axis=0)
    img  = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    W_col = rows[0].shape[1] // 3
    for col, (lbl, clr) in enumerate(zip(
            ["ORIGINAL", "EDITED (PFD)", "DIFFERENCE"],
            [(200, 200, 200), (180, 255, 180), (255, 180, 180)])):
        draw.text((col * W_col + 5, 4), lbl, fill=clr, font=font)
    info = f"Grade {grade}  |  {pattern.replace('_', ' ').upper()}  |  {population.upper()}"
    draw.text((5, canvas.shape[0] - 18), info, fill=(180, 180, 255), font=font)
    img.save(out_path)


# ── Single-instance lock (prevents duplicate runs from autostart / manual) ─────
import msvcrt as _msvcrt, atexit as _atexit
_lock_path = Path("logs/.pfd_real_v1.lock")
_lock_path.parent.mkdir(parents=True, exist_ok=True)
try:
    _lock_fh = open(_lock_path, 'w')
    _msvcrt.locking(_lock_fh.fileno(), _msvcrt.LK_NBLCK, 1)
    _atexit.register(lambda: [_lock_fh.close(), _lock_path.unlink(missing_ok=True)])
except (IOError, OSError):
    print("Another pfd_real_v1 instance is already running. Exiting.")
    sys.exit(0)

# ── Step 1: Collect & quality-filter source CTs ───────────────────────────────
print("Scanning source cache for quality sources...")
good_sources = []

for ds_dir in sorted(cache_root.iterdir()):
    if not ds_dir.is_dir() or ds_dir.name == "masks":
        continue
    for npz_file in sorted(ds_dir.glob("*.npz")):
        uid = npz_file.stem
        try:
            with np.load(npz_file, allow_pickle=True) as nf:
                sp = Path(str(nf["source"]))
            if not sp.exists():
                continue
            if sp.is_dir():
                dcms = list(sp.glob("*.dcm"))
                n_slices = len(dcms)
                if n_slices < 80:
                    continue
                ds0 = pydicom.dcmread(str(sorted(dcms)[0]), stop_before_pixels=True)
                sz  = float(getattr(ds0, "SliceThickness", 99.0) or 99.0)
            else:
                img = nib.load(str(sp))
                n_slices = img.shape[2]
                sz = float(img.header.get_zooms()[2])
            if n_slices < 80 or sz > 3.0 or n_slices > 400:
                continue  # reject full-body CTs (>400 slices = likely chest-to-pelvis)

            # Reject sources with metal implants (HU > 2000 = metal prostheses/hardware)
            try:
                if sp.is_dir():
                    mid_dcm = sorted(dcms)[len(dcms) // 2]
                    chk = pydicom.dcmread(str(mid_dcm))
                    arr = chk.pixel_array.astype(np.int32)
                    slope = float(getattr(chk, 'RescaleSlope', 1) or 1)
                    intercept = float(getattr(chk, 'RescaleIntercept', -1024) or -1024)
                    hu_arr = arr * slope + intercept
                else:
                    mid_z = nib.load(str(sp)).get_fdata()[:, :, nib.load(str(sp)).shape[2] // 2]
                    hu_arr = mid_z
                if (hu_arr > 2000).sum() > 50:  # >50 metal voxels in mid-slice
                    continue
            except Exception:
                pass

            good_sources.append({
                "dataset": ds_dir.name, "uid": uid,
                "src_path": sp, "slices": n_slices, "sz_mm": sz,
            })
        except Exception:
            pass

print(f"  Good sources: {len(good_sources)}")
if len(good_sources) == 0:
    print("ERROR: no sources found"); sys.exit(1)


# ── Step 2: Pre-plan 350 balanced assignments ─────────────────────────────────
random.seed(42)

# Balanced (pattern, grade) pool: 350 slots, ~29 per combo
combos = [(p, g) for p in PATTERNS for g in GRADES]   # 12 combos
pool   = (combos * ((350 // len(combos)) + 2))[:350]
random.shuffle(pool)

# Split into plain (even slots) and hilly (odd slots)
plain_pool = [pool[i] for i in range(0,   350, 2)][:N_PLAIN]  # 175 plain
hilly_pool = [pool[i] for i in range(1,   350, 2)][:N_HILLY]  # 175 hilly

# Map sources round-robin — shuffle sources first for variety
shuffled_src = good_sources.copy()
random.shuffle(shuffled_src)

plain_plan = []
for n, (pat, grd) in enumerate(plain_pool):
    src = shuffled_src[n % len(shuffled_src)]
    plain_plan.append({"n": n+1, "population": "plain", "pattern": pat, "grade": grd, **src})

hilly_plan = []
for n, (pat, grd) in enumerate(hilly_pool):
    src = shuffled_src[(n + N_PLAIN) % len(shuffled_src)]
    hilly_plan.append({"n": n+1, "population": "hilly", "pattern": pat, "grade": grd, **src})

print(f"  Planned: {len(plain_plan)} plain + {len(hilly_plan)} hilly = {len(plain_plan)+len(hilly_plan)} total")


# ── Step 3: Count already-completed patients ──────────────────────────────────
def _done_count(pop_dir):
    if not pop_dir.exists():
        return set()
    done = set()
    for d in pop_dir.iterdir():
        if d.is_dir() and (d / "COMPARISON.png").exists() and list((d / "DICOM").glob("*.dcm")):
            done.add(d.name)
    return done

plain_done_names = _done_count(PLAIN_DIR)
hilly_done_names = _done_count(HILLY_DIR)
print(f"  Already done: {len(plain_done_names)} plain, {len(hilly_done_names)} hilly")


# ── Inline QA ────────────────────────────────────────────────────────────────
def _qa_check(pdir):
    """Check a generated patient folder. Returns (passed, reason)."""
    dcms = list((pdir / "DICOM").glob("*.dcm"))
    n = len(dcms)
    if n < 50:
        return False, f"too_few_slices({n})"
    if n > 280:
        return False, f"too_many_slices({n})"

    # Sample 3 slices: 25%, 50%, 75%
    dcms_sorted = sorted(dcms, key=lambda f: int(
        pydicom.dcmread(str(f), stop_before_pixels=True, force=True).get("InstanceNumber", 0)))
    bone_v, air_v, tis_v, metal_v = [], [], [], []
    for frac in [0.25, 0.5, 0.75]:
        ds = pydicom.dcmread(str(dcms_sorted[int(n * frac)]), force=True)
        arr = ds.pixel_array.astype(np.int32) - 1024
        total = arr.size
        bone_v.append((arr > 300).sum() / total * 100)
        air_v.append((arr < -500).sum() / total * 100)
        tis_v.append(((arr >= -100) & (arr <= 200)).sum() / total * 100)
        metal_v.append((arr > 2000).sum())

    avg_bone  = float(np.mean(bone_v))
    avg_air   = float(np.mean(air_v))
    avg_tis   = float(np.mean(tis_v))
    max_metal = max(metal_v)

    if max_metal > 100:
        return False, f"metal_implant({max_metal}px)"
    if avg_bone < 0.3:
        return False, f"no_bone({avg_bone:.1f}%)"
    if avg_air > 78:
        return False, f"excess_air({avg_air:.0f}%)"
    if avg_tis < 10:
        return False, f"low_tissue({avg_tis:.0f}%)"

    return True, f"bone={avg_bone:.1f}% air={avg_air:.0f}% tis={avg_tis:.0f}%"


# ── Step 4: Process ───────────────────────────────────────────────────────────
total_done = skipped = failed = qa_rejected = 0

def _process(plan, pop_dir, done_names):
    global total_done, skipped, failed, qa_rejected
    from scipy.ndimage import zoom as ndi_zoom

    # Build rotating source list for retries
    src_pool = good_sources.copy()
    random.shuffle(src_pool)

    for asn in plan:
        n   = asn["n"]
        pop = asn["population"]
        pat = asn["pattern"]
        grd = asn["grade"]
        pid = f"REAL_{n:03d}_{pop.upper()}_{pat.upper()}_G{grd}"
        pdir = pop_dir / pid

        # Limit: stop after N new patients
        if LIMIT is not None and total_done >= LIMIT:
            break

        # Resume: skip completed
        if pid in done_names:
            skipped += 1
            continue

        # Try sources until one passes QA
        src_attempts = [asn] + [
            {**s, "population": pop, "pattern": pat, "grade": grd, "n": n}
            for s in src_pool if s["uid"] != asn["uid"]
        ]

        accepted = False
        for attempt_i, src in enumerate(src_attempts):
            if attempt_i > 0:
                print(f"  QA retry {attempt_i} with source {src['uid']} ({src['dataset']})", flush=True)

            shutil.rmtree(pdir, ignore_errors=True)
            pdir.mkdir(exist_ok=True)
            total_so_far = len(plain_done_names) + len(hilly_done_names)
            if attempt_i == 0:
                print(f"\n[{total_so_far}/{N_PLAIN+N_HILLY}] {pid}", flush=True)

            try:
                # Load CT
                print("  Loading CT...", flush=True)
                vol_obj = load_any(src["src_path"])
                result  = preprocess_volume(vol_obj, cfg_wide)
                if result is None:
                    print("  SKIP: preprocess None"); continue
                vol_norm, spacing, _ = result
                vol_norm = vol_norm.astype(np.float32)
                if vol_norm.shape[0] < 30:
                    print(f"  SKIP: {vol_norm.shape[0]} slices post-preprocess"); continue

                # Cached masks
                print("  Loading masks...", flush=True)
                masks, _ = segment_original_volume(
                    source_path=src["src_path"], dataset=src["dataset"],
                    uid=src["uid"], cfg=cfg, cache_root=cache_root,
                    roi_subset=PFD_ROI_SUBSET)

                # Align mask Z
                for k in list(masks.keys()):
                    if masks[k].shape[0] != vol_norm.shape[0]:
                        ratio = vol_norm.shape[0] / masks[k].shape[0]
                        masks[k] = ndi_zoom(masks[k].astype(np.float32),
                                            (ratio, 1.0, 1.0), order=0).astype(bool)

                # Pelvic Z crop
                Z_orig = vol_norm.shape[0]
                z_top, z_bot = pelvic_z_range_from_masks(
                    masks, margin_voxels=6, fallback=(0, Z_orig))
                vol_norm = vol_norm[z_top:z_bot]
                masks    = {k: m[z_top:z_bot] for k, m in masks.items()}
                if vol_norm.shape[0] < 20:
                    print(f"  SKIP: {vol_norm.shape[0]} slices after crop"); continue
                if vol_norm.shape[0] > 280:
                    print(f"  SKIP: crop failed ({vol_norm.shape[0]} slices)"); continue

                # X-center
                W = vol_norm.shape[2]
                _hl = mask_stats(masks.get("hip_left"),  "hip_left")  if "hip_left"  in masks else None
                _hr = mask_stats(masks.get("hip_right"), "hip_right") if "hip_right" in masks else None
                if _hl and _hr:
                    midline_x = (_hl.center[2] + _hr.center[2]) / 2.0
                    offset_x  = max(-32, min(32, int(round(W / 2.0 - midline_x))))
                    if abs(offset_x) >= 2:
                        vol_norm = np.roll(vol_norm, offset_x, axis=2)
                        masks    = {k: np.roll(m, offset_x, axis=2) for k, m in masks.items()}

                # Organ stats
                if "colon" in masks and masks["colon"].any():
                    masks["rectum"] = rectum_subregion(masks["colon"], frac=0.66)
                stats = {nm: st for nm, m in masks.items()
                         if (st := mask_stats(m, nm)) is not None}

                missing = [nm for nm in NEEDED[pat]
                           if nm not in stats or stats[nm].voxels < 200]
                if missing:
                    print(f"  SKIP: missing {missing}"); continue

                vol_orig_for_compare = vol_norm.copy()

                # PFD deformation
                print(f"  Deforming {pat} G{grd} ({pop})...", flush=True)
                p_obj = build_pattern_graded(pat, stats, grade=grd, population=pop)
                vol_edited = apply_confined_deformation(
                    volume=vol_norm, blobs=p_obj.blobs, organ_masks=masks,
                    target_organs=DEFORM_ORGANS[pat], order=3, cval=-1.0,
                    dilation_voxels=8, blur_sigma=4.0)
                vol_edited = apply_levator_ani_deformation(
                    volume=vol_edited, masks=masks, grade=grd,
                    pixel_spacing_mm=float(spacing[1]),
                    slice_spacing_mm=float(spacing[0]), blur_sigma=4.0)

                # Back to HU
                vol_hu = np.clip(
                    unwindow(vol_edited, HU_MIN, HU_MAX), -1024.0, 3071.0
                ).astype(np.int16)

                # DICOM
                print("  Writing DICOM...", flush=True)
                vol_obj2 = SyntheticVolume(
                    pixels_hu=vol_hu, spacing=spacing,
                    region=pop, region_id=0 if pop == "plain" else 1,
                    patient_id=pid, seed=n)
                write_dicom_series(vol_obj2, pdir / "DICOM", cfg)
                write_png_jpg_metadata(vol_obj2, pdir / "PNG", pdir / "JPG",
                                       pdir / "metadata.json", cfg)

                # COMPARISON
                print("  Saving COMPARISON.png...", flush=True)
                _save_comparison(vol_orig_for_compare, vol_edited,
                                 pdir / "COMPARISON.png", grd, pat, pop)

                # ── Inline QA ────────────────────────────────────────────────
                print("  Running QA...", flush=True)
                qa_pass, qa_reason = _qa_check(pdir)
                if not qa_pass:
                    print(f"  QA FAIL: {qa_reason} — deleting, trying next source", flush=True)
                    shutil.rmtree(pdir, ignore_errors=True)
                    qa_rejected += 1
                    continue
                print(f"  QA PASS: {qa_reason}", flush=True)

                # metadata_real.json
                (pdir / "metadata_real.json").write_text(json.dumps({
                    "patient_id": pid, "population": pop,
                    "pfd_pattern": pat, "pfd_grade": grd,
                    "source_dataset": src["dataset"], "source_uid": src["uid"],
                    "hu_window": [HU_MIN, HU_MAX], "approach": "real_ct_wide_hu",
                    "slices": int(vol_hu.shape[0]), "qa_attempts": attempt_i + 1,
                }, indent=2))

                done_names.add(pid)
                total_done += 1
                total_so_far = len(plain_done_names) + len(hilly_done_names)
                pct = total_so_far / (N_PLAIN + N_HILLY) * 100
                print(f"  OK -> {pid}  [{total_so_far}/{N_PLAIN+N_HILLY}  {pct:.1f}%]", flush=True)
                accepted = True
                break  # patient accepted — move to next slot

            except Exception as e:
                print(f"  ERROR (attempt {attempt_i+1}): {e}", flush=True)
                traceback.print_exc(file=sys.stdout)
                shutil.rmtree(pdir, ignore_errors=True)

        if not accepted:
            print(f"  EXHAUSTED all sources for {pid} — skipping slot", flush=True)
            failed += 1


print("\n=== Processing Plain (175) ===")
_process(plain_plan, PLAIN_DIR, plain_done_names)

print("\n=== Processing Hilly (175) ===")
_process(hilly_plan, HILLY_DIR, hilly_done_names)

print(f"\n{'='*50}")
print(f"DONE.  generated={total_done}  skipped={skipped}  failed={failed}  qa_rejected={qa_rejected}")
print(f"Plain: {len(plain_done_names)}/175   Hilly: {len(hilly_done_names)}/175")
print(f"Output: {OUT_ROOT}")
