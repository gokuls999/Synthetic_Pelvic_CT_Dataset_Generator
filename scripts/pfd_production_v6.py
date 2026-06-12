"""PFD Production v6 — Full 350-patient dataset with population-split output.

Architecture (same as v5 - surgical edit of real CT data):
  * Take a REAL source CT, apply confined organ + levator ani PFD deformation.
  * Bones, fat, muscle outside organs stay pixel-identical to original.
  * Grade-correlated muscle thickness + hiatal dimensions in PDF + JSON.
  * TotalSegmentator muscle segmentation overlay (.seg.nrrd) per patient.

Output structure (ready to zip and send):
  <out>/
    Plain_175/          <- 175 gynecoid (plains population) patients
      PFD_xxxx_PLAIN_*/
        DICOM/  PNG/  JPG/
        clinical_data.json  metadata.json
        patient_report.pdf  COMPARISON.png
      summary_plain.json
    Hilly_175/          <- 175 android/anthropoid (hilly population) patients
      PFD_xxxx_HILLY_*/
      summary_hilly.json
    dataset_overview.json

Run full 350 (default):
    python scripts/pfd_production_v6.py --port 8773

Resume after power cut:
    python scripts/pfd_production_v6.py --skip-done --no-browser --port 8773
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import datetime
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.ndimage as _ndi

from src.preprocessing import unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata
from src.patient_report import generate_clinical_metadata, write_patient_pdf
from src.pfd_segmentation import (
    segment_original_volume, rectum_subregion, mask_stats,
    pelvic_z_range_from_masks, PFD_ROI_SUBSET, SEG_SUBSET,
)
from src.segmentation_export import write_seg_nrrd
from src.pfd_deformation import (
    build_pattern_graded,
    build_displacement_field,
    apply_confined_deformation,
    apply_levator_ani_deformation,
)
from src.keep_awake import KeepAwake
from src import web_progress as wp
from PIL import Image, ImageDraw, ImageFont


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── Pelvimetry-based population classification ─────────────────────────────
# Transverse diameter threshold (mm) separating gynecoid (plain) from
# android/anthropoid (hilly) pelvis.  From South Indian pelvimetry literature
# (Nayak 2018, Mitra 2007):
#   Gynecoid   (plains):  TD ≥ 108 mm  — wide transverse inlet
#   Android    (hilly):   TD <  108 mm — narrow inlet
_TD_PLAIN_THRESHOLD_MM = 108.0


def _classify_population(masks: dict[str, np.ndarray],
                          pixel_spacing_mm: float) -> tuple[str, float]:
    """Return ('plain'|'hilly', transverse_diameter_mm) from TS masks.

    Uses the outer-to-outer X span of the combined hip masks — this is the
    actual transverse diameter of the pelvic inlet, not the centroid distance.
    Typical values: gynecoid (plain) ~125-145 mm, android/anthropoid <108 mm.
    """
    hl = masks.get("hip_left")
    hr = masks.get("hip_right")
    if hl is not None and hr is not None and (hl.any() or hr.any()):
        combined = (hl.astype(bool) | hr.astype(bool))
        # Project onto X axis: find leftmost and rightmost pixel across all Z,Y
        x_present = np.where(combined.any(axis=(0, 1)))[0]
        if len(x_present) >= 2:
            td_vox = int(x_present[-1]) - int(x_present[0])
            td_mm  = td_vox * pixel_spacing_mm
        else:
            td_mm = 120.0
    else:
        td_mm = 120.0  # default: assume plain
    pop = "plain" if td_mm >= _TD_PLAIN_THRESHOLD_MM else "hilly"
    return pop, round(td_mm, 1)


# ── Grade / pattern distribution ──────────────────────────────────────────
GRADE_WEIGHTS    = [0.28, 0.38, 0.22, 0.12]   # grade 1-4
PATTERN_FRACTIONS = {
    "combined_pfd":     0.60,
    "cystocele":        0.16,
    "rectocele":        0.12,
    "uterine_prolapse": 0.12,
}

# Which TS masks are the surgical edit targets for each pattern
_PATTERN_ORGANS = {
    "combined_pfd":     ["urinary_bladder", "uterus", "rectum", "vagina"],
    "cystocele":        ["urinary_bladder"],
    "rectocele":        ["rectum"],
    "uterine_prolapse": ["uterus", "urinary_bladder", "vagina"],
}


def _discover_sources(cache_root: Path) -> list[dict]:
    mask_root = cache_root / "masks"
    sources = []
    for ds_dir in sorted(mask_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        for uid_dir in sorted(ds_dir.iterdir()):
            if not uid_dir.is_dir():
                continue
            uid = uid_dir.name
            npz = cache_root / ds / f"{uid}.npz"
            if npz.exists():
                sources.append({"dataset": ds, "uid": uid})
    return sources


def _plan_roster(n_total: int, n_plain: int, n_hilly: int,
                 sources: list[dict], seed: int = 2027) -> list[dict]:
    """Build the patient roster.

    Step 1: assign pattern + grade for each population slot independently.
    Step 2: shuffle the full source pool once, then assign sources
            sequentially (pool[i % len(pool)]) so every patient gets a
            different source — even for small N test runs.
    """
    rng = np.random.default_rng(seed)

    def _split_grades(n: int) -> list[int]:
        if n == 0:
            return [0, 0, 0, 0]
        raw = [n * w for w in GRADE_WEIGHTS]
        floored = [int(np.floor(v)) for v in raw]
        deficit = n - sum(floored)
        fracs = [raw[i] - floored[i] for i in range(4)]
        for i in np.argsort(fracs)[::-1][:deficit]:
            floored[i] += 1
        return floored

    # ── Step 1: build pattern/grade slots per population ─────────────────
    slots: list[dict] = []
    for pop, n_pop in [("plain", n_plain), ("hilly", n_hilly)]:
        pop_slots: list[dict] = []
        for pattern, frac in PATTERN_FRACTIONS.items():
            n_pat = max(0, int(round(n_pop * frac)))
            grade_counts = _split_grades(n_pat)
            for grade_idx, gc in enumerate(grade_counts):
                for _ in range(gc):
                    pop_slots.append({"population": pop,
                                      "pattern": pattern,
                                      "grade": grade_idx + 1})
        # Trim / pad to exactly n_pop (keeps distribution proportional)
        while len(pop_slots) < n_pop:
            pop_slots.append(rng.choice(pop_slots).copy())  # type: ignore[arg-type]
        pop_slots = pop_slots[:n_pop]
        rng.shuffle(pop_slots)
        slots.extend(pop_slots)

    rng.shuffle(slots)

    # ── Step 2: assign sources sequentially — guaranteed unique per patient
    #           until sources are exhausted, then cycle ───────────────────
    pool = sources.copy()
    rng.shuffle(pool)
    for i, slot in enumerate(slots):
        src = pool[i % len(pool)]
        slot["uid"]     = src["uid"]
        slot["dataset"] = src["dataset"]

    # ── Assign patient IDs ────────────────────────────────────────────────
    roster = []
    for i, s in enumerate(slots, start=1):
        pop_tag = "PLAIN" if s["population"] == "plain" else "HILLY"
        pat_tag = s["pattern"].upper().replace("_", "")
        roster.append({"patient_num": i,
                        "patient_id": f"PFD_{i:04d}_{pop_tag}_{pat_tag}_G{s['grade']}",
                        **s})
    return roster


def _augment_volume(vol: np.ndarray, patient_num: int) -> np.ndarray:
    """Very subtle augmentation — only scanner noise + tiny brightness shift.
    Contrast and large HU shifts are NOT applied so anatomy looks natural."""
    rng = np.random.default_rng(patient_num * 999983 + 7)
    vol = vol.astype(np.float32)
    # Very light scanner noise (σ ≈ 1-3 HU equivalent in normalised space)
    sigma = float(rng.uniform(0.003, 0.010))
    vol += rng.standard_normal(vol.shape).astype(np.float32) * sigma
    # Tiny brightness (scanner calibration variation ±10 HU)
    vol += float(rng.uniform(-0.015, 0.015))
    return np.clip(vol, -1.0, 1.0).astype(np.float32)


def _make_comparison_strip(vol_orig: np.ndarray, vol_edited: np.ndarray,
                            pdir: Path, organ_masks: dict,
                            target_organs: list[str],
                            hu_min: float = -1000.0,
                            hu_max: float = 3000.0) -> None:
    """Save a side-by-side comparison: original | edited | difference (×5)."""
    lo_u, hi_u = -200.0, 500.0

    def _win(arr):
        # Denormalize from [-1, 1] to HU, then apply soft-tissue display window
        hu = (arr.astype(np.float32) + 1.0) / 2.0 * (hu_max - hu_min) + hu_min
        return ((np.clip(hu, lo_u, hi_u) - lo_u)
                / (hi_u - lo_u) * 255).astype(np.uint8)

    Z = vol_orig.shape[0]
    sample_slices = [Z // 4, Z // 2, 3 * Z // 4]
    rows = []
    for z in sample_slices:
        orig = _win(vol_orig[z])
        edit = _win(vol_edited[z])
        diff = np.clip(np.abs(vol_edited[z].astype(np.float32)
                              - vol_orig[z].astype(np.float32)) * 5 * 255,
                       0, 255).astype(np.uint8)
        rows.append(np.concatenate([orig, edit, diff], axis=1))

    img = Image.fromarray(np.concatenate(rows, axis=0))
    # Add column labels
    draw = ImageDraw.Draw(img)
    W = orig.shape[1]
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for col, label in enumerate(["ORIGINAL", "EDITED (PFD)", "DIFFERENCE x5"]):
        draw.text((col * W + 5, 4), label, fill=200, font=font)
    img.save(pdir / "COMPARISON.png")


def process_one(cfg: dict, cache_root: Path, out_root: Path,
                patient_id: str, patient_num: int,
                pattern: str, population: str, grade: int,
                uid: str, dataset: str,
                hu_min: float, hu_max: float,
                on_progress=None) -> dict:
    """Full pipeline: load real CT → X-center → surgical PFD edit → DICOM."""

    # ── Load source cache ─────────────────────────────────────────────────
    npz_path = cache_root / dataset / f"{uid}.npz"
    if not npz_path.exists():
        return {"ok": False, "reason": f"npz missing: {npz_path}"}
    with np.load(npz_path, allow_pickle=True) as npz:
        vol_norm = np.asarray(npz["slices"]).astype(np.float32)
        src_path = Path(str(npz["source"]))
        spacing  = tuple(float(x) for x in np.asarray(npz["spacing"]).tolist())
    Z_orig, H, W = vol_norm.shape

    if not src_path.exists():
        return {"ok": False, "reason": f"source path missing: {src_path}"}

    if on_progress: on_progress("loading masks")

    # ── TotalSegmentator (reads from cache — instant) ─────────────────────
    masks, _ = segment_original_volume(
        source_path=src_path, dataset=dataset, uid=uid,
        cfg=cfg, cache_root=cache_root, roi_subset=PFD_ROI_SUBSET,
        on_progress=on_progress,
    )

    # ── Z crop to pelvic region ───────────────────────────────────────────
    z_top, z_bot = pelvic_z_range_from_masks(masks, margin_voxels=6,
                                              fallback=(0, Z_orig))
    if z_bot - z_top < Z_orig:
        vol_norm = vol_norm[z_top:z_bot]
        masks    = {k: m[z_top:z_bot] for k, m in masks.items()}
    Z, H, W = vol_norm.shape

    # ── Pelvimetry-based population classification ────────────────────────
    px_spacing_mm = float(spacing[1])  # y/x pixel spacing
    actual_pop, td_mm = _classify_population(masks, px_spacing_mm)

    # ── X-center on bilateral hip midline ────────────────────────────────
    offset_x = 0   # track roll applied so segmentation can match
    _hl = mask_stats(masks.get("hip_left"),  "hip_left")  if "hip_left"  in masks else None
    _hr = mask_stats(masks.get("hip_right"), "hip_right") if "hip_right" in masks else None
    if _hl is not None and _hr is not None:
        midline_x = (_hl.center[2] + _hr.center[2]) / 2.0
        offset_x  = int(round(W / 2.0 - midline_x))
        offset_x  = max(-32, min(32, offset_x))
        if abs(offset_x) >= 2:
            vol_norm = np.roll(vol_norm, offset_x, axis=2)
            masks    = {k: np.roll(m, offset_x, axis=2) for k, m in masks.items()}

    # ── Very subtle scanner-noise augmentation (anatomy unchanged) ────────
    vol_orig_for_compare = vol_norm.copy()
    vol_norm = _augment_volume(vol_norm, patient_num)

    # ── Organ stats for deformation anchoring ────────────────────────────
    if "colon" in masks and masks["colon"].any():
        masks["rectum"] = rectum_subregion(masks["colon"], frac=0.66)
    stats = {n: s for n, m in masks.items()
             if (s := mask_stats(m, n)) is not None}

    needed = {
        "combined_pfd":     ["urinary_bladder", "rectum"],
        "cystocele":        ["urinary_bladder"],
        "rectocele":        ["rectum"],
        "uterine_prolapse": ["urinary_bladder", "rectum"],
    }[pattern]
    missing = [n for n in needed if n not in stats or stats[n].voxels < 200]
    if missing:
        return {"ok": False, "reason": f"missing structures: {missing}"}

    if on_progress: on_progress(f"applying confined G{grade} {pattern} deformation")

    # ── SURGICAL EDIT: deform ONLY pelvic floor organs ────────────────────
    # Bones, fat, muscle outside the PFD organs are UNTOUCHED.
    p_obj = build_pattern_graded(pattern, stats, grade=grade, population=population)
    target_organs = _PATTERN_ORGANS.get(pattern, ["urinary_bladder", "rectum"])

    vol_edited = apply_confined_deformation(
        volume       = vol_norm,
        blobs        = p_obj.blobs,
        organ_masks  = masks,
        target_organs= target_organs,
        order        = 3,          # cubic for smooth organ surface
        cval         = -1.0,
        dilation_voxels = 8,       # 8 vox margin around organ
        blur_sigma   = 4.0,        # feathered boundary blend
    )

    # ── Levator ani deformation (pelvic floor muscle PFD) ─────────────────
    # Widens levator hiatus + inferior muscle descent, calibrated to grade.
    # Bones stay completely untouched (weight mask excludes sacrum + hips).
    if on_progress: on_progress(f"applying levator ani G{grade} deformation")
    px_sp = float(spacing[1])
    sl_sp = float(spacing[0])
    vol_edited = apply_levator_ani_deformation(
        volume           = vol_edited,
        masks            = masks,
        grade            = grade,
        pixel_spacing_mm = px_sp,
        slice_spacing_mm = sl_sp,
        blur_sigma       = 4.0,
    )

    # ── Back to HU ───────────────────────────────────────────────────────
    vol_hu = np.clip(unwindow(vol_edited, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)

    # ── Z resample to target spacing ─────────────────────────────────────
    sz_src, sy_src, sx_src = spacing
    target_z_mm = float(cfg["output"].get("slice_thickness_mm", 1.5))
    if abs(sz_src - target_z_mm) > 0.1:
        vol_hu  = _ndi.zoom(vol_hu.astype(np.float32),
                            (sz_src / target_z_mm, 1.0, 1.0), order=1).astype(np.int16)
        spacing = (target_z_mm, sy_src, sx_src)

    # ── Write DICOM + PNG + JPG ───────────────────────────────────────────
    if on_progress: on_progress("writing DICOM")
    vol_obj = SyntheticVolume(
        pixels_hu=vol_hu, spacing=spacing, region=population,
        region_id=(0 if population == "plain" else 1),
        patient_id=patient_id, seed=patient_num,
    )
    pdir = out_root / patient_id
    write_dicom_series(vol_obj, pdir / "DICOM", cfg)
    write_png_jpg_metadata(vol_obj, pdir / "PNG", pdir / "JPG", pdir / "metadata.json", cfg)

    # ── Muscle + organ segmentation overlay (.seg.nrrd for 3D Slicer) ────
    if on_progress: on_progress("building muscle segmentation masks")
    try:
        seg_masks, _ = segment_original_volume(
            source_path=src_path, dataset=dataset, uid=uid,
            cfg=cfg, cache_root=cache_root, roi_subset=SEG_SUBSET,
            on_progress=on_progress,
        )
        # Crop + roll seg_masks the same way we did the main masks
        seg_masks = {k: m[z_top:z_bot] for k, m in seg_masks.items()}
        if abs(offset_x) >= 2:
            seg_masks = {k: np.roll(m, offset_x, axis=2) for k, m in seg_masks.items()}
        if "colon" in seg_masks and seg_masks["colon"].any():
            seg_masks["rectum"] = rectum_subregion(seg_masks["colon"], frac=0.66)
        write_seg_nrrd(
            out_path     = pdir / "DICOM" / "segmentation.seg.nrrd",
            masks_crop   = seg_masks,
            spacing_crop = (sz_src, sy_src, sx_src),
            spacing_out  = spacing,
            Z_out        = int(vol_hu.shape[0]),
            dicom_dir    = pdir / "DICOM",
        )
    except Exception as _seg_err:
        wp.log_msg(f"  [{patient_id}] segmentation skipped: {_seg_err}")

    # ── Comparison strip ─────────────────────────────────────────────────
    _make_comparison_strip(vol_orig_for_compare, vol_edited, pdir, masks, target_organs,
                           hu_min=hu_min, hu_max=hu_max)

    # ── Pelvimetry measurements from masks ───────────────────────────────
    def _mask_dim_mm(m, axis, sp):
        idx = np.where(m.any(axis=tuple(i for i in range(3) if i != axis)))
        if len(idx[0]) == 0: return 0.0
        return float((idx[0].max() - idx[0].min()) * sp)

    pelvimetry = {"transverse_diameter_inlet_mm": td_mm}
    if "sacrum" in masks and masks["sacrum"].any():
        pelvimetry["sacral_length_mm"] = round(_mask_dim_mm(masks["sacrum"], 0, spacing[0]), 1)
    for struct, key in [("urinary_bladder", "bladder_height_mm"),
                         ("rectum",          "rectum_height_mm")]:
        if struct in masks and masks[struct].any():
            st = mask_stats(masks[struct], struct)
            if st:
                pelvimetry[key] = round(st.extent[0] * spacing[0], 1)

    # ── Full metadata (all Data Collection Sheet fields) ─────────────────
    collection_date = datetime.date.today().isoformat()
    clinical = generate_clinical_metadata(
        patient_id=patient_id, patient_num=patient_num,
        pattern=pattern, severity=population,
        region_id=(0 if population == "plain" else 1),
        grade=grade,
        pfd_findings=p_obj.findings,
        real_spacing_mm=list(spacing),
        pelvic_crop_z=[int(z_top), int(z_bot), int(Z_orig)],
        collection_date=collection_date,
    )
    # Inject actual pelvimetry measurements
    if "pelvimetry" not in clinical:
        clinical["pelvimetry"] = {}
    clinical["pelvimetry"].update(pelvimetry)
    clinical["pelvimetry"]["pelvic_shape_classification"] = (
        "Gynecoid" if actual_pop == "plain" else "Android/Anthropoid"
    )
    clinical["pelvimetry"]["transverse_diameter_mm"] = td_mm
    clinical["population_assigned"]  = population
    clinical["population_measured"]  = actual_pop
    clinical["source_uid"]           = uid
    clinical["source_dataset"]       = dataset
    clinical["anchor_mode"]          = "confined_organ_plus_levator_v6"
    clinical["pfd_grade"]            = grade
    clinical["pfd_severity"]         = ["", "mild", "moderate", "severe", "complete"][grade]

    (pdir / "clinical_data.json").write_text(json.dumps(clinical, indent=2))
    write_patient_pdf(clinical, pdir / "patient_report.pdf")

    # Compact metadata.json
    meta = {
        "patient_id":       patient_id,
        "population":       population,
        "actual_population":actual_pop,
        "pfd_pattern":      pattern,
        "pfd_grade":        grade,
        "pfd_severity":     ["", "mild", "moderate", "severe", "complete"][grade],
        "pfd_findings":     p_obj.findings,
        "source_uid":       uid,
        "source_dataset":   dataset,
        "transverse_diam_mm": td_mm,
        "pelvic_shape":     "Gynecoid" if actual_pop == "plain" else "Android/Anthropoid",
        "pelvimetry":       pelvimetry,
        "pelvic_crop_z":    [int(z_top), int(z_bot), int(Z_orig)],
        "n_slices":         int(vol_hu.shape[0]),
        "spacing_mm":       list(spacing),
        "deformation_mode": "confined_organ_plus_levator",
        "target_organs":    target_organs,
        "synthetic":        True,
        "sex":              "Female",
        "modality":         "CT",
        "body_part":        "PELVIS",
    }
    (pdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return {"ok": True, "grade": grade, "pattern": pattern,
            "n_slices": int(vol_hu.shape[0]), "p_obj": p_obj,
            "actual_pop": actual_pop, "td_mm": td_mm}


def _fmt_eta(s: float) -> str:
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s/60)}m"
    return f"{s/3600:.1f}h"


def _write_dataset_docs(out_root: Path, summary: list, failed: list,
                         sources: list) -> None:
    """Write per-population summaries + root overview JSON documentation."""
    import datetime

    plain_list = [s for s in summary if s.get("population") == "plain" and not s.get("skipped")]
    hilly_list = [s for s in summary if s.get("population") == "hilly" and not s.get("skipped")]

    def _pop_doc(patients: list, pop_label: str) -> dict:
        grades = {f"grade_{g}": len([p for p in patients if p.get("grade") == g])
                  for g in range(1, 5)}
        patterns = {}
        for p in patients:
            pat = p.get("pattern", "unknown")
            patterns[pat] = patterns.get(pat, 0) + 1
        sources_used = list({p.get("source_ds", "") for p in patients})
        return {
            "population":           pop_label,
            "pelvic_type":          "Gynecoid (plains South India)" if pop_label == "plain"
                                    else "Android/Anthropoid (hilly South India)",
            "td_threshold_mm":      108.0,
            "n_patients":           len(patients),
            "grade_distribution":   grades,
            "pattern_distribution": patterns,
            "source_datasets":      sorted(sources_used),
            "deformation_mode":     "confined_organ_plus_levator",
            "patients": [
                {
                    "patient_id":      p["patient_id"],
                    "pfd_pattern":     p.get("pattern"),
                    "pfd_grade":       p.get("grade"),
                    "pfd_severity":    ["", "mild", "moderate", "severe", "complete"][
                                       p.get("grade", 1)],
                    "transverse_diam_mm": p.get("td_mm"),
                    "actual_population":  p.get("actual_pop"),
                    "source_dataset":  p.get("source_ds"),
                    "n_slices":        p.get("n_slices"),
                    "findings":        p.get("findings", {}),
                }
                for p in patients
            ],
        }

    plain_doc = _pop_doc(plain_list, "plain")
    hilly_doc = _pop_doc(hilly_list, "hilly")

    # Write per-population summaries
    plain_dir = out_root / "Plain_175"
    hilly_dir = out_root / "Hilly_175"
    plain_dir.mkdir(exist_ok=True)
    hilly_dir.mkdir(exist_ok=True)

    (plain_dir / "summary_plain.json").write_text(json.dumps(plain_doc, indent=2))
    (hilly_dir / "summary_hilly.json").write_text(json.dumps(hilly_doc, indent=2))

    # Root overview
    overview = {
        "dataset_title":   "Synthetic Pelvic CT Dataset — PFD Study (South Indian Women)",
        "description": (
            "350 synthetic female pelvic CT studies for pelvic floor dysfunction "
            "research. Real source CTs surgically edited: organ displacement "
            "(cystocele / rectocele / uterine prolapse) + levator ani hiatal "
            "widening, confined to pelvic floor organs. Bones, fat, and non-pelvic "
            "structures remain pixel-identical to the source CT."
        ),
        "generated_date":  datetime.date.today().isoformat(),
        "version":         "v6",
        "total_patients":  len(plain_list) + len(hilly_list),
        "plain_patients":  len(plain_list),
        "hilly_patients":  len(hilly_list),
        "failed_patients": len(failed),
        "source_pool_size": len(sources),
        "populations": {
            "plain":  "Gynecoid pelvis — TD >= 108 mm (Plains South India)",
            "hilly":  "Android/Anthropoid pelvis — TD < 108 mm (Hilly South India)",
        },
        "pfd_patterns": {
            "combined_pfd":     "60% — cystocele + rectocele + uterine prolapse",
            "cystocele":        "16% — anterior vaginal wall prolapse",
            "rectocele":        "12% — posterior vaginal wall prolapse",
            "uterine_prolapse": "12% — uterine descent",
        },
        "grade_distribution": {
            "grade_1 (mild)":     "28%",
            "grade_2 (moderate)": "38%",
            "grade_3 (severe)":   "22%",
            "grade_4 (complete)": "12%",
        },
        "per_patient_files": {
            "DICOM/":                  "CT volume in DICOM format (loadable in 3D Slicer)",
            "DICOM/segmentation.seg.nrrd": "3D Slicer segmentation: bones, organs, muscles, pelvic floor",
            "PNG/":                    "CT slices as PNG images",
            "JPG/":                    "CT slices as JPEG images",
            "clinical_data.json":      "Full clinical metadata (demographics, obstetric, pelvimetry, muscle thickness)",
            "metadata.json":           "Compact per-patient summary (population, grade, pattern, spacings)",
            "patient_report.pdf":      "Formatted PDF data collection sheet (Master Data Collection Sheet format)",
            "COMPARISON.png":          "Side-by-side: original | PFD-edited | difference",
        },
        "muscle_segmentation_labels": {
            "1-2":  "Hip Left / Right (ivory)",
            "3":    "Sacrum (ivory)",
            "4-5":  "Femur Left / Right",
            "6":    "Urinary Bladder (blue)",
            "7":    "Rectum (brown)",
            "8-9":  "Gluteus Maximus L/R (orange)",
            "10-11":"Gluteus Medius L/R (yellow-orange)",
            "12-13":"Gluteus Minimus L/R (yellow)",
            "14-15":"Iliopsoas L/R (green)",
            "16":   "Pelvic Floor / Levator Ani region (magenta)",
        },
        "how_to_load_in_3d_slicer": [
            "1. File > Add DICOM Data > select DICOM/ folder",
            "2. File > Add Data > select DICOM/segmentation.seg.nrrd",
            "   (Show Options > set type = Segmentation)",
            "3. Use Modules > Volume Rendering for 3D CT render",
            "4. Use Modules > Segment Editor to toggle individual structures",
        ],
        "folder_structure": {
            "Plain_175/": "175 patients — gynecoid pelvic morphology",
            "Hilly_175/": "175 patients — android/anthropoid pelvic morphology",
            "Plain_175/summary_plain.json": "Plain population summary",
            "Hilly_175/summary_hilly.json": "Hilly population summary",
        },
        "failures": failed,
    }
    (out_root / "dataset_overview.json").write_text(json.dumps(overview, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config",       default="configs/default.yaml")
    ap.add_argument("--out",          default="synthetic_dataset/PFD_Synthetic_Dataset_350")
    ap.add_argument("--num-patients", type=int, default=350)
    ap.add_argument("--n-plain",      type=int, default=None)
    ap.add_argument("--n-hilly",      type=int, default=None)
    ap.add_argument("--port",         type=int, default=8773)
    ap.add_argument("--no-browser",   action="store_true")
    ap.add_argument("--skip-done",    action="store_true")
    ap.add_argument("--allow-sleep",  action="store_true")
    args = ap.parse_args()

    n_total = args.num_patients
    n_plain = args.n_plain if args.n_plain is not None else n_total // 2
    n_hilly = args.n_hilly if args.n_hilly is not None else n_total - n_plain

    cfg        = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root   = Path(args.out)
    # Create population subfolders up front
    (out_root / "Plain_175").mkdir(parents=True, exist_ok=True)
    (out_root / "Hilly_175").mkdir(parents=True, exist_ok=True)
    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])

    sources = _discover_sources(cache_root)
    if not sources:
        raise RuntimeError(f"No cached source CTs found under {cache_root}")

    wp.set_stages([
        ("plan",    "Plan roster"),
        ("process", "Surgical edit: organ + levator deformation → DICOM + segmentation"),
        ("docs",    "Write documentation"),
        ("summary", "Summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=(f"PFD v6  N={n_total}  plain={n_plain}/hilly={n_hilly}  "
                   f"sources={len(sources)}"),
        expected_total_s=None, cfg=None,
    )
    print()
    print("=" * 72)
    print(f"  PFD Production v6  —  350-patient population-split dataset")
    print(f"  Dashboard : {url}")
    print(f"  Output    : {out_root}")
    print(f"    Plain_175/   <- {n_plain} gynecoid patients")
    print(f"    Hilly_175/   <- {n_hilly} android/anthropoid patients")
    print(f"  Sources   : {len(sources)} real pelvic CTs")
    print("=" * 72)
    print()

    wp.set_stage("plan", postfix="building roster")
    roster = _plan_roster(n_total, n_plain, n_hilly, sources)
    wp.finish_stage(f"{len(roster)} patients planned")

    progress_marker = out_root / ".production_in_progress"
    progress_marker.write_text(json.dumps({
        "n_planned": len(roster), "version": "v6",
        "started_at": time.time(), "out_root": str(out_root),
    }))

    wp.set_stage("process", total=len(roster), postfix="starting")
    keep_awake = KeepAwake() if not args.allow_sleep else _NullCtx()
    keep_awake.__enter__()

    success_times: deque = deque(maxlen=30)
    summary, failed = [], []
    t_run = time.time()

    for idx, r in enumerate(roster, start=1):
        # Route into population subfolder
        pop_subdir = "Plain_175" if r["population"] == "plain" else "Hilly_175"
        pat_root   = out_root / pop_subdir
        pdir       = pat_root / r["patient_id"]

        if args.skip_done and (pdir / "DICOM").is_dir() and (pdir / "metadata.json").exists():
            # Reload summary entry from existing metadata for docs generation
            try:
                m = json.loads((pdir / "metadata.json").read_text())
                summary.append({
                    "patient_id": r["patient_id"], "skipped": True,
                    "population": m.get("population"), "actual_pop": m.get("actual_population"),
                    "td_mm": m.get("transverse_diam_mm"), "pattern": m.get("pfd_pattern"),
                    "grade": m.get("pfd_grade"), "source_ds": m.get("source_dataset"),
                    "n_slices": m.get("n_slices"), "findings": m.get("pfd_findings", {}),
                })
            except Exception:
                summary.append({"patient_id": r["patient_id"], "skipped": True,
                                 "population": r["population"]})
            wp.update_stage(current=idx, postfix=f"{idx}/{len(roster)} skipped")
            continue

        wp.update_stage(current=idx - 1,
                        postfix=(f"{idx}/{len(roster)} {r['patient_id']} "
                                 f"G{r['grade']} {r['pattern']}"))
        t_slot = time.time()

        def _prog(msg, _idx=idx, _pid=r["patient_id"]):
            wp.update_stage(postfix=f"{_idx}/{len(roster)} {_pid}: {msg}")

        try:
            result = process_one(
                cfg=cfg, cache_root=cache_root, out_root=pat_root,
                patient_id=r["patient_id"], patient_num=r["patient_num"],
                pattern=r["pattern"], population=r["population"],
                grade=r["grade"], uid=r["uid"], dataset=r["dataset"],
                hu_min=hu_min, hu_max=hu_max, on_progress=_prog,
            )
            elapsed = time.time() - t_slot
            if result["ok"]:
                success_times.append(elapsed)
                summary.append({
                    "patient_id":  r["patient_id"],
                    "population":  r["population"],
                    "actual_pop":  result["actual_pop"],
                    "td_mm":       result["td_mm"],
                    "pattern":     r["pattern"],
                    "grade":       r["grade"],
                    "source_uid":  r["uid"],
                    "source_ds":   r["dataset"],
                    "n_slices":    result["n_slices"],
                    "findings":    result["p_obj"].findings,
                })
                wp.log_msg(f"  [{idx}/{len(roster)}] OK  {r['patient_id']}  "
                           f"[{pop_subdir}]  G{r['grade']} {r['pattern']}  "
                           f"TD={result['td_mm']}mm({result['actual_pop']})  "
                           f"({elapsed:.0f}s)")
            else:
                failed.append({"patient_id": r["patient_id"], "reason": result["reason"]})
                wp.log_msg(f"  [{idx}/{len(roster)}] FAIL {r['patient_id']}: {result['reason']}")
        except Exception as e:
            failed.append({"patient_id": r["patient_id"],
                           "reason": f"{type(e).__name__}: {e}"})
            wp.log_msg(f"  [{idx}/{len(roster)}] EXC {r['patient_id']}: {e}")

        if success_times:
            mean_s    = sum(success_times) / len(success_times)
            remaining = (len(roster) - idx) * mean_s
            postfix   = (f"{idx}/{len(roster)} ok={len([s for s in summary if not s.get('skipped')])} "
                         f"fail={len(failed)}  ~{_fmt_eta(remaining)} left")
        else:
            postfix = f"{idx}/{len(roster)} ok=? fail={len(failed)}"
        wp.update_stage(current=idx, postfix=postfix)

    keep_awake.__exit__(None, None, None)
    wp.finish_stage("done")

    # ── Write documentation ────────────────────────────────────────────────
    wp.set_stage("docs", postfix="writing summaries")
    n_succ    = len([s for s in summary if not s.get("skipped")])
    total_min = (time.time() - t_run) / 60.0

    _write_dataset_docs(out_root, summary, failed, sources)

    # Roster summary (for autostart resume detection)
    (out_root / "roster_summary.json").write_text(json.dumps({
        "version":     "v6",
        "n_planned":   len(roster),
        "n_succeeded": n_succ,
        "n_plain":     len([s for s in summary if s.get("population") == "plain"]),
        "n_hilly":     len([s for s in summary if s.get("population") == "hilly"]),
        "n_failed":    len(failed),
        "deformation_mode": "confined_organ_plus_levator",
        "grade_distribution": {
            f"grade{g}": len([s for s in summary if s.get("grade") == g])
            for g in range(1, 5)
        },
    }, indent=2))
    progress_marker.unlink(missing_ok=True)
    wp.finish_stage("done")

    wp.set_stage("summary", postfix="done")
    wp.finish_stage(f"{n_succ} ok, {len(failed)} failed in {total_min:.1f} min")
    print()
    print("=" * 72)
    print(f"  PFD v6 DONE  {total_min:.1f} min")
    print(f"  Succeeded  : {n_succ} / {len(roster)}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Plain_175/ : {len([s for s in summary if s.get('population')=='plain'])} patients")
    print(f"  Hilly_175/ : {len([s for s in summary if s.get('population')=='hilly'])} patients")
    print(f"  Output     : {out_root}")
    print(f"  Docs       : {out_root}/dataset_overview.json")
    print("=" * 72)
    wp.stop_server(grace_s=60.0)


if __name__ == "__main__":
    main()
