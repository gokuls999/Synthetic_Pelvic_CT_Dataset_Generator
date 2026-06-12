"""PFD Production v3 — Study-faithful dataset generator.

Key differences from v2:
  * Uses TWO curated representative source CTs that match the study populations:
      - PLAIN  : dataset6_CLINIC_0006_data  (gynecoid pelvis, TC ~12 cm)
      - HILLY  : dataset6_CLINIC_0059_data  (android/anthropoid, TC ~10.5 cm)
  * Four POP-Q clinical grades (1–4) per pattern, calibrated from South Indian
    PFD literature (Nayak et al. 2018, ICS POP-Q Bump et al. 1996):
      Grade 1  8 mm   — early, POP-Q stage I
      Grade 2  18 mm  — moderate, POP-Q stage II
      Grade 3  28 mm  — severe, POP-Q stage III
      Grade 4  40 mm  — complete, POP-Q stage IV
  * Per-patient augmentation (noise, contrast, brightness) so every DICOM
    looks like a distinct individual despite sharing the same template CT.
  * plain / hilly labels describe POPULATION MORPHOLOGY, not severity — both
    populations present with all four PFD grades.
  * Power-cut resilient: .production_in_progress marker + --skip-done.

Grade distribution (from Indian PFD prevalence data):
  Grade 1 : 28%   (early / subclinical)
  Grade 2 : 38%   (moderate / most common referral)
  Grade 3 : 22%   (severe)
  Grade 4 : 12%   (complete prolapse / surgical candidates)

Pattern distribution (same as v2):
  combined_pfd     60%
  cystocele        16%
  rectocele        12%
  uterine_prolapse 12%
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
    pelvic_z_range_from_masks, PFD_ROI_SUBSET,
)
from src.pfd_deformation import (
    build_pattern_graded,
    build_displacement_field, apply_deformation,
)
from src.keep_awake import KeepAwake
from src import web_progress as wp
from PIL import Image


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── Study-representative source templates ─────────────────────────────────
# Chosen from ctpelvic1k by hip-centroid symmetry score (lowest offset = best).
TEMPLATE_SOURCES = {
    "plain": {
        "uid":        "dataset6_CLINIC_0006_data",
        "dataset":    "ctpelvic1k",
        "sym_offset": 1.0,          # px from bilateral midline
        "pelvis_type": "Gynecoid",
        "tc_range_cm": (11.0, 13.0),
    },
    "hilly": {
        "uid":        "dataset6_CLINIC_0059_data",
        "dataset":    "ctpelvic1k",
        "sym_offset": 0.0,
        "pelvis_type": "Android/Anthropoid",
        "tc_range_cm": (9.5, 11.5),
    },
}

# ── Clinical grade distribution (ICS / South Indian PFD literature) ────────
GRADE_WEIGHTS = [0.28, 0.38, 0.22, 0.12]   # grade 1-4

# ── Pattern mix ─────────────────────────────────────────────────────────────
PATTERN_FRACTIONS = {
    "combined_pfd":     0.60,
    "cystocele":        0.16,
    "rectocele":        0.12,
    "uterine_prolapse": 0.12,
}


def _plan_roster(n_total: int, n_plain: int, n_hilly: int,
                 seed: int = 2027) -> list[dict]:
    """Build the 350-patient roster round-robin over patterns × grades × population."""
    rng = np.random.default_rng(seed)

    def _split_grades(n: int) -> list[int]:
        raw = [n * w for w in GRADE_WEIGHTS]
        floored = [int(np.floor(v)) for v in raw]
        deficit = n - sum(floored)
        fracs = [raw[i] - floored[i] for i in range(4)]
        for i in np.argsort(fracs)[::-1][:deficit]:
            floored[i] += 1
        return floored   # [n_grade1, n_grade2, n_grade3, n_grade4]

    slots: list[dict] = []
    for pop, n_pop in (("plain", n_plain), ("hilly", n_hilly)):
        src = TEMPLATE_SOURCES[pop]
        for pattern, frac in PATTERN_FRACTIONS.items():
            n_pat = int(round(n_pop * frac))
            grade_counts = _split_grades(n_pat)
            for grade_idx, gc in enumerate(grade_counts):
                grade = grade_idx + 1
                for _ in range(gc):
                    slots.append({
                        "population": pop,
                        "pattern":    pattern,
                        "grade":      grade,
                        "uid":        src["uid"],
                        "dataset":    src["dataset"],
                    })

    # Trim / pad to exact total
    while len(slots) < n_total:
        slots.append(slots[rng.integers(len(slots))])
    slots = slots[:n_total]

    rng.shuffle(slots)

    # Assign patient IDs
    roster = []
    for i, s in enumerate(slots, start=1):
        pop_tag = "PLAIN" if s["population"] == "plain" else "HILLY"
        roster.append({
            "patient_num": i,
            "patient_id":  f"PFD_{i:04d}_{pop_tag}_{s['pattern'].upper()}_G{s['grade']}",
            **s,
        })
    return roster


def _augment_volume(vol: np.ndarray, patient_num: int) -> np.ndarray:
    """Per-patient deterministic augmentation: noise + contrast + brightness.

    Operates in the normalised [-1, 1] space of the cache volumes.
    Augmentation is seeded from patient_num so it is reproducible.
    """
    rng = np.random.default_rng(patient_num * 999983 + 7)
    vol = vol.astype(np.float32)

    # Gaussian noise (simulates scanner dose / reconstruction variation)
    sigma = float(rng.uniform(0.005, 0.022))
    vol += rng.standard_normal(vol.shape).astype(np.float32) * sigma

    # Contrast: scale around the mean (simulates tissue HU shift)
    contrast = float(rng.uniform(0.90, 1.10))
    mu = float(vol.mean())
    vol = (vol - mu) * contrast + mu

    # Brightness: global HU offset (simulates scanner calibration variation)
    bright = float(rng.uniform(-0.04, 0.04))
    vol = vol + bright

    return np.clip(vol, -1.0, 1.0).astype(np.float32)


def _to_u8(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return np.zeros_like(s, dtype=np.uint8)
    return ((s - lo) / (hi - lo) * 255).astype(np.uint8)


def process_one(cfg: dict, cache_root: Path, out_root: Path,
                patient_id: str, patient_num: int,
                pattern: str, population: str, grade: int,
                uid: str, dataset: str,
                hu_min: float, hu_max: float,
                on_progress=None) -> dict:
    """Full pipeline for one patient: load -> center -> augment -> deform -> DICOM."""

    # ── Load template cache volume ────────────────────────────────────────
    npz_path = cache_root / dataset / f"{uid}.npz"
    if not npz_path.exists():
        return {"ok": False, "reason": f"cache npz missing: {npz_path}"}
    with np.load(npz_path, allow_pickle=True) as npz:
        vol_norm = np.asarray(npz["slices"]).astype(np.float32)
        src_path = Path(str(npz["source"]))
        spacing = tuple(float(x) for x in np.asarray(npz["spacing"]).tolist())
    Z_orig, H, W = vol_norm.shape

    if not src_path.exists():
        return {"ok": False, "reason": f"original DICOM missing: {src_path}"}

    # ── TotalSegmentator (cached on first run, instant thereafter) ────────
    masks, _ = segment_original_volume(
        source_path=src_path, dataset=dataset, uid=uid,
        cfg=cfg, cache_root=cache_root, roi_subset=PFD_ROI_SUBSET,
        on_progress=on_progress,
    )

    # ── Z crop to pelvic region ────────────────────────────────────────────
    z_top, z_bot = pelvic_z_range_from_masks(masks, margin_voxels=6,
                                              fallback=(0, Z_orig))
    if z_bot - z_top < Z_orig:
        vol_norm = vol_norm[z_top:z_bot]
        masks = {k: m[z_top:z_bot] for k, m in masks.items()}
    Z, H, W = vol_norm.shape

    # ── X-center on bilateral hip midline (fix off-center positioning) ────
    _hl = mask_stats(masks["hip_left"],  "hip_left")  if "hip_left"  in masks else None
    _hr = mask_stats(masks["hip_right"], "hip_right") if "hip_right" in masks else None
    if _hl is not None and _hr is not None:
        midline_x = (_hl.center[2] + _hr.center[2]) / 2.0
        offset_x  = int(round(W / 2.0 - midline_x))
        offset_x  = max(-32, min(32, offset_x))
        if abs(offset_x) >= 2:
            vol_norm = np.roll(vol_norm, offset_x, axis=2)
            masks    = {k: np.roll(m, offset_x, axis=2) for k, m in masks.items()}

    # ── Per-patient augmentation (makes each patient look distinct) ───────
    vol_norm = _augment_volume(vol_norm, patient_num)

    # ── Organ stats for deformation anchoring ────────────────────────────
    if "colon" in masks and masks["colon"].any():
        masks["rectum"] = rectum_subregion(masks["colon"], frac=0.66)
    stats = {}
    for name, m in masks.items():
        st = mask_stats(m, name)
        if st is not None:
            stats[name] = st

    # Guard: required structures must be present
    needed = {
        "combined_pfd":     ["urinary_bladder", "rectum"],
        "cystocele":        ["urinary_bladder"],
        "rectocele":        ["rectum"],
        "uterine_prolapse": ["urinary_bladder", "rectum"],
    }[pattern]
    missing = [n for n in needed if n not in stats or stats[n].voxels < 200]
    if missing:
        return {"ok": False, "reason": f"missing structures: {missing}"}

    # ── Clinically graded PFD deformation ────────────────────────────────
    p_obj = build_pattern_graded(pattern, stats, grade=grade, population=population)
    field   = build_displacement_field((Z, H, W), p_obj.blobs)
    deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)

    vol_hu = np.clip(unwindow(deformed, hu_min, hu_max),
                     -1024.0, 3071.0).astype(np.int16)

    # ── Resample Z to target slice thickness ─────────────────────────────
    sz_src, sy_src, sx_src = spacing
    target_z_mm = float(cfg["output"].get("slice_thickness_mm", 1.5))
    if abs(sz_src - target_z_mm) > 0.1:
        zoom_z  = sz_src / target_z_mm
        vol_hu  = _ndi.zoom(vol_hu.astype(np.float32),
                            (zoom_z, 1.0, 1.0), order=1).astype(np.int16)
        spacing = (target_z_mm, sy_src, sx_src)

    # ── Write DICOM + PNG + metadata ─────────────────────────────────────
    vol_obj = SyntheticVolume(
        pixels_hu=vol_hu, spacing=spacing, region=population,
        region_id=(0 if population == "plain" else 1),
        patient_id=patient_id, seed=patient_num,
    )
    pdir      = out_root / patient_id
    dicom_dir = pdir / "DICOM"
    png_dir   = pdir / "PNG"
    jpg_dir   = pdir / "JPG"
    meta_path = pdir / "metadata.json"

    if dicom_dir.exists():
        for f in dicom_dir.glob("*.dcm"):
            f.unlink()
    write_dicom_series(vol_obj, dicom_dir, cfg)
    write_png_jpg_metadata(vol_obj, png_dir, jpg_dir, meta_path, cfg)

    # Comparison strip: original (template) | deformed (after augment+grade) | deformed
    lo_u, hi_u = -200.0, 500.0
    def to_u8_win(arr):
        s = np.clip(arr.astype(np.float32), lo_u, hi_u)
        return ((s - lo_u) / (hi_u - lo_u) * 255).astype(np.uint8)

    Z_final = vol_hu.shape[0]
    sample_z = [max(0, Z_final // 4), Z_final // 2, min(Z_final - 1, 3 * Z_final // 4)]
    rows = []
    for z in sample_z:
        orig_z  = min(int(z * vol_norm.shape[0] / Z_final), vol_norm.shape[0] - 1)
        orig_u8 = _to_u8(vol_norm[orig_z])
        def_u8  = to_u8_win(vol_hu[z])
        rows.append(np.concatenate([orig_u8, def_u8, def_u8], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(pdir / "COMPARISON.png")

    meta = {
        "patient_id":         patient_id,
        "population":         population,    # plain / hilly (morphological class)
        "pfd_pattern":        pattern,
        "pfd_grade":          grade,
        "pop_q_stage":        p_obj.findings.get("pop_q_stage",
                              p_obj.findings.get("uterine_prolapse_grade", grade)),
        "pfd_severity":       ["", "mild", "moderate", "severe", "complete"][grade],
        "pfd_findings":       p_obj.findings,
        "source_volume_uid":  uid,
        "source_dataset":     dataset,
        "template_type":      TEMPLATE_SOURCES[population]["pelvis_type"],
        "pelvic_crop_z":      [int(z_top), int(z_bot), int(Z_orig)],
        "n_slices":           int(vol_hu.shape[0]),
        "real_spacing_mm":    list(spacing),
        "anchor_mode":        "TS_graded_v3",
        "synthetic":          True,
        "sex":                "Female",
        "modality":           "CT",
        "body_part":          "PELVIS",
    }
    (pdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Clinical report
    collection_date = datetime.date.today().isoformat()
    clinical = generate_clinical_metadata(
        patient_id=patient_id, patient_num=patient_num,
        pattern=pattern, severity=population,
        region_id=(0 if population == "plain" else 1),
        pfd_findings=p_obj.findings,
        real_spacing_mm=list(spacing),
        pelvic_crop_z=[int(z_top), int(z_bot), int(Z_orig)],
        collection_date=collection_date,
    )
    (pdir / "clinical_data.json").write_text(json.dumps(clinical, indent=2))
    write_patient_pdf(clinical, pdir / "patient_report.pdf")

    return {
        "ok":      True,
        "grade":   grade,
        "pattern": pattern,
        "n_slices": int(vol_hu.shape[0]),
        "p_obj":   p_obj,
    }


def _fmt_eta(s: float) -> str:
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s/60)}m"
    return f"{s/3600:.1f}h"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config",      default="configs/default.yaml")
    ap.add_argument("--out",         default="synthetic_dataset/production_v3_350")
    ap.add_argument("--num-patients",type=int, default=350)
    ap.add_argument("--n-plain",     type=int, default=175)
    ap.add_argument("--n-hilly",     type=int, default=175)
    ap.add_argument("--port",        type=int, default=8768)
    ap.add_argument("--no-browser",  action="store_true")
    ap.add_argument("--skip-done",   action="store_true")
    ap.add_argument("--allow-sleep", action="store_true")
    args = ap.parse_args()

    cfg        = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root   = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])

    wp.set_stages([
        ("plan",    "Plan roster"),
        ("process", "Per-patient: load -> centre -> augment -> grade-deform -> DICOM"),
        ("summary", "Summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=(f"PFD v3 Study-faithful  N={args.num_patients}  "
                   f"plain={args.n_plain}/hilly={args.n_hilly}"),
        expected_total_s=None, cfg=None,
    )
    print()
    print("=" * 72)
    print(f"  PFD Production v3 dashboard : {url}")
    print(f"  Output root                 : {out_root}")
    print(f"  Plain template              : {TEMPLATE_SOURCES['plain']['uid']}")
    print(f"  Hilly template              : {TEMPLATE_SOURCES['hilly']['uid']}")
    print(f"  PFD grades                  : 1-4 (POP-Q calibrated)")
    print(f"  Augmentation                : noise + contrast + brightness per patient")
    print("=" * 72)
    print()

    wp.set_stage("plan", postfix="building roster")
    roster = _plan_roster(args.num_patients, args.n_plain, args.n_hilly)
    grade_dist = {}
    for r in roster:
        k = f"{r['population']}_G{r['grade']}_{r['pattern']}"
        grade_dist[k] = grade_dist.get(k, 0) + 1
    for k, v in sorted(grade_dist.items()):
        wp.log_msg(f"  {k}: {v}")
    wp.finish_stage(f"{len(roster)} slots planned")

    progress_marker = out_root / ".production_in_progress"
    progress_marker.write_text(json.dumps({
        "n_planned": len(roster), "version": "v3",
        "started_at": time.time(), "out_root": str(out_root),
    }))

    wp.set_stage("process", total=len(roster), postfix="starting")
    keep_awake = KeepAwake() if not args.allow_sleep else _NullCtx()
    keep_awake.__enter__()

    success_times: deque = deque(maxlen=30)
    summary = []
    failed  = []
    t_run   = time.time()

    for idx, r in enumerate(roster, start=1):
        pdir = out_root / r["patient_id"]

        if args.skip_done and (pdir / "DICOM").is_dir() and (pdir / "metadata.json").exists():
            summary.append({"patient_id": r["patient_id"], "skipped": True})
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(roster)} skipped (done)")
            continue

        wp.update_stage(current=idx - 1,
                        postfix=(f"{idx}/{len(roster)} {r['patient_id']} "
                                 f"G{r['grade']}/{r['pattern']}"))
        t_slot = time.time()

        def _prog(msg):
            wp.update_stage(postfix=f"{idx}/{len(roster)} {r['patient_id']}: {msg}")

        try:
            result = process_one(
                cfg=cfg, cache_root=cache_root, out_root=out_root,
                patient_id=r["patient_id"], patient_num=r["patient_num"],
                pattern=r["pattern"], population=r["population"],
                grade=r["grade"], uid=r["uid"], dataset=r["dataset"],
                hu_min=hu_min, hu_max=hu_max, on_progress=_prog,
            )
            if result["ok"]:
                success_times.append(time.time() - t_slot)
                summary.append({
                    "patient_id":  r["patient_id"],
                    "population":  r["population"],
                    "pattern":     r["pattern"],
                    "grade":       r["grade"],
                    "n_slices":    result["n_slices"],
                    "findings":    result["p_obj"].findings,
                })
                wp.log_msg(f"  [{idx}/{len(roster)}] OK  {r['patient_id']}  "
                           f"G{r['grade']} {r['pattern']}  "
                           f"({time.time()-t_slot:.0f}s)")
            else:
                failed.append({"patient_id": r["patient_id"],
                               "reason": result["reason"]})
                wp.log_msg(f"  [{idx}/{len(roster)}] FAIL {r['patient_id']}: "
                           f"{result['reason']}")
        except Exception as e:
            failed.append({"patient_id": r["patient_id"],
                           "reason": f"{type(e).__name__}: {e}"})
            wp.log_msg(f"  [{idx}/{len(roster)}] EXC {r['patient_id']}: {e}")

        if success_times:
            mean_s    = sum(success_times) / len(success_times)
            remaining = (len(roster) - idx) * mean_s
            postfix   = (f"{idx}/{len(roster)} ok={len(summary)} fail={len(failed)}  "
                         f"~{_fmt_eta(remaining)} left")
        else:
            postfix = f"{idx}/{len(roster)} ok={len(summary)} fail={len(failed)}"
        wp.update_stage(current=idx, postfix=postfix)

    keep_awake.__exit__(None, None, None)
    wp.finish_stage("done")

    wp.set_stage("summary", postfix="writing")
    n_succ = len([s for s in summary if not s.get("skipped")])
    n_plain_done = len([s for s in summary if s.get("population") == "plain"])
    n_hilly_done = len([s for s in summary if s.get("population") == "hilly"])
    summary_path = out_root / "roster_summary.json"
    summary_path.write_text(json.dumps({
        "version":       "v3",
        "n_planned":     len(roster),
        "n_succeeded":   n_succ,
        "n_plain":       n_plain_done,
        "n_hilly":       n_hilly_done,
        "n_failed":      len(failed),
        "grade_distribution": {
            f"grade{g}": len([s for s in summary if s.get("grade") == g])
            for g in range(1, 5)
        },
        "plain_template": TEMPLATE_SOURCES["plain"]["uid"],
        "hilly_template": TEMPLATE_SOURCES["hilly"]["uid"],
        "patients":  summary,
        "failures":  failed,
    }, indent=2))
    progress_marker.unlink(missing_ok=True)
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"{n_succ} ok, {len(failed)} failed in {total_min:.1f} min")
    wp.finish_stage("done")
    print()
    print("=" * 72)
    print(f"  PFD v3 DONE in {total_min:.1f} min")
    print(f"  Succeeded:  {n_succ} / {len(roster)}")
    print(f"  Plain:      {n_plain_done}   Hilly: {n_hilly_done}")
    print(f"  Grade dist: {[(g, len([s for s in summary if s.get('grade')==g])) for g in range(1,5)]}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Summary:    {summary_path}")
    print("=" * 72)
    wp.stop_server(grace_s=30.0)


if __name__ == "__main__":
    main()
