"""PFD Production v4 — All-sources, anatomy-preserving dataset generator.

Key differences from v3:
  * Uses ALL 198 real source CTs (across all 6 datasets) instead of 2 templates.
    Each patient gets a different real pelvic anatomy — unique bones, organ
    positions, body habitus. Sources cycle with varied PFD deformation so even
    when a source repeats it looks like a different patient.
  * Same clinical POP-Q grading as v3 (grades 1-4 per South Indian literature).
  * Same per-patient augmentation: noise + contrast + brightness (seeded).
  * Same plain/hilly population split (175/175).
  * Power-cut resilient: .production_in_progress marker + --skip-done.
  * Live dashboard: http://127.0.0.1:8770/

Population assignment (plain vs hilly):
  Sources are deterministically assigned via a seeded shuffle so the
  morphological label is consistent across reruns.

Grade distribution (South Indian PFD prevalence):
  Grade 1 : 28%   Grade 2 : 38%   Grade 3 : 22%   Grade 4 : 12%

Pattern distribution:
  combined_pfd 60% | cystocele 16% | rectocele 12% | uterine_prolapse 12%
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


# ── Grade / pattern distribution (South Indian PFD literature) ────────────
GRADE_WEIGHTS    = [0.28, 0.38, 0.22, 0.12]
PATTERN_FRACTIONS = {
    "combined_pfd":     0.60,
    "cystocele":        0.16,
    "rectocele":        0.12,
    "uterine_prolapse": 0.12,
}


def _discover_sources(cache_root: Path) -> list[dict]:
    """Return all source CTs that have both a mask directory and a .npz file."""
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

    Sources are shuffled once then cycled independently for plain and hilly
    populations so both get wide variety. Within each population the roster
    covers all pattern × grade combinations proportionally.
    """
    rng = np.random.default_rng(seed)

    # Shuffle source pool then split for plain / hilly
    pool = sources.copy()
    rng.shuffle(pool)
    # Interleave: plain takes even indices, hilly takes odd
    plain_pool = [pool[i] for i in range(0, len(pool), 2)]
    hilly_pool = [pool[i] for i in range(1, len(pool), 2)]
    # Both pools cycle if population count > pool size
    def _cycle(lst, n):
        return [lst[i % len(lst)] for i in range(n)]
    plain_srcs = _cycle(plain_pool, n_plain)
    hilly_srcs = _cycle(hilly_pool, n_hilly)

    def _split_grades(n: int) -> list[int]:
        raw = [n * w for w in GRADE_WEIGHTS]
        floored = [int(np.floor(v)) for v in raw]
        deficit = n - sum(floored)
        fracs = [raw[i] - floored[i] for i in range(4)]
        for i in np.argsort(fracs)[::-1][:deficit]:
            floored[i] += 1
        return floored

    slots: list[dict] = []
    for pop, n_pop, srcs in (
        ("plain", n_plain, plain_srcs),
        ("hilly", n_hilly, hilly_srcs),
    ):
        src_iter = iter(srcs)
        for pattern, frac in PATTERN_FRACTIONS.items():
            n_pat = int(round(n_pop * frac))
            grade_counts = _split_grades(n_pat)
            for grade_idx, gc in enumerate(grade_counts):
                grade = grade_idx + 1
                for _ in range(gc):
                    src = next(src_iter)
                    slots.append({
                        "population": pop,
                        "pattern":    pattern,
                        "grade":      grade,
                        "uid":        src["uid"],
                        "dataset":    src["dataset"],
                    })

    # Trim / pad to exact total
    while len(slots) < n_total:
        s = slots[rng.integers(len(slots))].copy()
        slots.append(s)
    slots = slots[:n_total]

    rng.shuffle(slots)

    roster = []
    for i, s in enumerate(slots, start=1):
        pop_tag = "PLAIN" if s["population"] == "plain" else "HILLY"
        pat_tag = s["pattern"].upper().replace("_", "")
        roster.append({
            "patient_num": i,
            "patient_id":  f"PFD_{i:04d}_{pop_tag}_{pat_tag}_G{s['grade']}",
            **s,
        })
    return roster


def _augment_volume(vol: np.ndarray, patient_num: int) -> np.ndarray:
    rng = np.random.default_rng(patient_num * 999983 + 7)
    vol = vol.astype(np.float32)
    sigma = float(rng.uniform(0.005, 0.022))
    vol += rng.standard_normal(vol.shape).astype(np.float32) * sigma
    contrast = float(rng.uniform(0.90, 1.10))
    mu = float(vol.mean())
    vol = (vol - mu) * contrast + mu + float(rng.uniform(-0.04, 0.04))
    return np.clip(vol, -1.0, 1.0).astype(np.float32)


def process_one(cfg: dict, cache_root: Path, out_root: Path,
                patient_id: str, patient_num: int,
                pattern: str, population: str, grade: int,
                uid: str, dataset: str,
                hu_min: float, hu_max: float,
                on_progress=None) -> dict:

    # ── Load source cache volume ──────────────────────────────────────────
    npz_path = cache_root / dataset / f"{uid}.npz"
    if not npz_path.exists():
        return {"ok": False, "reason": f"cache npz missing: {npz_path}"}
    with np.load(npz_path, allow_pickle=True) as npz:
        vol_norm = np.asarray(npz["slices"]).astype(np.float32)
        src_path = Path(str(npz["source"]))
        spacing  = tuple(float(x) for x in np.asarray(npz["spacing"]).tolist())
    Z_orig, H, W = vol_norm.shape

    if not src_path.exists():
        return {"ok": False, "reason": f"original path missing: {src_path}"}

    # ── TotalSegmentator (cached) ─────────────────────────────────────────
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

    # ── X-center on bilateral hip midline ────────────────────────────────
    _hl = mask_stats(masks.get("hip_left"),  "hip_left")  if "hip_left"  in masks else None
    _hr = mask_stats(masks.get("hip_right"), "hip_right") if "hip_right" in masks else None
    if _hl is not None and _hr is not None:
        midline_x = (_hl.center[2] + _hr.center[2]) / 2.0
        offset_x  = int(round(W / 2.0 - midline_x))
        offset_x  = max(-32, min(32, offset_x))
        if abs(offset_x) >= 2:
            vol_norm = np.roll(vol_norm, offset_x, axis=2)
            masks    = {k: np.roll(m, offset_x, axis=2) for k, m in masks.items()}

    # ── Per-patient augmentation ──────────────────────────────────────────
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

    # ── Clinical PFD deformation ──────────────────────────────────────────
    p_obj    = build_pattern_graded(pattern, stats, grade=grade, population=population)
    field    = build_displacement_field((Z, H, W), p_obj.blobs)
    deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)
    vol_hu   = np.clip(unwindow(deformed, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)

    # ── Z resample to target spacing ─────────────────────────────────────
    sz_src, sy_src, sx_src = spacing
    target_z_mm = float(cfg["output"].get("slice_thickness_mm", 1.5))
    if abs(sz_src - target_z_mm) > 0.1:
        vol_hu  = _ndi.zoom(vol_hu.astype(np.float32),
                            (sz_src / target_z_mm, 1.0, 1.0), order=1).astype(np.int16)
        spacing = (target_z_mm, sy_src, sx_src)

    # ── Write outputs ─────────────────────────────────────────────────────
    vol_obj = SyntheticVolume(
        pixels_hu=vol_hu, spacing=spacing, region=population,
        region_id=(0 if population == "plain" else 1),
        patient_id=patient_id, seed=patient_num,
    )
    pdir = out_root / patient_id
    write_dicom_series(vol_obj, pdir / "DICOM", cfg)
    write_png_jpg_metadata(vol_obj, pdir / "PNG", pdir / "JPG", pdir / "metadata.json", cfg)

    # Comparison strip: template | deformed
    lo_u, hi_u = -200.0, 500.0
    def _win(a):
        return ((np.clip(a.astype(np.float32), lo_u, hi_u) - lo_u)
                / (hi_u - lo_u) * 255).astype(np.uint8)
    Z_f = vol_hu.shape[0]
    rows = []
    for z in [Z_f // 4, Z_f // 2, 3 * Z_f // 4]:
        oz = min(int(z * vol_norm.shape[0] / Z_f), vol_norm.shape[0] - 1)
        orig_u8 = ((np.clip(vol_norm[oz], -1, 1) + 1) / 2 * 255).astype(np.uint8)
        rows.append(np.concatenate([orig_u8, _win(vol_hu[z])], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(pdir / "COMPARISON.png")

    meta = {
        "patient_id":        patient_id,
        "population":        population,
        "pfd_pattern":       pattern,
        "pfd_grade":         grade,
        "pfd_severity":      ["", "mild", "moderate", "severe", "complete"][grade],
        "pfd_findings":      p_obj.findings,
        "source_uid":        uid,
        "source_dataset":    dataset,
        "pelvic_crop_z":     [int(z_top), int(z_bot), int(Z_orig)],
        "n_slices":          int(vol_hu.shape[0]),
        "spacing_mm":        list(spacing),
        "anchor_mode":       "TS_graded_v4",
        "synthetic":         True,
        "sex":               "Female",
        "modality":          "CT",
        "body_part":         "PELVIS",
    }
    (pdir / "metadata.json").write_text(json.dumps(meta, indent=2))

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

    return {"ok": True, "grade": grade, "pattern": pattern,
            "n_slices": int(vol_hu.shape[0]), "p_obj": p_obj}


def _fmt_eta(s: float) -> str:
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s/60)}m"
    return f"{s/3600:.1f}h"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config",       default="configs/default.yaml")
    ap.add_argument("--out",          default="synthetic_dataset/production_v4_350")
    ap.add_argument("--num-patients", type=int, default=350)
    ap.add_argument("--n-plain",      type=int, default=175)
    ap.add_argument("--n-hilly",      type=int, default=175)
    ap.add_argument("--port",         type=int, default=8770)
    ap.add_argument("--no-browser",   action="store_true")
    ap.add_argument("--skip-done",    action="store_true")
    ap.add_argument("--allow-sleep",  action="store_true")
    args = ap.parse_args()

    cfg        = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root   = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])

    sources = _discover_sources(cache_root)
    if not sources:
        raise RuntimeError(f"No cached source CTs found under {cache_root}")

    wp.set_stages([
        ("plan",    "Plan roster"),
        ("process", "Per-patient: load real CT -> centre -> augment -> deform -> DICOM"),
        ("summary", "Summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=(f"PFD v4 All-sources  N={args.num_patients}  "
                   f"plain={args.n_plain}/hilly={args.n_hilly}  "
                   f"sources={len(sources)}"),
        expected_total_s=None, cfg=None,
    )
    print()
    print("=" * 72)
    print(f"  PFD Production v4  —  All-source anatomy-preserving")
    print(f"  Dashboard : {url}")
    print(f"  Output    : {out_root}")
    print(f"  Sources   : {len(sources)} real pelvic CTs across {len(set(s['dataset'] for s in sources))} datasets")
    print(f"  Patients  : {args.num_patients} ({args.n_plain} plain / {args.n_hilly} hilly)")
    print("=" * 72)
    print()

    wp.set_stage("plan", postfix="building roster")
    roster = _plan_roster(args.num_patients, args.n_plain, args.n_hilly, sources)

    src_dist = {}
    for r in roster:
        src_dist[r["dataset"]] = src_dist.get(r["dataset"], 0) + 1
    wp.log_msg("Source distribution:")
    for k, v in sorted(src_dist.items()):
        wp.log_msg(f"  {k}: {v}")
    wp.finish_stage(f"{len(roster)} slots planned from {len(sources)} real sources")

    progress_marker = out_root / ".production_in_progress"
    progress_marker.write_text(json.dumps({
        "n_planned": len(roster), "version": "v4",
        "started_at": time.time(), "out_root": str(out_root),
    }))

    wp.set_stage("process", total=len(roster), postfix="starting")
    keep_awake = KeepAwake() if not args.allow_sleep else _NullCtx()
    keep_awake.__enter__()

    success_times: deque = deque(maxlen=30)
    summary, failed = [], []
    t_run = time.time()

    for idx, r in enumerate(roster, start=1):
        pdir = out_root / r["patient_id"]

        if args.skip_done and (pdir / "DICOM").is_dir() and (pdir / "metadata.json").exists():
            summary.append({"patient_id": r["patient_id"], "skipped": True})
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(roster)} skipped")
            continue

        wp.update_stage(current=idx - 1,
                        postfix=(f"{idx}/{len(roster)} {r['patient_id']} "
                                 f"G{r['grade']} {r['pattern']} "
                                 f"src={r['uid'][:28]}"))
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
                    "patient_id": r["patient_id"],
                    "population": r["population"],
                    "pattern":    r["pattern"],
                    "grade":      r["grade"],
                    "source_uid": r["uid"],
                    "source_ds":  r["dataset"],
                    "n_slices":   result["n_slices"],
                    "findings":   result["p_obj"].findings,
                })
                wp.log_msg(f"  [{idx}/{len(roster)}] OK  {r['patient_id']}  "
                           f"G{r['grade']} {r['pattern']}  "
                           f"({time.time()-t_slot:.0f}s)  src={r['uid'][:20]}")
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
    summary_path = out_root / "roster_summary.json"
    summary_path.write_text(json.dumps({
        "version":     "v4",
        "n_planned":   len(roster),
        "n_succeeded": n_succ,
        "n_plain":     len([s for s in summary if s.get("population") == "plain"]),
        "n_hilly":     len([s for s in summary if s.get("population") == "hilly"]),
        "n_failed":    len(failed),
        "n_sources":   len(sources),
        "source_datasets": list(src_dist.keys()),
        "grade_distribution": {
            f"grade{g}": len([s for s in summary if s.get("grade") == g])
            for g in range(1, 5)
        },
        "patients": summary,
        "failures": failed,
    }, indent=2))
    progress_marker.unlink(missing_ok=True)
    total_min = (time.time() - t_run) / 60.0

    wp.update_stage(postfix=f"{n_succ} ok, {len(failed)} failed in {total_min:.1f} min")
    wp.finish_stage("done")
    print()
    print("=" * 72)
    print(f"  PFD v4 DONE  {total_min:.1f} min")
    print(f"  Succeeded : {n_succ} / {len(roster)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Summary   : {summary_path}")
    print("=" * 72)
    wp.stop_server(grace_s=30.0)


if __name__ == "__main__":
    main()
