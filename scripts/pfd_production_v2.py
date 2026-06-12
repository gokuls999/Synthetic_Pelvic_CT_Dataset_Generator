"""Production-scale PFD dataset generator v2.

What's different from v1 (scripts/pfd_production.py):
  * Per-slot retry: when a source fails TS or guard rails, the script
    auto-substitutes from the same-severity pool (up to 3 attempts per
    slot) before recording the slot as failed.
  * Session blocklist: a source that fails in this session is never tried
    again -- not for the same slot nor for any other slot. This caps wasted
    TS time at ~5 min per bad source.
  * REALISTIC ETA: the dashboard shows mean-time-per-SUCCESSFUL-patient
    multiplied by remaining slots, not the misleading per-attempt rate
    that v1 displayed. Recomputed after every patient.
  * Default count: 350 patients (175 plain + 175 hilly), 60/16/12/12 pattern
    mix kept.

Power-resilience guards retained:
  * KeepAwake context (powercfg standby 0 + SetThreadExecutionState)
  * `.production_in_progress` marker for autostart.ps1 to detect and resume
  * --skip-done picks up exactly where a previous attempt stopped
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import datetime
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
    build_pattern_from_masks, build_displacement_field, apply_deformation,
)
from src.keep_awake import KeepAwake
from src import web_progress as wp


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


PATTERN_FRACTIONS = {
    "combined_pfd":     0.60,
    "cystocele":        0.16,
    "rectocele":        0.12,
    "uterine_prolapse": 0.12,
}


def plan_counts(n_total: int, n_plain: int, n_hilly: int) -> dict:
    if n_plain + n_hilly != n_total:
        raise ValueError(f"plain ({n_plain}) + hilly ({n_hilly}) != total ({n_total})")
    out = {}
    for severity, target in (("plain", n_plain), ("hilly", n_hilly)):
        raw = {p: target * f for p, f in PATTERN_FRACTIONS.items()}
        floor = {p: int(np.floor(v)) for p, v in raw.items()}
        deficit = target - sum(floor.values())
        order = sorted(raw.items(), key=lambda kv: (raw[kv[0]] - floor[kv[0]]),
                       reverse=True)
        for i in range(deficit):
            p = order[i % len(order)][0]
            floor[p] += 1
        for p in PATTERN_FRACTIONS:
            out[(p, severity)] = floor[p]
    return out


def _filter_sources_by_mask_cache(df: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    """Drop sources whose cached masks already show TS failure (bladder OR
    colon below 200 voxels)."""
    masks_root = cache_root / "masks"
    if not masks_root.exists():
        return df

    bad_keys = set()
    for _, row in df.iterrows():
        uid = str(row["uid"])
        ds = str(row["dataset"])
        bladder_path = masks_root / ds / uid / "urinary_bladder.npy"
        colon_path = masks_root / ds / uid / "colon.npy"
        if not bladder_path.exists():
            continue
        bladder_voxels = int(np.load(bladder_path).sum())
        colon_voxels = int(np.load(colon_path).sum()) if colon_path.exists() else 0
        if bladder_voxels < 200 or colon_voxels < 200:
            bad_keys.add((ds, uid))
    if not bad_keys:
        return df
    mask = ~df.apply(lambda r: (str(r["dataset"]), str(r["uid"])) in bad_keys, axis=1)
    return df[mask].reset_index(drop=True)


def build_roster(n_total: int, n_plain: int, n_hilly: int,
                 labels_csv: Path, cache_root: Path,
                 seed: int = 2026) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Plan roster + return per-severity pools for substitution."""
    df = pd.read_csv(labels_csv)
    df = _filter_sources_by_mask_cache(df, cache_root)
    rng = np.random.default_rng(seed)

    pools = {
        "plain": df[df["region"] == "plain"].sample(frac=1.0, random_state=seed).reset_index(drop=True),
        "hilly": df[df["region"] == "hilly"].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True),
    }
    counts = plan_counts(n_total, n_plain, n_hilly)

    roster = []
    used_combos = set()
    pid = 0
    for (pattern, severity), n in counts.items():
        pool = pools[severity]
        if len(pool) == 0:
            raise RuntimeError(f"no '{severity}' source volumes")
        cursor = 0
        for _ in range(n):
            pid += 1
            attempts = 0
            row = None
            while attempts < len(pool):
                cand = pool.iloc[cursor % len(pool)]
                cursor += 1
                key = (str(cand["uid"]), pattern, severity)
                if key not in used_combos:
                    used_combos.add(key)
                    row = cand
                    break
                attempts += 1
            if row is None:
                row = pool.iloc[cursor % len(pool)]
                cursor += 1
            roster.append({
                "patient_num": pid,
                "patient_id": f"PFD_{pid:04d}_{severity.upper()}_{pattern.upper()}",
                "pattern": pattern,
                "severity": severity,
                "uid": str(row["uid"]),
                "dataset": str(row["dataset"]),
                "cache_path": str(row["cache_path"]),
                "region_id": int(row["region_id"]),
            })
    rng.shuffle(roster)
    return roster, pools


def _window_to_uint8(slice_hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = np.clip(slice_hu.astype(np.float32), lo, hi)
    s = (s - lo) / max(hi - lo, 1e-6)
    return (s * 255.0).astype(np.uint8)


def process_one(cfg: dict, cache_root: Path, out_root: Path,
                patient_id: str, patient_num: int, pattern: str, severity: str,
                region_id: int, uid: str, dataset: str, cache_path: Path,
                hu_min: float, hu_max: float, on_progress=None) -> dict:
    """Run TS + crop + deform + DICOM-write for one patient. Returns
    dict with 'ok', 'reason' (if failed), 'p_obj', 'stats', 'spacing',
    'pelvic_crop_z', 'n_slices'."""
    with np.load(cache_path, allow_pickle=True) as npz:
        vol_norm = np.asarray(npz["slices"]).astype(np.float32)
        src_path = Path(str(npz["source"]))
        spacing = tuple(float(x) for x in np.asarray(npz["spacing"]).tolist())
    Z_orig, H, W = vol_norm.shape

    if not src_path.exists():
        return {"ok": False, "reason": f"no original DICOM at {src_path}"}

    masks, stats = segment_original_volume(
        source_path=src_path, dataset=dataset, uid=uid,
        cfg=cfg, cache_root=cache_root, roi_subset=PFD_ROI_SUBSET,
        on_progress=on_progress,
    )

    z_top, z_bot = pelvic_z_range_from_masks(masks, margin_voxels=6,
                                              fallback=(0, Z_orig))
    if z_bot - z_top < Z_orig:
        vol_norm = vol_norm[z_top:z_bot]
        masks = {k: m[z_top:z_bot] for k, m in masks.items()}
    Z, H, W = vol_norm.shape

    # --- Fix: center pelvis on bilateral X midline -------------------------
    # Source CTs often have the patient off-center by up to ±30 px; the cache
    # preserves that offset, making left/right hip bones appear asymmetric in
    # the final 256×256 slices.  Rolling vol_norm and masks in X BEFORE stats
    # and deformation ensures blob anchors are symmetric around the midline.
    _hl = mask_stats(masks["hip_left"],  "hip_left")  if "hip_left"  in masks else None
    _hr = mask_stats(masks["hip_right"], "hip_right") if "hip_right" in masks else None
    if _hl is not None and _hr is not None:
        midline_x = (_hl.center[2] + _hr.center[2]) / 2.0
        offset_x  = int(round(W / 2.0 - midline_x))
        offset_x  = max(-32, min(32, offset_x))   # cap at 32 px safety
        if abs(offset_x) >= 2:
            vol_norm = np.roll(vol_norm, offset_x, axis=2)
            masks    = {k: np.roll(m, offset_x, axis=2) for k, m in masks.items()}

    if "colon" in masks and masks["colon"].any():
        rectum_mask = rectum_subregion(masks["colon"], frac=0.66)
        masks["rectum"] = rectum_mask
    stats = {}
    for name, m in masks.items():
        st = mask_stats(m, name)
        if st is not None:
            stats[name] = st

    need = {
        "cystocele":        ["urinary_bladder"],
        "rectocele":        ["rectum"],
        "uterine_prolapse": ["urinary_bladder", "rectum"],
        "combined_pfd":     ["urinary_bladder", "rectum"],
    }[pattern]
    missing = [n for n in need if n not in stats or stats[n].voxels < 200]
    if missing:
        return {"ok": False, "reason": f"missing {missing}"}

    p_obj = build_pattern_from_masks(pattern, stats, severity=severity)
    field = build_displacement_field((Z, H, W), p_obj.blobs)
    deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)

    vol_hu = np.clip(unwindow(deformed, hu_min, hu_max),
                     -1024.0, 3071.0).astype(np.int16)

    # --- Fix elongated 3D: resample z to target slice thickness ---
    sz_src, sy_src, sx_src = spacing
    target_z_mm = float(cfg["output"].get("slice_thickness_mm", 1.5))
    if abs(sz_src - target_z_mm) > 0.1:
        zoom_z = sz_src / target_z_mm
        vol_hu = _ndi.zoom(vol_hu.astype(np.float32),
                           (zoom_z, 1.0, 1.0), order=1).astype(np.int16)
        spacing = (target_z_mm, sy_src, sx_src)

    vol = SyntheticVolume(
        pixels_hu=vol_hu, spacing=spacing, region=severity,
        region_id=region_id, patient_id=patient_id, seed=int(patient_num),
    )
    pdir = out_root / patient_id
    dicom_dir = pdir / "DICOM"
    png_dir   = pdir / "PNG"
    jpg_dir   = pdir / "JPG"
    meta_path = pdir / "metadata.json"
    if dicom_dir.exists():
        for f in dicom_dir.glob("*.dcm"):
            f.unlink()
    write_dicom_series(vol, dicom_dir, cfg)
    write_png_jpg_metadata(vol, png_dir, jpg_dir, meta_path, cfg)

    def to_u8(slc):
        return ((np.clip(slc, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    z_picks = [int(Z * f) for f in np.linspace(0.30, 0.70, 3)]
    rows = []
    for z in z_picks:
        rows.append(np.concatenate(
            [to_u8(vol_norm[z]), to_u8(deformed[z]), to_u8(deformed[z])], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0), mode="L").save(
        pdir / "COMPARISON.png")

    meta = json.loads(meta_path.read_text())
    meta.update({
        "synthetic_strategy":      "hybrid_production_v2",
        "cvae_refinement":         False,
        "source_volume_uid":       uid,
        "source_dataset":          dataset,
        "pfd_pattern":             p_obj.name,
        "pfd_severity":            p_obj.severity,
        "pfd_findings":            p_obj.findings,
        "anchor_mode":             "TS_on_original_DICOM",
        "segmented_structures":    list(stats.keys()),
        "organ_voxels":            {k: int(v.voxels) for k, v in stats.items()},
        "real_spacing_mm":         list(spacing),
        "pelvic_crop_applied":     True,
        "pelvic_crop_z":           [int(z_top), int(z_bot), int(Z_orig)],
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    # --- Generate full clinical metadata + PDF data collection sheet ---
    collection_date = datetime.date.today().isoformat()
    clinical = generate_clinical_metadata(
        patient_id=patient_id,
        patient_num=patient_num,
        pattern=p_obj.name,
        severity=p_obj.severity,
        region_id=region_id,
        pfd_findings=p_obj.findings,
        real_spacing_mm=list(spacing),
        pelvic_crop_z=[int(z_top), int(z_bot), int(Z_orig)],
        collection_date=collection_date,
    )
    clinical_path = pdir / "clinical_data.json"
    clinical_path.write_text(json.dumps(clinical, indent=2))
    write_patient_pdf(clinical, pdir / "patient_report.pdf")

    return {
        "ok": True, "p_obj": p_obj, "stats": stats, "spacing": spacing,
        "pelvic_crop_z": [int(z_top), int(z_bot), int(Z_orig)],
        "n_slices": int(Z),
    }


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds/60)}m"
    return f"{seconds/3600:.1f}h"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="synthetic_dataset/production_350")
    ap.add_argument("--num-patients", type=int, default=350)
    ap.add_argument("--n-plain", type=int, default=None)
    ap.add_argument("--n-hilly", type=int, default=None)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--allow-sleep", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Substitute sources per slot before giving up (default 3)")
    args = ap.parse_args()

    if args.n_plain is None and args.n_hilly is None:
        args.n_plain = args.num_patients // 2
        args.n_hilly = args.num_patients - args.n_plain
    elif args.n_plain is None:
        args.n_plain = args.num_patients - args.n_hilly
    elif args.n_hilly is None:
        args.n_hilly = args.num_patients - args.n_plain
    if args.n_plain + args.n_hilly != args.num_patients:
        raise SystemExit(
            f"plain ({args.n_plain}) + hilly ({args.n_hilly}) != total ({args.num_patients})"
        )

    cfg = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    labels_csv = Path(cfg["paths"]["labels_csv"])

    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])

    wp.set_stages([
        ("plan",     f"Plan roster ({args.num_patients} patients)"),
        ("process",  "Per-patient pipeline (TS -> crop -> deform -> DICOM)"),
        ("summary",  "Summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=(f"PFD Production v2  N={args.num_patients}  "
                   f"plain={args.n_plain}/hilly={args.n_hilly}  "
                   f"max-attempts={args.max_attempts}"),
        expected_total_s=None,                   # we update via postfix below
        cfg=None,
    )

    print()
    print("=" * 72)
    print(f"  Production v2 dashboard:  {url}")
    print(f"  Output root:              {out_root}")
    print(f"  Mask cache:               {cache_root / 'masks'}")
    print(f"  Total patients:           {args.num_patients}")
    print(f"  Plain / hilly:            {args.n_plain} / {args.n_hilly}")
    print(f"  Max attempts per slot:    {args.max_attempts}")
    print("=" * 72)
    print()

    wp.set_stage("plan", postfix="building roster + filtering bad sources")
    roster, pools = build_roster(args.num_patients, args.n_plain, args.n_hilly,
                                  labels_csv, cache_root)
    counts_by_bucket = {}
    for r in roster:
        k = f"{r['pattern']}_{r['severity']}"
        counts_by_bucket[k] = counts_by_bucket.get(k, 0) + 1
    wp.update_stage(postfix=(f"{len(roster)} slots planned; "
                             f"pool sizes plain={len(pools['plain'])} hilly={len(pools['hilly'])}"))
    for k, c in sorted(counts_by_bucket.items()):
        wp.log_msg(f"  bucket {k}: {c} patients")
    wp.finish_stage("done")

    progress_marker = out_root / ".production_in_progress"
    progress_marker.write_text(json.dumps({
        "n_planned": len(roster), "n_plain": args.n_plain,
        "n_hilly": args.n_hilly, "out_root": str(out_root),
        "started_at": time.time(), "version": "v2",
    }, indent=2))

    wp.set_stage("process", total=len(roster), postfix="starting")
    if not args.allow_sleep:
        wp.log_msg("keep-awake enabled (system sleep blocked during run)")
    keep_awake = KeepAwake() if not args.allow_sleep else _NullCtx()
    keep_awake.__enter__()

    # State for retry + realistic ETA
    session_blocklist: set[str] = set()          # uids known bad this session
    success_times: deque = deque(maxlen=30)      # rolling window of recent success durations
    pool_cursors = {"plain": 0, "hilly": 0}       # per-severity round-robin for substitutes
    summary = []
    failed = []
    t_run = time.time()

    def pick_substitute(severity: str, tried_uids: set[str]) -> dict | None:
        """Return a fresh source for substitution: not yet tried this slot,
        not in session blocklist."""
        pool = pools[severity]
        for _ in range(len(pool)):
            row = pool.iloc[pool_cursors[severity] % len(pool)]
            pool_cursors[severity] += 1
            uid = str(row["uid"])
            if uid in session_blocklist or uid in tried_uids:
                continue
            return {
                "uid": uid, "dataset": str(row["dataset"]),
                "cache_path": str(row["cache_path"]),
                "region_id": int(row["region_id"]),
            }
        return None

    for idx, r in enumerate(roster, start=1):
        pdir = out_root / r["patient_id"]

        if args.skip_done and (pdir / "DICOM").is_dir() and (pdir / "metadata.json").exists():
            summary.append({"patient_id": r["patient_id"], "skipped": True})
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(roster)} skipped (done)")
            continue

        wp.update_stage(current=idx - 1,
                        postfix=f"{idx}/{len(roster)} {r['patient_id']}")
        t_slot = time.time()
        tried_uids: set[str] = set()
        result = None
        last_error = "none"

        for attempt in range(args.max_attempts):
            # Pick the source for this attempt
            if attempt == 0:
                if r["uid"] in session_blocklist:
                    sub = pick_substitute(r["severity"], tried_uids)
                    if sub is None:
                        last_error = "no usable source for this severity"
                        break
                    src = sub
                else:
                    src = {"uid": r["uid"], "dataset": r["dataset"],
                           "cache_path": r["cache_path"], "region_id": r["region_id"]}
            else:
                sub = pick_substitute(r["severity"], tried_uids)
                if sub is None:
                    last_error = f"no more substitutes (attempt {attempt})"
                    break
                src = sub
            tried_uids.add(src["uid"])

            def _seg_progress(msg, _src=src):
                wp.update_stage(postfix=(f"{idx}/{len(roster)} {r['patient_id']} "
                                          f"try{attempt+1}/{args.max_attempts} {_src['uid']}: {msg}"))

            try:
                t_attempt = time.time()
                result = process_one(
                    cfg=cfg, cache_root=cache_root, out_root=out_root,
                    patient_id=r["patient_id"], patient_num=r["patient_num"],
                    pattern=r["pattern"], severity=r["severity"],
                    region_id=src["region_id"], uid=src["uid"],
                    dataset=src["dataset"], cache_path=Path(src["cache_path"]),
                    hu_min=hu_min, hu_max=hu_max, on_progress=_seg_progress,
                )
                if result["ok"]:
                    success_times.append(time.time() - t_slot)
                    summary.append({
                        "patient_id": r["patient_id"],
                        "pattern": result["p_obj"].name,
                        "severity": result["p_obj"].severity,
                        "src_uid": src["uid"], "attempts": attempt + 1,
                        "n_slices": result["n_slices"],
                        "findings": result["p_obj"].findings,
                    })
                    wp.log_msg(f"  [{idx}/{len(roster)}] OK  {r['patient_id']}  "
                               f"src={src['uid']}  attempt {attempt+1}  "
                               f"({(time.time()-t_attempt):.0f}s)")
                    break
                else:
                    last_error = result["reason"]
                    # Most failure reasons (missing organs) are intrinsic
                    # to this source -> blocklist it.
                    if "missing" in last_error:
                        session_blocklist.add(src["uid"])
                    wp.log_msg(f"  [{idx}/{len(roster)}] FAIL src={src['uid']} "
                               f"attempt {attempt+1}: {last_error}")
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                session_blocklist.add(src["uid"])
                wp.log_msg(f"  [{idx}/{len(roster)}] EXC src={src['uid']} "
                           f"attempt {attempt+1}: {last_error}")

        if not result or not result.get("ok"):
            failed.append({"patient_id": r["patient_id"],
                           "reason": last_error,
                           "tried_uids": sorted(tried_uids)})

        # Realistic ETA based on last 30 successful slots
        if success_times:
            mean_success_s = sum(success_times) / len(success_times)
            remaining = len(roster) - idx
            eta_s = remaining * mean_success_s
            postfix = (f"{idx}/{len(roster)} ok={len(summary)} fail={len(failed)}  "
                       f"~{_fmt_eta(eta_s)} left (real, n={len(success_times)})")
        else:
            postfix = f"{idx}/{len(roster)} ok={len(summary)} fail={len(failed)}  ETA: warming up"
        wp.update_stage(current=idx, postfix=postfix)

    wp.finish_stage("done")

    wp.set_stage("summary", postfix="writing roster summary")
    summary_path = out_root / "roster_summary.json"
    summary_path.write_text(json.dumps({
        "n_planned":         len(roster),
        "n_succeeded":       len([s for s in summary if not s.get("skipped")]),
        "n_skipped_done":    len([s for s in summary if s.get("skipped")]),
        "n_failed":          len(failed),
        "n_blocklisted":     len(session_blocklist),
        "blocklisted_uids":  sorted(session_blocklist),
        "patients":          summary,
        "failures":          failed,
    }, indent=2))
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"{len(summary)} ok, {len(failed)} failed in {total_min:.1f} min")
    wp.finish_stage("done")
    print()
    print("=" * 72)
    print(f"  PRODUCTION v2 DONE in {total_min:.1f} min")
    print(f"  Succeeded:    {len([s for s in summary if not s.get('skipped')])} / {len(roster)}")
    print(f"  Skipped done: {len([s for s in summary if s.get('skipped')])}")
    print(f"  Failed:       {len(failed)}")
    print(f"  Blocklisted:  {len(session_blocklist)}")
    print(f"  Summary:      {summary_path}")
    print("=" * 72)

    try:
        progress_marker.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        keep_awake.__exit__(None, None, None)
    except Exception:
        pass
    wp.stop_server(grace_s=120.0)


if __name__ == "__main__":
    main()
