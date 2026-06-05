"""Production-scale PFD dataset generator.

Builds N hybrid patients with REAL anatomy + parametric PFD deformation,
correct per-volume DICOM spacing, and TS-anchored pelvic cropping -- all
inline in one pass (no separate respacing step).

Source-volume reuse: when N exceeds the size of the source pool, the same
real volume is reused under a DIFFERENT (pattern, severity) tuple so every
patient is a unique combination of anatomy + disease pattern.

Default: 375 patients, half plain / half hilly, with the 60/16/12/12 pattern
mix the 50-patient pilot used.

CPU-bound for I/O, GPU-bound for TotalSegmentator. ~2-3 h on idle GPU for
fresh runs; instant on re-runs when masks are cached.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing import unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata
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


# ----- Default pattern mix (matches the 50-patient pilot proportions) -----

PATTERN_FRACTIONS = {
    "combined_pfd":     0.60,
    "cystocele":        0.16,
    "rectocele":        0.12,
    "uterine_prolapse": 0.12,
}


def plan_counts(n_total: int, n_plain: int, n_hilly: int) -> dict:
    """Return {(pattern, severity): count} for the roster. Counts are rounded
    so that the total is exactly n_plain + n_hilly and per-severity totals are
    exactly n_plain / n_hilly."""
    if n_plain + n_hilly != n_total:
        raise ValueError(f"plain ({n_plain}) + hilly ({n_hilly}) != total ({n_total})")

    out = {}
    # First: how many of each pattern within each severity bucket.
    for severity, target in (("plain", n_plain), ("hilly", n_hilly)):
        raw = {p: target * f for p, f in PATTERN_FRACTIONS.items()}
        # Floor each and distribute the remainder by largest fractional part.
        floor = {p: int(np.floor(v)) for p, v in raw.items()}
        deficit = target - sum(floor.values())
        # Order patterns by (raw - floor) descending so we add 1 to those with
        # the largest fractional parts until the bucket sums to target.
        order = sorted(raw.items(), key=lambda kv: (raw[kv[0]] - floor[kv[0]]),
                       reverse=True)
        for i in range(deficit):
            p = order[i % len(order)][0]
            floor[p] += 1
        for p in PATTERN_FRACTIONS:
            out[(p, severity)] = floor[p]
    return out


def _filter_sources_by_mask_cache(df: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    """Drop volumes that already have empty cached masks (TS failures from
    previous runs). Volumes never attempted are kept (we'll try them).
    Volumes with usable cached masks are kept (instant on re-use)."""
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
            continue       # never attempted -- keep
        bladder_voxels = int(np.load(bladder_path).sum())
        colon_voxels = int(np.load(colon_path).sum()) if colon_path.exists() else 0
        # Drop a source if EITHER organ TS needs for the dominant patterns
        # is unusable: combined_pfd / uterine_prolapse need bladder AND
        # rectum; cystocele needs bladder; rectocele needs rectum (derived
        # from colon). Without bladder AND colon we have no useful pattern.
        if bladder_voxels < 200 or colon_voxels < 200:
            bad_keys.add((ds, uid))

    if not bad_keys:
        return df

    mask = ~df.apply(lambda r: (str(r["dataset"]), str(r["uid"])) in bad_keys, axis=1)
    return df[mask].reset_index(drop=True)


def build_roster(n_total: int, n_plain: int, n_hilly: int,
                 labels_csv: Path, cache_root: Path,
                 seed: int = 2026) -> list[dict]:
    """Plan N patients. Same source can be reused under a different
    (pattern, severity) tuple so every patient is anatomy+pattern unique.
    Sources are chosen round-robin from a shuffled pool, with a uniqueness
    guard on (uid, pattern, severity).

    Sources with known-bad TS cached masks are filtered out so we don't
    waste slots on them."""
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
    # Iterate (pattern, severity) buckets, assigning sources round-robin.
    # Within a bucket we cycle through the shuffled severity pool.
    for (pattern, severity), n in counts.items():
        pool = pools[severity]
        if len(pool) == 0:
            raise RuntimeError(f"no '{severity}' source volumes")
        cursor = 0
        for _ in range(n):
            pid += 1
            # Find the next source that hasn't been paired with this
            # (pattern, severity) before. Cycles at most len(pool) times.
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
                # All sources already paired with this (pattern, severity).
                # Pick the least-used source (just take cursor position).
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
    # Shuffle so the dashboard shows interleaved patterns.
    rng.shuffle(roster)
    return roster


def _window_to_uint8(slice_hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = np.clip(slice_hu.astype(np.float32), lo, hi)
    s = (s - lo) / max(hi - lo, 1e-6)
    return (s * 255.0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="synthetic_dataset/production_375",
                    help="Output root directory")
    ap.add_argument("--num-patients", type=int, default=375)
    ap.add_argument("--n-plain", type=int, default=None,
                    help="Plain count (default: half of --num-patients)")
    ap.add_argument("--n-hilly", type=int, default=None,
                    help="Hilly count (default: the rest)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--allow-sleep", action="store_true",
                    help="Allow the OS to sleep during the run (default: prevent sleep "
                         "for resilience over the multi-hour run).")
    args = ap.parse_args()

    if args.n_plain is None and args.n_hilly is None:
        # Default: equal split. If odd, give extra to plain.
        args.n_plain = (args.num_patients + 1) // 2
        args.n_hilly = args.num_patients // 2
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
    lo, hi = cfg["output"]["png_window"]

    wp.set_stages([
        ("plan",     f"Plan roster ({args.num_patients} patients)"),
        ("process",  f"Per-patient pipeline (TS -> crop -> deform -> DICOM)"),
        ("summary",  "Summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=(f"PFD Production  N={args.num_patients}  "
                   f"plain={args.n_plain}/hilly={args.n_hilly}"),
        expected_total_s=args.num_patients * 25.0, cfg=None,
    )

    print()
    print("=" * 72)
    print(f"  Production dashboard:  {url}")
    print(f"  Output root:           {out_root}")
    print(f"  Mask cache:            {cache_root / 'masks'}")
    print(f"  Total patients:        {args.num_patients}")
    print(f"  Plain / hilly:         {args.n_plain} / {args.n_hilly}")
    print("=" * 72)
    print()

    # --- Plan ---
    wp.set_stage("plan", postfix="building roster")
    roster = build_roster(args.num_patients, args.n_plain, args.n_hilly,
                          labels_csv, cache_root)
    counts_by_bucket = {}
    for r in roster:
        k = f"{r['pattern']}_{r['severity']}"
        counts_by_bucket[k] = counts_by_bucket.get(k, 0) + 1
    wp.update_stage(postfix=f"{len(roster)} patient slots planned")
    for k, c in sorted(counts_by_bucket.items()):
        wp.log_msg(f"  bucket {k}: {c} patients")
    wp.finish_stage("done")

    # --- Mark "production in progress" so autostart.ps1 can resume on cut ---
    progress_marker = out_root / ".production_in_progress"
    progress_marker.write_text(json.dumps({
        "n_planned": len(roster), "n_plain": args.n_plain,
        "n_hilly": args.n_hilly, "out_root": str(out_root),
        "started_at": time.time(),
    }, indent=2))

    # --- Process (KeepAwake unless --allow-sleep) ---
    wp.set_stage("process", total=len(roster), postfix="starting")
    if not args.allow_sleep:
        wp.log_msg("keep-awake enabled (system sleep blocked during run)")
    keep_awake = KeepAwake() if not args.allow_sleep else _NullCtx()
    summary = []
    failed = []
    t_run = time.time()

    # Enter KeepAwake; plain enter/exit, cleanup at the end of main().
    keep_awake.__enter__()
    for idx, r in enumerate(roster, start=1):
        pdir = out_root / r["patient_id"]

        if args.skip_done and (pdir / "DICOM").is_dir() and (pdir / "metadata.json").exists():
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(roster)} skipped (already done)")
            summary.append({"patient_id": r["patient_id"], "skipped": True})
            continue

        wp.update_stage(current=idx - 1,
                        postfix=f"{idx}/{len(roster)} {r['patient_id']}")
        try:
            # Load cache (volume + real spacing)
            with np.load(r["cache_path"], allow_pickle=True) as npz:
                vol_norm = np.asarray(npz["slices"]).astype(np.float32)
                src_path = Path(str(npz["source"]))
                spacing = tuple(float(x)
                                for x in np.asarray(npz["spacing"]).tolist())
            Z_orig, H, W = vol_norm.shape

            if not src_path.exists():
                wp.log_msg(f"  [{idx}] MISS: original DICOM not at {src_path}")
                failed.append({"patient_id": r["patient_id"],
                               "reason": "no original DICOM"})
                continue

            # Segment via cached TS (instant if already cached, slow first time)
            def _seg_progress(msg):
                wp.update_stage(postfix=f"{idx}/{len(roster)} {r['patient_id']}: {msg}")

            masks, stats = segment_original_volume(
                source_path=src_path, dataset=r["dataset"], uid=r["uid"],
                cfg=cfg, cache_root=cache_root,
                roi_subset=PFD_ROI_SUBSET,
                on_progress=_seg_progress,
            )

            # TS-anchored pelvic crop
            z_top, z_bot = pelvic_z_range_from_masks(masks, margin_voxels=6,
                                                    fallback=(0, Z_orig))
            if z_bot - z_top < Z_orig:
                vol_norm = vol_norm[z_top:z_bot]
                masks = {k: m[z_top:z_bot] for k, m in masks.items()}
            Z, H, W = vol_norm.shape

            # Rectum subregion
            if "colon" in masks and masks["colon"].any():
                rectum_mask = rectum_subregion(masks["colon"], frac=0.66)
                masks["rectum"] = rectum_mask
            stats = {}
            for name, m in masks.items():
                st = mask_stats(m, name)
                if st is not None:
                    stats[name] = st

            # Guard rails
            need = {
                "cystocele":        ["urinary_bladder"],
                "rectocele":        ["rectum"],
                "uterine_prolapse": ["urinary_bladder", "rectum"],
                "combined_pfd":     ["urinary_bladder", "rectum"],
            }[r["pattern"]]
            missing = [n for n in need if n not in stats or stats[n].voxels < 200]
            if missing:
                failed.append({"patient_id": r["patient_id"],
                               "reason": f"missing {missing}",
                               "src_uid": r["uid"]})
                wp.log_msg(f"  [{idx}] SKIP {r['patient_id']}: missing {missing}")
                continue

            # Build + apply deformation
            p_obj = build_pattern_from_masks(r["pattern"], stats,
                                             severity=r["severity"])
            field = build_displacement_field((Z, H, W), p_obj.blobs)
            deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)

            # HU + write
            vol_hu = np.clip(unwindow(deformed, hu_min, hu_max),
                             -1024.0, 3071.0).astype(np.int16)
            vol = SyntheticVolume(
                pixels_hu=vol_hu, spacing=spacing, region=r["severity"],
                region_id=r["region_id"], patient_id=r["patient_id"],
                seed=int(r["patient_num"]),
            )
            dicom_dir = pdir / "DICOM"
            png_dir   = pdir / "PNG"
            jpg_dir   = pdir / "JPG"
            meta_path = pdir / "metadata.json"

            # Clean stale outputs if any
            if dicom_dir.exists():
                for f in dicom_dir.glob("*.dcm"):
                    f.unlink()
            write_dicom_series(vol, dicom_dir, cfg)
            write_png_jpg_metadata(vol, png_dir, jpg_dir, meta_path, cfg)

            # Comparison png at 3 z slices
            def to_u8(slc):
                return ((np.clip(slc, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
            z_picks = [int(Z * f) for f in np.linspace(0.30, 0.70, 3)]
            rows = []
            for z in z_picks:
                rows.append(np.concatenate(
                    [to_u8(vol_norm[z]), to_u8(deformed[z]), to_u8(deformed[z])],
                    axis=1))
            Image.fromarray(np.concatenate(rows, axis=0), mode="L").save(
                pdir / "COMPARISON.png")

            # Augment metadata with PFD fields
            meta = json.loads(meta_path.read_text())
            meta.update({
                "synthetic_strategy": "hybrid_production_v1",
                "cvae_refinement":    False,
                "source_volume_uid":  r["uid"],
                "source_dataset":     r["dataset"],
                "pfd_pattern":        p_obj.name,
                "pfd_severity":       p_obj.severity,
                "pfd_findings":       p_obj.findings,
                "anchor_mode":        "TS_on_original_DICOM",
                "segmented_structures": list(stats.keys()),
                "organ_voxels":       {k: int(v.voxels) for k, v in stats.items()},
                "real_spacing_mm":    list(spacing),
                "pelvic_crop_applied": True,
                "pelvic_crop_z":      [int(z_top), int(z_bot), int(Z_orig)],
            })
            meta_path.write_text(json.dumps(meta, indent=2))

            summary.append({
                "patient_id": r["patient_id"], "pattern": p_obj.name,
                "severity": p_obj.severity, "src_uid": r["uid"],
                "n_slices": int(Z), "findings": p_obj.findings,
            })
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(roster)} ok={len(summary)} fail={len(failed)}")
        except Exception as e:
            failed.append({"patient_id": r["patient_id"],
                           "reason": f"{type(e).__name__}: {e}",
                           "src_uid": r["uid"]})
            wp.log_msg(f"  [{idx}] EXCEPTION {r['patient_id']}: {e}")
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(roster)} ok={len(summary)} fail={len(failed)}")

    wp.finish_stage("done")

    # --- Summary ---
    wp.set_stage("summary", postfix="writing roster summary")
    summary_path = out_root / "roster_summary.json"
    summary_path.write_text(json.dumps({
        "n_planned":     len(roster),
        "n_succeeded":   len([s for s in summary if not s.get("skipped")]),
        "n_skipped_done": len([s for s in summary if s.get("skipped")]),
        "n_failed":      len(failed),
        "patients":      summary,
        "failures":      failed,
    }, indent=2))
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"{len(summary)} ok, {len(failed)} failed in {total_min:.1f} min")
    wp.finish_stage("done")
    print()
    print("=" * 72)
    print(f"  PRODUCTION DONE in {total_min:.1f} min")
    print(f"  Succeeded:   {len([s for s in summary if not s.get('skipped')])} / {len(roster)}")
    print(f"  Skipped (already done): {len([s for s in summary if s.get('skipped')])}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Output root: {out_root}")
    print(f"  Summary:     {summary_path}")
    print("=" * 72)
    # Mark run as complete; autostart.ps1 watches this marker to know when
    # production is done so it stops resuming on subsequent boots.
    try:
        progress_marker.unlink(missing_ok=True)
    except Exception:
        pass
    # KeepAwake cleanup. (No try/finally because Python keep_awake exit
    # is best-effort; if the process is power-killed, the next KeepAwake
    # will re-zero the standby settings anyway.)
    try:
        keep_awake.__exit__(None, None, None)
    except Exception:
        pass
    wp.stop_server(grace_s=120.0)


if __name__ == "__main__":
    main()
