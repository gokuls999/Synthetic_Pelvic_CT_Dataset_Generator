"""Phase 1 FILL: substitute the patients that failed in the original full run.

Reads `synthetic_dataset/_pfd_phase1_full/` to find which of the planned 50
patient slots are missing, then picks FRESH source volumes from the cache
(volumes NOT already in cache/masks/) and runs them through the same
TS -> mask-anchored deformation pipeline.

Why fresh volumes: the original failures came from source volumes where the
rectum subregion (or bladder) ended up with too few voxels in the pelvic crop
to anchor a clean deformation. Picking unused volumes gives a better chance of
clean masks. The mask cache means we never re-run TS on the previously-used
50 volumes -- they stay in the dataset where they DID work.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import json
import re
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
    segment_original_volume, rectum_subregion, mask_stats, PFD_ROI_SUBSET,
)
from src.pfd_deformation import (
    build_pattern_from_masks, build_displacement_field, apply_deformation,
)
from src import web_progress as wp


# Must match pfd_phase1_full.py
PATTERN_MIX = {
    ("combined_pfd",     "plain"): 15,
    ("combined_pfd",     "hilly"): 15,
    ("cystocele",        "plain"):  4,
    ("cystocele",        "hilly"):  4,
    ("rectocele",        "plain"):  3,
    ("rectocele",        "hilly"):  3,
    ("uterine_prolapse", "plain"):  3,
    ("uterine_prolapse", "hilly"):  3,
}
assert sum(PATTERN_MIX.values()) == 50


def planned_slots() -> list[tuple[int, str, str]]:
    """Return [(patient_num, pattern, severity)] in the same order
    pfd_phase1_full.py assigned them BEFORE the roster was shuffled."""
    slots = []
    n = 0
    for (pat, sev), count in PATTERN_MIX.items():
        for _ in range(count):
            n += 1
            slots.append((n, pat, sev))
    return slots


PID_RE = re.compile(r"^PFD_(\d{3})_(PLAIN|HILLY)_(.+)$")


def existing_patient_nums(out_root: Path) -> set[int]:
    nums = set()
    if not out_root.exists():
        return nums
    for d in out_root.iterdir():
        if not d.is_dir():
            continue
        m = PID_RE.match(d.name)
        if m and (d / "metadata.json").exists() and (d / "DICOM").is_dir():
            nums.add(int(m.group(1)))
    return nums


def already_used_uids(cache_root: Path) -> set[str]:
    """UIDs that already have a mask cache (= source volumes attempted)."""
    used = set()
    masks_root = cache_root / "masks"
    if not masks_root.exists():
        return used
    for ds_dir in masks_root.iterdir():
        if not ds_dir.is_dir():
            continue
        for uid_dir in ds_dir.iterdir():
            if uid_dir.is_dir():
                used.add(uid_dir.name)
    return used


def pick_fresh_volume(labels_df: pd.DataFrame, severity: str,
                      used_uids: set[str]) -> dict | None:
    sub = labels_df[(labels_df["region"] == severity)
                    & (~labels_df["uid"].astype(str).isin(used_uids))]
    if len(sub) == 0:
        return None
    # Sort by closeness to median pelvic_inlet_mm -- representative volumes
    med = sub["pelvic_inlet_mm"].median()
    sub = sub.assign(dist=(sub["pelvic_inlet_mm"] - med).abs())
    sub = sub.sort_values("dist")
    row = sub.iloc[0]
    return {
        "uid": str(row["uid"]),
        "dataset": str(row["dataset"]),
        "cache_path": str(row["cache_path"]),
        "region_id": int(row["region_id"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root = Path(cfg["paths"]["outputs_dir"]) / "_pfd_phase1_full"
    out_root.mkdir(parents=True, exist_ok=True)
    labels_csv = Path(cfg["paths"]["labels_csv"])
    labels_df = pd.read_csv(labels_csv)

    # ---- Figure out what's missing ----
    all_slots = planned_slots()
    have = existing_patient_nums(out_root)
    missing = [s for s in all_slots if s[0] not in have]

    if not missing:
        print("Nothing to do -- all 50 patients are on disk.")
        return

    # Establish a fresh-volume pool for each severity.
    used_uids = already_used_uids(cache_root)

    wp.set_stages([
        ("plan",    f"Identify missing patients ({len(missing)})"),
        ("process", f"Substitute + TS + deform ({len(missing)} patients)"),
        ("summary", "Update roster summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=f"PFD Phase-1 FILL  {len(missing)} substitutions",
        expected_total_s=len(missing) * 60.0, cfg=None,
    )

    print()
    print("=" * 72)
    print(f"  FILL dashboard:      {url}")
    print(f"  Output root:         {out_root}")
    print(f"  Missing patients:    {len(missing)}")
    print(f"  Already-used UIDs:   {len(used_uids)} (excluded from substitution pool)")
    print("=" * 72)
    print()

    wp.set_stage("plan", postfix=f"{len(missing)} slots to refill")
    for n, pat, sev in missing:
        wp.log_msg(f"  missing: PFD_{n:03d}_{sev.upper()}_{pat.upper()}  (pattern={pat}, sev={sev})")
    wp.finish_stage("done")

    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])
    spacing = (float(cfg["output"]["slice_thickness_mm"]),
               float(cfg["output"]["pixel_spacing_mm"][0]),
               float(cfg["output"]["pixel_spacing_mm"][1]))

    wp.set_stage("process", total=len(missing), postfix="starting")
    completed = []
    failed_again = []
    t_run = time.time()

    for idx, (patient_num, pattern, severity) in enumerate(missing, start=1):
        wp.update_stage(current=idx - 1, postfix=f"{idx}/{len(missing)} PFD_{patient_num:03d}")

        # Up to 5 attempts with progressively fresh picks (in case some new
        # volume also has empty rectum etc.)
        attempts = 0
        success = False
        while attempts < 5:
            attempts += 1
            pick = pick_fresh_volume(labels_df, severity, used_uids)
            if pick is None:
                wp.log_msg(f"  PFD_{patient_num:03d}: no fresh {severity} volumes left")
                break
            used_uids.add(pick["uid"])     # never re-try the same one
            wp.update_stage(postfix=f"{idx}/{len(missing)} PFD_{patient_num:03d} try{attempts} src={pick['uid']}")
            wp.log_msg(f"  [{idx}/{len(missing)}] try {attempts}: PFD_{patient_num:03d}  src={pick['uid']}")

            try:
                with np.load(pick["cache_path"], allow_pickle=True) as npz:
                    vol_norm = np.asarray(npz["slices"]).astype(np.float32)
                    src_path = Path(str(npz["source"]))
                Z, H, W = vol_norm.shape
                if not src_path.exists():
                    wp.log_msg(f"    skip: original DICOM missing at {src_path}")
                    continue

                def seg_progress(msg):
                    wp.update_stage(postfix=f"{idx}/{len(missing)} PFD_{patient_num:03d}: {msg}")

                masks, stats = segment_original_volume(
                    source_path=src_path, dataset=pick["dataset"], uid=pick["uid"],
                    cfg=cfg, cache_root=cache_root,
                    roi_subset=PFD_ROI_SUBSET,
                    on_progress=seg_progress,
                )
                if "colon" in masks and masks["colon"].any():
                    rectum_mask = rectum_subregion(masks["colon"], frac=0.66)
                    masks["rectum"] = rectum_mask
                    rs = mask_stats(rectum_mask, "rectum")
                    if rs is not None:
                        stats["rectum"] = rs

                need = {
                    "cystocele":        ["urinary_bladder"],
                    "rectocele":        ["rectum"],
                    "uterine_prolapse": ["urinary_bladder", "rectum"],
                    "combined_pfd":     ["urinary_bladder", "rectum"],
                }[pattern]
                miss = [n for n in need if n not in stats or stats[n].voxels < 200]
                if miss:
                    wp.log_msg(f"    guardrail fail: {miss} (bladder={stats.get('urinary_bladder', None) and stats['urinary_bladder'].voxels}, rectum={stats.get('rectum', None) and stats['rectum'].voxels})")
                    continue

                # Anchor + deform
                p_obj = build_pattern_from_masks(pattern, stats, severity=severity)
                field = build_displacement_field((Z, H, W), p_obj.blobs)
                deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)
                refined = deformed   # raw, no CVAE refine

                vol_hu = np.clip(unwindow(refined, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)
                pid = f"PFD_{patient_num:03d}_{severity.upper()}_{pattern.upper()}"
                vol = SyntheticVolume(
                    pixels_hu=vol_hu, spacing=spacing, region=severity,
                    region_id=pick["region_id"], patient_id=pid, seed=int(patient_num),
                )
                pdir = out_root / pid
                dicom_dir = pdir / "DICOM"
                png_dir   = pdir / "PNG"
                jpg_dir   = pdir / "JPG"
                meta_path = pdir / "metadata.json"
                write_dicom_series(vol, dicom_dir, cfg)
                write_png_jpg_metadata(vol, png_dir, jpg_dir, meta_path, cfg)

                def to_u8(slc):
                    return ((np.clip(slc, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
                z_picks = [int(Z * f) for f in np.linspace(0.30, 0.70, 3)]
                rows = []
                for z in z_picks:
                    rows.append(np.concatenate(
                        [to_u8(vol_norm[z]), to_u8(deformed[z]), to_u8(refined[z])], axis=1))
                Image.fromarray(np.concatenate(rows, axis=0), mode="L").save(pdir / "COMPARISON.png")

                meta = json.loads(meta_path.read_text())
                meta.update({
                    "synthetic_strategy": "hybrid_phase1_full_fill",
                    "cvae_refinement": False,
                    "source_volume_uid": pick["uid"],
                    "source_dataset":    pick["dataset"],
                    "pfd_pattern":       p_obj.name,
                    "pfd_severity":      p_obj.severity,
                    "pfd_findings":      p_obj.findings,
                    "anchor_mode":       "TS_on_original_DICOM",
                    "segmented_structures": list(stats.keys()),
                    "organ_voxels":      {k: int(v.voxels) for k, v in stats.items()},
                    "fill_attempt":      attempts,
                })
                meta_path.write_text(json.dumps(meta, indent=2))
                wp.log_msg(f"    OK  PFD_{patient_num:03d} on try {attempts}")
                completed.append({"patient_num": patient_num, "patient_id": pid,
                                  "pattern": pattern, "severity": severity,
                                  "src_uid": pick["uid"], "attempts": attempts,
                                  "findings": p_obj.findings})
                success = True
                break
            except Exception as e:
                wp.log_msg(f"    exception: {e}")
                continue

        if not success:
            failed_again.append((patient_num, pattern, severity))
            wp.log_msg(f"  PFD_{patient_num:03d} STILL FAILED after {attempts} attempts")
        wp.update_stage(current=idx, postfix=f"{idx}/{len(missing)} done; ok={len(completed)} skip={len(failed_again)}")

    wp.finish_stage("done")

    # ---- Summary ----
    wp.set_stage("summary", postfix="writing summary + merged roster")
    fill_path = out_root / "fill_summary.json"
    fill_path.write_text(json.dumps({
        "n_missing": len(missing),
        "n_filled":  len(completed),
        "n_still_failed": len(failed_again),
        "still_failed": [{"patient_num": n, "pattern": p, "severity": s}
                         for n, p, s in failed_again],
        "filled":    completed,
    }, indent=2))
    # Merged roster: count total patient dirs
    have_after = existing_patient_nums(out_root)
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"on disk now: {len(have_after)}/50  ({total_min:.1f} min)")
    wp.finish_stage("done")

    print()
    print("=" * 72)
    print(f"  FILL DONE in {total_min:.1f} min")
    print(f"  Filled: {len(completed)} / {len(missing)} missing slots")
    if failed_again:
        print(f"  STILL FAILED ({len(failed_again)}):")
        for n, p, s in failed_again:
            print(f"    PFD_{n:03d}_{s.upper()}_{p.upper()}  pattern={p}  severity={s}")
    print(f"  Patients on disk now: {len(have_after)} / 50")
    print(f"  Fill summary: {fill_path}")
    print("=" * 72)
    wp.stop_server(grace_s=60.0)


if __name__ == "__main__":
    main()
