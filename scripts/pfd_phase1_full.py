"""Phase 1 FULL gallery: 50 synthetic PFD patients with REAL anatomy.

For each patient:
  1. Pick a real cached volume (50 unique volumes; 25 plain + 25 hilly).
  2. Load original DICOM, run TotalSegmentator on the FULL volume.
  3. Map masks back to the cache's (Z, H, W) grid via the same crop+resample
     preprocessing applied to the cached volume.
  4. Build a PFD deformation field anchored to the actual bladder / rectum
     centroids (and a midpoint for the uterus, which TS doesn't segment).
  5. Apply the deformation to the cached volume.
  6. Write DICOM + PNG + JPG + metadata + comparison.png.

Pattern mix (50 patients total):
  30 combined_pfd       (15 plain + 15 hilly)   -- multi-organ presentations
   8 cystocele           (4 plain + 4 hilly)
   6 rectocele           (3 plain + 3 hilly)
   6 uterine_prolapse    (3 plain + 3 hilly)

GPU is used for TotalSegmentator only (training has finished, so the 1080 Ti
is idle). Estimated runtime: 30-45 minutes for the first run (cache builds
masks on disk under cache/masks/<dataset>/<uid>/). Subsequent re-runs with
--skip-done are instant for already-built patients.

Output: synthetic_dataset/_pfd_phase1_full/PFD_NNN_<REGION>_<PATTERN>/
Dashboard: http://127.0.0.1:8766/
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


# Pattern mix for 50 patients
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
# Sanity: must sum to 50
assert sum(PATTERN_MIX.values()) == 50, "pattern mix does not sum to 50"


def build_roster(labels_csv: Path, seed: int = 2026) -> list[dict]:
    """Plan 50 patients. Each row in the labels CSV is used at most ONCE
    so we have 50 distinct real source volumes."""
    df = pd.read_csv(labels_csv)
    rng = np.random.default_rng(seed)

    plain_pool = df[df["region"] == "plain"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    hilly_pool = df[df["region"] == "hilly"].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    n_plain_needed = sum(n for (p, s), n in PATTERN_MIX.items() if s == "plain")
    n_hilly_needed = sum(n for (p, s), n in PATTERN_MIX.items() if s == "hilly")

    if len(plain_pool) < n_plain_needed or len(hilly_pool) < n_hilly_needed:
        raise RuntimeError(
            f"not enough volumes: need {n_plain_needed} plain / {n_hilly_needed} hilly, "
            f"have {len(plain_pool)} / {len(hilly_pool)}"
        )

    roster = []
    p_idx = 0
    h_idx = 0
    pid_num = 0
    for (pattern, severity), n in PATTERN_MIX.items():
        for _ in range(n):
            pid_num += 1
            if severity == "plain":
                row = plain_pool.iloc[p_idx]
                p_idx += 1
            else:
                row = hilly_pool.iloc[h_idx]
                h_idx += 1
            roster.append({
                "patient_num": pid_num,
                "patient_id": f"PFD_{pid_num:03d}_{severity.upper()}_{pattern.upper()}",
                "pattern": pattern,
                "severity": severity,
                "uid": str(row["uid"]),
                "dataset": str(row["dataset"]),
                "cache_path": str(row["cache_path"]),
                "region_id": int(row["region_id"]),
            })
    # Shuffle so plain/hilly patients interleave (less boring to watch live).
    rng.shuffle(roster)
    return roster


STAGES = [
    ("plan",      "Plan roster"),
    ("segment",   "TotalSegmentator (50 source volumes)"),
    ("generate",  "Apply deformation + write DICOM"),
    ("finish",    "Summary"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="TotalSegmentator device (training is done; GPU is free)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root = Path(cfg["paths"]["outputs_dir"]) / "_pfd_phase1_full"
    out_root.mkdir(parents=True, exist_ok=True)
    labels_csv = Path(cfg["paths"]["labels_csv"])

    wp.set_stages(STAGES)
    expected_s = 50 * (40 if args.device == "cuda" else 600)
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=f"PFD Phase-1 FULL  50 patients  TS device={args.device}",
        expected_total_s=expected_s, cfg=None,
    )

    print()
    print("=" * 72)
    print(f"  Phase-1 FULL dashboard:  {url}")
    print(f"  Output root:             {out_root}")
    print(f"  Mask cache:              {cache_root / 'masks'}")
    print(f"  TS device:               {args.device}")
    print("=" * 72)
    print()

    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])
    spacing = (float(cfg["output"]["slice_thickness_mm"]),
               float(cfg["output"]["pixel_spacing_mm"][0]),
               float(cfg["output"]["pixel_spacing_mm"][1]))

    # ---- Stage 1: plan ----
    wp.set_stage("plan", postfix="building 50-patient roster")
    roster = build_roster(labels_csv)
    wp.update_stage(postfix=f"{len(roster)} patients planned "
                            f"({sum(1 for r in roster if r['severity']=='plain')} plain / "
                            f"{sum(1 for r in roster if r['severity']=='hilly')} hilly)")
    for r in roster:
        wp.log_msg(f"  plan: {r['patient_id']:32s} pattern={r['pattern']:18s} src={r['uid']}")
    wp.finish_stage("done")

    # If skip-done is set, filter the roster
    if args.skip_done:
        filtered = []
        skipped = []
        for r in roster:
            pdir = out_root / r["patient_id"]
            if (pdir / "DICOM").is_dir() and (pdir / "metadata.json").is_file():
                skipped.append(r)
            else:
                filtered.append(r)
        if skipped:
            wp.log_msg(f"--skip-done: skipping {len(skipped)} already-complete patients")
        roster = filtered

    # ---- Stage 2 + 3 fused: segment + generate per-patient ----
    # Show two progress bars: TS (per-patient) and deformation+write (per-patient).
    # In practice we interleave them so each patient flows pick -> seg -> def -> write.
    wp.set_stage("segment", total=len(roster), postfix="starting")
    seg_done = 0
    gen_done = 0
    summary = []
    t_run = time.time()

    # We'll keep "segment" as the visible running stage during TS, then bump to
    # "generate" for the per-patient deformation/write block. To keep both bars
    # making sense, we accumulate both counters.
    def post(stage_id, current, total, msg):
        wp.update_stage(current=current, total=total, postfix=msg)

    for idx, r in enumerate(roster, start=1):
        pdir = out_root / r["patient_id"]
        wp.log_msg(f"[{idx}/{len(roster)}] start {r['patient_id']}  pattern={r['pattern']} sev={r['severity']}")
        t_pat = time.time()

        # ----- TotalSegmentator -----
        try:
            wp.set_stage("segment", total=len(roster), postfix=f"{idx}/{len(roster)} {r['uid']}")
            wp.update_stage(current=seg_done)

            with np.load(r["cache_path"], allow_pickle=True) as npz:
                vol_norm = np.asarray(npz["slices"]).astype(np.float32)
                src_path = Path(str(npz["source"]))
            Z, H, W = vol_norm.shape

            if not src_path.exists():
                wp.log_msg(f"  [{idx}] MISS: original DICOM not at {src_path}")
                seg_done += 1
                continue

            def seg_progress(msg):
                wp.update_stage(postfix=f"{idx}/{len(roster)} {r['uid']}: {msg}")

            t_seg = time.time()
            masks, stats = segment_original_volume(
                source_path=src_path, dataset=r["dataset"], uid=r["uid"],
                cfg=cfg, cache_root=cache_root,
                roi_subset=PFD_ROI_SUBSET,
                on_progress=seg_progress,
            )
            seg_t = time.time() - t_seg

            # Derive rectum from lower portion of colon
            if "colon" in masks and masks["colon"].any():
                rectum_mask = rectum_subregion(masks["colon"], frac=0.66)
                masks["rectum"] = rectum_mask
                rs = mask_stats(rectum_mask, "rectum")
                if rs is not None:
                    stats["rectum"] = rs
            seg_done += 1
            wp.log_msg(f"  [{idx}] seg done in {seg_t:.0f}s "
                       f"(bladder={stats.get('urinary_bladder', None) and stats['urinary_bladder'].voxels}, "
                       f"rectum={stats.get('rectum', None) and stats['rectum'].voxels})")
        except Exception as e:
            wp.log_msg(f"  [{idx}] SEG FAIL: {e}")
            seg_done += 1
            continue

        # Guard rails for required organs
        need = {
            "cystocele":        ["urinary_bladder"],
            "rectocele":        ["rectum"],
            "uterine_prolapse": ["urinary_bladder", "rectum"],
            "combined_pfd":     ["urinary_bladder", "rectum"],
        }[r["pattern"]]
        missing = [n for n in need if n not in stats or stats[n].voxels < 200]
        if missing:
            wp.log_msg(f"  [{idx}] SKIP: missing/tiny masks for {missing}")
            continue

        # ----- Deformation + write -----
        wp.set_stage("generate", total=len(roster), postfix=f"{idx}/{len(roster)} {r['patient_id']}")
        wp.update_stage(current=gen_done)

        pattern = build_pattern_from_masks(r["pattern"], stats, severity=r["severity"])
        field = build_displacement_field((Z, H, W), pattern.blobs)
        deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)
        refined = deformed   # no CVAE refine (default Phase 1 -- sharp output)

        vol_hu = np.clip(unwindow(refined, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)
        vol = SyntheticVolume(
            pixels_hu=vol_hu, spacing=spacing, region=r["severity"],
            region_id=r["region_id"], patient_id=r["patient_id"], seed=int(r["patient_num"]),
        )
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

        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta.update({
            "synthetic_strategy": "hybrid_phase1_full",
            "cvae_refinement": False,
            "source_volume_uid": r["uid"],
            "source_dataset":    r["dataset"],
            "pfd_pattern":       pattern.name,
            "pfd_severity":      pattern.severity,
            "pfd_findings":      pattern.findings,
            "anchor_mode":       "TS_on_original_DICOM",
            "segmented_structures": list(stats.keys()),
            "organ_voxels":      {k: int(v.voxels) for k, v in stats.items()},
        })
        meta_path.write_text(json.dumps(meta, indent=2))

        gen_done += 1
        wp.update_stage(current=gen_done, postfix=f"{gen_done}/{len(roster)} done; last={r['patient_id']}")
        dt = time.time() - t_pat
        wp.log_msg(f"[{idx}/{len(roster)}] done {r['patient_id']}  ({dt:.0f}s)")
        summary.append({"patient_id": r["patient_id"], "pattern": pattern.name,
                        "severity": pattern.severity, "n_slices": int(Z),
                        "findings": pattern.findings,
                        "organ_voxels": {k: int(v.voxels) for k, v in stats.items()}})

    # ---- Stage 4: summary ----
    wp.set_stage("finish", postfix="writing roster summary")
    summary_path = out_root / "roster_summary.json"
    summary_path.write_text(json.dumps({
        "n_patients_done": len(summary),
        "n_patients_planned": len(roster),
        "patients": summary,
    }, indent=2))
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"{len(summary)} patients in {total_min:.1f} min")
    wp.finish_stage("done")
    wp.log_msg(f"PHASE 1 FULL complete: {len(summary)} patients in {total_min:.1f} min")

    print()
    print("=" * 72)
    print(f"  PHASE 1 FULL DONE in {total_min:.1f} min")
    print(f"  {len(summary)} / {len(roster)} patients written")
    print(f"  Output root: {out_root}")
    print(f"  Summary:     {summary_path}")
    print("=" * 72)
    wp.stop_server(grace_s=120.0)


if __name__ == "__main__":
    main()
