"""Run TotalSegmentator anatomy validation on the 50 hybrid PFD patients.

Audits each generated DICOM volume for the presence and bilateral symmetry
of sacrum, both femoral heads, and both hip bones. Writes
synthetic_dataset/_pfd_phase1_full/anatomy_report.json with per-patient
verdicts and an aggregate summary.

Dashboard at port 8766 shows per-patient progress live.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import json
import time
from pathlib import Path

from src.anatomy_validator import (
    validate_volume_anatomy, reference_inlet_stats,
    REQUIRED_STRUCTURES, SOFT_STRUCTURES, MIN_VOXELS,
    SYMMETRY_THRESHOLD, INLET_Z_THRESHOLD,
)
from src import web_progress as wp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target", default="synthetic_dataset/_pfd_phase1_full",
                    help="Directory containing patient subdirs with DICOM/ subfolders")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = Path(args.target)
    if not target.is_dir():
        raise SystemExit(f"target not found: {target}")

    patient_dirs = sorted(p for p in target.iterdir()
                          if p.is_dir() and (p / "DICOM").is_dir())
    if not patient_dirs:
        raise SystemExit(f"no patient/DICOM dirs in {target}")

    # Reference distribution for soft inlet-width check
    ref_mean, ref_std = reference_inlet_stats(cfg["paths"]["labels_csv"])

    wp.set_stages([
        ("setup",   "Load reference distribution"),
        ("validate", f"TotalSegmentator + checks ({len(patient_dirs)} patients)"),
        ("summary", "Write report"),
    ])
    expected_s = len(patient_dirs) * 30.0   # ~30 s/patient on GPU at fast=True
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=f"Anatomy validation  {len(patient_dirs)} patients  target={target.name}",
        expected_total_s=expected_s, cfg=None,
    )
    print()
    print("=" * 72)
    print(f"  Anatomy validation dashboard: {url}")
    print(f"  Target:                       {target}")
    print(f"  Reference inlet (mm):         mean={ref_mean:.1f}  std={ref_std:.1f}")
    print("=" * 72)
    print()

    wp.set_stage("setup", postfix=f"ref inlet mean={ref_mean:.1f}mm std={ref_std:.1f}mm")
    wp.finish_stage("done")

    wp.set_stage("validate", total=len(patient_dirs), postfix="starting")
    reports = []
    n_ok = 0
    n_hard_fail = 0
    n_soft_fail = 0
    t_run = time.time()

    for i, pdir in enumerate(patient_dirs, start=1):
        wp.update_stage(current=i - 1, postfix=f"{i}/{len(patient_dirs)} {pdir.name}")
        t_p = time.time()
        try:
            rep = validate_volume_anatomy(
                pdir / "DICOM", ref_mean, ref_std, patient_id=pdir.name,
            )
            r = rep.__dict__
        except Exception as e:
            r = {"patient_id": pdir.name, "ok": False,
                 "issues": [f"validator error: {type(e).__name__}: {e}"],
                 "structures": {}, "symmetry": None,
                 "inlet_mm": None, "inlet_z": None}
        reports.append(r)
        if r["ok"]:
            n_ok += 1
            wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: OK")
        else:
            hard = any(s.startswith("missing required") for s in r["issues"])
            if hard:
                n_hard_fail += 1
                wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: HARD FAIL -- {r['issues']}")
            else:
                n_soft_fail += 1
                wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: soft issues {r['issues']}")
        dt = time.time() - t_p
        wp.update_stage(current=i,
                        postfix=f"{i}/{len(patient_dirs)} ok={n_ok} hardfail={n_hard_fail} ({dt:.0f}s/patient)")

    wp.finish_stage("done")

    wp.set_stage("summary", postfix="writing report")
    summary = {
        "n_patients": len(patient_dirs),
        "n_ok": n_ok,
        "n_hard_fail": n_hard_fail,
        "n_soft_fail": n_soft_fail,
        "reference_inlet_mm_mean": round(ref_mean, 1),
        "reference_inlet_mm_std":  round(ref_std, 1),
        "thresholds": {
            "min_voxels": MIN_VOXELS,
            "symmetry_threshold": SYMMETRY_THRESHOLD,
            "inlet_z_threshold": INLET_Z_THRESHOLD,
        },
        "required_structures": REQUIRED_STRUCTURES,
        "soft_structures": SOFT_STRUCTURES,
        "reports": reports,
    }
    report_path = target / "anatomy_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"{n_ok}/{len(patient_dirs)} ok  ({total_min:.1f} min)")
    wp.finish_stage("done")

    print()
    print("=" * 72)
    print(f"  ANATOMY VALIDATION DONE in {total_min:.1f} min")
    print(f"  Passed:  {n_ok} / {len(patient_dirs)}")
    print(f"  Hard fail: {n_hard_fail}")
    print(f"  Soft issues only: {n_soft_fail}")
    print(f"  Report: {report_path}")
    print("=" * 72)
    wp.stop_server(grace_s=60.0)


if __name__ == "__main__":
    main()
