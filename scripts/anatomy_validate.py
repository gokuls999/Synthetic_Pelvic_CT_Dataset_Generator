"""Run anatomy validation on the synthetic pelvic CT dataset.

Requires TotalSegmentator:
    pip install totalsegmentator
(First call downloads ~2 GB of nnUNet weights into the user cache. Uses GPU
if available, falls back to CPU; ~10-30 s/volume on a GTX 1080 Ti.)

Live progress dashboard opens automatically at http://127.0.0.1:8765/ with
one card showing patients processed and per-volume verdicts in the log.

Per patient, checks:
  HARD (any fail -> volume rejected by report):
    * sacrum, both femoral heads, both hip bones present (>= 1000 voxels)
  SOFT (logged in report, don't reject):
    * left-right hip symmetry >= 0.85
    * pelvic inlet within +/-2 sigma of the real-data distribution

Output: synthetic_dataset/anatomy_report.json + the dashboard summary.

Usage:
    python scripts\\anatomy_validate.py
    python scripts\\anatomy_validate.py --no-browser
"""

from _common import add_repo_to_path, base_parser, load_config
add_repo_to_path()

import torch                                          # noqa: E402

from pathlib import Path                              # noqa: E402
from src.anatomy_validator import (                   # noqa: E402
    validate_volume_anatomy, reference_inlet_stats,
)
from src import web_progress as wp                    # noqa: E402


def main():
    ap = base_parser("Anatomy validation via TotalSegmentator (with dashboard).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the dashboard in a browser")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_root = Path(cfg["paths"]["outputs_dir"])
    labels_csv = cfg["paths"]["labels_csv"]
    patient_dirs = sorted(p for p in out_root.iterdir()
                          if p.is_dir() and (p / "DICOM").is_dir())
    if not patient_dirs:
        print(f"[anatomy] no patients found under {out_root}/")
        return

    ref_mean, ref_std = reference_inlet_stats(labels_csv)

    label = (f"anatomy validation  device={'cuda' if torch.cuda.is_available() else 'cpu'}  "
             f"patients={len(patient_dirs)}  "
             f"ref inlet {ref_mean:.0f}+-{ref_std:.0f}mm")
    url = wp.start_server(port=args.port, open_browser=not args.no_browser,
                          run_label=label)
    print(f"\n{'=' * 72}\n  Dashboard: {url}\n{'=' * 72}\n")

    wp.set_stage("anatomy_validate", total=len(patient_dirs),
                 postfix="loading nnUNet models...")

    import json
    reports = []
    n_ok = 0
    n_soft_flag = 0
    for i, pdir in enumerate(patient_dirs, start=1):
        wp.update_stage(postfix=f"{pdir.name}: segmenting...")
        try:
            r = validate_volume_anatomy(pdir / "DICOM", ref_mean, ref_std,
                                        patient_id=pdir.name)
            r_dict = r.__dict__
        except Exception as e:
            r_dict = {"patient_id": pdir.name, "ok": False,
                      "issues": [f"validator error: {type(e).__name__}: {e}"],
                      "structures": {}, "symmetry": None,
                      "inlet_mm": None, "inlet_z": None}
        reports.append(r_dict)
        if r_dict["ok"]:
            n_ok += 1
        if r_dict.get("issues") and r_dict["ok"]:
            n_soft_flag += 1

        verdict = "OK" if r_dict["ok"] else "FAIL"
        issues_str = ("; ".join(r_dict["issues"])[:80]) if r_dict["issues"] else "clean"
        wp.log_msg(f"[{verdict}] {pdir.name}  {issues_str}")
        wp.update_stage(current=i,
                        postfix=f"{pdir.name}: {verdict}  (ok {n_ok}/{i}, soft {n_soft_flag})")

    summary = {
        "n_patients": len(patient_dirs),
        "n_ok": n_ok,
        "n_soft_flagged": n_soft_flag,
        "reference_inlet_mm_mean": round(ref_mean, 1),
        "reference_inlet_mm_std": round(ref_std, 1),
        "reports": reports,
    }
    (out_root / "anatomy_report.json").write_text(json.dumps(summary, indent=2))

    wp.update_stage(postfix=f"done: {n_ok}/{len(patient_dirs)} pass, {n_soft_flag} soft-flagged")
    wp.finish_stage("done")
    print(f"\n[anatomy] {n_ok}/{len(patient_dirs)} pass HARD checks, "
          f"{n_soft_flag} have soft flags")
    print(f"[anatomy] report: {out_root}/anatomy_report.json")
    wp.stop_server(grace_s=30.0)


if __name__ == "__main__":
    main()
