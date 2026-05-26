"""Run anatomy validation on the synthetic pelvic CT dataset.

Requires TotalSegmentator:
    pip install totalsegmentator
(First call downloads ~2 GB of nnUNet weights into the user cache. Uses GPU
if available, falls back to CPU; ~10-30 s/volume on a GTX 1080 Ti.)

Per patient, checks:
  HARD (any fail -> volume rejected by report):
    * sacrum, both femoral heads, both hip bones present (>= 1000 voxels)
  SOFT (logged in report, don't reject):
    * left-right hip symmetry >= 0.85
    * pelvic inlet within +/-2 sigma of the real-data distribution

Output: synthetic_dataset/anatomy_report.json + a console summary.

Usage:
    python scripts\\anatomy_validate.py
"""

from _common import add_repo_to_path, base_parser, load_config
add_repo_to_path()

import torch                                          # noqa: E402  (ensure CUDA visibility)

from src.anatomy_validator import validate_dataset_anatomy   # noqa: E402


def main():
    ap = base_parser("Anatomy validation via TotalSegmentator.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    print(f"[anatomy] CUDA available: {torch.cuda.is_available()}")
    print(f"[anatomy] outputs dir:    {cfg['paths']['outputs_dir']}/")
    print(f"[anatomy] reference CSV:  {cfg['paths']['labels_csv']}")
    print()

    def _on_done(pid, ok, issues):
        tag = "OK" if ok else "FAIL"
        msg = "; ".join(issues) if issues else ""
        print(f"  [{tag}] {pid}  {msg}")

    summary = validate_dataset_anatomy(
        cfg["paths"]["outputs_dir"],
        cfg["paths"]["labels_csv"],
        progress_cb=_on_done,
    )

    print()
    print(f"[anatomy] {summary['n_ok']}/{summary['n_patients']} patients pass HARD checks")
    print(f"[anatomy] real-data pelvic inlet reference: "
          f"{summary['reference_inlet_mm_mean']:.0f} +- "
          f"{summary['reference_inlet_mm_std']:.0f} mm")
    print(f"[anatomy] full report: {cfg['paths']['outputs_dir']}/anatomy_report.json")


if __name__ == "__main__":
    main()
