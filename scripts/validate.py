"""Validate every generated DICOM series under synthetic_dataset/.

Writes synthetic_dataset/validation_report.json and prints a summary.
"""

from _common import add_repo_to_path, base_parser, load_config
add_repo_to_path()

from src.validate import validate_dataset   # noqa: E402


def main():
    ap = base_parser("Validate the generated synthetic DICOM dataset.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    summary = validate_dataset(cfg["paths"]["outputs_dir"])
    print(f"[validate] {summary['n_ok']}/{summary['n_patients']} patients OK")
    bad = [r for r in summary["reports"] if not r["ok"]]
    for r in bad[:10]:
        print(f"  {r['patient_id']}: {'; '.join(r['issues'])}")


if __name__ == "__main__":
    main()
