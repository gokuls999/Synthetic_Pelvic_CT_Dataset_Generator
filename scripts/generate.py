"""Generate the synthetic dataset (DICOM + PNG + JPG + metadata.json).

Usage:
    python scripts/generate.py                  # all patients per cfg
    python scripts/generate.py --num-patients 8
    python scripts/generate.py --smoke
"""

from _common import add_repo_to_path, base_parser, load_config, apply_overrides
add_repo_to_path()

from src.generate import generate_dataset   # noqa: E402


def main():
    ap = base_parser("Generate the synthetic pelvic-CT dataset.")
    ap.add_argument("--num-patients", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    if args.num_patients is not None:
        cfg["generation"]["num_patients"] = int(args.num_patients)

    summaries = generate_dataset(cfg)
    print(f"[done] wrote {len(summaries)} synthetic patients -> {cfg['paths']['outputs_dir']}")


if __name__ == "__main__":
    main()
