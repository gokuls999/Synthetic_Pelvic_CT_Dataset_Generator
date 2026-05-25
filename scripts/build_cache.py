"""Walk all source datasets, preprocess each volume, and write the .npz cache.

Usage:
    python scripts/build_cache.py
    python scripts/build_cache.py --smoke
    python scripts/build_cache.py --max-per-dataset 5
"""

from _common import add_repo_to_path, base_parser, load_config, apply_overrides
add_repo_to_path()

from src.preprocessing import build_cache   # noqa: E402


def main():
    ap = base_parser("Build the preprocessed pelvic-CT cache.")
    ap.add_argument("--smoke", action="store_true", help="Cap volumes for a fast smoke test")
    ap.add_argument("--max-per-dataset", type=int, default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    print(f"[build_cache] cache_dir={cfg['paths']['cache_dir']}")
    summary = build_cache(cfg)
    print(f"\n[done] ok={summary['ok']} skipped={summary['skipped']} errors={summary['errors']}")
    print(f"       manifest -> {summary['manifest']}")


if __name__ == "__main__":
    main()
