"""Cluster cached volumes by pelvic morphometry -> Plain/Hilly labels CSV.

Run after `build_cache.py`.
"""

from _common import add_repo_to_path, base_parser, load_config
add_repo_to_path()

from src.pseudo_labels import make_pseudo_labels   # noqa: E402


def main():
    ap = base_parser("Generate Plain/Hilly pseudo-labels from pelvic morphometry.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = make_pseudo_labels(cfg)
    print(f"[labels] total={out['n_total']}  plain={out['n_plain']}  hilly={out['n_hilly']}")
    print(f"[labels] CSV -> {out['labels_csv']}")
    print(f"[labels] centroids: {out['centroids']}")


if __name__ == "__main__":
    main()
