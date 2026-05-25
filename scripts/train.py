"""Train stage A (CVAE) and/or stage B (latent diffusion).

Usage:
    python scripts/train.py                 # both stages
    python scripts/train.py --stage cvae    # only CVAE
    python scripts/train.py --stage diff    # only diffusion (CVAE must exist)
    python scripts/train.py --smoke         # 1 epoch x few batches per stage
"""

from _common import add_repo_to_path, base_parser, load_config, apply_overrides
add_repo_to_path()

from src.train import train_cvae, train_diffusion   # noqa: E402


def main():
    ap = base_parser("Train CVAE and/or latent-diffusion UNet.")
    ap.add_argument("--stage", choices=["cvae", "diff", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--cvae-epochs", type=int, default=None, dest="cvae_epochs")
    ap.add_argument("--diff-epochs", type=int, default=None, dest="diff_epochs")
    ap.add_argument("--cvae-batch", type=int, default=None, dest="cvae_batch")
    ap.add_argument("--diff-batch", type=int, default=None, dest="diff_batch")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    if args.stage in ("cvae", "both"):
        print("[train] === stage A: CVAE ===")
        cvae_ckpt = train_cvae(cfg)
        print(f"[train] cvae checkpoint -> {cvae_ckpt}")
    if args.stage in ("diff", "both"):
        print("[train] === stage B: latent diffusion ===")
        diff_ckpt = train_diffusion(cfg)
        print(f"[train] diffusion checkpoint -> {diff_ckpt}")


if __name__ == "__main__":
    main()
