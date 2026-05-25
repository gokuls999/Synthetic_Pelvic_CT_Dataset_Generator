"""One-shot end-to-end runner: cache -> labels -> train -> generate -> validate.

Run this directly in your PowerShell so the progress UI (tqdm bars + spinner)
renders live on YOUR terminal. Do NOT pipe to a file — that breaks in-place bar
rendering.

Presets (`--preset`):
  proof      : cap 5 vols/dataset, CVAE 3 ep, diff 3 ep, 8 patients.  (~1-2h)
  overnight  : cap 50 vols/dataset, CVAE 8 ep, diff 12 ep, 50 patients. (~6-10h)
  full       : no caps, CVAE 20 ep, diff 40 ep, 400 patients.          (days)

Examples:
    python scripts\\run_all.py --preset proof --device cuda
    python scripts\\run_all.py --preset overnight --device cuda
    python scripts\\run_all.py --preset overnight --skip cache    # reuse cache
    python scripts\\run_all.py --preset full --skip cache,labels
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import sys
import time

from src.preprocessing import build_cache
from src.pseudo_labels import make_pseudo_labels
from src.train import train_cvae, train_diffusion
from src.generate import generate_dataset
from src.validate import validate_dataset


PRESETS = {
    "proof": {
        "max_per_dataset": 5,
        "cvae_epochs": 3,
        "diff_epochs": 3,
        "num_patients": 8,
    },
    "overnight": {
        "max_per_dataset": 50,
        "cvae_epochs": 8,
        "diff_epochs": 12,
        "num_patients": 50,
    },
    "full": {
        "max_per_dataset": None,
        "cvae_epochs": 20,
        "diff_epochs": 40,
        "num_patients": 400,
    },
}


def _banner(s: str):
    bar = "=" * 72
    print(f"\n{bar}\n  {s}\n{bar}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--skip", default="",
                    help="Comma-separated stages to skip: cache,labels,cvae,diff,generate,validate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preset = PRESETS[args.preset]
    cfg["training"]["device"] = args.device
    cfg["preprocess"]["max_volumes_per_dataset"] = preset["max_per_dataset"]
    cfg["training"]["cvae"]["epochs"] = preset["cvae_epochs"]
    cfg["training"]["diffusion"]["epochs"] = preset["diff_epochs"]
    cfg["generation"]["num_patients"] = preset["num_patients"]

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    t0 = time.time()

    if "cache" not in skip:
        _banner(f"[1/5] BUILD CACHE  (max {preset['max_per_dataset']} vol/dataset)")
        out = build_cache(cfg)
        print(f"      cache: ok={out['ok']} skipped={out['skipped']} errors={out['errors']}")

    if "labels" not in skip:
        _banner("[2/5] PSEUDO-LABELS  (plain vs hilly via morphometry + KMeans)")
        out = make_pseudo_labels(cfg)
        print(f"      labels: total={out['n_total']}  plain={out['n_plain']}  hilly={out['n_hilly']}")

    if "cvae" not in skip:
        _banner(f"[3/5] TRAIN CVAE  ({preset['cvae_epochs']} epochs)")
        train_cvae(cfg)

    if "diff" not in skip:
        _banner(f"[4a/5] TRAIN DIFFUSION  ({preset['diff_epochs']} epochs)")
        train_diffusion(cfg)

    if "generate" not in skip:
        _banner(f"[4b/5] GENERATE  ({preset['num_patients']} synthetic patients)")
        summaries = generate_dataset(cfg)
        print(f"      wrote {len(summaries)} patients -> {cfg['paths']['outputs_dir']}")

    if "validate" not in skip:
        _banner("[5/5] VALIDATE  (DICOM sanity check)")
        rpt = validate_dataset(cfg["paths"]["outputs_dir"])
        print(f"      {rpt['n_ok']}/{rpt['n_patients']} patients OK")

    mins = (time.time() - t0) / 60.0
    _banner(f"DONE in {mins:.1f} min   (outputs: {cfg['paths']['outputs_dir']}/)")


if __name__ == "__main__":
    main()
