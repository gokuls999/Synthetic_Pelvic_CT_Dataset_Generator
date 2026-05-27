"""One-shot end-to-end runner: cache -> labels -> train -> generate -> validate.

Run this directly in your PowerShell. A small embedded HTTP dashboard opens
at http://127.0.0.1:8765/ in your browser showing live progress for every
stage. Close the tab when the pipeline finishes.

Presets:
  proof      : cap 5 vols/dataset, CVAE 3 ep, diff 3 ep, 8 patients.  (~1-2h)
  midday     : cap 10 vols/dataset, CVAE 3 ep, diff 4 ep, 12 patients. (~2h)
  overnight  : cap 50 vols/dataset, CVAE 8 ep, diff 12 ep, 50 patients. (~6-10h)
  full       : no caps, CVAE 20 ep, diff 40 ep, 400 patients.          (days)

Examples:
    python scripts\\run_all.py --preset proof --device cuda
    python scripts\\run_all.py --preset overnight --device cuda
    python scripts\\run_all.py --preset overnight --skip cache    # reuse cache
    python scripts\\run_all.py --preset midday --no-browser       # don't auto-open
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import time
import traceback

from src.preprocessing import build_cache
from src.pseudo_labels import make_pseudo_labels
from src.train import train_cvae, train_diffusion
from src.generate import generate_dataset
from src.validate import validate_dataset
from src import web_progress as wp
from src.keep_awake import KeepAwake


PRESETS = {
    "proof": {
        "max_per_dataset": 5,
        "cvae_epochs": 3,
        "diff_epochs": 3,
        "num_patients": 8,
    },
    "midday": {
        # Bigger than proof, fits in ~2h on a GTX 1080 Ti.
        "max_per_dataset": 10,
        "cvae_epochs": 3,
        "diff_epochs": 4,
        "num_patients": 12,
    },
    "overnight": {
        # Sized for the deeper-latent CVAE (64x64) introduced after the first
        # overnight run produced 0/50 anatomy pass. 20 + 20 epochs is roughly
        # 3-4 days on a 1080 Ti at 4x more compute per step than the old config.
        "max_per_dataset": 50,
        "cvae_epochs": 20,
        "diff_epochs": 20,
        "num_patients": 50,
    },
    "full": {
        "max_per_dataset": None,
        "cvae_epochs": 20,
        "diff_epochs": 40,
        "num_patients": 400,
    },
}


def _run_stage(stage_id: str, fn, *args, summary_log=None, **kwargs):
    """Run one stage with the web dashboard tracking start/end + errors."""
    wp.set_stage(stage_id)
    try:
        result = fn(*args, **kwargs)
        if summary_log and result is not None:
            wp.log_msg(summary_log(result) if callable(summary_log) else str(summary_log))
        wp.finish_stage("done")
        return result
    except Exception as e:
        wp.log_msg(f"ERROR in {stage_id}: {e}")
        wp.finish_stage("error", error=str(e))
        traceback.print_exc()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--skip", default="",
                    help="Comma-separated stages to skip: cache,labels,cvae,diff,generate,validate")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the dashboard in a browser")
    ap.add_argument("--allow-sleep", action="store_true",
                    help="Allow the OS to sleep during the run (default: prevent sleep)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume CVAE / diffusion from the highest-numbered epoch checkpoint on disk")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preset = PRESETS[args.preset]
    cfg["training"]["device"] = args.device
    cfg["preprocess"]["max_volumes_per_dataset"] = preset["max_per_dataset"]
    cfg["training"]["cvae"]["epochs"] = preset["cvae_epochs"]
    cfg["training"]["diffusion"]["epochs"] = preset["diff_epochs"]
    cfg["generation"]["num_patients"] = preset["num_patients"]
    cfg["training"]["resume"] = args.resume

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    label = (f"preset={args.preset}  device={args.device}  "
             f"vols/ds={preset['max_per_dataset']}  "
             f"cvae={preset['cvae_epochs']}ep  diff={preset['diff_epochs']}ep  "
             f"patients={preset['num_patients']}")
    url = wp.start_server(port=args.port, open_browser=not args.no_browser,
                          run_label=label)

    print()
    print("=" * 72)
    print(f"  Pipeline dashboard:  {url}")
    print(f"  Preset:              {args.preset}")
    print(f"  Device:              {args.device}")
    print("=" * 72)
    print("  (If your browser didn't open, paste the URL above into it.)")
    print()

    t0 = time.time()
    if not args.allow_sleep:
        wp.log_msg("keep-awake enabled (system sleep blocked during run)")
    keep_awake = KeepAwake() if not args.allow_sleep else _NullCtx()
    try:
        with keep_awake:
            if "cache" not in skip:
                _run_stage("build_cache", build_cache, cfg,
                           summary_log=lambda r: f"cache: ok={r['ok']} skipped={r['skipped']} errors={r['errors']}")
            if "labels" not in skip:
                _run_stage("pseudo_labels", make_pseudo_labels, cfg,
                           summary_log=lambda r: f"labels: total={r['n_total']} plain={r['n_plain']} hilly={r['n_hilly']}")
            if "cvae" not in skip:
                _run_stage("train_cvae", train_cvae, cfg)
            if "diff" not in skip:
                _run_stage("train_diffusion", train_diffusion, cfg)
            if "generate" not in skip:
                _run_stage("generate", generate_dataset, cfg,
                           summary_log=lambda r: f"wrote {len(r)} patients")
            if "validate" not in skip:
                _run_stage("validate", validate_dataset, cfg["paths"]["outputs_dir"],
                           summary_log=lambda r: f"{r['n_ok']}/{r['n_patients']} patients OK")

        mins = (time.time() - t0) / 60.0
        print(f"\nDONE in {mins:.1f} min   outputs: {cfg['paths']['outputs_dir']}/\n")
        wp.log_msg(f"pipeline finished in {mins:.1f} min")
        wp.stop_server(grace_s=30.0)

    except KeyboardInterrupt:
        wp.log_msg("interrupted by user")
        wp.stop_server(grace_s=2.0)
        raise
    except Exception:
        wp.stop_server(grace_s=60.0)
        raise


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
