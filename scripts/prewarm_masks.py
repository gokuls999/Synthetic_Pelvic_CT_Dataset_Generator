# -*- coding: utf-8 -*-
"""prewarm_masks.py — Pre-run TotalSegmentator on sources that are missing mask caches.
Runs in a separate process so the main generation doesn't stall on them.
Safe to run while pfd_real_v1.py is active — they use different source files.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from _common import add_repo_to_path, load_config
add_repo_to_path()

import numpy as np
from pathlib import Path
from src.pfd_segmentation import cached_masks_exist, segment_original_volume, PFD_ROI_SUBSET

cfg        = load_config("configs/default.yaml")
cache_root = Path(cfg["paths"]["cache_dir"])

print("Scanning for sources missing mask caches...")
missing = []
for ds_dir in sorted(cache_root.iterdir()):
    if not ds_dir.is_dir() or ds_dir.name == "masks":
        continue
    for npz in sorted(ds_dir.glob("*.npz")):
        uid = npz.stem
        try:
            with np.load(npz, allow_pickle=True) as nf:
                n_slices = int(nf.get("slices", 0))
                sp = Path(str(nf["source"]))
            # Only warm sources that pass the good_sources quality filter
            if n_slices < 80 or n_slices > 400:
                continue
            if not sp.exists():
                continue
            if not cached_masks_exist(cache_root, ds_dir.name, uid, PFD_ROI_SUBSET):
                missing.append((ds_dir.name, uid, sp))
        except Exception as e:
            print(f"  Skip {npz.name}: {e}")

print(f"Sources to pre-warm: {len(missing)}")
if not missing:
    print("All mask caches are warm. Nothing to do.")
    sys.exit(0)

for i, (ds, uid, src_path) in enumerate(missing):
    print(f"\n[{i+1}/{len(missing)}] Warming {ds}/{uid}...", flush=True)
    try:
        masks, _ = segment_original_volume(
            source_path=src_path, dataset=ds, uid=uid,
            cfg=cfg, cache_root=cache_root,
            roi_subset=PFD_ROI_SUBSET,
            on_progress=lambda msg: print(f"  {msg}", flush=True),
        )
        n_organs = sum(1 for m in masks.values() if m.any())
        print(f"  DONE — {n_organs} organs found, masks cached.", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

print("\nPre-warm complete.")
