"""Apply light 3D Gaussian smoothing to all patient DICOM series.

Fixes "zombified" 3D surface rendering caused by high-frequency noise
in synthetic CT volumes. Reads each DICOM series as a 3D volume,
smooths it, and re-saves in place (metadata unchanged).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import pydicom

SIGMA      = 0.8          # light smooth — preserves anatomy, removes noise spikes
OUT_ROOT   = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")
POPS       = ["Plain_175", "Hilly_175"]

total = ok = skipped = failed = 0

for pop in POPS:
    for pdir in sorted((OUT_ROOT / pop).iterdir()):
        if not pdir.is_dir():
            continue
        dcm_dir = pdir / "DICOM"
        if not dcm_dir.exists():
            skipped += 1; continue

        total += 1
        try:
            # ── Load all slices sorted by InstanceNumber ──────────────────
            dcm_files = sorted(
                dcm_dir.glob("*.dcm"),
                key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True)
                                  .get("InstanceNumber", 0))
            )
            if len(dcm_files) < 4:
                skipped += 1; continue

            # Read pixel data into 3D array
            slices = [pydicom.dcmread(f) for f in dcm_files]
            vol    = np.stack([s.pixel_array for s in slices]).astype(np.float32)

            # ── Smooth ────────────────────────────────────────────────────
            vol_smooth = gaussian_filter(vol, sigma=[SIGMA * 0.5, SIGMA, SIGMA])
            # (lighter on Z axis to preserve slice-to-slice sharpness)

            # ── Write back (same dtype, same scale) ───────────────────────
            for i, (s, f) in enumerate(zip(slices, dcm_files)):
                orig_dtype = s.pixel_array.dtype
                arr = np.clip(vol_smooth[i], 0,
                              np.iinfo(orig_dtype).max).astype(orig_dtype)
                s.PixelData = arr.tobytes()
                s.save_as(str(f))

            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{total} done...", flush=True)

        except Exception as e:
            failed += 1
            print(f"  FAIL {pdir.name}: {e}", flush=True)

print(f"\nDone.  ok={ok}  skipped={skipped}  failed={failed}  total={total}", flush=True)
