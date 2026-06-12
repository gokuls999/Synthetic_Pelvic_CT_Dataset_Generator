"""Strong 3D smoothing to fix inter-slice staircase artifacts.

CVAE generates slices independently → organ boundaries don't align
between adjacent slices → bumpy 3D surfaces.

Fix: heavy Z-direction smoothing + moderate XY smoothing.
  sigma = [3.0, 1.5, 1.5]  (Z, Y, X)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import pydicom

SIGMA    = [3.0, 1.5, 1.5]   # Z heavy, XY moderate
OUT_ROOT = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")

total = ok = skipped = failed = 0

for pop in ["Plain_175", "Hilly_175"]:
    for pdir in sorted((OUT_ROOT / pop).iterdir()):
        if not pdir.is_dir(): continue
        dcm_dir = pdir / "DICOM"
        if not dcm_dir.exists(): skipped += 1; continue

        total += 1
        try:
            dcm_files = sorted(
                dcm_dir.glob("*.dcm"),
                key=lambda f: int(pydicom.dcmread(
                    f, stop_before_pixels=True).get("InstanceNumber", 0))
            )
            if len(dcm_files) < 4: skipped += 1; continue

            slices = [pydicom.dcmread(str(f)) for f in dcm_files]
            vol    = np.stack([s.pixel_array for s in slices]).astype(np.float32)

            # Strong 3D smooth — fixes inter-slice inconsistency
            vol_smooth = gaussian_filter(vol, sigma=SIGMA)

            orig_dtype = slices[0].pixel_array.dtype
            max_val    = np.iinfo(orig_dtype).max

            for i, (s, f) in enumerate(zip(slices, dcm_files)):
                arr = np.clip(vol_smooth[i], 0, max_val).astype(orig_dtype)
                s.PixelData = arr.tobytes()
                s.save_as(str(f))

            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{total} done...", flush=True)

        except Exception as e:
            failed += 1
            print(f"  FAIL {pdir.name}: {e}", flush=True)

print(f"\nDone.  ok={ok}  skipped={skipped}  failed={failed}  total={total}", flush=True)
