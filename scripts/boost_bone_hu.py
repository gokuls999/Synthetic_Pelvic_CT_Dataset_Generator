"""Boost bone HU values in all patient DICOMs.

Root cause: synthetic model generates compressed HU range (max ~486 HU).
Bone should reach 700-1500 HU for clean 3D Slicer rendering.

Fix: piecewise linear stretch above 150 HU:
  HU <= 150  → unchanged  (soft tissue / fat / air)
  HU > 150   → 150 + (HU - 150) * 1.93  (bone boosted to ~900 HU max)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
import pydicom

BONE_THRESH  = 150        # HU above this = bone candidate
BONE_SCALE   = 1.93       # stretches 486→900, 300→579, 200→289
OUT_ROOT     = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")

total = ok = skipped = failed = 0

for pop in ["Plain_175", "Hilly_175"]:
    for pdir in sorted((OUT_ROOT / pop).iterdir()):
        if not pdir.is_dir(): continue
        dcm_dir = pdir / "DICOM"
        if not dcm_dir.exists(): skipped += 1; continue

        total += 1
        try:
            dcm_files = sorted(dcm_dir.glob("*.dcm"),
                               key=lambda f: int(pydicom.dcmread(
                                   f, stop_before_pixels=True
                               ).get("InstanceNumber", 0)))
            if len(dcm_files) < 4: skipped += 1; continue

            for f in dcm_files:
                ds       = pydicom.dcmread(str(f))
                px       = ds.pixel_array.astype(np.int32)
                slope    = float(getattr(ds, "RescaleSlope",     1.0))
                intercept= float(getattr(ds, "RescaleIntercept", -1024.0))
                hu       = px * slope + intercept          # actual HU

                # Piecewise stretch
                hu_new   = hu.copy().astype(np.float32)
                bone_mask = hu > BONE_THRESH
                hu_new[bone_mask] = (BONE_THRESH +
                    (hu[bone_mask] - BONE_THRESH) * BONE_SCALE)
                hu_new = np.clip(hu_new, -1024.0, 3071.0)

                # Back to stored pixel values
                stored = np.clip(
                    ((hu_new - intercept) / slope).astype(np.int32),
                    0, 4095
                ).astype(np.uint16)

                ds.PixelData = stored.tobytes()
                ds.save_as(str(f))

            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{total} done...", flush=True)

        except Exception as e:
            failed += 1
            print(f"  FAIL {pdir.name}: {e}", flush=True)

print(f"\nDone.  ok={ok}  skipped={skipped}  failed={failed}  total={total}", flush=True)
