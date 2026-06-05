"""Re-render the 50 Phase 1 FULL patients with the REAL per-volume spacing.

The original pfd_phase1_full.py wrote DICOM headers using the constant
output.slice_thickness_mm=1.5 / pixel_spacing_mm=[1.0,1.0] from
configs/default.yaml. Those are fine for pure-generative patients but
WRONG for the hybrid PFD patients whose pixels come from a real cached
volume that may have very different spacing.

For example PFD_001's source had real spacing (sz=0.801, sy=0.858, sx=0.858);
the DICOM was claiming 1.5 mm Z thickness, making the 3D reconstruction
1.87x too tall (~38 cm vs the real ~20 cm). 3D Slicer rendered an
elongated zombie.

This script reads each patient's metadata.json to find its source cache
.npz, reads the real spacing from that npz, and re-writes the DICOM
series with the real spacing in the headers. Mask cache is reused
(deformation is recomputed but no TS run).

Time: ~5 min for all 50 patients on CPU. No GPU needed.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.pfd_segmentation import (
    load_cached_masks, rectum_subregion, mask_stats, PFD_ROI_SUBSET,
)
from src.pfd_deformation import (
    build_pattern_from_masks, build_displacement_field, apply_deformation,
)
from src import web_progress as wp


def _window_to_uint8(slice_hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = np.clip(slice_hu.astype(np.float32), lo, hi)
    s = (s - lo) / max(hi - lo, 1e-6)
    return (s * 255.0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target", default="synthetic_dataset/_pfd_phase1_full")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_root = Path(cfg["paths"]["cache_dir"])
    out_root = Path(args.target)
    if not out_root.is_dir():
        raise SystemExit(f"target dir not found: {out_root}")

    patient_dirs = sorted(p for p in out_root.iterdir()
                          if p.is_dir() and (p / "metadata.json").is_file())
    if not patient_dirs:
        raise SystemExit("no patients to re-render")

    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])
    lo, hi = cfg["output"]["png_window"]

    wp.set_stages([
        ("scan",     f"Scan ({len(patient_dirs)} patients)"),
        ("respace",  f"Re-render with real spacing"),
        ("summary",  "Summary"),
    ])
    url = wp.start_server(
        port=args.port, open_browser=not args.no_browser,
        run_label=f"PFD respacing  {len(patient_dirs)} patients  (real per-volume spacing in DICOM)",
        expected_total_s=len(patient_dirs) * 6.0, cfg=None,
    )
    print()
    print("=" * 72)
    print(f"  Respacing dashboard:   {url}")
    print(f"  Target:                {out_root}")
    print(f"  Patients to fix:       {len(patient_dirs)}")
    print("=" * 72)
    print()

    wp.set_stage("scan", postfix="reading existing metadata")
    wp.finish_stage("done")

    wp.set_stage("respace", total=len(patient_dirs), postfix="starting")
    n_ok = 0
    n_skip = 0
    t_run = time.time()

    for i, pdir in enumerate(patient_dirs, start=1):
        meta_path = pdir / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: bad metadata: {e}")
            n_skip += 1
            continue
        uid = meta.get("source_volume_uid")
        dataset = meta.get("source_dataset")
        pattern = meta.get("pfd_pattern")
        severity = meta.get("pfd_severity")
        region_id = 0 if severity == "plain" else 1
        if not (uid and dataset and pattern and severity):
            wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: missing fields in metadata")
            n_skip += 1
            continue

        cache_path = cache_root / dataset / f"{uid}.npz"
        if not cache_path.exists():
            wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: cache npz not found at {cache_path}")
            n_skip += 1
            continue

        wp.update_stage(current=i - 1, postfix=f"{i}/{len(patient_dirs)} {pdir.name}")

        try:
            with np.load(cache_path, allow_pickle=True) as npz:
                vol_norm = np.asarray(npz["slices"]).astype(np.float32)
                cache_spacing = tuple(float(x)
                                      for x in np.asarray(npz["spacing"]).tolist())
            Z, H, W = vol_norm.shape
            sz, sy, sx = cache_spacing

            masks = load_cached_masks(cache_root, dataset, uid, PFD_ROI_SUBSET)
            if "colon" in masks and masks["colon"].any():
                rectum_mask = rectum_subregion(masks["colon"], frac=0.66)
                masks["rectum"] = rectum_mask
            stats = {}
            for name, m in masks.items():
                st = mask_stats(m, name)
                if st is not None:
                    stats[name] = st

            # Pattern + deformation
            p_obj = build_pattern_from_masks(pattern, stats, severity=severity)
            field = build_displacement_field((Z, H, W), p_obj.blobs)
            deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)

            vol_hu = np.clip(unwindow(deformed, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)
            spacing_zyx = (sz, sy, sx)
            vol = SyntheticVolume(
                pixels_hu=vol_hu, spacing=spacing_zyx, region=severity,
                region_id=region_id, patient_id=pdir.name,
                seed=meta.get("seed", 0) or 0,
            )

            # Re-write DICOM with REAL spacing
            dicom_dir = pdir / "DICOM"
            for f in dicom_dir.glob("*.dcm"):
                f.unlink()
            write_dicom_series(vol, dicom_dir, cfg)

            # Re-write PNGs and JPGs with corrected data (in case deformation
            # changed; same as before but harmless to refresh)
            png_dir = pdir / "PNG"
            jpg_dir = pdir / "JPG"
            png_dir.mkdir(parents=True, exist_ok=True)
            jpg_dir.mkdir(parents=True, exist_ok=True)
            for z in range(Z):
                img = Image.fromarray(_window_to_uint8(vol_hu[z], lo, hi), mode="L")
                img.save(png_dir / f"slice_{z+1:04d}.png", "PNG")
                img.save(jpg_dir / f"slice_{z+1:04d}.jpg", "JPEG", quality=92)

            # Update metadata in place -- preserve PFD findings
            meta["respacing_applied"] = True
            meta["real_spacing_mm"] = [sz, sy, sx]
            meta["n_slices"] = int(Z)
            meta["slice_thickness_mm"] = float(sz)
            meta["pixel_spacing_mm"] = [float(sy), float(sx)]
            meta_path.write_text(json.dumps(meta, indent=2))

            wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: "
                       f"spacing -> (sz={sz:.3f}, sy={sy:.3f}, sx={sx:.3f})  Z={Z}")
            n_ok += 1
        except Exception as e:
            wp.log_msg(f"  [{i}/{len(patient_dirs)}] {pdir.name}: ERROR {type(e).__name__}: {e}")
            n_skip += 1

        wp.update_stage(current=i, postfix=f"{i}/{len(patient_dirs)} ok={n_ok} skip={n_skip}")

    wp.finish_stage("done")

    wp.set_stage("summary", postfix="done")
    total_min = (time.time() - t_run) / 60.0
    wp.update_stage(postfix=f"{n_ok} re-rendered, {n_skip} skipped ({total_min:.1f} min)")
    wp.finish_stage("done")
    print()
    print("=" * 72)
    print(f"  RESPACING DONE in {total_min:.1f} min")
    print(f"  Re-rendered: {n_ok} patients")
    print(f"  Skipped:     {n_skip}")
    print(f"  Target:      {out_root}")
    print("=" * 72)
    wp.stop_server(grace_s=60.0)


if __name__ == "__main__":
    main()
