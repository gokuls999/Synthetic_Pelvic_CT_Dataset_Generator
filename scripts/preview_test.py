"""Generate a 2-patient preview DICOM set (1 plain + 1 hilly) from the CURRENT
CVAE checkpoint, on CPU, while training continues on GPU.

What it does:
    1. Loads `cvae_latest.pt` on CPU (does NOT touch the GPU, so training is safe).
    2. Picks one real cached volume per region (plain / hilly).
    3. Encodes each slice -> decodes back through the CVAE (no diffusion -> no
       novel sampling; this is a CVAE reconstruction preview).
    4. Writes a full DICOM series + PNG + JPG + metadata.json for each patient
       under `synthetic_dataset/_preview/`.

Use this to:
    * Verify the DICOM viewer (3D Slicer / RadiAnt / OHIF) opens the series
      cleanly with correct anatomy.
    * Spot-check whether the partly-trained CVAE bottleneck is preserving
      pelvic bone structure (the failure mode from the first overnight run).

NOT for: anatomy validation scoring or "novel patient" assessment. These are
reconstructions of REAL training cases, not synthesized novel patients.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import CVAE
from src.preprocessing import unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata


def pick_one_per_region(labels_csv: Path) -> dict[str, dict]:
    """Pick one volume per region. Prefer ones near the median pelvic_inlet_mm
    of their cluster (so the preview is representative, not extreme)."""
    df = pd.read_csv(labels_csv)
    picks = {}
    for region in ("plain", "hilly"):
        sub = df[df["region"] == region].copy()
        if len(sub) == 0:
            raise RuntimeError(f"No volumes labelled '{region}' in {labels_csv}")
        median = sub["pelvic_inlet_mm"].median()
        sub["dist"] = (sub["pelvic_inlet_mm"] - median).abs()
        row = sub.sort_values("dist").iloc[0]
        picks[region] = {
            "uid": row["uid"],
            "dataset": row["dataset"],
            "cache_path": row["cache_path"],
            "pelvic_inlet_mm": float(row["pelvic_inlet_mm"]),
            "region_id": int(row["region_id"]),
        }
    return picks


@torch.no_grad()
def reconstruct_volume(cvae: CVAE, npz_path: Path, region_id: int,
                       slice_size: int, batch: int = 4) -> np.ndarray:
    """Load cached slices, encode->decode through CVAE, return (Z,H,W) in [-1,1]."""
    with np.load(npz_path) as npz:
        slices = np.asarray(npz["slices"]).astype(np.float32)   # (Z, H, W)
    Z = slices.shape[0]
    out = np.empty_like(slices)

    region_t = torch.tensor([region_id], dtype=torch.long)
    for start in range(0, Z, batch):
        end = min(start + batch, Z)
        chunk = slices[start:end]                                # (B, H, W)
        x = torch.from_numpy(chunk).unsqueeze(1)                 # (B, 1, H, W)

        z_indices = np.arange(start, end)
        z_pos = (z_indices / max(Z - 1, 1)) * 2.0 - 1.0
        z_pos_t = torch.from_numpy(z_pos).float()

        rid = region_t.expand(end - start)
        cond_vec = cvae.cond(rid, z_pos_t)
        mu, _ = cvae.encode(x, cond_vec)                         # deterministic
        x_rec = cvae.decode(mu, cond_vec)                        # (B, 1, H, W) in [-1,1]
        out[start:end] = x_rec.squeeze(1).cpu().numpy()

        print(f"    slice {end}/{Z}  (region={region_id})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None,
                    help="Output root (default: <outputs_dir>/_preview)")
    ap.add_argument("--ckpt", default=None,
                    help="CVAE checkpoint (default: <checkpoints_dir>/cvae_latest.pt)")
    ap.add_argument("--batch", type=int, default=4,
                    help="Slices per CVAE forward batch on CPU (default 4)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Force CPU. Training has the GPU pinned at 98% VRAM.
    cfg["training"]["device"] = "cpu"

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg["paths"]["checkpoints_dir"]) / "cvae_latest.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"CVAE checkpoint not found: {ckpt_path}")

    labels_csv = Path(cfg["paths"]["labels_csv"])
    out_root = Path(args.out) if args.out else (Path(cfg["paths"]["outputs_dir"]) / "_preview")
    out_root.mkdir(parents=True, exist_ok=True)

    slice_size = int(cfg["preprocess"]["slice_size"])
    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])

    print(f"[preview] checkpoint:   {ckpt_path}")
    print(f"[preview] labels csv:   {labels_csv}")
    print(f"[preview] output root:  {out_root}")
    print(f"[preview] device:       CPU  (training keeps GPU)")

    picks = pick_one_per_region(labels_csv)
    for region, info in picks.items():
        print(f"[preview]   {region:5s} -> {info['uid']:25s}  inlet={info['pelvic_inlet_mm']:.1f}mm")

    cvae_mcfg = cfg["model"]["cvae"]
    print(f"[preview] building CVAE (latent {cvae_mcfg['latent_channels']}x"
          f"{cvae_mcfg['latent_size']}x{cvae_mcfg['latent_size']})...")
    cvae = CVAE(
        in_channels=cvae_mcfg["in_channels"], base_channels=cvae_mcfg["base_channels"],
        latent_channels=cvae_mcfg["latent_channels"], latent_size=cvae_mcfg["latent_size"],
        cond_dim=cvae_mcfg["cond_dim"],
    ).eval()

    state = torch.load(ckpt_path, map_location="cpu")
    cvae.load_state_dict(state["model"])
    print(f"[preview] loaded CVAE state from {ckpt_path.name}")

    spacing = (float(cfg["output"]["slice_thickness_mm"]),
               float(cfg["output"]["pixel_spacing_mm"][0]),
               float(cfg["output"]["pixel_spacing_mm"][1]))

    summaries = []
    for region, info in picks.items():
        pid = f"PREVIEW_{region.upper()}"
        npz_path = Path(info["cache_path"])
        if not npz_path.exists():
            print(f"[preview] WARNING: missing cache {npz_path}, skipping")
            continue

        t0 = time.time()
        print(f"[preview] reconstructing {pid}  source={info['uid']}  ...")
        vol_norm = reconstruct_volume(cvae, npz_path, info["region_id"],
                                      slice_size, batch=args.batch)
        vol_hu = unwindow(vol_norm, hu_min, hu_max)
        vol_hu = np.clip(vol_hu, -1024.0, 3071.0).astype(np.int16)

        vol = SyntheticVolume(
            pixels_hu=vol_hu, spacing=spacing, region=region,
            region_id=info["region_id"], patient_id=pid, seed=0,
        )

        pdir = out_root / pid
        dicom_dir = pdir / "DICOM"
        png_dir = pdir / "PNG"
        jpg_dir = pdir / "JPG"
        meta_path = pdir / "metadata.json"

        print(f"[preview]   writing DICOM series -> {dicom_dir}")
        write_dicom_series(vol, dicom_dir, cfg)
        print(f"[preview]   writing PNG/JPG/metadata -> {pdir}")
        write_png_jpg_metadata(vol, png_dir, jpg_dir, meta_path, cfg)

        dt = time.time() - t0
        print(f"[preview]   {pid}  Z={vol_hu.shape[0]}  H={vol_hu.shape[1]}  "
              f"W={vol_hu.shape[2]}  ({dt/60:.1f} min)")

        summaries.append({"pid": pid, "region": region, "source_uid": info["uid"],
                          "n_slices": int(vol_hu.shape[0]), "dir": str(pdir)})

    print()
    print("=" * 72)
    print(f"  Preview written to: {out_root}")
    for s in summaries:
        print(f"    {s['pid']:18s}  {s['n_slices']:3d} slices  source={s['source_uid']}")
    print("=" * 72)
    print("  Open the DICOM folder in 3D Slicer / RadiAnt / OHIF.")
    print("  PNG/JPG are quick previews (window -200..500 HU).")


if __name__ == "__main__":
    main()
