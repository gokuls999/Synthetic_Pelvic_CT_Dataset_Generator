"""Synthetic end-to-end smoke test (no real data, no training).

Verifies wiring of: models -> DDPM step -> CVAE decode -> DICOM writer -> PNG/JPG
exports -> DICOM validation. Run this after install to make sure the pipeline
imports and the file outputs are well-formed before pointing it at real data.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models import CVAE, LatentUNet, DiffusionSchedule, q_sample, ddim_sample
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata
from src.validate import validate_dicom_series


def main():
    cfg = load_config("configs/default.yaml")
    cfg["generation"]["slices_per_volume"] = 8

    device = "cpu"
    print("[smoke] building tiny model instances")
    cm = cfg["model"]["cvae"]
    cvae = CVAE(in_channels=cm["in_channels"], base_channels=cm["base_channels"],
                latent_channels=cm["latent_channels"], latent_size=cm["latent_size"],
                cond_dim=cm["cond_dim"]).to(device).eval()

    dm = cfg["model"]["diffusion"]
    unet = LatentUNet(image_size=dm["image_size"], in_channels=dm["in_channels"],
                      base_channels=64, channel_mults=[1, 2], num_res_blocks=1,
                      cond_dim=dm["cond_dim"]).to(device).eval()
    sched = DiffusionSchedule.make(50, "linear").to(device)

    print("[smoke] CVAE forward")
    x = torch.randn(2, 1, 256, 256)
    r = torch.tensor([0, 1])
    z = torch.tensor([-0.3, 0.7])
    x_rec, mu, logvar, latent = cvae(x, r, z)
    assert x_rec.shape == x.shape, x_rec.shape
    assert latent.shape == (2, cm["latent_channels"], cm["latent_size"], cm["latent_size"])
    print("       OK", x_rec.shape, "-> latent", latent.shape)

    print("[smoke] DDPM noising step")
    z_t, noise = q_sample(latent, torch.tensor([10, 20]), sched)
    eps = unet(z_t, torch.tensor([10, 20]), r, z)
    assert eps.shape == latent.shape, eps.shape
    print("       OK", eps.shape)

    print("[smoke] DDIM sampling -> 1 latent")
    samp = ddim_sample(unet, sched, (1, cm["latent_channels"], cm["latent_size"], cm["latent_size"]),
                       torch.tensor([0]), torch.tensor([0.0]), steps=4, device=device)
    decoded = cvae.decode(samp, cvae.cond(torch.tensor([0]), torch.tensor([0.0])))
    print("       decoded slice:", decoded.shape, "range", float(decoded.min()), float(decoded.max()))

    print("[smoke] DICOM builder + exports on dummy volume")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_pipeline_"))
    try:
        Z = 8
        # Fake HU volume that has bone-like values in a centered disc so morphometry isn't degenerate.
        H = W = 64
        yy, xx = np.mgrid[:H, :W]
        disc = ((yy - H/2)**2 + (xx - W/2)**2) < (H/4)**2
        vol_hu = np.where(disc[None].repeat(Z, axis=0), 300, -200).astype(np.int16)

        vol = SyntheticVolume(
            pixels_hu=vol_hu, spacing=(1.5, 1.0, 1.0),
            region="plain", region_id=0, patient_id="SMOKE_001", seed=42,
        )
        dicom_dir = tmp / "DICOM"
        write_dicom_series(vol, dicom_dir, cfg)
        write_png_jpg_metadata(vol, tmp / "PNG", tmp / "JPG", tmp / "metadata.json", cfg)
        report = validate_dicom_series(dicom_dir)
        print("       validation report:", {k: v for k, v in report.items() if k != "issues"})
        if report["issues"]:
            print("       ISSUES:", report["issues"])
        assert report["ok"], report["issues"]
        print("       OK — DICOM is 3D-Slicer loadable, exports written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n[smoke] all checks passed.")


if __name__ == "__main__":
    main()
