"""Sample synthetic pelvic CT volumes (latent diffusion -> CVAE decoder -> slice stack).

Adjacent slices are produced with the same region label and a smooth z_pos sweep
plus an init-noise EMA across z, which biases consecutive samples to be visually
continuous. This is a cheap stand-in for true 3D modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import CVAE, LatentUNet, DiffusionSchedule, ddim_sample
from .progress import pbar, Spinner


@dataclass
class SyntheticVolume:
    pixels_hu: np.ndarray             # (Z, H, W) HU int16
    spacing: tuple[float, float, float]   # (sz, sy, sx) mm
    region: str
    region_id: int
    patient_id: str
    seed: int


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_generators(cfg: dict, device: str | None = None
                    ) -> tuple[CVAE, LatentUNet, DiffusionSchedule, str]:
    device = device or _device(cfg["training"]["device"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])

    cvae_mcfg = cfg["model"]["cvae"]
    with Spinner(f"Loading CVAE checkpoint on {device}"):
        cvae = CVAE(in_channels=cvae_mcfg["in_channels"], base_channels=cvae_mcfg["base_channels"],
                    latent_channels=cvae_mcfg["latent_channels"], latent_size=cvae_mcfg["latent_size"],
                    cond_dim=cvae_mcfg["cond_dim"]).to(device).eval()
        cvae.load_state_dict(torch.load(ckpt_dir / "cvae_latest.pt", map_location=device)["model"])

    dmcfg = cfg["model"]["diffusion"]
    with Spinner(f"Loading latent-diffusion UNet on {device}"):
        unet = LatentUNet(image_size=dmcfg["image_size"], in_channels=dmcfg["in_channels"],
                          base_channels=dmcfg["base_channels"], channel_mults=dmcfg["channel_mults"],
                          num_res_blocks=dmcfg["num_res_blocks"], cond_dim=dmcfg["cond_dim"]).to(device).eval()
        state = torch.load(ckpt_dir / "diffusion_latest.pt", map_location=device)
        unet.load_state_dict(state.get("ema", state["model"]))

    sched = DiffusionSchedule.make(dmcfg["timesteps"], dmcfg["beta_schedule"]).to(device)
    return cvae, unet, sched, device


@torch.no_grad()
def generate_volume(cfg: dict, region_id: int, patient_id: str, seed: int,
                    cvae: CVAE, unet: LatentUNet, sched: DiffusionSchedule,
                    device: str) -> SyntheticVolume:
    gcfg = cfg["generation"]
    pp = cfg["preprocess"]
    n_slices = gcfg["slices_per_volume"]
    smoothing = float(gcfg["slice_smoothing"])
    steps = int(gcfg["sampler_steps"])

    g = torch.Generator(device="cpu").manual_seed(seed)

    latent_shape = (1, cfg["model"]["cvae"]["latent_channels"],
                    cfg["model"]["cvae"]["latent_size"], cfg["model"]["cvae"]["latent_size"])

    # Smooth z_pos sweep from -1 (cranial) -> +1 (caudal).
    z_positions = torch.linspace(-1.0, 1.0, n_slices)

    # Init-noise EMA: each slice's seed noise is a blend of fresh noise + previous noise.
    # This is the key trick for cross-slice consistency without 3D conditioning.
    prev_noise = torch.randn(latent_shape, generator=g)
    region_tensor = torch.tensor([region_id], dtype=torch.long, device=device)

    slices = []
    slice_bar = pbar(enumerate(z_positions), total=n_slices,
                     desc=f"    {patient_id} slices", unit="slice", leave=False)
    for i, zp in slice_bar:
        fresh = torch.randn(latent_shape, generator=g)
        noise = (1.0 - smoothing) * fresh + smoothing * prev_noise
        # Re-normalize so std stays ~1
        noise = noise / (noise.std() + 1e-6)
        prev_noise = noise

        z_pos_t = zp.view(1).to(device)
        latent = ddim_sample(unet, sched, latent_shape, region_tensor, z_pos_t,
                             steps=steps, device=device, init_noise=noise.to(device))
        cond_vec = cvae.cond(region_tensor, z_pos_t)
        decoded = cvae.decode(latent, cond_vec)            # (1, 1, H, W) in [-1, 1]
        slices.append(decoded.squeeze(0).squeeze(0).cpu().numpy())
    slice_bar.close()

    vol_norm = np.stack(slices, axis=0)                   # (Z, H, W)
    # Map [-1, 1] back to HU using the same window used at preprocessing.
    from .preprocessing import unwindow
    vol_hu = unwindow(vol_norm, pp["hu_min"], pp["hu_max"])
    # Clamp & cast to int16 for storage.
    vol_hu = np.clip(vol_hu, -1024.0, 3071.0).astype(np.int16)

    region_name = "plain" if region_id == 0 else "hilly"
    spacing = (float(cfg["output"]["slice_thickness_mm"]),
               float(cfg["output"]["pixel_spacing_mm"][0]),
               float(cfg["output"]["pixel_spacing_mm"][1]))

    return SyntheticVolume(
        pixels_hu=vol_hu, spacing=spacing, region=region_name,
        region_id=region_id, patient_id=patient_id, seed=seed,
    )


def generate_dataset(cfg: dict, out_root: str | Path | None = None):
    """Generate all synthetic patients per cfg.generation and write outputs.

    Returns a list of patient summaries (id, region, paths).
    """
    from .dicom_builder import write_dicom_series
    from .exports import write_png_jpg_metadata

    gcfg = cfg["generation"]
    out_root = Path(out_root or cfg["paths"]["outputs_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    cvae, unet, sched, device = load_generators(cfg)
    n = int(gcfg["num_patients"])
    n_plain = int(round(n * float(gcfg["plain_fraction"])))
    n_hilly = n - n_plain

    plan = [0] * n_plain + [1] * n_hilly
    rng = np.random.default_rng(int(gcfg["seed"]))
    rng.shuffle(plan)

    summaries = []
    patient_bar = pbar(enumerate(plan, start=1), total=len(plan),
                       desc="Generating patients", unit="pt")
    for i, region_id in patient_bar:
        pid = f"PF_{i:04d}"
        seed = int(gcfg["seed"]) + i
        patient_bar.set_postfix(now=pid, region="plain" if region_id == 0 else "hilly")

        vol = generate_volume(cfg, region_id, pid, seed, cvae, unet, sched, device)

        pdir = out_root / pid
        dicom_dir = pdir / "DICOM"
        with Spinner(f"  Writing DICOM {pid}"):
            write_dicom_series(vol, dicom_dir, cfg)
        png_dir = pdir / "PNG"
        jpg_dir = pdir / "JPG"
        meta_path = pdir / "metadata.json"
        with Spinner(f"  Writing PNG/JPG/metadata {pid}"):
            write_png_jpg_metadata(vol, png_dir, jpg_dir, meta_path, cfg)
        summaries.append({"patient_id": pid, "region": vol.region,
                          "dicom_dir": str(dicom_dir), "n_slices": int(vol.pixels_hu.shape[0])})
    patient_bar.close()
    return summaries
