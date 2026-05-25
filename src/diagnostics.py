"""Training-time diagnostics: sample previews + loss CSV logger.

The two callables here are hooked into `src/train.py` at the end of every
epoch so a long training run leaves visible evidence of convergence:

  * `save_cvae_previews` writes a 2x4 grid of (real | reconstruction) pairs
    so you can see by eye whether the CVAE is learning anything.
  * `save_diffusion_previews` samples 4 slices (2 plain, 2 hilly) at z=-0.5
    and z=+0.3 through the full diffusion + decoder pipeline. Visible
    progress = real progress.
  * `LossLogger` appends epoch summary lines to a CSV so you can plot
    convergence later.

All previews use the same display window as the rest of the project
(cfg.output.png_window, default [-200, 500] HU).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ----- shared image helpers -----------------------------------------------

def _to_uint8(tensor_or_array, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Map a [-1, 1] tensor/array slice to a uint8 image for PIL."""
    if isinstance(tensor_or_array, torch.Tensor):
        a = tensor_or_array.detach().cpu().numpy()
    else:
        a = np.asarray(tensor_or_array)
    a = np.clip(a, lo, hi)
    a = (a - lo) / max(hi - lo, 1e-6)
    return (a * 255.0).astype(np.uint8)


def _grid(slices: list[np.ndarray], cols: int, gap: int = 4) -> np.ndarray:
    """Lay out a list of HxW uint8 slices into a (rows*cols) grid PNG."""
    n = len(slices)
    rows = (n + cols - 1) // cols
    h, w = slices[0].shape
    canvas = np.full((rows * h + (rows - 1) * gap,
                      cols * w + (cols - 1) * gap), 255, dtype=np.uint8)
    for i, s in enumerate(slices):
        r, c = divmod(i, cols)
        y0 = r * (h + gap)
        x0 = c * (w + gap)
        canvas[y0:y0 + h, x0:x0 + w] = s
    return canvas


# ----- CVAE previews -------------------------------------------------------

@torch.no_grad()
def save_cvae_previews(model, sample_batch, out_dir: Path, epoch: int, device: str):
    """Save 2x4 grid: top row = real slices, bottom row = CVAE reconstructions."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    x, cond = sample_batch
    x = x.to(device)
    region_id = cond["region_id"].to(device)
    z_pos = cond["z_pos"].to(device)

    n = min(4, x.size(0))
    x = x[:n]
    region_id = region_id[:n]
    z_pos = z_pos[:n]
    x_rec, _, _, _ = model(x, region_id, z_pos)

    reals = [_to_uint8(x[i, 0]) for i in range(n)]
    recs = [_to_uint8(x_rec[i, 0]) for i in range(n)]
    grid = _grid(reals + recs, cols=n)
    Image.fromarray(grid, mode="L").save(out_dir / f"cvae_epoch{epoch:03d}.png")
    model.train()


# ----- Diffusion previews --------------------------------------------------

@torch.no_grad()
def save_diffusion_previews(unet, cvae, sched, out_dir: Path, epoch: int,
                             device: str, latent_size: int, latent_channels: int,
                             sampler_steps: int = 30):
    """Sample 4 slices (plain @z=-0.5, plain @z=0.3, hilly @z=-0.5, hilly @z=0.3)."""
    from .models import ddim_sample

    unet.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for region_id in [0, 1]:                                  # plain, hilly
        for z in [-0.5, 0.3]:
            shape = (1, latent_channels, latent_size, latent_size)
            r = torch.tensor([region_id], dtype=torch.long, device=device)
            zp = torch.tensor([z], dtype=torch.float32, device=device)
            latent = ddim_sample(unet, sched, shape, r, zp, steps=sampler_steps, device=device)
            decoded = cvae.decode(latent, cvae.cond(r, zp))
            panels.append(_to_uint8(decoded[0, 0]))
    grid = _grid(panels, cols=2)
    Image.fromarray(grid, mode="L").save(out_dir / f"diff_epoch{epoch:03d}.png")
    unet.train()


# ----- Loss CSV logger -----------------------------------------------------

class LossLogger:
    """Append per-epoch losses to a CSV (creates header on first write)."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = Path(path)
        self.fields = ["epoch", "elapsed_s"] + fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.fields)

    def log(self, epoch: int, **values):
        row = [epoch, round(time.time() - self._t0, 1)] + [values.get(k, "") for k in self.fields[2:]]
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
