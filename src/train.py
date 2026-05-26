"""Training loops for the CVAE and the latent diffusion UNet.

Two stages:
  1. Stage A -- train CVAE on slices (reconstruction + KL).
  2. Stage B -- freeze CVAE, train LatentUNet to denoise CVAE latents (DDPM ε-loss).

Both stages are sized for a single 8-16 GB consumer GPU: AMP fp16, gradient
accumulation, batch sizes tuned for 32x32 latents.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PelvicSliceDataset
from .models import (
    CVAE, LatentUNet, DiffusionSchedule, cvae_loss, q_sample,
)
from .progress import pbar, Spinner
from .diagnostics import save_cvae_previews, save_diffusion_previews, LossLogger


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def _make_loader(cfg: dict, batch_size: int) -> DataLoader:
    ds = PelvicSliceDataset(
        labels_csv=cfg["paths"]["labels_csv"],
        slice_size=cfg["preprocess"]["slice_size"],
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg["training"]["num_workers"], pin_memory=True, drop_last=True,
    )


# ----- Stage A: CVAE ------------------------------------------------------

def train_cvae(cfg: dict):
    device = _device(cfg["training"]["device"])
    tcfg = cfg["training"]["cvae"]
    mcfg = cfg["model"]["cvae"]
    smoke = cfg["training"].get("smoke", False)

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = _make_loader(cfg, tcfg["batch_size"])
    with Spinner("Building CVAE on " + device):
        model = CVAE(in_channels=mcfg["in_channels"], base_channels=mcfg["base_channels"],
                     latent_channels=mcfg["latent_channels"], latent_size=mcfg["latent_size"],
                     cond_dim=mcfg["cond_dim"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg["training"]["mixed_precision"]))

    # Resume from the last saved cvae_epochN.pt if --resume is set in cfg.
    # `start_epoch` is the next epoch index to train (0-based), so e.g. if
    # we resume from cvae_epoch1.pt we start training at epoch index 1 = "epoch 2".
    start_epoch = 0
    if cfg["training"].get("resume", False):
        existing = sorted(ckpt_dir.glob("cvae_epoch*.pt"),
                          key=lambda p: int(p.stem.replace("cvae_epoch", "")))
        if existing:
            last = existing[-1]
            with Spinner(f"Resuming CVAE from {last.name}"):
                model.load_state_dict(torch.load(last, map_location=device)["model"])
            start_epoch = int(last.stem.replace("cvae_epoch", ""))

    epochs = 1 if smoke else tcfg["epochs"]
    accum = tcfg["grad_accum"]
    step = 0
    preview_dir = ckpt_dir / "preview" / "cvae"
    loss_log = LossLogger(ckpt_dir / "logs" / "cvae_loss.csv",
                          fields=["loss", "rec", "kl"])
    sample_batch = next(iter(loader))                        # frozen batch for visual progress
    epoch_bar = pbar(range(start_epoch, epochs), desc="CVAE epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        iter_bar = pbar(loader, desc=f"  CVAE ep {epoch+1}/{epochs}", unit="batch", leave=False)
        running = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "n": 0}
        opt.zero_grad()
        for i, (x, cond) in enumerate(iter_bar):
            x = x.to(device, non_blocking=True)
            region_id = cond["region_id"].to(device)
            z_pos = cond["z_pos"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                x_rec, mu, logvar, _ = model(x, region_id, z_pos)
                loss, rec, kl = cvae_loss(x, x_rec, mu, logvar, tcfg["kl_weight"])
                loss = loss / accum

            # NaN guard: skip the batch if loss is non-finite (would poison the model).
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue

            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                # Gradient clipping at L2 norm = 1.0 prevents single-batch explosions.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            running["loss"] += float(loss.detach()) * accum * x.size(0)
            running["rec"] += float(rec) * x.size(0)
            running["kl"] += float(kl) * x.size(0)
            running["n"] += x.size(0)
            step += 1
            iter_bar.set_postfix(loss=f"{running['loss']/running['n']:.3f}",
                                 rec=f"{running['rec']/running['n']:.3f}",
                                 kl=f"{running['kl']/running['n']:.2f}")
            if smoke and i >= 4:
                break

        iter_bar.close()
        avg_loss = running["loss"] / max(running["n"], 1)
        avg_rec = running["rec"] / max(running["n"], 1)
        avg_kl = running["kl"] / max(running["n"], 1)
        epoch_bar.set_postfix(loss=f"{avg_loss:.3f}")
        loss_log.log(epoch + 1, loss=round(avg_loss, 4),
                     rec=round(avg_rec, 4), kl=round(avg_kl, 4))
        try:
            save_cvae_previews(model, sample_batch, preview_dir, epoch + 1, device)
        except Exception as e:
            print(f"[warn] CVAE preview save failed: {e}")
        ckpt_path = ckpt_dir / f"cvae_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "cfg": mcfg}, ckpt_path)
        latest = ckpt_dir / "cvae_latest.pt"
        torch.save({"model": model.state_dict(), "cfg": mcfg}, latest)
    epoch_bar.close()
    return str(ckpt_dir / "cvae_latest.pt")


# ----- Stage B: Latent diffusion UNet -------------------------------------

class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


def train_diffusion(cfg: dict, cvae_ckpt: str | None = None):
    device = _device(cfg["training"]["device"])
    tcfg = cfg["training"]["diffusion"]
    mcfg = cfg["model"]["diffusion"]
    cvae_mcfg = cfg["model"]["cvae"]
    smoke = cfg["training"].get("smoke", False)

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    if cvae_ckpt is None:
        cvae_ckpt = str(ckpt_dir / "cvae_latest.pt")
    if not Path(cvae_ckpt).exists():
        raise FileNotFoundError(f"CVAE checkpoint missing: {cvae_ckpt}. Train CVAE first.")

    with Spinner("Loading CVAE checkpoint"):
        cvae = CVAE(in_channels=cvae_mcfg["in_channels"], base_channels=cvae_mcfg["base_channels"],
                    latent_channels=cvae_mcfg["latent_channels"], latent_size=cvae_mcfg["latent_size"],
                    cond_dim=cvae_mcfg["cond_dim"]).to(device)
        cvae.load_state_dict(torch.load(cvae_ckpt, map_location=device)["model"])
        cvae.eval()
        for p in cvae.parameters():
            p.requires_grad_(False)

    with Spinner("Building LatentUNet + DDPM schedule on " + device):
        unet = LatentUNet(
            image_size=mcfg["image_size"], in_channels=mcfg["in_channels"],
            base_channels=mcfg["base_channels"], channel_mults=mcfg["channel_mults"],
            num_res_blocks=mcfg["num_res_blocks"], cond_dim=mcfg["cond_dim"],
        ).to(device)
        sched = DiffusionSchedule.make(mcfg["timesteps"], mcfg["beta_schedule"]).to(device)
        opt = torch.optim.AdamW(unet.parameters(), lr=tcfg["lr"])
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg["training"]["mixed_precision"]))
        ema = _EMA(unet, decay=tcfg["ema_decay"])

    # Resume from last saved diffusion_epochN.pt if requested.
    start_epoch = 0
    if cfg["training"].get("resume", False):
        existing = sorted(ckpt_dir.glob("diffusion_epoch*.pt"),
                          key=lambda p: int(p.stem.replace("diffusion_epoch", "")))
        if existing:
            last = existing[-1]
            with Spinner(f"Resuming diffusion from {last.name}"):
                state = torch.load(last, map_location=device)
                unet.load_state_dict(state["model"])
                if "ema" in state:
                    ema.shadow.load_state_dict(state["ema"])
            start_epoch = int(last.stem.replace("diffusion_epoch", ""))

    loader = _make_loader(cfg, tcfg["batch_size"])

    epochs = 1 if smoke else tcfg["epochs"]
    accum = tcfg["grad_accum"]
    T = mcfg["timesteps"]
    preview_dir = ckpt_dir / "preview" / "diffusion"
    loss_log = LossLogger(ckpt_dir / "logs" / "diffusion_loss.csv", fields=["loss"])
    epoch_bar = pbar(range(start_epoch, epochs), desc="Diffusion epochs", unit="epoch")
    for epoch in epoch_bar:
        unet.train()
        iter_bar = pbar(loader, desc=f"  Diff ep {epoch+1}/{epochs}", unit="batch", leave=False)
        running = {"loss": 0.0, "n": 0}
        opt.zero_grad()
        for i, (x, cond) in enumerate(iter_bar):
            x = x.to(device, non_blocking=True)
            region_id = cond["region_id"].to(device)
            z_pos = cond["z_pos"].to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                cond_vec = cvae.cond(region_id, z_pos)
                mu, _ = cvae.encode(x, cond_vec)
                z0 = mu                                     # use posterior mean as target latent

            # Guard against a poisoned CVAE producing non-finite latents.
            if not torch.isfinite(z0).all():
                continue

            t = torch.randint(0, T, (x.size(0),), device=device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                z_t, noise = q_sample(z0, t, sched)
                eps_pred = unet(z_t, t, region_id, z_pos)
                loss = F.mse_loss(eps_pred, noise) / accum

            # NaN guard on UNet output.
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue

            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(unet)

            running["loss"] += float(loss.detach()) * accum * x.size(0)
            running["n"] += x.size(0)
            iter_bar.set_postfix(loss=f"{running['loss']/running['n']:.4f}")
            if smoke and i >= 4:
                break

        iter_bar.close()
        avg_loss = running["loss"] / max(running["n"], 1)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")
        loss_log.log(epoch + 1, loss=round(avg_loss, 5))
        try:
            save_diffusion_previews(
                ema.shadow, cvae, sched, preview_dir, epoch + 1, device,
                latent_size=cvae_mcfg["latent_size"],
                latent_channels=cvae_mcfg["latent_channels"],
                sampler_steps=30,
            )
        except Exception as e:
            print(f"[warn] diffusion preview save failed: {e}")
        save = {"model": unet.state_dict(), "ema": ema.shadow.state_dict(), "cfg": mcfg}
        torch.save(save, ckpt_dir / f"diffusion_epoch{epoch+1}.pt")
        torch.save(save, ckpt_dir / "diffusion_latest.pt")

    epoch_bar.close()
    return str(ckpt_dir / "diffusion_latest.pt")
