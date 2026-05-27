"""Generative model components.

Two networks:
  * `CVAE` -- convolutional VAE that compresses a (1, 256, 256) HU-windowed slice
    into a (latent_channels, 32, 32) latent. Conditioning (region + z-position)
    is mixed into encoder/decoder bottlenecks via FiLM-style scale/shift.
  * `LatentUNet` + `DDPM` -- small 2D UNet operating in the CVAE latent space,
    trained with the standard ε-prediction DDPM objective. Conditioning enters
    through time-embedding-style add-and-modulate.

Slice continuity is achieved at sampling time (see `generate.py`): adjacent
slices share the same region label and a smoothly-varying z_pos, plus an EMA
on the latent across consecutive z samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Shared building blocks
# =========================================================================

class GroupNorm(nn.GroupNorm):
    def __init__(self, channels: int):
        super().__init__(num_groups=min(32, channels), num_channels=channels)


def silu(x):
    return F.silu(x)


class ResBlock(nn.Module):
    """Pre-activation residual block with optional FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 0):
        super().__init__()
        self.norm1 = GroupNorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = GroupNorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)

    def forward(self, x, cond_vec=None):
        h = self.conv1(silu(self.norm1(x)))
        if self.cond_dim > 0 and cond_vec is not None:
            scale_shift = self.cond_proj(silu(cond_vec))
            scale, shift = scale_shift.chunk(2, dim=-1)
            h = self.norm2(h)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = silu(h)
        else:
            h = silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# =========================================================================
# Conditioning embedder (shared by CVAE and UNet)
# =========================================================================

class ConditionEmbedder(nn.Module):
    """Encodes (region_id, z_pos) -> cond_dim vector."""

    def __init__(self, cond_dim: int, n_regions: int = 2):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, cond_dim)
        self.z_proj = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.out = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, region_id: torch.Tensor, z_pos: torch.Tensor):
        r = self.region_emb(region_id)
        z = self.z_proj(z_pos.view(-1, 1))
        return self.out(r + z)


# =========================================================================
# CVAE
# =========================================================================

class CVAE(nn.Module):
    """Convolutional VAE with configurable depth.

    The number of /2 downsample stages is computed from input_size and
    latent_size:
        n_down = log2(input_size / latent_size)
    e.g. 256 -> 64 latent = 2 downsamples (less aggressive, more detail kept)
         256 -> 32 latent = 3 downsamples (more compression, faster training)

    Channel ladder doubles each stage, capped at base_channels * channel_cap.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 latent_channels: int = 4, latent_size: int = 32,
                 cond_dim: int = 16, n_regions: int = 2,
                 input_size: int = 256, channel_cap: int = 8):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.cond = ConditionEmbedder(cond_dim, n_regions)

        n_down = int(round(math.log2(input_size / latent_size)))
        if input_size != latent_size * (2 ** n_down):
            raise ValueError(
                f"input_size {input_size} must equal latent_size {latent_size} * 2^n"
            )

        ch = base_channels
        # Channel mults: 1, 2, 4, ... capped at channel_cap.
        # mults[0] = input stem, mults[i] = after i-th downsample, mults[-1] = bottleneck.
        mults = []
        m = 1
        for _ in range(n_down + 1):
            mults.append(m)
            m = min(m * 2, channel_cap)
        # so for n_down=2: mults = [1, 2, 4]; for n_down=3: [1, 2, 4, 8]

        # Encoder
        self.enc_in = nn.Conv2d(in_channels, ch * mults[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = ch * mults[0]
        for i in range(n_down):
            nxt = ch * mults[i + 1]
            self.enc_blocks.append(ResBlock(cur, nxt, cond_dim))
            self.downsamples.append(Downsample(nxt))
            cur = nxt
        self.enc_mid = ResBlock(cur, cur, cond_dim)

        # mu, logvar heads
        self.to_mu = nn.Conv2d(cur, latent_channels, 1)
        self.to_logvar = nn.Conv2d(cur, latent_channels, 1)

        # Decoder (mirror)
        self.from_latent = nn.Conv2d(latent_channels, cur, 1)
        self.dec_mid = ResBlock(cur, cur, cond_dim)
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(n_down)):
            nxt = ch * mults[i]
            self.upsamples.append(Upsample(cur))
            self.dec_blocks.append(ResBlock(cur, nxt, cond_dim))
            cur = nxt
        self.dec_out_block = ResBlock(cur, cur, cond_dim)
        self.dec_out = nn.Conv2d(cur, in_channels, 3, padding=1)

    def encode(self, x, cond_vec):
        h = self.enc_in(x)
        for blk, ds in zip(self.enc_blocks, self.downsamples):
            h = blk(h, cond_vec)
            h = ds(h)
        h = self.enc_mid(h, cond_vec)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond_vec):
        h = self.from_latent(z)
        h = self.dec_mid(h, cond_vec)
        for ds, blk in zip(self.upsamples, self.dec_blocks):
            h = ds(h)
            h = blk(h, cond_vec)
        h = self.dec_out_block(h, cond_vec)
        return torch.tanh(self.dec_out(h))   # output in [-1, 1] same as input window

    def forward(self, x, region_id, z_pos):
        cond_vec = self.cond(region_id, z_pos)
        mu, logvar = self.encode(x, cond_vec)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, cond_vec)
        return x_rec, mu, logvar, z


def cvae_loss(x, x_rec, mu, logvar, kl_weight: float):
    rec = F.l1_loss(x_rec, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl_weight * kl, rec.detach(), kl.detach()


# =========================================================================
# Latent UNet + DDPM
# =========================================================================

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class LatentUNet(nn.Module):
    """Small UNet on the CVAE latent grid. ε-prediction (DDPM)."""

    def __init__(self, image_size: int = 32, in_channels: int = 4,
                 base_channels: int = 128, channel_mults=(1, 2, 2, 4),
                 num_res_blocks: int = 2, cond_dim: int = 16,
                 n_regions: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.cond_dim = cond_dim

        self.time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_dim), nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.cond_embed = ConditionEmbedder(cond_dim, n_regions)
        self.cond_proj = nn.Linear(cond_dim, self.time_dim)

        ch = base_channels
        self.in_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Encoder
        chs = [ch]
        self.downs = nn.ModuleList()
        cur = ch
        for i, mult in enumerate(channel_mults):
            out = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(cur, out, self.time_dim))
                cur = out
                chs.append(cur)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(cur))
                chs.append(cur)

        # Mid
        self.mid1 = ResBlock(cur, cur, self.time_dim)
        self.mid2 = ResBlock(cur, cur, self.time_dim)

        # Decoder (mirrors encoder; eats skips)
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(cur + chs.pop(), out, self.time_dim))
                cur = out
            if i != 0:
                self.ups.append(Upsample(cur))

        self.out_norm = GroupNorm(cur)
        self.out_conv = nn.Conv2d(cur, in_channels, 3, padding=1)

        self.base_channels = base_channels

    def forward(self, x, t, region_id, z_pos):
        t_emb = sinusoidal_time_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)
        c = self.cond_embed(region_id, z_pos)
        cond_vec = t_emb + self.cond_proj(c)

        h = self.in_conv(x)
        skips = [h]

        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, cond_vec)
            else:
                h = layer(h)
            skips.append(h)

        h = self.mid1(h, cond_vec)
        h = self.mid2(h, cond_vec)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, cond_vec)
            else:
                h = layer(h)

        h = silu(self.out_norm(h))
        return self.out_conv(h)


# =========================================================================
# Diffusion schedule + helpers
# =========================================================================

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    @classmethod
    def make(cls, timesteps: int, schedule: str = "linear") -> "DiffusionSchedule":
        if schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = torch.clamp(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 1e-5, 0.999)
        else:
            raise ValueError(schedule)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        return cls(
            betas=betas, alphas=alphas, alphas_cumprod=ac,
            sqrt_alphas_cumprod=torch.sqrt(ac),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - ac),
        )

    def to(self, device):
        return DiffusionSchedule(
            betas=self.betas.to(device), alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(device),
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(device),
        )


def q_sample(x0, t, sched: DiffusionSchedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sa = sched.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    so = sched.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sa * x0 + so * noise, noise


@torch.no_grad()
def ddim_sample(model: LatentUNet, sched: DiffusionSchedule, shape, region_id, z_pos,
                steps: int = 50, device: str = "cpu", eta: float = 0.0,
                init_noise: torch.Tensor | None = None) -> torch.Tensor:
    """Fast DDIM sampling using a subset of the training timesteps."""
    T = len(sched.betas)
    step_seq = torch.linspace(T - 1, 0, steps + 1).long().to(device)
    x = init_noise if init_noise is not None else torch.randn(shape, device=device)
    for i in range(steps):
        t = step_seq[i].repeat(shape[0])
        t_next = step_seq[i + 1].repeat(shape[0])
        eps = model(x, t, region_id, z_pos)
        a_t = sched.alphas_cumprod[t].view(-1, 1, 1, 1)
        a_next = sched.alphas_cumprod[t_next].view(-1, 1, 1, 1)
        x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x0_pred = x0_pred.clamp(-4, 4)        # latent space, gentle clamp
        # DDIM update with eta=0 (deterministic).
        x = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * eps
    return x
