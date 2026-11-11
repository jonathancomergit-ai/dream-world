"""PAINTER v1 (2026-10-24): Dream World's second stage, "dreamer + painter".

The dreamer (v9, 96x168) decides WHAT the world is; the painter paints it at 192x336. DIAMOND's Counter-Strike
world model splits the job the same way (a low-res dynamics model + a separate diffusion upsampler).
Why not just sharper eyes (train/sr, 2026-10-22)? It is a one-shot sharpener trained on mild damage: late dreams
become "crisp mush", and texture shimmers while turning (no memory of its last frame). The painter is:
  - GENERATIVE: a small EDM diffusion model (few steps), so it can invent believable detail where the dream is vague;
  - TEMPORAL: it sees its OWN previous painted frame, slid by the turn (estimated from the low-res frames), so
    detail stays put from frame to frame;
  - trained on all 518k face-blurred 2x frames (data/cache_v11) with heavy, dream-like damage on the low-res input.

Inputs (model space [-1, 1]):
  x_noisy  [B, 3, 192, 336]  the picture being cleaned
  lr       [B, 3, 96, 168]   the dreamer's frame (bilinear 2x inside)
  prev     [B, 3, 192, 336]  the painter's previous output, slid by the turn (ignored where has_prev = 0)
  has_prev [B]               1 / 0
Sampling starts from the 2x low-res frame + noise sigma_max (2.0) and takes `steps` Euler steps; training puts 30% of
rows on exactly that start (the v7 lesson: train on what sampling starts from)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FourierFeatures, ResBlock
from train_v7 import shift_frames


class PainterNet(nn.Module):
    """A U-Net that works at HALF size: the 192x336 input is folded into 96x168 (pixel unshuffle: every 2x2 block of
    pixels becomes 4 channels), the U-Net runs there (96x168 -> 48x84 -> 24x42, attention in the middle), and the
    output is unfolded back to 192x336 (pixel shuffle). Same idea as sharper eyes; ~4x cheaper than full size."""

    def __init__(self, in_ch=9, channels=(128, 192, 256), blocks=2, cond_dim=256):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, channels=list(channels), blocks=blocks, cond_dim=cond_dim)
        self.noise_emb = FourierFeatures(cond_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim + 1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.fold, self.unfold = nn.PixelUnshuffle(2), nn.PixelShuffle(2)
        self.conv_in = nn.Conv2d(in_ch * 4, channels[0], 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        skip_ch, cur = [channels[0]], channels[0]
        for i, ch in enumerate(channels):
            level = nn.ModuleList()
            for _ in range(blocks):
                level.append(ResBlock(cur, ch, cond_dim))
                cur = ch
                skip_ch.append(cur)
            self.down.append(level)
            if i < len(channels) - 1:
                self.downsample.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))
                skip_ch.append(cur)
        self.mid1 = ResBlock(cur, cur, cond_dim, attn=True)
        self.mid2 = ResBlock(cur, cur, cond_dim)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(channels))):
            level = nn.ModuleList()
            for _ in range(blocks + 1):
                level.append(ResBlock(cur + skip_ch.pop(), ch, cond_dim))
                cur = ch
            self.up.append(level)
            if i > 0:
                self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                                   nn.Conv2d(cur, cur, 3, padding=1)))
        self.norm_out = nn.GroupNorm(min(32, cur // 4), cur)
        self.conv_out = nn.Conv2d(cur, 3 * 4, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, inp, c_noise, has_prev):
        cond = self.cond_mlp(torch.cat([self.noise_emb(c_noise), has_prev[:, None].float()], 1))
        h = self.conv_in(self.fold(inp))
        hs = [h]
        for i, level in enumerate(self.down):
            for blk in level:
                h = blk(h, cond)
                hs.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
                hs.append(h)
        h = self.mid2(self.mid1(h, cond), cond)
        for i, level in enumerate(self.up):
            for blk in level:
                h = blk(torch.cat([h, hs.pop()], 1), cond)
            if i < len(self.upsample):
                h = self.upsample[i](h)
        return self.unfold(self.conv_out(F.silu(self.norm_out(h))))


class Painter(nn.Module):
    def __init__(self, sigma_data=0.5, **net_kw):
        super().__init__()
        self.sigma_data = sigma_data
        self.net = PainterNet(**net_kw)

    def precond(self, sigma):
        sd = self.sigma_data
        c_in = 1 / (sigma ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        return c_in, c_skip, c_out, sigma.log() / 4

    @staticmethod
    def up2(lr):
        return F.interpolate(lr.float(), scale_factor=2, mode="bilinear", align_corners=False)

    def _F(self, x_noisy, sigma, lr_up, prev, has_prev):
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        b = lambda t: t[:, None, None, None]
        hp = has_prev.float()
        inp = torch.cat([b(c_in) * x_noisy, lr_up / self.sigma_data, prev * b(hp) / self.sigma_data], 1)
        return c_skip, c_out, self.net(inp, c_noise, hp)

    def forward(self, x_noisy, sigma, lr, prev, has_prev):
        """D(x; sigma): the denoised guess of the 192x336 frame."""
        c_skip, c_out, F_out = self._F(x_noisy, sigma, self.up2(lr), prev, has_prev)
        b = lambda t: t[:, None, None, None]
        return b(c_skip) * x_noisy + b(c_out) * F_out.float()

    def loss(self, target, lr, prev, has_prev, sigma, start_lr):
        """EDM loss. start_lr [B] bool: those rows are noised around the 2x low-res frame (= sampling's start)."""
        b = lambda t: t[:, None, None, None]
        lr_up = self.up2(lr)
        centre = torch.where(start_lr[:, None, None, None], lr_up, target)
        x_noisy = centre + b(sigma) * torch.randn_like(target)
        c_skip, c_out, F_out = self._F(x_noisy, sigma, lr_up, prev, has_prev)
        F_target = (target - b(c_skip) * x_noisy) / b(c_out)
        return F.mse_loss(F_out.float(), F_target)

    @staticmethod
    def karras(steps, sigma_max=2.0, sigma_min=2e-3, rho=7.0, device="cpu"):
        i = torch.arange(steps, device=device, dtype=torch.float32)
        s = (sigma_max ** (1 / rho) + i / max(steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        return torch.cat([s, torch.zeros(1, device=device)])

    @torch.no_grad()
    def sample(self, lr, prev=None, has_prev=None, steps=6, sigma_max=2.0, generator=None, noise=None):
        """noise: the starting noise [B, 3, 192, 336] (drawn here if None; play's CUDA graph passes its own)."""
        B = lr.shape[0]
        dev = lr.device
        lr_up = self.up2(lr)
        if prev is None:
            prev = torch.zeros_like(lr_up)
            has_prev = torch.zeros(B, device=dev)
        sig = self.karras(steps, sigma_max, device=dev)
        if noise is None:
            noise = torch.randn(lr_up.shape, device=dev, generator=generator)
        x = lr_up + noise * sig[0]
        for i in range(steps):
            s = sig[i].expand(B)
            den = self(x, s, lr, prev, has_prev)
            x = x + (x - den) / sig[i] * (sig[i + 1] - sig[i])
        return x.clamp(-1, 1)


def luma_profiles(x):
    """[B, 3, H, W] -> (column profile [B, W], row profile [B, H]) of luma, mean removed."""
    y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    c, r = y.mean(1), y.mean(2)
    return c - c.mean(1, keepdim=True), r - r.mean(1, keepdim=True)


@torch.no_grad()
def est_shift(prev, cur, max_dx=12, max_dy=6):
    """How far the view moved from prev to cur (both [B, 3, H, W], same size): whole pixels [B, 2] in
    train_v7.shift_frames' convention (col 0: + = content moved LEFT; col 1: + = content moved DOWN), from 1-D luma
    profiles. shift_frames(prev, est_shift(prev, cur)) ~ cur."""
    cp, rp = luma_profiles(prev.float())
    cc, rc = luma_profiles(cur.float())
    W, H = cp.shape[1], rp.shape[1]

    def best(a, b, m, sign):
        errs = []
        for d in range(-m, m + 1):
            k = sign * d                               # out[j] = a[j + k]
            if k >= 0:
                e = (b[:, :b.shape[1] - k] - a[:, k:]).abs().mean(1)
            else:
                e = (b[:, -k:] - a[:, :a.shape[1] + k]).abs().mean(1)
            errs.append(e)
        return torch.stack(errs, 1).argmin(1).float() - m

    dx = best(cp, cc, max_dx, 1)                       # content moved left by dx: cur[j] = prev[j + dx]
    dy = best(rp, rc, max_dy, -1)                      # content moved down by dy: cur[i] = prev[i - dy]
    return torch.stack([dx, dy], 1)


def slide(x, shift):
    """train_v7.shift_frames (whole pixels, edge pixels fill the gap)."""
    return shift_frames(x.float(), shift)


def build_painter(cfg=None):
    return Painter(**(cfg or {}))
