"""The Dream World model: a small action-conditioned diffusion "next frame" model (DIAMOND-style).

THE IDEA IN ONE BREATH
  Show the model the last K=4 frames you saw + the key you pressed. Hand it a picture of pure
  static. It cleans the static up, a few steps at a time, into what you'd see next. Feed that
  frame back in as the newest "memory", press the next key, repeat: that's walking in a dream.

WHAT'S IN THIS FILE
  UNet      the network that does the cleaning-up (a picture-in, picture-out conv net).
  Denoiser  wraps the UNet with EDM "preconditioning" (explained below), computes the training
            loss, and has the sampler that turns static into a frame.

EDM PRECONDITIONING (Karras et al. 2022), IN PLAIN WORDS
  During training we take a real next frame x, add Gaussian noise of strength sigma, and ask the
  model for x back. sigma ranges from tiny (0.002: almost clean) to huge (20: pure static).
  That's awkward for a network: inputs and the right answer change scale wildly with sigma.
  EDM fixes it with four little numbers that depend only on sigma (sigma_data = 0.5 is roughly
  how spread out real pixel values are, in our [-1, 1] scale):
    c_in    = 1 / sqrt(sigma^2 + sigma_data^2)
              shrink the noisy input so the network always sees values of about size 1.
    c_noise = ln(sigma) / 4
              the "how noisy is it" number we tell the network (a nicer scale than sigma itself).
    c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
    c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
              final answer = c_skip * (noisy input) + c_out * (network output).
              Low noise: c_skip ~ 1, so "the input is nearly right, just nudge it".
              High noise: c_skip ~ 0, so "the input is useless, paint the frame yourself".
  So the network only ever predicts a well-scaled CORRECTION, which makes training stable.
  The loss is just mean-squared error on that correction (EDM's per-sigma weighting falls out).

CONTEXT NOISE AUGMENTATION (the anti-melt trick, from DIAMOND / GameNGen)
  In training, the 4 context frames are always REAL, crisp video. When playing, they're the
  model's own slightly-wrong dreams. Small errors pile up and the dream melts into mush within
  seconds. Fix: in training we add a random bit of noise to the context frames too, and TELL the
  model how much (sigma_cond). It learns "the past may be smudged, don't copy the smudge, fix it".
  When playing you can dial that context noise up or down (the "dream dial").
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 9   # 0 still, 1 fwd, 2 back, 3 strafe L, 4 strafe R, 5 turn L, 6 turn R, 7 up, 8 down


# ---------------------------------------------------------------- building blocks

class FourierFeatures(nn.Module):
    """Turns one number (like c_noise) into a vector of sines and cosines at many random
    frequencies. Networks find a 'wiggly' vector far easier to use than one raw number."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freqs", torch.randn(1, dim // 2) * 2 * math.pi)  # fixed, not learned

    def forward(self, x):                      # x: [B]
        f = x[:, None].float() * self.freqs
        return torch.cat([f.cos(), f.sin()], dim=1)


class AdaGroupNorm(nn.Module):
    """GroupNorm whose scale and shift are chosen by the conditioning vector
    (noise level + context noise level + keys pressed). This is how 'which key' and 'how noisy'
    reach every layer: each layer gets told how to tilt its features."""

    def __init__(self, ch, cond_dim, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch // 4), ch, affine=False)
        self.lin = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x, cond):
        scale, shift = self.lin(cond)[:, :, None, None].chunk(2, dim=1)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            # Speed: autocast would run GroupNorm in float32 (twice the memory traffic). The CUDA
            # kernel still adds up its statistics in float32 internally, so doing it in bf16 is
            # safe, and it made training ~25% faster on the 5090.
            with torch.autocast("cuda", enabled=False):
                h = F.group_norm(x.to(torch.bfloat16), self.norm.num_groups, eps=self.norm.eps)
            return h * (1 + scale.to(h.dtype)) + shift.to(h.dtype)
        return self.norm(x) * (1 + scale) + shift


class ResBlock(nn.Module):
    """Two 3x3 convs with conditioned norms, plus a skip connection. Optional self-attention
    afterwards (only used at the small resolutions, where it's cheap). Attention lets every
    spot in the picture look at every other spot, e.g. 'the street on the left continues on the
    right', which plain convs at this size can't see far enough to know."""

    def __init__(self, cin, cout, cond_dim, attn=False):
        super().__init__()
        self.n1 = AdaGroupNorm(cin, cond_dim)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = AdaGroupNorm(cout, cond_dim)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)           # each block starts as "do nothing": stabler start
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attn = SelfAttention(cout, cond_dim) if attn else None

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x, cond)))
        h = self.c2(F.silu(self.n2(h, cond)))
        x = self.skip(x) + h
        if self.attn is not None:
            x = self.attn(x, cond)
        return x


class SelfAttention(nn.Module):
    def __init__(self, ch, cond_dim, head_dim=64):
        super().__init__()
        self.heads = max(1, ch // head_dim)
        self.norm = AdaGroupNorm(ch, cond_dim)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x, cond)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        # [B, heads, tokens, head_dim] for PyTorch's fast fused attention
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.out(y)


# ---------------------------------------------------------------- the U-Net

class UNet(nn.Module):
    """A U-shaped network: squeeze the picture down (64x112 -> 32x56 -> 16x28 -> 8x14) to
    understand the big picture, then grow it back up, with 'skip' wires carrying fine detail
    across the U so it isn't lost.

    Input: 15 channels = noisy next frame (3) + 4 context frames (4 x 3 = 12), stacked.
    Output: 3 channels, the correction described in the EDM notes above."""

    def __init__(self, K=4, channels=(64, 128, 256, 384), blocks=2, attn_levels=(3,),
                 cond_dim=256, act_dim=64):
        super().__init__()
        self.K = K
        # --- conditioning: noise level, context-noise level, and the K keys pressed
        self.noise_emb = FourierFeatures(cond_dim)
        self.cond_noise_emb = FourierFeatures(cond_dim)
        # one embedding table per action slot: slot K-1 = the key that makes the NEXT frame
        # (the one you're pressing now), slots 0..K-2 = the keys between the context frames
        # (tells the model how it was already moving: momentum).
        self.act_emb = nn.Embedding(K * N_ACTIONS, act_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * cond_dim + K * act_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim))

        self.conv_in = nn.Conv2d(3 + 3 * K, channels[0], 3, padding=1)

        # --- down path
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skip_ch = []
        ch = channels[0]
        for lvl, c in enumerate(channels):
            level = nn.ModuleList()
            for _ in range(blocks):
                level.append(ResBlock(ch, c, cond_dim, attn=lvl in attn_levels))
                ch = c
                skip_ch.append(ch)
            self.down.append(level)
            # halve the size between levels (not after the last one)
            self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                                   if lvl < len(channels) - 1 else nn.Identity())

        # --- middle (smallest size, always with attention)
        self.mid1 = ResBlock(ch, ch, cond_dim, attn=True)
        self.mid2 = ResBlock(ch, ch, cond_dim)

        # --- up path (mirror image; each block also eats one skip wire)
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for lvl in reversed(range(len(channels))):
            c = channels[lvl]
            level = nn.ModuleList()
            for _ in range(blocks):
                level.append(ResBlock(ch + skip_ch.pop(), c, cond_dim, attn=lvl in attn_levels))
                ch = c
            self.up.append(level)
            self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                               nn.Conv2d(ch, ch, 3, padding=1))
                                 if lvl > 0 else nn.Identity())

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x_noisy, c_noise, c_noise_cond, context, actions):
        """x_noisy [B,3,H,W] (already scaled by c_in), c_noise [B], c_noise_cond [B],
        context [B, 3K, H, W], actions [B, K] int64 in 0..8."""
        B = x_noisy.shape[0]
        slots = torch.arange(self.K, device=actions.device)[None] * N_ACTIONS + actions
        cond = torch.cat([self.noise_emb(c_noise), self.cond_noise_emb(c_noise_cond),
                          self.act_emb(slots).reshape(B, -1)], dim=1)
        cond = self.cond_mlp(cond)

        h = self.conv_in(torch.cat([x_noisy, context], dim=1))
        skips = []
        for level, down in zip(self.down, self.downsample):
            for blk in level:
                h = blk(h, cond)
                skips.append(h)
            h = down(h)
        h = self.mid2(self.mid1(h, cond), cond)
        for level, up in zip(self.up, self.upsample):
            for blk in level:
                h = blk(torch.cat([h, skips.pop()], dim=1), cond)
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ---------------------------------------------------------------- EDM wrapper

class Denoiser(nn.Module):
    """UNet + EDM preconditioning + training loss + sampler.
    All frames live in [-1, 1] (pixel/127.5 - 1)."""

    def __init__(self, K=4, sigma_data=0.5, **unet_kw):
        super().__init__()
        self.K = K
        self.sigma_data = sigma_data
        self.unet = UNet(K=K, **unet_kw)

    # The four EDM numbers (see the notes at the top of the file).
    def precond(self, sigma):
        sd = self.sigma_data
        c_in = 1 / (sigma ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.log() / 4
        return c_in, c_skip, c_out, c_noise

    @staticmethod
    def cond_noise_code(sigma_cond):
        # same "log / 4" idea for the context noise; clamp so sigma_cond = 0 isn't log(0) = -inf
        return sigma_cond.clamp(min=1e-3).log() / 4

    def forward(self, x_noisy, sigma, context, sigma_cond, actions):
        """D(x; sigma): the denoised guess of the clean next frame.
        x_noisy [B,3,H,W], sigma [B], context [B,K,3,H,W] (already noised by sigma_cond),
        sigma_cond [B], actions [B,K]."""
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        b = lambda t: t[:, None, None, None]
        # context is divided by sigma_data so its scale (~1) matches the scaled noisy input
        ctx = context.flatten(1, 2) / self.sigma_data
        F_out = self.unet(b(c_in) * x_noisy, c_noise, self.cond_noise_code(sigma_cond), ctx, actions)
        return b(c_skip) * x_noisy + b(c_out) * F_out.float(), F_out

    def loss(self, target, context, actions, sigma, sigma_cond):
        """One training loss. target [B,3,H,W] clean next frame; sigma [B] noise for it;
        context [B,K,3,H,W] clean context; sigma_cond [B] how much noise to smudge the context with."""
        b = lambda t: t[:, None, None, None]
        # anti-melt: smudge the context with Gaussian noise of strength sigma_cond (often small, sometimes 0)
        context = context + sigma_cond[:, None, None, None, None] * torch.randn_like(context)
        noise = torch.randn_like(target)
        x_noisy = target + b(sigma) * noise
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        ctx = context.flatten(1, 2) / self.sigma_data
        F_out = self.unet(b(c_in) * x_noisy, c_noise, self.cond_noise_code(sigma_cond), ctx, actions)
        # what the network SHOULD have output so that c_skip*x_noisy + c_out*F == target
        F_target = (target - b(c_skip) * x_noisy) / b(c_out)
        return F.mse_loss(F_out.float(), F_target)

    # ------------------------------------------------------------ sampling (dreaming)

    @staticmethod
    def karras_sigmas(steps, sigma_min=2e-3, sigma_max=5.0, rho=7.0, device="cpu"):
        """The ladder of noise levels we walk down while cleaning static into a frame.
        rho=7 spaces the rungs so more of them sit at the low-noise end, where detail is decided."""
        i = torch.arange(steps, device=device, dtype=torch.float32)
        s = (sigma_max ** (1 / rho) + i / max(steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        return torch.cat([s, torch.zeros(1, device=device)])   # finish at exactly 0 = clean

    @torch.no_grad()
    def sample(self, context, actions, steps=5, sigma_cond=0.0, heun=False, generator=None,
               start="noise", sigma_max=5.0):
        """Dream the next frame. context [B,K,3,H,W] in [-1,1], actions [B,K] (last = key now).
        steps: fewer = faster and smearier. sigma_cond: smudge the context a little first
        (the dream dial) and tell the model we did. heun=True is a 2x-cost more accurate solver."""
        B, _, C, H, W = context.shape
        dev = context.device
        sc = torch.full((B,), float(sigma_cond), device=dev)
        if sigma_cond > 0:
            context = context + sigma_cond * torch.randn(context.shape, device=dev, generator=generator)
        sigmas = self.karras_sigmas(steps, sigma_max=sigma_max, device=dev)
        # WHERE THE DREAM STARTS (2026-10-20 root-cause fix):
        #   start="noise": pure static (the old way). Training never showed the model pure static: even
        #     its heaviest noise still hid a faint ghost of the real frame, so it learned to read that
        #     ghost, and from pure static its first guess was a hallucinated "average city".
        #   start="prev": the newest memory frame covered in the same static. That IS what training
        #     looks like (the next frame is nearly the last one), so the model only has to work out
        #     what changed. First-frame error 0.221 -> 0.037 on v4 (train/_exp/sampler_trace.py).
        x = torch.randn((B, C, H, W), device=dev, generator=generator) * sigmas[0]
        if start == "prev":
            x = x + context[:, -1].float()          # (context was smudged above if sigma_cond > 0: fine)
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            den, _ = self(x, s.expand(B), context, sc, actions)
            d = (x - den) / s                       # which way is "less noisy"
            x_next = x + d * (s_next - s)           # Euler step down the ladder
            if heun and s_next > 0:                 # Heun: look again at the landing spot, average
                den2, _ = self(x_next, s_next.expand(B), context, sc, actions)
                d2 = (x_next - den2) / s_next
                x_next = x + (d + d2) / 2 * (s_next - s)
            x = x_next
        return x.clamp(-1, 1)


# The architecture settings. Saved inside every checkpoint, so play.py (and --resume) rebuild
# exactly the network that was trained even if these defaults change later.
# 64x112 -> 32x56 -> 16x28 -> 8x14. Attention only at 8x14 (level 3) + the middle: at 16x28
# it cost ~25% speed for little gain at this size. The last level is a bit wider (384) because
# at 8x14 extra channels are nearly free and that's where the "big picture" is understood.
DEFAULT_CFG = {"K": 4, "channels": [64, 128, 256, 384], "blocks": 2, "attn_levels": [3]}


def build_model(cfg=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    return Denoiser(K=cfg["K"], channels=tuple(cfg["channels"]), blocks=cfg["blocks"],
                    attn_levels=tuple(cfg["attn_levels"]))


if __name__ == "__main__":
    m = build_model()
    n = sum(p.numel() for p in m.parameters())
    print(f"parameters: {n / 1e6:.1f} M")
