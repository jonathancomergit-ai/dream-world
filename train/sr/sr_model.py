"""The "sharper eyes" networks: a small 2x upscaler (generator) and a small patch critic (discriminator).

Generator (SharpEyes): 96x168 -> 192x336, all the heavy work at the LOW resolution.
    x (model space, [-1, 1])
      -> 3x3 conv to C channels
      -> n_blocks residual blocks (conv-LeakyReLU-conv, + input * 1), all at 96x168
      -> a second, coarse path: the features at half size (48x84), a few blocks there, back up and added in
         (wider view, so it can see what a smear "should" have been - cheap, 1/4 the pixels)
      -> 3x3 conv to 4C, pixel shuffle x2 (-> C channels at 192x336), LeakyReLU, 3x3 conv to 3
      -> + bilinear 2x upscale of x (so the net only learns the missing detail, colours are anchored to the input)
    No normalisation layers (they shift colours and break small-batch / batch-1 behaviour).

Critic (PatchCritic): spectral-norm conv net on [HR picture, bilinear-up input] (6 channels): it judges ~70x70 patches
    as "real 2x footage" or "upscaler output", given what the low-res input was (so detail has to fit the input).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)              # each block starts as identity: stable start
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.leaky_relu(self.c1(x), 0.2))


class SharpEyes(nn.Module):
    def __init__(self, c=64, n_blocks=10, n_coarse=4):
        super().__init__()
        self.cfg = {"c": c, "n_blocks": n_blocks, "n_coarse": n_coarse}
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(n_blocks)])
        self.down = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.coarse = nn.Sequential(*[RB(c) for _ in range(n_coarse)])
        self.up_c = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.up_c.weight)
        nn.init.zeros_(self.up_c.bias)
        self.body_end = nn.Conv2d(c, c, 3, padding=1)
        self.to_hr = nn.Conv2d(c, 4 * c, 3, padding=1)
        self.hr_conv = nn.Conv2d(c, c, 3, padding=1)
        self.out = nn.Conv2d(c, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)              # starts as plain bilinear upscaling
        nn.init.zeros_(self.out.bias)

    def detail(self, x):
        """The learned part only: what gets ADDED to the bilinear 2x upscale of x. [B, 3, 192, 336]."""
        f0 = self.head(x)
        f = self.body(f0)
        g = self.coarse(F.leaky_relu(self.down(f), 0.2))
        f = f + self.up_c(F.interpolate(g, size=f.shape[-2:], mode="bilinear", align_corners=False))
        f = self.body_end(f) + f0
        f = F.leaky_relu(F.pixel_shuffle(self.to_hr(f), 2), 0.2)
        f = F.leaky_relu(self.hr_conv(f), 0.2)
        return self.out(f)

    def forward(self, x):
        """x [B, 3, 96, 168] in [-1, 1] -> [B, 3, 192, 336] (not clamped; clamp for display)."""
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + self.detail(x)


class PatchCritic(nn.Module):
    """6 -> 64 -> 128 -> 256 -> 256 -> 1 logits per patch (spectral norm everywhere)."""

    def __init__(self, c=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(6, c, 3, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(c, c, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(c, 2 * c, 3, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(2 * c, 2 * c, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(2 * c, 4 * c, 3, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * c, 4 * c, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * c, 4 * c, 3, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * c, 1, 3, padding=1)))

    def forward(self, hr, lr_up):
        return self.net(torch.cat([hr, lr_up], 1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
