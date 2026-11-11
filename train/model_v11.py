"""v11.1's network (2026-10-23): v7's U-Net + a DEEP CONTEXT stage for 2x frames.

Why: run at 192x336, the v7 U-Net's deepest level (and its only attention) sits at 24x42 instead of 12x21, so every
deep feature sees half as much of the street as at 96x168 (the receptive-field mismatch HiDiffusion, ECCV 2024, blames
for high-resolution failures). v11 (the plain 2x run) melted sooner than v9 (judge: 0.60 vs 0.43).
The deep context stage gives the 2x model its 12x21 "whole street at once" view back:
    h -> h + out(up2(b2(b1(avgpool2(h)))))           right before the middle blocks
b1 = a ResBlock with attention, b2 = a plain ResBlock, at 12x21: the same shapes as the middle blocks, so they can start
as copies of v9's own middle blocks (trained to read a street at 12x21). `out` (a 3x3 conv) starts at ZERO, so a fresh
v11.1 dreams exactly like its parent; the stage fades in as it trains.
Built by model_v7.build_model_v7 whenever the config has deep_ctx=True (so play.py, the judges and the trainers that
build models through it load v11.1 saves unchanged)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResBlock
from model_v7 import DEFAULT_CFG_V7, DenoiserV7, UNetV7
from model_v2 import check_frame_size


class DeepContext(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.b1 = ResBlock(ch, ch, cond_dim, attn=True)
        self.b2 = ResBlock(ch, ch, cond_dim)
        self.out = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, cond):
        d = F.avg_pool2d(h, 2)
        d = self.b2(self.b1(d, cond), cond)
        return h + self.out(F.interpolate(d, size=h.shape[-2:], mode="nearest"))


class UNetV11(UNetV7):
    def __init__(self, cond_dim=256, channels=(64, 128, 256, 384), **kw):
        super().__init__(cond_dim=cond_dim, channels=channels, **kw)
        self.deep = DeepContext(channels[-1], cond_dim)

    def forward(self, x_noisy, c_noise, c_noise_cond, c_noise_frames, context, actions, colour, far_act):
        """UNetV7.forward word for word, plus the deep context stage right before the middle blocks."""
        cond = torch.cat([self.noise_emb(c_noise), self.cond_noise_emb(c_noise_cond),
                          self.key_embedding(actions)], dim=1)
        cond = self.cond_mlp(cond)
        cond = cond + self.frame_noise_mlp(c_noise_frames.to(cond.dtype))
        cond = cond + self.colour_mlp(colour.to(cond.dtype))
        far_act = far_act.to(cond.dtype)
        n0 = self.band_sizes[0]
        cond = cond + self.far_act_mlp(far_act[:, :n0])
        a = n0
        for n, mlp in zip(self.band_sizes[1:], self.extra_act_mlps):
            cond = cond + mlp(far_act[:, a:a + n])
            a += n
        cond = cond + self.motion_mlp(far_act[:, a:a + self.motion_in])

        h = self.conv_in(torch.cat([x_noisy, context], dim=1))
        skips = []
        for level, extra, down in zip(self.down, self.down_extra, self.downsample):
            for blk in level:
                h = blk(h, cond)
                skips.append(h)
            for blk in extra:
                h = blk(h, cond)
            h = down(h)
        h = self.deep(h, cond)                                   # NEW (v11.1)
        h = self.mid2(self.mid1(h, cond), cond)
        for level, extra, up in zip(self.up, self.up_extra, self.upsample):
            for blk in level:
                h = blk(torch.cat([h, skips.pop()], dim=1), cond)
            for blk in extra:
                h = blk(h, cond)
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class DenoiserV11(DenoiserV7):
    def __init__(self, K=8, mem_offsets=None, key_bands=None, extra_blocks=1, fp_offset=48, **unet_kw):
        super().__init__(K=K, mem_offsets=mem_offsets, key_bands=key_bands, extra_blocks=extra_blocks,
                         fp_offset=fp_offset, **unet_kw)
        old = self.unet
        self.unet = UNetV11(K=K, n_mem=self.n_mem, band_sizes=old.band_sizes, extra_blocks=extra_blocks, **unet_kw)
        self.motion_in = self.unet.motion_in


def build_model_v11(cfg):
    cfg = dict(DEFAULT_CFG_V7, **cfg)
    check_frame_size(*cfg["frame_hw"], halvings=len(cfg["channels"]))      # one more halving (the deep stage)
    return DenoiserV11(K=cfg["K"], mem_offsets=cfg["mem_offsets"], key_bands=cfg["key_bands"],
                       extra_blocks=cfg["extra_blocks"], fp_offset=cfg.get("fp_offset", 48),
                       channels=tuple(cfg["channels"]), blocks=cfg["blocks"], attn_levels=tuple(cfg["attn_levels"]))


def init_deep_from_mid(model, donor_state):
    """Copy a donor's middle blocks (unet.mid1 -> deep.b1, unet.mid2 -> deep.b2) into the deep stage. `out` stays zero."""
    sd = model.state_dict()
    n = 0
    for src, dst in (("unet.mid1.", "unet.deep.b1."), ("unet.mid2.", "unet.deep.b2.")):
        for k, v in donor_state.items():
            if k.startswith(src):
                sd[dst + k[len(src):]] = v.clone()
                n += 1
    model.load_state_dict(sd)
    return n
