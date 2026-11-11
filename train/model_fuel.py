"""LEARNED FUEL (2026-10-23): v7/v9's network + ONE extra input, a REAL reference frame.

Why: melting = the dream losing detail bit by bit, with nothing real left to
re-sharpen from. Dream fuel (dream_fuel.py) and edge fuel pasted real frames into the GHOST by hand (double exposure /
edge strips). Learned fuel hands the model a real frame with a similar layout (FuelLibrary.nearest of the newest
frame) as an extra input every frame and lets TRAINING decide what to borrow (texture, light, detail), keeping its
own layout. The match may come from another city: overlaid realities, on a dial.

HOW IT TRAVELS: as memory slot 0. context [B, 1 + n_mem, 3, H, W], oldest-first as always, so context[:, -1] is still
the newest frame (every "start from the previous frame" sampler keeps working). Callers that pass n_mem slots get a
ZERO reference (= "no fuel"; training drops it 20% of the time so that means something).
The reference enters through its own conv (ref_conv: 3 -> conv_in's channels, added to conv_in's output) that starts
at ZERO: at step 0 the model dreams exactly like its parent (v9 pick), and the new conv can have its own learning
rate. ref_gain (default 1) scales it at play time: 0 = off, >1 = borrow more.
Built by model_v7.build_model_v7 whenever the config has fuel=True."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_v3 import to_model_space
from dream_fuel import layout_desc
from model_v2 import check_frame_size
from model_v7 import DEFAULT_CFG_V7, DenoiserV7, UNetV7


class UNetF(UNetV7):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ref_conv = nn.Conv2d(3, self.conv_in.out_channels, 3, padding=1)
        nn.init.zeros_(self.ref_conv.weight)
        nn.init.zeros_(self.ref_conv.bias)
        self.ref_gain = 1.0

    def forward(self, x_noisy, c_noise, c_noise_cond, c_noise_frames, context, actions, colour, far_act):
        """UNetV7.forward word for word, except: the LAST 3 context channels are the reference, which goes
        through ref_conv and is added to conv_in's output."""
        ref, context = context[:, -3:], context[:, :-3]
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

        h = self.conv_in(torch.cat([x_noisy, context], dim=1)) + self.ref_gain * self.ref_conv(ref)   # NEW
        skips = []
        for level, extra, down in zip(self.down, self.down_extra, self.downsample):
            for blk in level:
                h = blk(h, cond)
                skips.append(h)
            for blk in extra:
                h = blk(h, cond)
            h = down(h)
        h = self.mid2(self.mid1(h, cond), cond)
        for level, extra, up in zip(self.up, self.up_extra, self.upsample):
            for blk in level:
                h = blk(torch.cat([h, skips.pop()], dim=1), cond)
            for blk in extra:
                h = blk(h, cond)
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class DenoiserF(DenoiserV7):
    def __init__(self, K=8, mem_offsets=None, key_bands=None, extra_blocks=1, fp_offset=48, **unet_kw):
        super().__init__(K=K, mem_offsets=mem_offsets, key_bands=key_bands, extra_blocks=extra_blocks,
                         fp_offset=fp_offset, **unet_kw)
        old = self.unet
        self.unet = UNetF(K=K, n_mem=self.n_mem, band_sizes=old.band_sizes, extra_blocks=extra_blocks, **unet_kw)
        self.motion_in = self.unet.motion_in

    def with_ref(self, context, ref=None):
        """context [B, n_mem or 1 + n_mem, 3, H, W] -> [B, 1 + n_mem, ...]; slot 0 = ref (zeros if none)."""
        if context.shape[1] == self.n_mem + 1:
            return context if ref is None else torch.cat([ref[:, None].to(context.dtype), context[:, 1:]], 1)
        r = torch.zeros_like(context[:, :1]) if ref is None else ref[:, None].to(context.dtype)
        return torch.cat([r, context], 1)

    def per_frame(self, sigma_cond, B, device):
        """[B, 1 + n_mem]: the reference slot never gets memory noise (it is real)."""
        sc = super().per_frame(sigma_cond, B, device)
        if sc.shape[1] == self.n_mem:
            sc = torch.cat([torch.zeros_like(sc[:, :1]), sc], 1)
        return sc

    def _unet(self, x_noisy, sigma, context, sc, actions, colour, far_act):
        context = self.with_ref(context)
        if sc.shape[1] == self.n_mem:
            sc = torch.cat([torch.zeros_like(sc[:, :1]), sc], 1)
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        b = lambda t: t[:, None, None, None]
        ctx = torch.cat([context[:, 1:].flatten(1, 2), context[:, 0]], 1) / self.sigma_data   # memory | reference
        sm = sc[:, 1:]
        code_max = self.cond_noise_code(sm[:, -self.K:].max(dim=1).values)
        code_frames = self.cond_noise_code(sm)
        F_out = self.unet(b(c_in) * x_noisy, c_noise, code_max, code_frames, ctx, actions, colour, far_act)
        return c_skip, c_out, F_out

    def loss(self, target, context, *a, **k):
        return super().loss(target, self.with_ref(context), *a, **k)

    def loss_ghost(self, target, context, *a, **k):
        return super().loss_ghost(target, self.with_ref(context), *a, **k)

    def sample(self, context, *a, **k):
        return super().sample(self.with_ref(context), *a, **k)


def build_model_fuel(cfg):
    cfg = dict(DEFAULT_CFG_V7, **cfg)
    check_frame_size(*cfg["frame_hw"], halvings=len(cfg["channels"]) - 1)
    return DenoiserF(K=cfg["K"], mem_offsets=cfg["mem_offsets"], key_bands=cfg["key_bands"],
                     extra_blocks=cfg["extra_blocks"], fp_offset=cfg.get("fp_offset", 48),
                     channels=tuple(cfg["channels"]), blocks=cfg["blocks"], attn_levels=tuple(cfg["attn_levels"]))


class RefPicker:
    """TRAINING: per sample, a real library frame whose layout is close to the query (the newest memory frame, as in
    play), picked at random among the `topk` closest, NEVER from within `avoid` frames of the sample's own frame (no
    peeking at the true next frame; the library holds only training-part frames, so the test walks never appear)."""

    def __init__(self, lib, avoid=300, topk=4):
        self.lib, self.avoid, self.topk = lib, avoid, topk
        self.idx_t = torch.from_numpy(lib.idx).to(lib.dev)

    @torch.no_grad()
    def pick(self, query, tix):
        d = torch.cdist(layout_desc(query.float()), self.lib.desc)                     # [B, N]
        d = d.masked_fill((self.idx_t[None] - tix[:, None]).abs() < self.avoid, float("inf"))
        top = d.topk(self.topk, dim=1, largest=False).indices
        j = top.gather(1, torch.randint(0, self.topk, (len(query), 1), device=query.device)).squeeze(1)
        return to_model_space(self.lib.u8.index_select(0, j)).float(), j


class RefKeeper:
    """PLAY / JUDGE: per row, the library frame closest in layout to the newest dreamed frame, KEPT for `every`
    frames (a steady reference = steady borrowed texture), then re-picked (FuelLibrary.nearest: never one of the
    last 200 picks, so the dream doesn't loop)."""

    def __init__(self, lib, every=10):
        self.lib, self.every, self.t, self.ref = lib, every, 0, None

    def get(self, newest):
        if self.ref is None or self.t % self.every == 0:
            picks = self.lib.nearest(newest.float())
            self.ref = torch.stack([self.lib.frame(i) for i in picks])
        self.t += 1
        return self.ref
