"""Dream World model, version 6a: v5's network with LONGER MEMORY and a DEEPER U-Net.

    (v5 = model_v3.py's network, trained with the 2026-10-20 "previous frame" dream start.)

WHY A v6
  v5 walks 15-20 s without melting, but its memory only reaches 4.8 s back. Turn around for a
  few seconds and the street you came from has fallen out of its memory, so it invents a new
  one ("invents new scenery when you turn", PLAN.md). v6a gives it two things:
    1. LONGER MEMORY: 8 extra "very far" frames, one every 1.5 s, reaching ~16.8 s back.
    2. A DEEPER NETWORK: one extra residual block per U-Net level (down and up), so it has more
       room to learn what to do with all that memory.
  Both are added so that ON DAY ONE v6a dreams EXACTLY what v5 dreams (checked numerically by
  train_v6.py before training starts), and training then teaches it to use the new parts.

THE MEMORY, AS AN EXPLICIT LIST (cfg["mem_offsets"])
  Every memory slot is "the frame N frames before the one being dreamed". v3/v5 computed that
  list from far_n/far_stride; v6 stores the list itself in the config, so a slot can never quietly
  change its meaning. Oldest first:

      very far (NEW)                      far (v5's)                  recent (v5's)
      168 153 138 123 108 93 78 63  |  48 43 38 33 28 23 18 13  |  8 7 6 5 4 3 2 1   -> next frame
      (16.8 s ............ 6.3 s)      (4.8 s ......... 1.3 s)      (0.8 s .. 0.1 s)

  conv_in (the first layer) stacks them as colour channels, in that order:
      channels  0..2    the noisy next frame
                3..26   the 8 very-far frames   (NEW, weights start at ZERO = "ignore")
               27..50   the 8 far frames        (v5's weights, copied)
               51..74   the 8 recent frames     (v5's weights, copied)
  Each memory frame also has its own "how smudged is it" number (per-frame context noise), so
  frame_noise_mlp's first layer gets 8 new input columns, also starting at zero.

THE KEYS, SUMMARISED PER STRETCH (cfg["key_bands"])
  The network gets the 8 most recent keys one by one (as v0-v5). For the older stretches it gets
  a summary: for each gap between two old memory frames, what share of the time each of the 9
  keys was pressed. Without it, it couldn't tell "that old frame is from before I turned around"
  from "that old frame is straight ahead". A band [from, to, chunk] = the keys pressed between
  the frames `from` and `to` frames back, cut into pieces of `chunk` keys:
      band 0  [48, 8, 5]    v5's far stretch:      8 pieces x 9 shares -> far_act_mlp (v5's)
      band 1  [168, 48, 15] the very-far stretch:  8 pieces x 9 shares -> extra_act_mlps[0] (NEW,
                                                    last layer zero = "add nothing" at first)
  All bands travel as ONE flat input (far_act, band 0 first), so every caller that handed v5 its
  far-keys summary hands v6 this longer one in the same place: GraphedSampler, play.py, ...

THE EXTRA BLOCKS (cfg["extra_blocks"], train_v6.py --extra-blocks)
  A residual block computes  out = in + branch(in), where branch = norm -> conv -> norm -> conv.
  If the LAST conv of the branch is all zeros, branch(in) = 0 and the block is exactly "out = in":
  it does nothing. That's how every new block starts (model.py's ResBlock already zero-inits its
  last conv; we zero it again here to be explicit). Gradients still reach that zero conv, so it
  starts learning on step 1, and the rest of the branch follows.
  Where they go: at each level of the U-Net, after that level's v5 blocks.
    down path: after the level's blocks have handed their "skip wires" across the U, before
               shrinking. So the skip wires carry exactly what v5's did, and the extra block only
               works on what goes DOWN to the coarser level.
    up path:   after the level's blocks have eaten their skip wires, before growing.
  The extra blocks don't add skip wires, so v5's up-path blocks still get exactly the inputs
  they were trained on. Blocks at attention levels (the coarsest, 12x21) get attention too.

WARM START FROM v5: warm_start_from_v5() below. Anything with the same name and shape is copied;
conv_in and frame_noise_mlp.0 are copied slot by slot BY OFFSET (slot "13 frames back" in v6
gets v5's slot "13 frames back"); everything else new keeps its zero ending.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import N_ACTIONS, ResBlock
from model_v1 import DenoiserV1, UNetV1
from model_v2 import check_frame_size
from model_v3 import COLOUR_DIM, DEFAULT_CFG_V3

# v5's memory (model_v3 with far_n=8, far_stride=5), as an offsets list, oldest first
V5_OFFSETS = [48, 43, 38, 33, 28, 23, 18, 13, 8, 7, 6, 5, 4, 3, 2, 1]


def v6_offsets(vfar_n=8, vfar_stride=15, base=48):
    """v5's 16 slots + vfar_n very-far slots every vfar_stride frames before v5's oldest (48).
    Default: [168, 153, 138, 123, 108, 93, 78, 63] + V5_OFFSETS."""
    return [base + vfar_stride * (vfar_n - j) for j in range(vfar_n)] + list(V5_OFFSETS)


def v6_key_bands(vfar_n=8, vfar_stride=15, base=48):
    """Band 0 = v5's far keys [48 -> 8, pieces of 5]; band 1 = the very-far keys."""
    return [[48, 8, 5], [base + vfar_stride * vfar_n, base, vfar_stride]]


# ---------------------------------------------------------------- the U-Net, v6

class UNetV6(UNetV1):
    """v5's U-Net (model_v3.UNetV3) with: a longer memory (n_mem frames in conv_in and in the
    per-frame noise numbers), one key-summary network per key band, and extra_blocks identity-start
    residual blocks per level on the down and up paths."""

    def __init__(self, K=8, n_mem=24, band_sizes=(72, 72), extra_blocks=1, cond_dim=256,
                 channels=(64, 128, 256, 384), attn_levels=(3,), **kw):
        super().__init__(K=K, cond_dim=cond_dim, frame_noise=True, channels=channels,
                         attn_levels=attn_levels, **kw)
        self.n_mem = n_mem
        self.band_sizes = list(band_sizes)
        ch0 = self.conv_in.out_channels
        # memory = n_mem frames, oldest first (see the channel map at the top)
        self.conv_in = nn.Conv2d(3 + 3 * n_mem, ch0, 3, padding=1)
        self.frame_noise_mlp = nn.Sequential(nn.Linear(n_mem, cond_dim), nn.SiLU(),
                                             nn.Linear(cond_dim, cond_dim))
        self.colour_mlp = nn.Sequential(nn.Linear(COLOUR_DIM, cond_dim), nn.SiLU(),
                                        nn.Linear(cond_dim, cond_dim))
        # band 0 keeps v5's name (so its weights copy by name); later bands are new
        self.far_act_mlp = nn.Sequential(nn.Linear(band_sizes[0], cond_dim), nn.SiLU(),
                                         nn.Linear(cond_dim, cond_dim))
        self.extra_act_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(n, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
            for n in band_sizes[1:])
        for mlp in [self.frame_noise_mlp, self.colour_mlp, self.far_act_mlp, *self.extra_act_mlps]:
            nn.init.zeros_(mlp[2].weight)          # each starts as "add nothing"
            nn.init.zeros_(mlp[2].bias)

        # --- the extra blocks (separate lists, so v5's block names stay exactly the same)
        self.extra_blocks = extra_blocks
        self.down_extra = nn.ModuleList()
        for lvl, c in enumerate(channels):
            self.down_extra.append(nn.ModuleList(
                ResBlock(c, c, cond_dim, attn=lvl in attn_levels) for _ in range(extra_blocks)))
        self.up_extra = nn.ModuleList()
        for lvl in reversed(range(len(channels))):            # same order as self.up
            c = channels[lvl]
            self.up_extra.append(nn.ModuleList(
                ResBlock(c, c, cond_dim, attn=lvl in attn_levels) for _ in range(extra_blocks)))
        for lst in (self.down_extra, self.up_extra):
            for level in lst:
                for blk in level:
                    zero_branch_end(blk)

    def forward(self, x_noisy, c_noise, c_noise_cond, c_noise_frames, context, actions, colour, far_act):
        """x_noisy [B,3,H,W] (scaled by c_in), c_noise [B], c_noise_cond [B] (worst RECENT smudge),
        c_noise_frames [B, n_mem], context [B, 3*n_mem, H, W] oldest first,
        actions [B, K] recent keys (last = key now), colour [B, 25], far_act [B, sum(band_sizes)]."""
        B = x_noisy.shape[0]
        slots = torch.arange(self.K, device=actions.device)[None] * N_ACTIONS + actions
        cond = torch.cat([self.noise_emb(c_noise), self.cond_noise_emb(c_noise_cond),
                          self.act_emb(slots).reshape(B, -1)], dim=1)
        cond = self.cond_mlp(cond)
        cond = cond + self.frame_noise_mlp(c_noise_frames.to(cond.dtype))
        cond = cond + self.colour_mlp(colour.to(cond.dtype))
        far_act = far_act.to(cond.dtype)
        n0 = self.band_sizes[0]
        cond = cond + self.far_act_mlp(far_act[:, :n0])          # (v5's order of additions, then the new ones)
        a = n0
        for n, mlp in zip(self.band_sizes[1:], self.extra_act_mlps):
            cond = cond + mlp(far_act[:, a:a + n])
            a += n

        h = self.conv_in(torch.cat([x_noisy, context], dim=1))
        skips = []
        for level, extra, down in zip(self.down, self.down_extra, self.downsample):
            for blk in level:
                h = blk(h, cond)
                skips.append(h)
            for blk in extra:                  # NEW: after the skip wires, before shrinking
                h = blk(h, cond)
            h = down(h)
        h = self.mid2(self.mid1(h, cond), cond)
        for level, extra, up in zip(self.up, self.up_extra, self.upsample):
            for blk in level:
                h = blk(torch.cat([h, skips.pop()], dim=1), cond)
            for blk in extra:                  # NEW: after the skip wires, before growing
                h = blk(h, cond)
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def zero_branch_end(blk):
    """Make a residual block an exact identity: zero the last conv of its residual branch
    (and the output of its attention, if any). out = in + 0."""
    nn.init.zeros_(blk.c2.weight)
    nn.init.zeros_(blk.c2.bias)
    if blk.attn is not None:
        nn.init.zeros_(blk.attn.out.weight)
        nn.init.zeros_(blk.attn.out.bias)


# ---------------------------------------------------------------- EDM wrapper, v6

class DenoiserV6(DenoiserV1):
    """v5's Denoiser (EDM maths, loss, sampler: model_v3.DenoiserV3) with the memory given as an
    explicit offsets list. Same call signature as DenoiserV3, so GraphedSampler (train_v3.py)
    and play.py's graph speed-up work unchanged:

    context    [B, n_mem, 3, H, W]      memory, OLDEST FIRST, at self.offsets
    sigma_cond number, [B] or [B, n_mem]: how much each memory frame was smudged
    actions    [B, K]                   recent keys (last = the key pressed now)
    colour     [B, 25] or None          colour fingerprint (None = blanked = all zeros)
    far_act    [B, sum of band sizes]   key summaries of the older stretches (key_summary())"""

    def __init__(self, K=8, mem_offsets=None, key_bands=None, extra_blocks=1, fp_offset=48,
                 sigma_data=0.5, **unet_kw):
        nn.Module.__init__(self)          # skip DenoiserV1.__init__: it would build a v1 UNet
        self.K = K
        self.offsets = list(mem_offsets or v6_offsets())
        self.key_bands = [list(b) for b in (key_bands or v6_key_bands())]
        # sanity: oldest first, the last K slots are the recent frames K..1 (v5's recent memory)
        assert self.offsets == sorted(self.offsets, reverse=True), "mem_offsets must be oldest first"
        assert self.offsets[-K:] == list(range(K, 0, -1)), "the last K memory slots must be offsets K..1"
        for f, t, c in self.key_bands:
            assert (f - t) % c == 0, f"key band {f}->{t} doesn't split into pieces of {c}"
        self.n_mem = len(self.offsets)
        self.history = max(self.offsets[0], max(b[0] for b in self.key_bands))   # frames (and keys) needed
        self.fp_offset = fp_offset        # colour fingerprint = the 8 real frames fp_offset..fp_offset-7 back
        self.sigma_data = sigma_data
        band_sizes = [(f - t) // c * N_ACTIONS for f, t, c in self.key_bands]
        self.unet = UNetV6(K=K, n_mem=self.n_mem, band_sizes=band_sizes, extra_blocks=extra_blocks, **unet_kw)

    def mem_offsets(self):
        """Where each memory frame sits, in frames BEFORE the frame being dreamed (oldest first)."""
        return list(self.offsets)

    def key_summary(self, keys):
        """keys [B, history] int: the keys pressed before the frame being dreamed, oldest first
        (keys[:, -1] = the key pressed now, which moves the newest memory frame on to the dreamed one).
        -> float [B, sum of band sizes]: for each band and each piece of it, the share of each of
        the 9 keys. Band [48, 8, 5] = keys[:, -48:-8] in pieces of 5 = exactly v5's far summary."""
        B, H = keys.shape
        assert H == self.history, f"key_summary wants the last {self.history} keys, got {H}"
        out = []
        for f, t, c in self.key_bands:
            a = keys[:, H - f:H - t].clamp(0, N_ACTIONS - 1).reshape(B, (f - t) // c, c)
            out.append(F.one_hot(a, N_ACTIONS).float().mean(dim=2).reshape(B, -1))
        return torch.cat(out, dim=1)

    def per_frame(self, sigma_cond, B, device):
        """Any sigma_cond shape -> float [B, n_mem]."""
        if not torch.is_tensor(sigma_cond):
            return torch.full((B, self.n_mem), float(sigma_cond), device=device)
        sigma_cond = sigma_cond.to(device=device, dtype=torch.float32)
        if sigma_cond.ndim == 1:
            return sigma_cond[:, None].expand(B, self.n_mem)
        return sigma_cond

    def _extras(self, colour, far_act, B, device):
        if colour is None:
            colour = torch.zeros(B, COLOUR_DIM, device=device)
        if far_act is None:
            far_act = torch.zeros(B, sum(self.unet.band_sizes), device=device)
        return colour.float(), far_act.float()

    def _unet(self, x_noisy, sigma, context, sc, actions, colour, far_act):
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        b = lambda t: t[:, None, None, None]
        ctx = context.flatten(1, 2) / self.sigma_data
        # the old one-number input keeps its meaning: the worst smudge of the RECENT frames
        code_max = self.cond_noise_code(sc[:, -self.K:].max(dim=1).values)
        code_frames = self.cond_noise_code(sc)
        F_out = self.unet(b(c_in) * x_noisy, c_noise, code_max, code_frames, ctx, actions, colour, far_act)
        return c_skip, c_out, F_out

    def forward(self, x_noisy, sigma, context, sigma_cond, actions, colour=None, far_act=None):
        """D(x; sigma): the denoised guess of the clean next frame (context already noised)."""
        B = x_noisy.shape[0]
        sc = self.per_frame(sigma_cond, B, x_noisy.device)
        colour, far_act = self._extras(colour, far_act, B, x_noisy.device)
        c_skip, c_out, F_out = self._unet(x_noisy, sigma, context, sc, actions, colour, far_act)
        b = lambda t: t[:, None, None, None]
        return b(c_skip) * x_noisy + b(c_out) * F_out.float(), F_out

    def loss(self, target, context, actions, sigma, sigma_cond, colour=None, far_act=None):
        """Same training loss as v1-v5, with the extra inputs passed through."""
        B = target.shape[0]
        sc = self.per_frame(sigma_cond, B, target.device)
        colour, far_act = self._extras(colour, far_act, B, target.device)
        b = lambda t: t[:, None, None, None]
        context = context + sc[:, :, None, None, None] * torch.randn_like(context)   # anti-melt smudge
        noise = torch.randn_like(target)
        x_noisy = target + b(sigma) * noise
        c_skip, c_out, F_out = self._unet(x_noisy, sigma, context, sc, actions, colour, far_act)
        F_target = (target - b(c_skip) * x_noisy) / b(c_out)
        return F.mse_loss(F_out.float(), F_target)

    @torch.no_grad()
    def sample(self, context, actions, steps=5, sigma_cond=0.0, heun=False, generator=None,
               colour=None, far_act=None, start="noise", sigma_max=5.0):
        """Dream the next frame: word for word DenoiserV3.sample (Euler/Heun, start="prev" = begin
        from the newest memory frame + static, the 2026-10-20 fix)."""
        B, _, C, H, W = context.shape
        dev = context.device
        sc = self.per_frame(sigma_cond, B, dev)
        colour, far_act = self._extras(colour, far_act, B, dev)
        if bool((sc > 0).any()):
            context = context + sc[:, :, None, None, None] * torch.randn(
                context.shape, device=dev, generator=generator)
        sigmas = self.karras_sigmas(steps, sigma_max=sigma_max, device=dev)
        x = torch.randn((B, C, H, W), device=dev, generator=generator) * sigmas[0]
        if start == "prev":
            x = x + context[:, -1].float()          # newest memory frame (oldest-first order)
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            den, _ = self(x, s.expand(B), context, sc, actions, colour, far_act)
            d = (x - den) / s
            x_next = x + d * (s_next - s)
            if heun and s_next > 0:
                den2, _ = self(x_next, s_next.expand(B), context, sc, actions, colour, far_act)
                d2 = (x_next - den2) / s_next
                x_next = x + (d + d2) / 2 * (s_next - s)
            x = x_next
        return x.clamp(-1, 1)


# Saved inside every v6 checkpoint, so --resume and play.py rebuild it exactly.
DEFAULT_CFG_V6 = {k: v for k, v in DEFAULT_CFG_V3.items() if k not in ("far_n", "far_stride")}
DEFAULT_CFG_V6.update(version="v6", mem_offsets=v6_offsets(), key_bands=v6_key_bands(),
                      extra_blocks=1, fp_offset=48)


def build_model_v6(cfg=None):
    cfg = dict(DEFAULT_CFG_V6, **(cfg or {}))
    check_frame_size(*cfg["frame_hw"], halvings=len(cfg["channels"]) - 1)
    return DenoiserV6(K=cfg["K"], mem_offsets=cfg["mem_offsets"], key_bands=cfg["key_bands"],
                      extra_blocks=cfg["extra_blocks"], fp_offset=cfg.get("fp_offset", 48),
                      channels=tuple(cfg["channels"]), blocks=cfg["blocks"],
                      attn_levels=tuple(cfg["attn_levels"]))


def v6_cfg_from_v5(v5_cfg, vfar_n=8, vfar_stride=15, extra_blocks=1):
    """A v6 config that grows a v5 (model_v3) config: same widths/blocks/frame size."""
    if v5_cfg.get("version") != "v3" or (v5_cfg["far_n"], v5_cfg["far_stride"], v5_cfg["K"]) != (8, 5, 8):
        raise ValueError(f"expected a v5 (model_v3, 8 recent + 8 far every 5) config, got {v5_cfg}")
    cfg = {k: v for k, v in v5_cfg.items() if k not in ("far_n", "far_stride")}
    cfg.update(version="v6", mem_offsets=v6_offsets(vfar_n, vfar_stride), key_bands=v6_key_bands(vfar_n, vfar_stride),
               extra_blocks=extra_blocks, fp_offset=48)
    return dict(DEFAULT_CFG_V6, **cfg)


# ---------------------------------------------------------------- warm start: v5 weights -> v6

def warm_start_from_v5(v6, v5_state, v5_offsets=V5_OFFSETS, n_bands_v5=1):
    """Copy a v5 state_dict into a v6 model so that, at init, v6 dreams exactly what v5 dreams.
      - same name + same shape: copied as-is (every U-Net block, the key/noise/colour networks...)
      - conv_in.weight and frame_noise_mlp.0.weight: copied SLOT BY SLOT, matched by offset:
        v6's "13 frames back" slot gets v5's "13 frames back" weights; slots v5 never had get 0.
      - anything else new (the extra blocks, the very-far keys network) keeps its fresh init,
        whose last layer is zero (= no effect yet).
      - a v5 tensor v6 has no place for, or any other shape change: stop with an error.
    Returns (plain-words notes for run.log, {name: how it was mapped} for warm_start_opt_state)."""
    new = v6.state_dict()
    slot_of = {o: i for i, o in enumerate(v6.mem_offsets())}
    v5_slots = [slot_of[o] for o in v5_offsets]          # where each v5 slot lands in v6
    notes, fresh, copied, how = [], [], 0, {}
    for name, t in new.items():
        if name not in v5_state:
            fresh.append(name)
            continue
        old = v5_state[name].to(t.dtype)
        if old.shape == t.shape:
            new[name] = old.clone()
            copied += 1
            how[name] = ("same",)
        elif name == "unet.conv_in.weight":
            # [out, 3 + 3*n_mem, 3, 3]: noisy frame | memory slot 0 | slot 1 | ...
            w = torch.zeros_like(t)
            w[:, :3] = old[:, :3]
            idx = [0, 1, 2]
            for j5, j6 in enumerate(v5_slots):
                w[:, 3 + 3 * j6:6 + 3 * j6] = old[:, 3 + 3 * j5:6 + 3 * j5]
                idx += [3 + 3 * j6, 4 + 3 * j6, 5 + 3 * j6]
            new[name] = w
            how[name] = ("dim1", idx)                        # new channel c <- old channel position in idx
            notes.append(f"conv_in: noisy frame + v5's {len(v5_slots)} memory slots copied by offset; "
                         f"{v6.n_mem - len(v5_slots)} new slots ({3 * (v6.n_mem - len(v5_slots))} channels) zeroed")
        elif name == "unet.frame_noise_mlp.0.weight":
            # [cond_dim, n_mem]: one column per memory slot
            w = torch.zeros_like(t)
            for j5, j6 in enumerate(v5_slots):
                w[:, j6] = old[:, j5]
            new[name] = w
            how[name] = ("dim1", v5_slots)
            notes.append(f"frame_noise_mlp.0: v5's {len(v5_slots)} columns copied by offset; "
                         f"{v6.n_mem - len(v5_slots)} new columns zeroed")
        else:
            raise RuntimeError(f"v5 -> v6 warm start: unexpected shape change in {name}: "
                               f"v5 {tuple(old.shape)} vs v6 {tuple(t.shape)}")
    extra = [k for k in v5_state if k not in new]
    if extra:
        raise RuntimeError(f"v5 -> v6 warm start: v5 has tensors v6 doesn't: {extra}")
    v6.load_state_dict(new)
    notes.insert(0, f"copied {copied} tensors unchanged from v5")
    groups = sorted({".".join(n.split(".")[1:3]) if n.split(".")[1] in ("down_extra", "up_extra", "extra_act_mlps")
                     else n.split(".")[1] for n in fresh})
    notes.append(f"new in v6 ({len(fresh)} tensors, fresh init with a zero last layer = no effect yet): {groups}")
    return notes, how


def warm_start_opt_state(opt6, model6, opt5_state, v5_param_names, how):
    """Carry v5's Adam memory (its running averages of each weight's gradient and squared gradient)
    over to the matching v6 weights, so the warm start doesn't begin with a fresh optimizer's
    jumpy first steps on weights that are already well trained. New weights simply start with an
    empty memory (Adam fills it on their first step); the slot-mapped layers get v5's numbers on
    the old slots and zeros on the new ones.
    v5_param_names = v5's parameter names in the order its optimizer listed them.
    Returns the number of weights whose memory was carried over."""
    st5 = opt5_state["state"]
    by_name = {}
    for i, n in enumerate(v5_param_names):
        if i in st5:
            by_name[n] = st5[i]
    sd6 = opt6.state_dict()
    names6 = dict((id(p), n) for n, p in model6.named_parameters())
    order6 = [names6[id(p)] for g in opt6.param_groups for p in g["params"]]
    params6 = dict(model6.named_parameters())
    new_state = {}
    for i, n in enumerate(order6):
        if n not in by_name or n not in how:
            continue
        s5 = by_name[n]
        if how[n][0] == "same":
            new_state[i] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s5.items()}
        else:                                   # slot-mapped: old values on old slots, zeros on new ones
            p = params6[n]
            idx = torch.tensor(how[n][1])
            s = {}
            for k, v in s5.items():
                if torch.is_tensor(v) and v.ndim == p.ndim:
                    z = torch.zeros(p.shape, dtype=v.dtype)
                    z.index_copy_(1, idx, v)
                    s[k] = z
                else:
                    s[k] = v.clone() if torch.is_tensor(v) else v
            new_state[i] = s
    sd6["state"] = new_state
    opt6.load_state_dict(sd6)
    return len(new_state)


if __name__ == "__main__":
    for eb in (0, 1, 2):
        m = build_model_v6(dict(extra_blocks=eb))
        print(f"v6 (extra_blocks={eb}) parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M "
              f"(memory {m.n_mem} frames, history {m.history} frames = {m.history / 10:.1f} s)")
    print("memory offsets (frames before the dreamed one):", m.mem_offsets())
    print("key bands [from, to, piece]:", m.key_bands)
