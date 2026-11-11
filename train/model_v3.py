"""Dream World model, version 3: v2's network with LONG MEMORY and COLOUR MEMORY.

WHY A v3
  v2 dreams sharp first frames, but in long walks three things go wrong:
    1. It only remembers 0.8 s (8 frames). Turn around and the street behind you is gone, so it
       invents a new one.
    2. Colours slowly fade to beige. The "colour anchor" in play.py fixes that AFTER each frame
       is dreamed, but the model itself never learned to hold its colours.
    3. (Blur and vertical "curtain" streaks: train_v3.py works on those, not this file.)
  This file fixes 1 and 2 by giving the network two new inputs.

NEW INPUT 1: FAR MEMORY (long-term memory)
  Besides the 8 most recent frames (the "recent" memory, exactly v2's), the network also sees
  8 OLDER frames, one every 5 frames (0.5 s) going back before the recent ones:

      time ->   far far far far far far far far | recent x 8 | NEXT
      offset:  -48 -43 -38 -33 -28 -23 -18 -13  | -8 ... -1  |  0      (in frames, 10 per second)

  So it can look back ~4.8 s instead of 0.8 s. (The counts and the 5-frame stride are settings:
  far_n and far_stride in the config, --far-n / --far-stride in train_v3.py.)
  How the far frames get in: the very first layer (conv_in) simply gets more colour channels,
  like v1 did when it went from 4 to 8 frames. The memory is stacked OLDEST FIRST:
      channels  0..2   the noisy next frame (what is being cleaned up)
               3..26   the 8 far frames   (NEW, 3 colours each)
              27..50   the 8 recent frames (v2's)
  Each far frame also gets its own "how smudged is it" number (per-frame context noise), like the
  recent frames already do.

  A short summary of the KEYS pressed during the far stretch goes in too: for each 0.5 s gap
  between far frames, what share of the time was W, turn-left, turn-right, ... (9 shares per gap,
  72 numbers). Without it the network couldn't tell "that far frame is from before I turned
  around" from "that far frame is straight behind me". It goes through a tiny network
  (far_act_mlp) into the conditioning vector.

NEW INPUT 2: COLOUR MEMORY (the colour anchor, baked in)
  A "colour fingerprint" of REAL frames from ~4.5 s earlier: per colour channel, the average and
  the contrast (standard deviation) = 6 numbers, plus the average colour of a coarse 2 x 3 grid
  (top-left, top-middle, ..., bottom-right) = 18 numbers, plus a 1 = "fingerprint present" flag.
  25 numbers through a tiny network (colour_mlp) into the conditioning vector, which tilts every
  layer (see AdaGroupNorm in model.py): exactly the kind of "global look" knob colour needs.
  In play the fingerprint comes from the real starting frames and stays fixed for the whole
  walk: "this is roughly what this place looks like, don't fade away from it".
  10% of the time in training it is blanked out (all 25 numbers 0, flag 0), so the model still
  works without it (play.py's C key turns it off to compare).

WARM START FROM v2 (warm_start_from_v2 below): AT INIT v3 == v2
  Every v2 weight is copied. The recent memory channels/noise inputs get v2's weights exactly;
  everything new (far channels in conv_in, far columns of the per-frame-noise layer, and the
  LAST layer of colour_mlp and far_act_mlp) is zero. Zero weights = "ignore this input", so on day
  one v3 dreams exactly what v2 dreams, and training slowly teaches it to use the new inputs.
  train_v3.py checks this with real frames before training starts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import N_ACTIONS
from model_v1 import DenoiserV1, UNetV1
from model_v2 import DEFAULT_CFG_V2, check_frame_size

COLOUR_DIM = 25          # 3 means + 3 contrasts + 2x3 grid x 3 colours (18) + 1 "present" flag
COLOUR_GRID = (2, 3)     # rows x columns of the coarse colour grid
FP_FRAMES = 8            # how many real frames the colour fingerprint is averaged over


# ---------------------------------------------------------------- the two new summaries

def colour_fingerprint(frames):
    """The colour fingerprint of some REAL frames.
    frames [B, N, 3, H, W] in model space (-1..1) -> [B, 25]:
      [0:3]   average of R, G, B over all N frames and pixels
      [3:6]   contrast (standard deviation) of R, G, B
      [6:24]  average R, G, B of each cell of a 2 x 3 grid (top row first)
      [24]    1.0 = "a fingerprint is present" (a blanked one is all zeros, flag included)"""
    f = frames.float()
    B = f.shape[0]
    mean = f.mean(dim=(1, 3, 4))                                          # [B, 3]
    std = f.std(dim=(1, 3, 4))                                            # [B, 3]
    grid = F.adaptive_avg_pool2d(f.mean(dim=1), COLOUR_GRID).flatten(1)   # [B, 3*2*3] (channel-major)
    flag = torch.ones(B, 1, device=f.device)
    return torch.cat([mean, std, grid, flag], dim=1)


def far_action_summary(acts, far_n, far_stride):
    """Keys pressed during the far stretch, squeezed into shares.
    acts [B, far_n * far_stride] int keys, oldest first: the key that moved the oldest far frame
    on by one, ..., up to the key just before the oldest recent frame.
    -> float [B, far_n * 9]: for each far frame, the share of each of the 9 keys over the
    far_stride steps that FOLLOW it (the gap until the next far frame, or until the recent memory)."""
    B = acts.shape[0]
    a = acts.clamp(0, N_ACTIONS - 1).reshape(B, far_n, far_stride)
    return F.one_hot(a, N_ACTIONS).float().mean(dim=2).reshape(B, far_n * N_ACTIONS)


# ---------------------------------------------------------------- the U-Net, v3

class UNetV3(UNetV1):
    """v2's U-Net (= v1's) with: far frames as extra conv_in channels, per-frame noise numbers
    for the far frames, and two small zero-ending networks for the colour fingerprint and the
    far-keys summary. K = recent frames (8), far_n = far frames (8)."""

    def __init__(self, K=8, far_n=8, cond_dim=256, **kw):
        super().__init__(K=K, cond_dim=cond_dim, frame_noise=True, **kw)
        self.far_n = far_n
        ch0 = self.conv_in.out_channels
        # memory = far_n + K frames, oldest first (see the channel map at the top)
        self.conv_in = nn.Conv2d(3 + 3 * (far_n + K), ch0, 3, padding=1)
        self.frame_noise_mlp = nn.Sequential(nn.Linear(far_n + K, cond_dim), nn.SiLU(),
                                             nn.Linear(cond_dim, cond_dim))
        self.colour_mlp = nn.Sequential(nn.Linear(COLOUR_DIM, cond_dim), nn.SiLU(),
                                        nn.Linear(cond_dim, cond_dim))
        self.far_act_mlp = nn.Sequential(nn.Linear(far_n * N_ACTIONS, cond_dim), nn.SiLU(),
                                         nn.Linear(cond_dim, cond_dim))
        for mlp in (self.frame_noise_mlp, self.colour_mlp, self.far_act_mlp):
            nn.init.zeros_(mlp[2].weight)          # each starts as "add nothing"
            nn.init.zeros_(mlp[2].bias)

    def forward(self, x_noisy, c_noise, c_noise_cond, c_noise_frames, context, actions, colour, far_act):
        """x_noisy [B,3,H,W] (scaled by c_in), c_noise [B], c_noise_cond [B] (worst RECENT smudge),
        c_noise_frames [B, far_n+K], context [B, 3*(far_n+K), H, W] oldest first,
        actions [B, K] recent keys (last = key now), colour [B, 25], far_act [B, far_n*9]."""
        B = x_noisy.shape[0]
        slots = torch.arange(self.K, device=actions.device)[None] * N_ACTIONS + actions
        cond = torch.cat([self.noise_emb(c_noise), self.cond_noise_emb(c_noise_cond),
                          self.act_emb(slots).reshape(B, -1)], dim=1)
        cond = self.cond_mlp(cond)
        cond = cond + self.frame_noise_mlp(c_noise_frames.to(cond.dtype))
        cond = cond + self.colour_mlp(colour.to(cond.dtype))
        cond = cond + self.far_act_mlp(far_act.to(cond.dtype))

        # (from here on, word for word the v0/v1/v2 U-Net)
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


# ---------------------------------------------------------------- EDM wrapper, v3

class DenoiserV3(DenoiserV1):
    """v1/v2's Denoiser (EDM maths, loss, sampler) with the two new inputs.

    context    [B, far_n + K, 3, H, W]  memory, OLDEST FIRST (far frames, then recent frames)
    sigma_cond number, [B] or [B, far_n + K]: how much each memory frame was smudged
    actions    [B, K]                   recent keys (last = the key pressed now)
    colour     [B, 25] or None          colour fingerprint (None = blanked = all zeros)
    far_act    [B, far_n * 9] or None   far keys summary (None = all zeros)"""

    def __init__(self, K=8, far_n=8, far_stride=5, sigma_data=0.5, **unet_kw):
        nn.Module.__init__(self)          # skip DenoiserV1.__init__: it would build a v1 UNet
        self.K = K                        # recent frames (what v0-v2 called K)
        self.far_n = far_n
        self.far_stride = far_stride
        self.n_mem = far_n + K            # all memory frames
        self.history = K + far_n * far_stride   # frames of history needed (48 = 4.8 s)
        self.sigma_data = sigma_data
        self.unet = UNetV3(K=K, far_n=far_n, **unet_kw)

    def per_frame(self, sigma_cond, B, device):
        """Any sigma_cond shape -> float [B, far_n + K]."""
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
            far_act = torch.zeros(B, self.far_n * N_ACTIONS, device=device)
        return colour.float(), far_act.float()

    def _unet(self, x_noisy, sigma, context, sc, actions, colour, far_act):
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        b = lambda t: t[:, None, None, None]
        ctx = context.flatten(1, 2) / self.sigma_data
        # the old one-number input keeps v2's meaning: the worst smudge of the RECENT frames
        code_max = self.cond_noise_code(sc[:, self.far_n:].max(dim=1).values)
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
        """Same training loss as v1/v2, with the extra inputs passed through."""
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
        """Dream the next frame (v1's Euler/Heun sampler, with the extra inputs)."""
        B, _, C, H, W = context.shape
        dev = context.device
        sc = self.per_frame(sigma_cond, B, dev)
        colour, far_act = self._extras(colour, far_act, B, dev)
        if bool((sc > 0).any()):
            context = context + sc[:, :, None, None, None] * torch.randn(
                context.shape, device=dev, generator=generator)
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

    # ------------------------------------------------------------ memory bookkeeping
    def mem_offsets(self):
        """Where each memory frame sits, in frames BEFORE the frame being dreamed (oldest first).
        Default: [48, 43, 38, 33, 28, 23, 18, 13, 8, 7, 6, 5, 4, 3, 2, 1]."""
        far = [self.K + self.far_stride * (self.far_n - j) for j in range(self.far_n)]
        return far + list(range(self.K, 0, -1))


# Saved inside every v3 checkpoint, so --resume and play.py rebuild it exactly.
DEFAULT_CFG_V3 = dict(DEFAULT_CFG_V2, version="v3", far_n=8, far_stride=5)


def build_model_v3(cfg=None):
    cfg = dict(DEFAULT_CFG_V3, **(cfg or {}))
    check_frame_size(*cfg["frame_hw"], halvings=len(cfg["channels"]) - 1)
    return DenoiserV3(K=cfg["K"], far_n=cfg["far_n"], far_stride=cfg["far_stride"],
                      channels=tuple(cfg["channels"]), blocks=cfg["blocks"],
                      attn_levels=tuple(cfg["attn_levels"]))


# ---------------------------------------------------------------- warm start: v2 weights -> v3

def warm_start_from_v2(v3, v2_state):
    """Copy a v2 state_dict into a v3 model so that, at init, v3 dreams exactly what v2 dreams.
    Anything with the same shape is copied as-is; the two memory-size-dependent layers get v2's
    weights on the RECENT slots and zeros on the far slots; the new little networks keep their
    zero last layer. Anything else that doesn't match stops with an error (fail loudly).
    Returns plain-words notes for run.log."""
    new = v3.state_dict()
    Fn, K = v3.far_n, v3.K
    notes, fresh, copied = [], [], 0
    for name, t in new.items():
        if name not in v2_state:
            fresh.append(name)
            continue
        old = v2_state[name].to(t.dtype)
        if old.shape == t.shape:
            new[name] = old.clone()
            copied += 1
        elif name == "unet.conv_in.weight":
            # [out, 3 + 3*(far_n+K), 3, 3]: noisy frame | far frames | recent frames
            w = torch.zeros_like(t)
            w[:, :3] = old[:, :3]                    # the noisy next frame
            w[:, 3 + 3 * Fn:] = old[:, 3:]           # v2's 8 memory frames -> the 8 recent slots
            new[name] = w
            notes.append(f"conv_in: noisy-frame + recent-memory channels = v2's; {Fn} far frames "
                         f"({3 * Fn} channels) zeroed")
        elif name == "unet.frame_noise_mlp.0.weight":
            # [cond_dim, far_n + K]: one column per memory frame, oldest first
            w = torch.zeros_like(t)
            w[:, Fn:] = old
            new[name] = w
            notes.append(f"frame_noise_mlp.0: recent-frame columns = v2's; {Fn} far columns zeroed")
        else:
            raise RuntimeError(f"v2 -> v3 warm start: unexpected shape change in {name}: "
                               f"v2 {tuple(old.shape)} vs v3 {tuple(t.shape)}")
    extra = [k for k in v2_state if k not in new]
    if extra:
        raise RuntimeError(f"v2 -> v3 warm start: v2 has tensors v3 doesn't: {extra}")
    v3.load_state_dict(new)
    notes.insert(0, f"copied {copied} tensors unchanged from v2")
    notes.append(f"new in v3 (fresh init, last layers zero = no effect yet): "
                 f"{sorted({n.split('.')[1] for n in fresh})}")
    return notes


if __name__ == "__main__":
    m = build_model_v3()
    print(f"v3 parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M "
          f"(recent K={m.K}, far {m.far_n} every {m.far_stride} frames, history {m.history} frames)")
    print("memory offsets (frames before the dreamed one):", m.mem_offsets())
