"""Dream World model, version 1: same network as v0 (model.py), with two upgrades.

WHY A v1 AT ALL
  v0 dreams a good FIRST frame, but in a long walk its mistakes pile up and by ~6-8 s the scene
  washes out into beige/pink mush. Two reasons:
    1. It only remembers 0.4 s (K=4 frames). Short memory = it forgets what the street looked like.
    2. In training its memory frames were always real video (with a little noise). In play they
       are its own slightly-wrong dreams, a kind of "wrong" it never practised fixing.
  train_v1.py fixes reason 2 (it sometimes feeds the model its own dreams as memory, "self-forcing")
  and this file fixes reason 1 and gives the model a finer way to hear about noisy memory.

WHAT CHANGED VERSUS model.py (everything else is imported from there, unchanged)
  a) K = 8 memory frames (0.8 s) instead of 4. Only the parts whose SIZE depends on K change:
       - conv_in: the first layer now eats 3 + 8*3 = 27 colour channels instead of 15;
       - act_emb + cond_mlp: 8 key slots instead of 4.
  b) PER-FRAME context noise. v0 smudged all 4 memory frames by the same amount and told the
     network one number (sigma_cond). v1 can smudge each memory frame by its OWN amount and tells
     the network all 8 numbers through a tiny new layer ("frame_noise_mlp"). The old one-number
     input is kept and fed the LARGEST of the 8 amounts, so the v0 weights still mean what they
     meant ("the memory is at most this smudged").

WARM START FROM v0 (warm_start_from_v0 below)
  We don't start v1 from scratch: we copy v0's trained weights in. Anything with the same shape is
  copied as-is. For the K-dependent layers we put v0's weights on the 4 MOST RECENT memory slots
  (5..8) and zeros on the 4 older slots (1..4). Zero weights mean "ignore this input", so on day
  one v1 behaves exactly like v0 (it ignores the extra old frames) and training then slowly
  teaches it to use them. The new frame_noise_mlp ends in a zero layer for the same reason.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DEFAULT_CFG, N_ACTIONS, Denoiser, UNet   # the v0 building blocks, untouched


# ---------------------------------------------------------------- the U-Net, v1

class UNetV1(UNet):
    """v0's U-Net plus one small extra input: the noise level of EACH memory frame.

    How it enters: v0 already squeezes (noise level, memory noise level, keys) into one
    "conditioning vector" that tilts every layer (see AdaGroupNorm in model.py). We compute a
    second little vector from the K per-frame noise numbers and simply ADD it to that one. The
    last layer of the little network starts at zero, so at first it adds nothing at all."""

    def __init__(self, K=8, cond_dim=256, frame_noise=True, **kw):
        super().__init__(K=K, cond_dim=cond_dim, **kw)
        self.frame_noise = frame_noise
        if frame_noise:
            self.frame_noise_mlp = nn.Sequential(
                nn.Linear(K, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
            nn.init.zeros_(self.frame_noise_mlp[2].weight)   # starts as "add nothing"
            nn.init.zeros_(self.frame_noise_mlp[2].bias)

    def forward(self, x_noisy, c_noise, c_noise_cond, c_noise_frames, context, actions):
        """Same as UNet.forward in model.py, plus c_noise_frames [B, K]: per-frame noise codes."""
        B = x_noisy.shape[0]
        slots = torch.arange(self.K, device=actions.device)[None] * N_ACTIONS + actions
        cond = torch.cat([self.noise_emb(c_noise), self.cond_noise_emb(c_noise_cond),
                          self.act_emb(slots).reshape(B, -1)], dim=1)
        cond = self.cond_mlp(cond)
        if self.frame_noise:
            cond = cond + self.frame_noise_mlp(c_noise_frames.to(cond.dtype))

        # (from here on, word for word the v0 U-Net)
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


# ---------------------------------------------------------------- EDM wrapper, v1

class DenoiserV1(Denoiser):
    """v0's Denoiser (EDM maths, loss, sampler) with sigma_cond allowed to be per memory frame.

    sigma_cond may be:
      a plain number      -> every memory frame gets that much noise (like v0 / play.py's dial)
      a tensor [B]        -> one amount per example, same for all its frames (like v0 training)
      a tensor [B, K]     -> one amount per memory frame (new in v1)
    Inside, everything is turned into [B, K]."""

    def __init__(self, K=8, sigma_data=0.5, frame_noise=True, **unet_kw):
        nn.Module.__init__(self)          # skip Denoiser.__init__: it would build a v0 UNet
        self.K = K
        self.sigma_data = sigma_data
        self.unet = UNetV1(K=K, frame_noise=frame_noise, **unet_kw)

    def per_frame(self, sigma_cond, B, device):
        """Any of the three sigma_cond shapes -> a float tensor [B, K]."""
        if not torch.is_tensor(sigma_cond):
            return torch.full((B, self.K), float(sigma_cond), device=device)
        sigma_cond = sigma_cond.to(device=device, dtype=torch.float32)
        if sigma_cond.ndim == 1:
            return sigma_cond[:, None].expand(B, self.K)
        return sigma_cond

    def _unet(self, x_noisy, sigma, context, sc, actions):
        """Shared by forward() and loss(): run the U-Net with EDM scaling.
        sc [B, K] = the per-frame context noise that was ALREADY added to `context`."""
        c_in, c_skip, c_out, c_noise = self.precond(sigma)
        b = lambda t: t[:, None, None, None]
        ctx = context.flatten(1, 2) / self.sigma_data
        # the old single-number input gets the WORST (largest) frame noise: "at most this smudged"
        code_max = self.cond_noise_code(sc.max(dim=1).values)
        code_frames = self.cond_noise_code(sc)                       # [B, K], same log/4 scale
        F_out = self.unet(b(c_in) * x_noisy, c_noise, code_max, code_frames, ctx, actions)
        return c_skip, c_out, F_out

    def forward(self, x_noisy, sigma, context, sigma_cond, actions):
        """D(x; sigma): the denoised guess of the clean next frame (context already noised)."""
        B = x_noisy.shape[0]
        sc = self.per_frame(sigma_cond, B, x_noisy.device)
        c_skip, c_out, F_out = self._unet(x_noisy, sigma, context, sc, actions)
        b = lambda t: t[:, None, None, None]
        return b(c_skip) * x_noisy + b(c_out) * F_out.float(), F_out

    def loss(self, target, context, actions, sigma, sigma_cond):
        """Training loss, same as v0, except the context smudge can differ per frame.
        context [B,K,3,H,W] clean (or self-forced) memory; sigma_cond: see class notes."""
        B = target.shape[0]
        sc = self.per_frame(sigma_cond, B, target.device)
        b = lambda t: t[:, None, None, None]
        # anti-melt smudge: frame k of example i gets noise of strength sc[i, k]
        context = context + sc[:, :, None, None, None] * torch.randn_like(context)
        noise = torch.randn_like(target)
        x_noisy = target + b(sigma) * noise
        c_skip, c_out, F_out = self._unet(x_noisy, sigma, context, sc, actions)
        F_target = (target - b(c_skip) * x_noisy) / b(c_out)
        return F.mse_loss(F_out.float(), F_target)

    @torch.no_grad()
    def sample(self, context, actions, steps=5, sigma_cond=0.0, heun=False, generator=None,
               start="noise", sigma_max=5.0):
        """Dream the next frame. Same as v0's sampler (model.py), but sigma_cond can be a number,
        [B] or [B, K]. context [B,K,3,H,W] in [-1,1], actions [B,K] (last = key now)."""
        B, _, C, H, W = context.shape
        dev = context.device
        sc = self.per_frame(sigma_cond, B, dev)
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
            x = x + context[:, -1].float()          # (context was smudged above if sigma_cond > 0: fine)
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            den, _ = self(x, s.expand(B), context, sc, actions)
            d = (x - den) / s                       # which way is "less noisy"
            x_next = x + d * (s_next - s)           # Euler step down the noise ladder
            if heun and s_next > 0:
                den2, _ = self(x_next, s_next.expand(B), context, sc, actions)
                d2 = (x_next - den2) / s_next
                x_next = x + (d + d2) / 2 * (s_next - s)
            x = x_next
        return x.clamp(-1, 1)


# Saved inside every v1 checkpoint (like v0's DEFAULT_CFG), so --resume and play rebuild it exactly.
DEFAULT_CFG_V1 = dict(DEFAULT_CFG, K=8, frame_noise=True, version="v1")


def build_model_v1(cfg=None):
    cfg = dict(DEFAULT_CFG_V1, **(cfg or {}))
    return DenoiserV1(K=cfg["K"], frame_noise=cfg.get("frame_noise", True),
                      channels=tuple(cfg["channels"]), blocks=cfg["blocks"],
                      attn_levels=tuple(cfg["attn_levels"]))


# ---------------------------------------------------------------- warm start: v0 weights -> v1

def warm_start_from_v0(v1, v0_state, K0):
    """Copy a v0 state_dict (K0 memory frames) into a v1 model (K1 >= K0 memory frames).

    The rule: memory slot k of v1 (0 = oldest, K1-1 = newest) takes v0's slot k - (K1 - K0).
    So v0's 4 slots land on v1's 4 NEWEST slots, and v1's older slots get zero weights.
    Returns a list of plain-words notes about what was done (printed into run.log)."""
    K1 = v1.K
    shift = K1 - K0
    assert shift >= 0, "v1 must remember at least as many frames as v0"
    new = v1.state_dict()
    notes = []
    copied = 0
    for name, t in new.items():
        if name not in v0_state:
            notes.append(f"new in v1, kept fresh init: {name} {tuple(t.shape)}")
            continue
        old = v0_state[name].to(t.dtype)
        if old.shape == t.shape:
            new[name] = old.clone()
            copied += 1
            continue

        if name == "unet.conv_in.weight":
            # [out, 3 + 3*K, 3, 3]. Channels 0..2 = the noisy next frame. Then memory frame k sits
            # at channels 3+3k .. 3+3k+2 (oldest first).
            w = torch.zeros_like(t)
            w[:, :3] = old[:, :3]                                      # noisy-frame channels
            w[:, 3 + 3 * shift:] = old[:, 3:]                          # v0's 4 frames -> newest 4
            new[name] = w                                              # older 4 stay at zero
            notes.append(f"conv_in: noisy channels copied, v0 memory -> newest {K0} of {K1} slots, "
                         f"oldest {shift} zeroed")
        elif name == "unet.act_emb.weight":
            # [K * 9, act_dim]: row k*9 + a = "key a pressed at slot k". Newest slots get v0's
            # rows. The older slots copy v0's oldest slot too: harmless, because the cond_mlp
            # columns that READ those slots are zero (next case), so they have no effect yet.
            w = torch.empty_like(t)
            for k in range(K1):
                k0 = max(0, k - shift)
                w[k * N_ACTIONS:(k + 1) * N_ACTIONS] = old[k0 * N_ACTIONS:(k0 + 1) * N_ACTIONS]
            new[name] = w
            notes.append(f"act_emb: slots {shift}..{K1 - 1} <- v0 slots 0..{K0 - 1}")
        elif name == "unet.cond_mlp.0.weight":
            # [cond_dim, 2*cond_dim + K*act_dim]: first come the two noise vectors (copied), then
            # one act_dim-wide block of columns per key slot.
            act_dim = (t.shape[1] - old.shape[1]) // shift               # columns per key slot (64)
            fixed = old.shape[1] - K0 * act_dim                          # = 2 * cond_dim
            w = torch.zeros_like(t)
            w[:, :fixed] = old[:, :fixed]
            w[:, fixed + shift * act_dim:] = old[:, fixed:]              # v0 slots -> newest slots
            new[name] = w
            notes.append(f"cond_mlp.0: noise columns copied, key slots -> newest {K0}, "
                         f"oldest {shift} key slots zeroed")
        else:
            notes.append(f"!! shape mismatch NOT handled, kept fresh init: {name} "
                         f"v0 {tuple(old.shape)} vs v1 {tuple(t.shape)}")
    v1.load_state_dict(new)
    notes.insert(0, f"copied {copied} tensors unchanged from v0")
    return notes


if __name__ == "__main__":
    m = build_model_v1()
    n = sum(p.numel() for p in m.parameters())
    print(f"v1 parameters: {n / 1e6:.2f} M (K={m.K})")
