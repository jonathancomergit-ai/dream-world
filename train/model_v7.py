"""Dream World model, version 7: v6's network + REAL STEERING inputs.

WHY A v7 (PLAN.md, 2026-10-20 "Steering is weak")
  v5/v6 hardly steer: holding "turn right" for 4 s turns the dream ~1 deg (real footage: ~90 deg).
  Three likely reasons, and what v7 does about each:
    1. The keys are coarse labels ("turn" = anything over ~1 deg per step). -> v7 ALSO gets the
       measured camera motion as numbers (yaw, pitch, forward, strafe; data_v7.py).
    2. The memory frames already show the motion, so the key adds nothing. -> train_v7.py practises
       turn ONSETS (the memory shows no turning yet, only the key/motion says "turn now").
    3. Nothing in training forces it to obey. -> steering dropout + guidance (below).

NEW INPUT: THE MOTION BLOCK (MOTION_IN = 8 x 4 + 1 = 33 numbers)
  For the frame being dreamed, the motion of the 8 steps that the 8 recent key slots describe
  (oldest first; the LAST row is the step being dreamed now, the one that matters most), 4 numbers
  each (normalised yaw, pitch, fwd, strafe), plus one flag: 1 = "steering absent" (see dropout).
  It goes through a NEW small network, motion_mlp (33 -> 256 -> 256), whose LAST LAYER STARTS AT
  ZERO, and is ADDED to the conditioning vector (the vector that tilts every layer of the U-Net,
  like the noise level and the keys do). Zero last layer = adds exactly nothing at step 0, so v7
  dreams EXACTLY what its parent (v6, or v5) dreams until training teaches it to use the motion.

HOW IT TRAVELS: inside the "far_act" input
  Every piece of code that runs the network (GraphedSampler's CUDA-graph recording, play.py, the
  recovery practice) already passes one flat float vector called far_act (v6: the key summaries).
  v7 simply makes that vector longer: [key summaries (v6's 144 numbers) | motion block (33)].
  model.extras(keys, motion) builds it. So all those callers work on v7 unchanged.

STEERING DROPOUT + THE "NULL" KEY
  In training, 10% of examples get their steering removed: all 8 recent key slots get the new key
  id NULL_ACTION (9 = "no key given") and the motion block becomes zeros with the flag = 1. The
  model then has to dream "what probably happens next" from the memory alone. That gives us two
  dreams to compare at play time:
      den(steer)  the clean-up guess WITH your keys/motion
      den(null)   the guess with NO steering
  and GUIDANCE pushes the dream further in the direction your steering pulls it:
      den = den(steer) + w * (den(steer) - den(null))        w = 0: plain dream (one network call)
  (the same trick as "classifier-free guidance" in image generators; w > 0 costs a second call,
  done as ONE batch of two).
  The key embedding table (act_emb) has one row per (slot, key). The null key gets its own new row
  per slot, null_act_emb [8, 64], copied from the 'W forward' row of that slot at the start, so on
  day one "no key" means "W" (what the old guidance test compared against) and real keys are
  untouched. The old table keeps its exact shape and name, so the parent's weights load as-is.

WARM START: warm_start_v7() below, from a v6 save (primary) or a v5 save (fallback, through v6's
  v5 -> v6 mapping). train_v7.py checks "v7 == parent" numerically before training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import N_ACTIONS
from model_v2 import check_frame_size
from model_v6 import DEFAULT_CFG_V6, DenoiserV6, UNetV6, build_model_v6, v6_cfg_from_v5, warm_start_from_v5

NULL_ACTION = N_ACTIONS          # 9: "no key given" (steering dropout / the guidance's null dream)
MOTION_CH = 4                    # yaw, pitch, fwd, strafe (data_v7.py)
FORWARD = 1


def motion_in(K):
    return K * MOTION_CH + 1     # K steps x 4 numbers + the "steering absent" flag


# ---------------------------------------------------------------- the U-Net, v7

class UNetV7(UNetV6):
    """v6's U-Net + null key rows + the motion network. Everything else is v6, word for word."""

    def __init__(self, K=8, cond_dim=256, **kw):
        super().__init__(K=K, cond_dim=cond_dim, **kw)
        act_dim = self.act_emb.embedding_dim
        # one "no key" row per key slot; warm_start_v7 copies the 'W' rows in (fresh model: also 'W')
        with torch.no_grad():
            w = self.act_emb.weight.view(K, N_ACTIONS, act_dim)[:, FORWARD].clone()
        self.null_act_emb = nn.Parameter(w)
        self.motion_in = motion_in(K)
        self.motion_mlp = nn.Sequential(nn.Linear(self.motion_in, cond_dim), nn.SiLU(),
                                        nn.Linear(cond_dim, cond_dim))
        nn.init.zeros_(self.motion_mlp[2].weight)        # adds nothing at step 0
        nn.init.zeros_(self.motion_mlp[2].bias)

    def key_embedding(self, actions):
        """actions [B, K] in 0..9 -> [B, K * act_dim]. 0..8 = v6's table, 9 = the null row of that slot."""
        B = actions.shape[0]
        is_null = actions >= N_ACTIONS
        slots = torch.arange(self.K, device=actions.device)[None] * N_ACTIONS + actions.clamp(max=N_ACTIONS - 1)
        e = self.act_emb(slots)                                              # [B, K, act_dim]
        e = torch.where(is_null[..., None], self.null_act_emb[None].to(e.dtype), e)
        return e.reshape(B, -1)

    def forward(self, x_noisy, c_noise, c_noise_cond, c_noise_frames, context, actions, colour, far_act):
        """As UNetV6.forward; far_act = [key summaries | motion block]; actions may hold NULL_ACTION."""
        cond = torch.cat([self.noise_emb(c_noise), self.cond_noise_emb(c_noise_cond),
                          self.key_embedding(actions)], dim=1)
        cond = self.cond_mlp(cond)
        cond = cond + self.frame_noise_mlp(c_noise_frames.to(cond.dtype))
        cond = cond + self.colour_mlp(colour.to(cond.dtype))
        far_act = far_act.to(cond.dtype)
        n0 = self.band_sizes[0]
        cond = cond + self.far_act_mlp(far_act[:, :n0])          # (v6's order of additions ...)
        a = n0
        for n, mlp in zip(self.band_sizes[1:], self.extra_act_mlps):
            cond = cond + mlp(far_act[:, a:a + n])
            a += n
        cond = cond + self.motion_mlp(far_act[:, a:a + self.motion_in])   # (... then the NEW one, last)

        h = self.conv_in(torch.cat([x_noisy, context], dim=1))
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


# ---------------------------------------------------------------- EDM wrapper, v7

class DenoiserV7(DenoiserV6):
    """v6's Denoiser (same maths, loss, sampler, call signature) with the v7 U-Net.
    far_act is now [B, n_keysum + motion_in]: build it with extras()."""

    def __init__(self, K=8, mem_offsets=None, key_bands=None, extra_blocks=1, fp_offset=48,
                 sigma_data=0.5, **unet_kw):
        super().__init__(K=K, mem_offsets=mem_offsets, key_bands=key_bands, extra_blocks=extra_blocks,
                         fp_offset=fp_offset, sigma_data=sigma_data, **unet_kw)
        band_sizes = self.unet.band_sizes
        self.unet = UNetV7(K=K, n_mem=self.n_mem, band_sizes=band_sizes, extra_blocks=extra_blocks, **unet_kw)
        self.n_keysum = sum(band_sizes)
        self.motion_in = self.unet.motion_in

    def _extras(self, colour, far_act, B, device):
        if colour is None:
            colour = torch.zeros(B, 25, device=device)
        if far_act is None:
            far_act = torch.zeros(B, self.n_keysum + self.motion_in, device=device)
        return colour.float(), far_act.float()

    def motion_block(self, motion, absent=None):
        """motion [B, K, 4] normalised (oldest first, last = the step being dreamed) -> [B, motion_in].
        absent: None (= steering present) or a bool [B]: those rows get zeros + flag 1."""
        B = motion.shape[0]
        flat = motion.reshape(B, -1).float()
        flag = torch.zeros(B, 1, device=motion.device)
        if absent is not None:
            flat = flat * (~absent)[:, None]
            flag = absent[:, None].float()
        return torch.cat([flat, flag], dim=1)

    def null_block(self, B, device):
        """The motion block of 'no steering': zeros + flag 1."""
        z = torch.zeros(B, self.motion_in, device=device)
        z[:, -1] = 1.0
        return z

    def extras(self, keys, motion, absent=None):
        """The full far_act vector: keys [B, history] (as v6's key_summary wants), motion [B, K, 4]."""
        return torch.cat([self.key_summary(keys), self.motion_block(motion, absent)], dim=1)

    def loss_ghost(self, target, context, actions, sigma, sigma_cond, colour, far_act, ghost, use_ghost):
        """v6's training loss (DenoiserV6.loss, word for word) with ONE change for the rows in use_ghost [B]:
        the noisy input is built around `ghost` (the PREVIOUS frame, or the previous frame slid by the real
        turn) instead of around the target:   x_noisy = ghost + sigma * noise,   but the answer is still
        the target. Why (found 2026-10-20 while building v7): normal training hides a faint copy of the
        ANSWER in the noisy input even at the heaviest noise, so the network learned to copy that ghost
        and never needed the keys. When dreaming, each frame starts from the PREVIOUS frame + static, so
        the ghost it copies is un-turned -> it doesn't turn. Ghost rows make the training input look like
        the dream's first step, where only the keys/motion can say "the view has turned"."""
        B = target.shape[0]
        sc = self.per_frame(sigma_cond, B, target.device)
        colour, far_act = self._extras(colour, far_act, B, target.device)
        b = lambda t: t[:, None, None, None]
        context = context + sc[:, :, None, None, None] * torch.randn_like(context)
        noise = torch.randn_like(target)
        centre = torch.where(use_ghost[:, None, None, None], ghost.to(target.dtype), target)
        x_noisy = centre + b(sigma) * noise
        c_skip, c_out, F_out = self._unet(x_noisy, sigma, context, sc, actions, colour, far_act)
        F_target = (target - b(c_skip) * x_noisy) / b(c_out)
        return F.mse_loss(F_out.float(), F_target)

    def null_of(self, actions, far_act):
        """The 'no steering' twin of (recent keys, far_act): all K key slots -> NULL, motion -> zeros + flag.
        (The key summaries of the older stretches stay: they describe how the MEMORY was filmed.)"""
        a0 = torch.full_like(actions, NULL_ACTION)
        f0 = far_act.clone()
        f0[:, self.n_keysum:] = self.null_block(far_act.shape[0], far_act.device)
        return a0, f0


# Saved inside every v7 checkpoint, so --resume and play.py rebuild it exactly.
DEFAULT_CFG_V7 = dict(DEFAULT_CFG_V6, version="v7", motion_ch=MOTION_CH, null_action=NULL_ACTION)


def build_model_v7(cfg=None):
    if (cfg or {}).get("deep_ctx"):          # v11.1 saves (model_v11.py): v7 + a deep context stage for 2x frames
        from model_v11 import build_model_v11
        return build_model_v11(cfg)
    if (cfg or {}).get("fuel"):              # learned-fuel saves (model_fuel.py): v7 + a real reference frame input
        from model_fuel import build_model_fuel
        return build_model_fuel(cfg)
    cfg = dict(DEFAULT_CFG_V7, **(cfg or {}))
    check_frame_size(*cfg["frame_hw"], halvings=len(cfg["channels"]) - 1)
    return DenoiserV7(K=cfg["K"], mem_offsets=cfg["mem_offsets"], key_bands=cfg["key_bands"],
                      extra_blocks=cfg["extra_blocks"], fp_offset=cfg.get("fp_offset", 48),
                      channels=tuple(cfg["channels"]), blocks=cfg["blocks"],
                      attn_levels=tuple(cfg["attn_levels"]))


def v7_cfg_from_parent(parent_cfg, extra_blocks=1):
    """A v7 config from a v6 config (kept as is) or a v5 config (grown to v6 first, as train_v6 did)."""
    if parent_cfg.get("version") == "v6":
        cfg = dict(parent_cfg)
    elif parent_cfg.get("version") == "v3":
        cfg = v6_cfg_from_v5(parent_cfg, extra_blocks=extra_blocks)
    else:
        raise ValueError(f"v7 warm-starts from a v6 or v5 save, got version {parent_cfg.get('version')}")
    cfg.update(version="v7", motion_ch=MOTION_CH, null_action=NULL_ACTION)
    return dict(DEFAULT_CFG_V7, **cfg)


def parent_model(parent_cfg, state, extra_blocks=1):
    """The parent as a v6 network holding `state` (a v6 state_dict, or a v5 one grown through
    warm_start_from_v5). Also returns the v5 -> v6 mapping notes/how (None for a v6 parent)."""
    if parent_cfg.get("version") == "v6":
        m = build_model_v6(parent_cfg)
        m.load_state_dict(state)
        return m, None, None
    m = build_model_v6(v6_cfg_from_v5(parent_cfg, extra_blocks=extra_blocks))
    notes, how = warm_start_from_v5(m, state)
    return m, notes, how


def warm_start_v7(v7, v6_state):
    """Copy a v6 state_dict (or a v5 one already grown into v6 shape) into a v7 model.
    Every v6 tensor has the same name and shape in v7, so it is a straight copy. New in v7:
      null_act_emb   <- the 'W forward' rows of act_emb (one per slot): "no key" starts as "W"
      motion_mlp.*   fresh, last layer zero (adds nothing)
    Returns plain-words notes."""
    sd = v7.state_dict()
    fresh = [k for k in sd if k not in v6_state]
    extra = [k for k in v6_state if k not in sd]
    if extra:
        raise RuntimeError(f"parent -> v7 warm start: the parent has tensors v7 doesn't: {extra}")
    for k, t in v6_state.items():
        if sd[k].shape != t.shape:
            raise RuntimeError(f"parent -> v7: shape change in {k}: {tuple(t.shape)} vs {tuple(sd[k].shape)}")
        sd[k] = t.clone().to(sd[k].dtype)
    K, act_dim = v7.K, sd["unet.act_emb.weight"].shape[1]
    sd["unet.null_act_emb"] = sd["unet.act_emb.weight"].view(K, N_ACTIONS, act_dim)[:, FORWARD].clone()
    sd["unet.motion_mlp.2.weight"].zero_()
    sd["unet.motion_mlp.2.bias"].zero_()
    v7.load_state_dict(sd)
    return [f"copied all {len(v6_state)} parent tensors unchanged",
            f"new in v7 ({len(fresh)} tensors): {sorted(fresh)}; null_act_emb = the 'W' rows, motion_mlp's last layer = 0"]


if __name__ == "__main__":
    m = build_model_v7()
    print(f"v7 parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.3f} M; far_act length "
          f"{m.n_keysum} key-summary + {m.motion_in} motion")
