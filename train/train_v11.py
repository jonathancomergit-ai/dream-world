"""Train Dream World v11: the v9 recipe at TWICE THE RESOLUTION (192 x 336 instead of 96 x 168), 2026-10-22 night.

    # the overnight run (train_v11_overnight.bat builds data/cache_v11 first, then runs this):
    python train/train_v11.py --run dream_v11 --init-v9 train/checkpoints/dream_v11_init.pt --budget-hours 6 ...
    python train/train_v11.py --run dream_v11 --resume ...           # carry on after a stop
    # a small test that leaves the real folders alone (everything under train/_smoke/v11/):
    python train/train_v11.py --smoke --run s1 --init-v9 <copy of v9_pick.pt> --max-steps 60 --first-sample-step 60 ...
    # score any v11 save on the check-up suite (no training; numbers at 96x168, comparable with v9's):
    python train/train_v11.py --eval-mem <save> --eval-out <file.json>

WHAT v11 IS: train_v9.py (copied, train_v9.py itself untouched) reading data/cache_v11 (train/build_cache_v11.py:
  the same 518,346 frames as cache_v3, same order and indices, at 192x336 from the original videos, faces blurred).
  The network is v7's, unchanged: all convolutions + attention with no position codes, so the v9 pick's weights run
  at 2x as they are (train/_exp/v11_speed.py). A new run warm-starts from a COPY of the v9 pick with the save's
  frame_hw set to [192, 336]; the WARM-START CHECK compares it with the v9 weights run at 2x (must be diff 0).
  Saves keep model_cfg version "v7" (+ trained_as "v11", frame_hw [192, 336]).

WHAT CHANGED FOR THE BIGGER FRAMES (everything else is train_v9.py word for word, see its notes):
  - shifted ghost: motion_shift_px uses fpx = (W/2)/tan(35 deg) with W = 336 (so a 2.4 deg turn slides ~10 px,
    not ~5): already W-based in train_v9 / dream_mem / dream_walks_v7 / mem_ghost - checked, nothing hard-coded.
  - panorama memory (check-ups): PanoramaMemory(H, fpx) gets 2x H and fpx (1508 columns per full circle instead of
    754); the full-circle test is 151 steps of 2.4 deg either way (degrees, not pixels).
  - colour fingerprint: means / std / adaptive 2x3 pooling - size-free. mem_ghost: W-based (its patchwork block
    widths are in pixels, but patchwork is off in this recipe: --mem-patch-p 0). Wrecks: size-free. Zoom: not used.
  - THE CHECK-UPS dream at 2x (real 2x history from cache_v11), then every number is measured on the dreams
    AREA-SHRUNK to 96x168: the melt critic and glitch meter were made for 96x168, and so were the old meters
    (sharpness, clean2, scene meter, SSIM, layout). The real reference frames they compare with are cache_v11's
    own frames shrunk the same way (--ref-cache shrink; NOT cache_v3, whose colour detail is 2.3x lower - see
    ShrunkFrames). The melt critic reads real shrunk-2x and cache_v3 frames alike (0.184 vs 0.175), so melt
    scores compare directly with v9's. Videos / pictures / live view show the 2x dreams.
  - best_melt.pt logic unchanged (lowest melt critic with turn-back return >= 0.9).
  - GPU room check scaled for 4x the pixels; --vram-cap-gb caps PyTorch's memory (24 GB tonight).
  - --budget-hours H (relative, from the process start) or --end-at HH:MM: stop training in time for one last
    check-up + save to finish within the budget (uses the last check-up's measured length).
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
import functools
import json
import math
import pathlib
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from build_cache import compute_valid                                          # noqa: E402
from data_v3 import (NON_WALKING, InfiniteWeightedSampler, load_cache_any,       # noqa: E402
                     parse_source_weights, source_shares, to_model_space)
from data_v6 import needed_positions, position_lookup                           # noqa: E402
from data_v7 import SparseWindowDatasetV7, describe_table, key_motion_table, load_motion  # noqa: E402
from data_v8 import mem_ghost                                                   # noqa: E402
from model_v3 import FP_FRAMES, colour_fingerprint                              # noqa: E402
from model_v7 import NULL_ACTION, build_model_v7                                # noqa: E402
from model_v11 import init_deep_from_mid                                        # noqa: E402
from train import Live, safe_replace                                            # noqa: E402
from train_v1 import pose_string, sample_sigma_cond, sharpness, grain          # noqa: E402
from train_v2 import gpu_free_gb, write_live_frames, write_mp4                  # noqa: E402
from train_v3 import GraphedSampler, eager_sample, sample_sigma, sf_counts      # noqa: E402
from train_v6 import SCENE_FIT, V5_RULER_STARTS                                 # noqa: E402
from train_v7 import MemIndexV7, motion_shift_px, self_force_v7, shift_frames, turn_clip  # noqa: E402
from train_v8 import (combined_score, eval_main as eval_main_v8, memory_suite,   # noqa: E402
                      real_collage_ref, ruler_key_list)
from turn_test import TURN_STEPS, TURN_WALK, turn_picture                       # noqa: E402
from glitch_meter import glitch_time                                           # noqa: E402
# v11: the check-up suite's parts (train_v8.memory_suite, rebuilt below to score at 96x168)
from scene_meter import features as scene_features                              # noqa: E402
from train_v3 import score_walks                                                # noqa: E402
from train_v6 import scene_score                                                # noqa: E402
from train_v7 import dream_walks_v7                                             # noqa: E402
from train_v8 import CIRCLE_STEPS, collage_numbers, dream_mem, ssim             # noqa: E402
from turn_test import layout, turn_scores                                       # noqa: E402

REF_HW = (96, 168)        # every check-up number is measured at this size (the melt critic's / v9's)


# ---------------------------------------------------------------- v11: 2x helpers

def down_to(u8, hw=REF_HW):
    """uint8 [..., H, W, 3] -> [..., h, w, 3] by area averaging (exact block mean when H, W are multiples of h, w;
    cv2 INTER_AREA otherwise). Done row by row to keep RAM low. Frames already at hw come back unchanged."""
    H, W = u8.shape[-3:-1]
    h, w = hw
    if (H, W) == (h, w):
        return u8
    lead = u8.shape[:-3]
    flat = u8.reshape(-1, H, W, 3)
    out = np.empty((flat.shape[0], h, w, 3), np.uint8)
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        for i in range(0, flat.shape[0], 256):
            blk = flat[i:i + 256].reshape(-1, h, fy, w, fx, 3).astype(np.uint16).sum((2, 4))
            out[i:i + 256] = ((blk + (fy * fx) // 2) // (fy * fx)).astype(np.uint8)
    else:
        import cv2
        for i in range(flat.shape[0]):
            out[i] = cv2.resize(flat[i], (w, h), interpolation=cv2.INTER_AREA)
    return out.reshape(*lead, h, w, 3)


class ShrunkFrames:
    """A read-only stand-in for a 2x frame memmap that hands out every slice shrunk to 96x168 (down_to).
    The check-up meters read their REAL reference frames through it, so dreams (shrunk the same way) and references
    are the same kind of picture. (data/cache_v3 is NOT: it went through 256x144 video with half-res colour, so its
    fine colour detail is 2.3x lower and its grey sharpness 0.81x of the shrunk 2x footage - measured on 600 real
    frames 2026-10-22; the melt critic reads both alike: 0.175 vs 0.184.)"""

    def __init__(self, fm, hw=REF_HW):
        self.fm, self.hw = fm, hw
        self.shape = (fm.shape[0], hw[0], hw[1], 3)
        self.dtype = np.dtype(np.uint8)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return down_to(np.asarray(self.fm[idx]), self.hw)


def ref_frames(spec, frames_mm, index, actions_np):
    """--ref-cache: 'shrink' (default) = the training cache's own frames shrunk to 96x168; or a 96x168 cache folder
    (e.g. data/cache_v3) that lines up frame for frame."""
    if spec == "shrink":
        return ShrunkFrames(frames_mm)
    ref_index, fr, ref_actions, _ = load_cache_any(pathlib.Path(spec))
    if ([(c["name"], c["start"], c["end"]) for c in ref_index["cities"]] != [(c["name"], c["start"], c["end"]) for c in index["cities"]]
            or not np.array_equal(ref_actions, actions_np)):
        sys.exit(f"{spec} does not line up frame for frame with the training cache")
    if list(ref_index["shape"][1:3]) != list(REF_HW):
        sys.exit(f"--ref-cache must hold {REF_HW} frames (the meters' size), it holds {ref_index['shape'][1:3]}")
    return fr


def v11_cfg(parent_cfg, frame_hw):
    """A v9 (or v7/v8) save's config -> the same network, dreaming frame_hw frames, marked as trained by v11."""
    if parent_cfg.get("version") != "v7":
        raise ValueError(f"v11 warm-starts from a v9/v8/v7 save (model_cfg version 'v7'), got {parent_cfg.get('version')}")
    return dict(parent_cfg, frame_hw=list(frame_hw), trained_as="v11", parent_frame_hw=list(parent_cfg["frame_hw"]))


def end_deadline(hhmm, now=None):
    """'09:00' -> the next time it is 09:00 (epoch seconds). If that is more than 16 h away (we are just past it),
    give the run 1 hour from now instead."""
    import datetime as dt
    now = dt.datetime.now() if now is None else now
    hh, mm = [int(x) for x in hhmm.split(":")]
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    if (t - now).total_seconds() > 16 * 3600:
        t = now + dt.timedelta(hours=1)
    return t.timestamp()


def memory_suite_v11(model, fm, fm_ref, actions_np, motion_mm, starts, seeds, table, motion_scale, sd, ruler_keys,
                     steps=10, smax_mem=2.0, walk=TURN_WALK, n=TURN_STEPS, nomem=True, log=print, write_rule="newest",
                     smax=5.0):
    """train_v8.memory_suite for a 2x model: the dreams are made at the model's size from fm (cache_v11: real 2x
    history), then shrunk to 96x168 and scored against fm_ref (real frames shrunk the same way) with exactly v8's / v9's meters.
    Returns (numbers, dreams): dreams[...] are the 96x168 versions (what the meters saw), dreams[... + '_2x'] the
    model's own frames (for videos and pictures)."""
    rows = [s for _ in range(seeds) for s in starts]
    k = len(rows)
    res, dreams = {}, {}
    t0 = time.time()
    tk, ck = [1] * walk + [6] * n + [5] * n, [1] * walk + [6] * n + [1] * n
    d, _ = dream_mem(model, fm, actions_np, motion_mm, rows + rows, [tk] * k + [ck] * k, table, motion_scale,
                     steps, smax_mem=smax_mem, write_rule=write_rule, smax=smax)
    dreams["turn_2x"], dreams["ctrl_2x"] = d[:k], d[k:]
    d = down_to(d)
    td, cd = d[:k], d[k:]
    r = turn_scores(td, cd, len(starts), scene_features, sd, walk, n)
    res["turn"] = {kk: r[kk] for kk in ("return_sim", "ctrl_sim", "memory", "away_sim", "return_sim_scene",
                                       "turned_right_deg", "turned_back_deg")}
    res["back"] = collage_numbers(td[:, walk + 2 * n - 1], rows, fm_ref, sd)
    res["back"]["ssim_pre"] = round(float(ssim(td[:, walk - 1], td[:, walk + 2 * n - 1]).mean()), 4)
    res["back"]["ssim_ctrl"] = round(float(ssim(td[:, walk - 1], cd[:, walk + 2 * n - 1]).mean()), 4)
    dreams["turn"], dreams["ctrl"] = td, cd
    log(f"  memory turn test ({time.time() - t0:.0f}s): {res['turn']}; turned-back frames {res['back']}")
    circ = [1] * walk + [6] * CIRCLE_STEPS
    d2, _ = dream_mem(model, fm, actions_np, motion_mm, rows, circ, table, motion_scale, steps, smax_mem=smax_mem,
                      write_rule=write_rule, smax=smax)
    d2 = down_to(d2)
    pre, end = layout(d2[:, walk - 1]), layout(d2[:, -1])
    other = [float(((pre[i] - pre[j]) ** 2).mean()) for i in range(k) for j in range(k) if i != j and i // len(starts) == j // len(starts)]
    res["circle_return_sim"] = round(float(1 - np.mean([float(((pre[i] - end[i]) ** 2).mean()) for i in range(k)]) / np.mean(other)), 3)
    res["circle_end"] = collage_numbers(d2[:, -1], rows, fm_ref, sd)
    res["circle_end"]["ssim_pre"] = round(float(ssim(d2[:, walk - 1], d2[:, -1]).mean()), 4)
    dreams["circle"] = d2
    log(f"  memory full circle ({time.time() - t0:.0f}s): return_sim {res['circle_return_sim']}; end frames {res['circle_end']}")
    T = len(ruler_keys)
    d3x, low = dream_mem(model, fm, actions_np, motion_mm, rows, ruler_keys, table, motion_scale, steps, smax_mem=smax_mem,
                         write_rule=write_rule, smax=smax)
    d3 = down_to(d3x)
    summ, reps = score_walks(d3, rows, fm_ref, model.K, T)
    res["ruler_mem"] = {"melt_median": summ["melt_median"], "clean2_median": summ["clean2_median"],
                        "clean2_mean": summ["clean2_mean"], "scene": scene_score(d3, rows, fm_ref, sd),
                        "low_noise_share": round(low, 3)}
    dreams["ruler_mem"], dreams["ruler_mem_2x"], dreams["ruler_mem_sharp"] = d3, d3x, reps[0][1]
    log(f"  {T // 10} s ruler WITH memory ({time.time() - t0:.0f}s): {res['ruler_mem']}")
    if nomem:
        dev = next(model.parameters()).device
        d4 = down_to(dream_walks_v7(model, fm, actions_np, motion_mm, rows, ruler_keys, steps, dev, table,
                                    start="shift", motion_scale=motion_scale))
        summ4, _ = score_walks(d4, rows, fm_ref, model.K, T)
        res["ruler_nomem"] = {"melt_median": summ4["melt_median"], "clean2_median": summ4["clean2_median"],
                              "clean2_mean": summ4["clean2_mean"], "scene": scene_score(d4, rows, fm_ref, sd)}
        dreams["ruler_nomem"] = d4
        log(f"  {T // 10} s ruler WITHOUT memory, shift start ({time.time() - t0:.0f}s): {res['ruler_nomem']}")
    res["suite_secs"] = round(time.time() - t0)
    return res, dreams


def honest_numbers(mc, dreams, rows, fm_ref, T):
    """The honest meters (melt critic + glitch meter) on the 96x168 dreams; glitch start frames from fm_ref."""
    out = {}
    for key in ("ruler_mem", "ruler_nomem"):
        if key in dreams:
            scs = [mc.report(dd)["melt_score"] for dd in dreams[key]]
            gts = [glitch_time(dd, np.array(fm_ref[s_ - 48:s_])) for dd, s_ in zip(dreams[key], rows)]
            out[key] = {"melt_critic": round(float(np.mean(scs)), 3),
                        "glitch_time_mean": round(float(np.mean([g_ if g_ is not None else T / 10 for g_ in gts])), 2),
                        "melt_critic_per_walk": [round(float(x), 3) for x in scs]}
    return out


def eval_main_v11(args):
    """--eval-mem for a 2x save: the check-up suite + honest meters, numbers at 96x168 (comparable with v9's)."""
    import os
    from melt_critic import MeltCritic
    device = "cuda"
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v11"
    index, fm, act, _ = load_cache_any(cache_dir)
    fm_ref = ref_frames(args.ref_cache, fm, index, act)
    mot, minfo = load_motion(cache_dir)
    tmp = HERE / "_exp" / f"v11_eval_ck_{os.getpid()}.pt"                # copy first: never hold a live run's file open
    shutil.copy(args.eval_mem, tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    cfg = ck["model_cfg"]
    if list(cfg["frame_hw"]) != list(fm.shape[1:3]):
        cfg = v11_cfg(cfg, fm.shape[1:3])
    m = build_model_v7(cfg).to(device).eval()
    m.load_state_dict(ck["ema"])
    sd = np.load(SCENE_FIT)["sd"]
    starts = [int(x) for x in args.ruler_starts.split(",")]
    table = key_motion_table(minfo, args.test_turn_deg)
    ruler_keys = [a for a, n in ruler_key_list(args.ruler_secs) for _ in range(n)]
    print(f"{args.eval_mem}: step {ck['step']}, dreaming at {cfg['frame_hw']}", flush=True)
    res, dreams = memory_suite_v11(m, fm, fm_ref, act, mot, starts, args.ruler_seeds, table, minfo["scale"], sd,
                                   ruler_keys, args.rollout_steps, args.mem_smax, args.turn_walk, args.turn_steps,
                                   log=lambda s: print(s, flush=True), nomem=not args.eval_skip_nomem,
                                   write_rule=args.mem_write, smax=5.0 * args.noise_scale)
    rows = [s_ for _ in range(args.ruler_seeds) for s_ in starts]
    for key, v in honest_numbers(MeltCritic(device), dreams, rows, fm_ref, len(ruler_keys)).items():
        res[key].update(v)
    res["real"] = real_collage_ref(starts, fm_ref, sd)
    res["combined_score_no_guard"] = combined_score(res, args.ruler_secs, res["real"]["scene"])
    res["ckpt"], res["step"] = str(args.eval_mem), ck["step"]
    print(json.dumps(res, indent=1), flush=True)
    if args.eval_out:
        out = pathlib.Path(args.eval_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

def zoom_frames(x, z, cy=0.0):
    """Zoom frames x [B, 3, H, W] IN by z [B] (>= 1) around a point cy (-1 top .. +1 bottom; 0 = middle), bicubic.
    'Zoom ghost' (2026-10-22): real footage zooms ~1.5% per forward W-step (train/_exp/v9_zoom_fit.py: zooming the
    slid previous frame by 1 + 0.015 * forward makes it 34% closer to the real next frame), so a ghost that is only
    slid lags behind the walk, the model copies the lag and walking feels sluggish (v7's turning story, for W)."""
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0] = 1.0 / z
    th[:, 1, 1] = 1.0 / z
    th[:, 1, 2] = cy * (1.0 - 1.0 / z)
    grid = torch.nn.functional.affine_grid(th, x.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(x.float(), grid, mode="bicubic", padding_mode="border", align_corners=False)
    return torch.where((z > 1.0)[:, None, None, None], out.clamp(-1, 1), x.float())


def noise_fill_edges(ghost, prev, dx):
    """Turn-edge practice (2026-10-22): the strip a turn reveals (dx [B] whole px; + = content moved left, so the
    RIGHT edge is new) is normally the edge column STRETCHED (shift_frames' border padding), which builds horizontal
    streaks on long turns. Fill it instead with noise in the frame's own colours (mean + 0.5 x std per channel), so
    the model learns that a noisy strip means 'invent here' (train/_exp/v9_edge.py: a flat fill at play time alone
    gave beige texture walls - it has to be practised)."""
    B, C, H, W = ghost.shape
    colsx = torch.arange(W, device=ghost.device)[None, :]
    rev = ((dx[:, None] > 0) & (colsx >= W - dx[:, None])) | ((dx[:, None] < 0) & (colsx < -dx[:, None]))
    m = prev.mean(dim=(2, 3), keepdim=True)
    sd = prev.std(dim=(2, 3), keepdim=True)
    fill = (m + 0.5 * sd * torch.randn_like(prev)).clamp(-1, 1)
    return torch.where(rev[:, None, None, :], fill, ghost)


BANK_WIDE_P = 0.2         # error bank: share of draws from ANY bin (the rest from --err-bank-draw)
BANK_MIN_FILL = 16        # error bank: a bin is drawn from once it holds this many errors


def v9_cfg_from_v8(cfg):
    """A v8 (or v7) save's config -> the same config, marked as trained by train_v9 (version stays "v7")."""
    if cfg.get("version") != "v7":
        raise ValueError(f"v9 warm-starts from a v8/v7 save (model_cfg version 'v7'), got {cfg.get('version')}")
    return dict(cfg, trained_as="v9")


def window_layout(model, sf_w, resample_n):
    """-> (SF_MAX, L, fp_train, positions, mem_positions).
    SF_MAX = how many frames before the target may be dreamed/wrecked (the loader must read their memories).
    fp_train and mem_positions are v8's (same offsets from the target), so the colour fingerprint and the
    memory-collage ghosts see exactly the frames they saw in v8 whatever --resample-n is."""
    Hn = model.history
    n_sf = len(sf_w)
    SF_MAX = max(n_sf, resample_n)
    L = Hn + SF_MAX
    fp_train = [model.fp_offset + n_sf - i for i in range(FP_FRAMES)]
    positions = needed_positions(model.mem_offsets(), L, SF_MAX, fp_train)
    L8 = Hn + n_sf
    v8_offsets = [L8 - p for p in needed_positions(model.mem_offsets(), L8, n_sf, fp_train)]
    mem_positions = sorted(L - o for o in v8_offsets)
    assert set(mem_positions) <= set(positions)
    return SF_MAX, L, fp_train, positions, mem_positions


# ---------------------------------------------------------------- A: self-resampling of the newest memory slots

class OneCall:
    """ONE no-grad clean-up call of the EMA model, D(x; sigma) with a sigma PER ROW, memory label 0.
    With a GraphedSampler, it replays (or records) that sampler's CUDA graph for this batch size: the recording
    takes x and sigma [B] as inputs, so the same recording serves both (same maths, checked at start-up)."""

    def __init__(self, ema, graphed=None):
        self.ema, self.graphed = ema, graphed

    @torch.no_grad()
    def __call__(self, x, sig, ctx, a, col, far):
        if self.graphed is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                den, _ = self.ema(x, sig, ctx, 0.0, a, col, far)
            return den.float()
        B = x.shape[0]
        if B not in self.graphed.graphs:
            self.graphed.graphs[B] = self.graphed._build(ctx, a, col, far)
        g, st, out = self.graphed.graphs[B]
        st["ctx"].copy_(ctx)
        st["a"].copy_(a)
        st["col"].copy_(col)
        st["far"].copy_(far)
        st["x"].copy_(x)
        st["s"].copy_(sig)
        g.replay()
        return out.clone()


def draw_wreck_sigmas(n, args, device):
    """Per-row wreck level: log(sigma_s) ~ Normal(log median, std), clipped (research Pick 1)."""
    z = torch.randn(n, device=device) * args.resample_sigma_std + math.log(args.resample_sigma_median)
    return z.exp().clamp(args.resample_sigma_min, args.resample_sigma_max)


@torch.no_grad()
def resample_memory(call, mi, fr, act, mot, colour, a, b, n_slots, sig, L, generator=None):
    """Wreck the target's newest n_slots memory slots (offsets n_slots..1) for rows a..b of the batch, OLDEST
    FIRST: real frame + sigma_s noise -> one EMA call with the memory as it stands at that position (older
    slots already wrecked, real keys + motion, label 0) -> its x0 guess replaces the frame in fr (in place, so
    every later read of that window position, the target's memory, ghosts and panorama, sees the wrecked one).
    sig [b - a]. Returns fr."""
    if b <= a:
        return fr
    sb = sig[:, None, None, None]
    for o in range(n_slots, 0, -1):
        g = L - o
        r = mi.row(g)
        ctx, k, ext = mi(fr[a:b], act[a:b], mot[a:b], g)
        real = fr[a:b, r].float()
        noise = torch.randn(real.shape, device=real.device, generator=generator)
        fr[a:b, r] = call(real + sb * noise, sig, ctx, k, colour[a:b], ext).clamp(-1, 1).to(fr.dtype)
    return fr


# ---------------------------------------------------------------- B: the error bank for the older slots

class ErrorBank:
    """The model's own x0 errors (D - target) from the normal training pass, binned by the target's sigma
    (n_bins log-spaced bins over [lo, hi]), per_bin of them per bin, FIFO, bf16 on the GPU."""

    def __init__(self, n_bins, per_bin, frame_shape, lo, hi, device):
        self.n, self.per = n_bins, per_bin
        self.llo, self.lhi = math.log(lo), math.log(hi)
        self.buf = torch.zeros((n_bins * per_bin, *frame_shape), dtype=torch.bfloat16, device=device)
        self.fill, self.ptr = [0] * n_bins, [0] * n_bins
        self.fill_t = torch.zeros(n_bins, dtype=torch.long, device=device)
        self.dev = device

    def gb(self):
        return self.buf.numel() * 2 / 2**30

    def bins(self, sigma):
        u = (sigma.float().clamp(min=1e-8).log() - self.llo) / (self.lhi - self.llo) * self.n
        b = u.floor().long()
        return torch.where((u >= 0) & (u < self.n), b, torch.full_like(b, -1))

    @torch.no_grad()
    def add(self, err, sigma):
        """err [B, 3, H, W] (float), sigma [B]: store every row whose sigma falls in a bin."""
        bc = self.bins(sigma).cpu().tolist()
        dst, src = [], []
        for i, k in enumerate(bc):
            if k < 0:
                continue
            dst.append(k * self.per + self.ptr[k])
            src.append(i)
            self.ptr[k] = (self.ptr[k] + 1) % self.per
            self.fill[k] = min(self.per, self.fill[k] + 1)
        if src:
            self.buf.index_copy_(0, torch.tensor(dst, device=self.dev),
                                 err.index_select(0, torch.tensor(src, device=self.dev)).to(torch.bfloat16))
            self.fill_t.copy_(torch.tensor(self.fill))
        return len(src)

    @torch.no_grad()
    def draw(self, n, draw_lo, draw_hi, wide_p=BANK_WIDE_P, min_fill=BANK_MIN_FILL):
        """n errors -> ([n, 3, H, W] bf16, ok [n] bool = its bin held >= min_fill errors)."""
        u = torch.rand(n, device=self.dev)
        lo = torch.where(u < wide_p, torch.full_like(u, self.llo), torch.full_like(u, math.log(draw_lo)))
        hi = torch.where(u < wide_p, torch.full_like(u, self.lhi - 1e-6), torch.full_like(u, math.log(draw_hi)))
        s = (lo + (hi - lo) * torch.rand(n, device=self.dev)).exp()
        b = self.bins(s).clamp(min=0)
        f = self.fill_t[b]
        j = (torch.rand(n, device=self.dev) * f.clamp(min=1)).long().clamp(max=self.per - 1)
        return self.buf[b * self.per + j], f >= min_fill


@torch.no_grad()
def apply_bank(context, bank, rows, n_old, args):
    """Add banked errors (x a random scale) to the older memory slots [0, n_old) of the chosen rows, each slot
    with probability --err-bank-p, then clip to [-1, 1]. context [B, n_mem, 3, H, W] (a copy, changed in
    place). Returns (rows touched, slots touched)."""
    B = context.shape[0]
    pick = rows[:, None] & (torch.rand(B, n_old, device=context.device) < args.err_bank_p)
    m = int(pick.sum())
    if m == 0:
        return 0, 0
    e, ok = bank.draw(m, args.err_bank_draw[0], args.err_bank_draw[1])
    lo, hi = args.err_bank_scale
    sc = (lo + (hi - lo) * torch.rand(m, device=context.device)) * ok.float()
    sub = context[:, :n_old]
    sub[pick] = (sub[pick] + sc[:, None, None, None] * e.float()).clamp(-1, 1)
    used = torch.zeros_like(pick)
    used[pick] = ok
    return int(used.any(1).sum()), int(ok.sum())


def loss_ghost_d(model, target, context, actions, sigma, sigma_cond, colour, far_act, ghost, use_ghost):
    """model_v7.DenoiserV7.loss_ghost word for word (same random draws in the same order -> the same loss),
    plus the x0 guess D (detached) for the error bank."""
    B = target.shape[0]
    sc = model.per_frame(sigma_cond, B, target.device)
    colour, far_act = model._extras(colour, far_act, B, target.device)
    b = lambda t: t[:, None, None, None]
    context = context + sc[:, :, None, None, None] * torch.randn_like(context)
    noise = torch.randn_like(target)
    centre = torch.where(use_ghost[:, None, None, None], ghost.to(target.dtype), target)
    x_noisy = centre + b(sigma) * noise
    c_skip, c_out, F_out = model._unet(x_noisy, sigma, context, sc, actions, colour, far_act)
    F_target = (target - b(c_skip) * x_noisy) / b(c_out)
    loss = F.mse_loss(F_out.float(), F_target)
    with torch.no_grad():
        D = b(c_skip) * x_noisy + b(c_out) * F_out.detach().float()
    return loss, D


# ---------------------------------------------------------------- the picture: wrecked memory vs play's dreams

def wreck_picture_main(args):
    """Real newest memory slot vs the same slot WRECKED by resample_memory (the training code, the warm-start
    model's EMA, all --resample-n newest slots wrecked oldest first at one sigma) at several sigma levels, for
    the ruler walks' first dreamed moment; then play's own dreams (v8 final + panorama memory, from
    train/_exp/v9/diag_v8final_dreams.npy) at a few times. Numbers (sharpness/grain vs real) go to a .json."""
    import cv2
    device = "cuda"
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    ck = torch.load(args.init_v8, map_location="cpu", weights_only=False)
    model = build_model_v7(ck["model_cfg"]).to(device).eval()
    model.load_state_dict(ck["ema"])
    step = ck["step"]
    del ck
    sf_w = [float(x) for x in args.sf_weights.split(",")]
    SF_MAX, L, fp_train, positions, _ = window_layout(model, sf_w, args.resample_n)
    lut = position_lookup(positions, L)
    ds = SparseWindowDatasetV7(cache_dir, positions, L, flip_p=0.0)
    starts = [int(x) for x in args.ruler_starts.split(",")]
    items = [ds[s - L] for s in starts]                                 # target (window position L) = frame s
    fr0 = to_model_space(torch.stack([v[0] for v in items]).to(device))
    act = torch.stack([v[1] for v in items]).to(device)
    mot = torch.stack([v[3] for v in items]).to(device)
    mi = MemIndexV7(model, lut, device)
    fp_rows = lut[torch.tensor([L - o for o in fp_train])].to(device)
    colour = colour_fingerprint(fr0.index_select(1, fp_rows))
    call = OneCall(model)
    r1, rN = mi.row(L - 1), mi.row(L - args.resample_n)
    u8 = lambda x: ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    real1, realN = u8(fr0[:, r1]), u8(fr0[:, rN])
    levels = [float(x) for x in args.wreck_levels.split(",")]
    wrecked, wreckedN, nums = {}, {}, {"ckpt": str(args.init_v8), "step": step, "resample_n": args.resample_n,
                                       "walk_starts": starts, "levels": {}}
    sr = lambda a, b: float(np.median(sharpness(a) / np.maximum(sharpness(b), 1e-6)))
    for s in levels:
        fr = fr0.clone()
        g = torch.Generator(device=device).manual_seed(0)
        resample_memory(call, mi, fr, act, mot, colour, 0, len(starts), args.resample_n,
                        torch.full((len(starts),), s, device=device), L, generator=g)
        wrecked[s], wreckedN[s] = u8(fr[:, r1]), u8(fr[:, rN])
        d = np.abs(wrecked[s].astype(np.float32) - real1.astype(np.float32))
        nums["levels"][str(s)] = {
            "sharp_ratio_newest": round(sr(wrecked[s], real1), 3),
            "sharp_ratio_first_wrecked": round(sr(wreckedN[s], realN), 3),
            "grain_newest": round(float(np.median(grain(wrecked[s]))), 3),
            "mean_abs_diff_255": round(float(d.mean()), 2),
            "p99_abs_diff_255": round(float(np.percentile(d, 99)), 1)}
    nums["real_grain"] = round(float(np.median(grain(real1))), 3)
    # how likely is each level in training? (share of rows at <= that sigma)
    zs = torch.randn(200000) * args.resample_sigma_std + math.log(args.resample_sigma_median)
    ss = zs.exp().clamp(args.resample_sigma_min, args.resample_sigma_max)
    nums["training_sigma_quantiles"] = {q: round(float(ss.quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    # play's own dreams
    diag = HERE / "_exp" / "v9" / "diag_v8final_dreams.npy"
    dream_t = [2, 5, 12, 20, 30]
    dreams = None
    if diag.exists():
        dd = np.load(diag, mmap_mode="r")
        if dd.shape[0] == len(starts):
            dreams = np.array(dd[:, [t * 10 - 1 for t in dream_t]])
            _, fm, _, _ = load_cache_any(cache_dir)
            nums["play_dreams"] = {}
            for j, t in enumerate(dream_t):
                ref = np.array([sharpness(np.array(fm[s:s + 600])).mean() for s in starts])
                nums["play_dreams"][f"{t}s"] = {
                    "sharp_ratio": round(float(np.median(sharpness(dreams[:, j]) / ref)), 3),
                    "grain": round(float(np.median(grain(dreams[:, j]))), 3)}
    # the sheet
    S = 2
    up = lambda im: cv2.resize(im, (im.shape[1] * S, im.shape[0] * S), interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def tile(im, text):
        im = up(np.ascontiguousarray(im))[:, :, ::-1].copy()
        cv2.putText(im, text, (4, 16), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, text, (4, 16), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return im

    def banner(text, w):
        b = np.full((30, w, 3), 24, np.uint8)
        cv2.putText(b, text, (6, 21), font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return b

    show = list(range(min(args.wreck_walks, len(starts))))
    rows_ = []
    for i in show:
        cells = [tile(real1[i], "REAL newest slot")]
        for s in levels:
            cells.append(tile(wrecked[s][i], f"wrecked s={s:g}"))
        rows_.append(np.concatenate(cells, 1))
    W = rows_[0].shape[1]
    parts = [banner(f"A. Training's wrecked memory: newest slot (offset 1) after all {args.resample_n} newest slots were wrecked "
                    f"oldest-first, one call each ({pathlib.Path(args.init_v8).name}, step {step}). Training draws "
                    f"sigma median {args.resample_sigma_median}, 10-90%: {nums['training_sigma_quantiles'][0.1]}-"
                    f"{nums['training_sigma_quantiles'][0.9]}", W)] + rows_
    i = show[0]
    cells = [tile(np.zeros_like(real1[i]), "|wrecked - real| x4")]
    for s in levels:
        d = np.clip(np.abs(wrecked[s][i].astype(np.int16) - real1[i].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        cells.append(tile(d, f"diff x4 s={s:g}"))
    parts += [banner("B. Where the wreck changed the frame (walk 1, x4)", W), np.concatenate(cells, 1)]
    parts.append(banner("Wrecked newest slot, sharpness vs real (mean |diff| /255), 5 walks: " + ", ".join(
        f"s={s:g}: {nums['levels'][str(s)]['sharp_ratio_newest']:.2f} ({nums['levels'][str(s)]['mean_abs_diff_255']:.1f})"
        for s in levels), W))
    if dreams is not None:
        prow = []
        for i in show:
            cells = [tile(real1[i], "REAL (start)")]
            for j, t in enumerate(dream_t):
                cells.append(tile(dreams[i, j], f"PLAY dream {t}s"))
            while len(cells) < len(levels) + 1:
                cells.append(tile(np.full_like(real1[i], 40), ""))
            prow.append(np.concatenate(cells[:len(levels) + 1], 1))
        txt = ", ".join(f"{k} {v['sharp_ratio']:.2f}" for k, v in nums["play_dreams"].items())
        parts += [banner(f"C. Play's own dreams (v8 final + panorama memory, diag run): sharpness vs real {txt}", W)] + prow
    sheet = np.concatenate(parts, 0)
    out = pathlib.Path(args.wreck_picture)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out.with_suffix(".json").write_text(json.dumps(nums, indent=1), encoding="utf-8")
    print(json.dumps(nums, indent=1))
    print(f"-> {out}")


# ---------------------------------------------------------------- main

def pair(text):
    a, b = [float(x) for x in text.split(",")]
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v11", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-v9", "--init-v8", "--init-v7", dest="init_v8", default="",
                    help="v8 (or v7) save to warm-start a NEW run from (use a COPY, not the live file)")
    ap.add_argument("--eval-mem", default="", help="just run the 2x check-up suite (+ honest meters, scored at 96x168) on this save, then exit")
    ap.add_argument("--eval-out", default="", help="--eval-mem: write the numbers here (.json) + first-seed dreams (.npz)")
    ap.add_argument("--wreck-picture", default="", help="just draw real vs wrecked memory slots (+ play's dreams) here (.jpg), then exit")
    ap.add_argument("--wreck-levels", default="0.05,0.15,0.3,0.6,1.2,2.5", help="--wreck-picture: the sigma levels shown")
    ap.add_argument("--wreck-walks", type=int, default=3, help="--wreck-picture: how many ruler walks to show")
    ap.add_argument("--smoke", action="store_true", help="checkpoints/out/live all under train/_smoke/v11/")
    ap.add_argument("--cache", default=None, help="training cache (default data/cache_v11, 192x336)")
    ap.add_argument("--ref-cache", default="shrink",
                    help="v11: real frames the check-up meters compare with: 'shrink' = the training cache shrunk to "
                         "96x168 (default, same kind of picture as the shrunk dreams), or a 96x168 cache folder")
    ap.add_argument("--budget-hours", type=float, default=0,
                    help="v11: total time for this process (from its start, incl. the warm-start check and the LAST "
                         "check-up): training stops early enough for the last check-up + save to fit (0 = no budget)")
    ap.add_argument("--end-at", default="", help="v11: or an absolute end time (HH:MM), same rule")
    ap.add_argument("--log-copy", default="", help="v11: also append every log line to this file")
    ap.add_argument("--checkup-reserve-min", type=float, default=25,
                    help="v11: --end-at: expected length of one check-up until the first one has been timed")
    ap.add_argument("--colour-drop", type=float, default=0.1)
    # v9 A: self-resampling recovery practice
    ap.add_argument("--resample-frac", type=float, default=0.4,
                    help="v9: share of rows whose newest memory slots are WRECKED (real frame + noise -> one EMA call)")
    ap.add_argument("--resample-n", type=int, default=8, help="v9: how many of the newest memory slots get wrecked (offsets N..1, max 8)")
    ap.add_argument("--resample-sigma-median", type=float, default=0.3, help="v9: median wreck noise (one per row, log-normal)")
    ap.add_argument("--resample-sigma-std", type=float, default=1.0, help="v9: spread of log(wreck noise)")
    ap.add_argument("--resample-sigma-min", type=float, default=0.03)
    ap.add_argument("--resample-sigma-max", type=float, default=3.0)
    # v9 B: error bank for the older slots
    ap.add_argument("--err-bank-frac", type=float, default=0.5,
                    help="v9: share of rows whose OLDER memory slots (offset > --resample-n) get the model's own banked errors")
    ap.add_argument("--err-bank-p", type=float, default=0.5, help="v9: chance per older slot of those rows")
    ap.add_argument("--err-bank-bins", type=int, default=10, help="v9: log-spaced sigma bins of the bank")
    ap.add_argument("--err-bank-size", type=int, default=300, help="v9: errors kept per bin (FIFO)")
    ap.add_argument("--err-bank-range", type=pair, default=(0.03, 3.0), help="v9: sigma range of the bins")
    ap.add_argument("--err-bank-draw", type=pair, default=(0.1, 1.0), help="v9: 80%% of draws come from bins in this sigma range")
    ap.add_argument("--err-bank-scale", type=pair, default=(0.5, 1.5), help="v9: random scale of each pasted error")
    # v7 steering + ghost practice (unchanged)
    ap.add_argument("--steer-drop", type=float, default=0.1)
    ap.add_argument("--onset-weight", type=float, default=4.0)
    ap.add_argument("--bigturn-weight", type=float, default=2.0)
    ap.add_argument("--onset-deg", type=float, default=1.5)
    ap.add_argument("--onset-ratio", type=float, default=0.6)
    ap.add_argument("--bigturn-deg", type=float, default=3.0)
    ap.add_argument("--test-turn-deg", type=float, default=2.4)
    ap.add_argument("--ghost-frac", type=float, default=0.15, help="v7's shifted-ghost rows")
    ap.add_argument("--ghost-sigma-min", type=float, default=4.0)
    ap.add_argument("--ghost-sigma-max", type=float, default=5.0)
    ap.add_argument("--ghost-shift-p", type=float, default=1.0)
    ap.add_argument("--ghost-zoom-k", type=float, default=0.0,
                    help="zoom the shifted-ghost rows by 1 + k * forward (W-steps); real footage ~0.015 (0 = off, v9)")
    ap.add_argument("--ghost-zoom-cy", type=float, default=0.0, help="zoom centre height (-1 top .. +1 bottom)")
    ap.add_argument("--ghost-edge-fill", choices=["border", "noise"], default="border",
                    help="shifted-ghost rows: fill the strip a turn reveals by stretching the edge (v7-v9) or with noise")
    # v8 memory-collage ghosts (unchanged)
    ap.add_argument("--mem-ghost-frac", type=float, default=0.2, help="share of rows with a panorama-memory ghost")
    ap.add_argument("--mem-patch-p", type=float, default=0.65, help="of those: share of synthetic-revisit patchwork rows")
    ap.add_argument("--mem-gap-max", type=int, default=10)
    ap.add_argument("--mem-age-min", type=int, default=8)
    ap.add_argument("--mem-age-max", type=int, default=60)
    ap.add_argument("--mem-jitter", type=float, default=0.04)
    ap.add_argument("--mem-sigma-low", default="1.8,3.2")
    ap.add_argument("--mem-low-p", type=float, default=0.0)
    ap.add_argument("--mem-write", choices=["newest", "centre"], default="centre")
    ap.add_argument("--eval-skip-nomem", action="store_true")
    ap.add_argument("--mem-smax", type=float, default=2.0)
    # v11.1 (2026-10-23): the whole-street view + noise sized for 2x pictures
    ap.add_argument("--deep-ctx", action="store_true",
                    help="v11.1: a NEW run gets model_v11's deep context stage (12x21 view; starts as a no-op)")
    ap.add_argument("--deep-donor", default="",
                    help="v11.1: a copy of the save whose middle blocks start the deep stage (default: the parent)")
    ap.add_argument("--deep-lr-mult", type=float, default=3.0, help="v11.1: learning-rate multiplier for the deep stage")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="v11.1: multiply EVERY noise level by k (training draws, ghost/wreck noise, dream start noise). "
                         "Simple Diffusion (2023): 2x the side length needs ~2x the noise to hide the same picture")
    ap.add_argument("--noise-scale-low", type=float, default=None,
                    help="v11.1: shift of the low-noise DETAIL bump (--sigma-low-*) only (default: = --noise-scale); "
                         "1 keeps v11's fine-detail practice (noise x2 sprints came out softer)")
    ap.add_argument("--fresh-opt", action="store_true", help="start Adam fresh even if the parent's could be carried over")
    # batch / optimiser
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--nonfwd-weight", type=float, default=2.0)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--source-weights", default="")
    ap.add_argument("--city-balance", type=float, default=0.3)
    ap.add_argument("--p-mean", type=float, default=-0.4)
    ap.add_argument("--p-std", type=float, default=1.2)
    ap.add_argument("--sigma-low-frac", type=float, default=0.2)
    ap.add_argument("--sigma-low-mean", type=float, default=-1.8)
    ap.add_argument("--sigma-low-std", type=float, default=0.8)
    ap.add_argument("--sf-frac", type=float, default=0.10,
                    help="share of rows on v8's ghost-start self-forcing (3-step dreams of the newest 1-4 frames); v8: 0.30")
    ap.add_argument("--sf-weights", default="0.4,0.3,0.2,0.1")
    ap.add_argument("--sf-steps", type=int, default=3)
    ap.add_argument("--sf-start", choices=["noise", "prev"], default="prev")
    ap.add_argument("--sf-walk-only", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--cn-mean", type=float, default=-1.8)
    ap.add_argument("--cn-std", type=float, default=1.1)
    ap.add_argument("--cn-max", type=float, default=1.5)
    ap.add_argument("--cn-zero", type=float, default=0.1)
    ap.add_argument("--cn-perframe", type=float, default=0.5)
    ap.add_argument("--sample-every-min", type=float, default=20)
    ap.add_argument("--first-sample-step", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=20)
    ap.add_argument("--keep-every-hours", type=float, default=1)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--ruler-seeds", type=int, default=2)
    ap.add_argument("--ruler-starts", default=V5_RULER_STARTS)
    ap.add_argument("--turn-steps", type=int, default=TURN_STEPS)
    ap.add_argument("--turn-walk", type=int, default=TURN_WALK)
    ap.add_argument("--decay-hours", type=float, default=8)
    ap.add_argument("--lr-min-frac", type=float, default=0.1)
    ap.add_argument("--no-baseline", action="store_true", help="skip scoring the warm start (= v8) at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0)
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--ruler-secs", type=int, default=40)
    args = ap.parse_args()
    if args.noise_scale != 1.0:          # v11.1: shift every noise level by k (log-normal centres move by ln k)
        k = args.noise_scale
        args.p_mean += math.log(k)
        args.sigma_low_mean += math.log(k if args.noise_scale_low is None else args.noise_scale_low)
        args.ghost_sigma_min *= k
        args.ghost_sigma_max *= k
        args.resample_sigma_median *= k
        args.resample_sigma_min *= k
        args.resample_sigma_max *= k
        args.mem_smax *= k
    if args.eval_mem:
        return eval_main_v11(args)
    if args.wreck_picture:
        if not args.init_v8:
            sys.exit("--wreck-picture needs --init-v8 <a copy of the save to wreck with>")
        return wreck_picture_main(args)

    t_proc = time.time()                     # v11: --budget-hours counts from here
    base = HERE / "_smoke" / "v11" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v11"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "v11" / "live" if args.smoke else HERE / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:                 # a hard cap: PyTorch fails with out-of-memory instead of spilling
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    if not args.allow_low_vram:              # v11: checked with a cap too (a spill into system RAM froze the PC once)
        free = gpu_free_gb()
        need = 1.6 * args.batch + 3.0        # v11: 4x the pixels of v9 (v9: 0.40 x batch + 3)
        if args.vram_cap_gb > 0:
            need = min(need, args.vram_cap_gb)
        if free is not None and free < need + 2.5:
            sys.exit(f"Only {free:.1f} GB of GPU memory is free, and batch {args.batch} needs about {need:.0f} GB.\n"
                     f"Is another training run (or a play window) still running? Stop it first (Ctrl+C in its window; "
                     f"it saves), or use a smaller --batch, or --allow-low-vram to start anyway.")
    torch.manual_seed(args.seed)

    latest = ckpt_dir / "latest.pt"
    ck = None
    if args.resume:
        if not latest.exists():
            sys.exit(f"--resume: no checkpoint at {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
    elif latest.exists():
        sys.exit(f"{latest} already exists. Use --resume to continue it, or pick another --run name.")

    index, frames_mm, actions_np, cut_np = load_cache_any(cache_dir)
    motion_mm, minfo = load_motion(cache_dir)
    # v11: the 96x168 real frames every check-up number is measured against (default: cache_v11 shrunk)
    frames_ref = ref_frames(args.ref_cache, frames_mm, index, actions_np)
    if ck and ck.get("motion_info") and ck["motion_info"]["scale"] != minfo["scale"]:
        sys.exit("motion.npy was rebuilt with different scales since this run started. Rebuild it as before, or start a new run.")
    frame_hw = list(index["shape"][1:3])
    parent_ck = None
    if ck:
        model_cfg = ck["model_cfg"]
    else:
        if not args.init_v8:
            sys.exit("A new v11 run needs --init-v9 <a copy of a v9 save>.")
        parent_ck = torch.load(args.init_v8, map_location="cpu", weights_only=False)
        model_cfg = v11_cfg(parent_ck["model_cfg"], frame_hw)          # same weights, dreaming at the cache's size
        if args.deep_ctx:
            model_cfg["deep_ctx"] = True                               # v11.1: model_v7.build_model_v7 -> model_v11
        model_cfg["noise_scale"] = args.noise_scale                    # play / judges start dreams from 5 * k static
        if parent_ck.get("motion_info") and parent_ck["motion_info"]["scale"] != minfo["scale"]:
            sys.exit("the v8 save was trained with other motion scales than data/cache_v3/motion.npy has now")
    if model_cfg.get("noise_scale", 1.0) != args.noise_scale:
        sys.exit(f"this run trains with --noise-scale {model_cfg.get('noise_scale', 1.0)}; pass the same value to resume it")
    if list(model_cfg["frame_hw"]) != frame_hw:
        sys.exit(f"the model dreams {model_cfg['frame_hw']} frames but {cache_dir} holds {frame_hw}")
    model = build_model_v7(model_cfg).to(device)
    K, Hn = model.K, model.history
    if not 0 <= args.resample_n <= K:
        sys.exit(f"--resample-n must be 0..{K} (the dense recent slots, offsets {K}..1)")
    if args.sf_frac + args.resample_frac > 1:
        sys.exit("--sf-frac + --resample-frac must be <= 1")
    sf_w = [float(x) for x in args.sf_weights.split(",")]
    SF_MAX, L, fp_train, positions, mem_positions = window_layout(model, sf_w, args.resample_n)
    lut = position_lookup(positions, L)
    ruler_script = ruler_key_list(args.ruler_secs)
    T = sum(n for _, n in ruler_script)
    ruler_keys = [a for a, n in ruler_script for _ in range(n)]
    table = key_motion_table(minfo, args.test_turn_deg)
    yaw_scale = float(minfo["scale"][0])
    mscale_t = torch.tensor(minfo["scale"], dtype=torch.float32, device=device)
    fwd_w = float(table[1][2] * minfo["scale"][2])
    lo_s, hi_s = [float(x) for x in args.mem_sigma_low.split(",")]

    valid = compute_valid(actions_np, cut_np, index["cities"], L)
    train_ok = valid.copy()
    held_ok = np.zeros_like(valid)
    for c in index["cities"]:
        n = c["frames"]
        hold_n = min(max(int(args.holdout * n), Hn + T + 40), n // 3)
        h0 = c["end"] - hold_n
        train_ok[max(c["start"], h0 - L):c["end"]] = False
        held_ok[h0:c["end"]] = valid[h0:c["end"]]
    if train_ok.sum() == 0:
        sys.exit("No valid training windows.")

    starts = list(ck["ruler_starts"]) if ck else [int(x) for x in args.ruler_starts.split(",") if x.strip()]
    ruler = [next(c for c in index["cities"] if c["start"] <= s < c["end"]) for s in starts]
    for s, c in zip(starts, ruler):
        if s - Hn < c["start"] or s + T > c["end"]:
            sys.exit(f"ruler start {s} ({c['name']}) has no room for {Hn} frames of history + {T} frames of walk")
    real_clip = np.array(frames_mm[starts[0]:starts[0] + T])
    scene_sd = np.load(SCENE_FIT)["sd"]
    real_ref = real_collage_ref(starts, frames_ref, scene_sd)          # v11: measured at 96x168

    # ---- sampling weights: exactly v7's / v8's
    src_w = parse_source_weights(args.source_weights)
    usable = {c["name"]: int(train_ok[c["start"]:c["end"]].sum()) for c in index["cities"]}
    shares = source_shares(index["cities"], usable, src_w, args.city_balance)
    N = len(actions_np)
    target_key = np.full(N, -1, np.int64)
    target_key[:N - L + 1] = actions_np[L - 1:]
    yaw_deg = np.abs(np.asarray(motion_mm[:, 0], np.float32)) * yaw_scale
    tyaw = np.zeros(N, np.float32)
    tyaw[:N - L + 1] = yaw_deg[L - 1:]
    prev4 = np.zeros(N, np.float32)
    for j in range(2, 6):
        prev4[:N - L + 1] += yaw_deg[L - j:N - j + 1] / 4
    onset = (tyaw >= args.onset_deg) & (prev4 <= args.onset_ratio * tyaw)
    bigturn = tyaw >= args.bigturn_deg

    def per_city_norm(w):
        w = w.copy()
        for c in index["cities"]:
            seg = w[c["start"]:c["end"]]
            if seg.sum() > 0:
                seg *= shares[c["name"]] / seg.sum()
        return w
    boost = np.where(onset, args.onset_weight, 1.0) * np.where(bigturn, args.bigturn_weight, 1.0)
    weights = per_city_norm(train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight) * boost)
    del yaw_deg, tyaw, prev4

    # ---- model, EMA, optimizer (+ the warm start: a straight copy, v9 = v8 = v7's network)
    n_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    def make_opt(m):
        decay, no_decay, deep_d, deep_nd = [], [], [], []
        for name, p in m.named_parameters():
            nd = p.ndim < 2 or "emb" in name
            if name.startswith("unet.deep."):                         # v11.1's new stage: its own (faster) groups
                (deep_nd if nd else deep_d).append(p)
            else:
                (no_decay if nd else decay).append(p)
        groups = [{"params": decay, "weight_decay": args.wd, "mult": 1.0},
                  {"params": no_decay, "weight_decay": 0.0, "mult": 1.0}]
        if deep_d or deep_nd:
            groups += [{"params": deep_d, "weight_decay": args.wd, "mult": args.deep_lr_mult},
                       {"params": deep_nd, "weight_decay": 0.0, "mult": args.deep_lr_mult}]
        return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), fused=True)
    opt = make_opt(model)
    init_notes = []
    if parent_ck is not None:
        new_keys = set(model.state_dict()) - set(parent_ck["model"])
        if new_keys and not all(k_.startswith("unet.deep.") for k_ in new_keys):
            sys.exit(f"the parent save is missing more than the deep stage: {sorted(new_keys)[:8]}")
        for m_, sd_ in ((model, parent_ck["model"]), (ema, parent_ck["ema"])):
            missing, unexpected = m_.load_state_dict(sd_, strict=False)
            if unexpected or set(missing) != new_keys:
                sys.exit(f"warm start does not fit: missing {missing[:8]}, unexpected {unexpected[:8]}")
        init_notes.append(f"from {args.init_v8}: {parent_ck['model_cfg'].get('trained_as', 'v7')} run {parent_ck['run_id']} "
                          f"step {parent_ck['step']} ({n_params / 1e6:.3f}M params): raw + EMA weights copied unchanged")
        if new_keys:
            donor = parent_ck
            if args.deep_donor:
                donor = torch.load(args.deep_donor, map_location="cpu", weights_only=False)
            n_raw = init_deep_from_mid(model, donor["model"])
            n_ema = init_deep_from_mid(ema, donor["ema"])
            n_new = sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("unet.deep."))
            init_notes.append(f"v11.1 deep context stage: {len(new_keys)} new tensors ({n_new / 1e6:.2f}M params); "
                              f"{n_raw} raw / {n_ema} EMA tensors copied from the middle blocks of "
                              f"{args.deep_donor or 'the parent'}; its output conv starts at ZERO (= the parent's dreams); "
                              f"learning rate x{args.deep_lr_mult}")
            if donor is not parent_ck:
                del donor
        try:
            if args.fresh_opt:
                raise RuntimeError("--fresh-opt")
            opt.load_state_dict(parent_ck["opt"])
            init_notes.append("Adam memory carried over from the parent (same weights, same groups)")
        except Exception as e:
            opt = make_opt(model)
            init_notes.append("Adam starts fresh (--fresh-opt)" if args.fresh_opt else
                              "could not carry the parent's Adam memory over, starting it fresh:\n" + traceback.format_exc())
    if args.noise_scale != 1.0:
        init_notes.append(f"v11.1 noise x{args.noise_scale}: training noise centre exp({args.p_mean:.2f}) (low bump "
                          f"exp({args.sigma_low_mean:.2f})), ghost noise {args.ghost_sigma_min:g}-{args.ghost_sigma_max:g}, "
                          f"wreck noise median {args.resample_sigma_median:g} ({args.resample_sigma_min:g}-"
                          f"{args.resample_sigma_max:g}), dreams start from {5.0 * args.noise_scale:g} static "
                          f"({args.mem_smax:g} on remembered turns)")

    step, train_seconds = 0, 0.0
    run_id = f"{args.tag}-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline, best, best_walk, warm_check = None, None, None, None
    best_melt = None          # lowest honest melt-critic score (2026-10-22): best_melt.pt
    if ck:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline, best, best_walk, warm_check = ck.get("parent_baseline"), ck.get("best"), ck.get("best_walk"), ck.get("warm_check")
        best_melt = ck.get("best_melt")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    names = ", ".join(c["name"] for c in index["cities"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": f"dream model {args.tag}",
        "prompt": f"dream {args.tag} ({names}), {n_params / 1e6:.0f}M params, {frame_hw[0]}x{frame_hw[1]}, "
                  f"memory {model.n_mem} frames back to {Hn / 10:.1f} s, steered by keys + camera motion, "
                  f"practising recovery from its own dreamed memory",
        "pose": pose_string(ruler_script), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "melt_median": None, "melt_mean": None, "clean2_median": None, "scene": None,
        "return_sim": None, "circle_return_sim": None, "back_scene": None, "back_seam": None,
        "nomem_clean2_median": None, "combined_score": None,
        "best_step": best["step"] if best else None, "best_walk_step": best_walk["step"] if best_walk else None})
    if args.log_copy:                        # v11: every run.log line also goes to this file (train/_exp/v11/...)
        pathlib.Path(args.log_copy).parent.mkdir(parents=True, exist_ok=True)
        _log1 = live.log

        def _log2(msg, _o=_log1):
            _o(msg)
            try:
                with open(args.log_copy, "a", encoding="utf-8") as f_:
                    f_.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}" + "\n")
            except Exception:
                pass
        live.log = _log2
        live.log(f"run.log is also copied to {args.log_copy}; live folder {live.dir}")
    counts = sf_counts(int(round(args.sf_frac * args.batch)), sf_w)
    n_sf = sum(counts)
    n_rs = min(int(round(args.resample_frac * args.batch)), args.batch - n_sf) if args.resample_n > 0 else 0
    n_old = sum(1 for o in model.mem_offsets() if o > args.resample_n)
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v11 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.3f}M params; {frame_hw[0]}x{frame_hw[1]}; memory {model.n_mem} frames ({Hn / 10:.1f} s); "
             f"window {L + 1} frames, {len(positions)} read; batch {args.batch}; {int(train_ok.sum())} train windows")
    zq = (torch.randn(100000) * args.resample_sigma_std + math.log(args.resample_sigma_median)).exp().clamp(
        args.resample_sigma_min, args.resample_sigma_max)
    live.log(f"v9 A: {n_rs} of {args.batch} rows/step ({args.resample_frac:.0%}) get their newest {args.resample_n} memory slots "
             f"WRECKED (real + noise sigma_s -> one EMA call, oldest first, label 0); sigma_s median {args.resample_sigma_median}, "
             f"log-std {args.resample_sigma_std}, clipped {args.resample_sigma_min}-{args.resample_sigma_max} (10/50/90%: "
             f"{zq.quantile(0.1):.3f}/{zq.quantile(0.5):.3f}/{zq.quantile(0.9):.3f}). {n_sf} rows/step ({args.sf_frac:.0%}) on "
             f"v8's ghost-start self-forcing (groups {counts}, {args.sf_steps} steps).")
    live.log(f"v9 B: error bank {'OFF' if args.err_bank_frac <= 0 else 'ON'}: {args.err_bank_frac:.0%} of rows, each of the "
             f"{n_old} older slots (offset > {args.resample_n}) with p {args.err_bank_p}: a banked x0 error (D - target) x "
             f"U{tuple(args.err_bank_scale)}, {1 - BANK_WIDE_P:.0%} from sigma {args.err_bank_draw[0]}-{args.err_bank_draw[1]}; "
             f"bank {args.err_bank_bins} bins over {args.err_bank_range[0]}-{args.err_bank_range[1]} x {args.err_bank_size}")
    live.log(f"memory-collage ghosts: {args.mem_ghost_frac:.0%} of rows ({args.mem_patch_p:.0%} of them synthetic-revisit "
             f"patchwork: newest 0..{args.mem_gap_max} frames left out, oldest {args.mem_age_min}..{args.mem_age_max} steps, "
             f"jitter {args.mem_jitter}); noise {lo_s}-{hi_s} for {args.mem_low_p:.0%} of them, else "
             f"{args.ghost_sigma_min}-{args.ghost_sigma_max}. v7 shifted-ghost rows {args.ghost_frac:.0%}. "
             f"Tests dream WITH panorama memory (no fade turning in place, start noise {args.mem_smax} on remembered turns).")
    live.log(f"real frames' collage numbers (the target for turned-back frames): {real_ref}")
    live.log(f"key -> motion table (turn = {args.test_turn_deg} deg/step): " + describe_table(minfo, args.test_turn_deg))
    for n_ in init_notes:
        live.log("warm start: " + n_)
    live.update(stage="starting")

    # ---- loader
    ds = SparseWindowDatasetV7(cache_dir, positions, L, flip_p=args.flip_p)
    walking_frame = torch.zeros(N, dtype=torch.bool)
    for c in index["cities"]:
        if c["name"] not in NON_WALKING:
            walking_frame[c["start"]:c["end"]] = True
    sampler = InfiniteWeightedSampler(weights, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch if args.workers > 0 else None)
    mi = MemIndexV7(model, lut, device)
    mi_ema = MemIndexV7(ema, lut, device)
    fp_rows = lut[torch.tensor([L - o for o in fp_train])].to(device)
    target_row = int(lut[L])
    prev_row = int(lut[L - 1])
    fpx_train = (frame_hw[1] / 2) / math.tan(math.radians(35))
    bank = None
    if args.err_bank_frac > 0 and n_old > 0:
        bank = ErrorBank(args.err_bank_bins, args.err_bank_size, (3, frame_hw[0], frame_hw[1]),
                         args.err_bank_range[0], args.err_bank_range[1], device)
        live.log(f"error bank: {bank.gb():.2f} GB on the GPU")

    held_idx = np.flatnonzero(held_ok)
    rs = np.random.default_rng(0)
    pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
    val_ds = SparseWindowDatasetV7(cache_dir, positions, L, flip_p=0.0)
    vb = [val_ds[int(t)] for t in pick]
    val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]), torch.stack([v[3] for v in vb]))

    def run_val():
        tot, sig_list = 0.0, [0.05, 0.2, 0.5, 1.0, 2.0]
        B = val_batch[0].shape[0]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, B, 8):
                    fr = to_model_space(val_batch[0][i:i + 8].to(device))
                    act = val_batch[1][i:i + 8].to(device)
                    mot = val_batch[2][i:i + 8].to(device)
                    b = fr.shape[0]
                    ctx, a, ext = mi_ema(fr, act, mot, L)
                    col = colour_fingerprint(fr.index_select(1, fp_rows))
                    for s in sig_list:
                        tot += ema.loss(fr[:, target_row], ctx, a, torch.full((b,), s, device=device), 0.05,
                                        col, ext).item() * b / B
        return tot / len(sig_list)

    def save_ckpt(path):
        try:
            tmp = str(path) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                        "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg,
                        "motion_info": minfo, "parent_baseline": baseline, "best": best, "best_walk": best_walk, "best_melt": best_melt,
                        "warm_check": warm_check, "ruler_cities": [c["name"] for c in ruler], "ruler_starts": starts,
                        "warm_started": True}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def suite(label):
        t_suite = time.time()
        res, dreams = memory_suite_v11(ema, frames_mm, frames_ref, actions_np, motion_mm, starts, args.ruler_seeds, table,
                                       minfo["scale"], scene_sd, ruler_keys, args.rollout_steps, args.mem_smax,
                                       args.turn_walk, args.turn_steps, log=live.log, write_rule=args.mem_write,
                                       smax=5.0 * args.noise_scale)
        # HONEST meters (added 2026-10-22): the old clean2 / melt meters were fooled by colour
        # glitch and trip on real video 20-40% of the time. melt_critic.py (lower = better, real video ~0.13) and
        # glitch_meter.py (seconds until colour-bar glitch) on the same 40 s dreams; best_melt.pt follows the critic.
        try:
            mc = honest_critic()
            rows_ = [s_ for _ in range(args.ruler_seeds) for s_ in starts]
            parts = []
            # v11: dreams shrunk to 96x168 (memory_suite_v11), start frames shrunk the same way (frames_ref)
            for key, v in honest_numbers(mc, dreams, rows_, frames_ref, T).items():
                if key in res:
                    res[key].update(v)
                    parts.append(f"{'with' if key == 'ruler_mem' else 'without'} memory: melt critic {v['melt_critic']}, "
                                 f"glitch after {v['glitch_time_mean']} s")
            live.log(f"{label}: HONEST METERS at 96x168 (lower melt = better; real video ~0.13; v9 pick ~0.44 is the 10-walk judge, not these 5 walks) " + "; ".join(parts))
        except Exception:
            live.log("honest meters FAILED (training continues):\n" + traceback.format_exc())
        pc = baseline["ruler_nomem"]["clean2_median"] if baseline else None
        res["combined_score"] = combined_score(res, T / 10, real_ref["scene"], pc)
        live.log(f"{label}: COMBINED SCORE {res['combined_score']} = clean2 with memory {res['ruler_mem']['clean2_median']} / "
                 f"{T / 10:.0f} s x return {res['turn']['return_sim']} / 0.85 x real scene {real_ref['scene']} / turned-back "
                 f"scene {res['back']['scene']} x detail SSIM (back {res['back']['ssim_pre']} + full circle {res['circle_end']['ssim_pre']}) / 2"
                 + (f" x no-memory clean2 {res['ruler_nomem']['clean2_median']} / (0.75 x parent {pc})" if pc else "")
                 + (f"   [parent: {baseline.get('combined_score')}]" if baseline else ""))
        checkup_secs[0] = time.time() - t_suite
        live.log(f"{label}: check-up took {checkup_secs[0] / 60:.1f} min")
        return res, dreams

    critic_box = []
    checkup_secs = [args.checkup_reserve_min * 60]

    def honest_critic():
        if not critic_box:
            from melt_critic import MeltCritic
            critic_box.append(MeltCritic(device))
        return critic_box[0]

    def do_rollout():
        nonlocal best, best_walk, best_melt
        try:
            t0 = time.time()
            torch.cuda.empty_cache()             # v11: hand training's cached blocks back first (2x check-ups are big)
            res, dreams = suite(f"{args.tag} step {step}")
            d3 = dreams["ruler_mem_2x"]                       # v11: show the model's own 2x frames
            write_live_frames(live, d3[0], ruler_keys)
            path = write_mp4(out_dir / f"{args.tag}_rollout_step{step:07d}.mp4", d3[0], ruler_keys, real_clip,
                             f"{args.tag} step {step} (panorama memory)", dreams["ruler_mem_sharp"])
            turn_picture(out_dir / f"{args.tag}_turn_step{step:07d}.jpg", dreams["turn_2x"], dreams["ctrl_2x"], len(starts),
                         f"{args.tag} {step} memory", args.turn_walk, args.turn_steps)
            turn_clip(out_dir / f"{args.tag}_turnclip_step{step:07d}.mp4", dreams["turn_2x"][0],
                      np.array(frames_mm[starts[0]:starts[0] + 1]), f"{args.tag} {step} memory")
            v = run_val()
            live.log(f"rollout at step {step} in {time.time() - t0:.0f}s -> {path.name}; val loss {v:.4f}")
            sc = res["combined_score"]
            is_best = best is None or sc > best["combined_score"]
            if is_best:
                best = dict(step=step, val=v, **{k_: res[k_] for k_ in ("combined_score", "turn", "back", "ruler_mem",
                                                                        "ruler_nomem", "circle_return_sim")})
                save_ckpt(ckpt_dir / "best.pt")
                live.log(f"NEW BEST combined score {sc} -> best.pt")
            nm = res["ruler_nomem"]
            if best_walk is None or (nm["clean2_median"], nm["clean2_mean"]) > (best_walk["clean2_median"], best_walk["clean2_mean"]):
                best_walk = dict(step=step, **nm)
                save_ckpt(ckpt_dir / "best_walk.pt")
                live.log(f"NEW BEST no-memory clean2 {nm['clean2_median']} -> best_walk.pt")
            mcs = res["ruler_mem"].get("melt_critic")
            if mcs is not None and res["turn"]["return_sim"] >= 0.9 and (best_melt is None or mcs < best_melt["melt_critic"]):
                best_melt = dict(step=step, melt_critic=mcs, glitch_time_mean=res["ruler_mem"].get("glitch_time_mean"),
                                 return_sim=res["turn"]["return_sim"], circle_return_sim=res["circle_return_sim"])
                save_ckpt(ckpt_dir / "best_melt.pt")
                live.log(f"NEW BEST honest melt score {mcs} -> best_melt.pt (the one to play; best.pt follows the old score)")
            live.metric({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name, "is_best": is_best, **res})
            rm = res["ruler_mem"]
            live.update(last_rollout_step=step, val_loss=v, melt_s=rm["melt_median"], melt_median=rm["melt_median"],
                        clean2_median=rm["clean2_median"], clean2_mean=rm["clean2_mean"], scene=rm["scene"],
                        return_sim=res["turn"]["return_sim"], circle_return_sim=res["circle_return_sim"],
                        back_scene=res["back"]["scene"], back_seam=res["back"]["seam"],
                        nomem_clean2_median=nm["clean2_median"], combined_score=sc,
                        best_step=best["step"], best_walk_step=best_walk["step"])
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    # ---- start-up checks on a NEW warm-started run: v9 at init must BE the parent (v8)
    if parent_ck is not None:
        live.update(stage="warm-start check")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # v11: the reference is the PARENT's weights run at the same 2x size (the network has no size-bound parts)
        ref = build_model_v7(dict(parent_ck["model_cfg"], frame_hw=list(frame_hw))).to(device).eval()
        ref.load_state_dict(parent_ck["ema"])
        # compared BY NAME (v11.1's deep stage adds tensors the parent does not have)
        ema_sd, raw_sd = ema.state_dict(), model.state_dict()
        wdiff = max((ema_sd[k_] - q.to(device)).abs().max().item() for k_, q in ref.state_dict().items())
        rawdiff = max((raw_sd[k_].detach() - q.to(device)).abs().max().item() for k_, q in parent_ck["model"].items())
        del ema_sd, raw_sd
        with torch.no_grad():
            fr = to_model_space(val_batch[0][:16].to(device))
            act = val_batch[1][:16].to(device)
            mot = val_batch[2][:16].to(device)
            B = fr.shape[0]
            ctx, a, ext = mi_ema(fr, act, mot, L)
            col = colour_fingerprint(fr.index_select(1, fp_rows))
            tgt = fr[:, target_row]
            res = {"weights_maxabs": wdiff, "raw_weights_maxabs": rawdiff}
            for s in [0.1, 0.5, 2.0]:
                g = torch.Generator(device=device).manual_seed(0)
                xn = tgt + s * torch.randn(tgt.shape, device=device, generator=g)
                sig = torch.full((B,), s, device=device)
                res[f"denoise_sigma{s}_maxabs"] = (ema(xn, sig, ctx, 0.0, a, col, ext)[0] - ref(xn, sig, ctx, 0.0, a, col, ext)[0]).abs().max().item()
            p9 = ema.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=ext, start="prev",
                            generator=torch.Generator(device=device).manual_seed(1))
            p8 = ref.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=ext, start="prev",
                            generator=torch.Generator(device=device).manual_seed(1))
            res["dream_prev_start_maxabs"] = (p9 - p8).abs().max().item()
            # the v9 loss function (loss_ghost_d) must be the v7/v8 loss (model.loss_ghost), same random draws
            sigma = torch.full((B,), 0.5, device=device)
            ghost = fr[:, prev_row].float()
            use_g = torch.arange(B, device=device) % 3 == 0
            torch.manual_seed(5)
            l7 = ema.loss_ghost(tgt, ctx, a, sigma, 0.1, col, ext, ghost, use_g)
            torch.manual_seed(5)
            l9, _ = loss_ghost_d(ema, tgt, ctx, a, sigma, 0.1, col, ext, ghost, use_g)
            res["loss_fn_absdiff"] = abs(l7.item() - l9.item())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        del ref
        warm_check = res
        ok = (res["weights_maxabs"] == 0 and res["raw_weights_maxabs"] == 0 and res["dream_prev_start_maxabs"] == 0
              and res["loss_fn_absdiff"] == 0)
        live.log(f"WARM-START CHECK (v11 at init vs its parent's weights run at {frame_hw[0]}x{frame_hw[1]}, float32, 16 held-out "
                 f"moments): EMA weights max diff {wdiff:.1e} "
                 f"(raw {rawdiff:.1e}); one clean-up call at noise 0.1/0.5/2.0: {res['denoise_sigma0.1_maxabs']:.1e}/"
                 f"{res['denoise_sigma0.5_maxabs']:.1e}/{res['denoise_sigma2.0_maxabs']:.1e}; a full {args.rollout_steps}-step "
                 f"dream: {res['dream_prev_start_maxabs']:.1e}; v9 loss fn vs v8's: {res['loss_fn_absdiff']:.1e} "
                 f"-> {'OK: v11 starts out as its parent (at 2x)' if ok else 'WARNING: v11 differs from its parent'}")
        live.metric({"step": step, "time": time.time(), "warm_check": res})
        torch.cuda.empty_cache()
        if not args.no_baseline:
            live.update(stage="v8 on the memory suite")
            res, _ = suite(f"parent BASELINE (= v9 at init; v8 step {parent_ck['step']})")
            baseline = dict(res, parent_step=parent_ck["step"])
            baseline["combined_score"] = combined_score(res, T / 10, real_ref["scene"], res["ruler_nomem"]["clean2_median"])
            live.metric({"step": step, "time": time.time(), "parent_baseline": baseline})
            torch.cuda.empty_cache()
    del parent_ck

    # ---- the loop
    graphed = None
    if args.no_graphs:
        sf_sampler = eager_sample(ema, start=args.sf_start)
    else:
        graphed = GraphedSampler(ema, start=args.sf_start)
        sf_sampler = graphed.sample
        try:
            fr = to_model_space(val_batch[0][:max(1, counts[0])].to(device))
            act = val_batch[1][:fr.shape[0]].to(device)
            mot = val_batch[2][:fr.shape[0]].to(device)
            ctx, a, ext = mi_ema(fr, act, mot, L)
            diff = graphed.check(ctx, a, colour_fingerprint(fr.index_select(1, fp_rows)), ext)
            graphed.graphs.clear()
            msg = f"CUDA-graph check (self-forcing): {diff:.2e}"
            if n_rs:
                nb = min(n_rs, val_batch[0].shape[0])
                fr = to_model_space(val_batch[0][:nb].to(device))
                act = val_batch[1][:nb].to(device)
                mot = val_batch[2][:nb].to(device)
                ctx, a, ext = mi_ema(fr, act, mot, L - 1)
                col = colour_fingerprint(fr.index_select(1, fp_rows))
                sig = torch.linspace(0.03, 3.0, nb, device=device)
                x = fr[:, prev_row] + sig[:, None, None, None] * torch.randn_like(fr[:, prev_row])
                d1 = OneCall(ema, graphed)(x, sig, ctx, a, col, ext)
                d2 = OneCall(ema, None)(x, sig, ctx, a, col, ext)
                diff2 = (d1 - d2).abs().max().item()
                graphed.graphs.clear()
                msg += f"; (wreck call, per-row sigma): {diff2:.2e}"
                diff = max(diff, diff2)
            live.log(msg + f" ({'OK' if diff < 2e-2 else 'WARNING: large; consider --no-graphs'})")
        except Exception:
            live.log("CUDA graphs FAILED, falling back to normal calls:\n" + traceback.format_exc())
            sf_sampler = eager_sample(ema, start=args.sf_start)
            graphed = None
    if graphed is not None:
        # Record EVERY batch size the loop will need NOW, before the loader starts: a recording made while the
        # loader's pin-memory thread allocates pinned buffers gets invalidated ("operation failed due to a
        # previous error during capture", seen 2026-10-21 in the first v9 smoke test at the first SF call).
        sizes = {sum(counts[j - 1:]) for j in range(1, len(counts) + 1)} | {n_rs}
        sizes = sorted(s_ for s_ in sizes if 0 < s_ <= val_batch[0].shape[0])
        try:
            for nb in sizes:
                fr = to_model_space(val_batch[0][:nb].to(device))
                act = val_batch[1][:nb].to(device)
                mot = val_batch[2][:nb].to(device)
                ctx, a, ext = mi_ema(fr, act, mot, L - 1)
                graphed.graphs[nb] = graphed._build(ctx, a, colour_fingerprint(fr.index_select(1, fp_rows)), ext)
            torch.cuda.synchronize()
            live.log(f"CUDA graphs recorded up front for batch sizes {sizes} (self-forcing groups {counts}, wreck rows {n_rs})")
        except Exception:
            live.log("CUDA graph recording FAILED, falling back to normal calls:\n" + traceback.format_exc())
            sf_sampler = eager_sample(ema, start=args.sf_start)
            graphed = None
    if args.noise_scale != 1.0:          # v11.1: the quick self-forcing dreams start from 5k static too
        smax_sf = 5.0 * args.noise_scale
        if graphed is not None:
            sf_sampler = functools.partial(graphed.sample, sigma_max=smax_sf)
        else:
            def sf_sampler(ctx, a, steps, colour, far):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return ema.sample(ctx, a, steps=steps, colour=colour, far_act=far, start=args.sf_start,
                                      sigma_max=smax_sf).float()
    wreck_call = OneCall(ema, graphed)
    torch.cuda.reset_peak_memory_stats()
    now = time.time()
    last_status, last_ckpt, last_keep = 0.0, now, now
    next_sample = None if step < args.first_sample_step else now + args.sample_every_min * 60
    sample_soon = step < args.first_sample_step
    loss_avg = None
    win_t, win_steps, it_s = time.time(), 0, 0.0
    bad_steps = 0
    t_loop = time.time()
    n_seen = n_ghost = n_mem = n_patch = 0
    n_sf_rows = n_rs_rows = n_bank_rows = n_bank_slots = n_banked = 0
    mem_share_sum = 0.0
    rs_logsig = torch.zeros((), device=device)
    rs_change = torch.zeros((), device=device)
    deadline = end_deadline(args.end_at) if args.end_at else None
    if args.budget_hours > 0:
        deadline = t_proc + args.budget_hours * 3600 if deadline is None else min(deadline, t_proc + args.budget_hours * 3600)
    if deadline is not None:
        live.log(f"time budget: training stops around {time.strftime('%H:%M', time.localtime(deadline - checkup_secs[0] - 180))} "
                 f"(a check-up takes ~{checkup_secs[0] / 60:.0f} min, re-measured at each one) so the last check-up + save "
                 f"end by {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))}")
    live.update(stage="training")

    try:
        it = iter(loader)
        while True:
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_hours and (time.time() - t_loop) / 3600 >= args.max_hours:
                break
            if deadline is not None and time.time() + checkup_secs[0] + 180 >= deadline:
                live.log(f"time budget reached at step {step}: last check-up + save now, to end by {args.end_at}")
                break
            frames_u8, act, tix, mot = next(it)
            fr = to_model_space(frames_u8.to(device, non_blocking=True))
            act = act.to(device, non_blocking=True)
            mot = mot.to(device, non_blocking=True)
            B = fr.shape[0]
            if args.sf_walk_only:
                order = torch.argsort((~walking_frame[tix]).to(torch.int8), stable=True).to(device)
                fr, act, mot = fr.index_select(0, order), act.index_select(0, order), mot.index_select(0, order)
            colour = colour_fingerprint(fr.index_select(1, fp_rows))
            colour = colour * (torch.rand(B, 1, device=device) >= args.colour_drop)
            # rows [0, n_sf): v8's ghost-start self-forcing
            if n_sf:
                fr = self_force_v7(sf_sampler, mi_ema, fr, act, mot, colour, counts, args.sf_steps, L)
                n_sf_rows += n_sf
            # rows [n_sf, n_sf + n_rs): v9 A, the newest slots wrecked by one EMA call each
            if n_rs:
                sig_s = draw_wreck_sigmas(n_rs, args, device)
                before = fr[n_sf:n_sf + n_rs, prev_row].clone()
                fr = resample_memory(wreck_call, mi_ema, fr, act, mot, colour, n_sf, n_sf + n_rs, args.resample_n, sig_s, L)
                rs_logsig += sig_s.log().sum()
                rs_change += (fr[n_sf:n_sf + n_rs, prev_row] - before).abs().mean() * 127.5 * n_rs
                n_rs_rows += n_rs
            drop = torch.rand(B, device=device) < args.steer_drop
            context, actions, ext = mi(fr, act, mot, L, absent=drop)
            actions = torch.where(drop[:, None], torch.full_like(actions, NULL_ACTION), actions)
            n_seen += B
            target = fr[:, target_row]
            sigma = sample_sigma(B, args, device)
            sigma_cond = sample_sigma_cond(B, model.n_mem, args, device)
            if n_rs:
                # wrecked slots: memory-noise label 0 and no extra Gaussian noise, as in play
                sigma_cond[n_sf:n_sf + n_rs, model.n_mem - args.resample_n:] = 0.0
            # v9 B: banked errors on the older slots
            if bank is not None:
                brows = torch.rand(B, device=device) < args.err_bank_frac
                r_, s_ = apply_bank(context, bank, brows, n_old, args)
                n_bank_rows += r_
                n_bank_slots += s_
            # v7's shifted-ghost rows
            u = torch.rand(B, device=device)
            use_v7 = u < args.ghost_frac
            use_mem = (u >= args.ghost_frac) & (u < args.ghost_frac + args.mem_ghost_frac)
            lo, hi = math.log(args.ghost_sigma_min), math.log(args.ghost_sigma_max)
            sigma = torch.where(use_v7, torch.exp(lo + (hi - lo) * torch.rand(B, device=device)), sigma)
            ghost = fr[:, prev_row].float()
            px = motion_shift_px(mot[:, L - 1], minfo["scale"], fpx_train) * (torch.rand(B, device=device) < args.ghost_shift_p)[:, None]
            ghost = shift_frames(ghost, px)
            if args.ghost_edge_fill == "noise":
                ghost = noise_fill_edges(ghost, fr[:, prev_row].float(), torch.round(px[:, 0]).long())
            if args.ghost_zoom_k > 0:                 # walking forward makes the world grow: zoom the ghost too
                fz = (mot[:, L - 1, 2].float() * mscale_t[2] / fwd_w).clamp(min=0)
                ghost = zoom_frames(ghost, 1.0 + args.ghost_zoom_k * fz, args.ghost_zoom_cy)
            n_ghost += int(use_v7.sum())
            # v8's memory-collage ghost rows (v8's candidate frames: mem_positions)
            if bool(use_mem.any()):
                sel = torch.nonzero(use_mem).squeeze(1)
                m = len(sel)
                patch = torch.rand(m, device=device) < args.mem_patch_p
                gap = torch.where(patch, torch.randint(0, args.mem_gap_max + 1, (m,), device=device), torch.zeros(m, dtype=torch.long, device=device))
                age = torch.randint(args.mem_age_min, args.mem_age_max + 1, (m,), device=device)
                age = torch.where(patch, age, torch.full_like(age, 10 ** 6))
                mg, msh = mem_ghost(fr.index_select(0, sel), lut, mem_positions, mot.index_select(0, sel), mscale_t, fpx_train,
                                    fwd_w, prev_row, L, gap, patch, jitter=args.mem_jitter, max_age=age)
                ghost = ghost.index_copy(0, sel, mg)
                low = torch.rand(m, device=device) < args.mem_low_p
                ls = math.log(lo_s) + (math.log(hi_s) - math.log(lo_s)) * torch.rand(m, device=device)
                hs = lo + (hi - lo) * torch.rand(m, device=device)
                sigma = sigma.index_copy(0, sel, torch.exp(torch.where(low, ls, hs)))
                n_mem += m
                n_patch += int(patch.sum())
                mem_share_sum += float(msh.sum())
            use_ghost = use_v7 | use_mem

            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            if args.decay_hours > 0:
                prog = min(1.0, train_seconds / 3600.0 / args.decay_hours)
                lr *= args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
            for gq in opt.param_groups:
                gq["lr"] = lr * gq.get("mult", 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, D = loss_ghost_d(model, target, context, actions, sigma, sigma_cond, colour, ext, ghost, use_ghost)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lv = loss.item()
            if not (math.isfinite(lv) and math.isfinite(gn.item())):
                bad_steps += 1
                live.log(f"non-finite loss/grad at step {step}, skipping ({bad_steps} in a row)")
                if bad_steps >= 50:
                    raise RuntimeError("50 non-finite steps in a row: training has diverged")
                continue
            bad_steps = 0
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - args.ema)
            if bank is not None:
                n_banked += bank.add(D - target, sigma)
            del D
            step += 1
            win_steps += 1
            loss_avg = lv if loss_avg is None else 0.98 * loss_avg + 0.02 * lv

            now = time.time()
            if now - win_t >= 5:
                train_seconds += now - win_t
                it_s = win_steps / (now - win_t)
                win_t, win_steps = now, 0
            if step % 50 == 0:
                live.metric({"step": step, "time": now, "loss": round(loss_avg, 5), "lr": lr, "it_s": round(it_s, 2)})
            if sample_soon and step >= args.first_sample_step:
                sample_soon = False
                do_rollout()
                next_sample = time.time() + args.sample_every_min * 60
            elif next_sample is not None and now >= next_sample:
                if deadline is not None and now + 2 * checkup_secs[0] + 180 >= deadline:
                    next_sample = None                # too close to the end: the final check-up covers it
                    live.log("scheduled check-up skipped: the final one at the time budget comes soon after")
                else:
                    do_rollout()
                    next_sample = time.time() + args.sample_every_min * 60
            if step % 100 == 0 or step == 1:
                vram = torch.cuda.max_memory_reserved() / 2**30
                eta = (f"{(next_sample - now) / 60:.1f} min" if next_sample else f"at step {args.first_sample_step}")
                rs_txt = (f"resampled {n_rs_rows} (median-ish sigma_s {math.exp(rs_logsig.item() / n_rs_rows):.2f}, newest slot "
                          f"changed {rs_change.item() / n_rs_rows:.1f}/255)" if n_rs_rows else "resampled 0")
                bk_txt = (f"err-bank {n_bank_rows} rows / {n_bank_slots} slots (bank holds {sum(bank.fill)}/"
                          f"{bank.n * bank.per}, {n_banked} stored)" if bank is not None else "err-bank off")
                live.log(f"step {step}  loss {loss_avg:.4f}  {it_s:.2f} it/s  lr {lr:.2e}  grad {gn.item():.2f}  VRAM {vram:.1f} GB  "
                         f"rows: ghost-SF {n_sf_rows}, {rs_txt}, {bk_txt}, v7 ghost {n_ghost}, memory ghost {n_mem} "
                         f"({n_patch} patchwork, {mem_share_sum / max(1, n_mem):.0%} of their columns remembered) of {n_seen}  "
                         f"next dream {eta}")
            if now - last_status >= 10:
                last_status = now
                live.update(step=step, loss=round(loss_avg, 5), it_s=round(it_s, 2), lr=lr,
                            peak_vram_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            if now - last_ckpt >= args.ckpt_every_min * 60:
                save_ckpt(latest)
                last_ckpt = time.time()
                live.log(f"checkpoint saved (step {step})")
                if now - last_keep >= args.keep_every_hours * 3600:
                    shutil.copyfile(latest, ckpt_dir / f"step{step:07d}.pt")
                    last_keep = time.time()
                    live.log(f"keeper checkpoint step{step:07d}.pt saved")

        save_ckpt(latest)
        if step >= args.first_sample_step and live.status.get("last_rollout_step") != step:
            do_rollout()
            save_ckpt(latest)
        live.log(f"finished at step {step}; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB allocated, "
                 f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved; {train_seconds:.0f} s of training")
        live.update(stage="done", exit=0, step=step, loss=loss_avg)
    except KeyboardInterrupt:
        live.log(f"stopped by Ctrl+C at step {step}; saving checkpoint")
        save_ckpt(latest)
        live.update(stage="done", exit=130, step=step, loss=loss_avg)
    except Exception:
        live.log("CRASHED:\n" + traceback.format_exc())
        try:
            save_ckpt(ckpt_dir / "crash.pt")
            live.log("saved crash.pt (latest.pt left untouched; --resume uses latest.pt)")
        except Exception:
            pass
        live.update(stage="crashed", exit=1, step=step)
        raise


if __name__ == "__main__":
    main()
