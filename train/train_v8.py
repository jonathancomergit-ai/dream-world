"""Train Dream World v8: REBUILD CLEAN VIEWS FROM PANORAMA MEMORY, warm-started from v7 (2026-10-21).

    # the real run (or double-click train_v8.bat, which first copies v7's best.pt):
    python train/train_v8.py --run dream_v8 --init-v7 train/checkpoints/dream_v8_init.pt ...
    python train/train_v8.py --run dream_v8 --resume ...           # carry on after a stop
    # a small test that leaves the real folders alone (train/_smoke/v8/, live: train/_smoke/live_v8/):
    python train/train_v8.py --smoke --run sanity_v8 --init-v7 <copy of v7 best> --batch 16 --max-steps 2000 ...
    # score any v7/v8 save on the memory suite (no training), json + dreams for contact sheets:
    python train/train_v8.py --eval-mem train/checkpoints/dream_v7/best.pt --eval-out train/_smoke/v8/eval_v7.json

WHY v8 (PLAN.md 2026-10-21 "panorama memory PROVEN")
  Panorama memory (panorama.py, inference only) makes the dream come back to the same place when you turn
  away and back (return_sim 0.13 -> 0.88; full 360 0.02 -> 0.87), but the place returns as a fragmented
  COLLAGE: the model never practised (a) rebuilding a view from a patchwork of OLD frames, nor (b) the lower
  start noise (2.0) that play uses while turning. v8 trains exactly that; the network is v7's (model_v8.py).

WHAT v8 ADDS TO v7's RECIPE (everything else = train_v7.bat's settings)
  1. MEMORY-COLLAGE GHOST rows (--mem-ghost-frac, data_v8.mem_ghost): the window's own frames are painted
     into a panorama at their REAL headings (whole-pixel slides, exactly like panorama.py), and the target's
     starting ghost is cut from it; answer = the real target frame.
       - "newest" rows (1 - --mem-patch-p): newest frame wins per column, every frame up to the previous one
         = play's own ghost (the slid previous frame + older memory in the strip it leaves uncovered).
       - "patchwork" rows (--mem-patch-p): SYNTHETIC REVISITS. Real footage almost never comes back to a
         direction it saw earlier in a 17 s window (data_v8.py --revisit-stats: 0.18% of windows have even
         5% of the view 'revisited', none 25%), so: leave out the newest 0..--mem-gap-max frames, cut the
         view into 4-24 px blocks, take each block from a RANDOM older remembered frame at its real heading
         (+ a small brightness drift per frame, --mem-jitter). The model must rebuild a clean photo from it.
  2. START NOISE of memory rows: v7's 4-5 by default (--mem-low-p 0). The low band (--mem-sigma-low 1.8-3.2,
     what play starts turning frames at) was the first plan and FAILED the sanity test (2026-10-21, see below):
     sigma 2-3 is also the MIDDLE of every normal dream, so the model can't tell "first step from a wrong
     ghost" from "step 3 of a normal dream" and everything got softer (no-memory melt 12.4 -> 10.2 s).
  3. v7's shifted-ghost rows (--ghost-frac) and the normal denoising rows stay.
  4. THE RULER RUNS WITH PANORAMA MEMORY (play's setting: no fading while turning in place, start noise
     --mem-smax 2.0 on turning frames whose ghost is mostly remembered; --mem-write centre = the panorama's
     opt-in write_centre rule, the sanity test's biggest win, see below). Every rollout logs:
       turn-back return_sim (+ control), full-360 return_sim, 40 s ruler WITH memory (melt / clean2 / scene),
       40 s ruler WITHOUT memory (v7's shift start: normal walking must not regress), and COLLAGE
       CLEANLINESS of the turned-back frames: scene distance to the walk's real past (lower = more
       photo-like), 'seam' (straight vertical cuts) and 'hf' (fine-detail energy), next to real frames' values.
  5. best.pt = the best COMBINED SCORE (documented in combined_score below):
         clean2 WITH memory / ruler length  x  min(1, turn-back return_sim / 0.85)
           x  min(1, real frames' scene distance / turned-back frames' scene distance)
           x  DETAIL RETURN = mean SSIM(view before the turn, view after turning back / after a full 360)
           x  min(1, clean2 WITHOUT memory / (0.75 x the parent's))   (normal walking must not get worse;
              0.75 because the parent v7 best.pt is a lucky max on these exact walks + noise draws)
     Collage noise can't win: a collage keeps the layout (return_sim) but loses the detail SSIM measures
     (the scene meter alone was FOOLED by v7's collage: 0.64 vs real 0.61, so SSIM is the factor that bites);
     mush/static raise the scene distance; ignoring memory loses return_sim. best_walk.pt = the best
     no-memory clean2 (v7's rule).

SANITY TEST (2026-10-21, train/_smoke/v8/, 2500 steps at batch 16 from v7 best, lr 3e-5, EMA 0.998; same 5 walks):
  - R1 = the first plan (memory ghosts 20%, 60% of them at noise 1.8-3.2, patchwork gap <= 30): WORSE everywhere
    (20-dream rulers: no-memory clean2 5.0/5.5 s vs v7 5.8/6.9, memory ruler 6.0/6.9 vs 9.2/11.9, softer pictures).
  - R2 = memory ghosts at 4-5 only, easier patchwork (gap <= 10, oldest <= 60 steps) = the defaults now:
    memory ruler 10.9/15.0 s (v7 9.2/11.9, a v7-recipe control fine-tune 7.5/10.8), scene 1.09 (v7 1.17);
    no-memory walking unchanged (5.6/6.9, scene 1.29 vs 1.71); turn-back about v7's (0.97, detail SSIM 0.62 vs 0.61).
  - The COLLAGE is mostly made by the panorama's write rule, not the model: with write() the view before a turn
    is repainted ~34 times by drifting re-dreams while it slides out of sight. PanoramaMemory.write_centre (keep
    the most-central sighting while turning) on plain v7: turn-back 0.88 -> 0.97, detail SSIM 0.37 -> 0.61.

Dream Lab contract (as v7): train/live/v8-<date-time>/ (image "dream model v8"), run.log, metrics.jsonl,
frame_*.jpg (the memory ruler's walk 0). Videos: train/out/v8_rollout_step<step>.mp4 (memory ruler, walk 0),
train/out/v8_turnclip_step<step>.mp4 (turn away and back WITH memory), train/out/v8_turn_step<step>.jpg.
v8 saves keep model_cfg version "v7" (+ trained_as "v8"), so play.py plays them as v7 with no changes.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import copy
import json
import math
import pathlib
import shutil
import time
import traceback

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from build_cache import compute_valid                                          # noqa: E402
from data_v3 import (NON_WALKING, InfiniteWeightedSampler, load_cache_any,       # noqa: E402
                     parse_source_weights, source_shares, to_model_space)
from data_v6 import needed_positions, position_lookup                           # noqa: E402
from data_v7 import TEST_TURN_DEG, SparseWindowDatasetV7, describe_table, key_motion_table, load_motion  # noqa: E402
from data_v8 import FADE, MIN_CONF, mem_ghost                                   # noqa: E402
from model import N_ACTIONS                                                     # noqa: E402
from model_v3 import FP_FRAMES, colour_fingerprint                              # noqa: E402
from model_v7 import NULL_ACTION, build_model_v7                                # noqa: E402
from model_v8 import v8_cfg_from_v7                                             # noqa: E402
from panorama import PanoramaMemory                                             # noqa: E402
from scene_meter import features as scene_features                              # noqa: E402
from train import Live, safe_replace                                            # noqa: E402
from train_v1 import pose_string, sample_sigma_cond                             # noqa: E402
from train_v2 import gpu_free_gb, write_live_frames, write_mp4                  # noqa: E402
from train_v3 import (FAIR_SCRIPT, RIPPLE_LEVEL, GraphedSampler, eager_sample,   # noqa: E402
                      ruler_line, sample_sigma, score_walks, sf_counts)
from train_v6 import SCENE_FIT, V5_RULER_STARTS, scene_score                    # noqa: E402
from train_v7 import (MemIndexV7, dream_walks_v7, load_as_v7, motion_shift_px,  # noqa: E402
                      self_force_v7, shift_frames, turn_clip)
from turn_test import TURN_STEPS, TURN_WALK, layout, turn_picture, turn_scores  # noqa: E402

TURN_KEYS = (5, 6)
CIRCLE_STEPS = 151        # 151 x 5 px ~ the panorama's 754 columns = a full turn at 2.4 deg/step


# ---------------------------------------------------------------- dreaming WITH panorama memory (play's setting)

@torch.no_grad()
def dream_mem(model, frames_mm, actions_np, motion_mm, row_starts, keys, table, motion_scale, steps=10, seed=1234,
              fade_on_turn=False, smax_mem=2.0, smax=5.0, fade=FADE, min_conf=MIN_CONF, share_min=0.5,
              write_rule="newest"):
    """pano_test.dream_pano, generalised (any model, any keys per row) and set up like play.py:
      - the real history is painted into the panorama at its real headings (newest last), with its real fading;
      - every dreamed frame starts from ghost = remembered columns (confidence >= min_conf) else the previous
        frame slid sideways by the asked turn, then slid up/down by the asked tilt;
      - turning rows (keys 5/6) whose ghost is mostly remembered (share > share_min) start at smax_mem (2.0),
        others at smax (5.0) - decided PER ROW (play does it for its one row);
      - no fading on turn-in-place steps (fade_on_turn=False), real forward fading otherwise.
      - write_rule "newest" = panorama.py (every dreamed frame repaints all its columns). EXPERIMENT "centre":
        on turning steps a column is only repainted if this frame sees it at least as close to the middle of
        the view as the stored version was (or the stored one is forgotten); walking steps repaint everything.
        So the view before a turn is NOT repainted ~34 times by drifting re-dreams while it slides out of sight.
    Returns (uint8 [rows, T, H, W, 3], share of low-noise starts)."""
    dev = next(model.parameters()).device
    K, Hn = model.K, model.history
    rows_keys = [list(k) for k in keys] if isinstance(keys[0], (list, tuple)) else [list(keys)] * len(row_starts)
    B, T = len(row_starts), len(rows_keys[0])
    tab = torch.as_tensor(table, dtype=torch.float32, device=dev)
    mscale = torch.as_tensor(motion_scale, dtype=torch.float32, device=dev)
    kseq = torch.tensor([[int(a) if a >= 0 else 1 for a in actions_np[s - Hn:s - 1]] + rk
                         for s, rk in zip(row_starts, rows_keys)], device=dev)
    mseq = torch.cat([torch.stack([torch.from_numpy(np.asarray(motion_mm[s - Hn:s - 1], np.float32)) for s in row_starts]).to(dev),
                      tab[kseq[:, Hn - 1:].clamp(0, N_ACTIONS - 1)]], dim=1)
    hist = torch.stack([to_model_space(torch.from_numpy(np.array(frames_mm[s - Hn:s])).to(dev)) for s in row_starts])
    H, W = hist.shape[-2:]
    f0 = Hn - model.fp_offset
    colour = colour_fingerprint(hist[:, f0:f0 + FP_FRAMES])
    offs = torch.tensor(model.mem_offsets(), device=dev)
    fpx = (W / 2) / math.tan(math.radians(35))
    fwd_w = float(tab[1, 2] * mscale[2])
    pano = PanoramaMemory(B, H, fpx, dev, fade_per_forward=fade, min_conf=min_conf)
    real_slide = torch.stack([motion_shift_px(mseq[:, i], mscale, fpx)[:, 0].round() for i in range(Hn - 1)], 1)
    head_hist = torch.cat([torch.zeros(B, 1, device=dev), real_slide.cumsum(1)], 1)
    for i in range(Hn):
        pano.heading = head_hist[:, i].clone()
        pano.write(hist[:, i])
        if i < Hn - 1:
            pano.forward((mseq[:, i, 2] * mscale[2]).clamp(min=0) / fwd_w)
    sig_hi = model.karras_sigmas(steps, sigma_max=smax, device=dev).float()
    sig_lo = model.karras_sigmas(steps, sigma_max=smax_mem, device=dev).float()
    turn_ids = torch.tensor(TURN_KEYS, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    head, out, n_low = 0, [], 0
    b4 = lambda v: v[:, None, None, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(T):
            kk = kseq[:, t:t + Hn]
            ext = model.extras(kk, mseq[:, t + Hn - K:t + Hn])
            ctx = hist.index_select(1, (head - offs) % Hn)
            sh = motion_shift_px(mseq[:, t + Hn - 1], mscale, fpx).round()
            slid = shift_frames(ctx[:, -1].float(), torch.stack([sh[:, 0], torch.zeros_like(sh[:, 1])], 1))
            pano.turn(sh[:, 0])
            cols = pano._cols(W)                                                               # [B, W]
            mem = torch.gather(pano.img, 3, cols[:, None, None, :].expand(B, 3, H, W))
            use = torch.gather(pano.conf, 1, cols) >= min_conf                                 # [B, W]
            ghost = torch.where(use[:, None, None, :], mem, slid)
            if bool((sh[:, 1] != 0).any()):
                ghost = shift_frames(ghost, torch.stack([torch.zeros_like(sh[:, 1]), sh[:, 1]], 1))
            turning = torch.isin(kk[:, -1], turn_ids)
            low = turning & (use.float().mean(1) > share_min)
            n_low += int(low.sum())
            sig = torch.where(low[:, None], sig_lo[None], sig_hi[None])                       # [B, steps+1]
            x = torch.randn(ghost.shape, device=dev, generator=g) * b4(sig[:, 0]) + ghost
            for i in range(steps):
                s, sn = sig[:, i], sig[:, i + 1]
                den, _ = model(x, s, ctx, 0.0, kk[:, -K:], colour, ext)
                x = x + (x - den.float()) / b4(s) * b4(sn - s)
            nxt = x.clamp(-1, 1)
            if write_rule == "centre":
                pano.write_centre(nxt, turning)
            else:
                pano.write(nxt)
            fwd = (mseq[:, t + Hn - 1, 2] * mscale[2]).clamp(min=0) / fwd_w
            if not fade_on_turn:
                fwd = fwd * (~turning).float()
            pano.forward(fwd)
            out.append(((nxt + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            hist[:, head] = nxt
            head = (head + 1) % Hn
    return np.stack(out, 1), n_low / max(1, B * T)


# ---------------------------------------------------------------- collage cleanliness

def frame_stats(u8):
    """[N, H, W, 3] uint8 -> (seam [N], hf [N]).
    seam: how much of the left-right change between neighbouring columns is a STRAIGHT VERTICAL CUT
          (the same jump all the way down the column, as at the edge of a pasted strip):
          mean over columns of |column-mean of the x-difference| / mean |x-difference|.
    hf:   fine-detail energy: mean |Laplacian| of the grey picture / its contrast (shreds and grain raise it,
          mush lowers it). Both are only meaningful next to REAL frames' values (reported alongside)."""
    g = u8.astype(np.float32).mean(-1) / 255.0
    dx = np.diff(g, axis=2)
    seam = np.abs(dx.mean(1)).mean(1) / (np.abs(dx).mean((1, 2)) + 1e-6)
    lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    hf = np.abs(lap).mean((1, 2)) / (g.std((1, 2)) + 1e-3)
    return seam, hf


def ssim(a_u8, b_u8):
    """Mean grey-level SSIM (Gaussian 7x7, sigma 1.5, the usual constants) of frame pairs [N, H, W, 3] -> [N].
    Used as 'did the SAME view come back in detail': the turned-back frame vs the dreamed frame just before
    the turn (turning in place, so ideally identical). A collage keeps the layout but loses this."""
    import cv2
    out = []
    for a, b in zip(a_u8, b_u8):
        x = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY).astype(np.float64)
        y = cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_RGB2GRAY).astype(np.float64)
        bl = lambda z: cv2.GaussianBlur(z, (7, 7), 1.5)
        mx, my = bl(x), bl(y)
        vx, vy, cxy = bl(x * x) - mx * mx, bl(y * y) - my * my, bl(x * y) - mx * my
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        out.append(float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()))
    return np.array(out)


def scene_dists(u8, row_starts, frames_mm, sd):
    """Scene distance of each frame [N] from ITS walk's real past (4.8 s before the walk), real-footage units."""
    o = []
    for f, s in zip(u8, row_starts):
        mu = scene_features(np.array(frames_mm[s - 48:s])).mean(0)
        o.append(float(np.sqrt((((scene_features(f[None]) - mu) / sd) ** 2).mean())))
    return np.array(o)


def collage_numbers(u8, row_starts, frames_mm, sd):
    seam, hf = frame_stats(u8)
    return {"scene": round(float(scene_dists(u8, row_starts, frames_mm, sd).mean()), 3),
            "seam": round(float(seam.mean()), 4), "hf": round(float(hf.mean()), 4)}


def real_collage_ref(starts, frames_mm, sd, n=40):
    """The same numbers for REAL frames: the n real frames after each walk start (vs that walk's past)."""
    fr = np.concatenate([np.array(frames_mm[s:s + n]) for s in starts])
    rs = [s for s in starts for _ in range(n)]
    return collage_numbers(fr, rs, frames_mm, sd)


# ---------------------------------------------------------------- the memory suite (rollouts + --eval-mem)

def memory_suite(model, frames_mm, actions_np, motion_mm, starts, seeds, table, motion_scale, sd, ruler_keys,
                 steps=10, smax_mem=2.0, walk=TURN_WALK, n=TURN_STEPS, nomem=True, log=print, write_rule="newest"):
    """Turn test + full circle + 40 s ruler WITH memory, the 40 s ruler WITHOUT memory (v7's shift start),
    and collage cleanliness of the turned-back / full-circle frames. Returns (numbers, dreams dict)."""
    dev = next(model.parameters()).device
    rows = [s for _ in range(seeds) for s in starts]
    k = len(rows)
    res, dreams = {}, {}
    t0 = time.time()
    tk, ck = [1] * walk + [6] * n + [5] * n, [1] * walk + [6] * n + [1] * n
    d, _ = dream_mem(model, frames_mm, actions_np, motion_mm, rows + rows, [tk] * k + [ck] * k, table, motion_scale,
                     steps, smax_mem=smax_mem, write_rule=write_rule)
    td, cd = d[:k], d[k:]
    r = turn_scores(td, cd, len(starts), scene_features, sd, walk, n)
    res["turn"] = {kk: r[kk] for kk in ("return_sim", "ctrl_sim", "memory", "away_sim", "return_sim_scene",
                                       "turned_right_deg", "turned_back_deg")}
    res["back"] = collage_numbers(td[:, walk + 2 * n - 1], rows, frames_mm, sd)
    res["back"]["ssim_pre"] = round(float(ssim(td[:, walk - 1], td[:, walk + 2 * n - 1]).mean()), 4)
    res["back"]["ssim_ctrl"] = round(float(ssim(td[:, walk - 1], cd[:, walk + 2 * n - 1]).mean()), 4)
    dreams["turn"], dreams["ctrl"] = td, cd
    log(f"  memory turn test ({time.time() - t0:.0f}s): {res['turn']}; turned-back frames {res['back']}")
    circ = [1] * walk + [6] * CIRCLE_STEPS
    d2, _ = dream_mem(model, frames_mm, actions_np, motion_mm, rows, circ, table, motion_scale, steps, smax_mem=smax_mem, write_rule=write_rule)
    pre, end = layout(d2[:, walk - 1]), layout(d2[:, -1])
    other = [float(((pre[i] - pre[j]) ** 2).mean()) for i in range(k) for j in range(k) if i != j and i // len(starts) == j // len(starts)]
    res["circle_return_sim"] = round(float(1 - np.mean([float(((pre[i] - end[i]) ** 2).mean()) for i in range(k)]) / np.mean(other)), 3)
    res["circle_end"] = collage_numbers(d2[:, -1], rows, frames_mm, sd)
    res["circle_end"]["ssim_pre"] = round(float(ssim(d2[:, walk - 1], d2[:, -1]).mean()), 4)
    dreams["circle"] = d2
    log(f"  memory full circle ({time.time() - t0:.0f}s): return_sim {res['circle_return_sim']}; end frames {res['circle_end']}")
    T = len(ruler_keys)
    d3, low = dream_mem(model, frames_mm, actions_np, motion_mm, rows, ruler_keys, table, motion_scale, steps, smax_mem=smax_mem, write_rule=write_rule)
    summ, reps = score_walks(d3, rows, frames_mm, model.K, T)
    res["ruler_mem"] = {"melt_median": summ["melt_median"], "clean2_median": summ["clean2_median"],
                        "clean2_mean": summ["clean2_mean"], "scene": scene_score(d3, rows, frames_mm, sd),
                        "low_noise_share": round(low, 3)}
    dreams["ruler_mem"], dreams["ruler_mem_sharp"] = d3, reps[0][1]
    log(f"  {T // 10} s ruler WITH memory ({time.time() - t0:.0f}s): {res['ruler_mem']}")
    if nomem:
        d4 = dream_walks_v7(model, frames_mm, actions_np, motion_mm, rows, ruler_keys, steps, dev, table,
                            start="shift", motion_scale=motion_scale)
        summ4, _ = score_walks(d4, rows, frames_mm, model.K, T)
        res["ruler_nomem"] = {"melt_median": summ4["melt_median"], "clean2_median": summ4["clean2_median"],
                              "clean2_mean": summ4["clean2_mean"], "scene": scene_score(d4, rows, frames_mm, sd)}
        dreams["ruler_nomem"] = d4
        log(f"  {T // 10} s ruler WITHOUT memory, shift start ({time.time() - t0:.0f}s): {res['ruler_nomem']}")
    return res, dreams


def combined_score(res, ruler_secs, real_scene, parent_nomem_clean2=None):
    """best.pt's score (see the top of the file). Each factor is <= 1 except the first (0..1 as well)."""
    s = res["ruler_mem"]["clean2_median"] / ruler_secs
    s *= max(0.0, min(1.0, res["turn"]["return_sim"] / 0.85))
    s *= min(1.0, real_scene / max(res["back"]["scene"], 1e-6))
    s *= max(0.0, 0.5 * (res["back"]["ssim_pre"] + res["circle_end"]["ssim_pre"]))
    if parent_nomem_clean2 and "ruler_nomem" in res:
        # 0.75 x the parent: v7's best.pt was picked as the max of ~100 noisy rulers on THESE walks + seeds
        # (fresh noise: 5.8 s instead of 8.9), so any later save looks worse on them by luck alone
        s *= min(1.0, res["ruler_nomem"]["clean2_median"] / max(0.75 * parent_nomem_clean2, 1e-6))
    return round(float(s), 4)


def eval_main(args):
    device = "cuda"
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    _, fm, act, _ = load_cache_any(cache_dir)
    mot, minfo = load_motion(cache_dir)
    tmp = HERE / "_exp" / f"v8_eval_ck_{os.getpid()}.pt"                 # copy first: never hold a live run's file open
    shutil.copy(args.eval_mem, tmp)
    m, step, ver = load_as_v7(tmp, device)
    tmp.unlink()
    sd = np.load(SCENE_FIT)["sd"]
    starts = [int(x) for x in args.ruler_starts.split(",")]
    table = key_motion_table(minfo, args.test_turn_deg)
    ruler_keys = [a for a, n in ruler_key_list(args.ruler_secs) for _ in range(n)]
    print(f"{args.eval_mem}: {ver} step {step}", flush=True)
    res, dreams = memory_suite(m, fm, act, mot, starts, args.ruler_seeds, table, minfo["scale"], sd, ruler_keys,
                               args.rollout_steps, args.mem_smax, args.turn_walk, args.turn_steps,
                               log=lambda s: print(s, flush=True), nomem=not args.eval_skip_nomem, write_rule=args.mem_write)
    res["real"] = real_collage_ref(starts, fm, sd)
    res["combined_score_no_guard"] = combined_score(res, args.ruler_secs, res["real"]["scene"])
    res["ckpt"], res["step"] = str(args.eval_mem), step
    print(json.dumps(res, indent=1), flush=True)
    if args.eval_out:
        out = pathlib.Path(args.eval_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        nw = len(starts)
        np.savez_compressed(out.with_suffix(".npz"), turn=dreams["turn"][:nw], circle=dreams["circle"][:nw])
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def ruler_key_list(ruler_secs):
    script, t_ = [], 0
    while t_ < ruler_secs * 10:
        for a_, n_ in FAIR_SCRIPT:
            n_ = min(n_, ruler_secs * 10 - t_)
            if n_ > 0:
                script.append((a_, n_)); t_ += n_
    return script


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v8", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-v7", default="", help="v7 save to warm-start a NEW run from (use a COPY, not the live file)")
    ap.add_argument("--eval-mem", default="", help="just run the memory suite on this v7/v8 save, then exit")
    ap.add_argument("--eval-out", default="", help="--eval-mem: write the numbers here (.json) + first-seed dreams (.npz)")
    ap.add_argument("--smoke", action="store_true", help="checkpoints/out under train/_smoke/v8/, live in train/_smoke/live_v8/")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--colour-drop", type=float, default=0.1)
    # v7 steering + ghost practice (unchanged)
    ap.add_argument("--steer-drop", type=float, default=0.1)
    ap.add_argument("--onset-weight", type=float, default=4.0)
    ap.add_argument("--bigturn-weight", type=float, default=2.0)
    ap.add_argument("--onset-deg", type=float, default=1.5)
    ap.add_argument("--onset-ratio", type=float, default=0.6)
    ap.add_argument("--bigturn-deg", type=float, default=3.0)
    ap.add_argument("--test-turn-deg", type=float, default=TEST_TURN_DEG)
    ap.add_argument("--ghost-frac", type=float, default=0.15, help="v7's shifted-ghost rows")
    ap.add_argument("--ghost-sigma-min", type=float, default=4.0)
    ap.add_argument("--ghost-sigma-max", type=float, default=5.0)
    ap.add_argument("--ghost-shift-p", type=float, default=1.0)
    # v8 memory-collage ghosts
    ap.add_argument("--mem-ghost-frac", type=float, default=0.2, help="share of rows with a panorama-memory ghost")
    ap.add_argument("--mem-patch-p", type=float, default=0.65, help="of those: share of synthetic-revisit patchwork rows")
    ap.add_argument("--mem-gap-max", type=int, default=10, help="patchwork: leave out the newest 0..this many frames")
    ap.add_argument("--mem-age-min", type=int, default=8, help="patchwork: oldest usable frame at least this many steps back ...")
    ap.add_argument("--mem-age-max", type=int, default=60, help="... and at most this many (drawn per row)")
    ap.add_argument("--mem-jitter", type=float, default=0.04, help="patchwork: per-frame brightness/contrast drift")
    ap.add_argument("--mem-sigma-low", default="1.8,3.2", help="memory rows: low start-noise band (play turns at 2.0)")
    ap.add_argument("--mem-low-p", type=float, default=0.0, help="share of memory rows in the low band (rest: 4-5)")
    ap.add_argument("--mem-write", choices=["newest", "centre"], default="centre",
                    help="tests: panorama write rule (newest = panorama.py / play; centre = experiment, see dream_mem)")
    ap.add_argument("--eval-skip-nomem", action="store_true", help="--eval-mem: skip the no-memory ruler")
    ap.add_argument("--mem-smax", type=float, default=2.0, help="tests: start noise on remembered ghosts while turning (play's)")
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
    ap.add_argument("--sf-frac", type=float, default=0.30)
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
    ap.add_argument("--no-baseline", action="store_true", help="skip scoring the warm start (= v7) at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0)
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--ruler-secs", type=int, default=40)
    args = ap.parse_args()
    if args.eval_mem:
        return eval_main(args)

    base = HERE / "_smoke" / "v8" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "live_v8" if args.smoke else HERE / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    elif not args.allow_low_vram:
        free = gpu_free_gb()
        need = 0.30 * args.batch + 7.0
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
    if ck and ck.get("motion_info") and ck["motion_info"]["scale"] != minfo["scale"]:
        sys.exit("motion.npy was rebuilt with different scales since this run started. Rebuild it as before, or start a new run.")
    frame_hw = list(index["shape"][1:3])
    parent_ck = None
    if ck:
        model_cfg = ck["model_cfg"]
    else:
        if not args.init_v7:
            sys.exit("A new v8 run needs --init-v7 <a copy of a v7 save>.")
        parent_ck = torch.load(args.init_v7, map_location="cpu", weights_only=False)
        model_cfg = v8_cfg_from_v7(parent_ck["model_cfg"])
        if parent_ck.get("motion_info") and parent_ck["motion_info"]["scale"] != minfo["scale"]:
            sys.exit("the v7 save was trained with other motion scales than data/cache_v3/motion.npy has now")
    if list(model_cfg["frame_hw"]) != frame_hw:
        sys.exit(f"the model dreams {model_cfg['frame_hw']} frames but {cache_dir} holds {frame_hw}")
    model = build_model_v7(model_cfg).to(device)
    K, Hn = model.K, model.history
    sf_w = [float(x) for x in args.sf_weights.split(",")]
    SF_MAX = len(sf_w)
    L = Hn + SF_MAX
    fp_train = [model.fp_offset + SF_MAX - i for i in range(FP_FRAMES)]
    positions = needed_positions(model.mem_offsets(), L, SF_MAX, fp_train)
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
    real_ref = real_collage_ref(starts, frames_mm, scene_sd)

    # ---- sampling weights: exactly v7's
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

    # ---- model, EMA, optimizer (+ the warm start: a straight copy, v8 = v7's network)
    n_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    def make_opt(m):
        decay, no_decay = [], []
        for name, p in m.named_parameters():
            (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": args.wd, "mult": 1.0},
                                  {"params": no_decay, "weight_decay": 0.0, "mult": 1.0}],
                                 lr=args.lr, betas=(0.9, 0.99), fused=True)
    opt = make_opt(model)
    init_notes = []
    if parent_ck is not None:
        model.load_state_dict(parent_ck["model"])
        ema.load_state_dict(parent_ck["ema"])
        init_notes.append(f"from {args.init_v7}: v7 run {parent_ck['run_id']} step {parent_ck['step']} "
                          f"({n_params / 1e6:.3f}M params): raw + EMA weights copied unchanged")
        try:
            opt.load_state_dict(parent_ck["opt"])
            init_notes.append("Adam memory carried over from v7 (same weights, same groups)")
        except Exception:
            opt = make_opt(model)
            init_notes.append("could not carry v7's Adam memory over, starting it fresh:\n" + traceback.format_exc())

    step, train_seconds = 0, 0.0
    run_id = f"{args.tag}-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline, best, best_walk, warm_check = None, None, None, None
    if ck:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline, best, best_walk, warm_check = ck.get("parent_baseline"), ck.get("best"), ck.get("best_walk"), ck.get("warm_check")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    names = ", ".join(c["name"] for c in index["cities"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": f"dream model {args.tag}",
        "prompt": f"dream {args.tag} ({names}), {n_params / 1e6:.0f}M params, {frame_hw[0]}x{frame_hw[1]}, "
                  f"memory {model.n_mem} frames back to {Hn / 10:.1f} s, steered by keys + camera motion, "
                  f"practising rebuilding views from panorama memory",
        "pose": pose_string(ruler_script), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "melt_median": None, "melt_mean": None, "clean2_median": None, "scene": None,
        "return_sim": None, "circle_return_sim": None, "back_scene": None, "back_seam": None,
        "nomem_clean2_median": None, "combined_score": None,
        "best_step": best["step"] if best else None, "best_walk_step": best_walk["step"] if best_walk else None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v8 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.3f}M params; {frame_hw[0]}x{frame_hw[1]}; memory {model.n_mem} frames ({Hn / 10:.1f} s); "
             f"window {L + 1} frames, {len(positions)} read; batch {args.batch}; {int(train_ok.sum())} train windows")
    live.log(f"memory-collage ghosts: {args.mem_ghost_frac:.0%} of rows ({args.mem_patch_p:.0%} of them synthetic-revisit "
             f"patchwork: newest 0..{args.mem_gap_max} frames left out, oldest {args.mem_age_min}..{args.mem_age_max} steps, "
             f"jitter {args.mem_jitter}); noise {lo_s}-{hi_s} for {args.mem_low_p:.0%} of them, else "
             f"{args.ghost_sigma_min}-{args.ghost_sigma_max}. v7 shifted-ghost rows {args.ghost_frac:.0%}. "
             f"Tests dream WITH panorama memory (no fade turning in place, start noise {args.mem_smax} on remembered turns).")
    live.log(f"real frames' collage numbers (the target for turned-back frames): {real_ref}")
    live.log(f"key -> motion table (turn = {args.test_turn_deg} deg/step): " + describe_table(minfo, args.test_turn_deg))
    for n_ in init_notes:
        live.log("warm start: " + n_)
    counts = sf_counts(int(round(args.sf_frac * args.batch)), sf_w)
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
                        "motion_info": minfo, "parent_baseline": baseline, "best": best, "best_walk": best_walk,
                        "warm_check": warm_check, "ruler_cities": [c["name"] for c in ruler], "ruler_starts": starts,
                        "warm_started": True}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def suite(label):
        res, dreams = memory_suite(ema, frames_mm, actions_np, motion_mm, starts, args.ruler_seeds, table, minfo["scale"],
                                   scene_sd, ruler_keys, args.rollout_steps, args.mem_smax, args.turn_walk,
                                   args.turn_steps, log=live.log, write_rule=args.mem_write)
        pc = baseline["ruler_nomem"]["clean2_median"] if baseline else None
        res["combined_score"] = combined_score(res, T / 10, real_ref["scene"], pc)
        live.log(f"{label}: COMBINED SCORE {res['combined_score']} = clean2 with memory {res['ruler_mem']['clean2_median']} / "
                 f"{T / 10:.0f} s x return {res['turn']['return_sim']} / 0.85 x real scene {real_ref['scene']} / turned-back "
                 f"scene {res['back']['scene']} x detail SSIM (back {res['back']['ssim_pre']} + full circle {res['circle_end']['ssim_pre']}) / 2"
                 + (f" x no-memory clean2 {res['ruler_nomem']['clean2_median']} / (0.75 x parent {pc})" if pc else "")
                 + (f"   [parent: {baseline.get('combined_score')}]" if baseline else ""))
        return res, dreams

    def do_rollout():
        nonlocal best, best_walk
        try:
            t0 = time.time()
            res, dreams = suite(f"{args.tag} step {step}")
            d3 = dreams["ruler_mem"]
            write_live_frames(live, d3[0], ruler_keys)
            path = write_mp4(out_dir / f"{args.tag}_rollout_step{step:07d}.mp4", d3[0], ruler_keys, real_clip,
                             f"{args.tag} step {step} (panorama memory)", dreams["ruler_mem_sharp"])
            turn_picture(out_dir / f"{args.tag}_turn_step{step:07d}.jpg", dreams["turn"], dreams["ctrl"], len(starts),
                         f"{args.tag} {step} memory", args.turn_walk, args.turn_steps)
            turn_clip(out_dir / f"{args.tag}_turnclip_step{step:07d}.mp4", dreams["turn"][0],
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

    # ---- start-up checks on a NEW warm-started run: v8 at init must BE v7
    if parent_ck is not None:
        live.update(stage="warm-start check")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ref = build_model_v7(parent_ck["model_cfg"]).to(device).eval()
        ref.load_state_dict(parent_ck["ema"])
        wdiff = max((p - q.to(device)).abs().max().item() for p, q in zip(ema.state_dict().values(), ref.state_dict().values()))
        with torch.no_grad():
            fr = to_model_space(val_batch[0][:16].to(device))
            act = val_batch[1][:16].to(device)
            mot = val_batch[2][:16].to(device)
            B = fr.shape[0]
            ctx, a, ext = mi_ema(fr, act, mot, L)
            col = colour_fingerprint(fr.index_select(1, fp_rows))
            tgt = fr[:, target_row]
            res = {"weights_maxabs": wdiff}
            for s in [0.1, 0.5, 2.0]:
                g = torch.Generator(device=device).manual_seed(0)
                xn = tgt + s * torch.randn(tgt.shape, device=device, generator=g)
                sig = torch.full((B,), s, device=device)
                res[f"denoise_sigma{s}_maxabs"] = (ema(xn, sig, ctx, 0.0, a, col, ext)[0] - ref(xn, sig, ctx, 0.0, a, col, ext)[0]).abs().max().item()
            p8 = ema.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=ext, start="prev",
                            generator=torch.Generator(device=device).manual_seed(1))
            p7 = ref.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=ext, start="prev",
                            generator=torch.Generator(device=device).manual_seed(1))
            res["dream_prev_start_maxabs"] = (p8 - p7).abs().max().item()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        del ref
        warm_check = res
        ok = res["weights_maxabs"] == 0 and res["dream_prev_start_maxabs"] < 1e-5
        live.log(f"WARM-START CHECK (v8 at init vs v7, float32, 16 held-out moments): weights max diff {wdiff:.1e}; "
                 f"one clean-up call at noise 0.1/0.5/2.0: {res['denoise_sigma0.1_maxabs']:.1e}/{res['denoise_sigma0.5_maxabs']:.1e}/"
                 f"{res['denoise_sigma2.0_maxabs']:.1e}; a full {args.rollout_steps}-step dream: {res['dream_prev_start_maxabs']:.1e} "
                 f"-> {'OK: v8 starts out as v7' if ok else 'WARNING: v8 differs from v7'}")
        live.metric({"step": step, "time": time.time(), "warm_check": res})
        torch.cuda.empty_cache()
        if not args.no_baseline:
            live.update(stage="v7 on the memory suite")
            res, _ = suite(f"parent BASELINE (= v8 at init; v7 step {parent_ck['step']})")
            baseline = dict(res, parent_step=parent_ck["step"])
            baseline["combined_score"] = combined_score(res, T / 10, real_ref["scene"], res["ruler_nomem"]["clean2_median"])
            live.metric({"step": step, "time": time.time(), "parent_baseline": baseline})
            torch.cuda.empty_cache()
    del parent_ck

    # ---- the loop
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
            live.log(f"CUDA-graph check: {diff:.2e} ({'OK' if diff < 2e-2 else 'WARNING: large; consider --no-graphs'})")
        except Exception:
            live.log("CUDA graphs FAILED, falling back to normal calls:\n" + traceback.format_exc())
            sf_sampler = eager_sample(ema, start=args.sf_start)
    torch.cuda.reset_peak_memory_stats()
    n_sf = sum(counts)
    now = time.time()
    last_status, last_ckpt, last_keep = 0.0, now, now
    next_sample = None if step < args.first_sample_step else now + args.sample_every_min * 60
    sample_soon = step < args.first_sample_step
    loss_avg = None
    win_t, win_steps, it_s = time.time(), 0, 0.0
    bad_steps = 0
    t_loop = time.time()
    n_seen = n_ghost = n_mem = n_patch = 0
    mem_share_sum = 0.0
    live.update(stage="training")

    try:
        it = iter(loader)
        while True:
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_hours and (time.time() - t_loop) / 3600 >= args.max_hours:
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
            if n_sf:
                fr = self_force_v7(sf_sampler, mi_ema, fr, act, mot, colour, counts, args.sf_steps, L)
            drop = torch.rand(B, device=device) < args.steer_drop
            context, actions, ext = mi(fr, act, mot, L, absent=drop)
            actions = torch.where(drop[:, None], torch.full_like(actions, NULL_ACTION), actions)
            n_seen += B
            target = fr[:, target_row]
            sigma = sample_sigma(B, args, device)
            sigma_cond = sample_sigma_cond(B, model.n_mem, args, device)
            # v7's shifted-ghost rows
            u = torch.rand(B, device=device)
            use_v7 = u < args.ghost_frac
            use_mem = (u >= args.ghost_frac) & (u < args.ghost_frac + args.mem_ghost_frac)
            lo, hi = math.log(args.ghost_sigma_min), math.log(args.ghost_sigma_max)
            sigma = torch.where(use_v7, torch.exp(lo + (hi - lo) * torch.rand(B, device=device)), sigma)
            ghost = fr[:, prev_row].float()
            px = motion_shift_px(mot[:, L - 1], minfo["scale"], fpx_train) * (torch.rand(B, device=device) < args.ghost_shift_p)[:, None]
            ghost = shift_frames(ghost, px)
            n_ghost += int(use_v7.sum())
            # v8's memory-collage ghost rows
            if bool(use_mem.any()):
                sel = torch.nonzero(use_mem).squeeze(1)
                m = len(sel)
                patch = torch.rand(m, device=device) < args.mem_patch_p
                gap = torch.where(patch, torch.randint(0, args.mem_gap_max + 1, (m,), device=device), torch.zeros(m, dtype=torch.long, device=device))
                age = torch.randint(args.mem_age_min, args.mem_age_max + 1, (m,), device=device)
                age = torch.where(patch, age, torch.full_like(age, 10 ** 6))
                mg, msh = mem_ghost(fr.index_select(0, sel), lut, positions, mot.index_select(0, sel), mscale_t, fpx_train,
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
                loss = model.loss_ghost(target, context, actions, sigma, sigma_cond, colour, ext, ghost, use_ghost)
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
                do_rollout()
                next_sample = time.time() + args.sample_every_min * 60
            if step % 100 == 0 or step == 1:
                vram = torch.cuda.max_memory_reserved() / 2**30
                eta = (f"{(next_sample - now) / 60:.1f} min" if next_sample else f"at step {args.first_sample_step}")
                live.log(f"step {step}  loss {loss_avg:.4f}  {it_s:.2f} it/s  lr {lr:.2e}  grad {gn.item():.2f}  VRAM {vram:.1f} GB  "
                         f"rows: v7 ghost {n_ghost}, memory ghost {n_mem} ({n_patch} patchwork, {mem_share_sum / max(1, n_mem):.0%} "
                         f"of their columns remembered) of {n_seen}  next dream {eta}")
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
        live.log(f"finished at step {step}")
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
