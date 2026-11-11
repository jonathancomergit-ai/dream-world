"""Train Dream World v7: REAL STEERING, warm-started from v6 (or v5).

    # the real run (or just double-click train_v7.bat, which first copies v6's newest best.pt):
    python train/train_v7.py --run dream_v7 --init-v6 train/checkpoints/dream_v7_init.pt ...
    python train/train_v7.py --run dream_v7 --resume ...          # carry on after a stop
    # a small test that leaves the real folders alone (train/_smoke/v7/, live: train/_smoke/live_v7/):
    python train/train_v7.py --smoke --run smoke_v7 --batch 4 --max-steps 200 --first-sample-step 100 ...
    # score any save (v5 / v6 / v7) on the turn test at guidance 0 and 2, no training:
    python train/train_v7.py --eval train/checkpoints/dream_v6/best.pt

WHAT v7 CHANGES (everything else is v6's recipe: train_v6.bat)
  The 2026-10-20 finding: the model barely steers. Holding "turn right" 4 s turns the dream ~1 deg;
  swapping the key changes its guess by only 2-4% of a real frame change. v7 attacks that four ways:
  1. CONTINUOUS MOTION (data_v7.py, model_v7.py). Besides the 9 coarse keys, the model gets the
     measured camera motion (yaw, pitch, forward, strafe) of the 8 recent steps, the last one being
     the step it is dreaming. It enters through a new little network whose last layer starts at
     zero, so at step 0 v7 dreams EXACTLY what its parent dreams (checked below, with numbers).
  2. TURN-ONSET PRACTICE (this file, --onset-weight, --bigturn-weight). Training windows are drawn
     more often when the step being dreamed is a clear turn but the 4 steps before it weren't: the
     memory shows no turning, so ONLY the key/motion can tell the model to turn. And windows with a big
     turn are drawn more often in general. run.log reports their share before/after.
  3. STEERING DROPOUT (--steer-drop 0.1). 10% of examples get no steering at all (the 8 recent key
     slots -> the new "no key" id, motion -> zeros + an "absent" flag). That teaches the model a
     "no steering" dream, which makes GUIDANCE possible when dreaming:
         den = den(steer) + w * (den(steer) - den(no steering))
     w = 0 is the normal dream; w = 2 exaggerates whatever your steering asks for.
  4. THE TURN TEST IS THE KEY NUMBER (turn_test.py's design, dreamed through the v7 key->motion table):
     walk 2 s, turn right 60 steps at 2.4 deg/step (asked ~144 deg, as v6's test), turn back left 60.
     Logged every rollout at w = 0 AND w = 2: turned_deg (how far the dream really turned, measured by
     the same yaw meter), turned fraction = achieved / asked, and return_sim.
     best.pt is still chosen by the clean2 ruler (as v6). NEW best_steer.pt = the save with the best
         steer score = clamp(turned fraction at w=0, 0..1) x (clean2 median / ruler length)
     i.e. "turns as asked AND stays clean"; a model that turns perfectly but melts scores low, and so
     does one that is clean but never turns.
  5. A THIRD TURN TEST with the experimental 'shift' start (--no-turn-test-shift skips it): each dreamed
     frame starts from the previous frame SLID by the asked turn (+ static) instead of the plain previous
     frame. Found while building v7: v6 itself turns ~170 of 144 deg asked that way, with no training,
     but the strip it must invent at the edge is smeary. Watching it tells us whether v7 learns to fill
     that strip in well (and whether 'shift' should become the default dream start later).

Dream Lab contract (as v6): train/live/v7-<date-time>/ with status.json (image "dream model v7", plus
turned_deg, turned_frac_w0, turned_frac_w2, steer_score, best_steer_step, return_sim, ...), run.log,
metrics.jsonl, frame_*.jpg. Videos: train/out/v7_rollout_step<step>.mp4, v7_anchor_rollout_step<step>.mp4,
the turn-around picture train/out/v7_turn_step<step>.jpg and a short dreamed turn clip
train/out/v7_turnclip_step<step>.mp4 (walk 0: the turn rows at w = 0, real start on the left).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
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
from data_v7 import (TEST_TURN_DEG, SparseWindowDatasetV7, describe_table,       # noqa: E402
                     key_motion_table, load_motion)
from model import N_ACTIONS                                                     # noqa: E402
from model_v3 import FP_FRAMES, build_model_v3, colour_fingerprint              # noqa: E402
from model_v6 import V5_OFFSETS                                                 # noqa: E402
from model_v7 import (NULL_ACTION, build_model_v7, parent_model, v7_cfg_from_parent,  # noqa: E402
                      warm_start_v7)
from model_v6 import warm_start_opt_state                                       # noqa: E402
from scene_meter import features as scene_features                              # noqa: E402
from train import Live, safe_replace                                            # noqa: E402
from train_v1 import melt_report, pose_string, sample_sigma_cond, secs          # noqa: E402
from train_v2 import gpu_free_gb, write_live_frames, write_mp4                  # noqa: E402
from train_v3 import (FAIR_SCRIPT, RIPPLE_LEVEL, GraphedSampler, eager_sample,   # noqa: E402
                      ruler_line, sample_sigma, score_walks, sf_counts, walk_sharp_ref)
from train_v6 import SCENE_FIT, V5_RULER_STARTS, scene_score                    # noqa: E402
from turn_test import TURN_STEPS, TURN_WALK, layout, turn_picture, turn_scores  # noqa: E402

GUIDE_TEST_W = 2.0           # the second turn test of every rollout uses this guidance strength


# ---------------------------------------------------------------- memory helper (sparse batches + motion)

class MemIndexV7:
    """train_v6.MemIndex + motion: picks the memory, the 8 recent keys and the full far_act vector
    (key summaries + motion block) for dream-window position g out of a SPARSE batch."""

    def __init__(self, model, lut, device):
        self.model, self.lut, self.dev = model, lut, device
        self.rows = {}

    def row(self, g):
        r = int(self.lut[g])
        assert r >= 0, f"window position {g} was not read"
        return r

    def __call__(self, fr, act, mot, g, absent=None):
        """-> (memory [B, n_mem, 3, H, W] oldest first, recent keys [B, K], far_act [B, n_keysum + motion_in]).
        mot [B, L, 4]: motion of every step of the window (mot[:, i] goes with act[:, i])."""
        m = self.model
        if g not in self.rows:
            r = self.lut[torch.tensor([g - o for o in m.mem_offsets()])]
            assert bool((r >= 0).all()), f"a memory frame for position {g} was not read"
            self.rows[g] = r.to(self.dev)
        ctx = fr.index_select(1, self.rows[g])
        ext = m.extras(act[:, g - m.history:g], mot[:, g - m.K:g], absent)
        return ctx, act[:, g - m.K:g], ext


@torch.no_grad()
def self_force_v7(sampler, mi, fr, act, mot, colour, counts, steps, P):
    """Recovery practice, exactly as train_v6.self_force_v6, with the motion passed along (the
    practice dreams are always steered with the real keys + motion: no dropout here)."""
    for j in range(len(counts), 0, -1):
        rows = sum(counts[j - 1:])
        if rows <= 0:
            continue
        g = P - j
        ctx, a, ext = mi(fr[:rows], act[:rows], mot[:rows], g)
        fr[:rows, mi.row(g)] = sampler(ctx, a, steps, colour[:rows], ext)
    return fr


# ---------------------------------------------------------------- dreaming with (optional) guidance

def shift_frames(x, shift_px):
    """Slide frames by shift_px [B, 2] pixels (float): column 0 = sideways (+ = content moves LEFT = the
    camera turned RIGHT), column 1 = up/down (+ = content moves DOWN = the camera tilted UP). The uncovered
    edge is filled with the edge pixels. Used by the experimental 'shift' start (see guided_sample)."""
    B, C, H, W = x.shape
    # WHOLE PIXELS ONLY (fixed 2026-10-21 00:2x): a fractional slide is a bilinear blur, and the dream
    # slides every frame (even "walk forward" drifts ~0.09 px), so the blur compounded: after 100 frames
    # only ~2% of the fine detail was left and straight walks "melted" at ~12 s. Rounding makes every
    # slide an exact copy; tiny drifts round to 0 (= the plain previous-frame start), turns still slide
    # ~5 px (~2.4 deg) per step.
    shift_px = torch.round(shift_px.float())
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift_px[:, 0].float() / W     # sample from x + dx: content moves left by dx
    theta[:, 1, 2] = -2.0 * shift_px[:, 1].float() / H    # sample from y - dy: content moves down by dy
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    out = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    still = (shift_px == 0).all(dim=1)                    # rows that don't slide: return them untouched
    return torch.where(still[:, None, None, None], x.float(), out)


def motion_shift_px(mrow, motion_scale, fpx):
    """Normalised motion rows [B, 4] of the step being dreamed -> pixel slide [B, 2] for the 'shift' start
    (the labeler's own rule backwards: a turn of yaw degrees slides the picture f * tan(yaw) pixels)."""
    yaw = mrow[:, 0].float() * motion_scale[0]
    pitch = mrow[:, 1].float() * motion_scale[1]
    return torch.stack([fpx * torch.tan(torch.deg2rad(yaw)), fpx * torch.tan(torch.deg2rad(pitch))], dim=1)


@torch.no_grad()
def guided_sample(model, ctx, a, steps, colour, ext, w=0.0, generator=None, start="prev", sigma_max=5.0,
                  shift_px=None, ghost=None):
    """Dream one frame per row. w = 0: EXACTLY model.sample (same noise use), so the parent is scored by
    the same code as v6 did. w > 0: every clean-up call runs ONE batch of two (with steering, without),
    den = den(steer) + w * (den(steer) - den(null)).
    start = "prev" (as trained), "noise" (pure static, the pre-2026-10-20 way) or "shift": the previous
    frame SLID by the asked turn (yaw -> sideways, pitch -> up/down; shift_px [B, 2]) before the static is
    added. Why it works (found 2026-10-20 while building v7): at heavy noise the model mostly copies the
    faint 'ghost' of where it starts from (that is why the prev start was such a big fix), so a ghost that
    already looks turned gets dreamed as a turned view. v6 at step 500 turned ~3 deg of 144 asked with
    'prev' and ~170 with 'shift', with NO training; but the strip it has to invent at the edge is smeary,
    because it never practised inventing it (train_v7's turn test logs both starts)."""
    if w == 0 and start == "prev" and ghost is None:
        return model.sample(ctx, a, steps=steps, generator=generator, start="prev", colour=colour,
                            far_act=ext, sigma_max=sigma_max).float()
    B, _, C, H, W = ctx.shape
    dev = ctx.device
    sigmas = model.karras_sigmas(steps, sigma_max=sigma_max, device=dev)
    x = torch.randn((B, C, H, W), device=dev, generator=generator) * sigmas[0]
    base = ctx[:, -1].float()
    if start == "shift" and shift_px is not None:
        base = shift_frames(base, shift_px)
    if ghost is not None:                     # panorama memory (play.py): start from exactly this image
        x = x + ghost.float()
    elif start in ("prev", "shift"):
        x = x + base
    col = colour if colour is not None else torch.zeros(B, 25, device=dev)
    if w != 0:
        a0, e0 = model.null_of(a, ext)
        ctx2, a2, col2, e2 = torch.cat([ctx, ctx]), torch.cat([a, a0]), torch.cat([col, col]), torch.cat([ext, e0])
    for i in range(steps):
        s, s_next = sigmas[i], sigmas[i + 1]
        if w == 0:
            den, _ = model(x, s.expand(B), ctx, 0.0, a, col, ext)
        else:
            d2, _ = model(torch.cat([x, x]), s.expand(2 * B), ctx2, 0.0, a2, col2, e2)
            den_s, den_n = d2.float().chunk(2)
            den = den_s + w * (den_s - den_n)
        x = x + (x - den.float()) / s * (s_next - s)
    return x.clamp(-1, 1)


class GuidedGraphedSampler:
    """play.py's speed-up for GUIDED dreams: GraphedSampler (train_v3.py) records the network call for a
    batch of two (row 0 = with your keys/motion, row 1 = no steering) once, and every clean-up step
    replays that recording, then mixes  den = den(steer) + w * (den(steer) - den(null)).
    Shares the GraphedSampler's recordings dict (batch 1 = the plain dream, batch 2 = this one)."""

    def __init__(self, graphed):
        self.gs = graphed

    @torch.no_grad()
    def sample(self, ctx, a, steps, colour, far, w, sigma_max=5.0, start="prev", shift_px=None, ghost=None):
        """w = 0: the plain dream through the batch-1 recording; w > 0: the batch-of-two recording.
        start: prev / noise / shift (shift_px [B, 2], see guided_sample).
        ghost [B, 3, H, W]: start from exactly this image + static instead (play.py's panorama memory;
        2026-10-21). None = the start above, unchanged."""
        m = self.gs.model
        B = ctx.shape[0]
        if w == 0:
            ctx2, a2, c2, f2 = ctx, a, colour, far
        else:
            a0, f0 = m.null_of(a, far)
            ctx2, a2, c2, f2 = torch.cat([ctx, ctx]), torch.cat([a, a0]), torch.cat([colour, colour]), torch.cat([far, f0])
        nb = ctx2.shape[0]
        if nb not in self.gs.graphs:
            self.gs.graphs[nb] = self.gs._build(ctx2, a2, c2, f2)
        g, st, out = self.gs.graphs[nb]
        st["ctx"].copy_(ctx2)
        st["a"].copy_(a2)
        st["col"].copy_(c2)
        st["far"].copy_(f2)
        sigmas = m.karras_sigmas(steps, sigma_max=sigma_max, device=ctx.device)
        x = torch.randn_like(ctx[:, -1].float()) * sigmas[0]
        if ghost is not None:
            x = x + ghost.float()
        elif start == "prev":
            x = x + ctx[:, -1].float()
        elif start == "shift":
            x = x + shift_frames(ctx[:, -1].float(), shift_px)
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            st["x"].copy_(x if w == 0 else torch.cat([x, x]))
            st["s"].fill_(s)
            g.replay()
            if w == 0:
                den = out
            else:
                den_s, den_n = out.chunk(2)
                den = den_s + w * (den_s - den_n)
            x = x + (x - den) / s * (s_next - s)
        return x.clamp(-1, 1)


@torch.no_grad()
def dream_walks_v7(model, frames_mm, actions_np, motion_mm, row_starts, keys, steps, device, table,
                   anchor=0.0, seed=1234, start="prev", use_colour=True, w=0.0, motion_scale=None):
    """train_v6.dream_walks_v6 for v7: one walk per row, all in ONE batch (+ an anchored twin of row 0).
    The real history before each start brings its REAL keys and REAL motion; every dreamed step presses
    keys[t] and gets the motion table[key] (data_v7.key_motion_table). Fixed-seed noise (fair from
    rollout to rollout). w = guidance strength. Returns uint8 [rows, T, H, W, 3]."""
    K, Hn = model.K, model.history
    rows_keys = [list(k) for k in keys] if isinstance(keys[0], (list, tuple)) else [list(keys)] * len(row_starts)
    starts = list(row_starts)
    if anchor > 0:
        starts.append(starts[0])
        rows_keys.append(rows_keys[0])
    T = len(rows_keys[0])
    B = len(starts)
    tab = torch.as_tensor(table, dtype=torch.float32, device=device)
    # full key + motion sequences per row: Hn - 1 real steps, then the T pressed ones
    kseq = torch.tensor([[int(a) if a >= 0 else 1 for a in actions_np[s - Hn:s - 1]] + rk
                         for s, rk in zip(starts, rows_keys)], device=device)                  # [B, Hn-1+T]
    mseq = torch.cat([torch.stack([torch.from_numpy(np.asarray(motion_mm[s - Hn:s - 1], np.float32))
                                   for s in starts]).to(device),
                      tab[kseq[:, Hn - 1:].clamp(0, N_ACTIONS - 1)]], dim=1)                   # [B, Hn-1+T, 4]
    hist = torch.empty((B, Hn, 3, *frames_mm.shape[1:3]), device=device)
    for i, s in enumerate(starts):
        hist[i] = to_model_space(torch.from_numpy(np.array(frames_mm[s - Hn:s])).to(device))
    head = 0
    f0 = Hn - model.fp_offset
    colour = colour_fingerprint(hist[:, f0:f0 + FP_FRAMES]) if use_colour else None
    ref_mean = hist[-1:, -K:].mean(dim=(1, 3, 4))[:, :, None, None]
    ref_std = hist[-1:, -K:].std(dim=(1, 3, 4))[:, :, None, None]
    offs = torch.tensor(model.mem_offsets(), device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    fpx = (frames_mm.shape[2] / 2) / math.tan(math.radians(35))       # the labeler's focal length, in pixels
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(T):
            kk = kseq[:, t:t + Hn]                                         # last Hn keys, the one pressed now last
            ext = model.extras(kk, mseq[:, t + Hn - K:t + Hn])
            idx = (head - offs) % Hn
            shift = None
            if start == "shift":
                shift = motion_shift_px(mseq[:, t + Hn - 1], motion_scale, fpx)
            nxt = guided_sample(model, hist.index_select(1, idx), kk[:, -K:], steps, colour, ext, w=w,
                                generator=g, start=start, shift_px=shift).float()
            if anchor > 0:
                x = nxt[-1:]
                m = x.mean(dim=(2, 3), keepdim=True)
                sd = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
                fixed = (x - m) / sd * ref_std + ref_mean
                nxt = torch.cat([nxt[:-1], ((1 - anchor) * x + anchor * fixed).clamp(-1, 1)])
            out.append(((nxt.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            hist[:, head] = nxt
            head = (head + 1) % Hn
    return np.stack(out, 1)


def turn_keys_(walk, n):
    return [1] * walk + [6] * n + [5] * n


def ctrl_keys_(walk, n):
    return [1] * walk + [6] * n + [1] * n


def run_turn_test_v7(model, frames_mm, actions_np, motion_mm, starts, seeds, steps, device, table, sd,
                     walk=TURN_WALK, n=TURN_STEPS, w=0.0, with_ctrl=True, turn_deg=TEST_TURN_DEG,
                     start="prev", motion_scale=None):
    """turn_test.run_turn_test for v7 (keys -> motion table, optional guidance w).
    with_ctrl=False dreams only the turning rows (half the work; then 'memory'/'ctrl' are not reported).
    Adds turned_frac = turned_right_deg / asked_deg. Returns (turn dreams, control dreams or None, scores)."""
    rs = [s for _ in range(seeds) for s in starts]
    keys = [turn_keys_(walk, n)] * len(rs) + ([ctrl_keys_(walk, n)] * len(rs) if with_ctrl else [])
    d = dream_walks_v7(model, frames_mm, actions_np, motion_mm, rs + (rs if with_ctrl else []), keys, steps,
                       device, table, w=w, start=start, motion_scale=motion_scale)
    k = len(rs)
    td, cd = d[:k], (d[k:] if with_ctrl else None)
    r = turn_scores(td, cd if with_ctrl else td, len(starts), scene_features, sd, walk, n)
    r["asked_deg"] = round(n * turn_deg, 1)
    r["turned_frac"] = round(r["turned_right_deg"] / max(r["asked_deg"], 1e-6), 3)
    r["guidance"] = w
    if not with_ctrl:
        for kk in [k_ for k_ in r if k_.startswith(("ctrl_sim", "memory"))]:
            r.pop(kk)
    return td, cd, r


def turn_line_v7(name, r):
    s = (f"{name}: asked ~{r['asked_deg']:.0f} deg right, the dream turned ~{r['turned_right_deg']:.0f} right "
         f"(TURNED FRACTION {r['turned_frac']:.2f}) and ~{r['turned_back_deg']:.0f} back; return_sim {r['return_sim']}"
         f" (1 = same view as before the turn, 0 = like another city), facing away {r['away_sim']}")
    if "memory" in r:
        s += f", turned-away control {r['ctrl_sim']}, memory {r['memory']:+.3f}"
    return s


def steer_score(frac_w0, clean2_median, ruler_secs):
    """best_steer.pt's score: turns as asked (capped at 1, negative = 0) x stays clean (share of the ruler)."""
    return round(max(0.0, min(1.0, frac_w0)) * (clean2_median / ruler_secs), 4)


def turn_clip(path, td, real, title):
    """A short mp4 of one dreamed turn (row 0) next to the real start frame (for the evolution video)."""
    keys = turn_keys_(TURN_WALK, (td.shape[0] - TURN_WALK) // 2)        # td = ONE dream [T, H, W, 3]
    keys += [5] * (td.shape[0] - len(keys))
    write_mp4(path, td, keys, real, title, np.zeros(td.shape[0]))


# ---------------------------------------------------------------- loading any save as a v7 network

def load_as_v7(path, device, which="ema"):
    """A v5, v6 or v7 save -> (v7 model in eval mode, step, version). Parents are wrapped exactly
    (warm_start_v7: at init v7 == parent), so all of them are scored by the same dreaming code."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["model_cfg"]
    if cfg.get("version") == "v7":
        m = build_model_v7(cfg)
        m.load_state_dict(ck[which])
    else:
        p, _, _ = parent_model(cfg, ck[which])
        m = build_model_v7(v7_cfg_from_parent(cfg))
        warm_start_v7(m, p.state_dict())
    return m.to(device).eval(), ck["step"], cfg.get("version")


def eval_main(args):
    """--eval CKPT: the turn test at w = 0 (with controls) and w = GUIDE_TEST_W, optionally the ruler too."""
    device = "cuda"
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    _, fm, act, _ = load_cache_any(cache_dir)
    mot, minfo = load_motion(cache_dir)
    tmp = HERE / "_exp" / "v7_eval_ck.pt"                    # copy first: never hold a live run's file open
    shutil.copy(args.eval, tmp)
    m, step, ver = load_as_v7(tmp, device)
    tmp.unlink()
    sd = np.load(SCENE_FIT)["sd"]
    starts = [int(x) for x in args.ruler_starts.split(",")]
    table = key_motion_table(minfo, args.test_turn_deg)
    print(f"{args.eval}: {ver} step {step}; turn test {args.turn_steps} steps at {args.test_turn_deg} deg/step")
    for w in [float(x) for x in args.eval_w.split(",")]:
        for st in args.eval_starts.split(","):
            t0 = time.time()
            td, cd, r = run_turn_test_v7(m, fm, act, mot, starts, args.ruler_seeds, args.rollout_steps, device, table,
                                         sd, args.turn_walk, args.turn_steps, w=w, with_ctrl=(w == 0 and st == "prev"),
                                         turn_deg=args.test_turn_deg, start=st, motion_scale=minfo["scale"])
            print(turn_line_v7(f"  w={w} start={st} ({time.time() - t0:.0f}s)", r), flush=True)
            if args.eval_picture:
                turn_picture(pathlib.Path(args.eval_picture.replace(".jpg", f"_w{w:g}_{st}.jpg")), td,
                             cd if cd is not None else td, len(starts), f"{ver} {step} w={w:g} {st}",
                             args.turn_walk, args.turn_steps)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v7", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-v6", default="", help="v6 checkpoint to warm-start a NEW run from (primary)")
    ap.add_argument("--init-v5", default="", help="v5 checkpoint (model_v3) to warm-start from instead (fallback)")
    ap.add_argument("--eval", default="", help="just score this save (v5/v6/v7) on the turn test, then exit")
    ap.add_argument("--eval-w", default="0,2", help="--eval: guidance strengths")
    ap.add_argument("--eval-starts", default="prev", help="--eval: 'prev' and/or 'shift' (experiment)")
    ap.add_argument("--eval-picture", default="", help="--eval: picture path (.jpg), one per w")
    ap.add_argument("--smoke", action="store_true",
                    help="checkpoints/out under train/_smoke/v7/, live view in train/_smoke/live_v7/")
    ap.add_argument("--cache", default=None, help="default data/cache_v3 (needs motion.npy: data_v7.py --build)")
    ap.add_argument("--extra-blocks", type=int, default=1, help="only for --init-v5: v6's extra blocks per level")
    ap.add_argument("--colour-drop", type=float, default=0.1)
    # v7 steering
    ap.add_argument("--steer-drop", type=float, default=0.1, help="share of examples with NO steering (keys -> null, motion -> 0)")
    ap.add_argument("--onset-weight", type=float, default=4.0, help="draw turn-onset windows this many times more often")
    ap.add_argument("--bigturn-weight", type=float, default=2.0, help="draw big-turn windows this many times more often")
    ap.add_argument("--onset-deg", type=float, default=1.5, help="turn onset: |yaw| of the dreamed step at least this (deg/step) ...")
    ap.add_argument("--onset-ratio", type=float, default=0.6,
                    help="... while the 4 steps before turned on average at most this share of it (the labeler's yaw is "
                         "smoothed, so turns ramp up; 'the 4 before were all below 0.75 deg' matched only 130 steps in 14 h)")
    ap.add_argument("--bigturn-deg", type=float, default=3.0, help="big turn: |yaw| of the dreamed step at least this")
    ap.add_argument("--test-turn-deg", type=float, default=TEST_TURN_DEG,
                    help="deg/step a turn key asks for in the ruler + turn test (2.4 = the data's average turn = v6's asked 144 deg)")
    ap.add_argument("--guide-w", type=float, default=GUIDE_TEST_W, help="guidance of the second turn test")
    ap.add_argument("--ghost-frac", type=float, default=0.0,
                    help="share of examples trained as 'the dream's first step': heavy noise around the PREVIOUS frame "
                         "(not the target), answer = the target, so only the keys/motion can say how the view moved "
                         "(model_v7.loss_ghost). 0 = off (v6's recipe)")
    ap.add_argument("--ghost-sigma-min", type=float, default=4.0,
                    help="ghost rows: noise drawn log-uniform in [min, max]. Only the FIRST dream step starts around the "
                         "previous frame; later steps are centred on the guess, so keep the band near sigma_max "
                         "(sanity 2026-10-20: [2, 5] steered but cost clean2 20 -> 7 s in 500 steps)")
    ap.add_argument("--ghost-sigma-max", type=float, default=5.0, help="(5.0 = where every dream starts)")
    ap.add_argument("--ghost-shift-p", type=float, default=0.5,
                    help="share of ghost rows whose previous frame is first slid by the REAL turn of that step "
                         "(practice for the 'shift' dream start: invent the uncovered strip)")
    # batch / optimiser
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--new-lr-mult", type=float, default=1.0, help="learning-rate multiplier for v7's NEW weights (motion net, null keys)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--nonfwd-weight", type=float, default=2.0)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--source-weights", default="", help='e.g. "Wildlife=0.1"')
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
    ap.add_argument("--sf-start-step", type=int, default=0)
    ap.add_argument("--sf-walk-only", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--cn-mean", type=float, default=-1.8)
    ap.add_argument("--cn-std", type=float, default=1.1)
    ap.add_argument("--cn-max", type=float, default=1.5)
    ap.add_argument("--cn-zero", type=float, default=0.1)
    ap.add_argument("--cn-perframe", type=float, default=0.5)
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--first-sample-step", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=20)
    ap.add_argument("--keep-every-hours", type=float, default=1)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--ruler-seeds", type=int, default=2)
    ap.add_argument("--ruler-starts", default=V5_RULER_STARTS)
    ap.add_argument("--anchor", type=float, default=0.15)
    ap.add_argument("--best-by", choices=["melt", "clean", "clean2"], default="clean2")
    ap.add_argument("--ripple-max", type=float, default=RIPPLE_LEVEL)
    ap.add_argument("--dream-start", choices=["noise", "prev", "shift"], default="prev",
                    help="ruler + turn tests: prev (as v5/v6), or shift (prev frame slid by the asked turn; experiment)")
    ap.add_argument("--no-turn-test-shift", action="store_true",
                    help="skip the extra turn test with the 'shift' start (w = 0, turning rows only) in rollouts")
    ap.add_argument("--turn-steps", type=int, default=TURN_STEPS)
    ap.add_argument("--turn-walk", type=int, default=TURN_WALK)
    ap.add_argument("--no-turn-test", action="store_true")
    ap.add_argument("--decay-hours", type=float, default=8)
    ap.add_argument("--lr-min-frac", type=float, default=0.1)
    ap.add_argument("--no-baseline", action="store_true", help="skip scoring the warm start (= the parent) at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0, help="hard cap for this process (tests next to a live run)")
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--ruler-secs", type=int, default=40)
    args = ap.parse_args()
    if args.eval:
        return eval_main(args)

    base = HERE / "_smoke" / "v7" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "live_v7" if args.smoke else HERE / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    elif not args.allow_low_vram:
        # Room check, as train_v6 (v6a measured 18.8 GB at batch 48). v7 adds a tiny network and,
        # in rollouts only, a guided turn test that runs 10 rows as one batch of 20 (= v6's turn test size).
        free = gpu_free_gb()
        need = 0.30 * args.batch + 7.0
        if free is not None and free < need + 2.5:
            sys.exit(f"Only {free:.1f} GB of GPU memory is free, and batch {args.batch} needs about {need:.0f} GB.\n"
                     f"Is v6 (or a play window) still running? Stop it first (Ctrl+C in its window; it saves), "
                     f"or use a smaller --batch, or --allow-low-vram to start anyway.")
    torch.manual_seed(args.seed)

    latest = ckpt_dir / "latest.pt"
    ck = None
    if args.resume:
        if not latest.exists():
            sys.exit(f"--resume: no checkpoint at {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
    elif latest.exists():
        sys.exit(f"{latest} already exists. Use --resume to continue it, or pick another --run name.")

    # ---- data, motion, model config
    index, frames_mm, actions_np, cut_np = load_cache_any(cache_dir)
    motion_mm, minfo = load_motion(cache_dir)
    if ck and ck.get("motion_info") and ck["motion_info"]["scale"] != minfo["scale"]:
        sys.exit("motion.npy was rebuilt with different scales since this run started: the model's motion "
                 "input would change meaning. Rebuild it as before, or start a new run.")
    frame_hw = list(index["shape"][1:3])
    parent_ck, parent_path = None, None
    if ck:
        model_cfg = ck["model_cfg"]
    else:
        parent_path = args.init_v6 or args.init_v5
        if not parent_path:
            sys.exit("A new v7 run needs --init-v6 <v6 save> (or --init-v5 <v5 save>).")
        parent_ck = torch.load(parent_path, map_location="cpu", weights_only=False)
        pv = parent_ck["model_cfg"].get("version")
        if (args.init_v6 and pv != "v6") or (not args.init_v6 and pv != "v3"):
            sys.exit(f"{parent_path} is a '{pv}' save; --init-v6 wants a v6 save, --init-v5 a v5 (model_v3) save")
        model_cfg = v7_cfg_from_parent(parent_ck["model_cfg"], args.extra_blocks)
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
    ruler_script, t_ = [], 0
    while t_ < args.ruler_secs * 10:
        for a_, n_ in FAIR_SCRIPT:
            n_ = min(n_, args.ruler_secs * 10 - t_)
            if n_ > 0:
                ruler_script.append((a_, n_)); t_ += n_
    T = sum(n for _, n in ruler_script)
    ruler_keys = [a for a, n in ruler_script for _ in range(n)]
    table = key_motion_table(minfo, args.test_turn_deg)         # keys -> motion for the ruler + turn test
    yaw_scale = float(minfo["scale"][0])

    valid = compute_valid(actions_np, cut_np, index["cities"], L)
    train_ok = valid.copy()
    held_ok = np.zeros_like(valid)
    bad_tr = (actions_np < 0) | cut_np
    holds = {}
    for c in index["cities"]:
        n = c["frames"]
        hold_n = min(max(int(args.holdout * n), Hn + T + 40), n // 3)
        h0 = c["end"] - hold_n
        holds[c["name"]] = h0
        train_ok[max(c["start"], h0 - L):c["end"]] = False
        held_ok[h0:c["end"]] = valid[h0:c["end"]]
    if train_ok.sum() == 0:
        sys.exit("No valid training windows.")

    starts = list(ck["ruler_starts"]) if ck else [int(x) for x in args.ruler_starts.split(",") if x.strip()]
    ruler = [next(c for c in index["cities"] if c["start"] <= s < c["end"]) for s in starts]
    ruler_notes = []
    for s, c in zip(starts, ruler):
        if s - Hn < c["start"] or s + T > c["end"]:
            sys.exit(f"ruler start {s} ({c['name']}) has no room for {Hn} frames of history + {T} frames of walk")
        if bad_tr[s - Hn:s - 1].any():
            ruler_notes.append(f"WARNING: ruler walk {s} ({c['name']}) has a cut/unknown key in its {Hn}-frame history")
    row_starts = [s for _ in range(args.ruler_seeds) for s in starts]
    real_clip = np.array(frames_mm[starts[0]:starts[0] + T])
    scene_sd = np.load(SCENE_FIT)["sd"]

    # ---- sampling weights: v6's (source shares x2 for non-W keys) x the v7 turn practice
    src_w = parse_source_weights(args.source_weights)
    usable = {c["name"]: int(train_ok[c["start"]:c["end"]].sum()) for c in index["cities"]}
    shares = source_shares(index["cities"], usable, src_w, args.city_balance)
    N = len(actions_np)
    target_key = np.full(N, -1, np.int64)
    target_key[:N - L + 1] = actions_np[L - 1:]
    # |yaw| (degrees) of the step being dreamed (window step L-1) and the MEAN |yaw| of the 4 steps before it
    # (what the newest memory frames show). A turn ONSET = a clear turn now, while the memory shows at most
    # --onset-ratio of that turning speed (the turn is starting / speeding up: only the key/motion can say so).
    yaw_deg = np.abs(np.asarray(motion_mm[:, 0], np.float32)) * yaw_scale
    tyaw = np.zeros(N, np.float32)
    tyaw[:N - L + 1] = yaw_deg[L - 1:]
    prev4 = np.zeros(N, np.float32)
    for j in range(2, 6):                                    # window steps L-2 .. L-5
        prev4[:N - L + 1] += yaw_deg[L - j:N - j + 1] / 4
    onset = (tyaw >= args.onset_deg) & (prev4 <= args.onset_ratio * tyaw)
    bigturn = tyaw >= args.bigturn_deg
    anyturn = tyaw >= args.onset_deg

    def per_city_norm(w):
        w = w.copy()
        for c in index["cities"]:
            seg = w[c["start"]:c["end"]]
            if seg.sum() > 0:
                seg *= shares[c["name"]] / seg.sum()
        return w
    w_v6 = per_city_norm(train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight))
    boost = np.where(onset, args.onset_weight, 1.0) * np.where(bigturn, args.bigturn_weight, 1.0)
    weights = per_city_norm(train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight) * boost)
    share = lambda w, m: float(w[m].sum() / w.sum())
    turn_share_note = (f"turn practice: share of training draws (v6 weights -> v7 weights): turn ONSET windows "
                       f"(dreamed step |yaw| >= {args.onset_deg} deg/step, previous 4 on average <= {args.onset_ratio} of it) "
                       f"{share(w_v6, onset):.1%} -> {share(weights, onset):.1%}; BIG turns (>= {args.bigturn_deg}) "
                       f"{share(w_v6, bigturn):.1%} -> {share(weights, bigturn):.1%}; any turn (>= {args.onset_deg}) "
                       f"{share(w_v6, anyturn):.1%} -> {share(weights, anyturn):.1%} "
                       f"(windows: {int((onset & train_ok).sum())} onset, {int((bigturn & train_ok).sum())} big)")
    del yaw_deg, tyaw, prev4

    # ---- model, EMA, optimizer (+ the warm start)
    n_params = sum(p.numel() for p in model.parameters())
    init_notes, parent = [], None
    if parent_ck is not None:
        parent, notes5, how5 = parent_model(parent_ck["model_cfg"], parent_ck["model"], args.extra_blocks)
        init_notes = (notes5 or []) + warm_start_v7(model, parent.state_dict())
        init_notes.insert(0, f"from {parent_path}: {parent_ck['model_cfg'].get('version')} run {parent_ck['run_id']} "
                             f"step {parent_ck['step']} ({parent_ck['params'] / 1e6:.2f}M params) -> v7 {n_params / 1e6:.3f}M")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if parent_ck is not None:
        parent, _, _ = parent_model(parent_ck["model_cfg"], parent_ck["ema"], args.extra_blocks)
        warm_start_v7(ema, parent.state_dict())                 # the EMA weights (what dreams) get the parent's EMA
        parent = parent.to(device).eval()                       # kept for the warm-start check

    new_names = {"unet.null_act_emb"} | {n for n, _ in model.named_parameters() if "motion_mlp" in n}

    def make_opt(m):
        decay, no_decay, new = [], [], []
        for name, p in m.named_parameters():
            if name in new_names and args.new_lr_mult != 1.0:
                new.append(p)
            else:
                (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
        groups = [{"params": decay, "weight_decay": args.wd, "mult": 1.0},
                  {"params": no_decay, "weight_decay": 0.0, "mult": 1.0}]
        if new:
            groups.append({"params": new, "weight_decay": 0.0, "mult": args.new_lr_mult})
        return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), fused=True)
    opt = make_opt(model)
    if parent_ck is not None and "opt" in parent_ck:
        try:
            # the parent's optimizer listed its weights in its trainer's order (same grouping rule)
            pv = parent_ck["model_cfg"].get("version")
            shape = parent if pv == "v6" else build_model_v3(parent_ck["model_cfg"])
            names_p = [n for n, p in shape.named_parameters() if not (p.ndim < 2 or "emb" in n)] + \
                      [n for n, p in shape.named_parameters() if p.ndim < 2 or "emb" in n]
            how = {n: ("same",) for n in names_p} if pv == "v6" else how5
            n_carried = warm_start_opt_state(opt, model, parent_ck["opt"], names_p, how)
            init_notes.append(f"Adam memory carried over from the parent for {n_carried} of "
                              f"{sum(1 for _ in model.parameters())} weight tensors (the new ones start fresh)")
        except Exception:
            opt = make_opt(model)
            init_notes.append("could not carry the parent's Adam memory over, starting it fresh:\n" + traceback.format_exc())

    step, train_seconds = 0, 0.0
    run_id = f"{args.tag}-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline, best, best_steer, warm_check = None, None, None, None
    if ck:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline, best, warm_check = ck.get("parent_baseline"), ck.get("best"), ck.get("warm_check")
        best_steer = ck.get("best_steer")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    names = ", ".join(c["name"] for c in index["cities"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": f"dream model {args.tag}",
        "prompt": f"dream {args.tag} ({names}), {n_params / 1e6:.0f}M params, {frame_hw[0]}x{frame_hw[1]}, "
                  f"memory {model.n_mem} frames back to {Hn / 10:.1f} s, steered by keys + camera motion",
        "pose": pose_string(ruler_script), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "melt_median": None, "melt_mean": None, "streak": None,
        "return_sim": None, "away_sim": None, "ctrl_sim": None, "turn_memory": None, "scene": None,
        "turned_deg": None, "turned_frac_w0": None, "turned_frac_w2": None, "turned_frac_shift": None, "steer_score": None,
        "parent_turned_frac_w0": baseline["turn_w0"]["turned_frac"] if baseline and baseline.get("turn_w0") else None,
        "parent_clean2_median": baseline["clean2_median"] if baseline else None,
        "best_step": best["step"] if best else None, "best_steer_step": best_steer["step"] if best_steer else None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v7 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.3f}M params; {frame_hw[0]}x{frame_hw[1]}; memory {model.n_mem} frames ({Hn / 10:.1f} s); "
             f"window {L + 1} frames, {len(positions)} read; batch {args.batch}; {int(train_ok.sum())} train windows, "
             f"{int(held_ok.sum())} held-out; cache {cache_dir}")
    live.log("sources (share of training draws): " + ", ".join(
        f"{c['name']} {c['frames'] / 36000:.2f} h -> {shares[c['name']]:.1%}"
        + (f" (weight {src_w[c['name']]})" if c["name"] in src_w else "") for c in index["cities"]))
    live.log(turn_share_note)
    live.log(f"motion input: yaw/pitch/fwd/strafe scaled by {[round(x, 3) for x in minfo['scale']]} (95th pct of |value|), "
             f"clipped +-{minfo['clip']}; corr(yaw, turn key) {minfo['corr_yaw_vs_turnkey']:.3f}. Steering dropout "
             f"{args.steer_drop:.0%}; new weights' lr x{args.new_lr_mult}; first-dream-step (ghost) practice "
             f"{args.ghost_frac:.0%} of examples (noise {args.ghost_sigma_min}-{args.ghost_sigma_max}, {args.ghost_shift_p:.0%} slid)")
    live.log(f"key -> motion table for the ruler + turn test (turn = {args.test_turn_deg} deg/step): " + describe_table(minfo, args.test_turn_deg))
    for n in ruler_notes:
        live.log(n)
    counts = sf_counts(int(round(args.sf_frac * args.batch)), sf_w)
    live.log(f"recovery practice: {sum(counts)} of {args.batch} examples per batch; with 1/2/3/4 dreamed "
             f"frames: {counts}; {args.sf_steps} sampler steps, start '{args.sf_start}'"
             f"{', walking cities only' if args.sf_walk_only else ''}. Dreams start '{args.dream_start}'. "
             f"best.pt chases median {args.best_by} time; best_steer.pt chases the steer score.")
    live.log(f"fair ruler: {len(starts)} walks ({', '.join(c['name'] for c in ruler)}) x {args.ruler_seeds} noise draws, "
             f"{T / 10:.0f} s each; turn test: W x {args.turn_walk}, right x {args.turn_steps}, left x {args.turn_steps} "
             f"at {args.test_turn_deg} deg/step, guidance 0 (with controls) and {args.guide_w} (turn rows only)")
    for n in init_notes:
        live.log("warm start: " + n)
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
    prev_row = int(lut[L - 1])                             # the frame just before the target (ghost practice)
    fpx_train = (frame_hw[1] / 2) / math.tan(math.radians(35))

    held_idx = np.flatnonzero(held_ok)
    rs = np.random.default_rng(0)
    pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
    val_ds = SparseWindowDatasetV7(cache_dir, positions, L, flip_p=0.0)
    vb = [val_ds[int(t)] for t in pick]
    val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]), torch.stack([v[3] for v in vb]))

    def run_val(m=None):
        """EMA loss on 64 fixed held-out windows at 5 fixed noise levels (as v3-v6), steering present."""
        m = ema if m is None else m
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
                        tot += m.loss(fr[:, target_row], ctx, a, torch.full((b,), s, device=device), 0.05,
                                      col, ext).item() * b / B
        return tot / len(sig_list)

    def save_ckpt(path):
        try:
            tmp = str(path) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                        "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg,
                        "motion_info": minfo, "parent_baseline": baseline, "best": best, "best_steer": best_steer,
                        "warm_check": warm_check, "ruler_cities": [c["name"] for c in ruler], "ruler_starts": starts,
                        "warm_started": True}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def run_ruler(m, use_anchor):
        t0 = time.time()
        dreams = dream_walks_v7(m, frames_mm, actions_np, motion_mm, row_starts, ruler_keys, args.rollout_steps,
                                device, table, anchor=args.anchor if use_anchor else 0.0, start=args.dream_start,
                                motion_scale=minfo["scale"])
        n = len(row_starts)
        summ, reps = score_walks(dreams[:n], row_starts, frames_mm, K, T, args.ripple_max)
        summ["scene"] = scene_score(dreams[:n], row_starts, frames_mm, scene_sd)
        anc = None
        if use_anchor and args.anchor > 0:
            arep, acurve = melt_report(dreams[n], walk_sharp_ref(frames_mm, starts[0], T),
                                        np.array(frames_mm[starts[0] - K:starts[0]]))
            anc = (dreams[n], arep, acurve)
        return dreams, summ, reps, anc, time.time() - t0

    def run_turns(m, label, pic_name, clip_name=None):
        """Turn test at w = 0 (with controls), at w = guide_w (turning rows only) and, unless
        --no-turn-test-shift, with the 'shift' start at w = 0 (turning rows only). Logs all."""
        res = {}
        runs = [("turn_w0", 0.0, True, args.dream_start), ("turn_w2", args.guide_w, False, args.dream_start)]
        if not args.no_turn_test_shift and args.dream_start != "shift":
            runs.append(("turn_shift", 0.0, False, "shift"))
        for name, w, ctrl, st in runs:
            t0 = time.time()
            td, cd, r = run_turn_test_v7(m, frames_mm, actions_np, motion_mm, starts, args.ruler_seeds,
                                         args.rollout_steps, device, table, scene_sd, args.turn_walk, args.turn_steps,
                                         w=w, with_ctrl=ctrl, turn_deg=args.test_turn_deg, start=st,
                                         motion_scale=minfo["scale"])
            live.log(turn_line_v7(f"{label} turn test, guidance {w:g}, start {st} ({time.time() - t0:.0f}s)", r))
            res[name] = r
            if name == "turn_w0":
                turn_picture(out_dir / pic_name, td, cd, len(starts), label, args.turn_walk, args.turn_steps)
                if clip_name:
                    turn_clip(out_dir / clip_name, td[0], np.array(frames_mm[starts[0]:starts[0] + 1]), label)
        return res

    def do_rollout():
        """Ruler + turn tests on the EMA weights, videos, live frames, best.pt, best_steer.pt."""
        nonlocal best, best_steer
        try:
            dreams, summ, reps, anc, took = run_ruler(ema, True)
            write_live_frames(live, dreams[0], ruler_keys)
            path = write_mp4(out_dir / f"{args.tag}_rollout_step{step:07d}.mp4", dreams[0], ruler_keys, real_clip,
                             f"{args.tag} step {step}", reps[0][1])
            anc_txt = ""
            if anc is not None:
                write_mp4(out_dir / f"{args.tag}_anchor_rollout_step{step:07d}.mp4", anc[0], ruler_keys, real_clip,
                          f"{args.tag} {step} anchor {args.anchor}", anc[2])
                anc_txt = f"; anchored twin of walk 0: melt {secs(anc[1]['melt_s'])}"
            v = run_val()
            live.log(f"rollout at step {step}: {len(dreams)} dreams x {T} frames in one batch, "
                     f"{args.rollout_steps} sampler steps, in {took:.1f}s -> {path.name}; val loss {v:.4f}")
            live.log(ruler_line(f"{args.tag} step {step} (fair ruler)", summ, T)
                     + f"\n    scene score {summ['scene']} (real video ~0.78, lower = more photo-like)"
                     + (f"\n    [parent on the same ruler (= v7 at init): clean2 median {baseline['clean2_median']} s, "
                        f"mean {baseline['clean2_mean']} s; scene {baseline.get('scene')}]" if baseline else "") + anc_txt)
            turns = {}
            if not args.no_turn_test:
                turns = run_turns(ema, f"{args.tag} step {step}", f"{args.tag}_turn_step{step:07d}.jpg",
                                  f"{args.tag}_turnclip_step{step:07d}.mp4")
            t0r, t2r = turns.get("turn_w0"), turns.get("turn_w2")
            sc = steer_score(t0r["turned_frac"], summ["clean2_median"], T / 10) if t0r else None
            if t0r:
                live.log(f"    STEER SCORE {sc} = turned fraction at w=0 {t0r['turned_frac']} (capped 0..1) x clean2 "
                         f"{summ['clean2_median']} / {T / 10:.0f} s"
                         + (f"   [parent: turned fraction w=0 {baseline['turn_w0']['turned_frac']}, "
                            f"w={args.guide_w:g} {baseline['turn_w2']['turned_frac']}; steer score {baseline.get('steer_score')}]"
                            if baseline and baseline.get("turn_w0") else ""))
            key = args.best_by
            score = (summ[key + "_median"], summ[key + "_mean"], -v)
            is_best = (best is None or best.get("by", "melt") != key
                       or score > (best[key + "_median"], best[key + "_mean"], -best["val"]))
            if is_best:
                best = {"step": step, "by": key, "val": v, "melt_median": summ["melt_median"],
                        "melt_mean": summ["melt_mean"], "clean_median": summ["clean_median"],
                        "clean_mean": summ["clean_mean"], "clean2_median": summ["clean2_median"],
                        "clean2_mean": summ["clean2_mean"], "scene": summ["scene"], "steer_score": sc}
                save_ckpt(ckpt_dir / "best.pt")
                live.log(f"NEW BEST by {key} time (median {summ[key + '_median']} s, mean "
                         f"{summ[key + '_mean']} s) -> best.pt")
            if sc is not None and (best_steer is None or sc > best_steer["steer_score"]):
                best_steer = {"step": step, "steer_score": sc, "turned_frac_w0": t0r["turned_frac"],
                              "turned_frac_w2": t2r["turned_frac"] if t2r else None,
                              "clean2_median": summ["clean2_median"]}
                save_ckpt(ckpt_dir / "best_steer.pt")
                live.log(f"NEW BEST STEER SCORE {sc} -> best_steer.pt")
            live.metric(dict({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name,
                              "is_best": is_best, "anchor_melt_s": anc[1]["melt_s"] if anc else None,
                              "turn": t0r, "turn_w2": t2r, "turn_shift": turns.get("turn_shift"),
                              "steer_score": sc}, **summ))
            live.update(last_rollout_step=step, val_loss=v, melt_s=summ["melt_median"],
                        melt_median=summ["melt_median"], melt_mean=summ["melt_mean"], streak=summ["streak"],
                        clean_median=summ["clean_median"], clean2_median=summ["clean2_median"],
                        clean2_mean=summ["clean2_mean"], scene=summ["scene"],
                        return_sim=t0r["return_sim"] if t0r else None,
                        away_sim=t0r["away_sim"] if t0r else None,
                        ctrl_sim=t0r["ctrl_sim"] if t0r else None,
                        turn_memory=t0r["memory"] if t0r else None,
                        turned_deg=t0r["turned_right_deg"] if t0r else None,
                        turned_frac_w0=t0r["turned_frac"] if t0r else None,
                        turned_frac_w2=t2r["turned_frac"] if t2r else None,
                        turned_frac_shift=turns["turn_shift"]["turned_frac"] if turns.get("turn_shift") else None,
                        steer_score=sc, best_step=best["step"],
                        best_steer_step=best_steer["step"] if best_steer else None)
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    # ---- start-up checks on a NEW warm-started run
    if parent is not None:
        live.update(stage="warm-start check")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            fr = to_model_space(val_batch[0][:16].to(device))
            act = val_batch[1][:16].to(device)
            mot = val_batch[2][:16].to(device)
            B = fr.shape[0]
            ctx, a, ext = mi_ema(fr, act, mot, L)
            far = ext[:, :ema.n_keysum]                        # what the parent gets: just the key summaries
            col = colour_fingerprint(fr.index_select(1, fp_rows))
            tgt = fr[:, target_row]
            res = {}
            for s in [0.1, 0.5, 2.0]:
                g = torch.Generator(device=device).manual_seed(0)
                xn = tgt + s * torch.randn(tgt.shape, device=device, generator=g)
                sig = torch.full((B,), s, device=device)
                d7, _ = ema(xn, sig, ctx, 0.0, a, col, ext)
                dp, _ = parent(xn, sig, ctx, 0.0, a, col, far)
                res[f"denoise_sigma{s}_maxabs"] = (d7 - dp).abs().max().item()
            # the 'no steering' version at init = the parent pressing W in all 8 slots (null rows = W rows)
            a0, e0 = ema.null_of(a, ext)
            d7n, _ = ema(xn, sig, ctx, 0.0, a0, col, e0)
            dpn, _ = parent(xn, sig, ctx, 0.0, torch.ones_like(a), col, far)
            res["null_vs_parent_all_W_maxabs"] = (d7n - dpn).abs().max().item()
            p_raw, _, _ = parent_model(parent_ck["model_cfg"], parent_ck["model"], args.extra_blocks)
            p_raw = p_raw.to(device).eval()
            g = torch.Generator(device=device).manual_seed(0)
            xn = tgt + 0.5 * torch.randn(tgt.shape, device=device, generator=g)
            sig = torch.full((B,), 0.5, device=device)
            res["raw_weights_denoise_sigma0.5_maxabs"] = (model.eval()(xn, sig, ctx, 0.0, a, col, ext)[0]
                                                          - p_raw(xn, sig, ctx, 0.0, a, col, far)[0]).abs().max().item()
            model.train()
            del p_raw
            p7 = ema.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=ext, start="prev",
                            generator=torch.Generator(device=device).manual_seed(1))
            pp = parent.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=far, start="prev",
                               generator=torch.Generator(device=device).manual_seed(1))
            res["dream_prev_start_maxabs"] = (p7 - pp).abs().max().item()
            res["dream_mse_vs_real"] = torch.mean((pp - tgt) ** 2).item()
            res["copy_last_frame_mse"] = torch.mean((ctx[:, -1] - tgt) ** 2).item()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        del parent
        warm_check = res
        ok = res["dream_prev_start_maxabs"] < 1e-3 and res["denoise_sigma2.0_maxabs"] < 1e-3
        live.log(f"WARM-START CHECK (16 held-out moments, float32): max |v7 - parent| of one clean-up call (EMA) at noise "
                 f"0.1/0.5/2.0 = {res['denoise_sigma0.1_maxabs']:.2e}/{res['denoise_sigma0.5_maxabs']:.2e}/"
                 f"{res['denoise_sigma2.0_maxabs']:.2e}; raw weights at 0.5 = {res['raw_weights_denoise_sigma0.5_maxabs']:.2e}; "
                 f"'no steering' vs parent pressing W = {res['null_vs_parent_all_W_maxabs']:.2e}; "
                 f"a full {args.rollout_steps}-step prev-start dream = {res['dream_prev_start_maxabs']:.2e} (pixel range 2.0; "
                 f"the parent's own error vs the real frame: MSE {res['dream_mse_vs_real']:.4f}, copying the last frame "
                 f"{res['copy_last_frame_mse']:.4f}) -> {'OK: v7 starts out as its parent' if ok else 'WARNING: v7 differs from its parent'}")
        live.metric({"step": step, "time": time.time(), "warm_check": res})
        torch.cuda.empty_cache()
        if not args.no_baseline:
            live.update(stage="parent on the ruler + turn tests")
            dreams, summ, reps, _, took = run_ruler(ema, False)
            write_mp4(out_dir / f"{args.tag}_baseline_parent_step{parent_ck['step']:07d}.mp4", dreams[0], ruler_keys,
                      real_clip, f"parent {parent_ck['step']}", reps[0][1])
            baseline = dict(summ, parent_step=parent_ck["step"], parent_version=parent_ck["model_cfg"].get("version"))
            live.log(ruler_line(f"parent BASELINE (= v7 at init; step {parent_ck['step']}, {took:.0f}s)", summ, T)
                     + f"\n    scene score {summ['scene']}")
            if not args.no_turn_test:
                baseline.update(run_turns(ema, "parent BASELINE", f"{args.tag}_turn_baseline_parent.jpg",
                                          f"{args.tag}_turnclip_baseline_parent.mp4"))
                baseline["steer_score"] = steer_score(baseline["turn_w0"]["turned_frac"], summ["clean2_median"], T / 10)
                live.log(f"    parent steer score {baseline['steer_score']}")
            live.metric({"step": step, "time": time.time(), "parent_baseline": baseline})
            live.update(parent_clean2_median=baseline["clean2_median"],
                        parent_turned_frac_w0=baseline["turn_w0"]["turned_frac"] if baseline.get("turn_w0") else None)
            torch.cuda.empty_cache()
    del parent_ck

    # ---- the loop
    if args.no_graphs:
        sf_sampler = eager_sample(ema, start=args.sf_start)
    else:
        graphed = GraphedSampler(ema, start=args.sf_start)     # far_act = the longer v7 vector: works as-is
        sf_sampler = graphed.sample
        try:
            fr = to_model_space(val_batch[0][:max(1, counts[0])].to(device))
            act = val_batch[1][:fr.shape[0]].to(device)
            mot = val_batch[2][:fr.shape[0]].to(device)
            ctx, a, ext = mi_ema(fr, act, mot, L)
            diff = graphed.check(ctx, a, colour_fingerprint(fr.index_select(1, fp_rows)), ext)
            graphed.graphs.clear()
            live.log(f"CUDA-graph check: max |recorded - normal| of one clean-up call = {diff:.2e} "
                     f"({'OK' if diff < 2e-2 else 'WARNING: large; consider --no-graphs'})")
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
    n_dropped = n_seen = n_ghost = 0
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
            if n_sf and step >= args.sf_start_step:
                fr = self_force_v7(sf_sampler, mi_ema, fr, act, mot, colour, counts, args.sf_steps, L)
            # steering dropout: these rows get no keys (null id) and no motion (zeros + flag)
            drop = torch.rand(B, device=device) < args.steer_drop
            context, actions, ext = mi(fr, act, mot, L, absent=drop)
            actions = torch.where(drop[:, None], torch.full_like(actions, NULL_ACTION), actions)
            n_dropped += int(drop.sum()); n_seen += B
            target = fr[:, target_row]
            sigma = sample_sigma(B, args, device)
            sigma_cond = sample_sigma_cond(B, model.n_mem, args, device)
            ghost = use_ghost = None
            if args.ghost_frac > 0:
                # 'first dream step' practice (model_v7.loss_ghost): these rows get heavy noise around the
                # previous frame (as dreamed, if recovery practice replaced it), half of them slid by the real turn
                use_ghost = torch.rand(B, device=device) < args.ghost_frac
                lo, hi = math.log(args.ghost_sigma_min), math.log(args.ghost_sigma_max)
                sigma = torch.where(use_ghost, torch.exp(lo + (hi - lo) * torch.rand(B, device=device)), sigma)
                ghost = fr[:, prev_row]
                slide = torch.rand(B, device=device) < args.ghost_shift_p
                if bool(slide.any()):
                    px = motion_shift_px(mot[:, L - 1], minfo["scale"], fpx_train) * slide[:, None]
                    ghost = shift_frames(ghost, px)
                n_ghost += int(use_ghost.sum())

            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            if args.decay_hours > 0:
                prog = min(1.0, train_seconds / 3600.0 / args.decay_hours)
                lr *= args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
            for gq in opt.param_groups:
                gq["lr"] = lr * gq.get("mult", 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_ghost is not None:
                    loss = model.loss_ghost(target, context, actions, sigma, sigma_cond, colour, ext, ghost, use_ghost)
                else:
                    loss = model.loss(target, context, actions, sigma, sigma_cond, colour, ext)
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
                # how big is the motion network's push on the conditioning vector so far? (0 at the start)
                mw = model.unet.motion_mlp[2].weight.abs().mean().item()
                live.log(f"step {step}  loss {loss_avg:.4f}  {it_s:.2f} it/s  lr {lr:.2e}  grad {gn.item():.2f}  "
                         f"VRAM {vram:.1f} GB  motion-net |w| {mw:.2e}  steering dropped {n_dropped}/{n_seen}  ghost rows {n_ghost}  next dream {eta}")
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
