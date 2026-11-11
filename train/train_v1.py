"""Train Dream World v1: the "stop melting" upgrade of train.py.

    # the real run (warm-starts from v0's weights, see model_v1.py):
    python train/train_v1.py --run dream_v1 --init train/checkpoints/dream_v1_init.pt
    python train/train_v1.py --run dream_v1 --resume          # carry on after a stop
    # a small test on the real data that leaves the real folders alone:
    python train/train_v1.py --smoke --run smoke_v1 --batch 32 --max-steps 300 --vram-cap-gb 8

WHAT'S NEW COMPARED WITH train.py (v0), in plain words
  1. RECOVERY PRACTICE ("self-forcing lite").
     v0 only ever saw REAL video as its memory. When you play, its memory is its own dreams,
     which are a bit wrong in ways real video never is, so the errors snowball into mush.
     Now, in every batch, a slice of the examples (--sf-frac, default 25%) gets its newest 1 or 2
     memory frames REPLACED by frames the model dreamed itself (with the EMA weights, quick
     2-step sampling, no learning during that part). The answer it must produce is still the
     REAL next frame. So it practises "my memory is my own slightly-off dream; steer back to
     reality", which is exactly the situation during play.
     To dream those 1-2 frames we need real frames from BEFORE them, so every training window
     is 2 frames longer than v0's (K + 2 + 1 frames). Examples that don't self-force just use the
     last K + 1 frames of the window.
  2. HEAVIER CONTEXT NOISE (the v0 anti-melt trick, turned up).
     v0 smudged memory with noise around 0.08 (max 0.7). v1 centres it higher (~0.17, max 1.5),
     still 10% of the time none, and half the examples get a DIFFERENT amount per memory frame.
     The model is told every amount (model_v1.py). Flags: --cn-*.
  3. LONGER MEMORY: K = 8 frames (0.8 s) instead of 4, warm-started from v0 (model_v1.py).
  4. LONGER TEST WALK + A MELT METER. Every rollout is now 300 frames (30 s). For each dreamed
     frame we measure SHARPNESS (how much fine detail/edges it has), divided by the average
     sharpness of real video, so 1.0 = "as crisp as real", 0.3 = "mostly mush".
     "Melt time" = the first moment the last-10-frames average drops below 0.5, OR the frames
     turn to grainy static (which fools a plain sharpness meter: static scores as "very sharp").
     Colour contrast and colour drift vs the starting frames are logged next to it.
     At startup (new runs) we run v0 on the exact same walk for a fair baseline.
     Rollouts use 10 sampler steps (the dial lab found 10 holds shapes longer than 5). Each
     rollout also dreams an "anchored" twin (colours gently pulled back to the start's, --anchor)
     saved as v1_anchor_rollout_step<step>.mp4: nicer to watch, but only the RAW one is scored.
  5. FINE-TUNE SETTINGS: lr 1e-4 (v0: 2e-4) with a 500-step warmup, and the EMA (the smoothed
     weights that dream) runs at full strength from step 0, because it starts from v0's already
     good EMA (see the note in main()). Mirror flips and extra turn practice as in v0.

Dashboard contract is the same as v0 (status.json, run.log, metrics.jsonl, frame_*.jpg in
train/live/<run_id>/), and videos go to train/out/v1_rollout_step<step>.mp4.
Everything shared with v0 (Windows-safe file writes, the Live helper, the rollout loop) is
imported from train.py unchanged, so there's one copy of it.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
import math
import pathlib
import time
import traceback

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from build_cache import compute_valid                                     # noqa: E402
from data import (ACTION_NAMES, InfiniteWeightedSampler, WindowDataset, load_cache,  # noqa: E402
                  to_model_space, to_uint8_image)
from model import build_model                                             # noqa: E402  (v0, for the baseline)
from model_v1 import DEFAULT_CFG_V1, build_model_v1, warm_start_from_v0   # noqa: E402
from train import Live, label, safe_replace, upscale              # noqa: E402  (reused as-is)

# The fixed 30-second scripted walk. (key, how many frames): 1 = W, 5 = turn left, 6 = turn right.
ROLLOUT_SCRIPT_V1 = [(1, 40), (6, 15), (1, 60), (5, 30), (1, 60), (6, 15), (1, 80)]   # 300 frames
EXTRA = 2          # extra real frames at the front of each window, so we can dream up to 2 memory frames
MELT_LEVEL = 0.5   # "melted" = sharpness below half of real video's
MELT_AVG = 10      # ... averaged over the last 10 frames (1 s), so one blurry frame doesn't count
GRAIN_MAX = 1.1    # ... or "grain" (see grain()) above 1.1 for 1 s = glitch static (also "melted")


def pose_string(script):
    return ", ".join(f"{ACTION_NAMES[a]} x{n}" for a, n in script)


# ---------------------------------------------------------------- the melt meter

def sharpness(frames_u8):
    """Sharpness of each frame: average size of the Laplacian of its grey version.
    The Laplacian is a tiny filter (centre pixel x4 minus its 4 neighbours) that is ~0 on smooth
    areas and big on edges and texture. Mush has no edges, so its number drops.
    frames_u8: [T, H, W, 3] uint8 RGB -> numpy float [T]."""
    out = np.empty(len(frames_u8), np.float64)
    for i, fr in enumerate(frames_u8):
        g = cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_RGB2GRAY).astype(np.float32)
        out[i] = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1)).mean()
    return out


def grain(frames_u8):
    """How much of each frame's detail is pixel-sized GRAIN rather than real edges.
    = Laplacian at full size / Laplacian after shrinking the frame to half size.
    Real edges (a lamp post, a window) survive shrinking, so for real video this is ~0.6-1.0
    (500 random real frames: median 0.76, max 0.97). Static is pixel-sized speckle that mostly
    averages away when shrunk: real frames + noise give 1.0-1.9, pure noise ~2.0. Mushy dreams
    and the anchored dreams sit at ~0.6-0.7 (measured 2026-10-19).
    frames_u8: [T, H, W, 3] uint8 RGB -> numpy float [T]."""
    out = np.empty(len(frames_u8), np.float64)
    lap = lambda g: np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1)).mean()
    for i, fr in enumerate(frames_u8):
        g = cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_RGB2GRAY).astype(np.float32)
        half = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        out[i] = lap(g) / max(lap(half), 1e-3)
    return out


def colour_stats(frames_u8):
    """Per-frame colour mean and contrast (standard deviation) of each channel R, G, B, in 0..255
    units. frames_u8 [T, H, W, 3] -> (mean [T, 3], std [T, 3])."""
    f = frames_u8.reshape(len(frames_u8), -1, 3).astype(np.float32)
    return f.mean(1), f.std(1)


def melt_report(frames_u8, ref, start_u8):
    """The melt meter for one dreamed walk.
      frames_u8 [T,H,W,3] the dream; ref = sharpness of average real video;
      start_u8 [K,H,W,3] the real frames the dream started from (for the colour numbers).

    Four readings per frame:
      sharp     detail vs real video (1.0 = as crisp as real, ~0 = smooth mush)
      grain     share of that detail that is pixel speckle (real video < 1.0, static > 1.1)
      contrast  colour contrast vs the starting frames (1.0 = same; the beige collapse -> ~0.3)
      shift     how far the average colour wandered from the starting frames (0..255 units)

    MELT TIME = the first moment the 1 s (10-frame) average of sharp either
      - drops below 0.5 ("mush"), or
      - or the 1 s average of grain rises above 1.1 ("static"). Static has huge Laplacian
        values, so a sharpness-only meter would call glitch static "very sharp"; this rule counts
        it as melted instead. (The dial lab found Heun sampling over-sharpens into static.)
        (A first try, "sharper than 1.5x its own first second", wrongly flagged the colour-anchored
        dreams: anchoring restores contrast, and sharpness rises with contrast.)
    melt_frame None = it survived the whole walk. Frame i is shown (i + 1) / 10 s into the dream."""
    s = sharpness(frames_u8) / ref
    ma = np.convolve(s, np.ones(MELT_AVG) / MELT_AVG, mode="valid")   # ma[j] = mean of s[j .. j+9]
    gr = grain(frames_u8)
    gma = np.convolve(gr, np.ones(MELT_AVG) / MELT_AVG, mode="valid")
    mush = ma < MELT_LEVEL
    # (static also needs real sharpness: on near-flat mush both Laplacians are tiny and their
    #  ratio is just rounding noise, so a high grain there means nothing)
    static = (gma > GRAIN_MAX) & (ma >= MELT_LEVEL)
    bad = np.flatnonzero(mush | static)
    melt = int(bad[0] + MELT_AVG - 1) if len(bad) else None           # index of the window's LAST frame
    kind = None if melt is None else ("static" if static[bad[0]] else "mush")
    m, sd = colour_stats(frames_u8)
    m0, sd0 = colour_stats(start_u8)
    contrast = sd.mean(1) / max(1e-3, float(sd0.mean()))              # [T]
    shift = np.abs(m - m0.mean(0)).mean(1)                            # [T]
    rep = {"melt_frame": melt, "melt_s": None if melt is None else round((melt + 1) / 10, 1),
           "melt_kind": kind, "sharp_mean": round(float(s.mean()), 3)}
    for name, a, b in [("0_1s", 0, 10), ("5_6s", 50, 60), ("10_11s", 100, 110), ("20_21s", 200, 210),
                       ("last_1s", len(s) - 10, len(s))]:
        if len(s) > a:
            rep["sharp_" + name] = round(float(s[a:b].mean()), 3)
            rep["grain_" + name] = round(float(gr[a:b].mean()), 2)
            rep["contrast_" + name] = round(float(contrast[a:b].mean()), 3)
            rep["shift_" + name] = round(float(shift[a:b].mean()), 1)
    return rep, s


def secs(v):
    """A melt time for printing: 6.0 -> "6.0 s", None -> "never"."""
    return "never" if v is None else f"{v} s"


def melt_line(name, r):
    when = ("never (whole walk)" if r["melt_frame"] is None else
            f"{r['melt_s']} s (frame {r['melt_frame']}, {r['melt_kind']})")
    cols = [("0-1 s", "0_1s"), ("5-6 s", "5_6s"), ("10-11 s", "10_11s"), ("20-21 s", "20_21s"), ("last", "last_1s")]
    part = lambda key: ", ".join(f"{t} {r.get(key + '_' + c)}" for t, c in cols)
    return (f"{name}: melt time {when}\n    sharpness vs real: {part('sharp')}"
            f"\n    grain (static > 1.1): {part('grain')}"
            f"\n    contrast vs start: {part('contrast')}\n    colour shift (0-255): {part('shift')}")


@torch.no_grad()
def rollout_v1(model, frames_mm, actions_np, dream_start, K, script, steps, device, anchor=0.15, seed=1234):
    """Dream the scripted walk TWICE in one batch (2 rows cost about the same as 1 on the GPU):
      row 0 = RAW: the model on its own. This is what the melt meter scores (honest progress).
      row 1 = ANCHORED: after each frame, pull each colour channel's average and contrast `anchor`
              (15%) of the way back to the starting real frames' values (the dial lab's "colour
              leash"; it stops the beige collapse). Just for looking at, not for scoring.
    Both start from the real frames just before `dream_start` (K of them, so v0 (K=4) and v1
    (K=8) start their dream at the same moment of the same video) and press the same keys.
    Returns (raw uint8 [T,H,W,3], anchored uint8 [T,H,W,3], keys [T])."""
    ctx = to_model_space(torch.from_numpy(np.array(frames_mm[dream_start - K:dream_start])).to(device))
    ctx = ctx[None].repeat(2, 1, 1, 1, 1)                                   # [2, K, 3, H, W]
    ref_mean = ctx[1:].mean(dim=(1, 3, 4))[:, :, None, None]                # [1, 3, 1, 1]
    ref_std = ctx[1:].std(dim=(1, 3, 4))[:, :, None, None]
    hist = [int(a) if a >= 0 else 1 for a in actions_np[dream_start - K:dream_start - 1]]
    keys = [a for a, n in script for _ in range(n)]
    g = torch.Generator(device=device).manual_seed(seed)    # same noise every rollout: fair comparison
    raw, anc = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for a in keys:
            act = torch.tensor([hist[-(K - 1):] + [a]] * 2, device=device)
            nxt = model.sample(ctx, act, steps=steps, generator=g).float()  # [2, 3, H, W]
            if anchor > 0:
                x = nxt[1:]
                m = x.mean(dim=(2, 3), keepdim=True)
                sd = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
                fixed = (x - m) / sd * ref_std + ref_mean                   # same picture, start's colours
                nxt = torch.cat([nxt[:1], ((1 - anchor) * x + anchor * fixed).clamp(-1, 1)])
            raw.append(to_uint8_image(nxt[0]))
            anc.append(to_uint8_image(nxt[1]))
            ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)              # slide the memory window
            hist.append(a)
    return np.stack(raw), np.stack(anc), keys


# ---------------------------------------------------------------- training-batch helpers

def sample_sigma_cond(B, K, args, device):
    """How much to smudge each memory frame this step. Returns [B, K].
    Log-normal: exp(mean + std * N(0,1)), capped at --cn-max. With the defaults (-1.8, 1.1) the
    middle value is e^-1.8 = 0.17 and ~2% hit the 1.5 cap (v0: middle 0.08, cap 0.7).
    --cn-perframe of the examples get an independent amount for every frame; the rest share one.
    --cn-zero of the amounts are set to exactly 0 (clean memory must stay familiar too)."""
    draw = lambda *shape: (torch.randn(*shape, device=device) * args.cn_std + args.cn_mean).exp().clamp(max=args.cn_max)
    shared = draw(B, 1).expand(B, K)
    own = draw(B, K)
    use_own = (torch.rand(B, 1, device=device) < args.cn_perframe)
    sc = torch.where(use_own, own, shared)
    # zeroing: shared examples zero all frames together, per-frame examples zero frames one by one
    zero = torch.where(use_own, torch.rand(B, K, device=device) < args.cn_zero,
                       (torch.rand(B, 1, device=device) < args.cn_zero).expand(B, K))
    return torch.where(zero, torch.zeros_like(sc), sc)


@torch.no_grad()
def self_force(ema, fr, act, K, n_sf, n_two, steps):
    """Recovery practice. fr [B, K+EXTRA+1, 3, H, W] real frames (model space), act [B, K+EXTRA].
    For the first n_sf examples, replace the newest memory frames with the EMA model's own dreams:
      the first n_two of them get their 2 newest memory frames dreamed,
      the rest (n_sf - n_two) get just the newest 1 dreamed.
    (The batch is already in random order, so "the first n" is a random pick. Using fixed counts
    instead of coin flips keeps tensor sizes constant, which keeps cuDNN fast.)

    Window layout (EXTRA = 2): frames 0 1 | 2 ... K+1 | K+2
                                extra   | memory      | target
    Dreaming frame g uses the K frames before it (g-K .. g-1) and the keys act[g-K .. g-1]
    (act[g-1] is the key that leads into frame g). We dream frame K (only for the n_two group)
    first, then frame K+1 (for everybody in the n_sf group); frame K+1's dream already sees the
    dreamed frame K where there is one, just like during play. Returns fr, changed in place."""
    if n_sf <= 0:
        return fr
    for j in range(EXTRA):                    # j = 0 dreams frame K, j = 1 dreams frame K+1
        g = K + j
        # an example needs frame g dreamed if its number of dreamed frames >= EXTRA - j
        rows = n_two if j < EXTRA - 1 else n_sf
        if rows <= 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dream = ema.sample(fr[:rows, g - K:g], act[:rows, g - K:g], steps=steps)
        fr[:rows, g] = dream.float()
    return fr


# ---------------------------------------------------------------- saving a rollout

def write_live_frames(live, dream, keys):
    """Overwrite frame_*.jpg in the live folder (the dashboard's newest dream)."""
    for i, (fr, a) in enumerate(zip(dream, keys)):
        big = label(upscale(fr), ACTION_NAMES[a])
        dst = live.dir / f"frame_{i:05d}.jpg"
        tmp = live.dir / f"frame_{i:05d}.jpg.tmp"       # not matched by frame_*.jpg: never shown half-written
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            tmp.write_bytes(buf.tobytes())
            safe_replace(tmp, dst)


def write_mp4(path, dream, keys, real, title, sharp):
    """Side-by-side video: real video (left) | dream (right), with the key and the sharpness
    number written on each dream frame so you can SEE where the melt meter says it melted."""
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.mp4")         # .tmp so the dashboard never lists half a video
    w = imageio.get_writer(str(tmp), fps=10, codec="libx264", pixelformat="yuv420p",
                           quality=8, macro_block_size=16, ffmpeg_log_level="error")
    for i, (fr, a) in enumerate(zip(dream, keys)):
        r = real[min(i, len(real) - 1)]
        left = label(upscale(r), "real video")
        right = label(upscale(fr), f"{title}: {ACTION_NAMES[a]}")
        label(right, f"{(i + 1) / 10:4.1f}s  sharp {sharp[i]:.2f}", y=246)
        w.append_data(np.concatenate([left, right], axis=1))
    w.close()
    safe_replace(tmp, path)
    return path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v1", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=str(HERE / "checkpoints" / "dream_v1_init.pt"),
                    help="v0 checkpoint to warm-start from on a NEW run ('none' = from scratch)")
    ap.add_argument("--smoke", action="store_true",
                    help="checkpoints/out under train/_smoke/, live view in train/_smoke/live_v1/ (data stays the real cache)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ema-ramp", action="store_true",
                    help="use v0's EMA warm-up ramp even on a warm start (normally off: see notes in main)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--max-hours", type=float, default=0, help="0 = no time limit")
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--nonfwd-weight", type=float, default=2.0)
    ap.add_argument("--holdout", type=float, default=0.03)
    # recovery practice (self-forcing lite)
    ap.add_argument("--sf-frac", type=float, default=0.25,
                    help="share of each batch whose newest memory frames are the model's own dreams (0 = off)")
    ap.add_argument("--sf-two", type=float, default=0.5,
                    help="of those, the share that get 2 dreamed frames (the rest get 1)")
    ap.add_argument("--sf-steps", type=int, default=2, help="sampler steps for those quick dreams")
    # context noise (anti-melt smudge)
    ap.add_argument("--cn-mean", type=float, default=-1.8, help="log-normal centre (v0: -2.5)")
    ap.add_argument("--cn-std", type=float, default=1.1, help="log-normal spread (v0: 1.0)")
    ap.add_argument("--cn-max", type=float, default=1.5, help="cap (v0: 0.7)")
    ap.add_argument("--cn-zero", type=float, default=0.1, help="share with no smudge at all (v0: 0.1)")
    ap.add_argument("--cn-perframe", type=float, default=0.5, help="share with a different amount per frame (v0: 0)")
    # rollouts / checkpoints
    ap.add_argument("--sample-every-min", type=float, default=15,
                    help="minutes between rollouts (a 10-step, 300-frame rollout takes ~1 min)")
    ap.add_argument("--first-sample-step", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=20)
    ap.add_argument("--keep-every-hours", type=float, default=2)
    ap.add_argument("--rollout-steps", type=int, default=10,
                    help="denoising steps per dreamed frame (v0 used 5; the dial lab found 10 holds shapes longer)")
    ap.add_argument("--anchor", type=float, default=0.15,
                    help="colour-anchor strength for the second (anchored) rollout video; 0 = skip it")
    ap.add_argument("--no-baseline", action="store_true", help="skip the v0 baseline rollout at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0,
                    help="hard VRAM cap for this process (0 = none). Over the cap = clean out-of-memory "
                         "error instead of Windows quietly spilling into system RAM and crawling.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = HERE / "_smoke" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "live_v1" if args.smoke else HERE / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    torch.manual_seed(args.seed)

    # ---- data. Windows are K + EXTRA + 1 frames long (see self_force).
    index, frames_mm, actions_np, cut_np = load_cache(cache_dir)
    K = args.K
    L = K + EXTRA                                      # "window K" as build_cache/data.py count it
    valid = compute_valid(actions_np, cut_np, index["cities"], L)
    train_ok = valid.copy()
    held_ok = np.zeros_like(valid)
    held_frames = np.zeros(len(valid), bool)
    roll_len = sum(n for _, n in ROLLOUT_SCRIPT_V1)
    rollout_start = None
    for c in index["cities"]:
        n = c["frames"]
        # same held-out tail as v0 (last 3% of each city), so v0 never trained on it either
        hold_n = min(max(int(args.holdout * n), L + roll_len + 20), n // 3)
        h0 = c["end"] - hold_n
        train_ok[h0 - L:c["end"]] = False
        held_ok[h0:c["end"]] = valid[h0:c["end"]]
        held_frames[h0:c["end"]] = True
        if rollout_start is None and n > 3 * (L + roll_len + 20):
            rollout_start = h0 + 5
    if rollout_start is None:
        sys.exit("No city is long enough for a 300-frame held-out rollout.")
    if train_ok.sum() == 0:
        sys.exit("No valid training windows.")
    # the dream starts at frame s0; v1 remembers frames s0-8..s0-1, the v0 baseline s0-4..s0-1
    dream_start = rollout_start + K
    real_clip = np.array(frames_mm[dream_start:dream_start + roll_len])     # what really came next
    start_clip = np.array(frames_mm[dream_start - K:dream_start])            # colour reference for the meter
    target_key = np.full(len(actions_np), -1, np.int64)
    target_key[:len(actions_np) - L + 1] = actions_np[L - 1:]          # key that makes the target frame
    weights = train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight)

    # sharpness of average real video = the "1.0" of the melt meter (all held-out frames)
    held_ix = np.flatnonzero(held_frames)
    sharp_ref = float(sharpness(np.array(frames_mm[held_ix[::2]])).mean())

    # ---- model, EMA, optimizer
    latest = ckpt_dir / "latest.pt"
    ck = None
    if args.resume:
        if not latest.exists():
            sys.exit(f"--resume: no checkpoint at {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
    elif latest.exists():
        sys.exit(f"{latest} already exists. Use --resume to continue it, or pick another --run name.")
    model_cfg = ck["model_cfg"] if ck else dict(DEFAULT_CFG_V1, K=K)
    if model_cfg["K"] != K:
        sys.exit(f"checkpoint was trained with K={model_cfg['K']}, not --K {K}")
    model = build_model_v1(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    init_notes, v0_ck = [], None
    if not ck and args.init.lower() != "none":
        # Warm start: v0's EMA weights (the smoothed ones that dream best) -> both v1 copies.
        v0_ck = torch.load(args.init, map_location="cpu", weights_only=False)
        init_notes = warm_start_from_v0(model, v0_ck["ema"], v0_ck["model_cfg"]["K"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), fused=True)

    step = 0
    train_seconds = 0.0
    run_id = "v1-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline = None                                     # v0's melt numbers on the same walk
    # EMA on a warm start. v0 ramps the EMA in (it follows the live weights closely for the first
    # few thousand steps), which is right when starting from random weights. Here it would be
    # WRONG: we start from v0's EMA, which dreams far better than raw weights do (measured
    # 2026-10-19 on the 30 s walk: v0 EMA melts at 5.9 s, v0 raw weights at 1.2 s). So on a warm
    # start the EMA runs at full strength (--ema) from step 0: it begins as v0's good weights and
    # only slowly blends in what v1 learns. --ema-ramp forces v0's ramp anyway.
    warm_started = v0_ck is not None
    if ck:
        warm_started = ck.get("warm_started", False)
    if ck:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline = ck.get("v0_baseline")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cities = ", ".join(c["name"] for c in index["cities"] if c["has_actions"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": "dream model v1",
        "prompt": f"city-walk dream v1 ({cities}), {n_params / 1e6:.0f}M params, 64x112, K={K}",
        "pose": pose_string(ROLLOUT_SCRIPT_V1), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "v0_melt_s": baseline["melt_s"] if baseline else None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v1 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.2f}M params; K={K}; batch {args.batch}; {int(train_ok.sum())} train windows, "
             f"{int(held_ok.sum())} held-out; cache {cache_dir}")
    live.log(f"recovery practice: {args.sf_frac:.0%} of each batch ({args.sf_two:.0%} of those with 2 dreamed "
             f"frames, {args.sf_steps} sampler steps); context noise: log-normal({args.cn_mean}, {args.cn_std}) "
             f"max {args.cn_max}, {args.cn_zero:.0%} zero, {args.cn_perframe:.0%} per-frame")
    live.log(f"melt meter: real held-out video sharpness = {sharp_ref:.2f} (= 1.0); rollout = "
             f"{roll_len} frames starting at cache frame {dream_start}")
    for n in init_notes:
        live.log("warm start: " + n)
    live.update(stage="starting")

    # ---- loader
    ds = WindowDataset(cache_dir, L, flip_p=args.flip_p)
    sampler = InfiniteWeightedSampler(weights, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)

    # fixed held-out batch for validation loss + the warm-start check (no flips)
    held_idx = np.flatnonzero(held_ok)
    rs = np.random.default_rng(0)
    pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
    val_ds = WindowDataset(cache_dir, L, flip_p=0.0)
    vb = [val_ds[int(t)] for t in pick]
    val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]))

    def run_val():
        """EMA loss on 64 fixed held-out windows at 5 fixed noise levels (memory smudge 0.05),
        using the same window ends as v0's val loss would, so the numbers are roughly comparable."""
        fr = to_model_space(val_batch[0].to(device))
        act = val_batch[1].to(device)
        B = fr.shape[0]
        tot, sig_list = 0.0, [0.05, 0.2, 0.5, 1.0, 2.0]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in sig_list:
                    tot += ema.loss(fr[:, L], fr[:, EXTRA:L], act[:, EXTRA:L],
                                    torch.full((B,), s, device=device), 0.05).item()
        return tot / len(sig_list)

    # ---- startup checks on a NEW warm-started run: does v1 still behave like v0, and how fast does v0 melt?
    if v0_ck is not None:
        v0 = build_model(v0_ck["model_cfg"]).to(device).eval()
        v0.load_state_dict(v0_ck["ema"])
        K0 = v0.K
        fr = to_model_space(val_batch[0][:16].to(device))
        act = val_batch[1][:16].to(device)
        B = fr.shape[0]
        g = torch.Generator(device=device).manual_seed(0)
        x_clean = fr[:, L]
        with torch.no_grad():
            # (a) one denoising call at several noise levels, full float32: the same noisy target goes
            # in; v0 sees the newest 4 memory frames, v1 sees all 8. Should match almost exactly.
            msgs = []
            for s in [0.1, 0.5, 2.0]:
                sig = torch.full((B,), s, device=device)
                xn = x_clean + s * torch.randn(x_clean.shape, device=device, generator=g)
                d0, _ = v0(xn, sig, fr[:, L - K0:L], torch.zeros(B, device=device), act[:, L - K0:L])
                d1, _ = model(xn, sig, fr[:, EXTRA:L], 0.0, act[:, EXTRA:L])
                msgs.append(f"sigma {s}: MSE(v0, v1) {torch.mean((d0 - d1) ** 2).item():.2e} "
                            f"(vs. their error on the real frame {torch.mean((d0 - x_clean) ** 2).item():.2e})")
            # (b) a full 5-step dreamed first frame from the same starting noise
            g0 = torch.Generator(device=device).manual_seed(1)
            g1 = torch.Generator(device=device).manual_seed(1)
            p0 = v0.sample(fr[:, L - K0:L], act[:, L - K0:L], steps=5, generator=g0)
            p1 = model.sample(fr[:, EXTRA:L], act[:, EXTRA:L], steps=5, generator=g1)
            mse_first = torch.mean((p0 - p1) ** 2).item()
        for m in msgs:
            live.log("warm-start check, one denoise call: " + m)
        live.log(f"warm-start check, dreamed first frame: MSE(v0, v1) = {mse_first:.2e} "
                 f"({'OK: v1 starts out as v0' if mse_first < 1e-4 else 'WARNING: v1 differs from v0 at init'})")
        live.metric({"step": step, "time": time.time(), "warmstart_first_frame_mse": mse_first})

        # (c) the v0 baseline: the exact same 300-frame walk, same start moment, same noise seed,
        # same sampler steps. Raw is scored; the anchored twin is saved as a video to look at.
        if not args.no_baseline:
            live.update(stage="v0 baseline rollout")
            t0 = time.time()
            raw0, anc0, keys = rollout_v1(v0, frames_mm, actions_np, dream_start, K0, ROLLOUT_SCRIPT_V1,
                                          args.rollout_steps, device, anchor=args.anchor)
            rep0, s0 = melt_report(raw0, sharp_ref, start_clip)
            arep0, as0 = melt_report(anc0, sharp_ref, start_clip)
            real_rep, _ = melt_report(real_clip, sharp_ref, start_clip)
            tag = f"v0_step{v0_ck['step']:07d}"
            path0 = write_mp4(out_dir / f"v1_baseline_{tag}.mp4", raw0, keys, real_clip, f"v0 {v0_ck['step']}", s0)
            write_mp4(out_dir / f"v1_anchor_baseline_{tag}.mp4", anc0, keys, real_clip,
                      f"v0 {v0_ck['step']} anchor {args.anchor}", as0)
            baseline = dict(rep0, v0_step=v0_ck["step"], video=path0.name, anchor_melt_s=arep0["melt_s"],
                            steps=args.rollout_steps)
            live.log(melt_line(f"v0 BASELINE (step {v0_ck['step']}, {args.rollout_steps} sampler steps, raw, "
                               f"{time.time() - t0:.0f}s)", rep0))
            live.log(f"v0 baseline with colour anchor {args.anchor} (video only): melt time "
                     f"{secs(arep0['melt_s'])} ({arep0['melt_kind'] or '-'})")
            live.log(melt_line("real video, same 30 s (for scale; should never 'melt')", real_rep))
            live.metric({"step": step, "time": time.time(), "v0_baseline": baseline, "real_clip": real_rep})
            live.update(v0_melt_s=baseline["melt_s"])
        del v0
        torch.cuda.empty_cache()
    del v0_ck

    def save_ckpt(path):
        try:
            tmp = str(path) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                        "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg,
                        "v0_baseline": baseline, "sharp_ref": sharp_ref, "warm_started": warm_started}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def do_rollout():
        """Raw + anchored dream of the 30 s walk (one batch). The RAW one is scored and shown live
        (frame_*.jpg); the anchored one only goes to train/out/v1_anchor_rollout_step<step>.mp4."""
        try:
            t0 = time.time()
            raw, anc, keys = rollout_v1(ema, frames_mm, actions_np, dream_start, K, ROLLOUT_SCRIPT_V1,
                                        args.rollout_steps, device, anchor=args.anchor)
            rep, s = melt_report(raw, sharp_ref, start_clip)
            arep, a_s = melt_report(anc, sharp_ref, start_clip)
            write_live_frames(live, raw, keys)
            path = write_mp4(out_dir / f"v1_rollout_step{step:07d}.mp4", raw, keys, real_clip,
                             f"v1 step {step}", s)
            if args.anchor > 0:
                write_mp4(out_dir / f"v1_anchor_rollout_step{step:07d}.mp4", anc, keys, real_clip,
                          f"v1 {step} anchor {args.anchor}", a_s)
            v = run_val()
            live.log(f"rollout at step {step}: {len(keys)} frames x2 (raw + anchored), {args.rollout_steps} "
                     f"sampler steps, in {time.time() - t0:.1f}s -> {path.name}; val loss {v:.4f}")
            live.log(melt_line(f"v1 step {step} (raw)", rep)
                     + (f"\n    [v0 baseline raw melt: {secs(baseline['melt_s'])}]" if baseline else "")
                     + f"\n    [with colour anchor {args.anchor}: melt {secs(arep['melt_s'])}]")
            live.metric(dict({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name,
                              "anchor_melt_s": arep["melt_s"]}, **rep))
            live.update(last_rollout_step=step, val_loss=v, melt_s=rep["melt_s"])
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    # ---- the loop
    n_sf = int(round(args.sf_frac * args.batch))          # examples per batch that get dreamed memory
    n_two = int(round(args.sf_two * n_sf))                 # ... of which this many get 2 dreamed frames
    now = time.time()
    last_status, last_ckpt, last_keep = 0.0, now, now
    next_sample = None if step < args.first_sample_step else now + args.sample_every_min * 60
    sample_soon = step < args.first_sample_step
    loss_avg = None
    win_t, win_steps, it_s = time.time(), 0, 0.0
    bad_steps = 0
    t_loop = time.time()
    live.update(stage="training")

    try:
        it = iter(loader)
        while True:
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_hours and (time.time() - t_loop) / 3600 >= args.max_hours:
                break
            frames_u8, act = next(it)
            fr = to_model_space(frames_u8.to(device, non_blocking=True))    # [B, K+3, 3, H, W]
            act = act.to(device, non_blocking=True)                         # [B, K+2]
            B = fr.shape[0]

            # 1. recovery practice: swap some newest memory frames for the model's own dreams
            fr = self_force(ema, fr, act, K, n_sf, n_two, args.sf_steps)
            context, target, actions = fr[:, EXTRA:L], fr[:, L], act[:, EXTRA:L]
            # 2. EDM noise for the target (same as v0) and the heavier per-frame memory smudge
            sigma = (torch.randn(B, device=device) * 1.2 - 0.4).exp().clamp(2e-3, 20.0)
            sigma_cond = sample_sigma_cond(B, K, args, device)

            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            for gq in opt.param_groups:
                gq["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(target, context, actions, sigma, sigma_cond)
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
            # EMA: ema = d*ema + (1-d)*model (see the warm-start note above for why no ramp)
            d = args.ema if (warm_started and not args.ema_ramp) else min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
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
                live.log(f"step {step}  loss {loss_avg:.4f}  {it_s:.2f} it/s  lr {lr:.2e}  "
                         f"grad {gn.item():.2f}  VRAM {vram:.1f} GB  next dream {eta}")
            if now - last_status >= 10:
                last_status = now
                live.update(step=step, loss=round(loss_avg, 5), it_s=round(it_s, 2), lr=lr,
                            peak_vram_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            if now - last_ckpt >= args.ckpt_every_min * 60:
                save_ckpt(latest)
                last_ckpt = time.time()
                live.log(f"checkpoint saved (step {step})")
                if now - last_keep >= args.keep_every_hours * 3600:
                    save_ckpt(ckpt_dir / f"step{step:07d}.pt")
                    last_keep = time.time()
                    live.log(f"keeper checkpoint step{step:07d}.pt saved")

        save_ckpt(latest)
        if step >= args.first_sample_step and live.status.get("last_rollout_step") != step:
            do_rollout()
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
