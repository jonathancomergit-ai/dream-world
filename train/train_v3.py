"""Train Dream World v3: long memory + colour memory + longer recovery practice + a fair ruler.

    # the real run (warm-starts from a copy of v2's best save, see model_v3.py):
    python train/train_v3.py --run dream_v3 --init train/checkpoints/dream_v3_init.pt
    python train/train_v3.py --run dream_v3 --resume          # carry on after a stop
    # a small test that leaves the real folders alone:
    python train/train_v3.py --smoke --run smoke_v3 --batch 16 --max-steps 300 --first-sample-step 150

    (or just double-click train_v3.bat / train_v3_resume.bat in the DreamWorld folder)

WHAT WENT WRONG IN v2 (and what each v3 change is for)
  v2's loss kept improving, but its long walks got WORSE after ~25k steps, and nobody noticed for
  hours because the melt meter only watched ONE walk (very noisy) and saves were 2 h apart. Its
  dreams also blurred, faded to beige, grew vertical "curtain" streaks when walking forward, and
  forgot what was behind you after 0.8 s.

  1. LONG MEMORY (model_v3.py). 8 recent frames + 8 far frames every 0.5 s = ~4.8 s of memory.
  2. COLOUR MEMORY (model_v3.py). A colour fingerprint of REAL frames ~4.5 s back (taken BEFORE
     any recovery-practice swap or smudge), blanked 10% of the time (--colour-drop).
  3. LONGER RECOVERY PRACTICE. --sf-frac (30%) of every batch gets its newest 1, 2, 3 or 4 recent
     memory frames replaced by the model's own quick dreams (EMA weights, 2 sampler steps), dreamed
     one after another just like in play, so frame 3's dream already sees dreamed frames 1-2.
     How many: --sf-weights "0.4,0.3,0.2,0.1" (= 40% of them get 1 dreamed frame, 30% get 2, ...).
     Each dreamed frame needs its own full 4.8 s of memory, so every training window is
     48 + 4 + 1 = 53 real frames long (data_v3.py).
  4. DETAIL PRACTICE. EDM picks a random noise level for every training example. Detail (edges,
     texture) is decided at LOW noise; the overall layout at high noise. v3 keeps v2's main recipe
     (log-normal, centre -0.4, spread 1.2) for 80% of examples and draws the other 20% from a
     low-noise bump (centre -1.8 = sigma 0.17, spread 0.8): --sigma-low-frac/-mean/-std.
     WHY a mix instead of just moving the centre down: moving the centre also takes practice away
     from the high-noise end, where the model decides the big shapes that keep a dream from
     melting. A 20% extra bump adds low-noise practice without starving the rest. With v2's
     recipe ~16% of examples had sigma < 0.2; with the mix it's ~41%.
  5. THE FAIR RULER, built in. Every rollout (every 15 min) dreams, in ONE batch:
       one held-out walk per city (up to 4 walking cities, never Wildlife) x 2 noise draws,
       20 s each, 10 sampler steps, RAW (no colour anchor),
     plus a colour-anchored twin of the first one (just to look at). Scores:
       melt median and mean over all dreams (train_v1.melt_report; a dream that never melts
       counts as 20 s), and the STREAK METER (see streak_ratio below) at 5/10/15/20 s.
     The melt meter's "1.0" is each walk's OWN real 20 s (some places are just softer than others).
     "Clean time" = melt time or streak onset, whichever comes first: the melt meter reads curtains
     as "sharp" (hard edges), so on its own it can reward a model for streaking more.
     best.pt is saved whenever the median melt beats the best so far (ties: mean melt, then val
     loss); --best-by clean makes it chase the median clean time instead. A keeper save every
     hour (v2: every 2 h). At the start of a new run v2-best is scored on the exact same ruler:
     the number to beat.
  7. SPEED: CUDA GRAPHS for the recovery-practice dreams (GraphedSampler below). Those dreams run
     on tiny batches where Python + launch overhead, not the GPU, is the cost; replaying a recorded
     graph makes 4 dreamed frames cheaper than v2's 2 were. Measured at batch 64: v2 3.8 it/s;
     v3 2.6 it/s without graphs, 4.15 it/s with them (19 GB VRAM).
  6. MORE SOURCES. Whatever is in data/cache_v3 (build_cache_v3.py), drawn by hours^0.7 x
     source weight (data_v3.py: Wildlife 0.33).

v3.1 SWITCHES (added 2026-10-19 evening, all OFF by default: a plain --resume trains exactly as v3)
  Problem: when v3 loses the scene it falls back to rippling-water texture. Ripples are full of
  edges (the melt meter calls them sharp) and are HORIZONTAL (the curtain meter only flags
  vertical streaks), so best.pt could be won by water texture.
  - Always on, extra numbers only: "clean2 time" = melt OR curtains OR ripples (streak ratio below
    --ripple-max, default 0.7, for a whole second). Printed in run.log, metrics.jsonl and
    status.json (clean2_median, clean2_mean). It changes nothing unless you also pass:
  - --best-by clean2      best.pt chases the median clean2 time.
  - --rebar-best          on a resume that switches --best-by, re-score the existing best.pt with
                          the new rule first (and keep a copy of it), so it must be beaten.
  - --source-weights      was already read from the command line on every resume; now a change
                          is logged, and a name that matches no source is warned about.
  - --sf-walk-only        recovery practice only on walking-city examples (never Wildlife).
  train/rescore_v3.py re-scores saved checkpoints offline with the same ruler.

Dream Lab contract unchanged:train/live/v3-<date-time>/ with status.json (now also melt_median,
melt_mean, streak), run.log, metrics.jsonl and frame_*.jpg (walk 0, first noise draw, raw).
Videos: train/out/v3_rollout_step<step>.mp4 and v3_anchor_rollout_step<step>.mp4.
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

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from build_cache import compute_valid                                          # noqa: E402
from data_v3 import (ACTION_NAMES, NON_WALKING, InfiniteWeightedSampler,         # noqa: E402
                     WindowDatasetV3, WindowDatasetV3Tagged, load_cache_any, parse_source_weights,
                     pick_ruler_cities, source_shares, to_model_space, to_uint8_image)
from model_v2 import build_model_v2                                             # noqa: E402  (v2 baseline)
from model_v3 import (DEFAULT_CFG_V3, FP_FRAMES, DenoiserV3, build_model_v3,    # noqa: E402
                      colour_fingerprint, far_action_summary, warm_start_from_v2)
from train import Live, safe_replace                                            # noqa: E402
from train_v1 import melt_report, pose_string, sample_sigma_cond, secs, sharpness  # noqa: E402
from train_v2 import gpu_free_gb, write_live_frames, write_mp4                  # noqa: E402

# The fair ruler's 20 s scripted walk: lots of forward walking (where the streaks show up) and
# turns both ways. (key, frames): 1 = W, 5 = turn left, 6 = turn right.  200 frames.
FAIR_SCRIPT = [(1, 40), (6, 15), (1, 40), (5, 30), (1, 45), (6, 15), (1, 15)]
STREAK_LEVEL = 1.3     # streak ratio above this for a whole second = "streaky"
# v3.1: the OTHER side of the same meter. When v3 loses the scene it often falls back to
# rippling-water texture: lots of horizontal lines, so the up-down change is big and the ratio
# drops WELL BELOW 1 (0.5-0.7 seen at 15-20 s). Ripples are full of edges, so the melt meter calls
# them "sharp", and the curtain rule only looks UP. Below this for a whole second = "ripply".
# (--ripple-max changes it; it only feeds the extra "clean2" number, see score_walks.)
RIPPLE_LEVEL = 0.7


# ---------------------------------------------------------------- the streak meter

def streak_ratio(frames_u8):
    """How "curtain-streaky" each frame is.
    Vertical streaks are lines running top to bottom. Going ACROSS the picture (left -> right)
    you cross them all the time, so the brightness changes a lot sideways; going DOWN you don't.
    So: ratio = average |brightness change between side-by-side pixels|
              / average |brightness change between pixels above each other|.
    Real street video sits around 1 (roughly as much sideways as up-down detail); we divide the
    dream's number by the same number for the real start frames, so 1.0 = "as real", and above
    1.3 for a second = streaky. (Checked 2026-10-19: real held-out walks read 0.95-1.13; v2's
    dreams with visible curtains read 1.4-7.) The up-down change is floored at 1 grey level so
    flat mush (both numbers ~0) can't produce a huge ratio by dividing tiny by tiny.
    frames_u8 [T, H, W, 3] -> numpy float [T]."""
    g = frames_u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)   # grey [T,H,W]
    dx = np.abs(np.diff(g, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2))
    return dx / np.maximum(dy, 1.0)


def streak_report(dream_u8, start_u8, ripple_max=RIPPLE_LEVEL):
    """Streak ratio vs the real start frames: 1 s averages at 5/10/15/20 s, and when it first
    stays above STREAK_LEVEL for a whole second (None = never).
    v3.1 adds "ripple_onset_s": when it first stays BELOW ripple_max for a whole second (the
    rippling-water fallback, see RIPPLE_LEVEL). Same 1 s average, same timing rule, other side."""
    r = streak_ratio(dream_u8) / max(float(streak_ratio(start_u8).mean()), 1e-3)
    ma = np.convolve(r, np.ones(10) / 10, mode="valid")
    bad = np.flatnonzero(ma > STREAK_LEVEL)
    rep = {"onset_s": None if not len(bad) else round((bad[0] + 10) / 10, 1)}
    rip = np.flatnonzero(ma < ripple_max)
    rep["ripple_onset_s"] = None if not len(rip) else round((rip[0] + 10) / 10, 1)
    for s in (5, 10, 15, 20):
        a = s * 10 - 10
        if len(r) >= a + 10:
            rep[f"{s}s"] = round(float(r[a:a + 10].mean()), 3)
    return rep


# ---------------------------------------------------------------- memory helpers

def mem_inputs(model, fr, act, g):
    """Everything the model needs to dream frame g of a window.
    fr [B, T, 3, H, W] frames (model space), act [B, T-1] keys (act[t] moves frame t -> t+1).
    Returns (memory [B, far_n+K, 3, H, W] oldest first, recent keys [B, K], far keys summary)."""
    K, Fn, S = model.K, model.far_n, model.far_stride
    idx = torch.tensor([g - o for o in model.mem_offsets()], device=fr.device)
    ctx = fr.index_select(1, idx)
    far = far_action_summary(act[:, g - K - S * Fn:g - K], Fn, S)
    return ctx, act[:, g - K:g], far


class GraphedSampler:
    """The recovery-practice dreams, made ~4x cheaper with CUDA graphs.

    WHY: the quick dreams run on small batches (1 to ~19 examples). At that size the GPU itself is
    barely busy; nearly all the time (~18 ms per network call, measured 2026-10-19) goes to Python
    and to launching the ~500 tiny GPU jobs one network call is made of, one by one. With 4
    dreamed frames x 2 sampler steps that's 8 calls in a row = most of a training step.
    A CUDA GRAPH records those ~500 launches once and then replays the whole recording with one
    command. Same maths, same weights (the EMA weights are updated IN PLACE every step, and the
    recording reads them from the same memory, so it always uses the current ones).
    Rules that come with it: the inputs must be copied into fixed "static" tensors before each
    replay, the batch size of a recording is fixed (we keep one recording per batch size: one per
    recovery-practice stage), and nothing random may happen inside it (the random start noise of
    the sampler is drawn outside). train_v3 checks graphed vs normal on real data at start-up."""

    def __init__(self, model, shared_pool=False, start="noise"):
        self.model = model
        self.start = start            # "prev" = start each quick dream from the newest memory frame + noise
        self.graphs = {}
        # exp --sf-shared-pool: all recordings share ONE memory pool (as torch's make_graphed_callables
        # does). Safe here because the recordings are replayed one after another and each one's output
        # is used up before the next replay. Saves ~0.5 GB per extra recording (8 dreamed frames = 7).
        self.pool = torch.cuda.graph_pool_handle() if shared_pool else None

    def _build(self, ctx, a, colour, far):
        st = {"x": torch.zeros_like(ctx[:, 0]), "s": torch.ones(ctx.shape[0], device=ctx.device),
              "ctx": ctx.clone(), "a": a.clone(), "col": colour.clone(), "far": far.clone()}

        def run():
            # cache_enabled=False: autocast's weight-copy cache can't live inside a recording
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                den, _ = self.model(st["x"], st["s"], st["ctx"], 0.0, st["a"], st["col"], st["far"])
            return den.float()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                # warm-up runs (cuDNN picks its algorithms here)
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = run()
        return g, st, out

    @torch.no_grad()
    def sample(self, ctx, a, steps, colour, far, sigma_max=5.0):
        """Same Euler sampler as DenoiserV1.sample (no context noise), via the recording.
        sigma_max = how much static each dream starts with (play.py's "trippiness" dial; 5.0 = as trained)."""
        B = ctx.shape[0]
        if B not in self.graphs:
            self.graphs[B] = self._build(ctx, a, colour, far)
        g, st, out = self.graphs[B]
        st["ctx"].copy_(ctx)
        st["a"].copy_(a)
        st["col"].copy_(colour)
        st["far"].copy_(far)
        sigmas = self.model.karras_sigmas(steps, sigma_max=sigma_max, device=ctx.device)
        x = torch.randn_like(st["x"]) * sigmas[0]
        if self.start == "prev":
            x = x + ctx[:, -1].float()
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            st["x"].copy_(x)
            st["s"].fill_(s)                     # (a GPU-side fill: no waiting on the GPU)
            g.replay()
            x = x + (x - out) / s * (s_next - s)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def check(self, ctx, a, colour, far):
        """Max |recording - normal call| of one clean-up call on these inputs (should be ~0)."""
        B = ctx.shape[0]
        if B not in self.graphs:
            self.graphs[B] = self._build(ctx, a, colour, far)
        g, st, out = self.graphs[B]
        x = torch.randn_like(st["x"])
        for k, v in (("ctx", ctx), ("a", a), ("col", colour), ("far", far), ("x", x)):
            st[k].copy_(v)
        st["s"].fill_(1.0)
        g.replay()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref, _ = self.model(x, torch.ones(B, device=x.device), ctx, 0.0, a, colour, far)
        return (out - ref.float()).abs().max().item()


def eager_sample(ema, start="noise"):
    """The plain (non-graph) version of GraphedSampler.sample, same signature."""
    def f(ctx, a, steps, colour, far):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return ema.sample(ctx, a, steps=steps, colour=colour, far_act=far, start=start).float()
    return f


@torch.no_grad()
def self_force_v3(sampler, ema, fr, act, colour, counts, steps):
    """Recovery practice. counts[c-1] = how many examples get their newest c recent frames dreamed
    (c = 1..4). Rows are ordered most-dreamed first (the batch is random, so this is a random pick;
    fixed counts keep tensor sizes steady, which keeps cuDNN fast):
        rows [0, n4)            4 dreamed frames
        rows [n4, n4+n3)        3 ...
    We dream frame P-4 (only the 4-group), then P-3 (the 4- and 3-groups), ..., then P-1 (everyone
    in recovery practice), each from the memory as it stands, which already includes the dreams
    before it: exactly what happens in play. P = the target (last frame of the window).
    Returns fr, changed in place."""
    P = fr.shape[1] - 1
    n_max = len(counts)
    for j in range(n_max, 0, -1):
        rows = sum(counts[j - 1:])                   # examples with at least j dreamed frames
        if rows <= 0:
            continue
        g = P - j
        ctx, a, far = mem_inputs(ema, fr[:rows], act[:rows], g)
        fr[:rows, g] = sampler(ctx, a, steps, colour[:rows], far)
    return fr


def sf_counts(n_sf, weights):
    """Split n_sf examples into groups with 1, 2, 3, 4 dreamed frames by the weights."""
    w = np.array(weights, np.float64)
    w = w / w.sum()
    c = np.floor(w * n_sf).astype(int)
    c[0] += n_sf - c.sum()                           # leftovers go to the 1-frame group
    return [int(x) for x in c]


def sample_sigma(B, args, device):
    """Target-frame noise: v2's log-normal for most examples, plus a low-noise bump (detail practice)."""
    main = (torch.randn(B, device=device) * args.p_std + args.p_mean).exp()
    low = (torch.randn(B, device=device) * args.sigma_low_std + args.sigma_low_mean).exp()
    use_low = torch.rand(B, device=device) < args.sigma_low_frac
    return torch.where(use_low, low, main).clamp(2e-3, 20.0)


# ---------------------------------------------------------------- the fair ruler

@torch.no_grad()
def dream_walks(model, frames_mm, actions_np, starts, n_seeds, keys, steps, device, anchor=0.0, seed=1234,
                use_colour=True, start="noise"):
    """Dream every walk x n_seeds in ONE batch (+ one colour-anchored twin of the first, if anchor > 0).
    Works for v3 (long memory + colour fingerprint) and v2 (just the 8 recent frames), so v2-best can
    be scored on exactly the same walks. Every walk starts dreaming at frame `start`, from the real
    frames before it (v3: the 48 before it; v2: the 8 before it), and presses the same keys.
    "Seeds" = independent noise draws: the rows share one fixed-seed generator, so each row gets
    its own noise but every rollout gets the SAME noise (fair from rollout to rollout).
    Returns uint8 [rows, T, H, W, 3]; row order = seed 0 walks..., seed 1 walks..., [anchored twin]."""
    v3 = isinstance(model, DenoiserV3)
    K = model.K
    Hn = model.history if v3 else K
    row_starts = [s for _ in range(n_seeds) for s in starts] + ([starts[0]] if anchor > 0 else [])
    B = len(row_starts)
    hist = torch.stack([to_model_space(torch.from_numpy(np.array(frames_mm[s - Hn:s])).to(device))
                        for s in row_starts])                                # [B, Hn, 3, H, W]
    acts = [[int(a) if a >= 0 else 1 for a in actions_np[s - Hn:s - 1]] for s in row_starts]
    colour = colour_fingerprint(hist[:, :FP_FRAMES]) if (v3 and use_colour) else None
    ref_mean = hist[-1:, -K:].mean(dim=(1, 3, 4))[:, :, None, None]          # anchor target (twin row)
    ref_std = hist[-1:, -K:].std(dim=(1, 3, 4))[:, :, None, None]
    if v3:
        idx = torch.tensor([Hn - o for o in model.mem_offsets()], device=device)
        S, Fn = model.far_stride, model.far_n
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for a in keys:
            for r in acts:
                r.append(a)
            A = torch.tensor([r[-K:] for r in acts], device=device)
            if v3:
                far = far_action_summary(torch.tensor([r[-(K + S * Fn):-K] for r in acts], device=device), Fn, S)
                nxt = model.sample(hist.index_select(1, idx), A, steps=steps, generator=g, start=start,
                                   colour=colour, far_act=far).float()
            else:
                nxt = model.sample(hist, A, steps=steps, generator=g, start=start).float()
            if anchor > 0:
                x = nxt[-1:]
                m = x.mean(dim=(2, 3), keepdim=True)
                sd = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
                fixed = (x - m) / sd * ref_std + ref_mean
                nxt = torch.cat([nxt[:-1], ((1 - anchor) * x + anchor * fixed).clamp(-1, 1)])
            out.append(((nxt.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1)
            for r in acts:
                del r[:-(Hn - 1)]
    return np.stack(out, 1)


def walk_sharp_ref(frames_mm, s, T, cache={}):
    """The melt meter's "1.0" for the walk starting at frame s: the average sharpness of the REAL
    video that follows (T frames). Per walk, not one number for all video: some places are just
    less detailed (Stockholm's held-out tail is dim and soft), and with one global number even
    the real footage there scored as "melted after 1 s"."""
    if (s, T) not in cache:
        cache[(s, T)] = float(sharpness(np.array(frames_mm[s:s + T])).mean())
    return cache[(s, T)]


def score_walks(dreams, row_starts, frames_mm, K, T, ripple_max=RIPPLE_LEVEL):
    """Melt + streak readings for each dreamed walk, and the summary numbers.
    v3.1 adds clean2 (see the bottom); every old number is computed exactly as before."""
    melts, reps, streaks = [], [], []
    for dr, s in zip(dreams, row_starts):
        start_u8 = np.array(frames_mm[s - K:s])
        rep, curve = melt_report(dr, walk_sharp_ref(frames_mm, s, T), start_u8)
        reps.append((rep, curve))
        melts.append(rep["melt_s"] if rep["melt_s"] is not None else T / 10)
        streaks.append(streak_report(dr, start_u8, ripple_max))
    summ = {"melts": melts, "melt_median": round(float(np.median(melts)), 2),
            "melt_mean": round(float(np.mean(melts)), 2)}
    st = {}
    for k in ("5s", "10s", "15s", "20s"):
        vals = [x[k] for x in streaks if k in x]
        if vals:
            st[k] = round(float(np.median(vals)), 3)
    ons = [x["onset_s"] if x["onset_s"] is not None else T / 10 for x in streaks]
    st["onset_median_s"] = round(float(np.median(ons)), 1)
    # "clean time" = how long a dream lasts before it EITHER melts OR grows curtains. Needed because
    # the melt meter scores curtains as "sharp" (they are full of hard edges), so a model could
    # look better on melt time just by streaking more (seen in the 2026-10-19 smoke runs).
    clean = [min(m, o) for m, o in zip(melts, ons)]
    summ["clean_median"] = round(float(np.median(clean)), 2)
    summ["clean_mean"] = round(float(np.mean(clean)), 2)
    st["streaky_share"] = round(float(np.mean([x["onset_s"] is not None for x in streaks])), 2)
    # v3.1 "clean2 time" = how long a dream lasts before it melts OR grows curtains OR turns into
    # ripples (the water-texture fallback: ratio below ripple_max for a second). Extra numbers
    # only: "clean" above is unchanged, and best.pt only chases clean2 with --best-by clean2.
    rons = [x["ripple_onset_s"] if x["ripple_onset_s"] is not None else T / 10 for x in streaks]
    clean2 = [round(float(min(c, r)), 1) for c, r in zip(clean, rons)]
    summ["clean2s"] = clean2
    summ["clean2_median"] = round(float(np.median(clean2)), 2)
    summ["clean2_mean"] = round(float(np.mean(clean2)), 2)
    st["ripple_level"] = ripple_max
    st["ripple_onset_median_s"] = round(float(np.median(rons)), 1)
    st["ripply_share"] = round(float(np.mean([x["ripple_onset_s"] is not None for x in streaks])), 2)
    summ["streak"] = st
    return summ, reps


def ruler_line(name, summ, T):
    st = summ["streak"]
    sline = "/".join(f"{st[k]:.2f}" if k in st else "-" for k in ("5s", "10s", "15s", "20s"))
    onset = "never" if st["onset_median_s"] >= T / 10 else f"{st['onset_median_s']} s"
    line = (f"{name}: median melt {summ['melt_median']} s, mean {summ['melt_mean']} s over "
            f"{len(summ['melts'])} dreams {summ['melts']} (a dream that never melts counts as {T / 10:.0f} s); "
            f"clean time (no melt AND no curtains) median {summ['clean_median']} s, mean {summ['clean_mean']} s"
            f"\n    streak ratio (1.0 = like real video, > {STREAK_LEVEL} = curtains) at 5/10/15/20 s: {sline}; "
            f"median streak onset {onset}; {st['streaky_share']:.0%} of dreams got streaky")
    if "clean2_median" in summ:      # v3.1 extra line (older saved summaries don't have it)
        ron = "never" if st["ripple_onset_median_s"] >= T / 10 else f"{st['ripple_onset_median_s']} s"
        line += (f"\n    clean2 time (no melt, no curtains AND no ripples, ratio < {st['ripple_level']}) median "
                 f"{summ['clean2_median']} s, mean {summ['clean2_mean']} s {summ['clean2s']}; median ripple "
                 f"onset {ron}; {st['ripply_share']:.0%} of dreams got ripply")
    return line


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v3", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=str(HERE / "checkpoints" / "dream_v3_init.pt"),
                    help="v2 checkpoint to warm-start from on a NEW run ('none' = from scratch)")
    ap.add_argument("--smoke", action="store_true",
                    help="checkpoints/out under train/_smoke/, live view in train/_smoke/live_v3/")
    ap.add_argument("--cache", default=None, help="default data/cache_v3")
    # memory
    ap.add_argument("--K", type=int, default=8, help="recent memory frames (v2: 8)")
    ap.add_argument("--far-n", type=int, default=8, help="far memory frames")
    ap.add_argument("--far-stride", type=int, default=5, help="frames between far memory frames (5 = 0.5 s)")
    ap.add_argument("--colour-drop", type=float, default=0.1, help="share of examples with the colour fingerprint blanked")
    # batch / optimiser (as v2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ema-ramp", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=2, help="batches each loader worker prepares ahead (53-frame windows are big)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--max-hours", type=float, default=0, help="0 = no time limit")
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--nonfwd-weight", type=float, default=2.0)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--source-weights", default="", help='e.g. "Wildlife=0.33,Venice=1.5" (added to data_v3.SOURCE_WEIGHTS)')
    ap.add_argument("--city-balance", type=float, default=0.3,
                    help="0 = draw sources by hours, 1 = every source equally (default: by hours^0.7)")
    # target noise
    ap.add_argument("--p-mean", type=float, default=-0.4)
    ap.add_argument("--p-std", type=float, default=1.2)
    ap.add_argument("--sigma-low-frac", type=float, default=0.2, help="share drawn from the low-noise bump (0 = v2 recipe)")
    ap.add_argument("--sigma-low-mean", type=float, default=-1.8)
    ap.add_argument("--sigma-low-std", type=float, default=0.8)
    # recovery practice
    ap.add_argument("--sf-frac", type=float, default=0.30, help="share of each batch with dreamed recent memory")
    ap.add_argument("--sf-weights", default="0.4,0.3,0.2,0.1", help="shares with 1,2,3,4 dreamed frames")
    ap.add_argument("--sf-steps", type=int, default=2)
    ap.add_argument("--no-graphs", action="store_true",
                    help="dream the recovery-practice frames without CUDA graphs (slower; for debugging)")
    # context noise (as v1/v2), now for all far + recent frames
    ap.add_argument("--cn-mean", type=float, default=-1.8)
    ap.add_argument("--cn-std", type=float, default=1.1)
    ap.add_argument("--cn-max", type=float, default=1.5)
    ap.add_argument("--cn-zero", type=float, default=0.1)
    ap.add_argument("--cn-perframe", type=float, default=0.5)
    # rollouts / checkpoints
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--first-sample-step", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=20)
    ap.add_argument("--keep-every-hours", type=float, default=1)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--ruler-cities", type=int, default=4, help="max walking cities in the fair ruler")
    ap.add_argument("--ruler-seeds", type=int, default=2)
    ap.add_argument("--anchor", type=float, default=0.15, help="colour anchor for the extra anchored twin video; 0 = skip")
    ap.add_argument("--best-by", choices=["melt", "clean", "clean2"], default="melt",
                    help="what best.pt chases: 'melt' = median melt time (ties: mean, then val loss); "
                         "'clean' = median clean time (melt OR curtains, whichever comes first); "
                         "'clean2' (v3.1) = median clean2 time (melt OR curtains OR ripples)")
    # ---- v3.1 switches (all OFF / unchanged by default, so a plain --resume behaves as before)
    ap.add_argument("--ripple-max", type=float, default=RIPPLE_LEVEL,
                    help="v3.1: streak ratio below this for a whole second = ripples (only feeds clean2)")
    ap.add_argument("--rebar-best", action="store_true",
                    help="v3.1: on --resume, if --best-by changed, first score the existing best.pt with the "
                         "new rule so a new save has to BEAT it (and keep a copy of it as best_by_<old>_step<N>.pt). "
                         "Without this, the first ruler after the switch simply becomes best.pt.")
    ap.add_argument("--sf-walk-only", action="store_true",
                    help="v3.1: recovery-practice examples come only from walking cities (never Wildlife)")
    ap.add_argument("--no-baseline", action="store_true", help="skip scoring v2-best on the ruler at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0)
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # ---- experiment switches (added 2026-10-19 23:30 for train/_exp; all OFF by default)
    ap.add_argument("--exp-root", default="", help="exp: put checkpoints/, out/ and live/ under this folder")
    ap.add_argument("--init-v3", default="", help="exp: NEW run warm-started from a v3 checkpoint "
                    "(its raw weights, EMA weights and optimizer state; step counter starts at 0)")
    ap.add_argument("--tag", default="v3",
                    help="name prefix for this run's live folder and videos, e.g. v4 "
                         "(so a new run never overwrites v3_rollout_step*.mp4)")
    ap.add_argument("--dream-start", choices=["noise", "prev"], default="noise",
                    help="v5: where every dreamed frame starts in the ruler + rollouts: 'prev' = newest memory "
                         "frame + noise (the 2026-10-20 root-cause fix), 'noise' = pure static (v0-v4)")
    ap.add_argument("--sf-start", choices=["noise", "prev"], default="noise",
                    help="v5: same choice for the recovery-practice dreams, so it practises fixing the mistakes "
                         "it REALLY makes when dreaming")
    ap.add_argument("--decay-hours", type=float, default=0,
                    help="v4: cosine learning-rate slowdown over this many training hours (0 = constant, as before)")
    ap.add_argument("--lr-min-frac", type=float, default=0.1, help="v4: the slowdown ends at this share of --lr")
    ap.add_argument("--sf-start-step", type=int, default=0,
                    help="v4: no recovery practice before this step (a from-scratch model's own dreams are static at first)")
    ap.add_argument("--ruler-starts", default="",
                    help='v4: exact fair-ruler start frames for a NEW run, e.g. "47706,234507,517230,151029,441350" '
                         "(on --resume the run's saved walks are kept)")
    ap.add_argument("--channels", default="", help='exp: U-Net widths for a NEW run, e.g. "96,192,384,512"')
    ap.add_argument("--sf-shared-pool", action="store_true", help="exp: CUDA-graph recordings share one memory pool")
    ap.add_argument("--val-every", type=int, default=0, help="exp: log EMA + raw val loss every N steps")
    args = ap.parse_args()

    base = HERE / "_smoke" if args.smoke else HERE
    if args.exp_root:
        base = pathlib.Path(args.exp_root)
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "live_v3" if args.smoke else HERE / "live"
    if args.exp_root:
        live_root = base / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    elif not args.allow_low_vram:
        # Room check (see train_v2.py): Windows doesn't say "out of memory", it crawls.
        # Measured 2026-10-19 for v3: ~0.30 GB per example + ~2.5 GB fixed.
        free = gpu_free_gb()
        need = 0.30 * args.batch + 2.5
        if free is not None and free < need:
            sys.exit(f"Only {free:.1f} GB of GPU memory is free, and batch {args.batch} needs about {need:.0f} GB.\n"
                     f"Is another training run (or a play window) still running? Stop it first, "
                     f"or use a smaller --batch, or --allow-low-vram to start anyway.")
    torch.manual_seed(args.seed)

    # ---- checkpoint first (a resumed run keeps its own memory layout and ruler)
    latest = ckpt_dir / "latest.pt"
    ck = None
    if args.resume:
        if not latest.exists():
            sys.exit(f"--resume: no checkpoint at {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
    elif latest.exists():
        sys.exit(f"{latest} already exists. Use --resume to continue it, or pick another --run name.")

    # ---- data
    index, frames_mm, actions_np, cut_np = load_cache_any(cache_dir)
    frame_hw = list(index["shape"][1:3])
    model_cfg = ck["model_cfg"] if ck else dict(DEFAULT_CFG_V3, K=args.K, far_n=args.far_n,
                                                 far_stride=args.far_stride, frame_hw=frame_hw)
    v3_init = None
    if not ck and args.init_v3:
        v3_init = torch.load(args.init_v3, map_location="cpu", weights_only=False)
        model_cfg = dict(v3_init["model_cfg"])
    elif not ck and args.channels:
        model_cfg["channels"] = [int(x) for x in args.channels.split(",")]
    if list(model_cfg["frame_hw"]) != frame_hw:
        sys.exit(f"the model dreams {model_cfg['frame_hw']} frames but {cache_dir} holds {frame_hw}")
    model = build_model_v3(model_cfg).to(device)
    K, Fn, S = model.K, model.far_n, model.far_stride
    Hn = model.history                                   # 48 frames of memory
    sf_w = [float(x) for x in args.sf_weights.split(",")]
    SF_MAX = len(sf_w)                                   # up to 4 dreamed recent frames
    if SF_MAX > K:
        sys.exit("--sf-weights can't dream more frames than the recent memory holds")
    L = Hn + SF_MAX                                      # window = L + 1 frames; target at index L
    T = sum(n for _, n in FAIR_SCRIPT)
    ruler_keys = [a for a, n in FAIR_SCRIPT for _ in range(n)]

    valid = compute_valid(actions_np, cut_np, index["cities"], L)
    train_ok = valid.copy()
    held_ok = np.zeros_like(valid)
    held_frames = np.zeros(len(valid), bool)
    bad_tr = (actions_np < 0) | cut_np
    holds = {}
    for c in index["cities"]:
        n = c["frames"]
        hold_n = min(max(int(args.holdout * n), Hn + T + 40), n // 3)
        h0 = c["end"] - hold_n
        holds[c["name"]] = h0
        train_ok[max(c["start"], h0 - L):c["end"]] = False
        held_ok[h0:c["end"]] = valid[h0:c["end"]]
        held_frames[h0:c["end"]] = True
    if train_ok.sum() == 0:
        sys.exit("No valid training windows.")

    # the fair ruler's walks: for each chosen city, the first moment in its held-out tail with
    # 4.8 s of clean history (no cut, keys known) and 20 s of real video after it
    ruler = pick_ruler_cities(index["cities"], args.ruler_cities, names=ck.get("ruler_cities") if ck else None)
    starts = []
    if args.ruler_starts and not ck:
        # v4: hand-picked walks (Singapore's automatic start faces a blank wall and always scores 1.0 s)
        starts = [int(x) for x in args.ruler_starts.split(",") if x.strip()]
        ruler = [next(c for c in index["cities"] if c["start"] <= s < c["end"]) for s in starts]
    for c in ([] if starts else ruler):
        s = holds[c["name"]] + Hn + 5
        while s + T <= c["end"] and bad_tr[s - Hn:s - 1].any():
            s += 1
        if s + T > c["end"]:
            sys.exit(f"no clean fair-ruler start in {c['name']}'s held-out tail")
        starts.append(s)
    row_starts = [s for _ in range(args.ruler_seeds) for s in starts]
    real_clip = np.array(frames_mm[starts[0]:starts[0] + T])
    held_ix = np.flatnonzero(held_frames)
    sharp_ref = float(sharpness(np.array(frames_mm[held_ix[::2]])).mean())

    # sampling weights: by source (hours^(1-balance) x source weight), x2 for turns etc. as before
    src_w = parse_source_weights(args.source_weights)
    # (v3.1 note) The weights ALWAYS come from this command line, also on --resume: the checkpoint
    # remembers the old --source-weights text only so we can say below when it changed.
    src_notes = []
    if ck:
        old_text = ck.get("args", {}).get("source_weights", "")
        if parse_source_weights(old_text) != src_w:
            src_notes.append(f"source weights CHANGED on resume: before {parse_source_weights(old_text)} "
                             f"(--source-weights \"{old_text}\"), now {src_w} (--source-weights \"{args.source_weights}\")")
    unknown = [n for n in src_w if n not in {c["name"] for c in index["cities"]}]
    if unknown:
        src_notes.append(f"WARNING: --source-weights names {unknown} match no source in the cache "
                         f"(sources: {[c['name'] for c in index['cities']]}); those weights do nothing")
    usable ={c["name"]: int(train_ok[c["start"]:c["end"]].sum()) for c in index["cities"]}
    shares = source_shares(index["cities"], usable, src_w, args.city_balance)
    target_key = np.full(len(actions_np), -1, np.int64)
    target_key[:len(actions_np) - L + 1] = actions_np[L - 1:]
    weights = train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight)
    for c in index["cities"]:
        w = weights[c["start"]:c["end"]]
        if w.sum() > 0:
            w *= shares[c["name"]] / w.sum()

    # ---- model, EMA, optimizer
    n_params = sum(p.numel() for p in model.parameters())
    init_notes, v2_ck = [], None
    if v3_init is not None:
        model.load_state_dict(v3_init["model"])
        init_notes = [f"exp --init-v3 {args.init_v3}: v3 run {v3_init['run_id']} step {v3_init['step']} "
                      f"(raw + EMA weights + optimizer state)"]
    elif not ck and args.init.lower() != "none":
        v2_ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if v2_ck["model_cfg"].get("version") != "v2":
            sys.exit(f"--init {args.init} is not a v2 checkpoint (version {v2_ck['model_cfg'].get('version')})")
        init_notes = warm_start_from_v2(model, v2_ck["ema"])
        init_notes.insert(0, f"from {args.init}: v2 run {v2_ck['run_id']}, step {v2_ck['step']}")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), fused=True)

    step, train_seconds = 0, 0.0
    run_id = f"{args.tag}-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline = None             # v2-best's fair-ruler numbers
    best = None                 # {"step", "melt_median", "melt_mean", "val"} of best.pt
    warm_check = None
    warm_started = v2_ck is not None
    if v3_init is not None:
        ema.load_state_dict(v3_init["ema"])
        opt.load_state_dict(v3_init["opt"])
        warm_started = True
        del v3_init
    if ck:
        warm_started = ck.get("warm_started", False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline, best, warm_check = ck.get("v2_baseline"), ck.get("best"), ck.get("warm_check")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    names = ", ".join(c["name"] for c in index["cities"])
    live = Live(live_root / ((args.run + "_" + run_id) if args.exp_root else run_id), {
        "run": run_id, "kind": "train", "image": f"dream model {args.tag}",
        "prompt": f"dream {args.tag} ({names}), {n_params / 1e6:.0f}M params, {frame_hw[0]}x{frame_hw[1]}, "
                  f"memory {K} recent + {Fn} far every {S}",
        "pose": pose_string(FAIR_SCRIPT), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "melt_median": None, "melt_mean": None, "streak": None,
        "v2_melt_median": baseline["melt_median"] if baseline else None,
        "best_step": best["step"] if best else None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v3 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.2f}M params; {frame_hw[0]}x{frame_hw[1]}; memory {K} recent + {Fn} far "
             f"every {S} frames ({Hn / 10:.1f} s); window {L + 1} frames; batch {args.batch}; "
             f"{int(train_ok.sum())} train windows, {int(held_ok.sum())} held-out; cache {cache_dir}")
    live.log("sources (share of training draws): " + ", ".join(
        f"{c['name']} {c['frames'] / 36000:.2f} h -> {shares[c['name']]:.1%}"
        + (f" (weight {src_w[c['name']]})" if c["name"] in src_w else "") for c in index["cities"]))
    for n in src_notes:
        live.log(n)
    if args.sf_walk_only:
        live.log("v3.1 --sf-walk-only: recovery-practice rows are taken only from walking cities "
                 f"(never {sorted(NON_WALKING)})")
    if args.best_by == "clean2" or args.ripple_max != RIPPLE_LEVEL:
        live.log(f"v3.1: best.pt chases median {args.best_by} time; ripple level {args.ripple_max} "
                 f"(streak ratio below it for a whole second = ripples)")
    counts = sf_counts(int(round(args.sf_frac * args.batch)), sf_w)
    live.log(f"recovery practice: {sum(counts)} of {args.batch} examples per batch; with 1/2/3/4 dreamed "
             f"frames: {counts}; {args.sf_steps} sampler steps. Colour fingerprint blanked for "
             f"{args.colour_drop:.0%}. Target noise: {1 - args.sigma_low_frac:.0%} log-normal({args.p_mean}, "
             f"{args.p_std}) + {args.sigma_low_frac:.0%} low bump ({args.sigma_low_mean}, {args.sigma_low_std}). "
             f"Context noise log-normal({args.cn_mean}, {args.cn_std}) max {args.cn_max}.")
    live.log(f"fair ruler: {len(starts)} walks ({', '.join(c['name'] for c in ruler)}) x {args.ruler_seeds} "
             f"noise draws, {T / 10:.0f} s each, {args.rollout_steps} sampler steps, raw; starts at cache "
             f"frames {starts}; melt meter's 1.0 = each walk's own real 20 s: "
             f"{[round(walk_sharp_ref(frames_mm, s, T), 2) for s in starts]} (all held-out video: {sharp_ref:.2f})")
    for n in init_notes:
        live.log("warm start: " + n)
    live.update(stage="starting")

    # ---- loader
    # (v3.1 --sf-walk-only: the "Tagged" dataset also hands back each window's start frame, and
    #  walking_frame[t] says whether frame t belongs to a walking city, so the loop can put the
    #  walking examples first, where the recovery-practice rows are)
    ds = (WindowDatasetV3Tagged if args.sf_walk_only else WindowDatasetV3)(cache_dir, L, flip_p=args.flip_p)
    walking_frame = torch.zeros(len(actions_np), dtype=torch.bool)
    for c in index["cities"]:
        if c["name"] not in NON_WALKING:
            walking_frame[c["start"]:c["end"]] = True
    sampler =InfiniteWeightedSampler(weights, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch if args.workers > 0 else None)

    # fixed held-out batch (no flips) for val loss + the warm-start check
    held_idx = np.flatnonzero(held_ok)
    rs = np.random.default_rng(0)
    pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
    val_ds = WindowDatasetV3(cache_dir, L, flip_p=0.0)
    vb = [val_ds[int(t)] for t in pick]
    val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]))

    def run_val(m=None):
        """EMA loss on 64 fixed held-out windows at 5 fixed noise levels (memory smudge 0.05),
        real memory, colour fingerprint present. Same idea as v2's val loss (same noise levels),
        but not the same windows, so the numbers aren't exactly comparable."""
        m = ema if m is None else m              # (exp --val-every also scores the raw weights)
        tot, sig_list = 0.0, [0.05, 0.2, 0.5, 1.0, 2.0]
        B = val_batch[0].shape[0]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, B, 16):
                    fr = to_model_space(val_batch[0][i:i + 16].to(device))
                    act = val_batch[1][i:i + 16].to(device)
                    b = fr.shape[0]
                    ctx, a, far = mem_inputs(ema, fr, act, L)
                    col = colour_fingerprint(fr[:, :FP_FRAMES])
                    for s in sig_list:
                        tot += m.loss(fr[:, L], ctx, a, torch.full((b,), s, device=device), 0.05,
                                        col, far).item() * b / B
        return tot / len(sig_list)

    def save_ckpt(path):
        try:
            tmp = str(path) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                        "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg,
                        "v2_baseline": baseline, "best": best, "warm_check": warm_check,
                        "ruler_cities": [c["name"] for c in ruler], "ruler_starts": starts,
                        "sharp_ref": sharp_ref, "warm_started": warm_started}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def run_ruler(m, use_anchor):
        t0 = time.time()
        dreams = dream_walks(m, frames_mm, actions_np, starts, args.ruler_seeds, ruler_keys,
                             args.rollout_steps, device, anchor=args.anchor if use_anchor else 0.0,
                             start=args.dream_start)
        n = len(row_starts)
        summ, reps = score_walks(dreams[:n], row_starts, frames_mm, K, T, args.ripple_max)
        anc = None
        if use_anchor and args.anchor > 0:
            arep, acurve = melt_report(dreams[n], walk_sharp_ref(frames_mm, starts[0], T),
                                        np.array(frames_mm[starts[0] - K:starts[0]]))
            anc = (dreams[n], arep, acurve)
        return dreams, summ, reps, anc, time.time() - t0

    def do_rollout():
        """The fair ruler on the EMA weights + videos + live frames + best.pt."""
        nonlocal best
        try:
            dreams, summ, reps, anc, took = run_ruler(ema, True)
            write_live_frames(live, dreams[0], ruler_keys)
            path = write_mp4(out_dir / f"{args.tag}_rollout_step{step:07d}.mp4", dreams[0], ruler_keys, real_clip,
                             f"v3 step {step}", reps[0][1])
            anc_txt = ""
            if anc is not None:
                write_mp4(out_dir / f"{args.tag}_anchor_rollout_step{step:07d}.mp4", anc[0], ruler_keys, real_clip,
                          f"v3 {step} anchor {args.anchor}", anc[2])
                anc_txt = f"; anchored twin of walk 0: melt {secs(anc[1]['melt_s'])}"
            v = run_val()
            live.log(f"rollout at step {step}: {len(dreams)} dreams x {T} frames in one batch, "
                     f"{args.rollout_steps} sampler steps, in {took:.1f}s -> {path.name}; val loss {v:.4f}")
            live.log(ruler_line(f"{args.tag} step {step} (fair ruler)", summ, T)
                     + (f"\n    [v2-best on the same ruler: median {baseline['melt_median']} s, mean "
                        f"{baseline['melt_mean']} s; clean median {baseline.get('clean_median')} s]"
                        if baseline else "") + anc_txt)
            # best.pt: higher median wins; ties -> higher mean; ties -> lower val loss
            key = args.best_by                       # "melt", "clean" or (v3.1) "clean2"
            score = (summ[key + "_median"], summ[key + "_mean"], -v)
            is_best = (best is None or best.get("by", "melt") != key
                       or score > (best[key + "_median"], best[key + "_mean"], -best["val"]))
            if is_best:
                if args.rebar_best and best is not None and best.get("by", "melt") != key:
                    keep_old_best(best)              # v3.1: never lose the old rule's best.pt
                best = {"step": step, "by": key, "val": v, "melt_median": summ["melt_median"],
                        "melt_mean": summ["melt_mean"], "clean_median": summ["clean_median"],
                        "clean_mean": summ["clean_mean"], "clean2_median": summ["clean2_median"],
                        "clean2_mean": summ["clean2_mean"]}
                save_ckpt(ckpt_dir / "best.pt")
                live.log(f"NEW BEST by {key} time (median {summ[key + '_median']} s, mean "
                         f"{summ[key + '_mean']} s) -> best.pt")
            live.metric(dict({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name,
                              "is_best": is_best, "anchor_melt_s": anc[1]["melt_s"] if anc else None}, **summ))
            live.update(last_rollout_step=step, val_loss=v, melt_s=summ["melt_median"],
                        melt_median=summ["melt_median"], melt_mean=summ["melt_mean"], streak=summ["streak"],
                        clean_median=summ["clean_median"],
                        clean2_median=summ["clean2_median"], clean2_mean=summ["clean2_mean"],
                        best_step=best["step"])
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    def keep_old_best(old):
        """v3.1 (--rebar-best): before a best.pt chosen by an OLD rule is replaced, keep a copy
        named after that rule and step, e.g. best_by_clean_step0015011.pt."""
        src = ckpt_dir / "best.pt"
        dst = ckpt_dir / f"best_by_{old.get('by', 'melt')}_step{old['step']:07d}.pt"
        if src.exists() and not dst.exists():
            try:
                shutil.copyfile(src, dst)
                live.log(f"kept the old best.pt (chosen by {old.get('by', 'melt')} time) as {dst.name}")
            except Exception:
                live.log(f"could not copy best.pt to {dst.name}:\n" + traceback.format_exc())

    # ---- v3.1 --rebar-best: a resume that switches --best-by (e.g. clean -> clean2) re-scores the
    # existing best.pt with the NEW rule first. Without this, the old "best" was measured with a
    # different ruler, so the trainer can't compare and simply lets the next ruler become best.pt
    # (even if it is worse). With it, best.pt stays until a new save truly beats it.
    if (args.rebar_best and best is not None and best.get("by", "melt") != args.best_by
            and (ckpt_dir / "best.pt").exists()):
        live.update(stage="re-scoring best.pt")
        try:
            keep_old_best(best)
            bck = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
            old_m = build_model_v3(bck["model_cfg"]).to(device).eval()
            old_m.load_state_dict(bck["ema"])
            bstep = bck["step"]
            del bck
            _, summ, _, _, took = run_ruler(old_m, False)
            del old_m
            live.log(ruler_line(f"OLD best.pt (step {bstep}, chosen by {best.get('by', 'melt')} time) re-scored "
                                f"for --best-by {args.best_by} ({took:.0f}s)", summ, T))
            best = {"step": bstep, "by": args.best_by, "val": best["val"], "rebarred_from": best.get("by", "melt"),
                    **{k: summ[k] for k in ("melt_median", "melt_mean", "clean_median", "clean_mean",
                                            "clean2_median", "clean2_mean")}}
            live.log(f"best.pt bar is now median {args.best_by} {best[args.best_by + '_median']} s, mean "
                     f"{best[args.best_by + '_mean']} s: best.pt (step {bstep}) stays until a ruler beats that")
            live.metric({"step": step, "time": time.time(), "rebar": best})
            live.update(best_step=bstep)
        except Exception:
            live.log("re-scoring best.pt FAILED (the first ruler will set the bar instead):\n" + traceback.format_exc())
        torch.cuda.empty_cache()

    # ---- start-up checks on a NEW warm-started run
    if v2_ck is not None:
        live.update(stage="warm-start check")
        v2 = build_model_v2(v2_ck["model_cfg"]).to(device).eval()
        v2.load_state_dict(v2_ck["ema"])
        # Full float32, no TF32 shortcuts, so any difference is real and not rounding.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            fr = to_model_space(val_batch[0][:16].to(device))
            act = val_batch[1][:16].to(device)
            B = fr.shape[0]
            ctx, a, far = mem_inputs(model, fr, act, L)
            col = colour_fingerprint(fr[:, :FP_FRAMES])
            tgt = fr[:, L]
            res = {}
            for s in [0.1, 0.5, 2.0]:
                g = torch.Generator(device=device).manual_seed(0)
                xn = tgt + s * torch.randn(tgt.shape, device=device, generator=g)
                sig = torch.full((B,), s, device=device)
                d3, _ = model(xn, sig, ctx, 0.0, a, col, far)
                d2, _ = v2(xn, sig, fr[:, L - K:L], 0.0, a)
                res[f"denoise_sigma{s}_maxabs"] = (d3 - d2).abs().max().item()
            p3 = model.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=far,
                              generator=torch.Generator(device=device).manual_seed(1))
            p2 = v2.sample(fr[:, L - K:L], a, steps=args.rollout_steps,
                           generator=torch.Generator(device=device).manual_seed(1))
            res["first_frame_maxabs"] = (p3 - p2).abs().max().item()
            res["first_frame_mse_vs_real"] = torch.mean((p2 - tgt) ** 2).item()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        warm_check = res
        ok = res["first_frame_maxabs"] < 1e-3
        live.log(f"WARM-START CHECK (16 held-out moments, float32): max |v3 - v2| of one clean-up call at noise "
                 f"0.1/0.5/2.0 = {res['denoise_sigma0.1_maxabs']:.2e}/{res['denoise_sigma0.5_maxabs']:.2e}/"
                 f"{res['denoise_sigma2.0_maxabs']:.2e}; of the dreamed first frame ({args.rollout_steps} steps) = "
                 f"{res['first_frame_maxabs']:.2e} (pixel range is 2.0; v2's own error vs the real frame: MSE "
                 f"{res['first_frame_mse_vs_real']:.4f}) -> {'OK: v3 starts out as v2' if ok else 'WARNING: v3 differs from v2'}")
        live.metric({"step": step, "time": time.time(), "warm_check": res})
        if not args.no_baseline:
            live.update(stage="v2-best on the fair ruler")
            dreams, summ, reps, _, took = run_ruler(v2, False)
            tag = f"v2_step{v2_ck['step']:07d}"
            write_mp4(out_dir / f"v3_baseline_{tag}.mp4", dreams[0], ruler_keys, real_clip,
                      f"v2 {v2_ck['step']}", reps[0][1])
            baseline = dict(summ, v2_step=v2_ck["step"])
            live.log(ruler_line(f"v2-best BASELINE (step {v2_ck['step']}, {took:.0f}s)", summ, T))
            real = np.stack([np.array(frames_mm[s:s + T]) for s in starts])
            rsumm, _ = score_walks(real, starts, frames_mm, K, T)
            live.log(ruler_line("real video on the same walks (for scale; should never melt)", rsumm, T))
            live.metric({"step": step, "time": time.time(), "v2_baseline": baseline, "real_ruler": rsumm})
            live.update(v2_melt_median=baseline["melt_median"])
        del v2
        torch.cuda.empty_cache()
    del v2_ck

    # ---- the loop
    # recovery-practice sampler: CUDA-graph recordings of the EMA network (see GraphedSampler)
    if args.no_graphs:
        sf_sampler = eager_sample(ema, start=args.sf_start)
    else:
        graphed = GraphedSampler(ema, shared_pool=args.sf_shared_pool, start=args.sf_start)
        sf_sampler = graphed.sample
        try:
            fr = to_model_space(val_batch[0][:max(1, counts[0])].to(device))
            act = val_batch[1][:fr.shape[0]].to(device)
            ctx, a, far = mem_inputs(ema, fr, act, L)
            diff = graphed.check(ctx, a, colour_fingerprint(fr[:, :FP_FRAMES]), far)
            graphed.graphs.clear()
            if graphed.pool is not None:             # (exp) a fresh pool: the check's recording is gone
                graphed.pool = torch.cuda.graph_pool_handle()
            live.log(f"CUDA-graph check: max |recorded - normal| of one clean-up call = {diff:.2e} "
                     f"({'OK' if diff < 2e-2 else 'WARNING: large; consider --no-graphs'})")
        except Exception:
            live.log("CUDA graphs FAILED, falling back to normal calls:\n" + traceback.format_exc())
            sf_sampler = eager_sample(ema, start=args.sf_start)
    torch.cuda.reset_peak_memory_stats()                 # so "VRAM" below is training, not the start-up checks
    n_sf = sum(counts)
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
            batch = next(it)
            frames_u8, act = batch[0], batch[1]
            fr = to_model_space(frames_u8.to(device, non_blocking=True))    # [B, L+1, 3, H, W]
            act = act.to(device, non_blocking=True)                         # [B, L]
            B = fr.shape[0]
            if args.sf_walk_only:
                # v3.1: reorder the (random) batch so walking-city examples come first. Recovery
                # practice always uses the FIRST rows, so it now only ever sees walking examples.
                # (stable sort: walking rows keep their random order; Wildlife rows go to the end)
                order = torch.argsort((~walking_frame[batch[2]]).to(torch.int8), stable=True).to(device)
                fr, act = fr.index_select(0, order), act.index_select(0, order)

            # 1. colour fingerprint from the window's OLDEST real frames, before anything is swapped
            colour = colour_fingerprint(fr[:, :FP_FRAMES])
            colour = colour * (torch.rand(B, 1, device=device) >= args.colour_drop)      # blank 10%
            # 2. recovery practice: newest 1-4 recent frames -> the EMA model's own dreams
            if n_sf and step >= args.sf_start_step:
                fr = self_force_v3(sf_sampler, ema, fr, act, colour, counts, args.sf_steps)
            # 3. the memory for the target, the noise levels, and one training step
            context, actions, far = mem_inputs(model, fr, act, L)
            target = fr[:, L]
            sigma = sample_sigma(B, args, device)
            sigma_cond = sample_sigma_cond(B, Fn + K, args, device)

            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            if args.decay_hours > 0:
                # v4: full speed at first, then a smooth slowdown (cosine) to lr_min_frac over decay_hours of
                # training. Big steps explore; small steps at the end let it settle into a sharper model.
                prog = min(1.0, train_seconds / 3600.0 / args.decay_hours)   # train_seconds = time spent training (updated every few seconds)
                lr *= args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
            for gq in opt.param_groups:
                gq["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(target, context, actions, sigma, sigma_cond, colour, far)
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
            d = args.ema if (warm_started and not args.ema_ramp) else min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
            step += 1
            win_steps += 1
            loss_avg = lv if loss_avg is None else 0.98 * loss_avg + 0.02 * lv
            if args.val_every and step % args.val_every == 0:
                v_ema, v_raw = run_val(), run_val(model)
                live.metric({"step": step, "time": time.time(), "wall_s": time.time() - t_loop,
                             "val_ema": v_ema, "val_raw": v_raw, "loss": round(loss_avg, 5)})
                live.log(f"exp val at step {step} ({(time.time() - t_loop) / 60:.1f} min): EMA {v_ema:.4f}, "
                         f"raw {v_raw:.4f}, train loss {loss_avg:.4f}")

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
                    shutil.copyfile(latest, ckpt_dir / f"step{step:07d}.pt")
                    last_keep = time.time()
                    live.log(f"keeper checkpoint step{step:07d}.pt saved")

        save_ckpt(latest)
        if step >= args.first_sample_step and live.status.get("last_rollout_step") != step:
            do_rollout()
            save_ckpt(latest)            # so the resumed run knows about the last best
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
