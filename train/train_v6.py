"""Train Dream World v6a: LONGER MEMORY + A DEEPER NETWORK, warm-started from v5.

    # the real run (or just double-click train_v6.bat, which first copies v5's newest best.pt):
    python train/train_v6.py --run dream_v6 --init-v5 train/checkpoints/dream_v6_init.pt ...
    python train/train_v6.py --run dream_v6 --resume ...         # carry on after a stop
    # a small test that leaves the real folders alone (train/_smoke/v6/, live: train/_smoke/live_v6/):
    python train/train_v6.py --smoke --run smoke_v6 --batch 4 --max-steps 200 --first-sample-step 100

WHAT v6a CHANGES (everything else is v5's recipe: train_v5.bat)
  1. LONGER MEMORY (model_v6.py). v5 sees 8 recent frames + 8 far frames (back to 4.8 s). v6a adds
     8 "very far" frames, one every 1.5 s, back to 16.8 s, so a dream can still "see" a street it
     turned away from 10 s ago. The new slots start with zero weights: at step 0 v6a == v5.
  2. A DEEPER NETWORK (model_v6.py, --extra-blocks). One extra residual block per U-Net level on
     the way down and on the way up, each starting as an exact "do nothing". 39.6M -> 51.4M weights.
  3. SPARSE WINDOWS (data_v6.py). Each training example spans 173 frames, but only the 93 that some
     memory slot actually uses are read from disk.
  4. THE TURN-AROUND TEST (new ruler metric, turn_test.py). On the same 5 walks x 2 noise
     draws: walk forward 2 s, turn right for --turn-steps frames, turn back left for as many, and
     ask: does the street you come back to look like the street you left? Scored against a
     "different place" baseline so 0 = "no better than a random other city" and 1 = "identical",
     and against a control that turns away and keeps walking (same moment, never turned back):
     memory = return_sim - control. ~0 = it re-invents the street, > 0 = it remembers it.
     This is the yardstick for v6b (map memory) later.
  5. It also logs the scene score (train/scene_meter.py: how photo-like; real video ~0.78) of
     every ruler, which PLAN.md says to watch as a mean score.

Dream Lab contract (as v3/v5): train/live/v6-<date-time>/ with status.json (image "dream model v6",
plus return_sim, ctrl_sim, turn_memory, away_sim, turned_deg, scene), run.log, metrics.jsonl, frame_*.jpg (walk 0, first
noise draw, raw). Videos: train/out/v6_rollout_step<step>.mp4, v6_anchor_rollout_step<step>.mp4,
and a picture of the turn-around test: train/out/v6_turn_step<step>.jpg.
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
                     load_cache_any, parse_source_weights, source_shares, to_model_space)
from data_v6 import SparseWindowDataset, needed_positions, position_lookup       # noqa: E402
from model_v3 import FP_FRAMES, build_model_v3, colour_fingerprint              # noqa: E402
from model_v6 import (V5_OFFSETS, build_model_v6, v6_cfg_from_v5,               # noqa: E402
                      warm_start_from_v5, warm_start_opt_state)
from scene_meter import features as scene_features                              # noqa: E402
from train import Live, safe_replace                                            # noqa: E402
from train_v1 import melt_report, pose_string, sample_sigma_cond, secs          # noqa: E402
from train_v2 import gpu_free_gb, write_live_frames, write_mp4                  # noqa: E402
from turn_test import TURN_STEPS, TURN_WALK, layout, run_turn_test, turn_line, turn_picture  # noqa: E402
from train_v3 import (FAIR_SCRIPT, RIPPLE_LEVEL, GraphedSampler, eager_sample,   # noqa: E402
                      ruler_line, sample_sigma, score_walks, sf_counts, walk_sharp_ref)

V5_RULER_STARTS = "47706,234507,517230,151029,441350"   # Amsterdam, Istanbul, Zurich, Bangkok, Venice
# The scene meter's spread of real footage (fitted 2026-10-20, train/_exp/): its distances are in
# units of "how much real frames normally differ", per band.
SCENE_FIT = HERE / "_exp" / "scene_meter_fit.npz"

# The turn-around test lives in turn_test.py (shared with its command-line scorer).


# ---------------------------------------------------------------- memory helpers

class MemIndex:
    """Picks the memory for 'dream window position g' out of a SPARSE batch (data_v6.py).
    fr [B, P, 3, H, W] = only the window positions that were read; lut[position] = its row in fr.
    Index lists are worked out once per g and kept (g is always one of L, L-1, ..., L-4)."""

    def __init__(self, model, lut, device):
        self.model, self.lut, self.dev = model, lut, device
        self.rows = {}

    def row(self, g):
        r = int(self.lut[g])
        assert r >= 0, f"window position {g} was not read"
        return r

    def __call__(self, fr, act, g):
        """-> (memory [B, n_mem, 3, H, W] oldest first, recent keys [B, K], key summaries)."""
        m = self.model
        if g not in self.rows:
            r = self.lut[torch.tensor([g - o for o in m.mem_offsets()])]
            assert bool((r >= 0).all()), f"a memory frame for position {g} was not read"
            self.rows[g] = r.to(self.dev)
        ctx = fr.index_select(1, self.rows[g])
        return ctx, act[:, g - m.K:g], m.key_summary(act[:, g - m.history:g])


@torch.no_grad()
def self_force_v6(sampler, mi, fr, act, colour, counts, steps, P):
    """Recovery practice, as train_v3.self_force_v3: rows [0, n4) get 4 dreamed recent frames, ...
    Dream window position P-4 (only the 4-group), then P-3, ..., then P-1 (everyone in practice),
    each from the memory as it stands (which already includes the dreams before it). P = the
    target's window position (L). The only v6 difference: frames live in a sparse tensor, so
    'window position g' is row mi.row(g) of fr."""
    for j in range(len(counts), 0, -1):
        rows = sum(counts[j - 1:])                    # examples with at least j dreamed frames
        if rows <= 0:
            continue
        g = P - j
        ctx, a, far = mi(fr[:rows], act[:rows], g)
        fr[:rows, mi.row(g)] = sampler(ctx, a, steps, colour[:rows], far)
    return fr


# ---------------------------------------------------------------- dreaming walks (ruler + turn test)

@torch.no_grad()
def dream_walks_v6(model, frames_mm, actions_np, row_starts, keys, steps, device, anchor=0.0, seed=1234,
                   start="prev", use_colour=True):
    """Dream one walk per row in ONE batch (+ one colour-anchored twin of row 0 if anchor > 0).
    Like train_v3.dream_walks, for a v6 model: each row starts dreaming at cache frame row_starts[r]
    from the model.history (168) real frames before it. keys = one key list for every row, or one
    list per row (all the same length): the turn-around test dreams its turning rows and its
    controls together. All rows share one fixed-seed generator: every rollout gets the SAME
    noise (fair from rollout to rollout). Returns uint8 [rows, T, H, W, 3]."""
    K, Hn = model.K, model.history
    rows_keys = [list(k) for k in keys] if isinstance(keys[0], (list, tuple)) else [list(keys)] * len(row_starts)
    starts = list(row_starts)
    if anchor > 0:
        starts.append(starts[0])
        rows_keys.append(rows_keys[0])
    T = len(rows_keys[0])
    B = len(starts)
    # The history is a RING BUFFER: 168 frames per row is big (20 rows = 650 MB), so instead of
    # shifting the whole history by one frame every step (a full copy), we overwrite the oldest frame
    # in place. `head` = the slot holding the oldest frame (= where the next dream goes). The frame
    # o frames before the one being dreamed sits in slot (head - o) mod Hn.
    hist = torch.empty((B, Hn, 3, *frames_mm.shape[1:3]), device=device)
    for i, s in enumerate(starts):
        hist[i] = to_model_space(torch.from_numpy(np.array(frames_mm[s - Hn:s])).to(device))
    head = 0
    acts = [[int(a) if a >= 0 else 1 for a in actions_np[s - Hn:s - 1]] for s in starts]
    f0 = Hn - model.fp_offset                                                # the colour fingerprint: same frames as v5
    colour = colour_fingerprint(hist[:, f0:f0 + FP_FRAMES]) if use_colour else None
    ref_mean = hist[-1:, -K:].mean(dim=(1, 3, 4))[:, :, None, None]         # anchor target (twin row)
    ref_std = hist[-1:, -K:].std(dim=(1, 3, 4))[:, :, None, None]
    offs = torch.tensor(model.mem_offsets(), device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(T):
            for r, rk in zip(acts, rows_keys):
                r.append(rk[t])
            kk = torch.tensor(acts, device=device)                           # [B, Hn] keys, last = now
            idx = (head - offs) % Hn                                         # ring-buffer slots, oldest first
            nxt = model.sample(hist.index_select(1, idx), kk[:, -K:], steps=steps, generator=g, start=start,
                               colour=colour, far_act=model.key_summary(kk)).float()
            if anchor > 0:
                x = nxt[-1:]
                m = x.mean(dim=(2, 3), keepdim=True)
                sd = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
                fixed = (x - m) / sd * ref_std + ref_mean
                nxt = torch.cat([nxt[:-1], ((1 - anchor) * x + anchor * fixed).clamp(-1, 1)])
            out.append(((nxt.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            hist[:, head] = nxt                                              # overwrite the oldest frame
            head = (head + 1) % Hn
            for r in acts:
                del r[:-(Hn - 1)]
    return np.stack(out, 1)


def scene_score(dreams, row_starts, frames_mm, sd):
    """train/rescore_all.py's scene score: mean distance of every dreamed frame from the real 4.8 s
    before its walk, in units of real footage's spread. Real video ~0.78 on the 5 ruler walks."""
    o = []
    for d, s in zip(dreams, row_starts):
        mu = scene_features(np.array(frames_mm[s - 48:s])).mean(0)
        o.append(float(np.sqrt((((scene_features(d) - mu) / sd) ** 2).mean(1)).mean()))
    return round(float(np.mean(o)), 3)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v6", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-v5", default=str(HERE / "checkpoints" / "dream_v6_init.pt"),
                    help="v5 checkpoint (model_v3) to warm-start a NEW run from: raw + EMA weights + Adam memory "
                         "('none' = from scratch)")
    ap.add_argument("--smoke", action="store_true",
                    help="checkpoints/out under train/_smoke/v6/, live view in train/_smoke/live_v6/")
    ap.add_argument("--cache", default=None, help="default data/cache_v3")
    # v6 memory + depth (NEW run only; a resumed run keeps its saved layout)
    ap.add_argument("--vfar-n", type=int, default=8, help="v6: very-far memory frames")
    ap.add_argument("--vfar-stride", type=int, default=15, help="v6: frames between very-far frames (15 = 1.5 s)")
    ap.add_argument("--extra-blocks", type=int, default=1, help="v6: identity-start residual blocks added per U-Net level (down and up)")
    ap.add_argument("--colour-drop", type=float, default=0.1)
    # batch / optimiser
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
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
    # target noise (v3/v5)
    ap.add_argument("--p-mean", type=float, default=-0.4)
    ap.add_argument("--p-std", type=float, default=1.2)
    ap.add_argument("--sigma-low-frac", type=float, default=0.2)
    ap.add_argument("--sigma-low-mean", type=float, default=-1.8)
    ap.add_argument("--sigma-low-std", type=float, default=0.8)
    # recovery practice (v5: prev start, 3 steps, walking cities only)
    ap.add_argument("--sf-frac", type=float, default=0.30)
    ap.add_argument("--sf-weights", default="0.4,0.3,0.2,0.1", help="shares with 1,2,3,4 dreamed frames")
    ap.add_argument("--sf-steps", type=int, default=3)
    ap.add_argument("--sf-start", choices=["noise", "prev"], default="prev")
    ap.add_argument("--sf-start-step", type=int, default=0)
    ap.add_argument("--sf-walk-only", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    # context noise (v3/v5), now for all 24 memory frames
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
    ap.add_argument("--ruler-seeds", type=int, default=2)
    ap.add_argument("--ruler-starts", default=V5_RULER_STARTS, help="fair-ruler start frames (NEW run; a resume keeps its own)")
    ap.add_argument("--anchor", type=float, default=0.15)
    ap.add_argument("--best-by", choices=["melt", "clean", "clean2"], default="clean2")
    ap.add_argument("--ripple-max", type=float, default=RIPPLE_LEVEL)
    ap.add_argument("--dream-start", choices=["noise", "prev"], default="prev")
    ap.add_argument("--turn-steps", type=int, default=TURN_STEPS, help="turn-around test: turn steps each way (60 ~ 145 deg)")
    ap.add_argument("--turn-walk", type=int, default=TURN_WALK, help="turn-around test: forward steps before turning")
    ap.add_argument("--no-turn-test", action="store_true", help="skip the turn-around test in rollouts")
    ap.add_argument("--decay-hours", type=float, default=8)
    ap.add_argument("--lr-min-frac", type=float, default=0.1)
    ap.add_argument("--no-baseline", action="store_true", help="skip scoring the warm start (= v5 best) at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0, help="hard cap for this process (smoke tests next to a live run)")
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--ruler-secs", type=int, default=20,
                    help="length of each fair-ruler walk (20 = as v3-v5). v5 reached 20/20 s clean on 2026-10-20, "
                         "so the 20 s ruler is maxed out; 40 repeats the same route pattern to measure further")
    args = ap.parse_args()

    base = HERE / "_smoke" / "v6" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v3"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "live_v6" if args.smoke else HERE / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    elif not args.allow_low_vram:
        # Room check: Windows doesn't say "out of memory", it quietly spills into system RAM and crawls.
        # Estimated 2026-10-20 for v6a (51.4M, 93-frame sparse windows): a train step needs 0.28 GB per
        # example (v5: 0.21, train/_smoke/v6/bench.py) + ~0.02 GB of frames; v5's 20.4 GB at batch 64 says
        # ~7 GB fixed (weights, Adam, CUDA graphs, workspace). Plus 2.5 GB of margin, because other
        # programs (desktop, browser) also use the card and Windows spills silently near the top.
        # -> batch 48 ~ 21.5 GB (allowed), batch 64 ~ 26 GB (refused unless ~29 GB is free).
        free = gpu_free_gb()
        need = 0.30 * args.batch + 7.0
        if free is not None and free < need + 2.5:
            sys.exit(f"Only {free:.1f} GB of GPU memory is free, and batch {args.batch} needs about {need:.0f} GB.\n"
                     f"Is v5 (or a play window) still running? Stop it first (Ctrl+C in its window; it saves), "
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

    # ---- data + model config
    index, frames_mm, actions_np, cut_np = load_cache_any(cache_dir)
    frame_hw = list(index["shape"][1:3])
    v5_ck = None
    if ck:
        model_cfg = ck["model_cfg"]
    elif args.init_v5.lower() != "none":
        v5_ck = torch.load(args.init_v5, map_location="cpu", weights_only=False)
        model_cfg = v6_cfg_from_v5(v5_ck["model_cfg"], args.vfar_n, args.vfar_stride, args.extra_blocks)
    else:
        model_cfg = v6_cfg_from_v5(dict(build_cfg_v5_default(), frame_hw=frame_hw),
                                   args.vfar_n, args.vfar_stride, args.extra_blocks)
    if list(model_cfg["frame_hw"]) != frame_hw:
        sys.exit(f"the model dreams {model_cfg['frame_hw']} frames but {cache_dir} holds {frame_hw}")
    model = build_model_v6(model_cfg).to(device)
    K, Hn = model.K, model.history                       # 8 recent; 168 frames of history
    sf_w = [float(x) for x in args.sf_weights.split(",")]
    SF_MAX = len(sf_w)
    if SF_MAX > K:
        sys.exit("--sf-weights can't dream more frames than the recent memory holds")
    L = Hn + SF_MAX                                      # window = L + 1 frames; target at position L
    # The colour fingerprint in TRAINING: v5 used the oldest 8 frames of its 53-frame window, i.e. the
    # frames 52..45 before the target (fp_offset 48 + SF_MAX 4 = 52). v6 keeps exactly those frames,
    # so the warm start is exact. (In the ruler and in play it's 48..41 before the first dreamed frame,
    # also as v5.)
    fp_train = [model.fp_offset + SF_MAX - i for i in range(FP_FRAMES)]
    positions = needed_positions(model.mem_offsets(), L, SF_MAX, fp_train)
    lut = position_lookup(positions, L)
    # the ruler's route: v5's 20 s pattern, repeated if --ruler-secs asks for longer walks
    ruler_script, t_ = [], 0
    while t_ < args.ruler_secs * 10:
        for a_, n_ in FAIR_SCRIPT:
            n_ = min(n_, args.ruler_secs * 10 - t_)
            if n_ > 0:
                ruler_script.append((a_, n_)); t_ += n_
    T = sum(n for _, n in ruler_script)
    ruler_keys = [a for a, n in ruler_script for _ in range(n)]

    valid = compute_valid(actions_np, cut_np, index["cities"], L)
    n_valid_v5 = int(compute_valid(actions_np, cut_np, index["cities"], 48 + SF_MAX).sum())
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

    # the fair ruler: v5's 5 hand-picked walks (a resume keeps the run's own)
    starts = list(ck["ruler_starts"]) if ck else [int(x) for x in args.ruler_starts.split(",") if x.strip()]
    ruler = [next(c for c in index["cities"] if c["start"] <= s < c["end"]) for s in starts]
    ruler_notes = []
    for s, c in zip(starts, ruler):
        if s - Hn < c["start"] or s + T > c["end"]:
            sys.exit(f"ruler start {s} ({c['name']}) has no room for {Hn} frames of history + {T} frames of walk")
        if bad_tr[s - Hn:s - 1].any():
            ruler_notes.append(f"WARNING: ruler walk {s} ({c['name']}) has a cut/unknown key in its {Hn}-frame history")
        if s - Hn < holds[c["name"]]:
            ruler_notes.append(f"note: ruler walk {s} ({c['name']}): its oldest {holds[c['name']] - (s - Hn)} memory "
                               f"frames come from before the held-out tail (inputs only; every frame it is scored on is held out)")
    row_starts = [s for _ in range(args.ruler_seeds) for s in starts]
    real_clip = np.array(frames_mm[starts[0]:starts[0] + T])
    scene_sd = np.load(SCENE_FIT)["sd"]

    # sampling weights: by source (hours^(1-balance) x source weight), x2 for turns etc. (as v3/v5)
    src_w = parse_source_weights(args.source_weights)
    src_notes = []
    if ck:
        old_text = ck.get("args", {}).get("source_weights", "")
        if parse_source_weights(old_text) != src_w:
            src_notes.append(f"source weights CHANGED on resume: before {parse_source_weights(old_text)}, now {src_w}")
    unknown = [n for n in src_w if n not in {c["name"] for c in index["cities"]}]
    if unknown:
        src_notes.append(f"WARNING: --source-weights names {unknown} match no source in the cache")
    usable = {c["name"]: int(train_ok[c["start"]:c["end"]].sum()) for c in index["cities"]}
    shares = source_shares(index["cities"], usable, src_w, args.city_balance)
    target_key = np.full(len(actions_np), -1, np.int64)
    target_key[:len(actions_np) - L + 1] = actions_np[L - 1:]
    weights = train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight)
    for c in index["cities"]:
        w = weights[c["start"]:c["end"]]
        if w.sum() > 0:
            w *= shares[c["name"]] / w.sum()

    # ---- model, EMA, optimizer (+ the warm start from v5)
    n_params = sum(p.numel() for p in model.parameters())
    init_notes, how = [], None
    if v5_ck is not None:
        init_notes, how = warm_start_from_v5(model, v5_ck["model"])
        init_notes.insert(0, f"from {args.init_v5}: v5 run {v5_ck['run_id']} step {v5_ck['step']} "
                             f"({v5_ck['params'] / 1e6:.2f}M params) -> v6a {n_params / 1e6:.2f}M")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if v5_ck is not None:
        warm_start_from_v5(ema, v5_ck["ema"])            # the EMA weights (what dreams) get v5's EMA

    def make_opt(m):
        decay, no_decay = [], []
        for name, p in m.named_parameters():
            (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=args.lr, betas=(0.9, 0.99), fused=True)
    opt = make_opt(model)
    if v5_ck is not None and "opt" in v5_ck:
        try:
            # v5's optimizer listed its weights in train_v3's order (same grouping rule as make_opt)
            v5_shape = build_model_v3(v5_ck["model_cfg"])
            names5 = [n for n, p in v5_shape.named_parameters() if not (p.ndim < 2 or "emb" in n)] + \
                     [n for n, p in v5_shape.named_parameters() if p.ndim < 2 or "emb" in n]
            del v5_shape
            n_carried = warm_start_opt_state(opt, model, v5_ck["opt"], names5, how)
            init_notes.append(f"Adam memory carried over from v5 for {n_carried} of "
                              f"{sum(1 for _ in model.parameters())} weight tensors (the rest start fresh)")
        except Exception:
            opt = make_opt(model)
            init_notes.append("could not carry v5's Adam memory over, starting it fresh:\n" + traceback.format_exc())

    step, train_seconds = 0, 0.0
    run_id = f"{args.tag}-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline = None             # the warm start (= v5 best) on the ruler + turn test
    best = None
    warm_check = None
    warm_started = v5_ck is not None
    if ck:
        warm_started = ck.get("warm_started", False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline, best, warm_check = ck.get("v5_baseline"), ck.get("best"), ck.get("warm_check")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    names = ", ".join(c["name"] for c in index["cities"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": f"dream model {args.tag}",
        "prompt": f"dream {args.tag} ({names}), {n_params / 1e6:.0f}M params, {frame_hw[0]}x{frame_hw[1]}, "
                  f"memory {model.n_mem} frames back to {Hn / 10:.1f} s",
        "pose": pose_string(ruler_script), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "melt_median": None, "melt_mean": None, "streak": None,
        "return_sim": None, "away_sim": None, "ctrl_sim": None, "turn_memory": None, "scene": None,
        "v5_return_sim": baseline["turn"]["return_sim"] if baseline and baseline.get("turn") else None,
        "v5_clean2_median": baseline["clean2_median"] if baseline else None,
        "best_step": best["step"] if best else None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v6 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.2f}M params ({model_cfg['extra_blocks']} extra block(s) per level); "
             f"{frame_hw[0]}x{frame_hw[1]}; memory {model.n_mem} frames at offsets {model.mem_offsets()} "
             f"({Hn / 10:.1f} s); window {L + 1} frames, of which {len(positions)} are read; batch {args.batch}; "
             f"{int(train_ok.sum())} train windows, {int(held_ok.sum())} held-out; valid windows {int(valid.sum())} "
             f"(v5's 53-frame windows: {n_valid_v5}, {100 * (int(valid.sum()) / n_valid_v5 - 1):+.1f}%); cache {cache_dir}")
    live.log("sources (share of training draws): " + ", ".join(
        f"{c['name']} {c['frames'] / 36000:.2f} h -> {shares[c['name']]:.1%}"
        + (f" (weight {src_w[c['name']]})" if c["name"] in src_w else "") for c in index["cities"]))
    for n in src_notes + ruler_notes:
        live.log(n)
    counts = sf_counts(int(round(args.sf_frac * args.batch)), sf_w)
    live.log(f"recovery practice: {sum(counts)} of {args.batch} examples per batch; with 1/2/3/4 dreamed "
             f"frames: {counts}; {args.sf_steps} sampler steps, start '{args.sf_start}'"
             f"{', walking cities only' if args.sf_walk_only else ''}. Dreams start '{args.dream_start}'. "
             f"best.pt chases median {args.best_by} time.")
    live.log(f"fair ruler: {len(starts)} walks ({', '.join(c['name'] for c in ruler)}) x {args.ruler_seeds} "
             f"noise draws, {T / 10:.0f} s each, {args.rollout_steps} sampler steps, raw; starts {starts}. "
             f"Turn-around test: {'off' if args.no_turn_test else f'W x {args.turn_walk}, turn right x {args.turn_steps}, turn left x {args.turn_steps}'}")
    for n in init_notes:
        live.log("warm start: " + n)
    live.update(stage="starting")

    # ---- loader
    ds = SparseWindowDataset(cache_dir, positions, L, flip_p=args.flip_p)
    walking_frame = torch.zeros(len(actions_np), dtype=torch.bool)
    for c in index["cities"]:
        if c["name"] not in NON_WALKING:
            walking_frame[c["start"]:c["end"]] = True
    sampler = InfiniteWeightedSampler(weights, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch if args.workers > 0 else None)
    mi = MemIndex(model, lut, device)                    # same offsets for model and ema
    mi_ema = MemIndex(ema, lut, device)
    fp_rows = lut[torch.tensor([L - o for o in fp_train])].to(device)
    target_row = int(lut[L])

    # fixed held-out batch (no flips) for val loss + the warm-start check
    held_idx = np.flatnonzero(held_ok)
    rs = np.random.default_rng(0)
    pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
    val_ds = SparseWindowDataset(cache_dir, positions, L, flip_p=0.0)
    vb = [val_ds[int(t)] for t in pick]
    val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]))

    def run_val(m=None):
        """EMA loss on 64 fixed held-out windows at 5 fixed noise levels (memory smudge 0.05), real
        memory, colour fingerprint present: the same recipe as v3/v5's val loss (not the same windows)."""
        m = ema if m is None else m
        tot, sig_list = 0.0, [0.05, 0.2, 0.5, 1.0, 2.0]
        B = val_batch[0].shape[0]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, B, 8):
                    fr = to_model_space(val_batch[0][i:i + 8].to(device))
                    act = val_batch[1][i:i + 8].to(device)
                    b = fr.shape[0]
                    ctx, a, far = mi_ema(fr, act, L)
                    col = colour_fingerprint(fr.index_select(1, fp_rows))
                    for s in sig_list:
                        tot += m.loss(fr[:, target_row], ctx, a, torch.full((b,), s, device=device), 0.05,
                                      col, far).item() * b / B
        return tot / len(sig_list)

    def save_ckpt(path):
        try:
            tmp = str(path) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                        "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg,
                        "v5_baseline": baseline, "best": best, "warm_check": warm_check,
                        "ruler_cities": [c["name"] for c in ruler], "ruler_starts": starts,
                        "warm_started": warm_started}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def run_ruler(m, use_anchor):
        t0 = time.time()
        dreams = dream_walks_v6(m, frames_mm, actions_np, row_starts, ruler_keys, args.rollout_steps, device,
                                anchor=args.anchor if use_anchor else 0.0, start=args.dream_start)
        n = len(row_starts)
        summ, reps = score_walks(dreams[:n], row_starts, frames_mm, K, T, args.ripple_max)
        summ["scene"] = scene_score(dreams[:n], row_starts, frames_mm, scene_sd)
        anc = None
        if use_anchor and args.anchor > 0:
            arep, acurve = melt_report(dreams[n], walk_sharp_ref(frames_mm, starts[0], T),
                                        np.array(frames_mm[starts[0] - K:starts[0]]))
            anc = (dreams[n], arep, acurve)
        return dreams, summ, reps, anc, time.time() - t0

    def run_turn(m):
        """The turn-around test (turn_test.py): turning rows and their turned-away controls in ONE batch."""
        t0 = time.time()
        td, cd, res = run_turn_test(m, frames_mm, actions_np, starts, args.ruler_seeds, args.rollout_steps, device,
                                    scene_features, scene_sd, walk=args.turn_walk, n=args.turn_steps,
                                    start=args.dream_start)
        return td, cd, res, time.time() - t0

    def do_rollout():
        """Fair ruler + turn-around test on the EMA weights, videos, live frames, best.pt."""
        nonlocal best
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
                     + (f"\n    [v5 best on the same ruler (= v6 at init): clean2 median {baseline['clean2_median']} s, "
                        f"mean {baseline['clean2_mean']} s; scene {baseline.get('scene')}]" if baseline else "") + anc_txt)
            turn = None
            if not args.no_turn_test:
                td, sd_, turn, ttook = run_turn(ema)
                turn_picture(out_dir / f"{args.tag}_turn_step{step:07d}.jpg", td, sd_, len(starts),
                             f"{args.tag} step {step}", args.turn_walk, args.turn_steps)
                live.log(turn_line(f"{args.tag} step {step} turn-around test ({ttook:.0f}s)", turn)
                         + (f"\n    [v5 best: return_sim {baseline['turn']['return_sim']}, control "
                            f"{baseline['turn']['ctrl_sim']}, memory {baseline['turn']['memory']:+.3f}]"
                            if baseline and baseline.get("turn") else ""))
            key = args.best_by
            score = (summ[key + "_median"], summ[key + "_mean"], -v)
            is_best = (best is None or best.get("by", "melt") != key
                       or score > (best[key + "_median"], best[key + "_mean"], -best["val"]))
            if is_best:
                best = {"step": step, "by": key, "val": v, "melt_median": summ["melt_median"],
                        "melt_mean": summ["melt_mean"], "clean_median": summ["clean_median"],
                        "clean_mean": summ["clean_mean"], "clean2_median": summ["clean2_median"],
                        "clean2_mean": summ["clean2_mean"], "scene": summ["scene"],
                        "return_sim": turn["return_sim"] if turn else None}
                save_ckpt(ckpt_dir / "best.pt")
                live.log(f"NEW BEST by {key} time (median {summ[key + '_median']} s, mean "
                         f"{summ[key + '_mean']} s) -> best.pt")
            live.metric(dict({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name,
                              "is_best": is_best, "anchor_melt_s": anc[1]["melt_s"] if anc else None,
                              "turn": turn}, **summ))
            live.update(last_rollout_step=step, val_loss=v, melt_s=summ["melt_median"],
                        melt_median=summ["melt_median"], melt_mean=summ["melt_mean"], streak=summ["streak"],
                        clean_median=summ["clean_median"], clean2_median=summ["clean2_median"],
                        clean2_mean=summ["clean2_mean"], scene=summ["scene"],
                        return_sim=turn["return_sim"] if turn else None,
                        away_sim=turn["away_sim"] if turn else None,
                        ctrl_sim=turn["ctrl_sim"] if turn else None,
                        turn_memory=turn["memory"] if turn else None,
                        turned_deg=turn["turned_right_deg"] if turn else None,
                        return_sim_scene=turn["return_sim_scene"] if turn else None,
                        best_step=best["step"])
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    # ---- start-up checks on a NEW warm-started run
    if v5_ck is not None:
        live.update(stage="warm-start check")
        v5 = build_model_v3(v5_ck["model_cfg"]).to(device).eval()
        v5.load_state_dict(v5_ck["ema"])
        # Full float32, no TF32 shortcuts, so any difference is real and not rounding.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        slot5 = torch.tensor([model.mem_offsets().index(o) for o in V5_OFFSETS], device=device)
        with torch.no_grad():
            fr = to_model_space(val_batch[0][:16].to(device))
            act = val_batch[1][:16].to(device)
            B = fr.shape[0]
            ctx, a, far = mi_ema(fr, act, L)
            ctx5, far5 = ctx.index_select(1, slot5), far[:, :72]          # v5's 16 slots, v5's far keys
            col = colour_fingerprint(fr.index_select(1, fp_rows))
            tgt = fr[:, target_row]
            res = {}
            for s in [0.1, 0.5, 2.0]:
                g = torch.Generator(device=device).manual_seed(0)
                xn = tgt + s * torch.randn(tgt.shape, device=device, generator=g)
                sig = torch.full((B,), s, device=device)
                d6, _ = ema(xn, sig, ctx, 0.0, a, col, far)
                d5, _ = v5(xn, sig, ctx5, 0.0, a, col, far5)
                res[f"denoise_sigma{s}_maxabs"] = (d6 - d5).abs().max().item()
            # the raw (training) weights too, one call
            v5r = build_model_v3(v5_ck["model_cfg"]).to(device).eval()
            v5r.load_state_dict(v5_ck["model"])
            g = torch.Generator(device=device).manual_seed(0)
            xn = tgt + 0.5 * torch.randn(tgt.shape, device=device, generator=g)
            sig = torch.full((B,), 0.5, device=device)
            res["raw_weights_denoise_sigma0.5_maxabs"] = (model.eval()(xn, sig, ctx, 0.0, a, col, far)[0]
                                                          - v5r(xn, sig, ctx5, 0.0, a, col, far5)[0]).abs().max().item()
            model.train()
            del v5r
            p6 = ema.sample(ctx, a, steps=args.rollout_steps, colour=col, far_act=far, start="prev",
                            generator=torch.Generator(device=device).manual_seed(1))
            p5 = v5.sample(ctx5, a, steps=args.rollout_steps, colour=col, far_act=far5, start="prev",
                           generator=torch.Generator(device=device).manual_seed(1))
            res["dream_prev_start_maxabs"] = (p6 - p5).abs().max().item()
            res["dream_mse_vs_real"] = torch.mean((p5 - tgt) ** 2).item()
            res["copy_last_frame_mse"] = torch.mean((ctx[:, -1] - tgt) ** 2).item()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        del v5
        warm_check = res
        ok = res["dream_prev_start_maxabs"] < 1e-3 and res["denoise_sigma2.0_maxabs"] < 1e-3
        live.log(f"WARM-START CHECK (16 held-out moments, float32): max |v6 - v5| of one clean-up call (EMA) at noise "
                 f"0.1/0.5/2.0 = {res['denoise_sigma0.1_maxabs']:.2e}/{res['denoise_sigma0.5_maxabs']:.2e}/"
                 f"{res['denoise_sigma2.0_maxabs']:.2e}; raw weights at 0.5 = {res['raw_weights_denoise_sigma0.5_maxabs']:.2e}; "
                 f"of a full {args.rollout_steps}-step prev-start dream = {res['dream_prev_start_maxabs']:.2e} (pixel range 2.0; "
                 f"v5's own error vs the real frame: MSE {res['dream_mse_vs_real']:.4f}, copying the last frame "
                 f"{res['copy_last_frame_mse']:.4f}) -> {'OK: v6 starts out as v5' if ok else 'WARNING: v6 differs from v5'}")
        live.metric({"step": step, "time": time.time(), "warm_check": res})
        torch.cuda.empty_cache()
        if not args.no_baseline:
            live.update(stage="v5-best on the ruler + turn test")
            dreams, summ, reps, _, took = run_ruler(ema, False)
            write_mp4(out_dir / f"{args.tag}_baseline_v5_step{v5_ck['step']:07d}.mp4", dreams[0], ruler_keys,
                      real_clip, f"v5 {v5_ck['step']}", reps[0][1])
            baseline = dict(summ, v5_step=v5_ck["step"])
            live.log(ruler_line(f"v5-best BASELINE (= v6 at init; v5 step {v5_ck['step']}, {took:.0f}s)", summ, T)
                     + f"\n    scene score {summ['scene']}")
            if not args.no_turn_test:
                td, sd_, turn, ttook = run_turn(ema)
                baseline["turn"] = turn
                turn_picture(out_dir / f"{args.tag}_turn_baseline_v5.jpg", td, sd_, len(starts),
                             f"v5 {v5_ck['step']}", args.turn_walk, args.turn_steps)
                live.log(turn_line(f"v5-best BASELINE turn-around test ({ttook:.0f}s)", turn))
                # the real frames' "different place" distance, for scale: real frame before the turn
                # moment of each walk vs the other walks' (no dreaming involved)
                real_pre = np.stack([np.array(frames_mm[s + args.turn_walk - 1]) for s in starts])
                lr_ = layout(real_pre)
                dd = [float(((lr_[i] - lr_[j]) ** 2).mean()) for i in range(len(starts)) for j in range(len(starts)) if i != j]
                live.log(f"    (for scale: real frames of the 5 walks vs each other, layout distance {np.mean(dd):.4f}; "
                         f"dreamed pre-turn frames vs each other {turn['d_diff_place']})")
            live.metric({"step": step, "time": time.time(), "v5_baseline": baseline})
            live.update(v5_clean2_median=baseline["clean2_median"],
                        v5_return_sim=baseline["turn"]["return_sim"] if baseline.get("turn") else None)
            torch.cuda.empty_cache()
    del v5_ck

    # ---- the loop
    if args.no_graphs:
        sf_sampler = eager_sample(ema, start=args.sf_start)
    else:
        graphed = GraphedSampler(ema, start=args.sf_start)
        sf_sampler = graphed.sample
        try:
            fr = to_model_space(val_batch[0][:max(1, counts[0])].to(device))
            act = val_batch[1][:fr.shape[0]].to(device)
            ctx, a, far = mi_ema(fr, act, L)
            diff = graphed.check(ctx, a, colour_fingerprint(fr.index_select(1, fp_rows)), far)
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
    live.update(stage="training")

    try:
        it = iter(loader)
        while True:
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_hours and (time.time() - t_loop) / 3600 >= args.max_hours:
                break
            frames_u8, act, tix = next(it)
            fr = to_model_space(frames_u8.to(device, non_blocking=True))    # [B, P, 3, H, W] (sparse positions)
            act = act.to(device, non_blocking=True)                         # [B, L] every key of the window
            B = fr.shape[0]
            if args.sf_walk_only:
                # walking-city examples first: recovery practice always uses the FIRST rows
                order = torch.argsort((~walking_frame[tix]).to(torch.int8), stable=True).to(device)
                fr, act = fr.index_select(0, order), act.index_select(0, order)

            # 1. colour fingerprint from v5's frames (52..45 before the target), before anything is swapped
            colour = colour_fingerprint(fr.index_select(1, fp_rows))
            colour = colour * (torch.rand(B, 1, device=device) >= args.colour_drop)      # blank 10%
            # 2. recovery practice: newest 1-4 recent frames -> the EMA model's own dreams
            if n_sf and step >= args.sf_start_step:
                fr = self_force_v6(sf_sampler, mi_ema, fr, act, colour, counts, args.sf_steps, L)
            # 3. the memory for the target, the noise levels, and one training step
            context, actions, far = mi(fr, act, L)
            target = fr[:, target_row]
            sigma = sample_sigma(B, args, device)
            sigma_cond = sample_sigma_cond(B, model.n_mem, args, device)

            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            if args.decay_hours > 0:
                prog = min(1.0, train_seconds / 3600.0 / args.decay_hours)
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
            d = args.ema if warm_started else min(args.ema, (1 + step) / (10 + step))
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


def build_cfg_v5_default():
    """v5's architecture (model_v3 defaults: 8 recent + 8 far every 5), for --init-v5 none."""
    from model_v3 import DEFAULT_CFG_V3
    return dict(DEFAULT_CFG_V3)


if __name__ == "__main__":
    main()
