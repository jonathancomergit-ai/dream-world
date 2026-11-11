"""Train Dream World v2: v1, dreaming at 96 x 168 instead of 64 x 112 (2.25x the pixels).

    # the real run (warm-starts from a copy of v1's save, see model_v2.py):
    python train/train_v2.py --run dream_v2 --init train/checkpoints/dream_v2_init.pt
    python train/train_v2.py --run dream_v2 --resume          # carry on after a stop
    # a small test on the real v2 data that leaves the real folders alone:
    python train/train_v2.py --smoke --run smoke_v2 --batch 16 --max-steps 200 --vram-cap-gb 6 --first-sample-step 100

    (or just double-click train_v2.bat / train_v2_resume.bat in the DreamWorld folder)

WHAT'S NEW COMPARED WITH train_v1.py, in plain words
  1. BIGGER PICTURES. Frames come from data/cache_v2 (96x168, built by build_cache_v2.py) instead
     of data/cache (64x112). Same videos, same frame numbers, same keys: frame i of one cache is
     the same moment as frame i of the other.
  2. WARM START FROM v1. The network is v1's, unchanged (model_v2.py explains why a convolutional
     network doesn't care about picture size). v1's EMA weights are copied into both the live
     weights and the EMA, like v1 did with v0. The optimizer starts fresh, hence the warmup.
  3. A START-UP CHECK, before any training: does v1's brain, now looking at 96x168 pictures,
     still dream a sensible next frame? It measures, on the same 32 held-out moments:
       - v2 at init, at 96x168        vs the real next frame
       - v2 at init, shrunk to 64x112 vs the real next frame (apples to apples with v1)
       - v1 itself, at 64x112         vs the real next frame
       - "just repeat the last memory frame" (a no-brain baseline, so the numbers mean something)
     and saves a picture grid (v2_init_check.png in the out folder) to look at.
     Then the v1 BASELINE: v1 dreams the 30 s test walk at 64x112 so v2's melt times have
     something to beat, and v2-at-init dreams it too (rollout "step 0").
  4. Everything else is v1's recipe, imported from train_v1.py so there's one copy of it:
     recovery practice (self-forcing), the heavier per-frame memory smudge, mirror flips, extra
     turn practice, EMA at full strength from step 0 (no ramp: we start from a good EMA),
     10-step rollouts, raw + colour-anchored videos, the melt meter (checked at 96x168 on
     2026-10-19: real video's "grain" is 0.70 median / 0.93 max, static is 1.2+, so the same
     1.1 line still separates them).
  5. NAMES: run dream_v2, live folder train/live/v2-<date-time>/, videos
     train/out/v2_rollout_step<step>.mp4 and v2_anchor_rollout_step<step>.mp4. The dream is
     shown at 448x256 (the same size the v1 videos used), so the two sit side by side nicely.

Dream Lab (the dashboard) contract is unchanged: status.json, run.log, metrics.jsonl and the
newest dream as frame_*.jpg in the live folder.
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
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from build_cache import compute_valid                                       # noqa: E402
from data_v2 import (ACTION_NAMES, InfiniteWeightedSampler, WindowDatasetV2,  # noqa: E402
                     cache_frame_size, load_cache_any, to_model_space, to_uint8_image)
from model_v1 import build_model_v1                                         # noqa: E402  (for the v1 baseline)
from model_v2 import DEFAULT_CFG_V2, build_model_v2, warm_start_from_v1     # noqa: E402
from train import Live, label, safe_replace                                 # noqa: E402  (reused as-is)
from train_v1 import (EXTRA, ROLLOUT_SCRIPT_V1, melt_line, melt_report,     # noqa: E402  (v1's recipe, as-is)
                      pose_string, rollout_v1, sample_sigma_cond, secs, self_force, sharpness)

DISPLAY_W = 448    # every dream frame is shown this wide in videos and Dream Lab (v1: 112 x 4 = 448)


# ---------------------------------------------------------------- showing frames

def upscale_disp(img):
    """Blow a small frame up to DISPLAY_W wide, keeping its shape: 96x168 -> 256x448 and
    64x112 -> 256x448 too, so v1 and v2 videos come out the same size. INTER_CUBIC = smooth."""
    h, w = img.shape[:2]
    return cv2.resize(img, (DISPLAY_W, round(DISPLAY_W * h / w)), interpolation=cv2.INTER_CUBIC)


def write_live_frames(live, dream, keys):
    """Overwrite frame_*.jpg in the live folder (the dashboard's newest dream).
    (train_v1's version, with the new upscale.)"""
    for i, (fr, a) in enumerate(zip(dream, keys)):
        big = label(upscale_disp(fr), ACTION_NAMES[a])
        dst = live.dir / f"frame_{i:05d}.jpg"
        tmp = live.dir / f"frame_{i:05d}.jpg.tmp"       # not matched by frame_*.jpg: never shown half-written
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            tmp.write_bytes(buf.tobytes())
            safe_replace(tmp, dst)


def write_mp4(path, dream, keys, real, title, sharp):
    """Side-by-side video: real video (left) | dream (right), with the key and the sharpness
    number on every dream frame. (train_v1's version, with the new upscale.)"""
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.mp4")         # .tmp so the dashboard never lists half a video
    w = imageio.get_writer(str(tmp), fps=10, codec="libx264", pixelformat="yuv420p",
                           quality=8, macro_block_size=16, ffmpeg_log_level="error")
    for i, (fr, a) in enumerate(zip(dream, keys)):
        r = real[min(i, len(real) - 1)]
        left = label(upscale_disp(r), "real video")
        right = label(upscale_disp(fr), f"{title}: {ACTION_NAMES[a]}")
        label(right, f"{(i + 1) / 10:4.1f}s  sharp {sharp[i]:.2f}", y=right.shape[0] - 10)
        w.append_data(np.concatenate([left, right], axis=1))
    w.close()
    safe_replace(tmp, path)
    return path


# ---------------------------------------------------------------- the start-up check

@torch.no_grad()
def init_check(model, v1, fr2, fr1, act, steps, png_path, device):
    """Does v2 (= v1's weights looking at 96x168 pictures) still predict a sensible next frame?

    fr2 [B, L+1, 3, 96, 168]  real windows from cache_v2 (model space, -1..1)
    fr1 [B, L+1, 3, 64, 112]  the SAME moments from the v1 cache
    act [B, L]                their keys
    Everything in float32 (no bf16 rounding), fixed random seeds, so the numbers are repeatable.

    MSE = mean squared difference per pixel, in the model's -1..1 scale (0 = identical).
    PSNR = the same thing in decibels, the usual "how close are two pictures" score (higher = closer).
    Returns a dict of numbers and writes a picture grid to png_path."""
    L = fr2.shape[1] - 1
    ctx2, tgt2, ctx1, tgt1, a = fr2[:, EXTRA:L], fr2[:, L], fr1[:, EXTRA:L], fr1[:, L], act[:, EXTRA:L]
    mse = lambda x, y: torch.mean((x.float() - y.float()) ** 2).item()
    small = lambda x: F.interpolate(x, size=tgt1.shape[-2:], mode="area")    # 96x168 -> 64x112 by averaging
    res = {}

    # (a) ONE clean-up call at a few noise levels: bury the real next frame in noise of strength
    #     sigma, ask each model for the clean frame back. (Different picture sizes can't share the
    #     exact same noise, but each uses a fixed seed.)
    for s in [0.1, 0.5, 2.0]:
        g2 = torch.Generator(device=device).manual_seed(0)
        g1 = torch.Generator(device=device).manual_seed(0)
        B = tgt2.shape[0]
        sig = torch.full((B,), s, device=device)
        d2, _ = model(tgt2 + s * torch.randn(tgt2.shape, device=device, generator=g2), sig, ctx2, 0.0, a)
        d1, _ = v1(tgt1 + s * torch.randn(tgt1.shape, device=device, generator=g1), sig, ctx1, 0.0, a)
        res[f"denoise_sigma{s}_v2"] = mse(d2, tgt2)
        res[f"denoise_sigma{s}_v1"] = mse(d1, tgt1)

    # (b) a full dreamed first frame from pure static (the same sampler the rollouts use)
    p2 = model.sample(ctx2, a, steps=steps, generator=torch.Generator(device=device).manual_seed(1))
    p1 = v1.sample(ctx1, a, steps=steps, generator=torch.Generator(device=device).manual_seed(1))
    res["dream_v2_96x168"] = mse(p2, tgt2)
    res["dream_v2_shrunk_64x112"] = mse(small(p2), tgt1)
    res["dream_v1_64x112"] = mse(p1, tgt1)
    res["repeat_last_frame_96x168"] = mse(ctx2[:, -1], tgt2)
    res["repeat_last_frame_64x112"] = mse(ctx1[:, -1], tgt1)
    # how different the two caches' "same" frames are after shrinking (the floor of the shrunk score)
    res["resize_floor"] = mse(small(tgt2), tgt1)

    # (c) the picture: one row per example, columns = newest memory | real next | v2 | v1.
    #     Everything is shown at 192x336 (v2 frames x2, v1 frames x3) so sizes match.
    cols = [("newest memory frame", ctx2[:, -1]), ("REAL next frame", tgt2),
            ("v2 at init (96x168)", p2), ("v1 (64x112)", p1)]
    rows = []
    for i in range(min(6, tgt2.shape[0])):
        cells = []
        for name, t in cols:
            im = cv2.resize(to_uint8_image(t[i]), (336, 192), interpolation=cv2.INTER_CUBIC)
            cells.append(label(im, name if i == 0 else ACTION_NAMES[int(a[i, -1])] if name.startswith("REAL") else ""))
        rows.append(np.concatenate(cells, axis=1))
    grid = np.concatenate(rows, axis=0)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return res


def gpu_free_gb():
    """Free GPU memory in GB as nvidia-smi sees it (all programs), or None if it can't be asked."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        return (total - used) / 1024
    except Exception:
        return None


def psnr(m):
    """MSE in the -1..1 scale -> PSNR in dB (the range is 2, so peak^2 = 4)."""
    return 10 * math.log10(4 / max(m, 1e-12))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v2", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=str(HERE / "checkpoints" / "dream_v2_init.pt"),
                    help="v1 checkpoint to warm-start from on a NEW run ('none' = from scratch)")
    ap.add_argument("--smoke", action="store_true",
                    help="checkpoints/out under train/_smoke/, live view in train/_smoke/live_v2/ (data stays the real cache)")
    ap.add_argument("--cache", default=None, help="default data/cache_v2")
    ap.add_argument("--v1-cache", default=str(DW / "data" / "cache"),
                    help="the 64x112 cache, only for the v1 comparisons at start-up")
    ap.add_argument("--K", type=int, default=8)
    # Batch 64 at 96x168 = about the same GPU work per step as v1's 144 at 64x112 (each example has
    # 2.25x the pixels; FLOP count says 2.26x the work). Measured 2026-10-19 at small batches:
    # memory grows ~0.23 GB per example, so 64 -> ~17 GB reserved (v1 at 128: 14 GB). Expected speed
    # ~3.5 it/s with the GPU to itself (v1's 4.2 it/s x 128 examples / 2.26 / 64).
    # If the console's "VRAM" creeps past ~28 GB, lower it (Windows spills into system RAM and crawls).
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300, help="steps to ramp lr up from 0 (fresh optimizer)")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ema-ramp", action="store_true",
                    help="ramp the EMA in like v0 did even on a warm start (normally off: see train_v1.py)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--max-hours", type=float, default=0, help="0 = no time limit")
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--nonfwd-weight", type=float, default=2.0)
    ap.add_argument("--holdout", type=float, default=0.03)
    # the target-frame noise (EDM's log-normal). Same as v0/v1; see model_v2.py note 4 about shifting it.
    ap.add_argument("--p-mean", type=float, default=-0.4, help="centre of log(sigma) for the target noise")
    ap.add_argument("--p-std", type=float, default=1.2, help="spread of log(sigma)")
    # recovery practice (self-forcing lite), as v1
    ap.add_argument("--sf-frac", type=float, default=0.25)
    ap.add_argument("--sf-two", type=float, default=0.5)
    ap.add_argument("--sf-steps", type=int, default=2)
    # context noise (anti-melt smudge), as v1
    ap.add_argument("--cn-mean", type=float, default=-1.8)
    ap.add_argument("--cn-std", type=float, default=1.1)
    ap.add_argument("--cn-max", type=float, default=1.5)
    ap.add_argument("--cn-zero", type=float, default=0.1)
    ap.add_argument("--cn-perframe", type=float, default=0.5)
    # rollouts / checkpoints
    ap.add_argument("--sample-every-min", type=float, default=15)
    ap.add_argument("--first-sample-step", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=20)
    ap.add_argument("--keep-every-hours", type=float, default=2)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--anchor", type=float, default=0.15)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the v1 baseline walk and the v2-at-init walk at startup")
    ap.add_argument("--vram-cap-gb", type=float, default=0,
                    help="hard VRAM cap (0 = none): a clean out-of-memory error instead of Windows "
                         "quietly spilling into system RAM and crawling")
    ap.add_argument("--allow-low-vram", action="store_true",
                    help="start even if the GPU doesn't have room for --batch (see the room check in main)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = HERE / "_smoke" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else DW / "data" / "cache_v2"
    ckpt_dir = base / "checkpoints" / args.run
    live_root = HERE / "_smoke" / "live_v2" if args.smoke else HERE / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    elif not args.allow_low_vram:
        # Room check. Windows doesn't say "out of memory" when the GPU is full: it quietly borrows
        # system RAM and every step gets many times slower. The usual cause would be v1 still
        # training in its window (14 GB). Measured: ~0.23 GB per example, plus ~2 GB fixed.
        # (Ask nvidia-smi, not torch: on Windows torch.cuda.mem_get_info() said 30 GB free while
        #  v1 was using 14 GB of the 32, tested 2026-10-19. nvidia-smi sees every program's use.)
        free = gpu_free_gb()
        need = 0.25 * args.batch + 2.5
        if free is not None and free < need:
            sys.exit(f"Only {free:.1f} GB of GPU memory is free, and batch {args.batch} needs about {need:.0f} GB.\n"
                     f"Is v1 training (or a play window) still running? Stop it first (Ctrl+C saves), "
                     f"or use a smaller --batch, or --allow-low-vram to start anyway.")
    torch.manual_seed(args.seed)

    # ---- data: exactly v1's windows and held-out split, just bigger frames
    index, frames_mm, actions_np, cut_np = load_cache_any(cache_dir)
    frame_hw = list(index["shape"][1:3])
    K = args.K
    L = K + EXTRA                                      # window = L + 1 frames (see train_v1.self_force)
    valid = compute_valid(actions_np, cut_np, index["cities"], L)
    train_ok = valid.copy()
    held_ok = np.zeros_like(valid)
    held_frames = np.zeros(len(valid), bool)
    roll_len = sum(n for _, n in ROLLOUT_SCRIPT_V1)
    rollout_start = None
    for c in index["cities"]:
        n = c["frames"]
        # the same held-out tail as v0 and v1 (last 3% of each city): neither ever trained on it
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
    dream_start = rollout_start + K                    # the same moment v1's rollouts start at
    real_clip = np.array(frames_mm[dream_start:dream_start + roll_len])
    start_clip = np.array(frames_mm[dream_start - K:dream_start])
    target_key = np.full(len(actions_np), -1, np.int64)
    target_key[:len(actions_np) - L + 1] = actions_np[L - 1:]
    weights = train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight)
    held_ix = np.flatnonzero(held_frames)
    sharp_ref = float(sharpness(np.array(frames_mm[held_ix[::2]])).mean())   # "1.0" of the melt meter, at 96x168

    # ---- model, EMA, optimizer
    latest = ckpt_dir / "latest.pt"
    ck = None
    if args.resume:
        if not latest.exists():
            sys.exit(f"--resume: no checkpoint at {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
    elif latest.exists():
        sys.exit(f"{latest} already exists. Use --resume to continue it, or pick another --run name.")
    model_cfg = ck["model_cfg"] if ck else dict(DEFAULT_CFG_V2, K=K, frame_hw=frame_hw)
    if model_cfg["K"] != K:
        sys.exit(f"checkpoint was trained with K={model_cfg['K']}, not --K {K}")
    if list(model_cfg["frame_hw"]) != frame_hw:
        sys.exit(f"the model dreams {model_cfg['frame_hw']} frames but {cache_dir} holds {frame_hw}")
    model = build_model_v2(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    init_notes, v1_ck = [], None
    if not ck and args.init.lower() != "none":
        # Warm start: v1's EMA weights (the smoothed ones that dream best) -> both v2 copies.
        v1_ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if v1_ck["model_cfg"].get("version") != "v1":
            sys.exit(f"--init {args.init} is not a v1 checkpoint (version {v1_ck['model_cfg'].get('version')})")
        init_notes = warm_start_from_v1(model, v1_ck["ema"])
        init_notes.insert(0, f"from {args.init}: v1 run {v1_ck['run_id']}, step {v1_ck['step']}")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), fused=True)

    step = 0
    train_seconds = 0.0
    run_id = "v2-" + time.strftime("%Y%m%d-%H%M")
    started = time.time()
    baseline = None                                     # v1's melt numbers on the same walk
    init_numbers = None                                 # the start-up check's numbers
    warm_started = v1_ck is not None
    if ck:
        warm_started = ck.get("warm_started", False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]
        baseline = ck.get("v1_baseline")
        init_numbers = ck.get("init_check")
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cities = ", ".join(c["name"] for c in index["cities"] if c["has_actions"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": "dream model v2",
        "prompt": f"city-walk dream v2 ({cities}), {n_params / 1e6:.0f}M params, "
                  f"{frame_hw[0]}x{frame_hw[1]}, K={K}",
        "pose": pose_string(ROLLOUT_SCRIPT_V1), "stage": "starting", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None,
        "melt_s": None, "v1_melt_s": baseline["melt_s"] if baseline else None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} v2 run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.2f}M params; {frame_hw[0]}x{frame_hw[1]}; K={K}; batch {args.batch}; "
             f"{int(train_ok.sum())} train windows, {int(held_ok.sum())} held-out; cache {cache_dir}")
    live.log(f"recovery practice: {args.sf_frac:.0%} of each batch ({args.sf_two:.0%} of those with 2 dreamed "
             f"frames, {args.sf_steps} sampler steps); context noise: log-normal({args.cn_mean}, {args.cn_std}) "
             f"max {args.cn_max}, {args.cn_zero:.0%} zero, {args.cn_perframe:.0%} per-frame; target noise "
             f"log-normal({args.p_mean}, {args.p_std}); lr {args.lr} with {args.warmup}-step warmup")
    live.log(f"melt meter: real held-out video sharpness at {frame_hw[0]}x{frame_hw[1]} = {sharp_ref:.2f} (= 1.0); "
             f"rollout = {roll_len} frames starting at cache frame {dream_start}")
    for n in init_notes:
        live.log("warm start: " + n)
    live.update(stage="starting")

    # ---- loader
    ds = WindowDatasetV2(cache_dir, L, flip_p=args.flip_p)
    sampler = InfiniteWeightedSampler(weights, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)

    # fixed held-out batch for validation loss + the start-up check (no flips)
    held_idx = np.flatnonzero(held_ok)
    rs = np.random.default_rng(0)
    pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
    val_ds = WindowDatasetV2(cache_dir, L, flip_p=0.0)
    vb = [val_ds[int(t)] for t in pick]
    val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]))

    def run_val():
        """EMA loss on 64 fixed held-out windows at 5 fixed noise levels (memory smudge 0.05).
        The same windows as v1's val loss, but the numbers are NOT directly comparable with v1's:
        a bigger picture has more fine detail per frame to get right."""
        fr = to_model_space(val_batch[0].to(device))
        act = val_batch[1].to(device)
        B = fr.shape[0]
        tot, sig_list = 0.0, [0.05, 0.2, 0.5, 1.0, 2.0]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, B, 16):                          # in slices of 16: small VRAM bump
                    for s in sig_list:
                        tot += ema.loss(fr[i:i + 16, L], fr[i:i + 16, EXTRA:L], act[i:i + 16, EXTRA:L],
                                        torch.full((min(16, B - i),), s, device=device), 0.05).item() \
                               * min(16, B - i) / B
        return tot / len(sig_list)

    def save_ckpt(path):
        try:
            tmp = str(path) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                        "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg,
                        "v1_baseline": baseline, "init_check": init_numbers, "sharp_ref": sharp_ref,
                        "warm_started": warm_started}, tmp)
            if not safe_replace(tmp, path, tries=40):
                live.log(f"could not replace {path} (file busy), checkpoint skipped this time")
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def do_rollout():
        """Raw + anchored dream of the 30 s walk (one batch of 2). The RAW one is scored and shown
        live (frame_*.jpg); the anchored one only goes to train/out/v2_anchor_rollout_step<step>.mp4."""
        try:
            t0 = time.time()
            raw, anc, keys = rollout_v1(ema, frames_mm, actions_np, dream_start, K, ROLLOUT_SCRIPT_V1,
                                        args.rollout_steps, device, anchor=args.anchor)
            rep, s = melt_report(raw, sharp_ref, start_clip)
            arep, a_s = melt_report(anc, sharp_ref, start_clip)
            write_live_frames(live, raw, keys)
            path = write_mp4(out_dir / f"v2_rollout_step{step:07d}.mp4", raw, keys, real_clip,
                             f"v2 step {step}", s)
            if args.anchor > 0:
                write_mp4(out_dir / f"v2_anchor_rollout_step{step:07d}.mp4", anc, keys, real_clip,
                          f"v2 {step} anchor {args.anchor}", a_s)
            v = run_val()
            live.log(f"rollout at step {step}: {len(keys)} frames x2 (raw + anchored), {args.rollout_steps} "
                     f"sampler steps, in {time.time() - t0:.1f}s -> {path.name}; val loss {v:.4f}")
            live.log(melt_line(f"v2 step {step} (raw)", rep)
                     + (f"\n    [v1 baseline raw melt: {secs(baseline['melt_s'])}]" if baseline else "")
                     + f"\n    [with colour anchor {args.anchor}: melt {secs(arep['melt_s'])}]")
            live.metric(dict({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name,
                              "anchor_melt_s": arep["melt_s"]}, **rep))
            live.update(last_rollout_step=step, val_loss=v, melt_s=rep["melt_s"])
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    # ---- start-up checks on a NEW warm-started run
    if v1_ck is not None:
        live.update(stage="start-up check")
        v1 = build_model_v1(v1_ck["model_cfg"]).to(device).eval()
        v1.load_state_dict(v1_ck["ema"])
        n_chk = 32
        fr2 = to_model_space(val_batch[0][:n_chk].to(device))
        act = val_batch[1][:n_chk].to(device)
        v1_ds = WindowDatasetV2(args.v1_cache, L, flip_p=0.0)           # same moments, 64x112
        fr1 = to_model_space(torch.stack([v1_ds[int(t)][0] for t in pick[:n_chk]]).to(device))
        png = out_dir / "v2_init_check.png"
        r = init_check(model, v1, fr2, fr1, act, args.rollout_steps, png, device)
        init_numbers = r
        live.log(f"START-UP CHECK on {n_chk} held-out moments (MSE per pixel in the -1..1 scale; lower = closer "
                 f"to the real next frame; PSNR in dB, higher = closer):")
        for s in [0.1, 0.5, 2.0]:
            live.log(f"  one clean-up call at noise {s}: v2 {r[f'denoise_sigma{s}_v2']:.4f}   "
                     f"v1 {r[f'denoise_sigma{s}_v1']:.4f}")
        live.log(f"  dreamed first frame ({args.rollout_steps} steps from static):")
        live.log(f"    v2 at init, 96x168           {r['dream_v2_96x168']:.4f}  ({psnr(r['dream_v2_96x168']):.1f} dB)")
        live.log(f"    v2 at init, shrunk to 64x112 {r['dream_v2_shrunk_64x112']:.4f}  "
                 f"({psnr(r['dream_v2_shrunk_64x112']):.1f} dB)")
        live.log(f"    v1, 64x112                   {r['dream_v1_64x112']:.4f}  ({psnr(r['dream_v1_64x112']):.1f} dB)")
        live.log(f"    (no-brain baseline 'repeat the newest memory frame': 96x168 "
                 f"{r['repeat_last_frame_96x168']:.4f}, 64x112 {r['repeat_last_frame_64x112']:.4f}; "
                 f"resize floor {r['resize_floor']:.4f})")
        live.log(f"  picture grid -> {png}")
        live.metric({"step": step, "time": time.time(), "init_check": r})

        if not args.no_baseline:
            # the v1 baseline: the same 30 s walk, same start moment and noise seed, at 64x112
            live.update(stage="v1 baseline rollout")
            t0 = time.time()
            _, frames1, actions1, _ = load_cache_any(args.v1_cache)
            real1 = np.array(frames1[dream_start:dream_start + roll_len])
            start1 = np.array(frames1[dream_start - K:dream_start])
            sharp_ref1 = float(sharpness(np.array(frames1[held_ix[::2]])).mean())
            raw1, anc1, keys = rollout_v1(v1, frames1, actions1, dream_start, K, ROLLOUT_SCRIPT_V1,
                                          args.rollout_steps, device, anchor=args.anchor)
            rep1, s1 = melt_report(raw1, sharp_ref1, start1)
            arep1, as1 = melt_report(anc1, sharp_ref1, start1)
            tag = f"v1_step{v1_ck['step']:07d}"
            path1 = write_mp4(out_dir / f"v2_baseline_{tag}.mp4", raw1, keys, real1, f"v1 {v1_ck['step']}", s1)
            write_mp4(out_dir / f"v2_anchor_baseline_{tag}.mp4", anc1, keys, real1,
                      f"v1 {v1_ck['step']} anchor {args.anchor}", as1)
            baseline = dict(rep1, v1_step=v1_ck["step"], video=path1.name, anchor_melt_s=arep1["melt_s"],
                            steps=args.rollout_steps)
            live.log(melt_line(f"v1 BASELINE (step {v1_ck['step']}, 64x112, {args.rollout_steps} sampler "
                               f"steps, raw, {time.time() - t0:.0f}s)", rep1))
            live.log(f"v1 baseline with colour anchor {args.anchor} (video only): melt time "
                     f"{secs(arep1['melt_s'])} ({arep1['melt_kind'] or '-'})")
            real_rep, _ = melt_report(real_clip, sharp_ref, start_clip)
            live.log(melt_line("real video at 96x168, same 30 s (for scale; should never 'melt')", real_rep))
            live.metric({"step": step, "time": time.time(), "v1_baseline": baseline, "real_clip": real_rep})
            live.update(v1_melt_s=baseline["melt_s"])
            del frames1
        del v1
        torch.cuda.empty_cache()
        if not args.no_baseline:
            do_rollout()                                     # v2 at init on the same walk ("step 0")
    del v1_ck

    # ---- the loop (word for word v1's, apart from the --p-mean/--p-std target noise)
    n_sf = int(round(args.sf_frac * args.batch))
    n_two = int(round(args.sf_two * n_sf))
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
            fr = to_model_space(frames_u8.to(device, non_blocking=True))    # [B, K+3, 3, 96, 168]
            act = act.to(device, non_blocking=True)                         # [B, K+2]
            B = fr.shape[0]

            # 1. recovery practice: swap some newest memory frames for the model's own dreams
            fr = self_force(ema, fr, act, K, n_sf, n_two, args.sf_steps)
            context, target, actions = fr[:, EXTRA:L], fr[:, L], act[:, EXTRA:L]
            # 2. EDM noise for the target and the heavier per-frame memory smudge
            sigma = (torch.randn(B, device=device) * args.p_std + args.p_mean).exp().clamp(2e-3, 20.0)
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
