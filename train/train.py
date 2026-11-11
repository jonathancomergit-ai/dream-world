"""Train the Dream World model (overnight-safe).

    python train/train.py                         # new run "dream_v0" on data/cache
    python train/train.py --resume                # carry on from train/checkpoints/dream_v0/latest.pt
    python train/train.py --smoke --max-steps 300 # tiny test run, everything under train/_smoke/

Each step:
  1. grab a batch of random windows: 4 real context frames + the real next frame + the 4 keys
     (mirrored left<->right half the time, with the keys swapped to match);
  2. bury the next frame in a random amount of noise (EDM's log-normal sigma: mostly medium,
     sometimes nearly clean, sometimes pure static);
  3. smudge the context frames with a little noise too (anti-melt trick, see model.py);
  4. the model guesses the clean next frame; the loss is how wrong it was; nudge the weights.
We also keep an EMA ("exponential moving average") copy of the weights: a slow, smoothed
average of the last few thousand steps. It's less jittery than the live weights, so it dreams
better. Rollouts and play.py use the EMA weights.

What the owner's dashboard (Dream Lab) sees, in train/live/<run_id>/:
  status.json    small summary, rewritten every ~10 s (atomic: tmp file, then os.replace)
  run.log        human-readable progress lines
  metrics.jsonl  one JSON line every ~50 steps
  frame_*.jpg    the newest dream rollout (120 frames, overwritten every ~10 min)
And train/out/rollout_step<step>.mp4: real video (left) vs dream (right), one per rollout,
so you can scroll through the night and watch the dreams get better.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
import json
import math
import os
import pathlib
import time
import traceback

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from build_cache import compute_valid                       # noqa: E402
from data import (ACTION_NAMES, InfiniteWeightedSampler, WindowDataset, load_cache,  # noqa: E402
                  to_model_space, to_uint8_image)
from model import DEFAULT_CFG, build_model                   # noqa: E402

# The fixed scripted walk every rollout uses, so rollouts are comparable across the night.
# (action, how many frames). 1 = W forward, 5 = turn left, 6 = turn right.  Total 120 frames.
ROLLOUT_SCRIPT = [(1, 20), (6, 10), (1, 30), (5, 20), (1, 20), (6, 10), (1, 10)]


def pose_string(script):
    return ", ".join(f"{ACTION_NAMES[a]} x{n}" for a, n in script)


# ---------------------------------------------------------------- Windows-safe file writing

def safe_replace(tmp, dst, tries=5):
    """os.replace, but on Windows it fails (PermissionError) if the dashboard happens to have
    dst open for reading at that instant. Retry a few times, then quietly skip this update:
    a missed status refresh is harmless, a crashed overnight run is not."""
    for _ in range(tries):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            time.sleep(0.05)
        except OSError:
            break
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def write_json_atomic(path, obj):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        safe_replace(tmp, path)
    except Exception:
        pass   # never let a status write kill training


class Live:
    """Everything the dashboard reads lives here."""

    def __init__(self, folder, status):
        self.dir = pathlib.Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = status
        self.log_f = open(self.dir / "run.log", "a", encoding="utf-8")
        self.metrics_f = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            self.log_f.write(line + "\n")
            self.log_f.flush()
        except Exception:
            pass

    def metric(self, d):
        try:
            self.metrics_f.write(json.dumps(d) + "\n")
            self.metrics_f.flush()
        except Exception:
            pass

    def update(self, **kw):
        self.status.update(kw)
        self.status["frames"] = len(list(self.dir.glob("frame_*.jpg")))
        write_json_atomic(self.dir / "status.json", self.status)


# ---------------------------------------------------------------- rollout (the dream)

@torch.no_grad()
def rollout(ema, frames_mm, actions_np, start, K, script, steps, device, seed=1234):
    """Start from K real frames at `start`, then dream len(script) frames with the scripted keys.
    Returns (dream uint8 [T,H,W,3], key per frame [T])."""
    ctx = to_model_space(torch.from_numpy(np.array(frames_mm[start:start + K])).to(device))[None]
    # keys between the real context frames (unknown -> pretend "forward", the most common move)
    hist = [int(a) if a >= 0 else 1 for a in actions_np[start:start + K - 1]]
    keys = [a for a, n in script for _ in range(n)]
    g = torch.Generator(device=device).manual_seed(seed)    # same noise every rollout: fair comparison
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for a in keys:
            act = torch.tensor([hist[-(K - 1):] + [a]], device=device)
            nxt = ema.sample(ctx, act, steps=steps, generator=g)            # [1,3,H,W]
            out.append(to_uint8_image(nxt[0]))
            ctx = torch.cat([ctx[:, 1:], nxt[:, None].float()], dim=1)      # slide the memory window
            hist.append(a)
    return np.stack(out), keys


def upscale(img, s=4):
    return cv2.resize(img, (img.shape[1] * s, img.shape[0] * s), interpolation=cv2.INTER_CUBIC)


def label(img, text, y=18):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_rollout(live, out_dir, step, dream, keys, real):
    """Overwrite frame_*.jpg in the live folder + write a side-by-side mp4 (real | dream)."""
    import imageio.v2 as imageio
    for i, (fr, a) in enumerate(zip(dream, keys)):
        big = label(upscale(fr), ACTION_NAMES[a])
        dst = live.dir / f"frame_{i:05d}.jpg"
        tmp = live.dir / f"frame_{i:05d}.jpg.tmp"       # doesn't match frame_*.jpg, so never shown half-written
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            tmp.write_bytes(buf.tobytes())
            safe_replace(tmp, dst)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rollout_step{step:07d}.mp4"
    w = imageio.get_writer(str(path), fps=10, codec="libx264", pixelformat="yuv420p",
                           quality=8, macro_block_size=16, ffmpeg_log_level="error")
    for i, (fr, a) in enumerate(zip(dream, keys)):
        r = real[min(i, len(real) - 1)]
        left = label(upscale(r), "real video")
        right = label(upscale(fr), f"dream step {step}: {ACTION_NAMES[a]}")
        w.append_data(np.concatenate([left, right], axis=1))
    w.close()
    return path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v0", help="checkpoint folder name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use train/_smoke/ for cache, live, checkpoints, out")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--max-hours", type=float, default=0, help="0 = no time limit")
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--nonfwd-weight", type=float, default=2.0,
                    help="sample windows whose key isn't forward this many times more often")
    ap.add_argument("--holdout", type=float, default=0.03, help="end fraction of each city kept out of training")
    ap.add_argument("--sample-every-min", type=float, default=10)
    ap.add_argument("--first-sample-step", type=int, default=200)
    ap.add_argument("--ckpt-every-min", type=float, default=20)
    ap.add_argument("--keep-every-hours", type=float, default=2)
    ap.add_argument("--rollout-steps", type=int, default=5, help="denoising steps per dreamed frame")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = HERE / "_smoke" if args.smoke else HERE
    cache_dir = pathlib.Path(args.cache) if args.cache else (
        HERE / "_smoke" / "cache" if args.smoke else DW / "data" / "cache")
    ckpt_dir = base / "checkpoints" / args.run
    live_root = base / "live"
    out_dir = base / "out"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True      # fixed input sizes: let cuDNN pick the fastest convs

    # ---- data
    index, frames_mm, actions_np, cut_np = load_cache(cache_dir)
    K = args.K
    valid = compute_valid(actions_np, cut_np, index["cities"], K)
    train_ok = valid.copy()
    held_ok = np.zeros_like(valid)
    rollout_start = None
    for c in index["cities"]:
        n = c["frames"]
        hold_n = min(max(int(args.holdout * n), K + 130), n // 3)
        h0 = c["end"] - hold_n
        train_ok[h0 - K:c["end"]] = False             # no training window may touch held-out frames
        held_ok[h0:c["end"]] = valid[h0:c["end"]]
        if rollout_start is None and n > 3 * (K + 130):
            rollout_start = h0 + 5                    # rollouts start in the first big-enough city's held-out tail
    if rollout_start is None:
        rollout_start = 0
    if train_ok.sum() == 0:
        sys.exit("No valid training windows. Are the actions files in place? Re-run build_cache.py.")
    # target key = actions[t+K-1]; give non-forward windows extra weight so turns get practised
    target_key = np.full(len(actions_np), -1, np.int64)
    target_key[:len(actions_np) - K + 1] = actions_np[K - 1:]
    weights = train_ok.astype(np.float64) * np.where(target_key == 1, 1.0, args.nonfwd_weight)

    # ---- model, EMA, optimizer (on --resume, rebuild the exact architecture that was saved)
    latest = ckpt_dir / "latest.pt"
    ck = None
    if args.resume:
        if not latest.exists():
            sys.exit(f"--resume: no checkpoint at {latest}")
        ck = torch.load(latest, map_location=device, weights_only=False)
    elif latest.exists():
        sys.exit(f"{latest} already exists. Use --resume to continue it, or pick another --run name.")
    model_cfg = ck["model_cfg"] if ck else dict(DEFAULT_CFG, K=K)
    if model_cfg["K"] != K:
        sys.exit(f"checkpoint was trained with K={model_cfg['K']}, not --K {K}")
    model = build_model(model_cfg).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), fused=True)

    step = 0
    train_seconds = 0.0
    run_id = time.strftime("%Y%m%d-%H%M")
    started = time.time()
    if ck:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds = ck["step"], ck.get("train_seconds", 0.0)
        run_id, started = ck["run_id"], ck["started"]    # same live folder: the dashboard just carries on
        del ck
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cities = ", ".join(c["name"] for c in index["cities"] if c["has_actions"])
    live = Live(live_root / run_id, {
        "run": run_id, "kind": "train", "image": "dream model",
        "prompt": f"city-walk dream ({cities}), {n_params / 1e6:.0f}M params, 64x112",
        "pose": pose_string(ROLLOUT_SCRIPT), "stage": "training", "started": started,
        "peak_vram_gb": 0.0, "exit": None, "step": step, "loss": None, "it_s": 0.0, "frames": 0,
        "batch": args.batch, "lr": 0.0, "val_loss": None, "last_rollout_step": None})
    live.log(f"{'RESUMED' if args.resume else 'NEW'} run {args.run} (live id {run_id}) at step {step}; "
             f"{n_params / 1e6:.1f}M params; batch {args.batch}; {int(train_ok.sum())} train windows, "
             f"{int(held_ok.sum())} held-out; cache {cache_dir}")
    for c in index["cities"]:
        if not c["has_actions"]:
            live.log(f"WARNING: {c['name']} has no actions, so it isn't trained on.")
    live.update(stage="training", exit=None)    # show up on the dashboard right away (startup takes ~40 s)

    # ---- loader
    ds = WindowDataset(cache_dir, K, flip_p=args.flip_p)
    sampler = InfiniteWeightedSampler(weights, seed=args.seed + step)   # new seed on resume: fresh order
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)

    # fixed held-out batch for a validation loss (same windows, noise, sigmas every time)
    held_idx = np.flatnonzero(held_ok)
    val_batch = None
    if len(held_idx):
        rs = np.random.default_rng(0)
        pick = rs.choice(held_idx, size=min(64, len(held_idx)), replace=len(held_idx) < 64)
        # separate dataset object: once a dataset has opened its memmap it must NOT be sent
        # to the workers (it would be copied whole into each one)
        val_ds = WindowDataset(cache_dir, K, flip_p=0.0)
        vb = [val_ds[int(t)] for t in pick]
        val_batch = (torch.stack([v[0] for v in vb]), torch.stack([v[1] for v in vb]))

    def run_val():
        """Loss of the EMA model on the same 64 held-out windows at 5 fixed noise levels.
        Comparable across the night (unlike the training loss, which jumps around with the
        random sigmas). fork_rng: use fixed noise here without disturbing training's RNG."""
        if val_batch is None:
            return None
        fr = to_model_space(val_batch[0].to(device))
        act = val_batch[1].to(device)
        B = fr.shape[0]
        tot, sig_list = 0.0, [0.05, 0.2, 0.5, 1.0, 2.0]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in sig_list:
                    tot += ema.loss(fr[:, K], fr[:, :K], act, torch.full((B,), s, device=device),
                                    torch.full((B,), 0.05, device=device)).item()
        return tot / len(sig_list)

    def save_ckpt(path):
        try:
            _save_ckpt(path)
        except Exception:
            live.log(f"checkpoint {path.name} FAILED (training continues):\n" + traceback.format_exc())

    def _save_ckpt(path):
        tmp = str(path) + ".tmp"
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "train_seconds": train_seconds, "run_id": run_id, "started": started,
                    "args": vars(args), "K": K, "params": n_params, "model_cfg": model_cfg}, tmp)
        if not safe_replace(tmp, path, tries=40):
            live.log(f"could not replace {path} (file busy), checkpoint skipped this time")

    def do_rollout():
        """A broken rollout (video writer, disk...) must never stop training: log it, move on."""
        try:
            _do_rollout()
        except Exception:
            live.log("rollout FAILED (training continues):\n" + traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    def _do_rollout():
        t0 = time.time()
        dream, keys = rollout(ema, frames_mm, actions_np, rollout_start, K, ROLLOUT_SCRIPT,
                              args.rollout_steps, device)
        real = np.array(frames_mm[rollout_start + K:rollout_start + K + len(keys)])
        path = save_rollout(live, out_dir, step, dream, keys, real)
        v = run_val()
        live.log(f"rollout at step {step}: {len(keys)} frames in {time.time() - t0:.1f}s -> {path.name}"
                 + (f"; val loss {v:.4f}" if v is not None else ""))
        live.metric({"step": step, "time": time.time(), "val_loss": v, "rollout": path.name})
        live.update(last_rollout_step=step, val_loss=v)

    # ---- the loop
    seconds = lambda m: m * 60
    now = time.time()
    last_status = 0.0
    last_ckpt = now
    last_keep = now                                # first keeper copy after keep-every-hours
    next_sample = None if step < args.first_sample_step else now + seconds(args.sample_every_min)
    sample_soon = step < args.first_sample_step    # the early "something to look at" rollout
    loss_avg = None
    win_t, win_steps = time.time(), 0
    it_s = 0.0
    bad_steps = 0
    t_loop = time.time()

    try:
        it = iter(loader)
        while True:
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_hours and (time.time() - t_loop) / 3600 >= args.max_hours:
                break
            frames_u8, act = next(it)
            fr = to_model_space(frames_u8.to(device, non_blocking=True))   # [B, K+1, 3, H, W]
            act = act.to(device, non_blocking=True)
            context, target = fr[:, :K], fr[:, K]
            if args.channels_last:
                target = target.contiguous(memory_format=torch.channels_last)
            B = target.shape[0]
            # EDM training noise: log-normal, median e^-0.4 = 0.67 (DIAMOND's settings)
            sigma = (torch.randn(B, device=device) * 1.2 - 0.4).exp().clamp(2e-3, 20.0)
            # anti-melt context noise: 10% of the time none, else log-normal around 0.08, max 0.7
            sigma_cond = (torch.randn(B, device=device) * 1.0 - 2.5).exp().clamp(max=0.7)
            sigma_cond = torch.where(torch.rand(B, device=device) < 0.1, torch.zeros_like(sigma_cond), sigma_cond)

            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))      # short linear warmup, then flat
            for gq in opt.param_groups:
                gq["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(target, context, act, sigma, sigma_cond)
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
            # EMA: ema = d*ema + (1-d)*model. d ramps up from ~0.9 so early rollouts aren't just noise
            d = min(args.ema, (1 + step) / (10 + step))
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
                next_sample = time.time() + seconds(args.sample_every_min)
            elif next_sample is not None and now >= next_sample:
                do_rollout()
                next_sample = time.time() + seconds(args.sample_every_min)
            if step % 100 == 0 or step == 1:
                vram = torch.cuda.max_memory_reserved() / 2**30
                eta = (f"{(next_sample - now) / 60:.1f} min" if next_sample else
                       f"at step {args.first_sample_step}")
                live.log(f"step {step}  loss {loss_avg:.4f}  {it_s:.2f} it/s  lr {lr:.2e}  "
                         f"grad {gn.item():.2f}  VRAM {vram:.1f} GB  next dream {eta}")
            if now - last_status >= 10:
                last_status = now
                live.update(step=step, loss=round(loss_avg, 5), it_s=round(it_s, 2), lr=lr,
                            peak_vram_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            if now - last_ckpt >= seconds(args.ckpt_every_min):
                save_ckpt(latest)
                last_ckpt = time.time()
                live.log(f"checkpoint saved (step {step})")
                if now - last_keep >= args.keep_every_hours * 3600:
                    save_ckpt(ckpt_dir / f"step{step:07d}.pt")
                    last_keep = time.time()
                    live.log(f"keeper checkpoint step{step:07d}.pt saved")

        # finished normally (max steps / max hours)
        save_ckpt(latest)
        if step >= args.first_sample_step:
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
            save_ckpt(ckpt_dir / "crash.pt")   # NOT latest.pt: the crash state might be broken
            live.log("saved crash.pt (latest.pt left untouched; --resume uses latest.pt)")
        except Exception:
            pass
        live.update(stage="crashed", exit=1, step=step)
        raise


if __name__ == "__main__":
    main()
