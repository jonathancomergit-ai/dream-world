"""Train PAINTER v1 (painter_model.py) on all the face-blurred 2x footage (2026-10-24).

  python train/painter/train_painter.py --run painter_v1 --max-hours 3
  python train/painter/train_painter.py --smoke --run p_smoke --max-steps 60      (quick test, train/_smoke/painter/)
  python train/painter/train_painter.py --run painter_v1 --resume

Data: data/cache_v11 (192x336, the target) + data/cache_v3 (96x168, the input), the SAME 518k frames in the same
order. Training frames: the 9 walking cities (no Wildlife), each city's training part only (stops 300 frames before
its held-out tail: the judge's test walks live in those tails and are never trained on).
Each example: frame t (low-res input + 2x target) and frame t-1 (2x, = "the painter's previous frame").
  - input damage (dream_damage): 25% clean; else blur / down-up / sideways smear / ripple warp / melted blobs / grain
    in random mixes, so the painter learns to REPAIR dream-like frames, not just sharpen clean ones;
  - previous frame: slid by the turn measured on the low-res frames (est_shift, what play will do), plus noise
    (it will be the painter's own imperfect output in play); 25% of rows get none;
  - noise levels: EDM log-normal; 30% of rows start from the 2x low-res frame at noise 1.5-2.5 (= sampling's start).
Every --val-every-min: held-out frames (never trained on) painted with 6 steps -> L1 vs the real 2x frame and
sharpness (fine detail vs real = 1.0), and a picture [low-res 2x | painter | real] for Dream Lab's screen."""
import argparse
import copy
import json
import math
import pathlib
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
TRAIN = HERE.parent
DW = TRAIN.parent
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRAIN / "sr"))

from data_v2 import load_cache_any                                    # noqa: E402
from painter_model import Painter, est_shift, slide                   # noqa: E402
from train import Live                                                # noqa: E402
from train_sr import gauss_blur, lap_energy                           # noqa: E402

HOLD_FRAC, MIN_HOLD, GAP = 0.03, 608, 300          # same held-out rule as dream_fuel.FuelLibrary / cache_hr


def splits(index, cut):
    """(train indices, val indices) - frame t needs t-1 in the same shot for the 'previous frame'."""
    tr, va = [], []
    for c in index["cities"]:
        if c["name"] == "Wildlife":
            continue
        n = c["end"] - c["start"]
        h0 = c["end"] - min(max(int(HOLD_FRAC * n), MIN_HOLD), n // 3)
        tr.append(np.arange(c["start"] + 1, h0 - GAP))
        va.append(np.arange(h0 + 1, c["end"], 47))
    tr, va = np.concatenate(tr), np.concatenate(va)
    ok = lambda ix: ix[~(cut[ix] | cut[ix - 1])]
    return ok(tr), ok(va)


class Frames(torch.utils.data.Dataset):
    def __init__(self, idx, hr_dir, lr_dir, flip=True):
        self.idx, self.hr_dir, self.lr_dir, self.flip = idx, hr_dir, lr_dir, flip
        self.hr = self.lr = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self.hr is None:                                 # open the memmaps in each worker
            _, self.hr, _, _ = load_cache_any(self.hr_dir)
            _, self.lr, _, _ = load_cache_any(self.lr_dir)
        i = int(self.idx[j])
        out = [np.array(self.hr[i]), np.array(self.lr[i]), np.array(self.hr[i - 1]), np.array(self.lr[i - 1])]
        if self.flip and np.random.rand() < 0.5:
            out = [np.ascontiguousarray(a[:, ::-1]) for a in out]
        return tuple(torch.from_numpy(a) for a in out)


def to_ms(u8, dev):
    return u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def hsmear(x, length):
    """Sideways box blur of `length` px (the dream's 'curtains')."""
    k = torch.ones(1, 1, 1, length, device=x.device) / length
    C = x.shape[1]
    xp = F.pad(x, (length // 2, length - 1 - length // 2, 0, 0), mode="replicate")
    return F.conv2d(xp, k.expand(C, 1, 1, length), groups=C)


def ripple(x, amp, wav, phase):
    """Rows slid sideways by amp * sin(2 pi row / wav + phase) px (per row of the batch: amp, wav, phase [B])."""
    B, C, H, W = x.shape
    rows = torch.arange(H, device=x.device).float()[None]
    dx = amp[:, None] * torch.sin(2 * math.pi * rows / wav[:, None] + phase[:, None])          # [B, H]
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device),
                            indexing="ij")
    grid = torch.stack([gx[None] + 2 * dx[:, :, None] / W, gy[None].expand(B, -1, -1)], -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def dream_damage(x, clean_frac=0.25):
    """[B, 3, 96, 168] model space -> the same with random dream-like damage (row-wise)."""
    B, dev = x.shape[0], x.device
    dmg = (torch.rand(B, device=dev) >= clean_frac).float()
    r = torch.rand(B, 10, device=dev)
    y = gauss_blur(x, torch.where(r[:, 0] < 0.6, 0.3 + 2.2 * r[:, 1], torch.zeros_like(r[:, 1])) * dmg)
    sel = (dmg > 0) & (r[:, 2] < 0.35)                                     # down-up: lost detail, soft blobs
    if sel.any():
        sc = float(0.3 + 0.5 * torch.rand(()))
        small = F.interpolate(y[sel], scale_factor=sc, mode="bilinear", align_corners=False, antialias=True)
        y[sel] = F.interpolate(small, size=y.shape[-2:], mode="bilinear", align_corners=False)
    sel = (dmg > 0) & (r[:, 3] < 0.3)                                      # sideways smear ("curtains")
    if sel.any():
        y[sel] = hsmear(y[sel], int(torch.randint(3, 16, ())))
    sel = (dmg > 0) & (r[:, 4] < 0.25)                                     # ripples
    if sel.any():
        n = int(sel.sum())
        y[sel] = ripple(y[sel], 0.5 + 2.0 * torch.rand(n, device=dev), 4 + 20 * torch.rand(n, device=dev),
                        6.283 * torch.rand(n, device=dev))
    sel = (dmg > 0) & (r[:, 5] < 0.3)                                      # melted blobs: heavy blur in patches
    if sel.any():
        n = int(sel.sum())
        m = F.interpolate(torch.rand(n, 1, 6, 10, device=dev), size=y.shape[-2:], mode="bicubic", align_corners=False)
        m = ((m - 0.5) * 3 + 0.5).clamp(0, 1) * (0.5 + 0.5 * torch.rand(n, 1, 1, 1, device=dev))
        y[sel] = y[sel] * (1 - m) + gauss_blur(y[sel], torch.full((n,), 4.0, device=dev)) * m
    std = torch.where(r[:, 6] < 0.4, 0.06 * r[:, 7], torch.zeros_like(r[:, 7])) * dmg
    y = y + std[:, None, None, None] * torch.randn_like(y)
    return y.clamp(-1, 1)


@torch.no_grad()
def validate(ema, val, dev, steps=6):
    """val = (hr [N,3,192,336], lr [N,3,96,168]) on dev. Returns numbers + a picture of the first 3 frames."""
    hr, lr = val
    outs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(lr), 16):
            outs.append(ema.sample(lr[i:i + 16], steps=steps, generator=torch.Generator(device=dev).manual_seed(i)))
    out = torch.cat(outs).float()
    bic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)
    l1 = lambda a: float((a - hr).abs().mean() * 127.5)
    sharp = lambda a: float((lap_energy(a) / lap_energy(hr).clamp(min=1e-3)).mean())
    res = {"val_l1": round(l1(out), 3), "val_sharp": round(sharp(out), 3), "bicubic_l1": round(l1(bic), 3),
           "bicubic_sharp": round(sharp(bic), 3)}
    rows = [torch.cat([bic[k], out[k], hr[k]], 2) for k in range(3)]
    pic = ((torch.cat(rows, 1).permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    return res, pic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="painter_v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--p-mean", type=float, default=-0.6)
    ap.add_argument("--p-std", type=float, default=1.2)
    ap.add_argument("--start-frac", type=float, default=0.3, help="rows noised around the 2x low-res frame")
    ap.add_argument("--clean-frac", type=float, default=0.25, help="rows whose low-res input is undamaged")
    ap.add_argument("--prev-drop", type=float, default=0.25)
    ap.add_argument("--prev-noise", type=float, default=0.15, help="max noise on the previous frame")
    ap.add_argument("--val-n", type=int, default=96)
    ap.add_argument("--val-every-min", type=float, default=15)
    ap.add_argument("--ckpt-every-min", type=float, default=10)
    ap.add_argument("--keep-every-hours", type=float, default=1)
    ap.add_argument("--vram-cap-gb", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.vram_cap_gb > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2 ** 30 / tot))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    base = TRAIN / "_smoke" / "painter" if args.smoke else TRAIN
    ck_dir = base / "checkpoints" / args.run
    ck_dir.mkdir(parents=True, exist_ok=True)
    latest = ck_dir / "latest.pt"
    hr_dir, lr_dir = DW / "data" / "cache_v11", DW / "data" / "cache_v3"
    index, hr_mm, _, cut = load_cache_any(hr_dir)
    tr_idx, va_idx = splits(index, cut)
    model = Painter().to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, fused=True)
    step, t_used, run_id, started = 0, 0.0, f"{args.run}-" + time.strftime("%Y%m%d-%H%M"), time.time()
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, t_used, run_id, started = ck["step"], ck["t_used"], ck["run_id"], ck["started"]
    n_params = sum(p.numel() for p in model.parameters())
    live_root = (TRAIN / "_smoke" / "painter" / "live") if args.smoke else (TRAIN / "live")
    live = Live(live_root / run_id, {"run": run_id, "kind": "train", "image": "painter v1 (2x)",
                                     "prompt": f"PAINTER: 96x168 dream -> 192x336, {n_params / 1e6:.1f}M params, "
                                               f"6-step diffusion + its own previous frame",
                                     "pose": "", "stage": "starting", "started": started, "exit": None,
                                     "step": step, "batch": args.batch})
    live.log(f"{'RESUMED' if step else 'NEW'} painter run {run_id} at step {step}: {n_params / 1e6:.2f}M params, "
             f"{len(tr_idx)} training frames, {len(va_idx)} held-out val frames; args {vars(args)}")
    # fixed val set (held-out tails, never trained on)
    _, lr_mm, _, _ = load_cache_any(lr_dir)
    vsel = va_idx[np.linspace(0, len(va_idx) - 1, args.val_n).astype(int)]
    val = (to_ms(torch.from_numpy(np.stack([hr_mm[i] for i in vsel])), dev),
           to_ms(torch.from_numpy(np.stack([lr_mm[i] for i in vsel])), dev))
    loader = torch.utils.data.DataLoader(
        Frames(tr_idx, hr_dir, lr_dir), batch_size=args.batch, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None,
        sampler=torch.utils.data.RandomSampler(tr_idx, replacement=True, num_samples=10 ** 9))

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "t_used": t_used, "run_id": run_id, "started": started, "cfg": model.net.cfg,
                    "kind": "painter_v1", "args": vars(args)}, tmp)
        tmp.replace(path)

    def do_val():
        t0 = time.time()
        res, pic = validate(ema, val, dev)
        import cv2
        n = len(list(live.dir.glob("frame_*.jpg")))
        cv2.imwrite(str(live.dir / f"frame_{n:05d}.jpg"), cv2.cvtColor(pic, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        live.metric({"step": step, "time": time.time(), **res})
        live.log(f"VAL step {step}: painter L1 {res['val_l1']} (bicubic {res['bicubic_l1']}), sharpness "
                 f"{res['val_sharp']} of real (bicubic {res['bicubic_sharp']}) [{time.time() - t0:.0f}s; picture "
                 f"frame_{n:05d}.jpg = low-res 2x | painter | real]")
        live.update(val_l1=res["val_l1"], val_sharp=res["val_sharp"])

    live.update(stage="training")
    t_start, last_ck, last_keep, last_val = time.time(), time.time(), time.time(), time.time()
    win_t, win_n, loss_avg = time.time(), 0, None
    if step == 0:
        do_val()
    try:
        it = iter(loader)
        while True:
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
                break
            hr_u8, lr_u8, hrp_u8, lrp_u8 = next(it)
            hr, lr, hrp, lrp = (to_ms(t, dev) for t in (hr_u8, lr_u8, hrp_u8, lrp_u8))
            B = hr.shape[0]
            with torch.no_grad():
                shift = est_shift(lrp, lr) * 2                                      # measured on the low-res frames
                prev = slide(hrp, shift)
                prev = prev + args.prev_noise * torch.rand(B, 1, 1, 1, device=dev) * torch.randn_like(prev)
                has = (torch.rand(B, device=dev) >= args.prev_drop).float()
                lr_in = dream_damage(lr, args.clean_frac)
                sigma = (torch.randn(B, device=dev) * args.p_std + args.p_mean).exp().clamp(2e-3, 20)
                start = torch.rand(B, device=dev) < args.start_frac
                sigma = torch.where(start, 1.5 + torch.rand(B, device=dev), sigma)
            lr_now = args.lr * min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(hr, lr_in, prev, has, sigma, start)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not (math.isfinite(loss.item()) and math.isfinite(gn.item())):
                live.log(f"non-finite loss at step {step}, skipped")
                continue
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)
            step += 1
            win_n += 1
            lv = loss.item()
            loss_avg = lv if loss_avg is None else 0.98 * loss_avg + 0.02 * lv
            now = time.time()
            if step % 50 == 0:
                it_s = win_n / (now - win_t)
                t_used += now - win_t
                win_t, win_n = now, 0
                live.metric({"step": step, "time": now, "loss": round(loss_avg, 5), "it_s": round(it_s, 2), "lr": lr_now})
                live.update(step=step, loss=round(loss_avg, 5), it_s=round(it_s, 2), lr=lr_now, seconds=t_used,
                            peak_vram_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2))
                if step % 500 == 0:
                    live.log(f"step {step}  loss {loss_avg:.4f}  {it_s:.2f} it/s  lr {lr_now:.1e}  grad {gn.item():.2f}  "
                             f"VRAM {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
            if now - last_val >= args.val_every_min * 60:
                do_val()
                last_val = time.time()
            if now - last_ck >= args.ckpt_every_min * 60:
                save(latest)
                last_ck = time.time()
                if now - last_keep >= args.keep_every_hours * 3600:
                    import shutil
                    shutil.copyfile(latest, ck_dir / f"step{step:07d}.pt")
                    last_keep = time.time()
                    live.log(f"keeper step{step:07d}.pt saved")
        save(latest)
        do_val()
        save(latest)
        live.log(f"finished at step {step}; peak VRAM {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
        live.update(stage="done", exit=0, step=step)
    except KeyboardInterrupt:
        save(latest)
        live.log(f"stopped by Ctrl+C at step {step} (saved)")
        live.update(stage="stopped", exit=130, step=step)
    except Exception:
        live.log("CRASHED:\n" + traceback.format_exc())
        try:
            save(ck_dir / "crash.pt")
        except Exception:
            pass
        live.update(stage="crashed", exit=1, step=step)
        raise


if __name__ == "__main__":
    main()
