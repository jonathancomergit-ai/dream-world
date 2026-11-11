"""Train "sharper eyes": the 2x upscaler (96x168 cache/dream frame -> 192x336), see README.md.

    data\\venv\\Scripts\\python.exe train\\sr\\train_sr.py --minutes 100 --tag sr1

DATA (data/cache_hr, built by build_hr.py + wreck_hr.py): train split only (every 13th frame of every walking city's
training part); val = the held-out tails (every 8th frame), test (judge walks) is never touched here.
  input  = the cache_v3 frame itself (the dream model's own world), and for part of the batch DAMAGED:
           - "wreck" (p 0.3): the v9 pick model's one-call clean-up of real frame + noise sigma 0.1-1 (wreck_hr.py)
           - "synthetic" (p 0.3): gaussian blur sigma 0.4-1.6 px and/or grain std 0-0.08 and/or a down-up resample
           - clean (the rest)
  target = the real 192x336 frame, colour-matched to the cache (a 3x4 colour fit, see fit_colour) so the upscaler
           learns detail, not a colour shift.
  Both mirrored left-right half the time.

LOSS: L1 (pixels, 0..1 scale) + after --gan-start steps a light adversarial term (--adv 0.05) from a small
spectral-norm patch critic that sees [picture, bilinear-upscaled input]. No pretrained networks (no VGG/LPIPS).
EMA of the generator weights (0.999) is what gets evaluated/used.

GPU: bf16 autocast, batch 16, ~2-3 GB. Saves train/sr/checkpoints/<tag>/latest.pt (+ every --keep-every steps).
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = pathlib.Path(__file__).resolve().parent
DW = SR.parent.parent
sys.path.insert(0, str(SR.parent))
sys.path.insert(0, str(SR))

from sr_model import PatchCritic, SharpEyes, n_params    # noqa: E402

HRD = DW / "data" / "cache_hr"
CACHE = DW / "data" / "cache_v3"


# ------------------------------------------------------------------------------------------------ data

def open_data():
    from data_v3 import load_cache_any
    info = json.loads((HRD / "index.json").read_text(encoding="utf-8"))
    hr = np.memmap(HRD / info["frame_file"], dtype=np.uint8, mode="r", shape=tuple(info["shape"]))
    _, fm, _, _ = load_cache_any(CACHE)
    wr = np.memmap(HRD / "wreck_96x168.u8", dtype=np.uint8, mode="r", shape=(info["shape"][0], 96, 168, 3))
    return info, hr, fm, wr


class Pairs(torch.utils.data.Dataset):
    """Item i (an HR row) -> (cache frame uint8 [96,168,3], HR uint8 [192,336,3], wreck uint8 [96,168,3], has_wreck)."""

    def __init__(self, rows):
        self.rows = np.asarray(rows)
        self.d = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if self.d is None:
            self.d = open_data()
            self.cidx = np.load(HRD / "cache_idx.npy")
            self.sig = np.load(HRD / "wreck_sigma.npy")
        _, hr, fm, wr = self.d
        r = int(self.rows[k])
        return (torch.from_numpy(np.array(fm[self.cidx[r]])), torch.from_numpy(np.array(hr[r])),
                torch.from_numpy(np.array(wr[r])), bool(np.isfinite(self.sig[r])))


def fit_colour(n=1500, seed=0):
    """3x4 affine colour map HR -> cache colours, least squares on 2x2-mean HR vs the cache frame, both box-blurred
    5x5 (so codec noise / face-blur spots don't matter). Returns float32 [4, 3] (rows R, G, B, 1)."""
    import cv2
    info, hr, fm, _ = open_data()
    cidx = np.load(HRD / "cache_idx.npy")
    split = np.load(HRD / "split.npy")
    rows = np.random.default_rng(seed).choice(np.flatnonzero(split == 0), n, replace=False)
    X, Y = [], []
    for r in np.sort(rows):
        lo = np.asarray(hr[r], np.float32).reshape(96, 2, 168, 2, 3).mean((1, 3))
        X.append(cv2.blur(lo, (5, 5))[::3, ::3].reshape(-1, 3))
        Y.append(cv2.blur(np.asarray(fm[cidx[r]], np.float32), (5, 5))[::3, ::3].reshape(-1, 3))
    X, Y = np.concatenate(X), np.concatenate(Y)
    M, *_ = np.linalg.lstsq(np.concatenate([X, np.ones((len(X), 1), np.float32)], 1), Y, rcond=None)
    Xa = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    return M.astype(np.float32), {"mean_shift_before": (Y - X).mean(0).round(3).tolist(),
                                  "mean_abs_before": round(float(np.abs(Y - X).mean()), 3),
                                  "mean_abs_after": round(float(np.abs(Y - Xa @ M).mean()), 3)}


def to_ms(u8, dev):
    """uint8 [B, H, W, 3] -> float [B, 3, H, W] in [-1, 1]."""
    return u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def colour_map(x, M):
    """x [B, 3, H, W] in [-1, 1] -> the same with the 0..255 affine colour map M [4, 3] applied."""
    u = (x + 1) * 127.5
    y = torch.einsum("bchw,cd->bdhw", u, M[:3]) + M[3][None, :, None, None]
    return (y / 127.5 - 1).clamp(-1, 1)


def gauss_blur(x, sigma):
    """Per-row gaussian blur, sigma [B] in pixels (0 = none)."""
    B = x.shape[0]
    r = 4
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t[None] / sigma.clamp(min=1e-3)[:, None]) ** 2)
    k = k / k.sum(1, keepdim=True)
    k = torch.where((sigma > 0)[:, None], k, F.one_hot(torch.tensor(r, device=x.device), 2 * r + 1).float()[None])
    C = x.shape[1]
    xp = F.pad(x, (r, r, r, r), mode="replicate").reshape(1, B * C, x.shape[2] + 2 * r, x.shape[3] + 2 * r)
    kh = k.repeat_interleave(C, 0)[:, None, None, :]
    kv = k.repeat_interleave(C, 0)[:, None, :, None]
    y = F.conv2d(xp, kh, groups=B * C)
    y = F.conv2d(y, kv, groups=B * C)
    return y.reshape(B, C, x.shape[2], x.shape[3])


def damage(lr, wreck, has_wreck, p_wreck, p_synth, g=None):
    """Mix of clean / wreck / synthetic damage per row (all in model space). Returns (input, kind [B] 0/1/2)."""
    B, dev = lr.shape[0], lr.device
    u = torch.rand(B, device=dev, generator=g)
    kind = torch.zeros(B, dtype=torch.long, device=dev)
    kind[(u < p_wreck) & has_wreck] = 1
    kind[(u >= p_wreck) & (u < p_wreck + p_synth)] = 2
    x = torch.where((kind == 1)[:, None, None, None], wreck, lr)
    syn = kind == 2
    if syn.any():
        xs = x[syn]
        n = xs.shape[0]
        r = torch.rand(n, 4, device=dev, generator=g)
        sig = torch.where(r[:, 0] < 0.7, 0.4 + 1.2 * r[:, 1], torch.zeros_like(r[:, 1]))
        xs = gauss_blur(xs, sig)
        dn = r[:, 2] < 0.3                                           # down-up resample (soft "dream" look)
        if dn.any():
            sc = float(0.5 + 0.3 * torch.rand((), device=dev, generator=g))
            small = F.interpolate(xs[dn], scale_factor=sc, mode="bilinear", align_corners=False, antialias=True)
            xs[dn] = F.interpolate(small, size=xs.shape[-2:], mode="bilinear", align_corners=False)
        std = torch.where(r[:, 3] < 0.6, 0.08 * torch.rand(n, device=dev, generator=g), torch.zeros(n, device=dev))
        xs = xs + std[:, None, None, None] * torch.randn(xs.shape, device=dev, generator=g)
        x = x.clone()
        x[syn] = xs.clamp(-1, 1)
    return x, kind


# ------------------------------------------------------------------------------------------------ metrics

def luma(x):
    """[-1,1] RGB -> 0..255 luma [B, 1, H, W]."""
    u = (x.clamp(-1, 1) + 1) * 127.5
    return (0.299 * u[:, 0] + 0.587 * u[:, 1] + 0.114 * u[:, 2])[:, None]


LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def lap_energy(x):
    """mean |Laplacian of luma| per image [B] (0..255 units) = how much fine detail there is."""
    k = LAP.to(x.device)[None, None]
    return F.conv2d(luma(x), k).abs().mean((1, 2, 3))


def bicubic_up(x):
    return F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)


@torch.no_grad()
def evaluate(G, rows, M, dev, wreck=False, bs=32):
    """-> dict: L1 (0..255) and sharpness ratio (|Lap| out / |Lap| real) for the model and bicubic."""
    ds = Pairs(rows)
    tot = {"l1": 0.0, "l1_bic": 0.0, "sharp": [], "sharp_bic": [], "n": 0}
    for a in range(0, len(rows), bs):
        items = [ds[k] for k in range(a, min(a + bs, len(rows)))]
        lr = to_ms(torch.stack([t[0] for t in items]), dev)
        hr = colour_map(to_ms(torch.stack([t[1] for t in items]), dev), M)
        if wreck:
            lr = torch.where(torch.tensor([t[3] for t in items], device=dev)[:, None, None, None],
                             to_ms(torch.stack([t[2] for t in items]), dev), lr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = G(lr)
        out = out.float().clamp(-1, 1)
        bic = bicubic_up(lr)
        tot["l1"] += float((out - hr).abs().mean((1, 2, 3)).sum()) * 127.5
        tot["l1_bic"] += float((bic - hr).abs().mean((1, 2, 3)).sum()) * 127.5
        e_hr = lap_energy(hr)
        tot["sharp"].append((lap_energy(out) / e_hr).cpu())
        tot["sharp_bic"].append((lap_energy(bic) / e_hr).cpu())
        tot["n"] += len(items)
    n = tot["n"]
    return {"l1": round(tot["l1"] / n, 3), "l1_bic": round(tot["l1_bic"] / n, 3),
            "sharp": round(float(torch.cat(tot["sharp"]).median()), 3),
            "sharp_bic": round(float(torch.cat(tot["sharp_bic"]).median()), 3)}


# ------------------------------------------------------------------------------------------------ training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sr1")
    ap.add_argument("--minutes", type=float, default=100.0, help="wall-clock budget (stops cleanly after)")
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--c", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--coarse", type=int, default=6)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--gan-start", type=int, default=6000, help="L1-only warm-up steps before the critic joins")
    ap.add_argument("--adv", type=float, default=0.05, help="adversarial weight (L1 is on a 0..1 scale)")
    ap.add_argument("--p-wreck", type=float, default=0.3)
    ap.add_argument("--p-synth", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--keep-every", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vram-cap-gb", type=float, default=12.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ck_dir = SR / "checkpoints" / args.tag
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_path = ck_dir / "log.txt"
    log = open(log_path, "a", encoding="utf-8")

    def say(s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    split = np.load(HRD / "split.npy")
    tr_rows, va_rows = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    va_eval = va_rows[np.linspace(0, len(va_rows) - 1, min(512, len(va_rows))).astype(int)]

    CL = torch.channels_last                                   # ~15-20% faster convs in bf16
    G = SharpEyes(args.c, args.blocks, args.coarse).to(dev).to(memory_format=CL)
    D = PatchCritic().to(dev).to(memory_format=CL)
    G_ema = SharpEyes(args.c, args.blocks, args.coarse).to(dev).to(memory_format=CL).eval()
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.9, 0.99))
    step, t_used = 0, 0.0
    M_np, cinfo = None, None
    if args.resume and (ck_dir / "latest.pt").exists():
        ck = torch.load(ck_dir / "latest.pt", map_location="cpu", weights_only=False)
        G.load_state_dict(ck["G"]); G_ema.load_state_dict(ck["G_ema"]); D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        step, t_used, M_np, cinfo = ck["step"], ck.get("t_used", 0.0), np.array(ck["colour_fit"], np.float32), ck["colour_info"]
        say(f"resumed {ck_dir / 'latest.pt'} at step {step}")
    if M_np is None:
        M_np, cinfo = fit_colour()
        say(f"colour fit HR -> cache: {cinfo}\n{M_np.round(4)}")
    M = torch.from_numpy(M_np).to(dev)
    say(f"G {n_params(G) / 1e6:.2f}M params ({G.cfg}), D {n_params(D) / 1e6:.2f}M; train {len(tr_rows)} frames, "
        f"val {len(va_rows)} (eval on {len(va_eval)}); args {vars(args)}")

    sampler = torch.utils.data.RandomSampler(Pairs(tr_rows), replacement=True, num_samples=10**9)
    dl = torch.utils.data.DataLoader(Pairs(tr_rows), batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    it = iter(dl)

    def save(name):
        ck = {"G": G.state_dict(), "G_ema": G_ema.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
              "opt_d": opt_d.state_dict(), "step": step, "t_used": t_used, "cfg": G.cfg, "colour_fit": M_np.tolist(),
              "colour_info": cinfo, "args": vars(args)}
        tmp = ck_dir / (name + ".tmp")
        torch.save(ck, tmp)
        tmp.replace(ck_dir / name)

    t_start = time.time() - t_used
    ema_p, g_p = list(G_ema.parameters()), list(G.parameters())
    hist = {"l1": [], "adv": [], "d": []}
    t_last = time.time()
    while step < args.max_steps:
        t_used = time.time() - t_start
        if t_used > args.minutes * 60:
            say(f"time budget reached ({args.minutes} min)")
            break
        lr_u8, hr_u8, wr_u8, has_w = next(it)
        lr = to_ms(lr_u8, dev)
        hr = colour_map(to_ms(hr_u8, dev), M)
        wr = to_ms(wr_u8, dev)
        has_w = has_w.to(dev)
        flip = torch.rand(lr.shape[0], device=dev) < 0.5
        f4 = flip[:, None, None, None]
        lr, hr, wr = torch.where(f4, lr.flip(-1), lr), torch.where(f4, hr.flip(-1), hr), torch.where(f4, wr.flip(-1), wr)
        x, kind = damage(lr, wr, has_w, args.p_wreck, args.p_synth)
        x, hr = x.contiguous(memory_format=CL), hr.contiguous(memory_format=CL)
        gan_on = step >= args.gan_start
        # ---- generator
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = G(x)
            l1 = (out.float() - hr).abs().mean() * 0.5                     # 0..1 scale
            loss = l1
            if gan_on:
                up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                adv = F.softplus(-D(out, up)).mean()                       # non-saturating
                loss = loss + args.adv * adv
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_g.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, g_p, 1 - (args.ema if step > 1000 else 0.99))
        hist["l1"].append(float(l1.detach()))
        # ---- critic
        if gan_on:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d_real = D(hr, up)
                d_fake = D(out.detach(), up)
                ld = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            hist["adv"].append(float(adv.detach())); hist["d"].append(float(ld.detach()))
        step += 1
        if step % 200 == 0:
            el = time.time() - t_last
            t_last = time.time()
            say(f"step {step} | L1 {np.mean(hist['l1'][-200:]) * 255:.2f} (0..255)"
                + (f" | adv {np.mean(hist['adv'][-200:]):.3f} D {np.mean(hist['d'][-200:]):.3f}" if hist["adv"] else "")
                + f" | {200 / el:.1f} it/s | GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                + f" | {t_used / 60:.1f} min")
            hist = {k: v[-1000:] for k, v in hist.items()}
        if step % args.eval_every == 0:
            ev = evaluate(G_ema, va_eval, M, dev)
            ew = evaluate(G_ema, va_eval, M, dev, wreck=True)
            say(f"VAL step {step}: clean input L1 {ev['l1']} (bicubic {ev['l1_bic']}), sharpness {ev['sharp']} "
                f"(bicubic {ev['sharp_bic']}, real 1.0) | wrecked input L1 {ew['l1']} (bic {ew['l1_bic']}), "
                f"sharp {ew['sharp']} (bic {ew['sharp_bic']})")
            with open(ck_dir / "val.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "min": round(t_used / 60, 1), "clean": ev, "wreck": ew}) + "\n")
            save("latest.pt")
        if step % args.keep_every == 0:
            save(f"step{step:06d}.pt")
        if step == args.gan_start:
            save("l1_only.pt")
    save("latest.pt")
    save("final.pt")
    say(f"done at step {step}")


if __name__ == "__main__":
    main()
