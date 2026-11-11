"""Honest check of the 2x upscaler (step 4). Writes train/sr/out/eval_<tag>.json + pictures.

    data\\venv\\Scripts\\python.exe train\\sr\\eval_sr.py <tag> name=ckpt [name=ckpt ...]
      e.g. eval_sr.py final l1only=train/sr/checkpoints/sr1/l1_only.pt gan=train/sr/checkpoints/sr1/final.pt

1. REAL held-out frames (the 10 judge walks' first 40 s = 4000 frames, never trained on), input = the cache frame,
   and again with the v9 one-call WRECK as input: L1 (0..255) vs the real 192x336 frame (colour-matched to the
   cache like in training), sharpness = median mean|Laplacian| ratio out / real (1.0 = as much fine detail as real
   footage), colour shift = mean of (2x2 mean of the output - input) per channel.
2. DREAM frames (no ground truth): train/_exp/v9/dreams/sanity_v7_nopatch_seed1234.npy (v9 pick, 10 judge walks x 2,
   40 s) at 0/5/10/20/30 s: sharpness vs bicubic, colour shift, faithfulness |2x2 mean(out) - dream frame|.
3. FLICKER: 40 consecutive frames (t = 100..139) of all 20 dreams and of the 10 real walks: mean |frame - previous|
   for bicubic vs upscaler, and "detail flicker" = mean |change of (out - bicubic)| / mean |out - bicubic|
   (how much the invented detail jumps around, relative to its size; real 2x footage gives the natural value).
4. FACES: deface's CenterFace (threshold 0.5 and 0.2) on 1280x720 resizes of bicubic vs upscaler outputs of dream
   frames and real test frames: if the upscaler painted faces back in, it would fire more often.
5. SPEED: ms per frame at batch 1 and 4, fp16, eager vs CUDA graph (upscaler.Upscaler).
Pictures (train/sr/out/): real_<tag>.jpg (bicubic | upscaler | real 2x), wreck_<tag>.jpg, dream_<tag>.jpg,
dream_zoom_<tag>.jpg (3x crops).
"""
import json
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = pathlib.Path(__file__).resolve().parent
DW = SR.parent.parent
sys.path.insert(0, str(SR.parent))
sys.path.insert(0, str(SR))
from train_sr import HRD, bicubic_up, colour_map, lap_energy, open_data, to_ms   # noqa: E402
from upscaler import Upscaler                                                      # noqa: E402

DREAMS = DW / "train" / "_exp" / "v9" / "dreams" / "sanity_v7_nopatch_seed1234.npy"
OUT = SR / "out"
dev = "cuda"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def u8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def down2(x):
    return F.avg_pool2d(x, 2)


def label(im, text):
    im = np.ascontiguousarray(im[:, :, ::-1]).copy()
    cv2.putText(im, text, (5, 18), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, text, (5, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def zoom(im, k):
    return cv2.resize(im, (im.shape[1] * k, im.shape[0] * k), interpolation=cv2.INTER_NEAREST)


def run_models(models, x, bs=32):
    """{name: [N,3,192,336] float32 cpu} for x [N,3,96,168] (cpu float)."""
    res = {"bicubic": []}
    for n in models:
        res[n] = []
    for a in range(0, len(x), bs):
        xb = x[a:a + bs].to(dev)
        res["bicubic"].append(bicubic_up(xb).cpu())
        for n, m in models.items():
            res[n].append(m.up(xb).cpu())
    return {k: torch.cat(v) for k, v in res.items()}


def main():
    tag = sys.argv[1]
    specs = [a.split("=", 1) for a in sys.argv[2:]]                 # name=ckpt or name=ckpt@strength
    def load(spec, **kw):
        path, _, st = spec.partition("@")
        return Upscaler(path, strength=float(st) if st else 1.0, **kw)
    models = {n: load(p, graphs=False) for n, p in specs}
    first = specs[0][1].partition("@")[0]
    ck = torch.load(first, map_location="cpu", weights_only=False)
    M = torch.tensor(ck["colour_fit"], dtype=torch.float32, device=dev)
    info, hr_mm, fm, wr_mm = open_data()
    cidx = np.load(HRD / "cache_idx.npy")
    split = np.load(HRD / "split.npy")
    walk = np.load(HRD / "walk.npy")
    sig = np.load(HRD / "wreck_sigma.npy")
    res = {"models": dict(specs), "steps": {n: m.step for n, m in models.items()}}

    # ---------------------------------------------------------------- 1. real held-out frames (streamed in batches)
    test = np.flatnonzero(split == 2)
    pick = [int(np.flatnonzero(walk[test] == w)[150]) for w in (0, 3, 4, 5, 7, 9)]     # picture frames
    idx = np.concatenate([np.flatnonzero(walk[test] == w)[100:140] for w in range(10)])  # flicker: 40 consecutive
    sub = np.arange(0, len(test), 20)                                                  # faces
    keep_ix = np.unique(np.concatenate([pick, idx, sub]))
    pos = {int(k): j for j, k in enumerate(keep_ix)}
    names = ["bicubic"] + list(models)
    real_outs = None
    for inp_name in ("clean", "wreck"):
        acc = {n: {"abs": 0.0, "sq": 0.0, "px": 0, "sharp": [], "cs": torch.zeros(3)} for n in names}
        kept = {n: torch.zeros(len(keep_ix), 3, 192, 336, dtype=torch.float16) for n in names + ["REAL 2x"]}
        n_used = 0
        for a in range(0, len(test), 64):
            rows = test[a:a + 64]
            x = to_ms(torch.from_numpy(np.stack([fm[cidx[r]] for r in rows])), dev)
            mask = torch.ones(len(rows), dtype=torch.bool, device=dev)
            if inp_name == "wreck":
                mask = torch.from_numpy(np.isfinite(sig[rows])).to(dev)
                x = torch.where(mask[:, None, None, None], to_ms(torch.from_numpy(np.array(wr_mm[rows])), dev), x)
            h = colour_map(to_ms(torch.from_numpy(np.array(hr_mm[rows])), dev), M)
            eh = lap_energy(h)
            outs = {"bicubic": bicubic_up(x)}
            for n, m in models.items():
                outs[n] = m.up(x)
            for n, o in outs.items():
                d = (o - h)[mask]
                acc[n]["abs"] += float(d.abs().sum()); acc[n]["sq"] += float((d / 2).pow(2).sum()); acc[n]["px"] += d.numel()
                acc[n]["sharp"].append((lap_energy(o) / eh)[mask].cpu())
                acc[n]["cs"] += ((down2(o) - x)[mask]).sum((0, 2, 3)).cpu() / (96 * 168)
            n_used += int(mask.sum())
            for j, k in enumerate(range(a, a + len(rows))):
                if k in pos:
                    for n, o in outs.items():
                        kept[n][pos[k]] = o[j].half().cpu()
                    kept["REAL 2x"][pos[k]] = h[j].half().cpu()
        r = {}
        for n in names:
            A = acc[n]
            r[n] = {"L1_255": round(A["abs"] / A["px"] * 127.5, 3), "PSNR": round(float(-10 * np.log10(A["sq"] / A["px"])), 2),
                    "sharpness_vs_real": round(float(torch.cat(A["sharp"]).median()), 3),
                    "colour_shift_rgb": [round(float(c) / n_used * 127.5, 2) for c in A["cs"]]}
        r["n"] = n_used
        res[f"real_{inp_name}"] = r
        print(f"REAL ({inp_name} input, {n_used} frames):", json.dumps(r), flush=True)
        rows_ = []
        for i in pick:
            tiles = [label(zoom(u8(kept["bicubic"][pos[i]][None])[0], 2), f"bicubic ({inp_name} input)")]
            tiles += [label(zoom(u8(kept[n][pos[i]][None])[0], 2), n) for n in models]
            tiles += [label(zoom(u8(kept["REAL 2x"][pos[i]][None])[0], 2), "REAL 2x")]
            rows_.append(np.concatenate(tiles, 1))
        cv2.imwrite(str(OUT / f"{'real' if inp_name == 'clean' else 'wreck'}_{tag}.jpg"), np.concatenate(rows_, 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if inp_name == "clean":
            real_outs = kept
    # ---------------------------------------------------------------- 2. dream frames
    dd = np.load(DREAMS, mmap_mode="r")
    times = [0, 49, 99, 199, 299]
    dx = to_ms(torch.from_numpy(np.array(dd[:, times]).reshape(-1, 96, 168, 3)), "cpu")      # [20*5]
    douts = run_models(models, dx)
    r = {}
    for n, o in douts.items():
        sh = torch.cat([lap_energy(o[a:a + 50].to(dev)).cpu() for a in range(0, len(o), 50)])
        r[n] = {"lap_energy_by_time": [round(float(sh.reshape(20, 5)[:, j].mean()), 2) for j in range(5)],
                "colour_shift_rgb": [round(c, 2) for c in ((down2(o) - dx).mean((0, 2, 3)) * 127.5).tolist()],
                "faithful_mad_255": round(float((down2(o) - dx).abs().mean()) * 127.5, 2)}
    eb = np.array(r["bicubic"]["lap_energy_by_time"])
    for n in models:
        r[n]["sharpness_vs_bicubic_by_time"] = (np.array(r[n]["lap_energy_by_time"]) / eb).round(2).tolist()
    r["times_s"] = [0, 5, 10, 20, 30]
    res["dream"] = r
    print("DREAM:", json.dumps(r), flush=True)
    # picture: 4 walks (Amsterdam, Bangkok, Chiang Mai, Stockholm = rows 0, 3, 5, 8) x 5 times, bicubic over model
    names = list(models)
    blocks = []
    for w in (0, 3, 5, 8):
        for n in ["bicubic"] + names:
            row = [label(zoom(u8(douts[n][w * 5 + j:w * 5 + j + 1])[0], 2), f"{n} {r['times_s'][j]}s") for j in range(5)]
            blocks.append(np.concatenate(row, 1))
        blocks.append(np.full((10, blocks[-1].shape[1], 3), 40, np.uint8))
    cv2.imwrite(str(OUT / f"dream_{tag}.jpg"), np.concatenate(blocks, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # zoomed crops (3x) of the middle of the 5 s and 20 s frames, bicubic | models
    crops = []
    for w in (0, 3, 5, 8, 13, 16):
        for j in (1, 3):
            k = w * 5 + j
            tiles = [label(zoom(u8(douts[n][k:k + 1])[0][48:144, 96:240], 3), f"{n} walk{w} {r['times_s'][j]}s")
                     for n in ["bicubic"] + names]
            crops.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(OUT / f"dream_zoom_{tag}.jpg"), np.concatenate(crops, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ---------------------------------------------------------------- 3. flicker
    def flicker(seq_outs):
        """seq_outs {name: [S, T, 3, H, W]} -> stats."""
        bic = seq_outs["bicubic"]
        st = {}
        for n, o in seq_outs.items():
            ch = (o[:, 1:] - o[:, :-1]).abs().mean() * 127.5
            st[n] = {"frame_change_255": round(float(ch), 3)}
            if n != "bicubic":
                det = o - bic
                st[n]["detail_size_255"] = round(float(det.abs().mean()) * 127.5, 3)
                st[n]["detail_change_255"] = round(float((det[:, 1:] - det[:, :-1]).abs().mean()) * 127.5, 3)
                st[n]["detail_flicker_ratio"] = round(st[n]["detail_change_255"] / max(st[n]["detail_size_255"], 1e-6), 3)
                st[n]["change_vs_bicubic"] = round(float(ch) / st["bicubic"]["frame_change_255"], 3)
        return st
    seq = to_ms(torch.from_numpy(np.array(dd[:, 100:140]).reshape(-1, 96, 168, 3)), "cpu")
    so = run_models(models, seq)
    fl = {"dream": flicker({k: v.reshape(20, 40, 3, 192, 336) for k, v in so.items()})}
    del so
    # real: the 10 walks' frames 100..139 (consecutive real 10 fps) + the real 2x footage itself
    rs = {k: v[[pos[int(i)] for i in idx]].float().reshape(10, 40, 3, 192, 336) for k, v in real_outs.items()}
    fl["real"] = flicker(rs)
    res["flicker"] = fl
    print("FLICKER:", json.dumps(fl), flush=True)
    del rs

    # stability: (a) slide the dream frame 1 px sideways (what turning does): is the output the same picture slid 2 px?
    #            (b) add tiny noise (1/255): how much does the invented detail change?
    st = {}
    xs = dx.to(dev)
    for n, m in models.items():
        y0 = m.up(xs)
        det = (y0 - bicubic_up(xs))[..., 8:-8, 8:-8].abs().mean()
        y1 = m.up(torch.roll(xs, 1, dims=-1))
        sh = (torch.roll(y0, 2, dims=-1) - y1)[..., 8:-8, 8:-8].abs().mean()
        y2 = m.up(xs + torch.randn_like(xs) * (2 / 255))
        nz = (y2 - y0)[..., 8:-8, 8:-8].abs().mean()
        st[n] = {"detail_size_255": round(float(det) * 127.5, 3), "shift1px_mismatch_255": round(float(sh) * 127.5, 3),
                 "noise1lvl_change_255": round(float(nz) * 127.5, 3)}
    yb = bicubic_up(xs)
    st["bicubic"] = {"shift1px_mismatch_255": round(float((torch.roll(yb, 2, dims=-1) - bicubic_up(torch.roll(xs, 1, dims=-1)))[..., 8:-8, 8:-8].abs().mean()) * 127.5, 3),
                     "noise1lvl_change_255": round(float((bicubic_up(xs + torch.randn_like(xs) * (2 / 255)) - yb)[..., 8:-8, 8:-8].abs().mean()) * 127.5, 3)}
    res["stability_dream"] = st
    print("STABILITY:", json.dumps(st), flush=True)

    # video: 2 dreams (Amsterdam, Chiang Mai), first 20 s, bicubic | last model, 2x zoom, 10 fps
    import subprocess
    last = list(models)[-1]
    vid = OUT / f"dream_{tag}.mp4"
    W2, H2 = 336 * 2 * 2, 192 * 2
    pr = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W2}x{H2}", "-r", "10",
                           "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    for row in (0, 5):
        xs = to_ms(torch.from_numpy(np.array(dd[row, :200])), "cpu")
        vo = run_models({last: models[last]}, xs)
        for t in range(200):
            a = label(zoom(u8(vo["bicubic"][t:t + 1])[0], 2), f"bicubic  {t / 10:.1f}s")
            b = label(zoom(u8(vo[last][t:t + 1])[0], 2), f"{last}")
            pr.stdin.write(np.ascontiguousarray(np.concatenate([a, b], 1)).tobytes())
    pr.stdin.close(); pr.wait()
    res["video"] = str(vid)

    # ---------------------------------------------------------------- 4. faces
    from deface.centerface import CenterFace
    cf = CenterFace(in_shape=None, backend="opencv")
    def count(frames_u8):
        c5 = c2 = 0
        for f in frames_u8:
            big = cv2.resize(f, (1280, 720), interpolation=cv2.INTER_CUBIC)
            d, _ = cf(big, threshold=0.2)
            c2 += len(d)
            c5 += int((d[:, 4] >= 0.5).sum()) if len(d) else 0
        return {"per_frame_thr0.5": round(c5 / len(frames_u8), 3), "per_frame_thr0.2": round(c2 / len(frames_u8), 3)}
    faces = {"dream": {}, "real": {}}
    for n, o in douts.items():
        faces["dream"][n] = count(u8(o))
    sub = np.arange(0, len(test), 20)
    for n, o in real_outs.items():
        faces["real"][n + (" (face-blurred)" if n == "REAL 2x" else "")] = count(u8(o[[pos[int(i)] for i in sub]].float()))
    faces["n_frames"] = {"dream": len(dx), "real": len(sub)}
    res["faces"] = faces
    print("FACES:", json.dumps(faces), flush=True)

    # ---------------------------------------------------------------- 5. speed
    sp = {}
    for n, p in specs:
        for graphs in (False, True):
            u = load(p, graphs=graphs)
            for B in (1, 4):
                x = torch.rand(B, 3, 96, 168, device=dev) * 2 - 1
                for _ in range(10):
                    u.up(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200):
                    u.up(x)
                torch.cuda.synchronize()
                sp[f"{n} {'graph' if graphs else 'eager'} batch{B}"] = round((time.perf_counter() - t0) / 200 * 1000 / B, 3)
            del u
        # fp16 vs fp32 agreement
        u16, u32 = load(p, graphs=False), load(p, graphs=False, dtype=torch.float32)
        x = dx[:8].to(dev)
        sp[f"{n} fp16 vs fp32 max diff (0..255)"] = round(float((u16.up(x) - u32.up(x)).abs().max()) * 127.5, 3)
    sp["note"] = "ms per frame, includes the bilinear base + clamp; GPU otherwise idle"
    res["speed_ms_per_frame"] = sp
    print("SPEED:", json.dumps(sp), flush=True)
    (OUT / f"eval_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("saved", OUT / f"eval_{tag}.json")


if __name__ == "__main__":
    main()
