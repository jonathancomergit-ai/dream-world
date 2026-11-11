"""Judge PAINTER v1 against bicubic and sharper eyes v1 (2026-10-24).

  python train/painter/eval_painter.py <painter.pt> [tag] [--steps=6] [--frames=300]

Two tests, the same 4 held-out walks the 20-walk judge shows (Amsterdam, Bangkok, Chiang Mai, Stockholm):
  REAL:  the real 96x168 frames in, painted one after another (each frame gets the painter's previous output, slid
         by the turn measured on the low-res frames) -> compared with the REAL 192x336 footage: L1 (0-255), sharpness
         (fine detail / real's = 1.0), flicker (how much fine detail jumps frame to frame / real's = 1.0).
  DREAM: v9 pick's own 40 s dreams from the judge (train/_exp/v9/dreams/v11judge_v9pick_seed1234.npy) painted the same
         way (no ground truth): sharpness vs the real walk's at 5 / 10 / 20 / 30 s, and flicker.
Writes train/painter/out/eval_<tag>.json, a phone card (eval_<tag>_card.jpg) and a phone video
(eval_<tag>_dream.mp4: sharper eyes v1 on top, painter below, 2 dreams x 20 s)."""
import json
import pathlib
import sys
import time

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
from train_sr import gauss_blur, lap_energy                           # noqa: E402

WALKS = [47706, 151029, 193856, 376281]                 # = the judge's rows 0, 3, 5, 8
NAMES = ["Amsterdam", "Bangkok", "Chiang Mai", "Stockholm"]
DREAM_ROWS = [0, 3, 5, 8]
DEV = "cuda"


def to_ms(u8):
    return torch.from_numpy(np.ascontiguousarray(u8)).to(DEV).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def to_u8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def hf(x):
    """fine detail = picture - its 1.5 px blur (luma-ish: mean of RGB)."""
    m = x.mean(1, keepdim=True)
    return m - gauss_blur(m, torch.full((m.shape[0],), 1.5, device=m.device))


@torch.no_grad()
def paint_chain(painter, lr_seq, steps, seed=0):
    """lr_seq [B, T, 3, 96, 168] -> [B, T, 3, 192, 336], each frame painted with the previous output slid by the turn."""
    B, T = lr_seq.shape[:2]
    out, prev = [], None
    g = torch.Generator(device=DEV).manual_seed(seed)
    for t in range(T):
        lr = lr_seq[:, t]
        if prev is None:
            p, has = torch.zeros(B, 3, 192, 336, device=DEV), torch.zeros(B, device=DEV)
        else:
            p, has = slide(prev, est_shift(lr_seq[:, t - 1], lr) * 2), torch.ones(B, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prev = painter.sample(lr, p, has, steps=steps, generator=g).float()
        out.append(prev)
    return torch.stack(out, 1)


@torch.no_grad()
def sharp_eyes(lr_seq):
    from upscaler import Upscaler
    up = Upscaler(graphs=False)
    B, T = lr_seq.shape[:2]
    return torch.stack([up.up(lr_seq[:, t]).float() for t in range(T)], 1)


def bicubic(lr_seq):
    B, T = lr_seq.shape[:2]
    return F.interpolate(lr_seq.flatten(0, 1), scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1).view(B, T, 3, 192, 336)


def sharpness(seq):
    """[B, T, 3, H, W] -> [B, T] mean |Laplacian of luma|."""
    B, T = seq.shape[:2]
    return lap_energy(seq.flatten(0, 1)).view(B, T)


def flicker(seq):
    """mean |change of fine detail| between consecutive frames, per walk [B]."""
    B, T = seq.shape[:2]
    h = hf(seq.flatten(0, 1)).view(B, T, 1, *seq.shape[-2:])
    return (h[:, 1:] - h[:, :-1]).abs().mean((1, 2, 3, 4))


_CF = None


def face_count(seq, every=20):
    """FACE CHECK (privacy): deface's CenterFace (the detector the blur used) on 1280x720 blow-ups of every
    `every`-th frame of seq [B, T, 3, H, W]. Faces per frame at confidence >= 0.5 and >= 0.2. If a painter painted
    faces back in, it would fire more often than on the face-blurred real footage."""
    global _CF
    import cv2
    if _CF is None:
        from deface.centerface import CenterFace
        _CF = CenterFace(in_shape=None, backend="opencv")
    frames = to_u8(seq[:, ::every].flatten(0, 1))
    c5 = c2 = 0
    for f in frames:
        d, _ = _CF(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_CUBIC), threshold=0.2)
        c2 += len(d)
        c5 += int((d[:, 4] >= 0.5).sum()) if len(d) else 0
    return {"per_frame_conf0.5": round(c5 / len(frames), 3), "per_frame_conf0.2": round(c2 / len(frames), 3),
            "frames": len(frames)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    ck_path = pathlib.Path(args[0])
    tag = args[1] if len(args) > 1 else ck_path.stem
    steps, T = int(opts.get("steps", 6)), int(opts.get("frames", 300))
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    painter = Painter(**{}).to(DEV).eval()
    painter.load_state_dict(ck["ema"])
    _, hr_mm, _, _ = load_cache_any(DW / "data" / "cache_v11")
    _, lr_mm, _, _ = load_cache_any(DW / "data" / "cache_v3")
    res = {"ckpt": str(ck_path), "step": ck.get("step"), "steps": steps, "frames": T}
    t0 = time.time()
    # ---- REAL walks
    lr_real = torch.stack([to_ms(lr_mm[s:s + T]) for s in WALKS])
    hr_real = torch.stack([to_ms(hr_mm[s:s + T]) for s in WALKS])
    real_sharp = sharpness(hr_real)
    real_flick = flicker(hr_real)
    outs = {"bicubic": bicubic(lr_real), "sharp eyes v1": sharp_eyes(lr_real), "painter": paint_chain(painter, lr_real, steps)}
    res["real"] = {}
    for k, v in outs.items():
        res["real"][k] = {"L1": round(float((v - hr_real).abs().mean() * 127.5), 3),
                          "sharpness": round(float((sharpness(v) / real_sharp).mean()), 3),
                          "flicker": round(float((flicker(v) / real_flick).mean()), 3)}
        print(f"REAL  {k:<14} {res['real'][k]}", flush=True)
    res["faces"] = {"real": {"REAL 2x (face-blurred)": face_count(hr_real)}}
    for k, v in outs.items():
        res["faces"]["real"][k] = face_count(v)
    print("FACES real:", res["faces"]["real"], flush=True)
    del outs
    torch.cuda.empty_cache()
    # ---- v9 pick's dreams
    dr = np.load(TRAIN / "_exp" / "v9" / "dreams" / "v11judge_v9pick_seed1234.npy", mmap_mode="r")
    lr_dream = torch.stack([to_ms(dr[r][:T]) for r in DREAM_ROWS])
    outs = {"bicubic": bicubic(lr_dream), "sharp eyes v1": sharp_eyes(lr_dream), "painter": paint_chain(painter, lr_dream, steps)}
    res["dream"] = {}
    at = [t for t in (49, 99, 199, 299) if t < T]
    for k, v in outs.items():
        s = sharpness(v) / real_sharp.mean(1, keepdim=True)
        res["dream"][k] = {"sharpness_at_5_10_20_30s": [round(float(s[:, t].mean()), 3) for t in at],
                           "flicker": round(float((flicker(v) / real_flick).mean()), 3)}
        print(f"DREAM {k:<14} {res['dream'][k]}", flush=True)
    res["faces"]["dream"] = {k: face_count(v) for k, v in outs.items()}
    print("FACES dream:", res["faces"]["dream"], flush=True)
    res["secs"] = round(time.time() - t0)
    od = HERE / "out"
    od.mkdir(exist_ok=True)
    (od / f"eval_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # ---- phone video: sharper eyes v1 (top) / painter (bottom), Amsterdam + Stockholm dreams, 20 s each
    import cv2
    import imageio.v2 as imageio
    se, pa = to_u8(outs["sharp eyes v1"][[0, 3], :200].flatten(0, 1)), to_u8(outs["painter"][[0, 3], :200].flatten(0, 1))
    w = imageio.get_writer(str(od / f"eval_{tag}_dream.mp4"), fps=10, codec="libx264", pixelformat="yuv420p",
                           quality=8, macro_block_size=16, ffmpeg_log_level="error")
    for i in range(len(se)):
        a, b = se[i].copy(), pa[i].copy()
        for img, txt in ((a, "sharper eyes v1"), (b, "PAINTER")):
            cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 120), 1)
        cv2.putText(b, f"{(i % 200 + 1) / 10:4.1f}s", (280, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        frame = np.concatenate([a, b], 0)
        w.append_data(cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4))
    w.close()
    # ---- phone card: one walk at 5 / 10 / 20 s, rows bicubic / sharper eyes / painter
    from PIL import Image, ImageDraw, ImageFont
    font = lambda px, b=True: ImageFont.truetype("segoeuib.ttf" if b else "segoeui.ttf", px)
    Wd, cw = 1080, 1056
    ch = int(cw * 192 / 336)
    img = Image.new("RGB", (Wd, 4000), (18, 20, 26))
    d = ImageDraw.Draw(img)
    d.text((24, 20), "Painter v1 vs sharper eyes", font=font(54), fill=(240, 240, 240))
    y = 96
    for k in ("bicubic", "sharp eyes v1", "painter"):
        rr, dd = res["real"][k], res["dream"][k]
        d.text((24, y), f"{k}:  detail {rr['sharpness']:.2f} of real · flicker {rr['flicker']:.2f} · error {rr['L1']:.1f}",
               font=font(34, k == "painter"), fill=(80, 210, 120) if k == "painter" else (220, 220, 220))
        y += 48
    d.text((24, y + 6), "(real walks; 1.00 = like real footage. Dreams below: v9 pick, Amsterdam)", font=font(28, False),
           fill=(150, 155, 165))
    y += 70
    for t in [t for t in (49, 99, 199) if t < T]:
        d.text((24, y), f"dream at {(t + 1) // 10} s", font=font(36), fill=(240, 240, 240))
        y += 50
        for k in ("sharp eyes v1", "painter"):
            fr = to_u8(outs[k][0, t:t + 1])[0]
            img.paste(Image.fromarray(fr).resize((cw, ch), Image.LANCZOS), (12, y))
            d.text((24, y + 8), k, font=font(32), fill=(255, 230, 120))
            y += ch + 10
        y += 20
    img.crop((0, 0, Wd, y + 10)).save(od / f"eval_{tag}_card.jpg", quality=86)
    print("saved", od / f"eval_{tag}.json", f"({res['secs']} s)", flush=True)


if __name__ == "__main__":
    main()
