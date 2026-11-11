"""Dream dial lab: which DREAMING settings keep a walk from melting? (no retraining)

    data\\venv\\Scripts\\python.exe train\\dial_lab.py                 # newest dream_v0 save
    data\\venv\\Scripts\\python.exe train\\dial_lab.py --frames 300    # longer walks

The model is the same in every test. Only HOW we ask it to dream changes:
  steps   how many clean-up passes turn static into a frame (more = sharper, slower)
  heun    a more careful clean-up pass (2x cost)
  cnoise  "context noise": blur the remembered frames a bit and TELL the model so. It was
          trained with that blur, so it learns not to copy its own mistakes too faithfully.
  churn   re-inject a little fresh noise during each clean-up pass (EDM "stochastic"
          sampling). Deterministic cleaning tends to settle on the safe average (beige mush);
          a bit of randomness keeps it committing to actual details.
  anchor  after each frame, nudge its brightness/contrast back toward the starting real
          frames' (a cheap "don't fade to beige" leash; not learned, just colour maths).

How "melting" is measured: sharpness = average edge strength (absolute Laplacian) of a frame,
divided by the average for real footage. Real frames score ~1.0; beige mush scores near 0.
Melt time = first moment the 1-second moving average drops below half of the walk's OWN first
second (even a perfect guess is softer than real video at 64x112, so we compare it to itself).

Every setting dreams the SAME 3 walks (3 cities, held-out footage the model never trained on)
with the SAME random seed, so differences come from the settings alone.
Outputs: train/out/dial_<name>.mp4 (shows up in Dream Lab), train/dial/summary.md, train/dial/grid.jpg
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))
from data import load_cache, to_model_space, to_uint8_image  # noqa: E402
from model import build_model                               # noqa: E402

OUT = HERE / "dial"
VIDS = HERE / "out"

# The walk: forward, turn right, forward, turn left (longer), forward. 1 = W, 5 = left, 6 = right.
def script(n):
    base = [1] * 40 + [6] * 15 + [1] * 40 + [5] * 30 + [1] * 40 + [6] * 15 + [1] * 20
    return (base * (n // len(base) + 1))[:n]

# The settings to compare. "now" = what Dream Lab's rollouts use today.
CONFIGS = [
    dict(name="now",            steps=5,  heun=False, cnoise=0.0,  churn=0.0,  anchor=False),
    dict(name="steps10",        steps=10, heun=False, cnoise=0.0,  churn=0.0,  anchor=False),
    dict(name="heun5",          steps=5,  heun=True,  cnoise=0.0,  churn=0.0,  anchor=False),
    dict(name="cnoise.05",      steps=5,  heun=False, cnoise=0.05, churn=0.0,  anchor=False),
    dict(name="cnoise.10",      steps=5,  heun=False, cnoise=0.10, churn=0.0,  anchor=False),
    dict(name="cnoise.20",      steps=5,  heun=False, cnoise=0.20, churn=0.0,  anchor=False),
    dict(name="churn",          steps=8,  heun=False, cnoise=0.0,  churn=4.0,  anchor=False),
    dict(name="churn+cnoise.1", steps=8,  heun=False, cnoise=0.10, churn=4.0,  anchor=False),
    dict(name="anchor",         steps=5,  heun=False, cnoise=0.0,  churn=0.0,  anchor=True),
    dict(name="churn+cn+anchor", steps=8, heun=False, cnoise=0.10, churn=4.0,  anchor=True),
    # round 2 (combining round 1's winners: more steps hold shapes, the anchor keeps colour alive)
    dict(name="steps10+anchor",  steps=10, heun=False, cnoise=0.0, churn=0.0,  anchor=0.2),
    dict(name="steps10+anchor.1", steps=10, heun=False, cnoise=0.0, churn=0.0, anchor=0.1),
    dict(name="heun3",           steps=3,  heun=True,  cnoise=0.0, churn=0.0,  anchor=False),
    dict(name="heun3+anchor.1",  steps=3,  heun=True,  cnoise=0.0, churn=0.0,  anchor=0.1),
]


@torch.no_grad()
def dream_frame(model, ctx, act, steps, heun, cnoise, churn, g):
    """One dreamed frame. Same maths as Denoiser.sample, plus optional EDM 'churn'
    (Karras et al. 2022, Algorithm 2): before each step, add a little noise back and pretend
    we're slightly noisier than we are. It stops the clean-up from collapsing onto the average."""
    B, _, C, H, W = ctx.shape
    dev = ctx.device
    sc = torch.full((B,), float(cnoise), device=dev)
    if cnoise > 0:
        ctx = ctx + cnoise * torch.randn(ctx.shape, device=dev, generator=g)
    sig = model.karras_sigmas(steps, device=dev)
    x = torch.randn((B, C, H, W), device=dev, generator=g) * sig[0]
    gamma = min(churn / steps, 2 ** 0.5 - 1) if churn > 0 else 0.0
    for i in range(steps):
        s, s_next = sig[i], sig[i + 1]
        # churn only in the middle of the ladder (EDM's S_tmin/S_tmax): not at pure static, not at the end
        if gamma > 0 and 0.05 <= float(s) <= 1.0:
            s_hat = s * (1 + gamma)
            x = x + (s_hat ** 2 - s ** 2).sqrt() * torch.randn(x.shape, device=dev, generator=g)
            s = s_hat
        den, _ = model(x, s.expand(B), ctx, sc, act)
        d = (x - den) / s
        x_next = x + d * (s_next - s)
        if heun and s_next > 0:
            den2, _ = model(x_next, s_next.expand(B), ctx, sc, act)
            x_next = x + (d + (x_next - den2) / s_next) / 2 * (s_next - s)
        x = x_next
    return x.clamp(-1, 1)


def sharpness(frames_u8):
    """Edge strength per frame: mean |Laplacian| of the grayscale image. [T,H,W,3] uint8 -> [T]"""
    g = torch.from_numpy(frames_u8).float().mean(-1)[:, None] / 255.0     # [T,1,H,W]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)[None, None]
    return F.conv2d(g, k).abs().mean((1, 2, 3)).numpy()


def melt_time(sharp_norm, fps=10):
    ma = np.convolve(sharp_norm, np.ones(fps) / fps, mode="valid")
    idx = np.where(ma < 0.5)[0]
    return None if len(idx) == 0 else round(float(idx[0]) / fps, 1)


@torch.no_grad()
def rollout(model, frames_mm, actions_np, starts, keys, cfg, seed):
    """Dream all the walks at once (one batch = 3x faster than one by one).
    Returns uint8 [walks, T, H, W, 3]."""
    K = model.K
    dev = "cuda"
    ctx = torch.stack([to_model_space(torch.from_numpy(np.array(frames_mm[s:s + K])).to(dev)) for s in starts])
    # colour statistics of each walk's real starting frames, for the "anchor" leash
    ref_mean = ctx.mean(dim=(1, 3, 4))[:, :, None, None]        # [B,3,1,1]
    ref_std = ctx.std(dim=(1, 3, 4))[:, :, None, None]
    hist = [[int(a) if a >= 0 else 1 for a in actions_np[s:s + K - 1]] for s in starts]
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for a in keys:
            act = torch.tensor([h[-(K - 1):] + [a] for h in hist], device=dev)
            nxt = dream_frame(model, ctx, act, cfg["steps"], cfg["heun"], cfg["cnoise"], cfg["churn"], g).float()
            if cfg["anchor"]:
                # pull each colour channel's mean/contrast part of the way (anchor strength) back toward the real start
                m = nxt.mean(dim=(2, 3), keepdim=True)
                sd = nxt.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
                fixed = (nxt - m) / sd * ref_std + ref_mean
                w = 0.2 if cfg["anchor"] is True else float(cfg["anchor"])
                nxt = ((1 - w) * nxt + w * fixed).clamp(-1, 1)
            out.append(np.stack([to_uint8_image(f) for f in nxt]))
            ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)
            for h in hist:
                h.append(a)
    return np.stack(out, 1)


def write_mp4(path, frames, fps=10):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                            macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "dream_v0" / "latest.pt"))
    ap.add_argument("--frames", type=int, default=200, help="frames per walk (10 = 1 second)")
    ap.add_argument("--only", default="", help="comma list of config names to run")
    ap.add_argument("--tag", default="", help="suffix for grid/summary files, e.g. r2")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model = build_model(ck["model_cfg"]).cuda().eval()
    model.load_state_dict(ck["ema"])
    step = ck["step"]
    print(f"model step {step}")
    index, frames_mm, actions_np, _ = load_cache(DW / "data" / "cache")

    # one start per city, inside the held-out tail (last 3%) that training never sees
    starts = []
    for c in index["cities"]:
        h0 = c["end"] - int((c["end"] - c["start"]) * 0.03)
        starts.append((c["name"], h0 + 20))
    keys = script(args.frames)

    # what "sharp" means: average edge strength of real held-out frames
    real = np.concatenate([np.array(frames_mm[s:s + 200]) for _, s in starts])
    real_sharp = float(sharpness(real).mean())

    cfgs = [c for c in CONFIGS if not args.only or c["name"] in args.only.split(",")]
    results = []
    import imageio.v2 as imageio  # noqa: F401  (fail early if missing)
    from PIL import Image
    grid_rows = []
    for cfg in cfgs:
        t0 = time.time()
        melts, curves = [], []
        dreams = rollout(model, frames_mm, actions_np, [s for _, s in starts], keys, cfg, seed=1000)
        for dream in dreams:
            sh = sharpness(dream)
            # measured against the walk's OWN first second: even a perfect one-step guess is
            # softer than real video at 64x112, so "melting" = losing half of where it started
            sn = sh / max(float(sh[:10].mean()), 1e-6)
            curves.append(sn)
            melts.append(melt_time(sn))
        first_walk = dreams[0]
        secs = time.time() - t0
        fps = len(keys) / secs   # batched: all walks advance together
        avg_sharp = float(np.mean([c.mean() for c in curves]))
        late_sharp = float(np.mean([c[-50:].mean() for c in curves]))
        r = dict(cfg, melt=[m if m is not None else f">{len(keys)/10:.0f}" for m in melts],
                 sharp_avg=round(avg_sharp, 2), real_sharp_ratio=round(float(np.mean([sharpness(d[:10]).mean() for d in dreams])) / real_sharp, 2), sharp_end=round(late_sharp, 2), dream_fps=round(fps, 1))
        results.append(r)
        print(f"{cfg['name']:<16} melt(s) {r['melt']}  sharp avg {r['sharp_avg']}  end {r['sharp_end']}  {fps:.1f} frames/s")
        # video of the first walk, 4x upscaled, labelled by the file name in Dream Lab
        big = [np.array(Image.fromarray(f).resize((448, 256), Image.BICUBIC)) for f in first_walk]
        write_mp4(VIDS / f"dial_{cfg['name'].replace('+', '_')}.mp4", big)
        # grid row: frames at 0, 5, 10, 15, 19.9 s
        picks = [min(int(t * 10), len(first_walk) - 1) for t in (0.5, 5, 10, 15, len(keys) / 10 - 0.1)]
        grid_rows.append(np.concatenate([np.array(Image.fromarray(first_walk[p]).resize((224, 128), Image.BICUBIC)) for p in picks], 1))

    Image.fromarray(np.concatenate(grid_rows, 0)).save(OUT / f"grid{args.tag}.jpg", quality=88)
    with open(OUT / f"summary{args.tag}.md", "w", encoding="utf-8") as f:
        f.write(f"# Dream dial lab (model step {step}, {len(keys)} frames per walk)\n\n")
        f.write("Sharpness: real footage = 1.0, beige mush ~0. Melt = when the 1 s average drops below 0.5.\n")
        f.write("Grid rows are in this order; columns are 0.5 s, 5 s, 10 s, 15 s and the end of walk 1.\n\n")
        f.write("| setting | steps | heun | ctx noise | churn | anchor | melt time per walk (s) | sharp avg | sharp at end | dream fps |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['steps']} | {r['heun']} | {r['cnoise']} | {r['churn']} | {r['anchor']} | "
                    f"{', '.join(map(str, r['melt']))} | {r['sharp_avg']} | {r['sharp_end']} | {r['dream_fps']} |\n")
    json.dump(results, open(OUT / f"summary{args.tag}.json", "w"), indent=1)
    print("wrote", OUT / f"summary{args.tag}.md")


if __name__ == "__main__":
    main()
