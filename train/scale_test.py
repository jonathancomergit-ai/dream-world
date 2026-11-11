"""Is v2's slow progress about the BIGGER FRAMES, or about v2's TRAINING? (2026-10-19)

    data\\venv\\Scripts\\python.exe train\\scale_test.py

Same idea as a science fair control: dream the same 3 held-out walks (one per city, never
trained on) with two sets of weights at two frame sizes, and compare melt times:

                     at 64x112 (v1's size)     at 96x168 (v2's size)
    v1 final         (v1's home turf)          (= v2 before any v2 training)
    v2 now           (did v2 keep v1's skill?) (v2's home turf)

Reading it:
  * v2-now melts late at 64x112 but early at 96x168 -> the skill is intact; the bigger frames
    are the problem (things look 1.5x bigger in pixels than the filters were tuned for).
  * v2-now melts early at BOTH sizes -> v2's training is wearing down the recovery skill.
Melt = the 1 s average sharpness drops below half of real video's, or turns to static
(train_v1.melt_report, the same meter the training runs use).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))
from data import to_model_space, to_uint8_image          # noqa: E402
from data_v2 import load_cache_any                       # noqa: E402
from model_v1 import build_model_v1                      # noqa: E402
from model_v2 import build_model_v2                      # noqa: E402
from train_v1 import melt_report, sharpness              # noqa: E402

FRAMES = 150                       # 15 s walks
SCRIPT = [1] * 40 + [6] * 15 + [1] * 40 + [5] * 30 + [1] * 25
STEPS = 10


def load(path, builder):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    m = builder(ck["model_cfg"]).cuda().eval()
    m.load_state_dict(ck["ema"])
    return m, ck["step"]


@torch.no_grad()
def dream_batch(model, frames_mm, actions_np, starts, seed=7):
    """All walks in one batch. Returns uint8 [walks, T, H, W, 3]."""
    K = model.K
    ctx = torch.stack([to_model_space(torch.from_numpy(np.array(frames_mm[s - K:s])).cuda()) for s in starts])
    hist = [[int(a) if a >= 0 else 1 for a in actions_np[s - K:s - 1]] for s in starts]
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for a in SCRIPT[:FRAMES]:
            act = torch.tensor([h[-(K - 1):] + [a] for h in hist], device="cuda")
            nxt = model.sample(ctx, act, steps=STEPS, generator=g).float()
            out.append(np.stack([to_uint8_image(f) for f in nxt]))
            ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)
            for h in hist:
                h.append(a)
    return np.stack(out, 1)


def main():
    v1, s1 = load(HERE / "checkpoints" / "dream_v1" / "latest.pt", build_model_v1)
    v2, s2 = load(HERE / "checkpoints" / "dream_v2" / "latest.pt", build_model_v2)
    print(f"v1 step {s1}, v2 step {s2}")
    caches = {"64x112": DW / "data" / "cache", "96x168": DW / "data" / "cache_v2"}
    rows = []
    for size, cdir in caches.items():
        index, frames_mm, actions_np, _ = load_cache_any(cdir)
        starts = [c["end"] - int((c["end"] - c["start"]) * 0.03) + 40 for c in index["cities"]]
        real = np.concatenate([np.array(frames_mm[s:s + 150]) for s in starts])
        ref = float(sharpness(real).mean())
        for name, m in [("v1 final", v1), ("v2 now", v2)]:
            dreams = dream_batch(m, frames_mm, actions_np, starts)
            melts = []
            for i, dr in enumerate(dreams):
                rep = melt_report(dr, ref, np.array(frames_mm[starts[i] - m.K:starts[i]]))
                rep = rep[0] if isinstance(rep, tuple) else rep   # melt_report returns (report, curves)
                melts.append(rep["melt_s"] if rep["melt_s"] is not None else f">{FRAMES // 10}")
            rows.append((name, size, melts))
            print(f"{name:<9} at {size}: melt per walk (s) {melts}")
    out = HERE / "dial" / "scale_test.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Scale test (v1 step {s1}, v2 step {s2}, {FRAMES // 10} s walks, 3 cities)\n\n")
        f.write("| weights | frame size | melt per walk (s) |\n|---|---|---|\n")
        for name, size, melts in rows:
            f.write(f"| {name} | {size} | {', '.join(map(str, melts))} |\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
