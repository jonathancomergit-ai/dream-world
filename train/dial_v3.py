"""Test A: does HOW v3 dreams (not how it trained) cause the texture collapse? (2026-10-19)

    data\\venv\\Scripts\\python.exe train\\dial_v3.py

v3 trained on BLURRY memories (context noise, per frame, often heavy) but its ruler and play
window dream from CLEAN memories. That mismatch could push it toward pole/ripple texture.
This dreams the same fair-ruler walks (4 walks x 2 seeds x 20 s) with v3's best weights under
different dreaming settings and scores each with the strict clean2 meter (no melt, no
curtains, no ripples). Same weights, same walks, same noise: only the settings change.

  cn     context noise at dream time: smudge every memory frame by this much and TELL the model
  steps  clean-up steps per frame
  heun   the more careful 2-pass clean-up
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import functools
import pathlib
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))
from data_v3 import load_cache_any                                          # noqa: E402
from model_v3 import build_model_v3                                         # noqa: E402
from train_v3 import FAIR_SCRIPT, dream_walks, ruler_line, score_walks      # noqa: E402

CONFIGS = [
    dict(name="ruler (10 steps, clean memory)", steps=10, cn=0.0,  heun=False),
    dict(name="cn 0.05",                         steps=10, cn=0.05, heun=False),
    dict(name="cn 0.1",                          steps=10, cn=0.1,  heun=False),
    dict(name="cn 0.2",                          steps=10, cn=0.2,  heun=False),
    dict(name="cn 0.35",                         steps=10, cn=0.35, heun=False),
    dict(name="20 steps",                        steps=20, cn=0.0,  heun=False),
    dict(name="20 steps + cn 0.1",               steps=20, cn=0.1,  heun=False),
    dict(name="heun 6",                          steps=6,  cn=0.0,  heun=True),
    dict(name="heun 6 + cn 0.1",                 steps=6,  cn=0.1,  heun=True),
]


def main():
    ck = torch.load(HERE / "checkpoints" / "v3_best_for_tests.pt", map_location="cuda", weights_only=False)
    model = build_model_v3(ck["model_cfg"]).cuda().eval()
    model.load_state_dict(ck["ema"])
    starts, step, K = list(ck["ruler_starts"]), ck["step"], model.K
    del ck
    index, frames_mm, actions_np, _ = load_cache_any(DW / "data" / "cache_v3")
    T = sum(n for _, n in FAIR_SCRIPT)
    keys = [a for a, n in FAIR_SCRIPT for _ in range(n)]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    plain = model.sample
    rows = []
    for c in CONFIGS:
        # dream_walks calls model.sample(...) with the ruler's settings; wrap it to add ours
        model.sample = functools.partial(plain, sigma_cond=c["cn"], heun=c["heun"])
        t0 = time.time()
        dreams = dream_walks(model, frames_mm, actions_np, starts, 2, keys, c["steps"], "cuda")
        summ, _ = score_walks(dreams, [x for _ in range(2) for x in starts], frames_mm, K, T)
        print(ruler_line(f"{c['name']} ({time.time() - t0:.0f}s)", summ, T), flush=True)
        rows.append((c["name"], summ))
        np.save(HERE / "_exp" / f"dialv3_{c['name'].split(' (')[0].replace(' ', '_').replace('+', '')}.npy", dreams[0])
    model.sample = plain
    out = HERE / "_exp" / "dial_v3.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Test A: dream dial on v3 best (step {step}), fair ruler, 8 dreams x 20 s\n\n")
        f.write("| setting | melt med/mean | clean2 med/mean | ripply |\n|---|---|---|---|\n")
        for n, s in rows:
            f.write(f"| {n} | {s['melt_median']} / {s['melt_mean']} | {s['clean2_median']} / {s['clean2_mean']} "
                    f"| {s['streak']['ripply_share']:.0%} |\n")
    print(open(out, encoding="utf-8").read())


if __name__ == "__main__":
    (HERE / "_exp").mkdir(exist_ok=True)
    main()
