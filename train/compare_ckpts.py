"""Fair ruler: compare saves of one run on several held-out walks and seeds (2026-10-19).

    data\venv\Scripts\python.exe train\compare_ckpts.py train\checkpoints\dream_v2\step0025225.pt train\checkpoints\dream_v2\latest.pt

The training runs score melt time on ONE fixed walk, which is noisy (scale_test.py showed v1
scoring 10-30 s on it but 1.6-6.9 s on others). This scores every save on 3 walks (one per city)
x 2 random seeds = 6 dreams of 15 s, using the same melt meter, and prints the median.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_v2 import load_cache_any                           # noqa: E402
from model_v2 import build_model_v2                          # noqa: E402
from scale_test import FRAMES, dream_batch                   # noqa: E402
from train_v1 import melt_report, sharpness                  # noqa: E402

cache = HERE.parent / "data" / "cache_v2"
index, frames_mm, actions_np, _ = load_cache_any(cache)
starts = [c["end"] - int((c["end"] - c["start"]) * 0.03) + 40 for c in index["cities"]]
ref = float(sharpness(np.concatenate([np.array(frames_mm[s:s + 150]) for s in starts])).mean())
for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cuda", weights_only=False)
    m = build_model_v2(ck["model_cfg"]).cuda().eval()
    m.load_state_dict(ck["ema"])
    melts = []
    for seed in (7, 8):
        for i, dr in enumerate(dream_batch(m, frames_mm, actions_np, starts, seed=seed)):
            rep = melt_report(dr, ref, np.array(frames_mm[starts[i] - m.K:starts[i]]))
            rep = rep[0] if isinstance(rep, tuple) else rep
            melts.append(rep["melt_s"] if rep["melt_s"] is not None else FRAMES / 10)
    print(f"step {ck['step']:>6}: melt per dream {melts}  median {np.median(melts):.1f} s  mean {np.mean(melts):.1f} s")
    del m
    torch.cuda.empty_cache()
