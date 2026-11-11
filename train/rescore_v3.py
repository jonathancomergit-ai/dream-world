"""Re-score saved v3 checkpoints on the fair ruler, offline, with the v3.1 meters (clean2).

    data\\venv\\Scripts\\python.exe train\\rescore_v3.py train\\checkpoints\\dream_v3\\best.pt ^
        train\\checkpoints\\dream_v3\\step0013406.pt train\\checkpoints\\dream_v3\\step0023524.pt

WHY: v3's best.pt is picked by "clean time" (no melt, no curtains). But when v3 loses the scene it
can fall back to rippling-water texture, which neither meter catches (see RIPPLE_LEVEL in
train_v3.py). So we dream the SAME fair-ruler walks again with each save and score them with the
new "clean2" number too, to see where the real peak was.

SAFETY (a run may be training while this runs):
  - every checkpoint is COPIED to a scratch folder first and loaded from the copy, so we never
    hold the training's own files open (on Windows an open file can block its save);
  - latest.pt is refused unless you pass --allow-latest (the trainer rewrites it every 20 min);
  - only the EMA weights go on the GPU (~0.2 GB) and the 8 dreams run as one small batch.

It reuses train_v3's own ruler code (dream_walks + score_walks), the walks saved inside the
checkpoint (ruler_starts) and the same fixed noise seed, so the melt / clean numbers should match
what run.log printed for that step (tiny differences can come from GPU rounding).
Also scores the REAL video of the same walks: a fair meter should give real video ~20 s.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import pathlib
import shutil

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))

from data_v3 import load_cache_any                                          # noqa: E402
from model_v3 import build_model_v3                                         # noqa: E402
from train_v3 import FAIR_SCRIPT, RIPPLE_LEVEL, dream_walks, ruler_line, score_walks  # noqa: E402


def load_copy(path, tmp_dir):
    """Copy a checkpoint to tmp_dir, load it from there (CPU), delete the copy. Returns the dict."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / ("_copy_" + path.name)
    shutil.copyfile(path, tmp)
    try:
        return torch.load(tmp, map_location="cpu", weights_only=False)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", help="v3 checkpoint files to re-score")
    ap.add_argument("--cache", default=str(DW / "data" / "cache_v3"))
    ap.add_argument("--tmp", default=str(HERE / "_smoke" / "rescore_tmp"), help="scratch folder for the copies")
    ap.add_argument("--ripple-max", type=float, default=RIPPLE_LEVEL)
    ap.add_argument("--seeds", type=int, default=2, help="noise draws per walk (the live ruler uses 2)")
    ap.add_argument("--steps", type=int, default=10, help="sampler steps (the live ruler uses 10)")
    ap.add_argument("--start", choices=["noise", "prev"], default="prev", help="dream start (prev = the 2026-10-20 fix)")
    ap.add_argument("--save-dreams", default="", help="folder to save each save's dreams (.npy) for later looks")
    ap.add_argument("--allow-latest", action="store_true", help="allow latest.pt (only when nothing is training)")
    args = ap.parse_args()

    index, frames_mm, actions_np, _ = load_cache_any(args.cache)
    T = sum(n for _, n in FAIR_SCRIPT)
    keys = [a for a, n in FAIR_SCRIPT for _ in range(n)]
    torch.backends.cuda.matmul.allow_tf32 = True      # same settings as the training run's ruler
    torch.backends.cudnn.allow_tf32 = True
    rows, starts = [], None
    for p in args.ckpts:
        p = pathlib.Path(p)
        if p.name == "latest.pt" and not args.allow_latest:
            sys.exit(f"refusing {p}: a running training rewrites latest.pt (use --allow-latest if it's stopped)")
        ck = load_copy(p, pathlib.Path(args.tmp))
        if ck["model_cfg"].get("version") != "v3":
            sys.exit(f"{p} is not a v3 checkpoint")
        s = list(ck["ruler_starts"])
        if starts is not None and s != starts:
            print(f"WARNING: {p.name} was scored on other walks {s} than the first save {starts}")
        starts = starts or s
        model = build_model_v3(ck["model_cfg"]).to("cuda").eval()
        model.load_state_dict(ck["ema"])
        step, K = ck["step"], model.K
        del ck
        dreams = dream_walks(model, frames_mm, actions_np, s, args.seeds, keys, args.steps, "cuda", start=args.start)
        row_starts = [x for _ in range(args.seeds) for x in s]
        summ, _ = score_walks(dreams, row_starts, frames_mm, K, T, args.ripple_max)
        print(ruler_line(f"{p.name} (step {step})", summ, T), flush=True)
        rows.append((f"{p.name} (step {step})", summ))
        if args.save_dreams:
            d = pathlib.Path(args.save_dreams)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"dreams_step{step:07d}.npy", dreams)
        del model
        torch.cuda.empty_cache()

    if starts is None:
        sys.exit("no checkpoints given")
    # the same walks in REAL video: the meters' "should never trigger" check
    real = np.stack([np.array(frames_mm[s:s + T]) for s in starts])
    rsumm, _ = score_walks(real, starts, frames_mm, 8, T, args.ripple_max)
    print(ruler_line("REAL video, same walks", rsumm, T))
    rows.append(("REAL video", rsumm))
    if args.save_dreams:
        np.save(pathlib.Path(args.save_dreams) / "real.npy", real)
        np.save(pathlib.Path(args.save_dreams) / "starts.npy", np.array(starts))

    print(f"\nwalks {starts}, {args.seeds} noise draws each, {T / 10:.0f} s, ripple level {args.ripple_max}")
    print(f"{'save':34s} {'melt med/mean':>14s} {'clean med/mean':>15s} {'clean2 med/mean':>16s} {'ripply':>7s}")
    for name, s in rows:
        print(f"{name:34s} {s['melt_median']:>6} /{s['melt_mean']:>6} {s['clean_median']:>7} /{s['clean_mean']:>6} "
              f"{s['clean2_median']:>8} /{s['clean2_mean']:>6} {s['streak']['ripply_share']:>7.0%}")


if __name__ == "__main__":
    main()
