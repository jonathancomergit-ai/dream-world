"""Dream-style damage for the upscaler's inputs: every HR frame's 96x168 cache frame, WRECKED the way train_v9 does
(train_v9.resample_memory with n_slots = 1): real frame + noise sigma -> ONE clean-up call of the v9 pick model
(train/checkpoints/dream_v9/v9_pick.pt, EMA weights, via train_v7.load_as_v7) with that frame's real memory, keys
and motion. The result is "what the dream model makes of this moment after one noisy step" - its own kind of mush -
while the real frame (and so the real HR frame) is known.

    data\\venv\\Scripts\\python.exe train\\sr\\wreck_hr.py [--batch 32]     # -> data/cache_hr/wreck_96x168.u8

sigma per frame: log-uniform in [0.1, 1.0] (seeded). Frames whose 172-frame memory window would leave the city or
cross a scene cut get no wreck (sigma NaN in wreck_sigma.npy; their wreck slot holds the clean frame).
GPU: one model (~100M params) in bf16 autocast, batch 32 -> a few GB; capped at 12 GB.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = pathlib.Path(__file__).resolve().parent
DW = SR.parent.parent
sys.path.insert(0, str(SR.parent))

from data_v3 import load_cache_any, to_model_space                      # noqa: E402
from data_v6 import position_lookup                                     # noqa: E402
from data_v7 import SparseWindowDatasetV7                               # noqa: E402
from model_v3 import colour_fingerprint                                 # noqa: E402
from train_v7 import MemIndexV7, load_as_v7                             # noqa: E402
from train_v9 import OneCall, resample_memory, window_layout            # noqa: E402

CACHE = DW / "data" / "cache_v3"
OUT = DW / "data" / "cache_hr"


class Items(torch.utils.data.Dataset):
    def __init__(self, positions, L, starts):
        self.positions, self.L, self.starts, self.ds = positions, L, starts, None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self.ds is None:
            self.ds = SparseWindowDatasetV7(CACHE, self.positions, self.L, flip_p=0.0)
        fr, act, t, mot = self.ds[int(self.starts[i])]
        return fr, act.clamp(min=0) + (act < 0).long(), mot, i      # unknown keys (-1) -> W (1), like dream_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DW / "train" / "checkpoints" / "dream_v9" / "v9_pick.pt"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sig-lo", type=float, default=0.1)
    ap.add_argument("--sig-hi", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (testing)")
    ap.add_argument("--vram-cap-gb", type=float, default=12.0)
    args = ap.parse_args()
    dev = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    index, fm, act_all, cut = load_cache_any(CACHE)
    cidx = np.load(OUT / "cache_idx.npy")
    N = len(cidx)
    model, step, ver = load_as_v7(pathlib.Path(args.ckpt), dev, which="ema")
    sf_w = [0.4, 0.3, 0.2, 0.1]                                          # train_v9 default --sf-weights
    SF_MAX, L, fp_train, positions, _ = window_layout(model, sf_w, 1)
    lut = position_lookup(positions, L)
    mi = MemIndexV7(model, lut, dev)
    fp_rows = lut[torch.tensor([L - o for o in fp_train])].to(dev)
    call = OneCall(model)
    r1 = mi.row(L - 1)
    print(f"{args.ckpt}: {ver} step {step}; window L={L}, {len(positions)} read positions; newest slot row {r1}", flush=True)

    # which frames can be wrecked: window frames (c + 1 - L) .. (c + 1) inside the city, no cut in between
    city_of = np.zeros(len(fm), np.int32)
    for k, c in enumerate(index["cities"]):
        city_of[c["start"]:c["end"]] = k
    ccut = np.concatenate([[0], np.cumsum(cut.astype(np.int64))])
    ok = np.zeros(N, bool)
    for i, c in enumerate(cidx):
        t0, s = c + 1 - L, c + 1
        cy = index["cities"][city_of[c]]
        ok[i] = t0 >= cy["start"] and s < cy["end"] and (ccut[s] - ccut[t0]) == 0
    todo = np.flatnonzero(ok)
    if args.limit:
        todo = todo[:args.limit]
    rng = np.random.default_rng(1234)
    sig_all = np.full(N, np.nan, np.float32)
    sig_all[todo] = np.exp(rng.uniform(math.log(args.sig_lo), math.log(args.sig_hi), len(todo))).astype(np.float32)
    print(f"{ok.sum()} of {N} frames can be wrecked ({N - ok.sum()} skipped: window leaves the city or crosses a cut)",
          flush=True)

    path = OUT / "wreck_96x168.u8"
    mm = np.memmap(path, dtype=np.uint8, mode="r+" if path.exists() else "w+", shape=(N, 96, 168, 3))
    for i in np.flatnonzero(~ok):                                        # no wreck -> the clean frame
        mm[i] = fm[cidx[i]]
    dl = torch.utils.data.DataLoader(Items(positions, L, cidx[todo] + 1 - L), batch_size=args.batch, num_workers=2,
                                     shuffle=False, persistent_workers=False)
    t0 = time.time()
    done = 0
    gen = torch.Generator(device=dev).manual_seed(0)
    mad = []
    for fr, act, mot, j in dl:
        rows = todo[j.numpy()]
        fr = to_model_space(fr.to(dev))
        act, mot = act.to(dev), mot.to(dev)
        colour = colour_fingerprint(fr.index_select(1, fp_rows))
        sig = torch.from_numpy(sig_all[rows]).to(dev)
        real = fr[:, r1].clone()
        resample_memory(call, mi, fr, act, mot, colour, 0, len(rows), 1, sig, L, generator=gen)
        w = ((fr[:, r1].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        chk = ((real.float() + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        assert np.array_equal(chk, np.stack([fm[cidx[r]] for r in rows])), "window row != the HR frame's cache frame"
        for k, r in enumerate(rows):
            mm[r] = w[k]
        mad.append(np.abs(w.astype(np.float32) - chk.astype(np.float32)).mean((1, 2, 3)))
        done += len(rows)
        if done % (args.batch * 100) < args.batch:
            el = time.time() - t0
            print(f"  {done}/{len(todo)} ({done / el:.0f} frames/s), GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)
    mm.flush()
    np.save(OUT / "wreck_sigma.npy", sig_all)
    mad = np.concatenate(mad)
    s = sig_all[todo]
    info = {"ckpt": args.ckpt, "step": step, "frames": int(len(todo)), "skipped": int(N - ok.sum()),
            "sigma": [args.sig_lo, args.sig_hi], "secs": round(time.time() - t0),
            "mean_abs_diff_255_by_sigma": {f"{a:.2f}-{b:.2f}": round(float(mad[(s >= a) & (s < b)].mean()), 2)
                                           for a, b in [(0.1, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.01)]}}
    (OUT / "wreck_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("DONE", info, flush=True)


if __name__ == "__main__":
    main()
