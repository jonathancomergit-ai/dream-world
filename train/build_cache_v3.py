"""Build the v3 training cache: EVERY source that is ready, at 96 x 168, into data/cache_v3/.

    data\\venv\\Scripts\\python.exe train\\build_cache_v3.py        # -> data/cache_v3/

WHICH VIDEOS GO IN
  Every <Name>.mp4 in data/clean that ALSO has data/actions/<Name>.npz (the key labels). A video
  without its labels is skipped with a note (v2's builder cached it as unusable instead; for v3
  there's no point storing frames we can't train on). Half-finished files (*.tmp*, *_small*) are
  skipped too. So when new cities (Bangkok, Istanbul, ...) or the Wildlife footage come out of
  prep/prep_videos.py + prep/label_actions.py, just re-run this: it rebuilds the whole cache
  from scratch with everything that's ready.
  !! Don't re-run it while a v3 training run is reading data/cache_v3 (stop training first, then
     rebuild, then resume). The fair ruler remembers which cities it walks in, so a resumed run
     keeps scoring on the same walks even after new cities arrive.

WHAT IT MAKES: the same layout as cache_v2 (see build_cache_v2.py), so the same loaders work:
  frames_96x168.u8, index.json (with "frame_file" and per-source "hours"), actions.npy, cut.npy,
  valid.npy (for the v3 window length: 52 + 1 frames by default).

At the end it compares every source that is also in data/cache_v2 (same frame count, same keys),
so you know frame i of a city in v3 is the same moment as in v2 (the v2 baseline relies on it).
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))
from build_cache import compute_valid, find_videos, load_actions   # noqa: E402  (shared, unchanged)
from build_cache_v2 import FRAME_FILE, H, W, decode_into           # noqa: E402  (96x168 decoder)


def compare_with_v2(index, actions, v2_dir):
    p = pathlib.Path(v2_dir) / "index.json"
    if not p.exists():
        print(f"  (no cache_v2 at {v2_dir}: skipping the comparison)")
        return
    v2 = json.loads(p.read_text(encoding="utf-8"))
    a2 = np.load(pathlib.Path(v2_dir) / "actions.npy")
    by_name = {c["name"]: c for c in v2["cities"]}
    for c in index["cities"]:
        o = by_name.get(c["name"])
        if o is None:
            print(f"  {c['name']}: new (not in cache_v2)")
            continue
        same_n = o["frames"] == c["frames"]
        same_a = same_n and np.array_equal(a2[o["start"]:o["end"]], actions[c["start"]:c["end"]])
        print(f"  {c['name']}: {'same frames + keys as cache_v2' if same_a else '!! DIFFERS from cache_v2'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DW / "data" / "clean"), help="folder of <Name>.mp4")
    ap.add_argument("--actions", default=str(DW / "data" / "actions"), help="folder of <Name>.npz")
    ap.add_argument("--out", default=str(DW / "data" / "cache_v3"))
    ap.add_argument("--v2-cache", default=str(DW / "data" / "cache_v2"), help="only for the comparison")
    ap.add_argument("--L", type=int, default=52,
                    help="window length for valid.npy (v3: 48 history + 4 recovery-practice frames; "
                         "train_v3.py recomputes it itself anyway)")
    ap.add_argument("--threads", type=int, default=4, help="OpenCV threads")
    args = ap.parse_args()
    import cv2
    cv2.setNumThreads(args.threads)

    t_start = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = []
    for v in find_videos(args.clean):
        if (pathlib.Path(args.actions) / f"{v.stem}.npz").exists():
            vids.append(v)
        else:
            print(f"  skipping {v.name}: no actions file yet (run prep/label_actions.py on it first)")
    if not vids:
        sys.exit(f"No <Name>.mp4 in {args.clean} with a matching .npz in {args.actions}")
    print(f"Building the {H}x{W} v3 cache from {len(vids)} videos -> {out}")

    raw_tmp = out / (FRAME_FILE + ".tmp")
    cities, all_a, all_c = [], [], []
    offset = 0
    with open(raw_tmp, "wb") as f:
        for v in vids:
            t0 = time.time()
            print(f"  decoding {v.name} ...", flush=True)
            n = decode_into(v, f)
            a, c, ok = load_actions(args.actions, v.stem, n)
            took = time.time() - t0
            cities.append({"name": v.stem, "start": offset, "end": offset + n, "frames": n,
                           "has_actions": ok, "hours": round(n / 36000, 3)})
            all_a.append(a)
            all_c.append(c)
            offset += n
            known = a[a >= 0]
            still = float((known == 0).mean()) if len(known) else 0.0
            print(f"  {v.stem}: {n} frames ({n / 36000:.2f} h) in {took:.0f} s "
                  f"({took / max(n / 36000, 1e-6) / 60:.1f} min per hour of video); "
                  f"keys: {still:.0%} still, {float((known == 1).mean()) if len(known) else 0:.0%} W",
                  flush=True)

    total = offset
    os.replace(raw_tmp, out / FRAME_FILE)          # only now does the real name exist
    actions = np.concatenate(all_a)
    cut = np.concatenate(all_c)
    valid = compute_valid(actions, cut, cities, args.L)
    np.save(out / "actions.npy", actions)
    np.save(out / "cut.npy", cut)
    np.save(out / "valid.npy", valid)
    for c in cities:
        c["valid_windows"] = int(valid[c["start"]:c["end"]].sum())
    index = {"shape": [total, H, W, 3], "dtype": "uint8", "frame_file": FRAME_FILE, "K": args.L,
             "cities": cities, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    compare_with_v2(index, actions, args.v2_cache)
    took = time.time() - t_start
    hours = total / 36000
    size = (out / FRAME_FILE).stat().st_size
    print(f"\nDone in {took / 60:.1f} min ({took / 60 / max(hours, 1e-6):.1f} min per hour of video): "
          f"{len(cities)} sources, {total} frames = {hours:.2f} h, {size / 1e9:.2f} GB, "
          f"{int(valid.sum())} valid windows (L={args.L}).")


if __name__ == "__main__":
    main()
