"""Build the v2 training cache: the same frames as data/cache, but at 96 x 168 instead of 64 x 112.

    python train/build_cache_v2.py                  # real data -> data/cache_v2/  (~6.3 GB, a few minutes)

WHY A SECOND CACHE
  v2 dreams at a higher resolution. The trainer reads frames straight out of one big raw file on
  disk (a "memmap"), so the frames have to be stored at the size the model eats. Rather than
  touch data/cache (the live v1 run reads it all day), v2 gets its own folder next to it.

WHY 96 x 168
  - 168 / 96 = 1.75, the same ~16:9 shape as 64 x 112 and as the 256 x 144 source videos.
  - The U-Net halves the picture 3 times: 96x168 -> 48x84 -> 24x42 -> 12x21. Every step divides
    evenly (no half pixels), so going down and coming back up gives exactly 96 x 168 again.
  - It is 1.5x the old size on each side = 2.25x the pixels (and roughly 2.25x the GPU work).

WHAT IT MAKES (in --out), the same layout as build_cache.py makes, so the same rules apply:
  frames_96x168.u8   raw uint8 RGB pixels, shape [total_frames, 96, 168, 3], no header.
  index.json         where each city starts/ends, the shape, AND (new) "frame_file", the name of
                     the frames file, so the loader doesn't have to guess it from the size.
  actions.npy        int8 [total_frames]  the key that moved frame i -> i+1 (-1 = unknown)
  cut.npy            bool [total_frames]  True where i -> i+1 is a scene cut
  valid.npy          bool [total_frames]  True if a window starting at t is safe (see compute_valid)

FRAME COUNT CHECK
  Frame i of cache_v2 must be the same moment as frame i of data/cache (v1), otherwise the
  v1-vs-v2 comparisons and the held-out "test walk" would point at different moments. So at the
  end this script compares every city's start/end and the actions/cut arrays with data/cache and
  stops with an error if anything differs.

Everything that isn't about the frame size (finding the videos, reading the key labels, the
valid-window rule) is imported from build_cache.py unchanged, so there's one copy of it.
Safe to re-run (rebuilds from scratch), but NOT while a v2 training run is reading --out.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))
from build_cache import compute_valid, find_videos, load_actions   # noqa: E402  (shared, unchanged)

H, W = 96, 168                                  # v2 cache frame size (height, width)
FRAME_FILE = f"frames_{H}x{W}.u8"


def decode_into(path, fout):
    """Decode one mp4 frame by frame, shrink 256x144 -> 168x96 and append the raw bytes to fout.
    INTER_AREA averages the source pixels that get merged into each new pixel: the right way to
    shrink (the same method build_cache.py used, so v1 and v2 frames look alike apart from size).
    Returns how many frames were written."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    n = 0
    t0 = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)   # cv2 sizes are (width, height)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        fout.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        n += 1
        if n % 10000 == 0:
            print(f"    {path.stem}: {n} frames ({n / (time.time() - t0):.0f} fps)", flush=True)
    cap.release()
    return n


def check_against_v1(index, actions, cut, v1_dir):
    """Stop with an error if this cache's frames don't line up 1:1 with the v1 cache."""
    p = pathlib.Path(v1_dir) / "index.json"
    if not p.exists():
        print(f"  (no v1 cache at {v1_dir}, skipping the frame-count check)")
        return
    v1 = json.loads(p.read_text(encoding="utf-8"))
    problems = []
    if v1["shape"][0] != index["shape"][0]:
        problems.append(f"total frames: v1 {v1['shape'][0]} vs v2 {index['shape'][0]}")
    a = [(c["name"], c["start"], c["end"]) for c in v1["cities"]]
    b = [(c["name"], c["start"], c["end"]) for c in index["cities"]]
    if a != b:
        problems.append(f"cities differ: v1 {a} vs v2 {b}")
    if not problems:
        if not np.array_equal(np.load(pathlib.Path(v1_dir) / "actions.npy"), actions):
            problems.append("actions.npy differs from v1's")
        if not np.array_equal(np.load(pathlib.Path(v1_dir) / "cut.npy"), cut):
            problems.append("cut.npy differs from v1's")
    if problems:
        sys.exit("!!! v2 cache does NOT line up with the v1 cache:\n  " + "\n  ".join(problems))
    print(f"  frame check OK: {index['shape'][0]} frames, same cities/starts/ends/actions/cuts as {v1_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DW / "data" / "clean"), help="folder of <City>.mp4")
    ap.add_argument("--actions", default=str(DW / "data" / "actions"), help="folder of <City>.npz")
    ap.add_argument("--out", default=str(DW / "data" / "cache_v2"))
    ap.add_argument("--v1-cache", default=str(DW / "data" / "cache"), help="for the frame-count check")
    ap.add_argument("--K", type=int, default=10,
                    help="window length for valid.npy (v2 trains on K=8 memory + 2 extra frames = 10; "
                         "train_v2.py recomputes this itself anyway)")
    ap.add_argument("--threads", type=int, default=2,
                    help="OpenCV threads. Kept low so the running v1 training's data loaders aren't starved.")
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)

    t_start = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = find_videos(args.clean)
    if not vids:
        sys.exit(f"No finished <City>.mp4 in {args.clean}")
    print(f"Building the {H}x{W} cache from {len(vids)} videos -> {out}")

    raw_tmp = out / (FRAME_FILE + ".tmp")
    cities, all_a, all_c = [], [], []
    offset = 0
    with open(raw_tmp, "wb") as f:
        for v in vids:
            city = v.stem
            print(f"  decoding {v.name} ...", flush=True)
            n = decode_into(v, f)
            a, c, ok = load_actions(args.actions, city, n)
            if not ok:
                print(f"  !!! WARNING: no actions file for {city}; its {n} frames are cached but INVALID.")
            cities.append({"name": city, "start": offset, "end": offset + n, "frames": n, "has_actions": ok})
            all_a.append(a)
            all_c.append(c)
            offset += n
            print(f"  {city}: {n} frames, actions {'OK' if ok else 'MISSING'}", flush=True)

    total = offset
    os.replace(raw_tmp, out / FRAME_FILE)          # only now does the real name exist
    actions = np.concatenate(all_a)
    cut = np.concatenate(all_c)
    valid = compute_valid(actions, cut, cities, args.K)
    np.save(out / "actions.npy", actions)
    np.save(out / "cut.npy", cut)
    np.save(out / "valid.npy", valid)
    for c in cities:
        c["valid_windows"] = int(valid[c["start"]:c["end"]].sum())
    index = {"shape": [total, H, W, 3], "dtype": "uint8", "frame_file": FRAME_FILE, "K": args.K,
             "cities": cities, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    check_against_v1(index, actions, cut, args.v1_cache)
    took = time.time() - t_start
    size = (out / FRAME_FILE).stat().st_size
    print(f"\nDone in {took / 60:.1f} min: {total} frames, {size / 1e9:.2f} GB, "
          f"{int(valid.sum())} valid windows (K={args.K}).")


if __name__ == "__main__":
    main()
