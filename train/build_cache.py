"""Build the training cache: every city's frames in ONE big file the trainer can read instantly.

    python train/build_cache.py                       # real data -> data/cache/
    python train/build_cache.py --clean train/_smoke/clean --actions train/_smoke/actions --out train/_smoke/cache

What it makes (in --out):
  frames_64x112.u8   raw uint8 pixels, shape [total_frames, 64, 112, 3] (RGB). No header: it's a
                     "memmap", so the trainer can treat a 3 GB file like an array without loading it.
  index.json         where each city starts/ends inside that big array, plus the shape.
  actions.npy        int8 [total_frames]: actions[i] = the key press that moved frame i -> i+1
                     (-1 = unknown, e.g. the city's actions file hasn't arrived yet).
  cut.npy            bool [total_frames]: True where i -> i+1 is a scene cut (never train across it).
  valid.npy          bool [total_frames]: True if a training window that STARTS at frame t is safe:
                     its K+1 frames stay inside one city, every action is known, and no cut is crossed.

Why 64 x 112? It keeps the videos' ~16:9 shape, and both numbers divide by 16, so the U-Net can
halve the picture 3 times (64x112 -> 32x56 -> 16x28 -> 8x14) without odd sizes.

Safe to re-run: it rebuilds everything from scratch. DON'T re-run it while a training run is
reading the same --out folder (stop training, rebuild, then --resume).
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

DW = pathlib.Path(__file__).resolve().parent.parent   # the DreamWorld folder
H, W = 64, 112                                         # cache frame size (height, width)
K_DEFAULT = 4                                          # context frames the model sees


def compute_valid(actions, cut, cities, K):
    """valid[t] is True when the window of frames t .. t+K (K context frames + 1 target)
    is safe to train on.

    The window uses the K actions actions[t] .. actions[t+K-1]:
      - actions[t .. t+K-2] are the moves BETWEEN the context frames,
      - actions[t+K-1] is the move that produces the target frame (the key you "pressed").
    So we need all K of them known (>= 0), none of them a cut, and all K+1 frames in one city.
    Shared with train.py so the rule lives in exactly one place.
    """
    total = len(actions)
    valid = np.zeros(total, dtype=bool)
    bad = (actions < 0) | cut                     # transitions we must not train across
    for c in cities:
        s, e = c["start"], c["end"]               # frames s .. e-1 belong to this city
        n_windows = (e - s) - K                   # last start t needs t+K <= e-1
        if n_windows <= 0:
            continue
        b = bad[s:e - 1].astype(np.int32)         # transitions s->s+1 ... e-2->e-1
        # window starting at local i uses transitions i .. i+K-1: it's OK if their sum is 0
        cs = np.concatenate([[0], np.cumsum(b)])
        win_bad = cs[K:K + n_windows] - cs[0:n_windows]
        valid[s:s + n_windows] = win_bad == 0
    return valid


def find_videos(clean_dir):
    vids = []
    for p in sorted(pathlib.Path(clean_dir).glob("*.mp4")):
        name = p.name
        if "_small" in name or ".tmp" in name:     # half-finished or side outputs: skip
            continue
        vids.append(p)
    return vids


def decode_into(path, fout):
    """Decode one mp4 frame by frame, shrink to 64x112 and append raw bytes to fout.
    Returns the number of frames written. INTER_AREA = average the pixels being merged,
    the right choice for shrinking (no shimmering/aliasing)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    n = 0
    t0 = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        fout.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        n += 1
        if n % 5000 == 0:
            print(f"    {path.stem}: {n} frames ({n / (time.time() - t0):.0f} fps)", flush=True)
    cap.release()
    return n


def load_actions(actions_dir, city, n_frames):
    """Return (actions int8 [n], cut bool [n], ok flag). Missing file -> all -1, ok=False."""
    p = pathlib.Path(actions_dir) / f"{city}.npz"
    if not p.exists():
        return np.full(n_frames, -1, np.int8), np.zeros(n_frames, bool), False
    z = np.load(p)
    a = z["actions"].astype(np.int8)
    c = z["cut"].astype(bool) if "cut" in z.files else np.zeros(len(a), bool)
    if len(a) != n_frames:
        # Off-by-a-few is normal (decoders disagree on the last frame). Pad unknown with -1.
        print(f"  ! {city}: actions has {len(a)} entries but video has {n_frames} frames; "
              f"{'truncating' if len(a) > n_frames else 'padding the end with -1'}")
    out_a = np.full(n_frames, -1, np.int8)
    out_c = np.zeros(n_frames, bool)
    m = min(len(a), n_frames)
    out_a[:m] = a[:m]
    out_c[:m] = c[:m]
    out_a[n_frames - 1] = -1          # the last frame has no "next frame", so no action
    bad = (out_a[:m] < -1) | (out_a[:m] > 8)
    if bad.any():
        print(f"  ! {city}: {bad.sum()} actions outside 0..8, marking them unknown (-1)")
        out_a[:m][bad] = -1
    return out_a, out_c, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DW / "data" / "clean"), help="folder of <City>.mp4")
    ap.add_argument("--actions", default=str(DW / "data" / "actions"), help="folder of <City>.npz")
    ap.add_argument("--out", default=str(DW / "data" / "cache"))
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = find_videos(args.clean)
    if not vids:
        sys.exit(f"No finished <City>.mp4 in {args.clean}")
    print(f"Building cache from {len(vids)} videos -> {out}")

    raw_tmp = out / "frames_64x112.u8.tmp"
    cities, all_a, all_c = [], [], []
    offset = 0
    with open(raw_tmp, "wb") as f:
        for v in vids:
            city = v.stem
            print(f"  decoding {v.name} ...", flush=True)
            n = decode_into(v, f)
            a, c, ok = load_actions(args.actions, city, n)
            if not ok:
                print(f"  !!! WARNING: no actions file for {city} ({pathlib.Path(args.actions) / (city + '.npz')})."
                      f" Its {n} frames are cached but marked INVALID (not trained on)."
                      f" Re-run this script once the actions arrive.")
            cities.append({"name": city, "start": offset, "end": offset + n, "frames": n,
                           "has_actions": ok})
            all_a.append(a)
            all_c.append(c)
            offset += n
            print(f"  {city}: {n} frames, actions {'OK' if ok else 'MISSING'}", flush=True)

    total = offset
    os.replace(raw_tmp, out / "frames_64x112.u8")   # only now does the real name exist
    actions = np.concatenate(all_a)
    cut = np.concatenate(all_c)
    valid = compute_valid(actions, cut, cities, args.K)
    np.save(out / "actions.npy", actions)
    np.save(out / "cut.npy", cut)
    np.save(out / "valid.npy", valid)
    index = {"shape": [total, H, W, 3], "dtype": "uint8", "K": args.K, "cities": cities,
             "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    for c in cities:
        c["valid_windows"] = int(valid[c["start"]:c["end"]].sum())
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"\nDone: {total} frames ({total * H * W * 3 / 1e9:.2f} GB), "
          f"{int(valid.sum())} valid training windows (K={args.K}).")
    hist = np.bincount(actions[actions >= 0], minlength=9)
    names = ["still", "fwd", "back", "strafeL", "strafeR", "turnL", "turnR", "up", "down"]
    print("  action counts: " + ", ".join(f"{n}={h}" for n, h in zip(names, hist)))
    if valid.sum() == 0:
        print("  !!! WARNING: zero valid windows. train.py will refuse to start until actions exist.")


if __name__ == "__main__":
    main()
