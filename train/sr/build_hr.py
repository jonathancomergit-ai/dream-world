"""Build the 2x "sharper eyes" data: 192x336 frames from the ORIGINAL 1280x720 videos, face-blurred, lined up
pixel-for-pixel with data/cache_v3's 96x168 frames.

    data\\venv\\Scripts\\python.exe train\\sr\\build_hr.py [--workers 5]      # -> data/cache_hr/

ALIGNMENT (measured by train/sr/align_check.py, see README):
  cache_v3 frame (city start + n) = output frame n of ffmpeg's "fps=10" filter on data/videos/<City>.mp4 (the newest
  source frame with pts < (n + 0.5) / 10 s, i.e. ~n/10 + 0.033 s at 60 fps), the WHOLE 16:9 picture squashed to
  168x96 (no crop). So this script runs the very same "-hwaccel cuda ... fps=10" chain, keeps the wanted n with a
  select filter, and shrinks the full 1280x720 frame to 336x192 with INTER_AREA. A 2x2 mean of an HR frame is then
  the cache frame's view; every frame's mean |difference| to its cache frame (and to the NEXT cache frame, as a
  control) is stored in mad.npy.

FACES (project rule): the same detector + settings as prep/prep_videos.py (deface 1.5.0's bundled CenterFace,
  threshold 0.2, mask scale 1.3, ellipse, "blur" = deface's own anonymize_frame), but scaled to the higher
  resolution: detection runs on the 1280x720 frame at THREE input sizes (full 1280x720, 640x360 and prep's own
  256x144) and every box from any of them is blurred, at 1280x720, BEFORE shrinking. Nothing is downloaded.

WHICH FRAMES (walking cities only, Wildlife left out):
  train  every 13th frame of each city's training part (everything before its held-out tail, minus a 300-frame gap)
  val    every 8th frame of each held-out tail (last 3%, >= 608 frames: v9_judge / train_v8's rule), judge walks excluded
  test   ALL 400 frames (40 s) of the 10 judge walks (train/_exp/v9_judge.py: STD5 + extra_starts)

OUTPUT (data/cache_hr/): frames_192x336.u8 (uint8 memmap [N,192,336,3]), cache_idx.npy (int64 global cache_v3
  frame of each HR frame), split.npy (int8 0 train / 1 val / 2 test), walk.npy (int8 judge walk 0..9 or -1),
  faces.npy (int16 boxes blurred), mad.npy (float32 [N,2]: |2x2 mean(HR) - cache frame| same / next frame),
  index.json, logs/, done/<City>.json (restart-safe: finished cities are skipped).
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = pathlib.Path(__file__).resolve().parent
DW = SR.parent.parent
sys.path.insert(0, str(SR.parent))

VID = DW / "data" / "videos"
CACHE = DW / "data" / "cache_v3"
OUT = DW / "data" / "cache_hr"
H, W = 192, 336
SH, SW = 720, 1280
FRAME_FILE = f"frames_{H}x{W}.u8"
STRIDE_TRAIN, STRIDE_VAL, GAP = 13, 8, 300
STD5 = [47706, 234507, 517230, 151029, 441350]          # v9_judge.py
EXTRA_CITIES = ["Chiang_Mai", "Kuala_Lumpur", "Singapore", "Stockholm", "Bangkok"]
T_WALK, HN = 400, 168
THRESH, MASK_SCALE = 0.2, 1.3                           # prep_videos.py: deface --thresh 0.2 (mask scale default 1.3)
DETECT_SHAPES = [None, (640, 360), (256, 144)]          # None = full 1280x720


def hold_n(c):
    return min(max(int(0.03 * c["frames"]), HN + T_WALK + 40), c["frames"] // 3)


def judge_walks(index, cut):
    """v9_judge.py's 10 walks (global start frames), copied exactly."""
    out = []
    for name in EXTRA_CITIES:
        c = next(c for c in index["cities"] if c["name"] == name)
        h0 = c["end"] - hold_n(c)
        s = h0 + (1500 if name == "Bangkok" else 200)
        while s + T_WALK <= c["end"] and np.asarray(cut[s - HN - 2:s + 1]).any():
            s += 25
        assert s + T_WALK <= c["end"], name
        out.append(int(s))
    return STD5 + out


def plan(index, cut):
    """-> list of per-city dicts {name, start, ns (local), split, walk, expr}."""
    from data_v3 import NON_WALKING
    walks = judge_walks(index, cut)
    plans = []
    for c in index["cities"]:
        if c["name"] in NON_WALKING:
            continue
        n_all = c["frames"]
        h0 = n_all - hold_n(c)
        tr_end = h0 - GAP
        wl = [(w - c["start"], i) for i, w in enumerate(walks) if c["start"] <= w < c["end"]]
        for a, _ in wl:
            assert a >= h0, (c["name"], a, h0)
        walk_of = {}
        for a, i in wl:
            for n in range(a, a + T_WALK):
                walk_of[n] = i
        ns, split, walk = [], [], []
        for n in range(n_all):
            if n in walk_of:
                ns.append(n); split.append(2); walk.append(walk_of[n])
            elif n < tr_end and n % STRIDE_TRAIN == 0:
                ns.append(n); split.append(0); walk.append(-1)
            elif n >= h0 and n % STRIDE_VAL == 0:
                ns.append(n); split.append(1); walk.append(-1)
        terms = [f"(lt(n\\,{tr_end})*not(mod(n\\,{STRIDE_TRAIN})))", f"(gte(n\\,{h0})*not(mod(n\\,{STRIDE_VAL})))"]
        terms += [f"between(n\\,{a}\\,{a + T_WALK - 1})" for a, _ in wl]
        plans.append({"name": c["name"], "start": c["start"], "frames": n_all, "h0": h0, "tr_end": tr_end,
                      "ns": ns, "split": split, "walk": walk, "expr": "+".join(terms)})
    return plans, walks


def down2(hr):
    return hr.astype(np.float32).reshape(96, 2, 168, 2, 3).mean((1, 3))


def worker(job):
    """One city: ffmpeg (fps=10 + select) -> face blur at 1280x720 -> INTER_AREA 336x192 -> memmap slots."""
    import cv2
    from deface.centerface import CenterFace
    from deface.deface import anonymize_frame
    from data_v3 import load_cache_any
    cv2.setNumThreads(job["threads"])
    name, slot0, ns = job["name"], job["slot0"], job["ns"]
    N = job["N"]
    log = open(OUT / "logs" / f"{name}.txt", "a", encoding="utf-8")
    say = lambda s: (log.write(time.strftime("%H:%M:%S ") + s + "\n"), log.flush())
    _, fm, _, _ = load_cache_any(CACHE)
    mm = np.memmap(OUT / FRAME_FILE, dtype=np.uint8, mode="r+", shape=(N, H, W, 3))
    faces = np.zeros(len(ns), np.int16)
    mad = np.zeros((len(ns), 2), np.float32)
    nets = [CenterFace(in_shape=s, backend="opencv") for s in DETECT_SHAPES]
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", str(VID / f"{name}.mp4"), "-an",
           "-vf", f"fps=10,select='{job['expr']}'", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    say(f"start {name}: {len(ns)} frames, slots {slot0}..{slot0 + len(ns) - 1}")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=SH * SW * 3 * 2)
    nbytes = SH * SW * 3
    buf = bytearray(nbytes)
    view = memoryview(buf)
    samples = 0
    i = 0
    while True:
        got = 0
        while got < nbytes:
            k = p.stdout.readinto(view[got:])
            if not k:
                break
            got += k
        if got == 0:
            break
        if got != nbytes:
            raise RuntimeError(f"{name}: partial frame ({got} bytes) at output {i}")
        if i >= len(ns):
            raise RuntimeError(f"{name}: ffmpeg gave MORE frames than planned ({len(ns)})")
        f = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3).copy()
        dets = [net(f, threshold=THRESH)[0] for net in nets]
        dets = np.concatenate([d for d in dets if len(d)], 0) if any(len(d) for d in dets) else np.zeros((0, 5), np.float32)
        if len(dets):
            anonymize_frame(dets, f, mask_scale=MASK_SCALE, replacewith="blur", ellipse=True, draw_scores=False,
                            replaceimg=None, mosaicsize=20)
        hr = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        mm[slot0 + i] = hr
        faces[i] = len(dets)
        g = job["start"] + ns[i]
        lo = down2(hr)
        mad[i, 0] = np.abs(lo - np.asarray(fm[g], np.float32)).mean()
        nxt = g + 1 if ns[i] + 1 < job["frames"] else g - 1
        mad[i, 1] = np.abs(lo - np.asarray(fm[nxt], np.float32)).mean()
        if len(dets) >= 4 and samples < 2:                      # a couple of blurred examples to eyeball
            cv2.imwrite(str(OUT / "samples" / f"{name}_{ns[i]}_faces{len(dets)}.jpg"),
                        cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
            samples += 1
        i += 1
        if i % 500 == 0:
            el = time.time() - t0
            say(f"{i}/{len(ns)} ({i / el:.1f} fps, eta {(len(ns) - i) / max(i / el, 1e-6) / 60:.1f} min) "
                f"faces/frame {faces[:i].mean():.2f} mad same {mad[:i, 0].mean():.2f} next {mad[:i, 1].mean():.2f}")
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    mm.flush()
    del mm
    if i != len(ns):
        raise RuntimeError(f"{name}: ffmpeg gave {i} frames, planned {len(ns)}. stderr: {err[-500:]}")
    took = time.time() - t0
    res = {"name": name, "frames": i, "secs": round(took), "faces_total": int(faces.sum()),
           "frames_with_faces": int((faces > 0).sum()), "mad_same": round(float(mad[:, 0].mean()), 3),
           "mad_next": round(float(mad[:, 1].mean()), 3)}
    np.save(OUT / "done" / f"{name}_faces.npy", faces)
    np.save(OUT / "done" / f"{name}_mad.npy", mad)
    (OUT / "done" / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")
    say(f"DONE {res}")
    log.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4, help="OpenCV threads per worker")
    ap.add_argument("--only", default="", help="comma list of cities (testing)")
    args = ap.parse_args()
    from data_v3 import load_cache_any
    index, fm, act, cut = load_cache_any(CACHE)
    plans, walks = plan(index, cut)
    N = sum(len(p["ns"]) for p in plans)
    for d in ("logs", "done", "samples"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    cache_idx = np.concatenate([p["start"] + np.array(p["ns"], np.int64) for p in plans])
    split = np.concatenate([np.array(p["split"], np.int8) for p in plans])
    walk = np.concatenate([np.array(p["walk"], np.int8) for p in plans])
    if (OUT / "cache_idx.npy").exists():
        assert np.array_equal(np.load(OUT / "cache_idx.npy"), cache_idx), "plan changed since the last run"
    np.save(OUT / "cache_idx.npy", cache_idx)
    np.save(OUT / "split.npy", split)
    np.save(OUT / "walk.npy", walk)
    print(f"plan: {N} frames = {(split == 0).sum()} train / {(split == 1).sum()} val / {(split == 2).sum()} test, "
          f"{N * H * W * 3 / 1e9:.2f} GB; judge walks {walks}", flush=True)
    for p in plans:
        sp = np.array(p["split"])
        print(f"  {p['name']:<13} train {(sp == 0).sum():>5} val {(sp == 1).sum():>4} test {(sp == 2).sum():>4} "
              f"(training part < {p['tr_end']}, tail from {p['h0']} of {p['frames']})")
    if not (OUT / FRAME_FILE).exists():
        mm = np.memmap(OUT / FRAME_FILE, dtype=np.uint8, mode="w+", shape=(N, H, W, 3))
        del mm
    jobs, slot = [], 0
    for p in plans:
        jobs.append({"name": p["name"], "slot0": slot, "ns": p["ns"], "expr": p["expr"], "start": p["start"],
                     "frames": p["frames"], "N": N, "threads": args.threads})
        slot += len(p["ns"])
    only = [s for s in args.only.split(",") if s]
    todo = [j for j in jobs if not (OUT / "done" / f"{j['name']}.json").exists() and (not only or j["name"] in only)]
    todo.sort(key=lambda j: -len(j["ns"]))
    print(f"to do: {[j['name'] for j in todo]}", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(worker, todo):
            print(f"  {r} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # gather
    if all((OUT / "done" / f"{j['name']}.json").exists() for j in jobs):
        faces = np.concatenate([np.load(OUT / "done" / f"{j['name']}_faces.npy") for j in jobs])
        mad = np.concatenate([np.load(OUT / "done" / f"{j['name']}_mad.npy") for j in jobs])
        np.save(OUT / "faces.npy", faces)
        np.save(OUT / "mad.npy", mad)
        per = {j["name"]: json.loads((OUT / "done" / f"{j['name']}.json").read_text(encoding="utf-8")) for j in jobs}
        info = {"shape": [N, H, W, 3], "dtype": "uint8", "frame_file": FRAME_FILE, "cache": "data/cache_v3",
                "lowres": [96, 168], "scale": 2,
                "alignment": "cache_v3 frame (city start + n) = ffmpeg -hwaccel cuda -i data/videos/<City>.mp4 "
                             "-vf fps=10 output frame n (newest source frame with pts < (n+0.5)/10 s); full 16:9 frame "
                             "INTER_AREA -> 336x192 (cache: -> 256x144 area -> x264 -> deface -> 168x96 area, no crop)",
                "face_blur": {"tool": "deface 1.5.0 CenterFace (bundled centerface.onnx, OpenCV backend)",
                              "threshold": THRESH, "mask_scale": MASK_SCALE, "replacewith": "blur", "ellipse": True,
                              "detect_at": ["1280x720", "640x360", "256x144"], "blurred_at": "1280x720"},
                "splits": {"train": int((split == 0).sum()), "val": int((split == 1).sum()), "test": int((split == 2).sum())},
                "strides": {"train": STRIDE_TRAIN, "val": STRIDE_VAL, "gap_before_tail": GAP},
                "judge_walks": walks, "mad_same_mean": round(float(mad[:, 0].mean()), 3),
                "mad_next_mean": round(float(mad[:, 1].mean()), 3), "per_city": per,
                "built": time.strftime("%Y-%m-%d %H:%M:%S")}
        (OUT / "index.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"ALL DONE: {N} frames; mean |2x2(HR) - cache| same frame {mad[:, 0].mean():.3f}, next frame "
              f"{mad[:, 1].mean():.3f}; faces blurred {int(faces.sum())} in {int((faces > 0).sum())} frames", flush=True)


if __name__ == "__main__":
    main()
