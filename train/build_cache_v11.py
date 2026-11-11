"""Build the v11 (2x) training cache: ALL of data/cache_v3 again, at 192 x 336, into data/cache_v11/.

    data\\venv\\Scripts\\python.exe train\\build_cache_v11.py [--workers 20] [--chunk 600]

WHAT: the same 518,346 frames in the same order (same cities incl. Wildlife, same starts/ends, same indices) as
  data/cache_v3, but twice the size: frames_192x336.u8 (uint8 memmap [N,192,336,3], ~100 GB). Everything else is
  copied unchanged from cache_v3 (actions.npy, cut.npy, valid.npy, motion.npy, motion_norm.json); index.json is
  cache_v3's with "shape" [N,192,336,3] and "frame_file" changed (+ notes). So data_v2/data_v3.load_cache_any and
  every loader read cache_v11 exactly like cache_v3. index.json is written LAST, only when every frame is done, so
  an unfinished cache cannot be trained on by accident.

HOW (exactly train/sr/build_hr.py's recipe, measured there; see train/sr/README.md section 1-2):
  cache_v3 frame (city start + n) = output frame n of `ffmpeg -hwaccel cuda -i data/videos/<City>.mp4 -vf fps=10`.
  The whole 16:9 1280x720 frame is face-blurred AT 1280x720 and then shrunk with INTER_AREA to 336x192 (no crop).
  FACES (project rule): deface 1.5.0's bundled CenterFace (OpenCV backend, nothing downloaded), threshold 0.2, mask
  scale 1.3, ellipse blur (deface's own anonymize_frame), detection at THREE sizes (1280x720, 640x360, 256x144),
  every box from any size blurred. (The 2-size shortcut was tested and NOT taken: train/_exp/v11/face_check.json -
  it misses real faces 4-7 px wide in the 336-wide output.)
  --device gpu (default): the same CenterFace network + weights + resize + decode run on the GPU with PyTorch
  (train/centerface_torch.py, ~25x faster than OpenCV on the CPU); checked identical: 237/237 boxes, IoU 0.999996,
  40/40 final frames byte-identical (train/_exp/v11/face_gpu_check.json). --device cpu = deface's OpenCV backend.
  The first 30 chunks of Amsterdam were made on the CPU, the rest on the GPU (2026-10-23, 8 workers x batch 4,
  ~160 frames/s; 12 workers overfilled the 32 GB GPU and slowed to ~45 frames/s).
  Speed: each city is cut into chunks of --chunk frames; a chunk = ffmpeg seeking 1 s before it (-ss, a multiple of
  0.1 s, so the fps=10 grid is unchanged - checked: 500/500 seeked frames byte-identical to data/cache_hr).

CHECKS (per frame, summed per chunk and city):
  mad = mean |2x2 mean(HR frame) - cache_v3 frame| for the SAME frame and the NEXT frame (alignment, like build_hr);
  a frame is "off" when the next frame matches as well as the same one. Every frame that data/cache_hr also has
  (41,087, built the slow way) must come out byte-identical.

RESTART-SAFE: a finished chunk leaves done/chunks/<City>_<start>.npz; re-running skips it. A finished city leaves
  done/<City>.json. Logs: data/cache_v11/logs/<City>.txt and the --log file (default train/_exp/v11/part1_cache.log).
Exit code 0 only when the whole cache is complete and checked (the overnight .bat relies on it).
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
VID = DW / "data" / "videos"
SRC = DW / "data" / "cache_v3"
HR = DW / "data" / "cache_hr"
OUT = DW / "data" / "cache_v11"
H, W = 192, 336
SH, SW = 720, 1280
FB = H * W * 3
FRAME_FILE = f"frames_{H}x{W}.u8"
THRESH, MASK_SCALE = 0.2, 1.3                     # prep_videos.py / build_hr.py
DETECT = {"3": [None, (640, 360), (256, 144)],    # build_hr.py (None = the full 1280x720 frame)
          "2": [(640, 360), (256, 144)]}
PRE = 10                                          # start the seek 1 s (10 output frames) before the chunk
COPY = ["actions.npy", "cut.npy", "valid.npy", "motion.npy", "motion_norm.json"]
LOGFILE = [None]


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if LOGFILE[0]:
        with open(LOGFILE[0], "a", encoding="utf-8") as f:
            f.write(line + "\n")


def blur_faces(f, dets):
    """deface's anonymize_frame on the 1280x720 frame (in place). A degenerate box that makes OpenCV complain is
    blurred on its own with a plain Gaussian over the same (mask-scaled) box instead, so no face is ever skipped."""
    import cv2
    from deface.deface import anonymize_frame
    if not len(dets):
        return 0
    try:
        anonymize_frame(dets, f, mask_scale=MASK_SCALE, replacewith="blur", ellipse=True, draw_scores=False,
                        replaceimg=None, mosaicsize=20)
        return 0
    except Exception:
        bad = 0
        for d in dets:
            try:
                anonymize_frame(d[None], f, mask_scale=MASK_SCALE, replacewith="blur", ellipse=True,
                                draw_scores=False, replaceimg=None, mosaicsize=20)
            except Exception:
                bad += 1
                x1, y1, x2, y2 = d[:4]
                cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * MASK_SCALE, (y2 - y1) * MASK_SCALE
                a, b = int(max(0, cx - w / 2 - 2)), int(max(0, cy - h / 2 - 2))
                c, e = int(min(SW, cx + w / 2 + 3)), int(min(SH, cy + h / 2 + 3))
                if c > a and e > b:
                    f[b:e, a:c] = cv2.GaussianBlur(f[b:e, a:c], (0, 0), max(1.0, max(w, h) / 4))
        return bad


GPU_NET = [None]


def detector(job):
    """-> f(frames uint8 [B, 720, 1280, 3]) -> list of B box arrays [n, 5] from all detection sizes together.
    'cpu': deface's own CenterFace (OpenCV DNN), one frame at a time - build_hr.py's way.
    'gpu': the same network + weights + resize + decode on the GPU (train/centerface_torch.py); checked against
           OpenCV: identical boxes (IoU 0.999996, score diff 5e-7) and byte-identical 336x192 frames
           (train/_exp/v11/face_gpu_check.json)."""
    sizes = DETECT[job["detect"]]
    if job.get("device", "cpu") == "cpu":
        from deface.centerface import CenterFace
        nets = [CenterFace(in_shape=s, backend="opencv") for s in sizes]

        def run(frames):
            out = []
            for f in frames:
                d = [net(f, threshold=THRESH)[0] for net in nets]
                d = [x for x in d if len(x)]
                out.append(np.concatenate(d, 0) if d else np.zeros((0, 5), np.float32))
            return out
        return run
    import torch
    torch.backends.cudnn.allow_tf32 = False                 # float32 like OpenCV
    torch.backends.cuda.matmul.allow_tf32 = False
    if GPU_NET[0] is None:
        sys.path.insert(0, str(HERE))
        from centerface_torch import CenterFaceTorch
        GPU_NET[0] = CenterFaceTorch(device="cuda")
    net = GPU_NET[0]

    def run(frames):
        per = [[] for _ in frames]
        for s in sizes:
            for j, d in enumerate(net(frames, s, THRESH)):
                if len(d):
                    per[j].append(d)
        return [np.concatenate(d, 0) if d else np.zeros((0, 5), np.float32) for d in per]
    return run


def run_chunk(job):
    """One chunk of one city: ffmpeg seek + fps=10 -> blur at 1280x720 -> INTER_AREA 336x192 -> write the slots."""
    import cv2
    cv2.setNumThreads(1)
    name, city, a, b = job["name"], job["city"], job["a"], job["b"]
    s0, cend, n_city = job["start"], job["end"], job["frames"]
    shape3 = tuple(job["src_shape"])
    fm = np.memmap(SRC / job["src_file"], dtype=np.uint8, mode="r", shape=shape3)
    hr_idx = np.load(HR / "cache_idx.npy") if (HR / "cache_idx.npy").exists() else np.zeros(0, np.int64)
    hrm = (np.memmap(HR / "frames_192x336.u8", dtype=np.uint8, mode="r", shape=(len(hr_idx), H, W, 3))
           if len(hr_idx) else None)
    detect = detector(job)
    BATCH = job.get("batch", 8)
    m = max(0, a - PRE)
    want = b - m
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-ss", f"{m / 10:.1f}", "-i", str(VID / f"{city}.mp4"),
           "-an", "-vf", "fps=10", "-frames:v", str(want), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = b - a
    faces = np.zeros(n, np.int16)
    mad = np.zeros((n, 2), np.float32)
    st = {"hr_same": 0, "hr_diff": 0, "hr_n": 0, "fallback": 0}
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=SH * SW * 3 * 2)
    nbytes = SH * SW * 3
    out = open(OUT / FRAME_FILE, "r+b")
    out.seek((s0 + a) * FB)
    sample_done = [(OUT / "samples" / f"{city}.jpg").exists()]

    def finish(frames, nls):
        """detect (batched), blur, shrink, write, check - in frame order."""
        dets_all = detect(frames)
        for f, nl, dets in zip(frames, nls, dets_all):
            st["fallback"] += blur_faces(f, dets)
            hr = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            out.write(hr.tobytes())
            i = nl - a
            g = s0 + nl
            faces[i] = len(dets)
            lo = hr.astype(np.float32).reshape(H // 2, 2, W // 2, 2, 3).mean((1, 3))
            mad[i, 0] = np.abs(lo - fm[g]).mean()
            nxt = g + 1 if nl + 1 < n_city else g - 1
            mad[i, 1] = np.abs(lo - fm[nxt]).mean()
            if hrm is not None:
                j = np.searchsorted(hr_idx, g)
                if j < len(hr_idx) and hr_idx[j] == g:
                    st["hr_n"] += 1
                    same = np.array_equal(hr, hrm[j])
                    st["hr_same"] += int(same)
                    if not same:
                        st["hr_diff"] = max(st["hr_diff"], int(np.abs(hr.astype(np.int16) - hrm[j]).max()))
            if not sample_done[0] and len(dets) >= 4:     # one blurred example per city to eyeball
                (OUT / "samples").mkdir(exist_ok=True)
                cv2.imwrite(str(OUT / "samples" / f"{city}.jpg"),
                            cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
                sample_done[0] = True

    k = 0
    pend, pend_nl = [], []
    while True:
        buf = bytearray(nbytes)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            r = p.stdout.readinto(view[got:])
            if not r:
                break
            got += r
        if got == 0:
            break
        if got != nbytes:
            raise RuntimeError(f"{name}: partial frame ({got} bytes) at output {k}")
        if k >= want:
            raise RuntimeError(f"{name}: ffmpeg gave more frames than asked ({want})")
        nl = m + k                                       # local frame index in this city
        k += 1
        if nl < a:
            continue
        pend.append(np.frombuffer(buf, np.uint8).reshape(SH, SW, 3))
        pend_nl.append(nl)
        if len(pend) >= BATCH:
            finish(np.stack(pend), pend_nl)
            pend, pend_nl = [], []
    if pend:
        finish(np.stack(pend), pend_nl)
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    out.close()
    if k != want:
        raise RuntimeError(f"{name}: ffmpeg gave {k} frames, wanted {want}. stderr: {err[-400:]}")
    took = time.time() - t0
    off = int((mad[:, 0] >= mad[:, 1]).sum())
    res = {"name": name, "city": city, "a": a, "b": b, "secs": round(took, 1), "fps": round(n / took, 2),
           "faces": int(faces.sum()), "frames_with_faces": int((faces > 0).sum()),
           "mad_same": round(float(mad[:, 0].mean()), 3), "mad_next": round(float(mad[:, 1].mean()), 3),
           "off": off, "hr_checked": st["hr_n"], "hr_identical": st["hr_same"], "hr_maxdiff": st["hr_diff"],
           "fallback_blurs": st["fallback"], "device": job.get("device", "cpu")}
    tmp = OUT / "done" / "chunks" / f"{name}.tmp.npz"
    np.savez(tmp, faces=faces, mad=mad, info=json.dumps(res))
    os.replace(tmp, OUT / "done" / "chunks" / f"{name}.npz")
    with open(OUT / "logs" / f"{city}.txt", "a", encoding="utf-8") as lf:
        lf.write(time.strftime("%H:%M:%S ") + json.dumps(res) + "\n")
    return res


def safe_chunk(job):
    for attempt in range(2):
        try:
            return run_chunk(job)
        except Exception as e:                           # one retry (e.g. a hiccup in ffmpeg), then report
            err = f"{type(e).__name__}: {e}"
            time.sleep(2)
    return {"name": job["name"], "city": job["city"], "error": err}


def make_frame_file(n):
    path = OUT / FRAME_FILE
    size = n * FB
    if path.exists() and path.stat().st_size == size:
        return
    if path.exists():
        raise SystemExit(f"{path} exists with the wrong size ({path.stat().st_size} vs {size}); not touching it")
    say(f"creating {path.name}: {size / 1e9:.1f} GB (the file system zero-fills it once)...")
    t0 = time.time()
    with open(path, "wb") as f:
        f.seek(size - 1)
        f.write(b"\0")
    say(f"  created in {time.time() - t0:.0f} s")


def city_summary(c, chunks):
    infos = [json.loads(str(np.load(OUT / "done" / "chunks" / f"{j['name']}.npz")["info"])) for j in chunks]
    tot = sum(i["b"] - i["a"] for i in infos)
    wsum = lambda key: sum(i[key] * (i["b"] - i["a"]) for i in infos) / tot
    return {"name": c["name"], "frames": tot, "chunks": len(infos),
            "worker_secs": round(sum(i["secs"] for i in infos)), "faces_total": sum(i["faces"] for i in infos),
            "frames_with_faces": sum(i["frames_with_faces"] for i in infos),
            "mad_same": round(wsum("mad_same"), 3), "mad_next": round(wsum("mad_next"), 3),
            "frames_off": sum(i["off"] for i in infos),
            "hr_checked": sum(i["hr_checked"] for i in infos), "hr_identical": sum(i["hr_identical"] for i in infos),
            "hr_maxdiff": max(i["hr_maxdiff"] for i in infos), "fallback_blurs": sum(i["fallback_blurs"] for i in infos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20, help="parallel chunk workers (1 OpenCV thread each)")
    ap.add_argument("--chunk", type=int, default=600, help="frames per chunk")
    ap.add_argument("--detect", default="3", choices=["3", "2"], help="face detection sizes (3 = build_hr's)")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"],
                    help="where CenterFace runs: gpu = train/centerface_torch.py (same net, checked identical), cpu = OpenCV")
    ap.add_argument("--batch", type=int, default=8, help="frames per detection call (gpu)")
    ap.add_argument("--only", default="", help="comma list of chunk names or cities (testing)")
    ap.add_argument("--max-chunks", type=int, default=0, help="stop after this many chunks (testing)")
    ap.add_argument("--log", default=str(DW / "train" / "_exp" / "v11" / "part1_cache.log"))
    args = ap.parse_args()
    LOGFILE[0] = args.log
    pathlib.Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    src_index = json.loads((SRC / "index.json").read_text(encoding="utf-8"))
    N = src_index["shape"][0]
    for d in ("logs", "done/chunks", "samples"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    if (OUT / "index.json").exists():
        info = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
        if info.get("complete"):
            say(f"cache_v11 already complete ({info['shape']}); nothing to do")
            return 0
    for fn in COPY:
        if not (OUT / fn).exists():
            shutil.copy2(SRC / fn, OUT / fn)
    make_frame_file(N)
    jobs = []
    for c in src_index["cities"]:
        for a in range(0, c["frames"], args.chunk):
            jobs.append({"name": f"{c['name']}_{a:06d}_{min(a + args.chunk, c['frames']):06d}", "city": c["name"], "a": a,
                         "b": min(a + args.chunk, c["frames"]), "start": c["start"], "end": c["end"],
                         "frames": c["frames"], "detect": args.detect, "src_shape": src_index["shape"],
                         "device": args.device, "batch": args.batch,
                         "src_file": src_index.get("frame_file", "frames_96x168.u8")})
    done = lambda j: (OUT / "done" / "chunks" / f"{j['name']}.npz").exists()
    only = [s for s in args.only.split(",") if s]
    todo = [j for j in jobs if not done(j) and (not only or j["name"] in only or j["city"] in only)]
    if args.max_chunks:
        todo = todo[:args.max_chunks]
    left0 = sum(j["b"] - j["a"] for j in todo)
    say(f"cache_v11: {N} frames, {len(jobs)} chunks of {args.chunk}; {len(todo)} chunks ({left0} frames) to do, "
        f"{args.workers} workers, face detection at {args.detect} sizes on the {args.device.upper()}")
    per_city = {c["name"]: [j for j in jobs if j["city"] == c["name"]] for c in src_index["cities"]}
    t0 = time.time()
    n_done, errors = 0, []
    if todo:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(safe_chunk, todo):
                if "error" in r:
                    errors.append(r)
                    say(f"  !! {r['name']} FAILED: {r['error']}")
                    continue
                n_done += r["b"] - r["a"]
                el = time.time() - t0
                fps = n_done / el
                say(f"  {r['name']}: {r['fps']} fps/worker, faces {r['faces']}, mad {r['mad_same']}/{r['mad_next']}"
                    f"{' OFF ' + str(r['off']) if r['off'] else ''}"
                    f"{', cache_hr identical ' + str(r['hr_identical']) + '/' + str(r['hr_checked']) if r['hr_checked'] else ''}"
                    f" | total {n_done}/{left0} ({fps:.1f} fps, eta {(left0 - n_done) / max(fps, 1e-6) / 3600:.2f} h)")
                city = r["city"]
                if all(done(j) for j in per_city[city]) and not (OUT / "done" / f"{city}.json").exists():
                    cs = city_summary(next(c for c in src_index["cities"] if c["name"] == city), per_city[city])
                    (OUT / "done" / f"{city}.json").write_text(json.dumps(cs), encoding="utf-8")
                    say(f"CITY DONE {city}: {json.dumps(cs)}")
    if not all(done(j) for j in jobs):
        left = [j["name"] for j in jobs if not done(j)]
        say(f"NOT COMPLETE: {len(left)} chunks left ({len(errors)} failed this run). Re-run to continue.")
        return 1
    # gather + checks + index.json (last)
    faces = np.zeros(N, np.int16)
    mad = np.zeros((N, 2), np.float32)
    per = {}
    for c in src_index["cities"]:
        for j in per_city[c["name"]]:
            z = np.load(OUT / "done" / "chunks" / f"{j['name']}.npz")
            faces[c["start"] + j["a"]:c["start"] + j["b"]] = z["faces"]
            mad[c["start"] + j["a"]:c["start"] + j["b"]] = z["mad"]
        per[c["name"]] = city_summary(c, per_city[c["name"]])
        if not (OUT / "done" / f"{c['name']}.json").exists():
            (OUT / "done" / f"{c['name']}.json").write_text(json.dumps(per[c["name"]]), encoding="utf-8")
    np.save(OUT / "faces.npy", faces)
    np.save(OUT / "mad.npy", mad)
    hr_chk = sum(p["hr_checked"] for p in per.values())
    hr_id = sum(p["hr_identical"] for p in per.values())
    off = int((mad[:, 0] >= mad[:, 1]).sum())
    info = dict(src_index)
    info.update({
        "shape": [N, H, W, 3], "frame_file": FRAME_FILE, "complete": True,
        "v11": {"from": "data/cache_v3 (same frames, order, indices; actions/cut/valid/motion copied unchanged)",
                "alignment": "cache_v3 frame (city start + n) = ffmpeg -hwaccel cuda -i data/videos/<City>.mp4 -vf "
                             "fps=10 output frame n; full 16:9 1280x720 frame -> face blur -> INTER_AREA 336x192",
                "face_blur": {"tool": "deface 1.5.0 CenterFace (bundled centerface.onnx, OpenCV backend)",
                              "threshold": THRESH, "mask_scale": MASK_SCALE, "replacewith": "blur", "ellipse": True,
                              "detect_at": ["1280x720" if s is None else f"{s[0]}x{s[1]}" for s in DETECT[args.detect]],
                              "blurred_at": "1280x720"},
                "mad_same_mean": round(float(mad[:, 0].mean()), 3), "mad_next_mean": round(float(mad[:, 1].mean()), 3),
                "frames_off": off, "cache_hr_checked": hr_chk, "cache_hr_identical": hr_id,
                "faces_total": int(faces.sum()), "frames_with_faces": int((faces > 0).sum()),
                "per_city": per, "built": time.strftime("%Y-%m-%d %H:%M:%S")}})
    tmp = OUT / "index.tmp.json"
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    os.replace(tmp, OUT / "index.json")
    say(f"ALL DONE: {N} frames at {H}x{W}; mad same {mad[:, 0].mean():.3f} vs next {mad[:, 1].mean():.3f}; "
        f"frames off {off}; cache_hr identical {hr_id}/{hr_chk}; faces blurred {int(faces.sum())} in "
        f"{int((faces > 0).sum())} frames")
    for c in src_index["cities"]:
        p = per[c["name"]]
        say(f"  {c['name']:<13} mad {p['mad_same']:.2f} / next {p['mad_next']:.2f}  off {p['frames_off']}  "
            f"cache_hr {p['hr_identical']}/{p['hr_checked']}  faces {p['faces_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
