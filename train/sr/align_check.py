"""Step 1 of "sharper eyes": how does a cache_v3 frame line up with the original 1280x720 video?

    data\\venv\\Scripts\\python.exe train\\sr\\align_check.py

The cache chain (prep/prep_videos.py + train/build_cache_v3.py):
    data/videos/<City>.mp4 (1280x720, ~60 fps)
      -> ffmpeg -hwaccel cuda  -vf "fps=10,scale=256:144:flags=area"  x264 crf 18      (small.mp4)
      -> deface --thresh 0.2 --replacewith blur  (imageio re-encode, 1:1 frames)          (clean/<City>.mp4)
      -> cv2 decode, cv2.resize(.., (168, 96), INTER_AREA)  appended in order          (cache_v3, frame i)
So cache frame (city start + n) = the n-th frame ffmpeg's fps=10 filter outputs, full 16:9 view squashed into
168x96 (no crop). This script checks that numerically two ways:

 A. 60 fps search (PyAV): for a few n per city, decode every SOURCE frame with pts in [n/10 - 0.12, n/10 + 0.12],
    shrink it straight to 168x96 (INTER_AREA) and compare with cache frame n -> which source frame wins.
 B. The HR pipeline itself (ffmpeg fps=10 + select, rgb24 1280x720 -> 336x192 INTER_AREA -> 2x2 mean = 168x96)
    vs cache frames n-1, n, n+1 -> mean abs difference (0..255) and the mean signed colour difference per channel.
"""
import json
import pathlib
import subprocess
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = pathlib.Path(__file__).resolve().parent
DW = SR.parent.parent
sys.path.insert(0, str(SR.parent))
from data_v3 import load_cache_any  # noqa: E402

VID = DW / "data" / "videos"
index, fm, act, cut = load_cache_any(DW / "data" / "cache_v3")
cities = {c["name"]: c for c in index["cities"]}


def down2(hr):
    """[.., 192, 336, 3] uint8 -> [.., 96, 168, 3] float32 by 2x2 mean (= INTER_AREA at exactly 1/2)."""
    h = hr.astype(np.float32)
    return h.reshape(*h.shape[:-3], 96, 2, 168, 2, 3).mean((-4, -2))


def ffmpeg_frames(path, ns, w=1280, h=720, lead=20):
    """Frames n in ns of ffmpeg's fps=10 output, as rgb24 [len(ns), h, w, 3]. Each one: input-seek to
    (n - lead) / 10 s (a whole number of 0.1 s steps, so the fps filter's 0.1 s bins don't move), then take
    output frame number `lead` (the first few outputs after a seek can differ, so they're skipped)."""
    out = []
    for n in ns:
        s0 = max(0, n - lead)
        cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-ss", f"{s0 / 10:.1f}", "-i", str(path), "-an",
               "-vf", f"fps=10,select='eq(n\,{n - s0})'", "-fps_mode", "passthrough", "-frames:v", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        out.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    return np.stack(out)


def pyav_search(path, n, cache_frame, win=0.12):
    import av
    with av.open(str(path)) as ct:
        st = ct.streams.video[0]
        st.thread_type = "AUTO"
        t0 = n / 10 - win
        ct.seek(int(max(0, t0 - 3) / st.time_base), stream=st, backward=True)
        rows = []
        for fr in ct.decode(st):
            t = float(fr.pts * st.time_base)
            if t < t0:
                continue
            if t > n / 10 + win:
                break
            rgb = fr.to_ndarray(format="rgb24")
            small = cv2.resize(rgb, (168, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
            rows.append((t, float(np.abs(small - cache_frame).mean())))
        return rows


def main():
    out = {"A_60fps_search": {}, "B_hr_pipeline": {}}
    # A: which 60 fps source frame is cache frame n?
    for name in ["Amsterdam", "Bangkok", "Zurich", "Venice"]:
        c = cities[name]
        for n in [37, c["frames"] // 2 + 3, c["frames"] - 60]:
            rows = pyav_search(VID / f"{name}.mp4", n, np.asarray(fm[c["start"] + n], np.float32))
            best = min(rows, key=lambda r: r[1])
            src_idx_60 = best[0] * 60
            line = " ".join(f"{t - n / 10:+.4f}s:{d:.1f}" for t, d in rows)
            print(f"A {name:<10} n={n:>6} best source pts - n/10 = {best[0] - n / 10:+.4f} s "
                  f"(diff {best[1]:.2f}) | {line}", flush=True)
            out["A_60fps_search"][f"{name}_{n}"] = {"best_dt": round(best[0] - n / 10, 4), "best_diff": round(best[1], 2),
                                                    "all": [(round(t - n / 10, 4), round(d, 2)) for t, d in rows]}
    # B: the HR pipeline vs cache frames n-1, n, n+1
    tot = []
    for name, c in cities.items():
        if name == "Wildlife":
            continue
        ns = sorted({200, c["frames"] // 3, (2 * c["frames"]) // 3, c["frames"] - 30})
        fr = ffmpeg_frames(VID / f"{name}.mp4", ns)
        assert len(fr) == len(ns), (name, len(fr), len(ns))
        for n, f in zip(ns, fr):
            hr = cv2.resize(f, (336, 192), interpolation=cv2.INTER_AREA)
            lo = down2(hr)
            d = {o: float(np.abs(lo - np.asarray(fm[c["start"] + n + o], np.float32)).mean()) for o in (-1, 0, 1)}
            sgn = (lo - np.asarray(fm[c["start"] + n], np.float32)).reshape(-1, 3).mean(0)
            # the same frame squeezed the way the cache was: 1280x720 -> 256x144 -> 168x96 (area both times)
            via = cv2.resize(cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA), (168, 96),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            dv = float(np.abs(via - np.asarray(fm[c["start"] + n], np.float32)).mean())
            tot.append(d[0])
            print(f"B {name:<12} n={n:>6}: |HR->96x168 - cache| n-1 {d[-1]:5.2f}  n {d[0]:5.2f}  n+1 {d[1]:5.2f}  "
                  f"| via 256x144 {dv:5.2f} | mean colour shift RGB {sgn.round(2).tolist()}", flush=True)
            out["B_hr_pipeline"][f"{name}_{n}"] = {"prev": round(d[-1], 2), "same": round(d[0], 2), "next": round(d[1], 2),
                                                   "via_256": round(dv, 2), "rgb_shift": sgn.round(2).tolist()}
    out["B_mean_abs_diff_same_frame"] = round(float(np.mean(tot)), 3)
    print(f"\nB overall mean abs diff (same frame, 0..255): {np.mean(tot):.3f}")
    (SR / "out" / "align_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
