"""Step 1: turn raw Walking Tours videos into clean training footage.

    data\\venv\\Scripts\\python prep\\prep_videos.py

For each video in data/videos/*.mp4:
  1. SHRINK  (ffmpeg)  720p 60 fps -> 256x144 at 10 fps. The model trains at 64x64 first
     (128 later), so this already has plenty of detail, and 10 fps is enough to see the
     camera move between frames. Shrinking first makes every later step about 50x faster.
  2. BLUR FACES (deface)  strangers in the street get their faces blurred. It runs on the
     small video, which is fast, and at 256x144 faces are already tiny.
  3. PREVIEW  a strip of 8 frames for Dream Lab, so you can see what the model will learn from.

Output:
  data/clean/<City>.mp4      the footage we train on (git-ignored, NAS-skipped)
  prep/preview/<City>.jpg    contact strip shown in Dream Lab
  prep/status/<City>.json    progress, read by Dream Lab

Safe to re-run: finished steps are skipped.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

DW = pathlib.Path(__file__).resolve().parent.parent
RAW = DW / "data" / "videos"
CLEAN = DW / "data" / "clean"
PREVIEW = DW / "prep" / "preview"
STATUS_DIR = DW / "prep" / "status"
DEFACE = DW / "data" / "venv" / "Scripts" / "deface.exe"

W, H, FPS = 256, 144, 10

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CLEAN.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)

status = {"videos": {}}
CURRENT = [None]  # city this process is working on


def save():
    city = CURRENT[0]
    tmp = STATUS_DIR / f"{city}.tmp"
    tmp.write_text(json.dumps(status["videos"][city]), encoding="utf-8")
    # Windows refuses to replace a file while Dream Lab is reading it. Retry a few
    # times; if it's still busy, skip this update (the next one comes in a second).
    for _ in range(5):
        try:
            os.replace(tmp, STATUS_DIR / f"{city}.json")
            return
        except PermissionError:
            time.sleep(0.05)


def duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)], text=True)
    return float(out.strip())


def run_ffmpeg_with_progress(cmd, total_secs, entry):
    """Run ffmpeg, parsing its -progress output so Dream Lab can show a % bar."""
    p = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats", "-loglevel", "error"],
                         stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.startswith("out_time_us="):
            try:
                entry["pct"] = round(min(100, int(line.split("=")[1]) / 1e6 / total_secs * 100), 1)
                save()
            except ValueError:
                pass
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed: " + " ".join(cmd[:6]))


def make_preview(src, dst, secs):
    """8 frames spread across the video, side by side."""
    shots = []
    for i in range(8):
        t = secs * (i + 0.5) / 8
        shot = PREVIEW / f"_tmp{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.1f}", "-i", str(src),
                        "-frames:v", "1", str(shot)], check=True)
        shots.append(shot)
    from PIL import Image
    ims = [Image.open(s) for s in shots]
    strip = Image.new("RGB", (W * 8, H))
    for i, im in enumerate(ims):
        strip.paste(im.resize((W, H)), (i * W, 0))
    strip.save(dst, quality=85)
    for s in shots:
        s.unlink()


# Optional city names on the command line: run one process per city to use all CPU cores
videos = sorted(RAW.glob("*.mp4"))
if len(sys.argv) > 1:
    videos = [v for v in videos if v.stem in sys.argv[1:]]
if not videos:
    sys.exit("no videos in data/videos yet")

for v in videos:
    city = v.stem
    CURRENT[0] = city
    e = status["videos"].setdefault(city, {})
    secs = duration(v)
    e.update(hours=round(secs / 3600, 2), frames=int(secs * FPS))

    small = CLEAN / f"{city}_small.mp4"
    final = CLEAN / f"{city}.mp4"

    # 1. shrink
    if not final.exists() and not small.exists():
        e.update(stage="shrinking", pct=0); save()
        print(f"[{city}] shrinking {secs/3600:.2f} h to {W}x{H} @ {FPS} fps")
        tmp = CLEAN / f"{city}_small.tmp.mp4"
        run_ffmpeg_with_progress(
            ["ffmpeg", "-y", "-hwaccel", "cuda", "-i", str(v), "-an",
             "-vf", f"fps={FPS},scale={W}:{H}:flags=area",
             "-c:v", "libx264", "-crf", "18", "-preset", "fast", str(tmp)], secs, e)
        os.replace(tmp, small)

    # 2. blur faces
    if not final.exists():
        e.update(stage="blurring faces", pct=None); save()
        print(f"[{city}] blurring faces")
        tmp = CLEAN / f"{city}.tmp.mp4"
        subprocess.run([str(DEFACE), str(small), "-o", str(tmp), "--thresh", "0.2",
                        "--replacewith", "blur"], check=True)
        os.replace(tmp, final)
        small.unlink()

    # 3. preview strip
    prev = PREVIEW / f"{city}.jpg"
    if not prev.exists():
        e.update(stage="preview"); save()
        make_preview(final, prev, secs)

    e.update(stage="ready", pct=100, preview=f"/files/prep/preview/{city}.jpg"); save()
    print(f"[{city}] ready")

print("all videos ready")
