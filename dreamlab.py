"""Dream Lab: watch Dream World come alive, live, in your browser.

    python dreamlab.py            -> http://localhost:8803

Shows, refreshing every second:
  - the DREAM SCREEN: frames from the newest run, played as a video while they're
    still being made (taste tests now, our own model's samples later)
  - GPU: memory used, load, temperature
  - downloads: how far along each model download is
  - the run's log tail, and every finished video

Stdlib only, so it runs with plain Python (no venv needed). It only READS files
the runs write into taste/live/<run>/ (status.json, run.log, frame_00000.jpg ...).
Training runs (step 3+) will write the same kind of folder into train/live/.
"""
import http.server
import json
import pathlib
import subprocess
import sys
import urllib.parse

DW = pathlib.Path(__file__).resolve().parent
PORT = 8803
LIVE_DIRS = [DW / "taste" / "live", DW / "train" / "live"]
# quick test runs (sprints): shown only if started in the last 36 h, and never the smoke tests themselves
RECENT_DIRS = [DW / "train" / "_smoke" / "v9" / "live", DW / "train" / "_smoke" / "v11" / "live"]
JUDGE_DIR = DW / "train" / "_exp" / "v9"            # the honest 20-walk judge's results (v11judge_*.json)
VIDEO_DIRS = [DW / "taste" / "out", DW / "train" / "out", DW / "prep" / "preview"]  # preview = WASD label checks

# Expected download sizes (GB), so the page can draw progress bars
DOWNLOADS = {
    "HY-WorldPlay 5B (Tencent)": (DW / "data" / "models" / "HY-WorldPlay", 52.3),
    "Wan2.2 text encoder + VAE": (DW / "data" / "models" / "Wan2.2-TI2V-5B-Diffusers", 14.2),
}


def folder_gb(p):
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=3).strip().split(", ")
        used, total, util, temp, power = out
        return {"used_gb": round(int(used) / 1024, 1), "total_gb": round(int(total) / 1024, 1),
                "util": int(util), "temp": int(temp), "power": round(float(power))}
    except Exception:
        return None


def checkup(p):
    """One check-up row of metrics.jsonl -> the numbers worth charting (None if it isn't a check-up)."""
    rm = p.get("ruler_mem")
    if not isinstance(rm, dict):
        return None
    nm = p.get("ruler_nomem") if isinstance(p.get("ruler_nomem"), dict) else {}
    turn = p.get("turn") if isinstance(p.get("turn"), dict) else {}
    return {"step": p.get("step"), "time": p.get("time"), "melt": rm.get("melt_critic"),
            "glitch": rm.get("glitch_time_mean"), "melt_nomem": nm.get("melt_critic"),
            "val_loss": p.get("val_loss"), "return_sim": turn.get("return_sim"), "video": p.get("rollout")}


_scores_cache = {"key": None, "rows": []}


def scores():
    """Every model the honest 20-walk judge has scored (train/_exp/v9/v11judge_*.json), best first."""
    files = sorted(JUDGE_DIR.glob("v11judge_*.json")) if JUDGE_DIR.exists() else []
    key = tuple((f.name, f.stat().st_mtime) for f in files)
    if key == _scores_cache["key"]:
        return _scores_cache["rows"]
    rows, seen = [], set()
    for f in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True):     # newest judge wins a repeated name
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for name, r in data.items():
            if not isinstance(r, dict) or r.get("MELT_SCORE_mean") is None or name.startswith("smk") or name in seen:
                continue
            seen.add(name)
            sharp = r.get("sharp_5_10_20_30_40s") or [None]
            rows.append({"name": name, "melt": r["MELT_SCORE_mean"], "melt_s": r.get("MELT_TIME_median"),
                         "glitch": r.get("glitch_time_mean"), "sharp5": sharp[0], "step": r.get("step"),
                         "judge": f.stem[len("v11judge_"):], "time": f.stat().st_mtime})
    rows.sort(key=lambda r: r["melt"])
    _scores_cache.update(key=key, rows=rows)
    return rows


def runs():
    found = []
    now = __import__("time").time()
    for base in LIVE_DIRS + RECENT_DIRS:
        if not base.exists():
            continue
        for d in base.iterdir():
            st = d / "status.json"
            if not st.exists():
                continue
            try:
                s = json.loads(st.read_text(encoding="utf-8"))
            except Exception:
                continue  # half-written, try again next poll
            if base in RECENT_DIRS and (d.name.startswith(("smk", "wd_smoke")) or now - s.get("started", 0) > 36 * 3600):
                continue
            frames = sorted(d.glob("frame_*.jpg"))
            s["frames"] = len(frames)
            s["url"] = "/files/" + d.relative_to(DW).as_posix() + "/"
            met = d / "metrics.jsonl"
            if met.exists():
                pts = []
                for ln in met.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        pts.append(json.loads(ln))
                    except Exception:
                        pass  # last line may be half-written
                s["checkups"] = [c for c in map(checkup, pts) if c]
                pts = [p for p in pts if p.get("loss") is not None]
                step = max(1, len(pts) // 400)  # keep the chart light: at most ~400 points
                s["metrics"] = [{"step": p.get("step"), "loss": p.get("loss"), "time": p.get("time"),
                                 "it_s": p.get("it_s")} for p in pts[::step] + pts[-1:]]
            log = d / "run.log"
            if log.exists():
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                # tqdm progress bars use \r, keep only the newest piece of each line
                s["log"] = [ln.split("\r")[-1] for ln in lines[-14:]]
            found.append(s)
    return sorted(found, key=lambda s: s.get("started", 0), reverse=True)


def videos():
    vids = []
    for base in VIDEO_DIRS:
        if base.exists():
            for f in base.glob("*.mp4"):
                vids.append({"name": f.name, "url": "/files/" + f.relative_to(DW).as_posix(),
                             "time": f.stat().st_mtime})
    return sorted(vids, key=lambda v: v["time"], reverse=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep the console quiet

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_video(self, f):
        """Serve an mp4 with HTTP Range support. Browsers ask for videos in pieces
        ("bytes=0-", "bytes=1234-5678"); without proper 206 answers Chrome keeps
        re-requesting the whole file, so the player buffers and flickers forever."""
        data = f.read_bytes()
        size = len(data)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            elif b:  # "bytes=-500" = the last 500 bytes
                start = max(0, size - int(b))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "max-age=60")  # finished videos don't change
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            self.wfile.write(data[start:end + 1])
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # the browser cancelled this piece; normal for video

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self.send(200, (DW / "dreamlab.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            dataset = None
            for f in sorted((DW / "prep" / "status").glob("*.json")):
                try:
                    dataset = dataset or {"videos": {}}
                    dataset["videos"][f.stem] = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass  # half-written, next poll gets it
            body = {"gpu": gpu(), "runs": runs(), "videos": videos(), "dataset": dataset, "scores": scores(),
                    "downloads": [{"name": n, "gb": round(folder_gb(p), 1), "of": of}
                                  for n, (p, of) in DOWNLOADS.items()]}
            return self.send(200, json.dumps(body).encode(), "application/json")
        if path.startswith("/files/"):
            # Only serve files inside the DreamWorld folder (and never data/)
            rel = urllib.parse.unquote(path[len("/files/"):])
            f = (DW / rel).resolve()
            if DW not in f.parents or (DW / "data") in [f, *f.parents] or not f.is_file():
                return self.send(404, b"not found", "text/plain")
            ctype = {".jpg": "image/jpeg", ".mp4": "video/mp4", ".png": "image/png"}.get(f.suffix, "text/plain")
            if f.suffix == ".mp4":
                return self.send_video(f)
            return self.send(200, f.read_bytes(), ctype)
        self.send(404, b"not found", "text/plain")


class Server(http.server.ThreadingHTTPServer):
    # On Windows, reuse_address lets a SECOND server silently share the port (see playpen gotchas)
    allow_reuse_address = False


if __name__ == "__main__":
    # Default: this PC only. To watch from another device on your own private network, run a second copy with
    #   python dreamlab.py --host <this PC's private address> --port 8804
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    try:
        srv = Server((a.host, a.port), Handler)
    except OSError:
        sys.exit(f"Port {a.port} on {a.host} is busy. Is Dream Lab already running there?")
    print(f"Dream Lab: http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}")
    srv.serve_forever()
