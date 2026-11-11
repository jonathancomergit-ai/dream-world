"""Step 2: turn camera motion into WASD-style "key presses" (the action labeler).

Walking-tour videos have no controller input. To train an action-conditioned world
model we still need a "key" for every step, so we INVENT the keys by watching how the
camera moved from one frame to the next.

Usage (from the DreamWorld folder):
    data\\venv\\Scripts\\python prep\\label_actions.py                 # every finished city
    data\\venv\\Scripts\\python prep\\label_actions.py Stockholm       # just one
    data\\venv\\Scripts\\python prep\\label_actions.py --force         # redo even if done
    data\\venv\\Scripts\\python prep\\label_actions.py --test-clip prep\\actions_test\\x.mp4
        (a short test file: outputs go next to it in prep/actions_test/, not data/actions/)

What it writes per city:
    data/actions/<City>.npz         actions (int8), yaw, pitch, fwd, strafe (float32), cut (bool),
                                    thresholds (float32: turn, look, fwd, back, strafe)
    prep/preview/<City>_actions.mp4 2-minute check video with the action drawn on every frame
    prep/actions_report.md          a short stats block appended per city

-----------------------------------------------------------------------------------------
HOW IT WORKS (plain version)
-----------------------------------------------------------------------------------------
1. OPTICAL FLOW. For each pair of neighbouring frames (i, i+1) we ask OpenCV's DIS
   optical flow "where did every pixel go?". The answer is a little arrow (u, v) per
   pixel: u = how many pixels it moved right, v = how many pixels it moved DOWN
   (image rows count downward).

2. FIT A TINY CAMERA MODEL TO THE ARROWS. A moving camera makes very typical arrow
   patterns. We describe the whole flow field with just 4 numbers:
        u = tx + s*x - r*y
        v = ty + s*y + r*x
   where (x, y) is the pixel position measured from the image centre.
     tx = everything slides sideways      -> the camera TURNED (yaw)
     ty = everything slides up/down       -> the camera TILTED (pitch)
     s  = everything spreads out from the centre (zoom-like) -> we walked FORWARD
          (s < 0 means things shrink toward the centre -> walking backward)
     r  = everything spins around the centre -> roll (we ignore it, it is not a key)
   We solve for the 4 numbers with least squares, then throw away the arrows that
   disagree badly (pedestrians, cars, flicker) and solve again. That "outlier
   rejection" is why one passing bus doesn't fake a turn.

3. STRAFE (A/D) IS THE HARD ONE. Turning and side-stepping both make the picture slide
   sideways. The difference is PARALLAX: when you turn, near and far things slide by the
   same amount; when you side-step, near things (the street at the bottom of the frame)
   slide much more than far things (buildings in the middle). So strafe score =
   "how much more the lower band slid than the upper band". It is weak and noisy on
   one camera, so we only emit A/D when it is strong and there is no clear turn.

4. PIXELS TO DEGREES. We assume ~70 degrees horizontal field of view over 256 px, which
   means a focal length f = 128 / tan(35 deg) ~= 183 px. A slide of tx pixels is a turn
   of atan(tx / f) degrees. Signs: if the camera turns RIGHT, the world slides LEFT
   (tx < 0), so yaw = -atan(tx/f). If the camera tilts UP the world slides DOWN
   (ty > 0), so pitch = +atan(ty/f).

5. SMOOTH. Walking makes the camera bob a little every step, so we run a short median
   filter over time (a median ignores single-frame spikes but keeps real turns).

6. THRESHOLDS FROM THE DATA. Instead of guessing "a turn is > X degrees", we look at the
   city's own distribution (percentiles) and clamp to sensible ranges. See the
   constants below; the report records the numbers each city actually used.

7. CUTS. If two neighbouring frames are wildly different (editor cut, glitch, black
   frame) the "motion" is meaningless. We flag those steps so the trainer never makes a
   training clip across them. Careful: a fast pan ALSO makes neighbouring frames very
   different, but the flow explains it. So we "undo" the motion first (slide frame i+1
   back along the flow arrows) and only then compare to frame i. Whatever is still
   different is change the motion can't explain -> a real cut.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

# ----------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------
DW = Path(__file__).resolve().parent.parent          # the DreamWorld folder
CLEAN = DW / "data" / "clean"
ACTIONS = DW / "data" / "actions"
PREVIEW = DW / "prep" / "preview"
REPORT = DW / "prep" / "actions_report.md"
TEST_DIR = DW / "prep" / "actions_test"

# ----------------------------------------------------------------------------------------
# The fixed action vocabulary (other code depends on these numbers, never reorder)
# ----------------------------------------------------------------------------------------
STILL, FORWARD, BACK, STRAFE_L, STRAFE_R, TURN_L, TURN_R, LOOK_UP, LOOK_DOWN = range(9)
ACTION_NAMES = ["still", "W forward", "S back", "A strafe left", "D strafe right",
                "turn left", "turn right", "look up", "look down"]

# ----------------------------------------------------------------------------------------
# Camera + measurement settings
# ----------------------------------------------------------------------------------------
HFOV_DEG = 70.0                                   # assumed horizontal field of view
FOCAL_PX = 128.0 / math.tan(math.radians(HFOV_DEG / 2))   # ~183 px for a 256 px wide frame
BLOCK = 8            # flow is averaged in 8x8 pixel blocks (fewer, cleaner arrows to fit)
BORDER_X = 16        # ignore 16 px on the left/right edges (flow is unreliable at edges)
BORDER_TOP = 8       # ignore the top 8 px
BOTTOM_CUT = 0.22    # ignore the bottom 22% (watermarks, the walker's own shadow/feet)
FIT_ITERS = 3        # least-squares -> drop outliers -> refit, this many times
OUTLIER_K = 3.0      # an arrow is an outlier if its error > K * (robust typical error)
OUTLIER_MIN_PX = 0.25  # ... but never call errors below this many pixels outliers

# Smoothing windows (in steps of 0.1 s). Yaw is already smooth (gimbal), pitch and
# forward bob with every footstep, so they get a slightly longer median.
SMOOTH_YAW = 3
SMOOTH_PITCH = 5
SMOOTH_FWD = 5
SMOOTH_STRAFE = 5

# ----------------------------------------------------------------------------------------
# Threshold rules. Each threshold = a percentile of this city's own signal, clamped to a
# sensible range. Turning is the main control after walking, so we aim for roughly
# 15-20% turn labels; looks up/down are rarer. Units: degrees per step (0.1 s) for
# turn/look, "% zoom per step" for fwd, pixels of extra near-vs-far slide for strafe.
# ----------------------------------------------------------------------------------------
TURN_PCTL, TURN_MIN, TURN_MAX = 82.0, 0.8, 2.5     # |yaw| above its 82nd percentile = turn
LOOK_PCTL, LOOK_MIN, LOOK_MAX = 96.0, 0.8, 3.0     # |pitch| above its 96th percentile = look
# Forward: threshold = a fraction of the city's typical walking "zoom" (median of the
# positive fwd values), clamped. Normal walking measures ~2 %/step; standing still sits
# near 0 with some noise, so a third of the walking value splits the two well.
FWD_FRACTION, FWD_MIN, FWD_MAX = 0.35, 0.20, 1.50
BACK_FACTOR = 1.5    # walking backwards is rare, so demand 1.5x the forward threshold
STRAFE_PCTL, STRAFE_MIN, STRAFE_MAX = 99.0, 1.0, 3.0  # strafe is weak: only the top ~1%
#   (of the calm, non-turning steps)

# Cut detection uses the MOTION-COMPENSATED difference: frame i+1 warped back along the
# flow, compared with frame i (mean absolute grey-level difference, centre region).
CUT_MULT = 6.0       # cut if that leftover difference is > 6x the city's median ...
CUT_MIN_ABS = 30.0   # ... and > 30 grey levels (out of 255) in absolute terms
MIN_INLIERS = 0.25   # if < 25% of arrows agree with the camera model, the step is junk

# Check video
PREVIEW_START, PREVIEW_LEN = 3000, 1200            # frames 3000-4199 = 2 minutes at 10 fps
PREVIEW_W, PREVIEW_H = 512, 288

W, H = 256, 144      # clean video size (checked against ffprobe)


# ========================================================================================
# 1. Reading frames
# ========================================================================================
def probe(path):
    """Return (width, height, fps) of a video using ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def decode_gray_to_memmap(path, w, h, mm_path):
    """Decode every frame as greyscale into a file on disk (a numpy "memmap").

    Why a file? Worker processes on Windows can't share a Python array directly, but they
    can all open the same file read-only. 49k frames x 256x144 bytes ~= 1.8 GB.
    `-fps_mode passthrough` makes ffmpeg output exactly the frames in the file, no
    duplicates/drops, so frame indices match what the trainer will decode.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 64)
    fsize = w * h
    # Size the scratch file from the video's own length (+5% slack). The old fixed 60,000
    # crashed on Bangkok (2.9 h = 105,000 frames at 10 fps), 2026-10-19.
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                               check=True).stdout.strip())
    cap = int(dur * 10 * 1.05) + 1000                # clean videos are 10 fps
    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8, shape=(cap, h, w))
    n = 0
    while True:
        buf = proc.stdout.read(fsize * 256)        # read 256 frames at a time
        if not buf:
            break
        k = len(buf) // fsize
        if n + k > cap:
            raise RuntimeError(f"video has more frames than expected ({cap}); raise `cap`")
        mm[n:n + k] = np.frombuffer(buf[:k * fsize], np.uint8).reshape(k, h, w)
        n += k
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed decoding {path}")
    mm.flush()
    del mm
    return n


# ========================================================================================
# 2. Optical flow + the 4-number camera model (runs inside worker processes)
# ========================================================================================
_G = None      # each worker keeps its own handle to the frame file
_DIS = None    # ... and its own optical flow object
_GRID = None   # precomputed block-centre coordinates + design matrix


def _worker_init(mm_path, n):
    global _G, _DIS, _GRID
    cv2.setNumThreads(1)                           # one core per worker, the Pool does the rest
    _G = np.load(mm_path, mmap_mode="r")[:n]
    _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    _GRID = make_grid()


def make_grid():
    """Work out which region of the frame we use and where each 8x8 block centre sits.

    Returns the crop box, the (x, y) of every block centre measured from the image
    centre, and the "design matrix" A for the least-squares fit (see fit_camera).
    """
    y0 = BORDER_TOP
    y1 = int(H * (1 - BOTTOM_CUT)) // BLOCK * BLOCK          # 112 for H=144
    x0, x1 = BORDER_X, W - BORDER_X
    gh, gw = (y1 - y0) // BLOCK, (x1 - x0) // BLOCK
    y1, x1 = y0 + gh * BLOCK, x0 + gw * BLOCK
    ys = y0 + BLOCK / 2 + BLOCK * np.arange(gh) - H / 2      # block centres, from image centre
    xs = x0 + BLOCK / 2 + BLOCK * np.arange(gw) - W / 2
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    m = X.size
    # Stack the u-equations on top of the v-equations. Unknowns p = [tx, ty, s, r]:
    #   u = 1*tx + 0*ty + x*s - y*r
    #   v = 0*tx + 1*ty + y*s + x*r
    A = np.zeros((2 * m, 4))
    A[:m, 0] = 1; A[:m, 2] = X; A[:m, 3] = -Y
    A[m:, 1] = 1; A[m:, 2] = Y; A[m:, 3] = X
    # Strafe bands: the lower third of the used area (near ground) vs the upper half.
    low = Y > (ys[0] + (ys[-1] - ys[0]) * 0.66)
    high = Y < (ys[0] + (ys[-1] - ys[0]) * 0.50)
    return dict(box=(y0, y1, x0, x1), gw=gw, gh=gh, X=X, Y=Y, A=A, low=low, high=high)


def fit_camera(flow, grid):
    """Robustly fit u = tx + s*x - r*y, v = ty + s*y + r*x to one flow field.

    Returns (tx, ty, s, r, strafe_px, inlier_fraction).
    """
    y0, y1, x0, x1 = grid["box"]
    # Average the flow in 8x8 blocks: INTER_AREA resizing is exactly a block mean.
    f = cv2.resize(flow[y0:y1, x0:x1], (grid["gw"], grid["gh"]), interpolation=cv2.INTER_AREA)
    u = f[..., 0].ravel().astype(np.float64)
    v = f[..., 1].ravel().astype(np.float64)
    b = np.concatenate([u, v])
    A = grid["A"]
    m = u.size
    keep = np.ones(m, bool)                       # a block is kept or dropped as a whole
    p = np.zeros(4)
    for _ in range(FIT_ITERS):
        k2 = np.concatenate([keep, keep])
        p, *_ = np.linalg.lstsq(A[k2], b[k2], rcond=None)
        res = b - A @ p
        err = np.hypot(res[:m], res[m:])          # how far each block's arrow is from the model
        typical = 1.4826 * np.median(err[keep])   # robust "standard deviation" (MAD)
        keep = err < max(OUTLIER_K * typical, OUTLIER_MIN_PX)
        if keep.sum() < 12:                        # almost nothing agrees: give up on this step
            break
    tx, ty, s, r = p
    # Strafe/parallax: horizontal motion LEFT OVER after the camera model, near band minus
    # far band. Side-stepping right makes near things slide left more -> negative -> we flip
    # the sign so + means "camera moved right".
    ru = (b - A @ p)[:m]
    lo, hi = grid["low"] & keep, grid["high"] & keep
    if lo.sum() >= 4 and hi.sum() >= 4:
        strafe = -(np.median(ru[lo]) - np.median(ru[hi]))
    else:
        strafe = 0.0
    return tx, ty, s, r, strafe, keep.mean()


def warped_diff(prev, nxt, flow):
    """Mean |frame i - (frame i+1 slid back along the flow)| over the central region.

    For pixel (x, y) in frame i the flow says it went to (x+u, y+v) in frame i+1, so we
    fetch frame i+1 at that spot (cv2.remap does the fetching with interpolation).
    """
    hh, ww = prev.shape
    gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    back = cv2.remap(nxt, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    m = 16                                         # skip edges: content there may have left
    d = np.abs(back[m:-m, m:-m].astype(np.int16) - prev[m:-m, m:-m].astype(np.int16))
    return d.mean()


def _work(rng):
    """Process steps a..b-1 (step i = frame i -> frame i+1). Returns a (b-a, 7) array."""
    a, b = rng
    out = np.zeros((b - a, 7), np.float32)
    prev = np.asarray(_G[a])
    for j, i in enumerate(range(a, b)):
        nxt = np.asarray(_G[i + 1])
        flow = _DIS.calc(prev, nxt, None)
        tx, ty, s, r, strafe, inl = fit_camera(flow, _GRID)
        # Photometric change for cut detection, AFTER undoing the motion (see warped_diff).
        diff = warped_diff(prev, nxt, flow)
        out[j] = (tx, ty, s, r, strafe, inl, diff)
        prev = nxt
    return out


def measure_motion(mm_path, n, workers):
    """Run flow + fit over all n-1 steps in parallel chunks."""
    steps = n - 1
    chunk = 500
    ranges = [(a, min(a + chunk, steps)) for a in range(0, steps, chunk)]
    with Pool(workers, initializer=_worker_init, initargs=(mm_path, n)) as pool:
        parts = pool.map(_work, ranges, chunksize=1)
    return np.concatenate(parts, axis=0)


# ========================================================================================
# 3. Turning raw measurements into continuous signals, thresholds and labels
# ========================================================================================
def to_signals(raw):
    """Pixels -> degrees and a forward score, then smooth over time."""
    tx, ty, s, r, strafe, inl, diff = raw.T
    # Strafe clean-up: a pure turn should show NO near-vs-far difference, but in practice
    # the flow is a bit less accurate during fast turns (blur, big shifts), so the strafe
    # score leaks some of the yaw. We measure that leak over the whole city with a robust
    # straight-line fit (strafe ~ beta * tx) and subtract it.
    beta = robust_slope(tx, strafe)
    strafe = strafe - beta * tx
    yaw = -np.degrees(np.arctan(tx / FOCAL_PX))     # + = turned right
    pitch = np.degrees(np.arctan(ty / FOCAL_PX))    # + = tilted up
    fwd = s * 100.0                                 # % the picture "zoomed in" this step
    # Median filters: a footstep spike lasting 1-2 frames vanishes, a real turn stays.
    yaw = median_filter(yaw, SMOOTH_YAW, mode="nearest")
    pitch = median_filter(pitch, SMOOTH_PITCH, mode="nearest")
    fwd = median_filter(fwd, SMOOTH_FWD, mode="nearest")
    strafe = median_filter(strafe, SMOOTH_STRAFE, mode="nearest")
    return yaw, pitch, fwd, strafe, inl, diff


def robust_slope(x, y, iters=5):
    """Least-squares slope of y ~ beta*x that ignores outliers (refit on points within
    3 robust standard deviations of the line)."""
    keep = np.isfinite(x) & np.isfinite(y)
    beta = 0.0
    for _ in range(iters):
        xx, yy = x[keep], y[keep]
        if xx.size < 10 or np.dot(xx, xx) == 0:
            return 0.0
        beta = float(np.dot(xx, yy) / np.dot(xx, xx))
        res = y - beta * x
        sd = 1.4826 * np.median(np.abs(res[keep])) + 1e-6
        keep = np.abs(res) < 3 * sd
    return beta


def find_cuts(diff, inl):
    """A step is a cut if the frames differ hugely, or the camera model fits almost nothing."""
    big = diff > max(CUT_MULT * np.median(diff), CUT_MIN_ABS)
    junk = inl < MIN_INLIERS
    return big | junk


def choose_thresholds(yaw, pitch, fwd, strafe, ok):
    """Pick thresholds from this city's own distribution (only non-cut steps count)."""
    clamp = lambda x, lo, hi: float(min(max(x, lo), hi))
    turn = clamp(np.percentile(np.abs(yaw[ok]), TURN_PCTL), TURN_MIN, TURN_MAX)
    look = clamp(np.percentile(np.abs(pitch[ok]), LOOK_PCTL), LOOK_MIN, LOOK_MAX)
    walking_speed = np.median(fwd[ok & (fwd > 0)]) if (ok & (fwd > 0)).any() else 0.3
    f = clamp(FWD_FRACTION * walking_speed, FWD_MIN, FWD_MAX)
    back = f * BACK_FACTOR
    # Strafe is judged only on "calm" steps (hardly turning), the same steps where label()
    # allows A/D, so leftover turn noise doesn't inflate the percentile.
    calm = ok & (np.abs(yaw) < 0.5 * turn)
    st = clamp(np.percentile(np.abs(strafe[calm]), STRAFE_PCTL) if calm.any() else STRAFE_MAX,
               STRAFE_MIN, STRAFE_MAX)
    return dict(turn=turn, look=look, fwd=f, back=back, strafe=st)


def label(yaw, pitch, fwd, strafe, th):
    """One label per step. Priority: clear rotation > translation > still.

    Each signal is divided by its threshold, so "1.0" means "just counts". Among the
    rotations the stronger one (turn vs look) wins; same for translations.
    """
    n = yaw.size
    act = np.full(n, STILL, np.int8)
    ry = np.abs(yaw) / th["turn"]
    rp = np.abs(pitch) / th["look"]
    # --- translations first (they get overwritten by rotations below) ---
    rf = np.where(fwd > 0, fwd / th["fwd"], -fwd / th["back"])
    rs = np.abs(strafe) / th["strafe"]
    rs[ry > 0.5] = 0          # side-step only counts when the camera is hardly turning
    trans = (rf >= 1) | (rs >= 1)
    use_strafe = trans & (rs > rf)
    act[trans & ~use_strafe & (fwd > 0)] = FORWARD
    act[trans & ~use_strafe & (fwd <= 0)] = BACK
    act[use_strafe & (strafe < 0)] = STRAFE_L
    act[use_strafe & (strafe > 0)] = STRAFE_R
    # --- rotations win over translations ---
    rot = (ry >= 1) | (rp >= 1)
    is_turn = rot & (ry >= rp)
    is_look = rot & (rp > ry)
    act[is_turn & (yaw < 0)] = TURN_L
    act[is_turn & (yaw > 0)] = TURN_R
    act[is_look & (pitch > 0)] = LOOK_UP
    act[is_look & (pitch < 0)] = LOOK_DOWN
    return act


# ========================================================================================
# 4. Check video
# ========================================================================================
GREEN, ORANGE, CYAN, GREY, RED, WHITE = (80, 220, 80), (0, 160, 255), (255, 220, 0), \
    (170, 170, 170), (60, 60, 255), (255, 255, 255)   # BGR colours for OpenCV


def draw_action(img, a, yaw, pitch, fwd, idx, cut):
    """Draw an arrow + label for action `a` on a 512x288 BGR frame."""
    cx, cy = PREVIEW_W // 2, PREVIEW_H // 2 + 40
    L = 55
    if a == FORWARD:
        cv2.arrowedLine(img, (cx, cy + L // 2), (cx, cy - L), GREEN, 5, tipLength=0.35)
    elif a == BACK:
        cv2.arrowedLine(img, (cx, cy - L // 2), (cx, cy + L), GREEN, 5, tipLength=0.35)
    elif a in (STRAFE_L, STRAFE_R):
        d = -1 if a == STRAFE_L else 1
        cv2.arrowedLine(img, (cx - d * L // 2, cy + 30), (cx + d * L, cy + 30), GREEN, 5,
                        tipLength=0.35)
    elif a in (TURN_L, TURN_R):
        # a curved arrow: an arc across the top with an arrow head at the end
        d = -1 if a == TURN_L else 1
        cv2.ellipse(img, (cx, cy + 10), (70, 40), 0, 200, 340, ORANGE, 5)
        tip_x = cx + d * 66
        cv2.arrowedLine(img, (tip_x - d * 10, cy - 12), (tip_x + d * 8, cy + 6), ORANGE, 5,
                        tipLength=1.2)
    elif a in (LOOK_UP, LOOK_DOWN):
        d = -1 if a == LOOK_UP else 1
        cv2.arrowedLine(img, (cx + 60, cy - d * L // 2), (cx + 60, cy + d * L), CYAN, 5,
                        tipLength=0.35)
        cv2.arrowedLine(img, (cx - 60, cy - d * L // 2), (cx - 60, cy + d * L), CYAN, 5,
                        tipLength=0.35)
    else:
        cv2.circle(img, (cx, cy), 14, GREY, 4)
    col = {FORWARD: GREEN, BACK: GREEN, STRAFE_L: GREEN, STRAFE_R: GREEN, TURN_L: ORANGE,
           TURN_R: ORANGE, LOOK_UP: CYAN, LOOK_DOWN: CYAN}.get(int(a), GREY)
    text = ACTION_NAMES[a]
    cv2.rectangle(img, (6, 6), (14 + 16 * len(text), 44), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    info = f"yaw {yaw:+5.2f}  pitch {pitch:+5.2f}  fwd {fwd:+5.2f}  #{idx}"
    cv2.rectangle(img, (6, PREVIEW_H - 26), (360, PREVIEW_H - 6), (0, 0, 0), -1)
    cv2.putText(img, info, (10, PREVIEW_H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1,
                cv2.LINE_AA)
    # little yaw bar under the label: centre = 0, full width = +-4 deg/step
    bx, by, bw = 400, PREVIEW_H - 16, 100
    cv2.line(img, (bx - bw // 2, by), (bx + bw // 2, by), GREY, 1)
    cv2.line(img, (bx, by - 6), (bx, by + 6), GREY, 1)
    px = int(bx + np.clip(yaw / 4.0, -1, 1) * bw / 2)
    cv2.circle(img, (px, by), 5, ORANGE, -1)
    if cut:
        cv2.putText(img, "CUT", (PREVIEW_W - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3,
                    cv2.LINE_AA)


def make_preview(path, out_path, n, actions, yaw, pitch, fwd, cut):
    start = PREVIEW_START if n >= PREVIEW_START + PREVIEW_LEN else max(0, n - PREVIEW_LEN)
    count = min(PREVIEW_LEN, n - start)
    # Decode exactly frames start..start+count-1 in colour (select by frame number n).
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an",
         "-vf", f"select='between(n\\,{start}\\,{start + count - 1})'",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{PREVIEW_W}x{PREVIEW_H}", "-r", "10", "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    fsize = W * H * 3
    for k in range(count):
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        i = start + k
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        img = cv2.resize(img, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_LINEAR)
        draw_action(img, int(actions[i]), yaw[i], pitch[i], fwd[i], i, bool(cut[i]))
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    dec.stdout.close()
    dec.wait()
    return start, count


# ========================================================================================
# 5. Per-city driver
# ========================================================================================
def process(city, path, out_npz, out_preview, workers):
    t0 = time.time()
    w, h, fps = probe(path)
    if (w, h) != (W, H):
        raise RuntimeError(f"{path} is {w}x{h}, expected {W}x{H}")
    tmpdir = tempfile.mkdtemp(prefix="dw_actions_")
    mm_path = os.path.join(tmpdir, "gray.npy")
    try:
        n = decode_gray_to_memmap(path, w, h, mm_path)
        t_dec = time.time() - t0
        print(f"[{city}] decoded {n} frames in {t_dec:.0f}s; measuring motion "
              f"with {workers} workers...")
        raw = measure_motion(mm_path, n, workers)
    finally:
        try:
            os.remove(mm_path)
            os.rmdir(tmpdir)
        except OSError:
            pass
    t_flow = time.time() - t0 - t_dec

    yaw, pitch, fwd, strafe, inl, diff = to_signals(raw)
    cut = find_cuts(diff, inl)
    th = choose_thresholds(yaw, pitch, fwd, strafe, ~cut)
    act = label(yaw, pitch, fwd, strafe, th)

    # Stretch every array from n-1 steps to n entries: the last frame has no "next"
    # frame, so it copies the step before it (as the contract says).
    pad = lambda x: np.append(x, x[-1])
    actions = pad(act).astype(np.int8)
    yaw_, pitch_, fwd_, strafe_ = (pad(x).astype(np.float32) for x in (yaw, pitch, fwd, strafe))
    cut_ = pad(cut).astype(bool)
    assert actions.size == n

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, actions=actions, yaw=yaw_, pitch=pitch_, fwd=fwd_, strafe=strafe_,
             cut=cut_, thresholds=np.array([th["turn"], th["look"], th["fwd"], th["back"],
                                            th["strafe"]], np.float32))
    out_preview.parent.mkdir(parents=True, exist_ok=True)
    pstart, pcount = make_preview(path, out_preview, n, actions, yaw_, pitch_, fwd_, cut_)
    total = time.time() - t0

    counts = np.bincount(actions, minlength=9)
    pct = 100.0 * counts / n
    print(f"[{city}] {n} frames, {total:.0f}s total (decode {t_dec:.0f}s, flow {t_flow:.0f}s)")
    print(f"[{city}] thresholds: " + ", ".join(f"{k}={v:.2f}" for k, v in th.items()))
    for a in range(9):
        print(f"    {a} {ACTION_NAMES[a]:<15} {pct[a]:5.1f}%")
    print(f"    cuts: {cut_.sum()}   median inliers {np.median(inl):.2f}")
    return dict(n=n, fps=fps, pct=pct, th=th, cuts=int(cut_.sum()), secs=total,
                yaw=yaw, pitch=pitch, fwd=fwd, ok=~cut, preview=(pstart, pcount),
                inl=float(np.median(inl)))


def append_report(city, r, src):
    yaw, pitch, fwd, ok = r["yaw"], r["pitch"], r["fwd"], r["ok"]
    th = r["th"]
    lines = [f"\n## {city}  ({time.strftime('%Y-%m-%d %H:%M')})\n",
             f"- Source `{src}`: {r['n']} frames at {r['fps']:.0f} fps, labelled in "
             f"{r['secs']:.0f} s. Cuts flagged: {r['cuts']}. Median inlier share of the "
             f"camera fit: {r['inl']:.2f}.",
             f"- Signals (non-cut steps): |yaw| median {np.median(np.abs(yaw[ok])):.2f}, "
             f"p90 {np.percentile(np.abs(yaw[ok]), 90):.2f} deg/step; |pitch| median "
             f"{np.median(np.abs(pitch[ok])):.2f} deg/step; fwd median {np.median(fwd[ok]):.2f} "
             f"%zoom/step.",
             f"- Thresholds: turn |yaw| > {th['turn']:.2f} deg/step (p{TURN_PCTL:.0f} of |yaw|, "
             f"clamped {TURN_MIN}-{TURN_MAX}); look |pitch| > {th['look']:.2f} deg/step "
             f"(p{LOOK_PCTL:.0f}, clamped {LOOK_MIN}-{LOOK_MAX}); forward fwd > {th['fwd']:.2f} "
             f"({FWD_FRACTION} x median walking fwd, clamped {FWD_MIN}-{FWD_MAX}); back fwd < "
             f"-{th['back']:.2f} ({BACK_FACTOR} x forward); strafe |parallax| > "
             f"{th['strafe']:.2f} px (p{STRAFE_PCTL:.0f}, clamped {STRAFE_MIN}-{STRAFE_MAX}).",
             f"- Check video: frames {r['preview'][0]}-{r['preview'][0] + r['preview'][1] - 1}.",
             "",
             "| action | % of steps |", "|---|---|"]
    for a in range(9):
        lines.append(f"| {a} {ACTION_NAMES[a]} | {r['pct'][a]:.1f} |")
    turns = r['pct'][TURN_L] + r['pct'][TURN_R]
    lines.append(f"\nTurns total: {turns:.1f}%.\n")
    new = not REPORT.exists()
    with open(REPORT, "a", encoding="utf-8") as fh:
        if new:
            fh.write("# Action labels report\n\nWritten by `prep/label_actions.py` "
                     "(one block per run).\n")
        fh.write("\n".join(lines))


def finished_cities():
    return sorted(p.stem for p in CLEAN.glob("*.mp4")
                  if not p.stem.endswith("_small") and ".tmp" not in p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--force", action="store_true", help="redo cities that already have output")
    ap.add_argument("--test-clip", help="label a short test mp4; outputs go to prep/actions_test/")
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 4) - 4)))
    args = ap.parse_args()

    if args.test_clip:
        p = Path(args.test_clip).resolve()
        name = p.stem
        r = process(name, p, TEST_DIR / f"{name}.npz", TEST_DIR / f"{name}_actions.mp4",
                    args.workers)
        print(f"test outputs in {TEST_DIR}")
        return

    cities = args.cities or finished_cities()
    if not cities:
        print("No finished clean videos in data/clean yet.")
        return
    for city in cities:
        src = CLEAN / f"{city}.mp4"
        out_npz = ACTIONS / f"{city}.npz"
        out_prev = PREVIEW / f"{city}_actions.mp4"
        if not src.exists():
            print(f"[{city}] no {src.name} yet, skipping")
            continue
        if out_npz.exists() and not args.force:
            print(f"[{city}] already labelled ({out_npz.name}); use --force to redo")
            continue
        r = process(city, src, out_npz, out_prev, args.workers)
        append_report(city, r, f"data/clean/{city}.mp4")


if __name__ == "__main__":
    main()
