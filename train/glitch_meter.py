"""Glitch meter: has the dream's COLOUR turned into sharp-edged confetti / colour bars? (2026-10-21)

Why: every older meter is colour-blind (sharpness, grain, streak ratio and the scene meter all look at
grey levels only). On 2026-10-21 inference tweaks that turned dreams into neon colour-bar glitch
"beat" v8 by 2x on clean2 (24 s vs 11 s) while visibly being worse: glitch is full of edges, so the
melt meter called it sharp.

The idea, in plain words: in real photos (and real video files) COLOUR changes smoothly; the fine
detail lives in the BRIGHTNESS. A glitch frame has sharp colour edges everywhere.
  colour_detail(frame) = mean |Laplacian| of the colour channels (Cb, Cr of YCbCr) at full resolution
  glitch(t) = log(colour_detail(dream t) + 0.5) - log(mean colour_detail(real start frames) + 0.5)
Compared with the real frames the dream started from, because night neon streets are naturally more
colourful than daytime ones. NOT divided by brightness detail: a first version was, and then plain mush
(low brightness detail) and smooth walls tripped it too (13.7% of real walks; this one: 5%).
GLITCH TIME = when its 1 s average first goes above GLITCH_LEVEL.
Calibrated 2026-10-21 (train/_exp/v9_glitch_calib2.py, version "A"): on 300 real 40 s walks vs their own
start frames, 5% ever trip 0.83 (1%: 1.12). On tonight's dreams the order matches the eye: epsilon
scaling 1.02 glitches at 1-2 s, 1.01 at 3-4 s, 1.005 at 7-8 s, v8 at ~9 s (Amsterdam) / never (the Bangkok
night walk, which goes to dark MUSH instead: the melt meter's job). Mush-type failures never trip it.
"""
import cv2
import numpy as np

GLITCH_LEVEL = 0.83      # 5% of real 40 s walks ever trip it (train/_exp/v9_glitch_calib2.py)


def _fine(ch):
    return np.array([np.abs(cv2.Laplacian(np.ascontiguousarray(c), cv2.CV_32F, ksize=1)).mean() for c in ch])


def colour_detail(frames_u8):
    """[T, H, W, 3] uint8 RGB -> [T] fine detail in the colour channels (mean |Laplacian| of Cb and Cr)."""
    f = frames_u8.astype(np.float32)
    cb = f @ np.array([-0.1687, -0.3313, 0.5], np.float32)
    cr = f @ np.array([0.5, -0.4187, -0.0813], np.float32)
    return 0.5 * (_fine(cb) + _fine(cr))


def glitch_curve(dream_u8, start_u8):
    """Per-frame glitch reading vs the real start frames [T] (0 = as colourful-in-detail as the start)."""
    return np.log(colour_detail(dream_u8) + 0.5) - np.log(float(colour_detail(start_u8).mean()) + 0.5)


def glitch_time(dream_u8, start_u8, level=GLITCH_LEVEL, fps=10):
    """Seconds until the 1 s average glitch reading first goes above level (None = never)."""
    g = glitch_curve(dream_u8, start_u8)
    ma = np.convolve(g, np.ones(fps) / fps, mode="valid")
    bad = np.flatnonzero(ma > level)
    return None if not len(bad) else round((bad[0] + fps) / fps, 1)
