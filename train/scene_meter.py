"""Scene meter: is a dreamed frame still "a photo of a place", or has it become texture? (2026-10-19)

Why: every meter so far checks ONE symptom (sharpness -> mush, grain -> static, vertical lines ->
curtains, horizontal lines -> ripples) and every time the dream found a NEW way to fail that the
meter couldn't see (Heun's shard-collage scored a "perfect" 20 s). This meter checks the general
property instead.

The idea, in plain words: any photo can be split into "how much big-shape energy, how much
medium, how much fine detail", in each direction (its power spectrum, via a Fourier transform).
Real street photos have a very typical recipe: lots of big shapes, smoothly less of each finer
level, roughly the same in every direction. Every collapse we've seen breaks that recipe:
  mush / beige -> almost no fine detail
  static       -> far too much of the finest detail
  curtains     -> one direction (left-right detail) dominates
  ripples      -> the other direction dominates
  shard-collage -> too much mid-to-fine detail everywhere
So: summarise each frame as 6 rings (coarse..fine) x 4 directions = 24 numbers (log energy, after
removing overall brightness/contrast), and measure how far a dream frame is from REAL footage's
typical values, in units of real footage's own spread (a z-score distance).

scene_distance(frame) ~ 1 for real frames, grows as the frame stops looking like a photo.
scene_time(walk) = seconds until the 1 s average distance stays above THRESH (calibrated on real
walks so real footage almost never triggers it).
"""
import numpy as np

RINGS = 6
DIRS = 4


def _bins(h, w):
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    ang = (np.degrees(np.arctan2(fy, fx)) + 180.0) % 180.0          # direction of the WAVE, 0..180
    # log-spaced rings between the lowest non-zero and the Nyquist frequency
    edges = np.geomspace(1.0 / max(h, w), 0.5 * np.sqrt(2) + 1e-6, RINGS + 1)
    ring = np.digitize(r, edges) - 1                                  # -1 = DC
    d = (np.floor((ang + 22.5) / 45.0) % DIRS).astype(int)            # 0 = horiz waves, 2 = vertical...
    return ring, d


_cache = {}


def features(frames_u8):
    """[T,H,W,3] uint8 -> [T, RINGS*DIRS] log band energies (brightness/contrast normalised)."""
    T, H, W, _ = frames_u8.shape
    if (H, W) not in _cache:
        _cache[(H, W)] = _bins(H, W)
    ring, d = _cache[(H, W)]
    g = frames_u8.astype(np.float32).mean(-1)
    g = g - g.mean(axis=(1, 2), keepdims=True)
    g = g / (g.std(axis=(1, 2), keepdims=True) + 1e-3)               # contrast-free: colour is another meter's job
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]             # soften the frame edges (no fake edge lines)
    P = np.abs(np.fft.fft2(g * win)) ** 2
    out = np.zeros((T, RINGS * DIRS), np.float32)
    for k in range(RINGS):
        for j in range(DIRS):
            m = (ring == k) & (d == j)
            if m.any():
                out[:, k * DIRS + j] = np.log(P[:, m].mean(axis=1) + 1e-6)
    return out


class SceneMeter:
    """Fit on real footage, then score dreams."""

    def fit(self, real_frames_u8):
        f = features(real_frames_u8)
        self.mu = f.mean(0)
        self.sd = f.std(0) + 1e-3
        return self

    def distance(self, frames_u8):
        z = (features(frames_u8) - self.mu) / self.sd
        return np.sqrt((z ** 2).mean(1))                              # RMS z-score: ~1 for typical real frames

    def scene_time(self, frames_u8, thresh, fps=10):
        """Seconds until the 1 s moving average of distance goes above thresh (T/fps if never)."""
        d = self.distance(frames_u8)
        ma = np.convolve(d, np.ones(fps) / fps, mode="valid")
        bad = np.flatnonzero(ma > thresh)
        return (len(frames_u8) / fps) if len(bad) == 0 else round((bad[0] + fps) / fps, 1), d
