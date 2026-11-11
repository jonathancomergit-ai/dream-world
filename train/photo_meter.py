"""Photo meter: how much does a frame look like a real street photo, in shape AND colour? (2026-10-21)

Why: the older meters each check one symptom against the 8 frames the dream started from, and the first
one to trip ends the dream's "clean time". Checked on REAL video (train/_exp/v9_judge.py) they trip far too
often: real footage scored only ~25-29 s of 40, and 2-3 s on two walks (Kuala Lumpur, Singapore). And they
are colour-blind, so neon colour-bar glitch beat v8 by 2x (2026-10-21).

The idea, in plain words: describe every frame with a few dozen numbers that real street photos keep within a
typical range, then measure how far a frame is from that range, in units of real footage's own spread:
  - grey shape recipe: scene_meter's 6 rings (coarse..fine) x 4 directions of the brightness spectrum (24)
  - colour recipe: the same 6 rings for each colour channel Cb, Cr, all directions together (12), in log
    energy RELATIVE to the brightness at the same ring (real photos keep colour smoother than brightness;
    colour-bar glitch has fine colour detail; beige mush has none)
  - overall contrast and colourfulness (2)
Fitted on thousands of real frames from every walking city (not the held-out tails), with the full
covariance (the numbers move together; Mahalanobis distance). distance ~ 1 for a typical real frame.
  photo_share(walk)  = share of a dream's frames whose 1 s average distance is inside the range 95% of
                       real frames' 1 s averages stay in (real video ~0.95 by construction)
  photo_time(walk)   = seconds until the 1 s average leaves that range for 3 s in a row
Not a judge of "is it the same place" (that's the turn test's job) - only "does it still look like a photo".
"""
import numpy as np

from scene_meter import RINGS, _bins, features as grey_features

FIT_FILE = "photo_meter_fit.npz"          # in train/_exp/


def _colour_rings(frames_u8):
    """[T, 6 + 6 + 2]: Cb and Cr ring log-energies relative to brightness's, + log contrast + log colourfulness."""
    T, H, W, _ = frames_u8.shape
    ring, _ = _bins(H, W)
    f = frames_u8.astype(np.float32)
    y = f @ np.array([0.299, 0.587, 0.114], np.float32)
    cb = f @ np.array([-0.1687, -0.3313, 0.5], np.float32)
    cr = f @ np.array([0.5, -0.4187, -0.0813], np.float32)
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    spec = lambda c: np.abs(np.fft.fft2((c - c.mean(axis=(1, 2), keepdims=True)) * win)) ** 2
    Py, Pb, Pr = spec(y), spec(cb), spec(cr)
    out = np.zeros((T, 2 * RINGS + 2), np.float32)
    for k in range(RINGS):
        m = ring == k
        ey = np.log(Py[:, m].mean(1) + 1.0)
        out[:, k] = np.log(Pb[:, m].mean(1) + 1.0) - ey
        out[:, RINGS + k] = np.log(Pr[:, m].mean(1) + 1.0) - ey
    out[:, -2] = np.log(y.std(axis=(1, 2)) + 1.0)
    out[:, -1] = np.log(np.sqrt(cb.std(axis=(1, 2)) ** 2 + cr.std(axis=(1, 2)) ** 2) + 1.0)
    return out


def features(frames_u8, chunk=256):
    """[T, H, W, 3] uint8 -> [T, 38]."""
    outs = []
    for i in range(0, len(frames_u8), chunk):
        u = frames_u8[i:i + chunk]
        outs.append(np.concatenate([grey_features(u), _colour_rings(u)], 1))
    return np.concatenate(outs, 0)


class PhotoMeter:
    def fit(self, real_u8, reg=1e-3):
        f = features(real_u8).astype(np.float64)
        self.mu = f.mean(0)
        cov = np.cov(f, rowvar=False)
        cov += reg * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.icov = np.linalg.inv(cov)
        self.level = None
        return self

    def save(self, path):
        np.savez(path, mu=self.mu, icov=self.icov, level=np.float64(self.level if self.level is not None else np.nan))

    def load(self, path):
        z = np.load(path)
        self.mu, self.icov, self.level = z["mu"], z["icov"], float(z["level"])
        return self

    def distance(self, frames_u8):
        """[T] Mahalanobis distance / sqrt(n features): ~1 for a typical real frame."""
        d = features(frames_u8).astype(np.float64) - self.mu
        return np.sqrt(np.einsum("ti,ij,tj->t", d, self.icov, d) / len(self.mu))

    def report(self, frames_u8, fps=10, hold_s=3):
        """photo_share, photo_time (s, T/fps if it never leaves the real range for hold_s seconds) and the curve."""
        dist = self.distance(frames_u8)
        ma = np.convolve(dist, np.ones(fps) / fps, mode="valid")
        bad = ma > self.level
        share = float(1 - bad.mean())
        run, t_out = 0, None
        for i, b in enumerate(bad):
            run = run + 1 if b else 0
            if run >= hold_s * fps:
                t_out = round((i - run + 1 + fps) / fps, 1)
                break
        return {"photo_share": round(share, 3), "photo_time": t_out if t_out is not None else len(frames_u8) / fps,
                "dist": dist}


class DriftMeter:
    """How far has each frame drifted from the REAL frames its walk started from, compared with how far real
    walks drift on their own? (2026-10-21, after the global PhotoMeter missed colour glitch: real footage
    includes neon night markets, so glitch looked 'normal' against all cities at once.)
      drift(frame) = Mahalanobis distance of features(frame) - mean features(48 real start frames), using the
                     covariance of that same difference measured on real 40 s walks (sqrt(d2 / n) ~ 1 for real)
    Real walks wander into new places too, so the meter only flags drift bigger or stranger than real walking."""

    def fit(self, diffs, reg=1e-3):
        self.mu = diffs.mean(0)
        cov = np.cov(diffs, rowvar=False)
        cov += reg * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.icov = np.linalg.inv(cov)
        self.level = None
        return self

    def save(self, path):
        np.savez(path, mu=self.mu, icov=self.icov, level=np.float64(self.level))

    def load(self, path):
        z = np.load(path)
        self.mu, self.icov, self.level = z["mu"], z["icov"], float(z["level"])
        return self

    def distance(self, frames_u8, start_u8):
        d = features(frames_u8).astype(np.float64) - features(start_u8).astype(np.float64).mean(0) - self.mu
        return np.sqrt(np.einsum("ti,ij,tj->t", d, self.icov, d) / len(self.mu))

    def report(self, frames_u8, start_u8, fps=10, hold_s=3):
        dist = self.distance(frames_u8, start_u8)
        ma = np.convolve(dist, np.ones(fps) / fps, mode="valid")
        bad = ma > self.level
        run, t_out = 0, None
        for i, b in enumerate(bad):
            run = run + 1 if b else 0
            if run >= hold_s * fps:
                t_out = round((i - run + 1 + fps) / fps, 1)
                break
        return {"drift_share": round(float(1 - bad.mean()), 3),
                "drift_time": t_out if t_out is not None else len(frames_u8) / fps, "dist": dist}
