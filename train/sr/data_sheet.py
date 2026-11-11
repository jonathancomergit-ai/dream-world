"""Picture of the training pairs: cache frame (bicubic 2x) | v9 wreck of it (bicubic 2x, sigma shown) | HR target.

    data\\venv\\Scripts\\python.exe train\\sr\\data_sheet.py      # -> train/sr/out/data_pairs.jpg
"""
import pathlib
import sys

import cv2
import numpy as np

SR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SR.parent))
sys.path.insert(0, str(SR))
from train_sr import HRD, open_data  # noqa: E402

info, hr, fm, wr = open_data()
cidx = np.load(HRD / "cache_idx.npy")
split = np.load(HRD / "split.npy")
sig = np.load(HRD / "wreck_sigma.npy")
rng = np.random.default_rng(3)
rows = []
cand = np.flatnonzero((split == 0) & np.isfinite(sig))
order = np.argsort(sig[cand])
picks = cand[order[np.linspace(0, len(order) - 1, 6).astype(int)]]        # from gentlest to hardest wreck
F = cv2.FONT_HERSHEY_SIMPLEX


def lab(im, t):
    im = np.ascontiguousarray(im[:, :, ::-1]).copy()
    cv2.putText(im, t, (5, 18), F, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, t, (5, 18), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


for r in picks:
    up = lambda a: cv2.resize(np.asarray(a), (336, 192), interpolation=cv2.INTER_CUBIC)
    rows.append(np.concatenate([lab(up(fm[cidx[r]]), f"cache frame {cidx[r]} (bicubic)"),
                                lab(up(wr[r]), f"v9 wreck sigma {sig[r]:.2f} (bicubic)"),
                                lab(np.asarray(hr[r]), "HR target (face-blurred)")], 1))
out = np.concatenate(rows, 0)
out = cv2.resize(out, (out.shape[1] * 2, out.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(SR / "out" / "data_pairs.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved", SR / "out" / "data_pairs.jpg")
