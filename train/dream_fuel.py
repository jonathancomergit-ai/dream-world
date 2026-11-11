"""Dream fuel (2026-10-22): keep the dream sharp by slipping REAL frames into its starting ghost.

Why: everything since the Oct 20 breakthrough says the model FOLLOWS ITS GHOST (the faint picture each frame is
dreamed from). Give it a sharp ghost and it paints a sharp frame. So when the dream starts to melt (or every few
seconds), look up a real street frame whose big colour blocks match the current dream, and blend it into the ghost
for a second or two: the dream morphs into a new, sharp, real-looking place instead of sliding into mush.

  FuelLibrary   ~20k real frames from the TRAINING part of every walking city (never the held-out test walks),
                on the GPU, each with a tiny 'layout' descriptor (6 x 10 blocks of colour = 180 numbers).
  nearest(x)    the library frame whose layout is closest to x (what the dream looks like right now), skipping
                ones used recently, so the morph is gentle and the world doesn't loop.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data_v3 import to_model_space


def layout_desc(x):
    """x [B, 3, H, W] model space -> [B, 180] block colours (6 x 10), mean-free per channel x 0.5 + raw x 0.5."""
    d = F.adaptive_avg_pool2d(x.float(), (6, 10))
    m = d.mean(dim=(2, 3), keepdim=True)
    return torch.cat([(d - m).flatten(1), 0.5 * m.flatten(1)], 1)


class FuelLibrary:
    def __init__(self, index, frames_mm, device, n=20000, seed=0, hold_frac=0.03, min_hold=608, skip=("Wildlife",)):
        rng = np.random.default_rng(seed)
        cities = [c for c in index["cities"] if c["name"] not in skip]
        idx = []
        for c in cities:
            h0 = c["end"] - min(max(int(hold_frac * c["frames"]), min_hold), c["frames"] // 3)   # held-out tail
            idx.append(rng.choice(np.arange(c["start"], h0), n // len(cities), replace=False))
        self.idx = np.sort(np.concatenate(idx))
        self.u8 = torch.from_numpy(np.stack([frames_mm[i] for i in self.idx])).to(device)     # [N, H, W, 3]
        self.desc = torch.cat([layout_desc(to_model_space(self.u8[i:i + 2048])) for i in range(0, len(self.u8), 2048)])
        self.dev = device
        self.recent = []

    def frame(self, i):
        return to_model_space(self.u8[i:i + 1])[0].float()             # [3, H, W]

    def nearest(self, x, avoid=200):
        """x [B, 3, H, W] -> list of library indices (closest layout, not among the last `avoid` picks)."""
        d = torch.cdist(layout_desc(x), self.desc)                        # [B, N]
        if self.recent:
            d[:, torch.tensor(self.recent[-avoid:], device=self.dev)] = float("inf")
        out = d.argmin(1).tolist()
        self.recent += out
        return out


class EdgeFuel:
    """EDGE FUEL (2026-10-22): when a turn reveals a new strip at the edge of the view, fill it from a matching REAL
    frame sliding in from that side, as if the real world continued past the edge (instead of stretching the edge
    column, which built horizontal streaks). A new real frame is picked when the current one has fully slid in or
    the turn changes direction.
    Tested on the v9 pick (train/_exp/v10_edgefuel.py, 10 held-out walks x 2): full-circle spin melt 0.42 -> 0.22,
    streak ratio 0.70 -> 1.01 (= like real video), return to the start view 0.93 -> 0.94; 40 s walks 0.43 -> 0.40.
    (Blending real frames into the whole ghost instead failed: double exposure -> soft, washed-out mush.)"""

    def __init__(self, lib, B=1):
        self.lib = lib
        self.img, self.off, self.dir = [None] * B, [0] * B, [0] * B
        self.picks = 0

    def fill(self, slid, prev, dx):
        """slid [B, 3, H, W] = previous frame slid by the turn; prev [B, 3, H, W] = previous frame; dx [B] whole px
        (+ = content moved LEFT = turned right). Returns slid with the revealed strips replaced."""
        out = slid.clone()
        W = slid.shape[-1]
        for b in range(slid.shape[0]):
            n = int(abs(int(dx[b])))
            if n == 0 or n >= W:
                continue
            d = 1 if int(dx[b]) > 0 else -1
            if self.img[b] is None or self.dir[b] != d or self.off[b] + n > W:
                self.img[b] = self.lib.frame(self.lib.nearest(prev[b:b + 1].float())[0])
                self.off[b], self.dir[b] = 0, d
                self.picks += 1
            if d > 0:        # turning right: new columns on the right, the real frame enters from its left edge
                out[b, :, :, W - n:] = self.img[b][:, :, self.off[b]:self.off[b] + n]
            else:            # turning left: new columns on the left, it enters from its right edge
                out[b, :, :, :n] = self.img[b][:, :, W - self.off[b] - n:W - self.off[b]]
            self.off[b] += n
        return out
