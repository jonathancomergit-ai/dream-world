"""Reading the v3 cache (data/cache_v3) for train_v6.py: LONG windows, but only the frames we need.

THE PROBLEM
  v6a remembers 16.8 s (168 frames), and recovery practice dreams up to 4 of the recent frames,
  each needing its OWN 168 frames of memory. So one training example spans
      168 (history) + 4 (recovery practice) + 1 (the target) = 173 frames.
  v5 simply read every frame of its 53-frame span (2.6 MB). All 173 frames would be ~8.4 MB per
  example: 3x the disk reads and 3x the GPU copies, for frames that are mostly never looked at.

THE FIX: READ ONLY THE FRAMES SOME MEMORY SLOT WILL ACTUALLY USE
  The trainer works out (needed_positions() below) which positions of the 173-frame window are
  ever used: the target, the recovery-practice frames, every memory slot of each of those, and
  the frames of the colour fingerprint. With the default memory that is:
      positions 120..172 (the last 53 frames: v5's whole window, all of it is needed)
      + 8 little runs of 5 frames for the very-far slots (one run per very-far slot, because the
        4 recovery-practice dreams each look 0-4 frames further back)
      = 93 frames (~4.5 MB) instead of 173.
  The dataset reads each contiguous run with one memmap slice (9 slices per example) and stacks
  them; the trainer keeps a little lookup table "window position -> row in the stacked frames".
  The KEYS (1 byte each) are always read for the whole span: the key summaries need all of them.

WHICH WINDOWS ARE ALLOWED (unchanged rule, longer span)
  build_cache.compute_valid with the full 173-frame span: inside one source, every key known, no
  scene cut anywhere in the span (even where we skip frames: a cut between two very-far frames
  would still mean "that old frame is from a different shot").
"""
import numpy as np
import torch

from data import FLIP_MAP
from data_v3 import WindowDatasetV3


def needed_positions(offsets, L, sf_max, fp_offsets):
    """Window positions (0 = the oldest frame, L = the target) that the trainer will read.
    offsets: the model's memory offsets; sf_max: how many recent frames recovery practice may
    dream; fp_offsets: offsets BEFORE THE TARGET of the colour-fingerprint frames."""
    need = {L}
    for j in range(sf_max + 1):                  # the target (j = 0) and the dreamed frames L-1..L-sf_max
        g = L - j
        need.add(g)
        need.update(g - o for o in offsets)
    need.update(L - o for o in fp_offsets)
    pos = sorted(need)
    if pos[0] < 0:
        raise ValueError(f"the window ({L + 1} frames) is too short for these offsets")
    return pos


def runs_of(pos):
    """Sorted positions -> list of (first, last + 1) contiguous runs. [1,2,3,7,8] -> [(1,4), (7,9)]."""
    runs, a = [], pos[0]
    for p, q in zip(pos, pos[1:]):
        if q != p + 1:
            runs.append((a, p + 1))
            a = q
    runs.append((a, pos[-1] + 1))
    return runs


class SparseWindowDataset(WindowDatasetV3):
    """Item t = (frames at window positions `positions` (uint8 [P, H, W, 3]),
                 keys t .. t+L-1 (int64 [L]),
                 t)
    mirrored left <-> right half the time (keys swapped to match), like every dataset before it.
    The start index t comes back too (for --sf-walk-only: which source is it from?)."""

    def __init__(self, cache_dir, positions, L, flip_p=0.5):
        super().__init__(cache_dir, L, flip_p=flip_p)
        self.L = L
        self.positions = list(positions)
        self.runs = runs_of(self.positions)

    def __getitem__(self, t):
        self._open()
        t = int(t)
        # one memmap slice per contiguous run (copied out of the memmap), then stacked in order
        fr = np.concatenate([np.asarray(self.frames[t + a:t + b]) for a, b in self.runs], axis=0)
        fr = torch.from_numpy(fr)
        act = self.actions[t:t + self.L].clone()
        if self.flip_p > 0 and torch.rand(()) < self.flip_p:
            fr = fr.flip(2)                    # [P, H, W, 3]: dim 2 is width
            act = FLIP_MAP[act]
        return fr, act, t


def position_lookup(positions, L):
    """LongTensor [L + 1]: window position -> row in the stacked frames (-1 = not read)."""
    lut = torch.full((L + 1,), -1, dtype=torch.long)
    lut[torch.tensor(positions)] = torch.arange(len(positions))
    return lut
