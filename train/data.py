"""Reading the cache built by build_cache.py, for train.py and play.py.

Lives in its own file (not inside train.py) because on Windows each DataLoader worker is a
brand-new Python process that has to import the dataset class by name. Keeping it in a small
plain module makes that import cheap and reliable.
"""
import json
import pathlib

import numpy as np
import torch

# Horizontal flip augmentation: mirror the picture left<->right, and the keys have to mirror too.
# strafe left (3) <-> strafe right (4), turn left (5) <-> turn right (6). Everything else stays.
FLIP_MAP = torch.tensor([0, 1, 2, 4, 3, 6, 5, 7, 8], dtype=torch.int64)

ACTION_NAMES = ["still", "W", "S", "A", "D", "turn-left", "turn-right", "look-up", "look-down"]


def load_cache(cache_dir):
    """Returns (index dict, frames memmap [N,64,112,3] uint8, actions int8 [N], cut bool [N])."""
    d = pathlib.Path(cache_dir)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    frames = np.memmap(d / "frames_64x112.u8", dtype=np.uint8, mode="r", shape=tuple(index["shape"]))
    actions = np.load(d / "actions.npy")
    cut = np.load(d / "cut.npy")
    return index, frames, actions, cut


def to_model_space(frames_u8):
    """uint8 [..., H, W, 3] (0..255) on any device -> float [..., 3, H, W] in [-1, 1]."""
    x = frames_u8.float().div_(127.5).sub_(1.0)
    return x.movedim(-1, -3)


def to_uint8_image(x):
    """float [3,H,W] in [-1,1] -> numpy uint8 [H,W,3]."""
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


class WindowDataset(torch.utils.data.Dataset):
    """Item t = frames t .. t+K (K context + 1 target) and actions t .. t+K-1.
    The sampler only hands out t values that are valid (see build_cache.compute_valid).
    The memmap is opened lazily inside each worker (an open memmap can't be pickled across)."""

    def __init__(self, cache_dir, K, flip_p=0.5):
        self.cache_dir = str(cache_dir)
        self.K = K
        self.flip_p = flip_p
        self.frames = None
        self.actions = None
        index = json.loads((pathlib.Path(cache_dir) / "index.json").read_text(encoding="utf-8"))
        self.n = index["shape"][0]

    def __len__(self):
        return self.n

    def _open(self):
        if self.frames is None:
            _, self.frames, a, _ = load_cache(self.cache_dir)
            self.actions = torch.from_numpy(a.astype(np.int64))

    def __getitem__(self, t):
        self._open()
        K = self.K
        fr = torch.from_numpy(np.array(self.frames[t:t + K + 1]))   # copy out of the memmap
        act = self.actions[t:t + K].clone()
        # torch's RNG is seeded differently in every worker, so this coin flip is truly random
        if self.flip_p > 0 and torch.rand(()) < self.flip_p:
            fr = fr.flip(2)                    # [K+1, H, W, 3]: dim 2 is width
            act = FLIP_MAP[act]
        return fr, act


class InfiniteWeightedSampler(torch.utils.data.Sampler):
    """Hands out window start indices forever, drawn with replacement in proportion to
    `weights` (0 = never: invalid or held-out windows). Drawn in chunks so we never build a
    giant list in memory."""

    def __init__(self, weights, seed=0, chunk=1 << 16):
        self.w = torch.as_tensor(weights, dtype=torch.float64)
        self.seed = seed
        self.chunk = chunk

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.multinomial(self.w, self.chunk, replacement=True, generator=g).tolist()

    def __len__(self):
        return 1 << 62
