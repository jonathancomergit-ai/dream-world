"""Reading a cache of ANY frame size (v2's 96x168 one, or v1's 64x112 one), for train_v2.py and play.py.

Why a new file instead of editing data.py: data.py's load_cache() has the v1 file name
("frames_64x112.u8") written into it, and the live v1 run's data-loader workers import data.py.
Changing a file a running training imports is asking for trouble, so v2 gets this small add-on.
Everything else (flip map, conversions, the sampler) is reused from data.py as-is.

How it finds the frames file: build_cache_v2.py writes its name into index.json ("frame_file").
The old v1 cache has no such entry, so we fall back to "frames_<H>x<W>.u8" built from the shape,
which gives "frames_64x112.u8" for v1: both caches load through the same function.
"""
import json
import pathlib

import numpy as np
import torch

from data import (ACTION_NAMES, FLIP_MAP, InfiniteWeightedSampler, WindowDataset,  # noqa: F401  (re-exported)
                  to_model_space, to_uint8_image)


def load_cache_any(cache_dir):
    """Returns (index dict, frames memmap [N,H,W,3] uint8, actions int8 [N], cut bool [N])."""
    d = pathlib.Path(cache_dir)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    shape = tuple(index["shape"])
    name = index.get("frame_file", f"frames_{shape[1]}x{shape[2]}.u8")
    frames = np.memmap(d / name, dtype=np.uint8, mode="r", shape=shape)
    actions = np.load(d / "actions.npy")
    cut = np.load(d / "cut.npy")
    return index, frames, actions, cut


def cache_frame_size(cache_dir):
    """(H, W) of a cache, read from its index.json without opening the frames."""
    index = json.loads((pathlib.Path(cache_dir) / "index.json").read_text(encoding="utf-8"))
    return tuple(index["shape"][1:3])


class WindowDatasetV2(WindowDataset):
    """data.py's WindowDataset (windows of frames + keys, with mirror flips), but it opens the
    cache through load_cache_any, so it works for any frame size. Everything else is inherited."""

    def _open(self):
        if self.frames is None:
            _, self.frames, a, _ = load_cache_any(self.cache_dir)
            self.actions = torch.from_numpy(a.astype(np.int64))
