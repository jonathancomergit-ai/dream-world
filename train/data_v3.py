"""Reading the v3 cache (data/cache_v3, built by build_cache_v3.py) for train_v3.py and play.py.

WHAT'S DIFFERENT FROM v2's DATA
  1. LONG WINDOWS. v3 remembers ~4.8 s (8 recent frames + 8 far frames every 0.5 s), and recovery
     practice may dream up to 4 of the recent frames, each of which needs ITS OWN 4.8 s of memory.
     So every training example is one long run of consecutive frames:
         history (48) + up to 4 extra for recovery practice + the target = 53 frames.
     The window must stay inside one video, every key must be known and no scene cut may be
     crossed: the same rule as before (build_cache.compute_valid), just with a longer window.
     We read all 53 frames as ONE contiguous slice of the memmap (cheap: ~2.6 MB per example), and
     the GPU picks out which of them each memory slot needs.
  2. SOURCES WITH WEIGHTS. Every video in the cache is a "source" (a city walk, or e.g. the
     Wildlife nature footage). How often each source is drawn is decided by source_shares() below:
       - by default each source is drawn in proportion to hours^(1 - balance) (balance 0.3):
         a 2 h city gets more draws than a 1 h city, but less than twice as many, so smaller
         cities aren't drowned out;
       - then each source's share is multiplied by its weight from SOURCE_WEIGHTS (or the
         --source-weights flag). Cities default to 1.0. Wildlife is 0.33: it is not first-person
         walking (tripod and pan shots), so we want a little of its look, not its camera habits.
  3. NON_WALKING sources (Wildlife) are never used for the fair-ruler test walks: a "walk forward
     for 20 s" test makes no sense on tripod footage.
"""
import numpy as np
import torch

from data import ACTION_NAMES, FLIP_MAP, InfiniteWeightedSampler, to_model_space, to_uint8_image  # noqa: F401
from data_v2 import WindowDatasetV2, cache_frame_size, load_cache_any                             # noqa: F401

# Sampling weight of each source, relative to the hours-based share. Not listed = 1.0.
SOURCE_WEIGHTS = {"Wildlife": 0.33}
# Sources that are not first-person walks: never used as fair-ruler test walks.
NON_WALKING = {"Wildlife"}


def parse_source_weights(text):
    """'Wildlife=0.33,Venice=1.5' -> dict, starting from SOURCE_WEIGHTS (the flag overrides it)."""
    w = dict(SOURCE_WEIGHTS)
    for part in (text or "").split(","):
        part = part.strip()
        if part:
            name, val = part.split("=")
            w[name.strip()] = float(val)
    return w


def source_shares(cities, usable, weights, balance=0.3):
    """What share of training draws each source gets.
    cities: index["cities"]; usable: {name: number of usable training windows};
    weights: {name: multiplier}; balance: 0 = purely by hours, 1 = every source equal.
    Returns {name: share} adding up to 1 (sources with no usable windows get 0)."""
    raw = {}
    for c in cities:
        n = usable.get(c["name"], 0)
        raw[c["name"]] = (n ** (1.0 - balance)) * weights.get(c["name"], 1.0) if n > 0 else 0.0
    tot = sum(raw.values()) or 1.0
    return {k: v / tot for k, v in raw.items()}


class WindowDatasetV3(WindowDatasetV2):
    """Item t = frames t .. t+L (L+1 consecutive frames, uint8 [L+1, H, W, 3]) and keys t .. t+L-1,
    mirrored left<->right half the time (keys swapped to match). The same as v2's dataset; only the
    window is much longer (L = 52 with the defaults). Lives in its own module so Windows'
    DataLoader workers can import it cheaply."""
    pass


class WindowDatasetV3Tagged(WindowDatasetV3):
    """v3.1 (--sf-walk-only): the same windows, plus the window's start index t as a third item,
    so the trainer can tell which source each example came from (walking city or Wildlife) and
    give recovery practice only to the walking ones. The default run keeps using WindowDatasetV3."""

    def __getitem__(self, t):
        fr, act = super().__getitem__(t)
        return fr, act, int(t)


def pick_ruler_cities(cities, n_max=4, exclude=NON_WALKING, names=None):
    """Which sources the fair ruler walks in: up to n_max walking sources, spread evenly over the
    (alphabetical) list. names = a list from an earlier run (resume) to keep the ruler identical,
    as long as those sources still exist."""
    walk = [c for c in cities if c.get("has_actions", True) and c["name"] not in exclude]
    if names:
        keep = [c for c in walk if c["name"] in names]
        if len(keep) == len(names):
            return keep
    if len(walk) <= n_max:
        return walk
    ix = np.linspace(0, len(walk) - 1, n_max).round().astype(int)
    return [walk[i] for i in ix]
