"""v7 data: the CONTINUOUS camera motion of every training step, and the key -> motion table.

    # build it once (CPU only, ~10 s; safe while another run trains: it only ADDS two files):
    data\\venv\\Scripts\\python.exe train\\data_v7.py --build
    -> data/cache_v3/motion.npy        float16 [N frames, 4]  (yaw, pitch, fwd, strafe), normalised
       data/cache_v3/motion_norm.json  the normalisation constants, the stats, and the key -> motion table

WHY (the 2026-10-20 steering finding, PLAN.md)
  Up to v6 the model only got a KEY per step: one of 9 labels (W, turn left, turn right, ...). The
  labeler (prep/label_actions.py) called a step "turn right" whenever the camera turned more than
  ~1 deg in that 0.1 s, so "turn right" meant anything from a 1 deg nudge to an 8 deg swing, and most
  of the time the memory frames already showed the turning anyway. The model learned "the key adds
  nothing the memory doesn't say" and ignored it: holding turn-right for 40 frames turned the dream ~1 deg.
  But the labeler had MEASURED how far the camera moved at every step, as 4 numbers:
      yaw    degrees turned per step   (+ = right)
      pitch  degrees tilted per step   (+ = up)
      fwd    % the picture zoomed in per step (walking forward ~2; standing ~0; backwards < 0)
      strafe extra near-vs-far sideways slide in pixels (+ = side-step right; weak and noisy)
  v7 hands those numbers to the model too ("continuous motion conditioning").

ALIGNMENT (the same rule as the keys): motion[i] = how the camera moved from frame i to frame i+1 of
  data/cache_v3, i.e. the step that the key actions[i] describes. The last frame of each video has no
  next frame: motion 0 there (its key is -1 = unknown, so no training window uses it).

NORMALISATION (so all 4 numbers are roughly "-1..1 for normal moves")
  Per channel: divide by that channel's 95th percentile of |value| (over every non-cut step of the
  walking cities), then clip to +-CLIP (4). No centring: 0 must keep meaning "no motion" (and the
  mirror flip must turn yaw into -yaw, which only works if 0 stays 0). Big outliers (flow glitches
  during fast pans) are clipped instead of blowing up the input.

MIRROR FLIPS (training flips half the windows left <-> right, keys swapped by FLIP_MAP):
  yaw -> -yaw, strafe -> -strafe; pitch and fwd stay. Done in SparseWindowDatasetV7 below.

THE KEY -> MOTION TABLE (what play.py, the ruler and the turn test press)
  The model is now steered by motion numbers, but a player still presses keys. For each of the 9 keys
  we take the MEDIAN motion of the training steps that carry that label (walking cities, no cuts), so
  "W" = a typical walking step, "turn right" = a typical turning step (it keeps walking a little while it
  turns, like the people in the videos did). Then the TURN keys get their yaw REPLACED by a chosen
  turning speed (turn_deg, degrees per step), because the median labelled turn (~2 deg/step) is the
  thing we want to be able to dial: play.py defaults to DEFAULT_TURN_DEG (4 deg/step = 40 deg per second)
  and T / Y change it. The ruler and the turn test use TEST_TURN_DEG = 2.4 deg/step, the AVERAGE
  labelled turn = what a discrete turn key meant in the data, so the "asked" angle of v7's turn test
  (60 x 2.4 = 144 deg) is exactly v6's and the numbers compare.
  Look up/down get their pitch replaced the same way (look_deg).
"""
import json
import pathlib
import sys

import numpy as np
import torch

from data import FLIP_MAP
from data_v3 import NON_WALKING
from data_v6 import SparseWindowDataset

MOTION_NAMES = ["yaw", "pitch", "fwd", "strafe"]
MOTION_FILE = "motion.npy"
NORM_FILE = "motion_norm.json"
CLIP = 4.0                    # normalised values are clipped to +-4 (= 4x the 95th percentile)
SCALE_PCTL = 95.0
DEFAULT_TURN_DEG = 4.0        # play.py's starting turn speed (deg per 0.1 s step)
TEST_TURN_DEG = 2.4           # ruler + turn test: the average labelled turn (as v6's turn test assumed)
DEFAULT_LOOK_DEG = 1.2         # ~ the median labelled look step (pitch clips at 4 x p95 = ~1.75)
# Flip: mirror left <-> right flips the sign of yaw and strafe.
FLIP_SIGN = torch.tensor([-1.0, 1.0, 1.0, -1.0])


# ---------------------------------------------------------------- building motion.npy

def build_motion(cache_dir, actions_dir, verbose=True):
    """Read data/actions/<Name>.npz for every source of the cache, line it up with the cache frames,
    normalise, and write motion.npy + motion_norm.json into cache_dir. Returns the norm dict."""
    cache_dir, actions_dir = pathlib.Path(cache_dir), pathlib.Path(actions_dir)
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    acts = np.load(cache_dir / "actions.npy")
    cut = np.load(cache_dir / "cut.npy")
    N = index["shape"][0]
    raw = np.zeros((N, 4), np.float32)
    known = np.zeros(N, bool)
    for c in index["cities"]:
        z = np.load(actions_dir / f"{c['name']}.npz")
        n = c["frames"]
        m = min(n, len(z["yaw"]))
        if m != n and verbose:
            print(f"  ! {c['name']}: npz has {len(z['yaw'])} steps, cache {n} frames (using {m})")
        # same check as build_cache_v3: the npz keys must be the cache's keys, or the motion is misaligned
        same = np.array_equal(z["actions"][:m - 1].astype(np.int8), acts[c["start"]:c["start"] + m - 1])
        if not same:
            raise RuntimeError(f"{c['name']}: the keys in {actions_dir} differ from the cache's: motion would be misaligned")
        for j, k in enumerate(MOTION_NAMES):
            raw[c["start"]:c["start"] + m, j] = z[k][:m]
        known[c["start"]:c["start"] + m - 1] = True            # the last frame has no next frame
    raw[~known] = 0.0
    raw[~np.isfinite(raw)] = 0.0
    walking = np.zeros(N, bool)
    for c in index["cities"]:
        if c["name"] not in NON_WALKING:
            walking[c["start"]:c["end"]] = True
    ok = known & ~cut & (acts >= 0)
    ref = ok & walking
    scale = np.array([np.percentile(np.abs(raw[ref, j]), SCALE_PCTL) for j in range(4)], np.float32)
    scale = np.maximum(scale, 1e-3)
    norm = np.clip(raw / scale, -CLIP, CLIP).astype(np.float16)
    tmp = cache_dir / (MOTION_FILE + ".tmp.npy")
    np.save(tmp, norm)
    tmp.replace(cache_dir / MOTION_FILE)

    # ---- stats + the key table (in RAW units: degrees, %zoom, px)
    names9 = ["still", "W", "S", "A", "D", "turn-left", "turn-right", "look-up", "look-down"]
    key_median, key_count = [], []
    for a in range(9):
        sel = ref & (acts == a)
        key_count.append(int(sel.sum()))
        key_median.append([float(np.median(raw[sel, j])) if sel.any() else 0.0 for j in range(4)])
    yaw_turn = np.abs(raw[ref & np.isin(acts, [5, 6]), 0])
    info = {
        "names": MOTION_NAMES, "scale": scale.tolist(), "clip": CLIP, "scale_pctl": SCALE_PCTL,
        "units": ["deg/step", "deg/step", "%zoom/step", "px parallax"],
        "key_names": names9, "key_median_raw": key_median, "key_count": key_count,
        "labelled_turn_yaw": {"mean": float(yaw_turn.mean()), "median": float(np.median(yaw_turn)),
                              "p90": float(np.percentile(yaw_turn, 90))},
        "abs_yaw_pctl": {str(p): float(np.percentile(np.abs(raw[ref, 0]), p)) for p in (50, 75, 90, 95, 99)},
        "clipped_share": [float((np.abs(raw[ref, j] / scale[j]) > CLIP).mean()) for j in range(4)],
        "corr_yaw_vs_turnkey": float(np.corrcoef(raw[ref, 0], (acts[ref] == 6).astype(float) - (acts[ref] == 5))[0, 1]),
        "corr_pitch_vs_lookkey": float(np.corrcoef(raw[ref, 1], (acts[ref] == 7).astype(float) - (acts[ref] == 8))[0, 1]),
        "corr_fwd_vs_Wkey": float(np.corrcoef(raw[ref, 2], (acts[ref] == 1).astype(float))[0, 1]),
        "frames": int(N), "built_from": str(actions_dir),
    }
    (cache_dir / NORM_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def load_motion(cache_dir):
    """(motion memmap float16 [N, 4] normalised, norm-info dict)."""
    cache_dir = pathlib.Path(cache_dir)
    p = cache_dir / MOTION_FILE
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run  data\\venv\\Scripts\\python.exe train\\data_v7.py --build")
    info = json.loads((cache_dir / NORM_FILE).read_text(encoding="utf-8"))
    return np.load(p, mmap_mode="r"), info


# ---------------------------------------------------------------- the key -> motion table

def key_motion_table(info, turn_deg=DEFAULT_TURN_DEG, look_deg=DEFAULT_LOOK_DEG):
    """float32 [9, 4] NORMALISED motion for each key (row = key id 0..8), from the medians in `info`
    (motion_norm.json, also saved inside every v7 checkpoint). Turn keys get yaw = -/+turn_deg,
    look keys pitch = +/-look_deg; everything else is the data's median for that key.
    Units in: degrees per 0.1 s step."""
    raw = np.array(info["key_median_raw"], np.float32)
    raw[5, 0], raw[6, 0] = -turn_deg, +turn_deg
    raw[7, 1], raw[8, 1] = +look_deg, -look_deg
    scale = np.array(info["scale"], np.float32)
    return np.clip(raw / scale, -info["clip"], info["clip"]).astype(np.float32)


def describe_table(info, turn_deg=DEFAULT_TURN_DEG, look_deg=DEFAULT_LOOK_DEG):
    """Plain-words table for logs / the report."""
    t = key_motion_table(info, turn_deg, look_deg) * np.array(info["scale"], np.float32)
    return "; ".join(f"{n}: yaw {r[0]:+.2f} pitch {r[1]:+.2f} fwd {r[2]:+.2f} strafe {r[3]:+.2f}"
                     for n, r in zip(info["key_names"], t))


# ---------------------------------------------------------------- training windows with motion

class SparseWindowDatasetV7(SparseWindowDataset):
    """data_v6.SparseWindowDataset + the motion of every step of the window.
    Item t = (frames at the read positions uint8 [P, H, W, 3], keys int64 [L], t, motion float32 [L, 4])
    motion[i] goes with keys[i] (the step from window frame i to i+1). Mirrored (frames, keys AND
    motion) together, half the time."""

    def __init__(self, cache_dir, positions, L, flip_p=0.5):
        super().__init__(cache_dir, positions, L, flip_p=flip_p)
        self.motion = None

    def __getitem__(self, t):
        self._open()
        if self.motion is None:
            self.motion = np.load(pathlib.Path(self.cache_dir) / MOTION_FILE, mmap_mode="r")
        t = int(t)
        fr = np.concatenate([np.asarray(self.frames[t + a:t + b]) for a, b in self.runs], axis=0)
        fr = torch.from_numpy(fr)
        act = self.actions[t:t + self.L].clone()
        mot = torch.from_numpy(np.asarray(self.motion[t:t + self.L], dtype=np.float32))
        if self.flip_p > 0 and torch.rand(()) < self.flip_p:
            fr = fr.flip(2)
            act = FLIP_MAP[act]
            mot = mot * FLIP_SIGN
        return fr, act, t, mot


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cache", default=str(here.parent / "data" / "cache_v3"))
    ap.add_argument("--actions", default=str(here.parent / "data" / "actions"))
    args = ap.parse_args()
    if args.build:
        info = build_motion(args.cache, args.actions)
    else:
        _, info = load_motion(args.cache)
    print(json.dumps({k: v for k, v in info.items() if k != "key_median_raw"}, indent=1))
    print("key medians (raw):")
    for n, c, r in zip(info["key_names"], info["key_count"], info["key_median_raw"]):
        print(f"  {n:<11} n={c:>7}  yaw {r[0]:+.2f}  pitch {r[1]:+.2f}  fwd {r[2]:+.2f}  strafe {r[3]:+.2f}")
    print("table at play default:", describe_table(info))
    print("table for the ruler/turn test:", describe_table(info, TEST_TURN_DEG))
