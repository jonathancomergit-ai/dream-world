"""v8 data: MEMORY-COLLAGE GHOSTS cut from a panorama of the training window's own frames (2026-10-21).

    # how often does real footage come BACK to a direction it saw earlier in a window? (CPU, ~1 min)
    data\\venv\\Scripts\\python.exe train\\data_v8.py --revisit-stats

WHY (PLAN.md, 2026-10-21 "panorama memory PROVEN")
  Panorama memory (train/panorama.py) makes v7 come back to the same place when you turn away and back
  (turn-back return_sim 0.13 -> 0.88, full 360 0.02 -> 0.87), but the place comes back as a fragmented
  COLLAGE. Two reasons, both "never practised":
    1. In play, each frame's starting ghost is the slid previous frame PLUS strips cut from memory, and the
       memory itself is a patchwork: every panorama column holds the newest frame that covered it, so a
       remembered view is stitched from strips of many different (dreamed) frames.
    2. While turning, play starts from LESS static (sigma_max 2.0 instead of 5) so the dream sticks to the
       memory. v7 only ever practised ghosts at noise 4-5: at 2 it keeps the seams and smears.
  v8 practises exactly that: ghosts cut from a panorama of the window's own frames, at noise 2-3 and 4-5,
  with the REAL next frame as the answer, so the model learns to turn a patchwork into a clean photo.

THE GHOST (mem_ghost below; all whole-pixel, the same geometry as panorama.py)
  - Heading of every window position: the real yaw of every step -> sideways slide f*tan(yaw) px, ROUNDED
    (train_v7.motion_shift_px + round, exactly like panorama.py / pano_test.dream_pano), summed.
  - Confidence of frame p when the target is dreamed: fade ** (forward walked from p to the frame before
    the target), as the panorama fades it (0.985 per 'W' step; below 0.35 = forgotten).
  - Target column j sees frame p's column j + (heading_target - heading_p) when that is inside the frame.
  - Two kinds of rows:
      "newest"    : each column comes from the NEWEST frame that covers it (= what play's panorama holds),
                    using every frame up to the one just before the target (gap 0). Mostly this is the slid
                    previous frame + a strip from older memory where the camera swings back: play's ghost.
      "patchwork" : SYNTHETIC REVISITS. The newest `gap` frames are left out (gap drawn 0..gap_max), and
                    the view is cut into blocks 4-24 px wide, each block taken from a RANDOM older frame
                    that covers it (at its real heading, still remembered, at most `max_age` steps older
                    than the newest allowed one: the trainer draws 8..160 per row = easy..hard). Every frame also gets a small
                    random brightness/contrast change (--mem-jitter), like the drift between dreamed frames.
                    = a remembered view stitched from old frames: the model must rebuild a clean photo
                    from it (real windows that turn away and come back are rare, see --revisit-stats).
  - Columns no remembered frame covers: the slid previous frame (v7's shift ghost), like play.
  - Then the up/down slide of the target step (the panorama is 1-D), like play.
"""
import math
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_v7 import motion_shift_px, shift_frames                   # noqa: E402

FADE = 0.985          # panorama.py's fade_per_forward (play + tests)
MIN_CONF = 0.35       # panorama.py's min_conf


def pano_columns(fpx):
    """Columns in the full circle (panorama.py's P)."""
    return int(round(360 * fpx * math.pi / 180))


def window_headings(mot, scale, fpx):
    """mot [B, L, 4] normalised motion (mot[:, i] = step from window position i to i+1) -> integer headings
    [B, L + 1] in panorama columns (position 0 = 0), each step's slide rounded to whole pixels first."""
    B, L, _ = mot.shape
    sl = motion_shift_px(mot.reshape(B * L, 4), scale, fpx)[:, 0].round().reshape(B, L)
    return torch.cat([torch.zeros(B, 1, device=mot.device), sl.cumsum(1)], 1)


def window_conf(mot, scale, fwd_w, L, fade=FADE):
    """Confidence [B, L] of every window position at the moment the target (position L) is dreamed:
    fade ** (forward walked from position p to position L-1), as PanoramaMemory.forward does it."""
    f = (mot[:, :L - 1, 2].float() * scale[2]).clamp(min=0) / fwd_w                  # steps 0..L-2
    tail = torch.flip(torch.cumsum(torch.flip(f, [1]), 1), [1])                       # sum_{i>=p} f_i, p = 0..L-2
    tail = torch.cat([tail, torch.zeros_like(tail[:, :1])], 1)                        # position L-1: 0
    return fade ** tail


@torch.no_grad()
def mem_ghost(fr, lut, positions, mot, scale, fpx, fwd_w, prev_row, L, gap, patch, jitter=0.0,
              min_conf=MIN_CONF, fade=FADE, generator=None, max_age=None):
    """The memory-collage ghost of the target (window position L) for every row.
    fr [B, R, 3, H, W] model-space frames at the read positions (lut: position -> row), mot [B, L, 4],
    gap [B] long (leave out the newest `gap` frames), patch [B] bool (random-block patchwork vs newest-wins),
    max_age [B] long or None: only frames at most this many steps older than the newest one allowed.
    Returns (ghost [B, 3, H, W] float32, share of columns that came from memory [B])."""
    B, R, C, H, W = fr.shape
    dev = fr.device
    P = pano_columns(fpx)
    pos = torch.tensor([p for p in positions if p < L], device=dev)                    # candidate frames
    rows = lut.to(dev)[pos]                                                             # [Np]
    head = window_headings(mot.float(), scale, fpx)                                     # [B, L+1]
    conf = window_conf(mot, scale, fwd_w, L, fade)[:, pos]                              # [B, Np]
    d = head[:, L:L + 1] - head[:, pos]                                                 # [B, Np]
    d = (torch.remainder(d + P // 2, P) - P // 2).long()
    j = torch.arange(W, device=dev)
    src_col = j[None, None] + d[:, :, None]                                             # [B, Np, W]
    cover = (src_col >= 0) & (src_col < W)
    elig = (pos[None] <= (L - 1 - gap[:, None])) & (conf >= min_conf)                   # [B, Np]
    if max_age is not None:
        elig = elig & (pos[None] >= (L - 1 - gap[:, None] - max_age[:, None]))
    valid = cover & elig[:, :, None]
    # newest-wins: the largest position index; patchwork: a random covering frame per block of columns
    rank = torch.arange(len(pos), device=dev, dtype=torch.float32)[None, :, None].expand(B, -1, W)
    if bool(patch.any()):
        bw = torch.randint(4, 25, (B, 1), device=dev, generator=generator)
        off = torch.randint(0, 24, (B, 1), device=dev, generator=generator)
        blk = (j[None] + off) // bw                                                     # [B, W]
        nb = int(blk.max()) + 1
        u = torch.rand(B, len(pos), nb, device=dev, generator=generator)
        rnd = torch.gather(u, 2, blk[:, None].expand(-1, len(pos), -1))                 # [B, Np, W]
        rank = torch.where(patch[:, None, None], rnd, rank)
    score = torch.where(valid, rank, torch.full_like(rank, -1.0))
    best = score.argmax(1)                                                              # [B, W]
    have = valid.any(1)                                                                 # [B, W]
    col = torch.gather(src_col, 1, best[:, None]).squeeze(1).clamp(0, W - 1)            # [B, W]
    r = rows[best]                                                                      # [B, W]
    b_idx = torch.arange(B, device=dev)[:, None].expand(-1, W)
    mem = fr[b_idx, r, :, :, col].float()                                               # [B, W, 3, H]
    mem = mem.permute(0, 2, 3, 1)                                                       # [B, 3, H, W]
    if jitter > 0:
        # per source frame: brightness/contrast drift (patchwork rows only), like dreamed frames drift
        g = 1 + jitter * (2 * torch.rand(B, len(pos), device=dev, generator=generator) - 1)
        o = jitter * (2 * torch.rand(B, len(pos), device=dev, generator=generator) - 1)
        g = torch.where(patch[:, None], g, torch.ones_like(g))
        o = torch.where(patch[:, None], o, torch.zeros_like(o))
        gg = torch.gather(g, 1, best)[:, None, None]                                    # [B, 1, 1, W]
        oo = torch.gather(o, 1, best)[:, None, None]
        mem = (mem * gg + oo).clamp(-1, 1)
    sh = motion_shift_px(mot[:, L - 1], scale, fpx).round()                             # the target step's slide
    slid = shift_frames(fr[:, prev_row].float(), torch.stack([sh[:, 0], torch.zeros_like(sh[:, 1])], 1))
    use = have[:, None, None, :].float()
    ghost = use * mem + (1 - use) * slid
    if bool((sh[:, 1] != 0).any()):
        ghost = shift_frames(ghost, torch.stack([torch.zeros_like(sh[:, 1]), sh[:, 1]], 1))
    return ghost, have.float().mean(1)


# ---------------------------------------------------------------- how common are real revisits?

def revisit_stats(cache_dir, L=172, recent=10, n=40000, seed=0):
    """For random valid training windows (motion only, no pictures): what share of the target view is
    remembered from frames OLDER than the newest `recent` frames but NOT covered by any of those newest
    frames (= the camera looked away and came back within the window)? Also: how often does older memory
    fill the strip the slid previous frame leaves uncovered (gap 0, play's case)?"""
    from build_cache import compute_valid
    from data_v3 import load_cache_any
    from data_v7 import key_motion_table, load_motion
    index, fm, act, cut = load_cache_any(cache_dir)
    mot_mm, minfo = load_motion(cache_dir)
    valid = compute_valid(act, cut, index["cities"], L)
    W = fm.shape[2]
    fpx = (W / 2) / math.tan(math.radians(35))
    scale = torch.tensor(minfo["scale"])
    fwd_w = float(key_motion_table(minfo, 2.4)[1][2] * minfo["scale"][2])
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(valid), size=min(n, int(valid.sum())), replace=False)
    out = {"windows": len(idx)}
    shares, strip_fill, strip_any = [], [], []
    for a in range(0, len(idx), 2000):
        t = idx[a:a + 2000]
        mot = torch.from_numpy(np.stack([np.asarray(mot_mm[s:s + L], np.float32) for s in t]))
        head = window_headings(mot, scale, fpx)
        conf = window_conf(mot, scale, fwd_w, L)
        d = head[:, L:L + 1] - head[:, :L]                                            # [B, L]
        j = torch.arange(W)
        sc = j[None, None] + d[:, :, None]
        cover = (sc >= 0) & (sc < W) & (conf >= MIN_CONF)[:, :, None]                 # [B, L, W]
        new = cover[:, L - recent:].any(1)
        old = cover[:, :L - recent].any(1)
        shares.append((old & ~new).float().mean(1))
        # gap 0: columns the previous frame doesn't cover, and whether older memory has them
        prev_cov = cover[:, L - 1]
        unc = ~prev_cov
        strip_any.append(unc.any(1))
        strip_fill.append(((cover[:, :L - 1].any(1) & unc).sum(1).float() / unc.sum(1).clamp(min=1)))
    s = torch.cat(shares).numpy()
    out["revisit_share_mean"] = float(s.mean())
    for th in (0.05, 0.25, 0.5):
        out[f"windows_with_revisit_share>={th}"] = float((s >= th).mean())
    sa = torch.cat(strip_any).numpy()
    sf = torch.cat(strip_fill).numpy()
    out["gap0_target_has_uncovered_strip"] = float(sa.mean())
    out["gap0_strip_filled_by_older_memory_mean(when a strip exists)"] = float(sf[sa].mean()) if sa.any() else 0.0
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--revisit-stats", action="store_true")
    ap.add_argument("--cache", default=str(HERE.parent / "data" / "cache_v3"))
    ap.add_argument("--recent", type=int, default=10)
    args = ap.parse_args()
    if args.revisit_stats:
        for rc in (10, 30):
            print(f"newest {rc} frames left out:", json.dumps(revisit_stats(args.cache, recent=rc), indent=1), flush=True)
