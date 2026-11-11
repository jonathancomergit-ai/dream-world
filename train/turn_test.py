"""The turn-around memory test (2026-10-20, for v6a; the yardstick for v6b's map memory later).

    # score any saved model (v3/v5 saves are wrapped into an exactly-equivalent v6 network, see below):
    data\\venv\\Scripts\\python.exe train\\turn_test.py --ckpt train\\checkpoints\\dream_v5\\best.pt
    (train_v6.py runs the same test inside every rollout and logs it.)

THE QUESTION: walk somewhere, turn around, turn back. Is the street you left still there?

THE WALK (on the fair ruler's 5 walks x 2 noise draws, dreamed with the previous-frame start)
    turn rows:     W x 20 (2 s), turn-right x N, turn-left x N      -> facing the start direction again
    control rows:  W x 20,       turn-right x N, W x N              -> same length, but still facing AWAY
  Frames compared (dream frame numbers):
    pre   = frame 20         the view just before turning (already a dream, not the real start)
    away  = frame 20 + N     facing away (the turn rows)
    back  = frame 20 + 2N    facing the start direction again (the turn rows)
    ctrl  = frame 20 + 2N    the control rows: same moment, same amount of dreaming, never turned back

HOW FAR IS ONE TURN KEY? (measured 2026-10-20)
  The labeler (prep/label_actions.py) converts the sideways slide of the picture into degrees,
  assuming a ~70 deg wide camera, and calls a step "turn" when |yaw| is above that city's 82nd
  percentile (0.9-1.6 deg per 0.1 s). On the labelled turn steps of data/actions/*.npz the camera
  turned 2.4 deg per step on average (median 2.0; per city 1.75 Zurich .. 2.9 Venice). The model
  learnt "turn = the average labelled turn", so N = 60 steps ~ 145 deg (120 deg by the median),
  well past the 70 deg field of view: the street you left is completely out of sight.
  We ALSO measure how far the DREAM really turned (dream_yaw below: phase correlation between
  consecutive frames, the same pixel -> degree rule). On real video that meter reads 1.9 deg per
  labelled turn step (it under-reads the labeler's 2.4 a little), so we report the dream's
  turning as "deg (real-video scale)" = measured x 2.4/1.9. If a model under-turns, the test is
  easier than it looks; this number shows it.
  FIRST RESULT (v5 best, step 6236, 2026-10-20): it turned ~2 deg for 60 turn keys. Holding a turn key,
  v5 keeps the same view and smears it (train/_smoke/v6/turn_strip.jpg); with the old pure-static start it
  jumps around at random instead. So today the test mostly says "it doesn't turn": turned_deg is the
  number to watch first, and memory only means something once turned_deg is past ~half of asked_deg.

THE DISTANCES between two frames
  layout: shrink both to 12 x 21 pixels, remove each frame's own average colour and contrast, mean
          squared difference. Only "where are the big light/dark/coloured shapes" survives. Raw
          colour is left out on purpose: the colour fingerprint keeps a whole dream in one palette,
          so raw colour says "same walk", not "same view" (first try, 2026-10-20: raw colour made
          "facing away" look MORE like the start than "back again").
  scene:  the scene meter's 24 spectrum numbers (train/scene_meter.py), in units of real footage's
          spread. No positions at all: "same KIND of picture". Reported as a second opinion.

THE SCORES (each distance d, 'different place' = pre vs the other walks' pre, same noise draw)
  return_sim = 1 - d(pre, back) / d(different place)   1 = the same view, 0 = like another city
  ctrl_sim   = 1 - d(pre, ctrl) / d(different place)   what NOT turning back gives at that moment
  memory     = return_sim - ctrl_sim                    the part of coming back that is memory:
               ~0 = turning back brings nothing back (it re-invents), > 0 = it remembers the place
  away_sim   = 1 - d(pre, away) / d(different place)   should be low: the view really changed
"""
import math
import sys

import cv2
import numpy as np

TURN_STEPS = 60
TURN_WALK = 20
LOWRES = (21, 12)                  # (width, height)
REAL_DEG_PER_KEY = 2.4             # labeler's average turn per turn key (data/actions/*.npz)
METER_DEG_PER_KEY = 1.9            # what dream_yaw reads on real labelled turn steps


def turn_keys(walk=TURN_WALK, n=TURN_STEPS):
    return [1] * walk + [6] * n + [5] * n


def ctrl_keys(walk=TURN_WALK, n=TURN_STEPS):
    return [1] * walk + [6] * n + [1] * n


def layout(u8):
    """[..., H, W, 3] uint8 -> [..., 12, 21, 3]: shrunk, with each frame's own mean colour removed
    and its contrast scaled to 1, so only the arrangement of shapes is left."""
    flat = u8.reshape(-1, *u8.shape[-3:])
    out = np.stack([cv2.resize(f, LOWRES, interpolation=cv2.INTER_AREA) for f in flat]).astype(np.float32)
    out -= out.mean(axis=(1, 2), keepdims=True)
    out /= out.std(axis=(1, 2, 3), keepdims=True) + 1.0
    return out.reshape(*u8.shape[:-3], LOWRES[1], LOWRES[0], 3)


_win = {}


def dream_yaw(frames_u8):
    """Degrees turned between consecutive frames (+ = right), from the sideways shift found by
    phase correlation, with the labeler's rule (70 deg wide camera). frames [T, H, W, 3] -> [T-1]."""
    H, W = frames_u8.shape[1:3]
    if (H, W) not in _win:
        _win[(H, W)] = cv2.createHanningWindow((W, H), cv2.CV_32F)
    f = (W / 2) / math.tan(math.radians(35))
    g = [cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2GRAY).astype(np.float32) for x in frames_u8]
    return np.array([-math.degrees(math.atan(cv2.phaseCorrelate(a, b, _win[(H, W)])[0][0] / f))
                     for a, b in zip(g, g[1:])])


def turn_scores(turn_d, ctrl_d, n_walks, scene_features, scene_sd, walk=TURN_WALK, n=TURN_STEPS):
    """turn_d, ctrl_d uint8 [rows, T, H, W, 3]; rows = seed 0 walks..., seed 1 walks..."""
    i_pre, i_away, i_back = walk - 1, walk + n - 1, walk + 2 * n - 1
    rows = turn_d.shape[0]
    pick = {"pre": turn_d[:, i_pre], "away": turn_d[:, i_away], "back": turn_d[:, i_back], "ctrl": ctrl_d[:, i_back]}
    lay = {k: layout(v) for k, v in pick.items()}
    sce = {k: scene_features(np.ascontiguousarray(v)) for k, v in pick.items()}
    dl = lambda a, b: float(((a - b) ** 2).mean())
    ds = lambda a, b: float(np.sqrt((((a - b) / scene_sd) ** 2).mean()))
    out = {"turn_steps": n, "walk_steps": walk}
    for suf, d, z in (("", dl, lay), ("_scene", ds, sce)):
        diff = [d(z["pre"][r], z["pre"][q]) for r in range(rows) for q in range(rows)
                if r != q and r // n_walks == q // n_walks]           # same noise draw, other city
        base = max(float(np.mean(diff)), 1e-9)
        per = {k: [d(z["pre"][r], z[k][r]) for r in range(rows)] for k in ("away", "back", "ctrl")}
        sim = {k: 1 - float(np.mean(v)) / base for k, v in per.items()}
        out["return_sim" + suf] = round(sim["back"], 3)
        out["ctrl_sim" + suf] = round(sim["ctrl"], 3)
        out["away_sim" + suf] = round(sim["away"], 3)
        out["memory" + suf] = round(sim["back"] - sim["ctrl"], 3)
        out["d_diff_place" + suf] = round(base, 4)
        # per walk (both noise draws averaged): does one walk carry the score?
        out["memory_walks" + suf] = [round(float(np.mean(per["ctrl"][w::n_walks]) - np.mean(per["back"][w::n_walks])) / base, 2)
                                     for w in range(n_walks)]
    # how far did the dream really turn? (real-video scale, see the notes at the top)
    scale = REAL_DEG_PER_KEY / METER_DEG_PER_KEY
    yaws = np.stack([dream_yaw(turn_d[r, i_pre:i_back + 1]) for r in range(rows)])     # [rows, 2n]
    out["turned_right_deg"] = round(float(yaws[:, :n].sum(1).mean() * scale), 1)
    out["turned_back_deg"] = round(float(-yaws[:, n:].sum(1).mean() * scale), 1)
    out["asked_deg"] = round(n * REAL_DEG_PER_KEY, 1)
    return out


def turn_line(name, r):
    # VALIDITY: on 2026-10-20 v5 best turned ~2 deg for 60 turn keys (asked ~144): it does not pan when
    # you hold a turn key, it keeps the view and smears it. Then "back again" vs "control" says nothing
    # about memory, because it never looked away. Say so in the log instead of reporting a fake number.
    warn = ("" if r["turned_right_deg"] >= 0.5 * r["asked_deg"] else
            f"\n    !! the dream barely turned ({r['turned_right_deg']:.0f} of ~{r['asked_deg']:.0f} deg): it never really "
            f"looked away, so MEMORY is not meaningful yet (watch turned_deg grow first)")
    return (f"{name}: walk {r['walk_steps'] / 10:.0f} s, turn right {r['turn_steps']} keys (asked ~{r['asked_deg']:.0f} deg; "
            f"the dream turned ~{r['turned_right_deg']:.0f} right and ~{r['turned_back_deg']:.0f} back), turn back left "
            f"{r['turn_steps']}; return_sim {r['return_sim']} (1 = same view as before the turn, 0 = like another city), "
            f"turned-away control {r['ctrl_sim']}, so MEMORY {r['memory']:+.3f} (0 = re-invents, > 0 = remembers) "
            f"per walk {r['memory_walks']}; facing away {r['away_sim']}"
            f"\n    scene-meter version: return {r['return_sim_scene']}, control {r['ctrl_sim_scene']}, memory "
            f"{r['memory_scene']:+.3f}, away {r['away_sim_scene']}" + warn)


def turn_picture(path, turn_d, ctrl_d, n_walks, title, walk=TURN_WALK, n=TURN_STEPS):
    """One row per walk (first noise draw): before the turn | facing away | back again | control."""
    from train import safe_replace
    i_pre, i_away, i_back = walk - 1, walk + n - 1, walk + 2 * n - 1
    rows = []
    for w in range(n_walks):
        cells = [turn_d[w, i_pre], turn_d[w, i_away], turn_d[w, i_back], ctrl_d[w, i_back]]
        rows.append(np.concatenate([cv2.resize(c, (336, 192), interpolation=cv2.INTER_CUBIC) for c in cells], 1))
    im = np.concatenate(rows, 0)
    im = np.concatenate([np.zeros((28, im.shape[1], 3), np.uint8), im], 0)
    for k, t in enumerate(["before the turn", "facing away", "back again", "control: never turned back"]):
        cv2.putText(im, t if k else f"{title}: {t}", (8 + 336 * k, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.jpg")
    cv2.imwrite(str(tmp), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    safe_replace(tmp, path)


def run_turn_test(model, frames_mm, actions_np, starts, seeds, steps, device, scene_features, scene_sd,
                  walk=TURN_WALK, n=TURN_STEPS, start="prev"):
    """Dream the turn rows and the control rows in ONE batch (same fixed noise every time) and score.
    Returns (turn dreams, control dreams, scores)."""
    from train_v6 import dream_walks_v6
    rs = [s for _ in range(seeds) for s in starts]
    d = dream_walks_v6(model, frames_mm, actions_np, rs + rs, [turn_keys(walk, n)] * len(rs) + [ctrl_keys(walk, n)] * len(rs),
                       steps, device, start=start)
    k = len(rs)
    return d[:k], d[k:], turn_scores(d[:k], d[k:], len(starts), scene_features, scene_sd, walk, n)


def main():
    import argparse
    import pathlib
    import shutil
    import time
    import torch
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from data_v3 import load_cache_any
    from model_v6 import build_model_v6, v6_cfg_from_v5, warm_start_from_v5
    from scene_meter import features
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--starts", default="47706,234507,517230,151029,441350")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--turn-steps", type=int, default=TURN_STEPS)
    ap.add_argument("--picture", default="", help="optional .jpg of the first noise draw")
    args = ap.parse_args()
    tmp = here / "_exp" / "turn_test_ck.pt"          # copy first: never hold a live run's file open
    shutil.copy(args.ckpt, tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    cfg = ck["model_cfg"]
    if cfg.get("version") == "v3":
        # A v3/v5 save, run through the v6 code: v6 with ZERO extra blocks, warm-started from it, is the same
        # network (new memory slots weigh zero; train_v6's warm-start check measured |v6 - v5| ~1e-5 over a
        # whole 10-step dream), so both are scored by exactly the same dreaming code.
        m = build_model_v6(v6_cfg_from_v5(cfg, extra_blocks=0))
        warm_start_from_v5(m, ck["ema"])
    else:
        m = build_model_v6(cfg)
        m.load_state_dict(ck["ema"])
    m = m.cuda().eval()
    step = ck["step"]
    del ck
    _, fm, act, _ = load_cache_any(here.parent / "data" / "cache_v3")
    sd = np.load(here / "_exp" / "scene_meter_fit.npz")["sd"]
    starts = [int(x) for x in args.starts.split(",")]
    t0 = time.time()
    td, cd, r = run_turn_test(m, fm, act, starts, args.seeds, 10, "cuda", features, sd, n=args.turn_steps)
    print(turn_line(f"{args.ckpt} (step {step}), {time.time() - t0:.0f}s", r))
    print(r)
    if args.picture:
        turn_picture(pathlib.Path(args.picture), td, cd, len(starts), f"step {step}", n=args.turn_steps)


if __name__ == "__main__":
    main()
