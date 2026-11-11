"""Walk around inside the dream.

    python train/play.py                                  # newest checkpoint of run dream_v0
    python train/play.py --run dream_v1                   # v1 (8-frame memory); same keys
    python train/play.py --run dream_v2                   # v2 (8-frame memory, 96x168 frames)
    python train/play.py --ckpt train/checkpoints/dream_v3/best.pt   # v3 (long memory + colour memory)
    python train/play.py --ckpt train/checkpoints/dream_v0/step0060000.pt
    python train/play.py --smoke --run smoke3             # the tiny smoke-test model

Keys (one action per frame, like the training data):
    W / S        walk forward / back            A / D      step left / right
    Left / Right turn left / right              Up / Down  look up / down
    nothing      stand still (action 0)
  Dream dial:
    [  /  ]      fewer / more clean-up steps per frame (fewer = faster, smearier)
    -  /  =      less / more context noise (more = the model trusts its memory less: it
                 "repairs" more, but also hallucinates more)
    H            Euler <-> Heun sampler (Heun: sharper, half the speed; at this training
                 stage it over-sharpens into glitch-art static)
    ,  /  .      weaker / stronger colour anchor (0 = off). Each dreamed frame's brightness and
                 contrast get pulled this fraction of the way back toward the real starting
                 frames. Without it, long walks fade into beige mush; with it they stay vivid
                 and painterly (found in train/dial_lab.py, 2026-10-19)
    P            dream start: previous frame + noise (default, the 2026-10-20 fix) <-> pure static
                 (how v0-v4 were always dreamed; compare them to see the difference)
                 v7: previous -> SHIFT -> static. SHIFT = the previous frame slid by the turn you ask for,
                 then noise: the dream really swings round (v6 turned ~170 of 144 deg asked this way, but
                 invents a smeary strip at the edge; train_v7.guided_sample)
    9  /  0      (v3+) trippiness dial: how much static each dream frame starts from
                 neon (1.0) <- vivid (2.5) <- FAITHFUL (5.0, as trained) -> airy (8.0) -> minimal (15)
                 low = drifts into rainbow-neon glitch, high = calm simplified dreamscape
                 (measured 2026-10-20, train/tier1_eval.py --what smax; 5.0 keeps the most detail)
    M  /  N      (v3+) mix-a-city: borrow a random other city's colours (colour memory + anchor),
                 N = back to this walk's own colours
    C            (v3 only) colour memory on/off. v3 is told the colour "fingerprint" of the real
                 starting frames and tries to keep the dream's colours near it by itself; turn it
                 off to see the model without it (it was trained to work both ways)
    T  /  Y      (v7 only) slower / faster turning: how many degrees one turn key asks for per frame
                 (1 ... 8; starts at 4 = 40 deg per second). v7 is steered by camera-motion numbers,
                 and each key is turned into numbers by a fixed table (data_v7.key_motion_table)
    G            (v7 only) steering guidance 0 -> 1 -> 2 -> 3 -> 0: exaggerate what your keys ask for,
                 den = den(keys) + w * (den(keys) - den(no keys)). Costs a second network call per step
                 (run as one batch of two); the HUD shows the model's own speed
    R            restart from a fresh real video moment     Esc   quit

Picture size: v0 and v1 dream 64x112 frames (data/cache); v2 dreams 96x168 frames (data/cache_v2).
A v2 save says its own frame size in its model config ("frame_hw"), and play picks the cache with
that frame size for its real starting moments. Either way the window shows it at 896x512.

How it works: the model keeps the last K frames as memory (v0: 4, v1/v2: 8). Every tick it reads your key,
dreams the next frame from (memory + key), shows it, and pushes it into the memory, dropping
the oldest. After K ticks every frame you see is pure dream.
v3 keeps a longer rolling history (48 frames = 4.8 s): its 8 recent frames are the newest 8, and its
8 "far" frames are picked out of the history every 5 frames before those (model_v3.py). It also
gets the colour fingerprint of the 8 OLDEST real starting frames, held fixed until you press R.
v3 reads its starting moments from data/cache_v3 (or data/cache_v2 if that's missing: same frames).
v6 (model_v6.py, 2026-10-20) keeps 168 frames (16.8 s) of rolling history: v5's 16 memory slots plus
8 "very far" ones every 1.5 s, picked by the explicit offsets list saved in the checkpoint. Its key
summaries come from model.key_summary(), and its colour fingerprint uses the same frames as v5's
(the 8 real frames 4.8..4.1 s before the first dream). All the v3 keys and the speed-up work the same.
v7 (model_v7.py, 2026-10-20) is v6 + steering by CAMERA MOTION: next to the keys, it gets the motion
(yaw, pitch, forward, strafe) of the last 8 steps. The real starting moment brings its real measured motion
(data/cache_v3/motion.npy); every key you press adds that key's row of the key -> motion table (turns at
the T/Y speed). Everything else (speed-up, 9/0, M/N, P, ...) is as v6.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import pathlib
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DW = HERE.parent
sys.path.insert(0, str(HERE))
from data import ACTION_NAMES, to_model_space, to_uint8_image  # noqa: E402
from data_v2 import cache_frame_size, load_cache_any           # noqa: E402  (any frame size)
from model import build_model                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dream_v0")
    ap.add_argument("--ckpt", default=None, help="checkpoint file (default: <run>/latest.pt)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--steps", type=int, default=10)   # dial lab: 10 holds shapes far longer than 3-5
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--anchor", type=float, default=0.15, help="colour anchor strength, 0 = off")
    ap.add_argument("--no-graphs", action="store_true", help="don't use the CUDA-graph speed-up")
    ap.add_argument("--start", choices=["prev", "noise", "shift"], default="prev",
                    help="where each dreamed frame starts: prev = last frame + noise (fixed), noise = pure static (old), "
                         "shift = (v7 only) last frame slid by the asked turn + noise")
    ap.add_argument("--fps", type=int, default=10, help="cap; the videos were 10 fps")
    ap.add_argument("--selftest", type=int, default=0, help="run N ticks with scripted keys, no window, then exit")
    ap.add_argument("--memtest", action="store_true",
                    help="(v7) memory check, no window: walk 1 s, turn right 6 s, turn back 6 s; same place again?")
    ap.add_argument("--mem-off", action="store_true", help="(v7) start with memory off (K cycles off/panorama/map)")
    ap.add_argument("--panorama", action="store_true", help="(v7) start with the plain panorama memory instead of map memory")
    ap.add_argument("--no-zoom", action="store_true", help="(v7) start with the zoom ghost OFF (Z toggles; on by default)")
    ap.add_argument("--no-fuel", action="store_true", help="(v7) start with edge fuel OFF (F toggles; on by default)")
    ap.add_argument("--no-sr", action="store_true", help="start with sharper eyes (the 2x upscaler) OFF (U cycles)")
    ap.add_argument("--painter", action="store_true", help="start with painter v1 painting the picture (U cycles)")
    ap.add_argument("--maptest", action="store_true",
                    help="(v7) map-memory check, no window: walk 2 s, out 4 s, turn round, back 4 s, turn round; same view?")
    ap.add_argument("--seed", type=int, default=None, help="pick the starting spot reproducibly")
    args = ap.parse_args()

    base = HERE / "_smoke" if args.smoke else HERE
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else base / "checkpoints" / args.run / "latest.pt"
    cache = pathlib.Path(args.cache) if args.cache else (HERE / "_smoke" / "cache" if args.smoke else DW / "data" / "cache")
    if args.memtest:
        args.selftest = 130
    if args.maptest:
        args.selftest = 250
    if args.selftest:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    dev = "cuda"
    print(f"loading {ckpt} ...")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    version = ck["model_cfg"].get("version", "v0")
    is_v3 = version == "v3"
    is_v7 = version == "v7"
    is_v6 = version == "v6" or is_v7  # v7 = v6 + motion steering: same memory layout, same helpers
    long_mem = is_v3 or is_v6        # long memory + colour memory (v3/v5 and v6/v7): same keys and dials
    if is_v7:                                            # v7 saves (v6 + camera-motion steering): model_v7.py
        from model_v3 import FP_FRAMES, colour_fingerprint
        from model_v7 import build_model_v7
        model = build_model_v7(ck["model_cfg"]).to(dev).eval()
    elif is_v6:                                          # v6 saves (16.8 s memory, deeper U-Net): model_v6.py
        from model_v3 import FP_FRAMES, colour_fingerprint
        from model_v6 import build_model_v6
        model = build_model_v6(ck["model_cfg"]).to(dev).eval()
    elif is_v3:                                          # v3 saves (long + colour memory): model_v3.py
        from model_v3 import FP_FRAMES, build_model_v3, colour_fingerprint, far_action_summary
        model = build_model_v3(ck["model_cfg"]).to(dev).eval()
    elif version == "v2":                                # v2 saves (v1's network, 96x168): model_v2.py
        from model_v2 import build_model_v2
        model = build_model_v2(ck["model_cfg"]).to(dev).eval()
    elif version == "v1":                                # v1 saves (8-frame memory): model_v1.py
        from model_v1 import build_model_v1
        model = build_model_v1(ck["model_cfg"]).to(dev).eval()
    else:
        model = build_model(ck["model_cfg"]).to(dev).eval()
    model.load_state_dict(ck["ema"])            # the EMA weights dream best
    K = model.K
    Hn = model.history if long_mem else K   # frames of history to keep (v3: 48, v6: 168, others: K)
    if long_mem:
        mem_idx = torch.tensor([Hn - o for o in model.mem_offsets()], device=dev)
    # where the colour fingerprint's 8 frames start in the history: v3 = the oldest 8 (48..41 back);
    # v6 = the SAME moments (48..41 frames back), which are no longer the oldest in its longer history
    fp0 = Hn - model.fp_offset if is_v6 else 0

    def far_of(keys, mots=None):
        """The summary of the older keys the model wants (keys = every key so far, the one pressed now last).
        v3: far_action_summary of the 40 keys before the recent 8 (exactly as before); v6: model.key_summary.
        v7: model.extras = the v6 key summary + the motion block of the last K steps (mots, same order as keys)."""
        if is_v7:
            return model.extras(torch.tensor([keys[-Hn:]], device=dev),
                                torch.tensor(np.array(mots[-K:], np.float32), device=dev)[None])
        if is_v6:
            return model.key_summary(torch.tensor([keys[-Hn:]], device=dev))
        S, Fn = model.far_stride, model.far_n
        return far_action_summary(torch.tensor([keys[-(K + S * Fn):-K]], device=dev), Fn, S)
    # The frame size this model dreams at: v2 saves record it; v0/v1 are always 64x112.
    frame_hw = tuple(ck["model_cfg"].get("frame_hw", (64, 112)))
    # v11.1 --noise-scale saves trained with every noise level x k: their dreams start from k x the static
    NOISE_K = float(ck["model_cfg"].get("noise_scale", 1.0))
    # LEARNED FUEL saves (model_fuel.py, 2026-10-23 night): memory slot 0 = a real reference frame; L toggles
    IS_FUEL = bool(ck["model_cfg"].get("fuel"))
    ref_keep = None

    def mctx_of(c):
        if not IS_FUEL:
            return c
        return model.with_ref(c, ref_keep.get(c[:, -1]) if (ref_keep is not None and v7s.get("lfuel")) else None)
    if not args.cache and version == "v2":
        # v2 always trains on the real 96x168 cache (its smoke tests too), so look there for starts
        cache = DW / "data" / "cache_v2"
    if not args.cache and long_mem:
        cache = DW / "data" / "cache_v3"
        if not (cache / "index.json").exists():
            cache = DW / "data" / "cache_v2"
        if tuple(frame_hw) == (192, 336):        # v11 (2026-10-23): the 2x dreamer starts from the 2x footage
            cache = DW / "data" / "cache_v11"
    if cache_frame_size(cache) != frame_hw:
        sys.exit(f"{cache} holds {cache_frame_size(cache)} frames but this model dreams {frame_hw}. "
                 f"Pass --cache <folder> with the right size (v2: data/cache_v2, v0/v1: data/cache).")
    print(f"{version}, step {ck['step']}, {ck['params'] / 1e6:.1f}M params, K={K}, "
          f"frames {frame_hw[0]}x{frame_hw[1]}, starts from {cache}")
    index, frames_mm, actions_np, _ = load_cache_any(cache)

    rng = np.random.default_rng(args.seed)
    # v7: the real measured motion of every frame (for the starting history) and the key -> motion table
    # default turn speed 2.4 deg/frame (2026-10-22, was 4): every memory / edge-fuel / spin test ran at 2.4, and faster
    # turning reveals ~67% more new strip per frame for the model to invent, so dreams melted faster in play tests
    v7s = {"mhist": [], "turn_i": 2, "w_i": 0}
    if args.maptest:          # the out-and-back script's 75 turn steps = ~179 deg at 2.4 deg/step (play's default is 4)
        v7s["turn_i"] = 2
    TURN_SPEEDS = [1.0, 2.0, 2.4, 3.0, 4.0, 5.0, 6.0, 8.0]      # degrees per frame a turn key asks for
    GUIDES = [0.0, 1.0, 2.0, 3.0]                                # G cycles the guidance strength
    if is_v7:
        from data_v7 import key_motion_table, load_motion
        from train_v7 import GuidedGraphedSampler, guided_sample, motion_shift_px
        import math
        FPX = (frame_hw[1] / 2) / math.tan(math.radians(35))     # the labeler's focal length in pixels
        motion_mm, minfo_cache = load_motion(cache)
        minfo = ck.get("motion_info") or minfo_cache            # the scales this model was trained with
        # PANORAMA MEMORY (2026-10-21, train/panorama.py): a 360-degree strip of everything seen, by
        # direction. Turning starts each frame from what's remembered in that direction, so turning away
        # and back finds the same place (tested: return 0.13 -> 0.97, full 360 0.02 -> 0.95 with the centre rule). K toggles it.
        from panorama import MapMemory, PanoramaMemory, zoom_frames
        from train_v7 import shift_frames
        # MAP MEMORY (2026-10-22): a panorama per spot on the ground, so walking back finds the place again
        # (out-and-back test 0.82-0.94 vs panorama 0.70). ZOOM GHOST: walking forward starts each frame from the
        # previous one zoomed in 1.5% per step (real footage's rate), so walking moves like real walking.
        MEM_MODES = ["off", "PANORAMA", "MAP"]
        ZOOM_K = 0.015
        MEM_SMAX = 2.0 * NOISE_K  # less static on remembered views while turning (sticks closer to memory)
        FWD_W = float(key_motion_table(minfo, 2.4)[1][2] * minfo["scale"][2])   # forward motion of one 'W'
        # zoom ON by default since 2026-10-22: on the v9 pick, melt 0.43 vs 0.44 without, walking moves ~45% more
        # (layout change at 4 s: 51 vs 35; real 62), colour glitch never within 40 s (train/_exp/v9/judge_zoomC.log)
        # EDGE FUEL ON by default since 2026-10-22 (dream_fuel.EdgeFuel): turning slides matching REAL places in at
        # the edge instead of stretched streaks (spin melt 0.42 -> 0.22, streaks gone). F toggles.
        from dream_fuel import EdgeFuel, FuelLibrary
        t_f = time.time()
        # (2x frames are 4x bigger: a smaller library keeps it ~1.5 GB of GPU memory)
        fuel_lib = FuelLibrary(index, frames_mm, dev, n=20000 if frame_hw[0] <= 96 else 8000)
        print(f"edge fuel: {len(fuel_lib.idx)} real frames ready ({time.time() - t_f:.0f}s)")
        v7s.update(fuel=not args.no_fuel, edge=EdgeFuel(fuel_lib))
        if IS_FUEL:
            from model_fuel import RefKeeper
            ref_keep = RefKeeper(fuel_lib, every=10)
            v7s["lfuel"] = True
            print("LEARNED FUEL model: a real reference frame every 1 s (L toggles)")
        v7s.update(mem_on=not args.mem_off, pano=None, share=0.0, map=None, zoom=not args.no_zoom,
                   mem_mode=0 if args.mem_off else (1 if args.panorama else 2))
        if minfo["scale"] != minfo_cache["scale"]:
            print("!! data/cache_v3/motion.npy was rebuilt with other scales than this model was trained on: "
                  "the starting moment's motion may be off (keys still use the model's own table)")

    def build_pano(ctx):
        """v7: a fresh panorama with the real starting frames painted in at their real headings."""
        if not is_v7:
            return None
        p = PanoramaMemory(1, ctx.shape[-2], FPX, dev, fade_per_forward=0.985, min_conf=0.35)
        rows = torch.tensor(v7s["mhist"], device=dev, dtype=torch.float32)            # [Hn-1, 4]
        slides = motion_shift_px(rows, minfo["scale"], FPX)[:, 0].round()
        heads = torch.cat([torch.zeros(1, device=dev), slides.cumsum(0)])
        for i in range(ctx.shape[1]):
            p.heading = heads[i:i + 1].clone()
            p.write(ctx[:, i].float())
            if i < len(rows):
                p.forward((rows[i, 2] * minfo["scale"][2]).clamp(min=0).reshape(1) / FWD_W)
        v7s["map"] = MapMemory(p, cell=12.0)
        return p

    def motion_row(a):
        """v7: the normalised motion numbers a key stands for, at the current turn speed."""
        return key_motion_table(minfo, TURN_SPEEDS[v7s["turn_i"]])[a].tolist()

    def fresh_context():
        """Hn consecutive real frames (v0-v2: K of them; v3: 48) from a random spot inside one city.
        ctx is the rolling history buffer [1, Hn, 3, H, W], hist the Hn - 1 keys between its frames."""
        c = index["cities"][rng.integers(len(index["cities"]))]
        t = int(rng.integers(c["start"], c["end"] - Hn))
        ctx = to_model_space(torch.from_numpy(np.array(frames_mm[t:t + Hn])).to(dev))[None]
        hist = [int(a) if a >= 0 else 1 for a in actions_np[t:t + Hn - 1]]
        if is_v7:                       # v7: the real motion that goes with those keys
            v7s["mhist"] = np.asarray(motion_mm[t:t + Hn - 1], np.float32).tolist()
        return ctx, hist, f"{c['name']} frame {t - c['start']}"

    def colour_stats(ctx):
        """Per-channel brightness (mean) and contrast (std) of the newest K real starting frames."""
        ctx = ctx[:, -K:]
        return ctx.mean(dim=(1, 3, 4))[:, :, None, None], ctx.std(dim=(1, 3, 4))[:, :, None, None]

    def fingerprint(ctx):
        """v3's colour memory: the fingerprint of the OLDEST real starting frames (as in training).
        (v6: the same moments, 4.8..4.1 s before the first dream: see fp0)"""
        return colour_fingerprint(ctx[:, fp0:fp0 + FP_FRAMES]) if long_mem else None

    # SHARPER EYES (2026-10-22, train/sr/): a 3.7M-param 2x upscaler for DISPLAY only (the dream still runs at 96x168,
    # so it dreams exactly as before). On held-out real frames: 0.90 of real 2x footage's fine detail (plain stretch
    # 0.32), no colour shift, no invented faces; ~1.5 ms a frame. U cycles SHARP -> BLEND -> off.
    # PAINTER v1 (2026-10-24, train/painter/): a 23M-param diffusion painter (4 steps, one CUDA graph) that paints the
    # 96x168 dream at 192x336 and remembers its own last frame. U cycles SHARP -> BLEND -> PAINTER -> off.
    SR_MODES = [("SHARP", 1.0), ("BLEND", 0.5), ("PAINTER", "paint"), ("off", None)]
    OFF_I = len(SR_MODES) - 1
    sr, sr_i, painter = None, OFF_I, None
    if tuple(frame_hw) in ((96, 168), (192, 336)):     # 2x models (v11): the same network makes 4x (eager, ~11 ms)
        try:
            sys.path.insert(0, str(HERE / "sr"))
            from upscaler import Upscaler
            sr = Upscaler()
            sr_i = OFF_I if args.no_sr else 0
            print("sharper eyes: ready (2x upscaler, U cycles sharp / blend / painter / off)")
        except Exception as e:                    # never let the upscaler break play
            print(f"sharper eyes unavailable ({type(e).__name__}: {e})")
    if tuple(frame_hw) == (96, 168) and sr is not None:
        try:
            sys.path.insert(0, str(HERE / "painter"))
            from painter_play import PainterPlay
            painter = PainterPlay()
            print(f"painter v1: ready (step {painter.step}; U -> PAINTER)")
            if args.painter:
                sr_i = [m[1] for m in SR_MODES].index("paint")
        except Exception as e:                    # never let the painter break play
            print(f"painter unavailable ({type(e).__name__}: {e})")
    pygame.init()
    W, H = 896, 512
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dream World")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    ctx, hist, where = fresh_context()
    if is_v7:
        v7s["pano"] = build_pano(ctx)
    ref_mean, ref_std = colour_stats(ctx)
    colour = fingerprint(ctx)
    colour_on = True
    own_colours = (colour, ref_mean, ref_std)     # this walk's own colours (N restores them)
    colours_from = "own"
    FLAVOURS = [(1.0, "neon"), (2.5, "vivid"), (5.0, "faithful"), (8.0, "airy"), (15.0, "minimal")]
    flav = 2                                      # 5.0 = what it was trained for

    def borrow_colours():
        """Mix-a-city: the colour fingerprint (+ anchor colours) of 8 real frames from another city."""
        here = where.split(" frame")[0]
        others = [c for c in index["cities"] if c["name"] != here]
        c = others[rng.integers(len(others))]
        t = int(rng.integers(c["start"], c["end"] - Hn))
        f = to_model_space(torch.from_numpy(np.array(frames_mm[t:t + Hn])).to(dev))[None]
        m_, s_ = colour_stats(f)
        return fingerprint(f), m_, s_, c["name"]

    # SPEED: replay the network as a recorded CUDA graph (same maths, ~2x less overhead at batch 1).
    # Only for the standard path (no memory blur, Euler); anything else falls back to model.sample.
    graphed = None
    if long_mem and not args.no_graphs:
        try:
            from train_v3 import GraphedSampler          # (v6 has the same call signature, so it works as-is)
            graphed = GraphedSampler(model, start=args.start)
            k0 = hist + [0]
            far0 = far_of(k0, v7s["mhist"] + [motion_row(0)] if is_v7 else None)
            pad = (lambda c: model.with_ref(c)) if IS_FUEL else (lambda c: c)   # (fuel: record with the ref slot)
            diff = graphed.check(pad(ctx.index_select(1, mem_idx)), torch.tensor([k0[-K:]], device=dev),
                                 colour, far0)
            if is_v7:                   # v7's guided dreams record a second graph (a batch of two): check it too
                a_ = torch.tensor([k0[-K:]], device=dev)
                a0, f0 = model.null_of(a_, far0)
                m_ = pad(ctx.index_select(1, mem_idx))
                diff = max(diff, graphed.check(torch.cat([m_, m_]), torch.cat([a_, a0]), torch.cat([colour, colour]),
                                               torch.cat([far0, f0])))
            print(f"speed-up check: graph vs normal = {diff:.2e} ({'OK' if diff < 2e-2 else 'too different, not using it'})")
            if diff >= 2e-2:
                graphed = None
        except Exception as e:                    # never let the speed-up break play
            print(f"speed-up unavailable ({type(e).__name__}: {e}); using the normal path")
            graphed = None
    steps, noise, heun, anchor = args.steps, args.noise, False, args.anchor
    start = args.start if (is_v7 or args.start != "shift") else "prev"     # 'shift' is v7-only
    fps_avg = 0.0
    tick = 0
    script = [1] * 15 + [6] * 8 + [1] * 10 + [5] * 8 + [0] * 4      # selftest keys
    guided = GuidedGraphedSampler(graphed) if (is_v7 and graphed is not None) else None
    fps_by_w = {}                                 # v7 selftest: model speed per guidance strength
    mem_script = [1] * 10 + [6] * 60 + [5] * 60  # memtest keys; frames kept at 9 (before), 69 (away), 129 (back)
    mem_shots = {}
    map_script = [1] * 20 + [1] * 40 + [6] * 75 + [1] * 40 + [6] * 75   # maptest: frames kept at 19 (start), 134, 249
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_LEFTBRACKET:
                    steps = max(1, steps - 1)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    steps = min(30, steps + 1)
                elif ev.key == pygame.K_MINUS:
                    noise = max(0.0, round(noise - 0.02, 2))
                elif ev.key == pygame.K_EQUALS:
                    noise = min(0.7, round(noise + 0.02, 2))
                elif ev.key == pygame.K_h:
                    heun = not heun
                elif ev.key == pygame.K_COMMA:
                    anchor = max(0.0, round(anchor - 0.05, 2))
                elif ev.key == pygame.K_PERIOD:
                    anchor = min(0.5, round(anchor + 0.05, 2))
                elif ev.key == pygame.K_p and is_v7:      # v7: prev -> shift -> static -> prev
                    start = {"prev": "shift", "shift": "noise", "noise": "prev"}[start]
                elif ev.key == pygame.K_p:
                    start = "noise" if start == "prev" else "prev"
                    if graphed is not None:
                        graphed.start = start
                elif ev.key == pygame.K_9:
                    flav = max(0, flav - 1)
                elif ev.key == pygame.K_0:
                    flav = min(len(FLAVOURS) - 1, flav + 1)
                elif ev.key == pygame.K_m and long_mem:
                    colour, ref_mean, ref_std, colours_from = borrow_colours()
                elif ev.key == pygame.K_n and long_mem:
                    (colour, ref_mean, ref_std), colours_from = own_colours, "own"
                elif ev.key == pygame.K_c:
                    colour_on = not colour_on
                elif ev.key == pygame.K_t and is_v7:
                    v7s["turn_i"] = max(0, v7s["turn_i"] - 1)
                elif ev.key == pygame.K_y and is_v7:
                    v7s["turn_i"] = min(len(TURN_SPEEDS) - 1, v7s["turn_i"] + 1)
                elif ev.key == pygame.K_g and is_v7:
                    v7s["w_i"] = (v7s["w_i"] + 1) % len(GUIDES)
                elif ev.key == pygame.K_k and is_v7:          # off -> panorama -> map -> off
                    v7s["mem_mode"] = (v7s["mem_mode"] + 1) % 3
                    v7s["mem_on"] = v7s["mem_mode"] > 0
                elif ev.key == pygame.K_z and is_v7:
                    v7s["zoom"] = not v7s["zoom"]
                elif ev.key == pygame.K_f and is_v7:
                    v7s["fuel"] = not v7s["fuel"]
                elif ev.key == pygame.K_l and IS_FUEL:
                    v7s["lfuel"] = not v7s.get("lfuel")
                elif ev.key == pygame.K_u and sr is not None:
                    sr_i = (sr_i + 1) % len(SR_MODES)
                    if SR_MODES[sr_i][1] == "paint" and painter is None:
                        sr_i = (sr_i + 1) % len(SR_MODES)            # no painter save: skip that mode
                    if isinstance(SR_MODES[sr_i][1], float):
                        sr.set_strength(SR_MODES[sr_i][1])
                    if SR_MODES[sr_i][1] == "paint":
                        painter.reset()
                elif ev.key == pygame.K_r:
                    if painter is not None:
                        painter.reset()
                    ctx, hist, where = fresh_context()
                    if is_v7:
                        v7s["pano"] = build_pano(ctx)
                    ref_mean, ref_std = colour_stats(ctx)
                    colour = fingerprint(ctx)
                    own_colours, colours_from = (colour, ref_mean, ref_std), "own"

        # which key is held? turning/looking wins over walking (one action per frame)
        k = pygame.key.get_pressed()
        a = 0
        for key, act in [(pygame.K_w, 1), (pygame.K_s, 2), (pygame.K_a, 3), (pygame.K_d, 4),
                         (pygame.K_LEFT, 5), (pygame.K_RIGHT, 6), (pygame.K_UP, 7), (pygame.K_DOWN, 8)]:
            if k[key]:
                a = act
        if args.maptest:
            a = map_script[tick]
        elif args.memtest:
            a = mem_script[tick]
        elif args.selftest:
            a = script[tick % len(script)]
            # exercise the Tier-1 controls too (no key presses in a self-test): borrow colours at
            # tick 20, "minimal" dial at 30, own colours + "neon" at 45; a frame is saved at each
            if long_mem and tick == 20:
                colour, ref_mean, ref_std, colours_from = borrow_colours()
            if tick == 30:
                flav = len(FLAVOURS) - 1
            if long_mem and tick == 45:
                (colour, ref_mean, ref_std), colours_from = own_colours, "own"
                flav = 0
            if is_v7:                   # v7: guidance 2 for ticks 12-29 and 50+, faster turning from tick 30
                v7s["w_i"] = 2 if (12 <= tick < 30 or tick >= 50) else 0
                if tick == 30:
                    v7s["turn_i"] = 6
                start = "shift" if 36 <= tick < 50 else args.start      # also exercise the 'shift' start

        t0 = time.time()
        keys = hist + [a]                                   # every key so far, the one pressed now last
        act = torch.tensor([keys[-K:]], device=dev)
        gw = GUIDES[v7s["w_i"]] if is_v7 else 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if long_mem:
                mrow = motion_row(a) if is_v7 else None
                far = far_of(keys, v7s["mhist"] + [mrow] if is_v7 else None)
                smax = FLAVOURS[flav][0] * NOISE_K
                col_in = colour if colour_on else torch.zeros_like(colour)   # blanked = all zeros, as in training
                mctx = mctx_of(ctx.index_select(1, mem_idx))
                if is_v7 and noise == 0 and not heun:
                    # v7: plain or guided (one batch of two: with keys / without), recorded graph if available;
                    # start "shift" = the previous frame slid by the asked turn first (train_v7.guided_sample)
                    sh = motion_shift_px(torch.tensor([mrow], device=dev), minfo["scale"], FPX).round()
                    shift = sh if start == "shift" else None
                    ghost, smax_t = None, smax
                    walking = a not in (5, 6)
                    fwd_u = max(0.0, mrow[2] * minfo["scale"][2] / FWD_W) if walking else 0.0
                    if v7s["pano"] is not None:
                        v7s["pano"].turn(sh[:, 0])                      # heading is tracked even with memory off
                        if v7s["mem_mode"] == 2 and walking:            # map memory: where are we on the ground?
                            v7s["map"].move(torch.tensor([mrow[2] * minfo["scale"][2] / FWD_W], device=dev),
                                            torch.tensor([mrow[3] * minfo["scale"][3] / FWD_W], device=dev))
                        if v7s["mem_on"]:
                            slid = shift_frames(ctx[:, -1].float(), sh * torch.tensor([[1.0, 0.0]], device=dev))
                            if v7s["fuel"]:                           # edge fuel: real places slide in at the edge
                                slid = v7s["edge"].fill(slid, ctx[:, -1].float(), sh[:, 0].round().long())
                            ghost, v7s["share"] = v7s["pano"].ghost(slid)
                            if float(sh[0, 1]) != 0:                 # tilting: the strip is 1-D, slide vertically
                                ghost = shift_frames(ghost, sh * torch.tensor([[0.0, 1.0]], device=dev))
                            if a in (5, 6) and v7s["share"] > 0.5:   # turning toward remembered views
                                smax_t = MEM_SMAX
                    if ghost is None and v7s["fuel"] and float(sh[0, 0]) != 0:   # memory off: still fuel the turn edge
                        ghost = v7s["edge"].fill(shift_frames(ctx[:, -1].float(), sh), ctx[:, -1].float(),
                                                 sh[:, 0].round().long())
                    if v7s["zoom"] and fwd_u > 0:                       # zoom ghost: the world grows as you walk in
                        if ghost is None:
                            ghost = shift_frames(ctx[:, -1].float(), sh)
                        ghost = zoom_frames(ghost, torch.tensor([1.0 + ZOOM_K * fwd_u], device=dev))
                    if guided is not None:
                        nxt = guided.sample(mctx, act, steps, col_in, far, gw,
                                            sigma_max=smax_t, start=start, shift_px=shift, ghost=ghost)
                    else:
                        nxt = guided_sample(model, mctx, act, steps, col_in, far, w=gw,
                                            start=start, sigma_max=smax_t, shift_px=shift, ghost=ghost)
                elif graphed is not None and noise == 0 and not heun:
                    nxt = graphed.sample(mctx, act, steps, col_in, far, sigma_max=smax)
                else:
                    nxt = model.sample(mctx, act, steps=steps, sigma_cond=noise, heun=heun,
                                       colour=col_in, far_act=far, start=start, sigma_max=smax)
            else:
                nxt = model.sample(ctx, act, steps=steps, sigma_cond=noise, heun=heun, start=start)
        nxt = nxt.float()
        if anchor > 0:
            # colour anchor: re-colour the frame to the start's brightness/contrast, then blend a bit of that in
            m = nxt.mean(dim=(2, 3), keepdim=True)
            sd = nxt.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
            nxt = ((1 - anchor) * nxt + anchor * ((nxt - m) / sd * ref_std + ref_mean)).clamp(-1, 1)
        if is_v7 and v7s["pano"] is not None:
            # centre rule (panorama.write_centre): while turning, keep each column's most-central sighting instead
            # of repainting it ~34 times as it slides out of view (turn-back 0.88 -> 0.97, much less collage)
            v7s["pano"].write_centre(nxt.float(), torch.tensor([a in (5, 6)], device=dev))
            if v7s["mem_mode"] == 2:
                v7s["map"].write(nxt.float(), torch.tensor([a in (5, 6)], device=dev))
            if a not in (5, 6):
                v7s["pano"].forward((torch.tensor([mrow[2]], device=dev) * minfo["scale"][2]).clamp(min=0) / FWD_W)
        ctx = torch.cat([ctx[:, 1:], nxt[:, None].float()], dim=1)
        hist.append(a)
        hist = hist[-(Hn - 1):]
        if is_v7:
            v7s["mhist"] = (v7s["mhist"] + [mrow])[-(Hn - 1):]
        if SR_MODES[sr_i][1] == "paint" and painter is not None and nxt.shape[-2] == 96:  # painter: display only
            img = to_uint8_image(painter.paint(nxt.float())[0])       # [192,336,3]
        elif sr is not None and isinstance(SR_MODES[sr_i][1], float) and nxt.shape[-2] == 96:   # sharper eyes
            img = to_uint8_image(sr.up(nxt.float())[0])               # [192,336,3]
        elif sr is not None and isinstance(SR_MODES[sr_i][1], float):  # 2x model: same network at 2x input -> 384x672
            with torch.no_grad():                                    # (eager; it's all convolutions - tested 2026-10-23)
                base = torch.nn.functional.interpolate(nxt.float(), scale_factor=2, mode="bilinear", align_corners=False)
                det = sr.sharp.detail(nxt.half()).float()
                if SR_MODES[sr_i][1] < 1.0 and sr.soft is not None:
                    det = torch.lerp(sr.soft.detail(nxt.half()).float(), det, SR_MODES[sr_i][1])
            img = to_uint8_image((base + det).clamp(-1, 1)[0])        # [384,672,3]
        else:
            img = to_uint8_image(nxt[0])                              # [64,112,3] (v2: [96,168,3])
        dt = time.time() - t0
        fps_avg = 0.9 * fps_avg + 0.1 * (1 / max(dt, 1e-6)) if tick else 1 / max(dt, 1e-6)
        if is_v7 and tick >= 3:          # (skip the first ticks: graph recording)
            fps_by_w.setdefault(gw, []).append(1 / max(dt, 1e-6))

        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))      # pygame wants [W,H,3]
        screen.blit(pygame.transform.smoothscale(surf, (W, H)), (0, 0))
        # two HUD lines: what's happening / the dream dials (one line got cut off on the right)
        hud1 = (f"{ACTION_NAMES[a]:>10}   {fps_avg:4.1f} fps   steps {steps} ({'Heun' if heun else 'Euler'})   "
                f"start {'PREV' if start == 'prev' else ('SHIFT' if start == 'shift' else 'static')} [P]   blur {noise:.2f} [-=]   anchor {anchor:.2f} [,.]")
        hud2 = ((f"dream: {FLAVOURS[flav][1]} [9/0]   colours: {colours_from} [M/N]   "
                 f"colour mem {'on' if colour_on else 'OFF'} [C]   " if long_mem else "")
                + (f"turn {TURN_SPEEDS[v7s['turn_i']]:g} deg/frame [T/Y]   guide {gw:g} [G]   "
                   f"memory {MEM_MODES[v7s['mem_mode']]} [K] {v7s['share']:.0%}   zoom {'ON' if v7s['zoom'] else 'off'} [Z]   "
                   f"fuel {'ON' if v7s['fuel'] else 'off'} [F]   "
                   + (f"learned fuel {'ON' if v7s.get('lfuel') else 'off'} [L]   " if IS_FUEL else "")
                   if is_v7 else "")
                + (f"picture {SR_MODES[sr_i][0]} [U]   " if sr is not None else "")
                + "R new spot  Esc quit")
        for dx, dy, col in [(1, 1, (0, 0, 0)), (0, 0, (255, 255, 255))]:
            screen.blit(font.render(hud1, True, col), (8 + dx, 8 + dy))
            screen.blit(font.render(hud2, True, col), (8 + dx, 28 + dy))
            screen.blit(font.render(f"started from: {where}", True, col), (8 + dx, H - 24 + dy))
        pygame.display.flip()
        if args.maptest and tick in (19, 134, 249):
            mem_shots[tick] = img.astype(np.float32)
            pygame.image.save(screen, str(HERE / "out" / f"play_maptest_{MEM_MODES[v7s['mem_mode']].lower()}_t{tick}.png"))
        elif args.memtest and tick in (9, 69, 129):
            mem_shots[tick] = img.astype(np.float32)
            pygame.image.save(screen, str(HERE / "out" / f"play_memtest_{'on' if v7s.get('mem_on') else 'off'}_t{tick}.png"))
        elif args.selftest and tick in (19, 29, 44, 59):
            pygame.image.save(screen, str(HERE / "out" / f"play_selftest_t{tick}.png"))
        clock.tick(args.fps)
        tick += 1
        if args.selftest and tick >= args.selftest:
            out = (HERE / "_smoke" if args.smoke else HERE / "out") / "play_selftest.png"
            pygame.image.save(screen, str(out))
            print(f"selftest: {tick} ticks, model {fps_avg:.1f} fps at {steps} steps, saved {out}")
            for w_, v_ in sorted(fps_by_w.items()):
                print(f"  v7 guidance {w_:g}: model {np.median(v_):.1f} fps (median of {len(v_)} ticks)")
            if args.maptest:
                lay = {t_: v_.reshape(8, v_.shape[0] // 8, 14, v_.shape[1] // 14, 3).mean((1, 3)) for t_, v_ in mem_shots.items()}
                d_back = float(((lay[249] - lay[19]) ** 2).mean()); d_mid = float(((lay[134] - lay[19]) ** 2).mean())
                print(f"maptest (memory {MEM_MODES[v7s['mem_mode']]}, zoom {'on' if v7s['zoom'] else 'off'}, "
                      f"{v7s['map'].swaps} cell swaps): start->end diff {d_back:.0f}, start->facing-back diff {d_mid:.0f}, "
                      f"return score {1 - d_back / max(d_mid, 1e-6):.2f} (1 = same view)")
            if args.memtest:            # rough layout match: 8x14 blocks, back vs before, compared to away vs before
                lay = {t_: v_.reshape(8, v_.shape[0] // 8, 14, v_.shape[1] // 14, 3).mean((1, 3)) for t_, v_ in mem_shots.items()}
                d_back = float(((lay[129] - lay[9]) ** 2).mean()); d_away = float(((lay[69] - lay[9]) ** 2).mean())
                print(f"memtest (memory {'ON' if v7s['mem_on'] else 'off'}): before->back diff {d_back:.0f}, "
                      f"before->away diff {d_away:.0f}, return score {1 - d_back / max(d_away, 1e-6):.2f} (1 = same place)")
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
