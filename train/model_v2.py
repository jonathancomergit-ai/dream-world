"""Dream World model, version 2: v1's network, dreaming at 96 x 168 instead of 64 x 112.

THE SHORT VERSION
  v2 is not a new network. It is exactly v1 (model_v1.py: K = 8 memory frames, per-frame memory
  noise) fed bigger pictures. That works because of what kind of network it is:

  A CONVOLUTIONAL network has no idea how big the picture is. Each layer is a small 3x3 stamp
  that slides over every spot of the picture and does the same sum everywhere. A bigger picture
  just means the stamp is pressed down in more places. So the weights (the numbers inside the
  stamps) are the same shape at any size, and v1's trained weights drop straight into v2.

WHAT COULD HAVE DEPENDED ON THE PICTURE SIZE (checked 2026-10-19, see audit() below)
  1. Positional terms in the attention layers. Some networks add a learned "this is position
     (row 3, column 7)" vector to every pixel before attention. Those tables have one entry per
     position, so they'd need stretching to a new size. OURS HAS NONE: model.py's SelfAttention
     only does q/k/v 1x1 convs + attention. (The network still knows roughly where the picture
     edges are, from the zero padding at the borders, and that works at any size.)
  2. Hard-coded H or W anywhere. None: the U-Net never reads the picture size; the sampler takes
     it from the memory frames it is given.
  3. The halving steps. The U-Net halves the picture 3 times and doubles it back 3 times. That
     only round-trips if the size divides by 2 x 2 x 2 = 8: 96x168 -> 48x84 -> 24x42 -> 12x21. OK.
     (check_frame_size() below refuses sizes that don't.)
  4. Things that change in MEANING, not in shape (worth knowing, nothing to copy):
     - The 3x3 stamps now see a smaller piece of the scene (a lamp post is 1.5x more pixels
       wide). Early on, v2's edges/textures are a bit "off scale". Fine-tuning fixes this fast.
     - The attention level (the smallest one) is 12x21 = 252 spots instead of 8x14 = 112. Still
       tiny and cheap, and attention handles any number of spots.
     - Noise: the same noise strength (sigma) hides a bit LESS of a bigger picture, because
       neighbouring pixels are more alike and averaging them cancels some noise. Research that
       scales diffusion models up a resolution often shifts the noise levels up to match
       (by about the size ratio, here 1.5x). We DON'T, on purpose: the dream is mostly copied from
       the memory frames, not invented from noise, and keeping v1's noise levels means the warm
       start sees exactly the regime it was trained on, and play.py's dials mean the same thing.
       train_v2.py has a --p-mean flag if we ever want to try the shift (v1: -0.4; shifted: 0.0).

WARM START (warm_start_from_v1 below)
  Every tensor has the same shape in v1 and v2, so it's a plain one-to-one copy. The function
  still checks every name and shape and refuses anything that doesn't match, so if a future
  change sneaks in a size-dependent layer, it fails loudly instead of quietly training from junk.
"""
import torch

from model_v1 import DEFAULT_CFG_V1, DenoiserV1, build_model_v1   # noqa: F401  (v2 = v1's network)

FRAME_HW_V2 = (96, 168)        # (height, width) of every v2 frame
N_HALVINGS = 3                 # the U-Net halves the picture this many times (4 channel levels)

# Saved inside every v2 checkpoint, so --resume and play.py rebuild it exactly.
# "frame_hw" is new: it tells play.py which cache (picture size) the model needs.
DEFAULT_CFG_V2 = dict(DEFAULT_CFG_V1, version="v2", frame_hw=list(FRAME_HW_V2))


def check_frame_size(h, w, halvings=N_HALVINGS):
    """Stop early with a clear message if the U-Net can't halve this picture size cleanly."""
    m = 2 ** halvings
    if h % m or w % m:
        raise ValueError(f"frame size {h}x{w} must divide by {m} (the U-Net halves it {halvings} times)")


def build_model_v2(cfg=None):
    """The same network as v1, built from a v2 config (extra keys like frame_hw are just ignored
    by the builder: the network itself never needs to know the picture size)."""
    cfg = dict(DEFAULT_CFG_V2, **(cfg or {}))
    check_frame_size(*cfg["frame_hw"], halvings=len(cfg["channels"]) - 1)
    return build_model_v1(cfg)


def warm_start_from_v1(v2, v1_state):
    """Copy a v1 state_dict into a v2 model. Every name and shape must match exactly.
    Returns plain-words notes for run.log."""
    own = v2.state_dict()
    missing = [k for k in own if k not in v1_state]
    extra = [k for k in v1_state if k not in own]
    wrong = [f"{k}: v1 {tuple(v1_state[k].shape)} vs v2 {tuple(t.shape)}"
             for k, t in own.items() if k in v1_state and v1_state[k].shape != t.shape]
    if missing or extra or wrong:
        raise RuntimeError("v1 -> v2 warm start: the networks don't match one-to-one.\n"
                           f"  missing in v1: {missing}\n  only in v1: {extra}\n  shape differs: {wrong}")
    v2.load_state_dict({k: v.to(own[k].dtype) for k, v in v1_state.items()}, strict=True)
    n_buf = sum(1 for k, _ in v2.named_buffers())
    return [f"copied all {len(own)} tensors unchanged from v1 ({n_buf} of them fixed buffers: "
            f"the random Fourier frequencies that turn noise levels into vectors)",
            "nothing in the network depends on the picture size (no positional tables, no "
            "hard-coded H/W); 96x168 halves cleanly to 48x84, 24x42, 12x21"]


@torch.no_grad()
def audit(device="cpu"):
    """Prove the size claims above by experiment: build the network, list anything whose shape
    could hold a picture size, and push a 64x112 and a 96x168 frame through it."""
    m = build_model_v2().to(device).eval()
    sizes = {64, 112, 96, 168, 8, 14, 12, 21, 32, 56, 48, 84, 16, 28, 24, 42}
    suspicious = [(k, tuple(t.shape)) for k, t in m.state_dict().items()
                  if t.ndim >= 2 and any(s in sizes for s in t.shape[2:])]
    print("tensors with a dimension that looks like a picture size:", suspicious or "none")
    for h, w in [(64, 112), FRAME_HW_V2]:
        ctx = torch.zeros(1, m.K, 3, h, w, device=device)
        act = torch.ones(1, m.K, dtype=torch.long, device=device)
        x = torch.zeros(1, 3, h, w, device=device)
        den, _ = m(x, torch.ones(1, device=device), ctx, 0.0, act)
        print(f"  {h}x{w} in -> {tuple(den.shape[2:])} out")


if __name__ == "__main__":
    mdl = build_model_v2()
    print(f"v2 parameters: {sum(p.numel() for p in mdl.parameters()) / 1e6:.2f} M "
          f"(K={mdl.K}, frames {FRAME_HW_V2[0]}x{FRAME_HW_V2[1]})")
    audit()
