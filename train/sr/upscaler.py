"""Sharper eyes for play: a tiny 2x upscaler for Dream World frames (96x168 -> 192x336). See README.md.

    import sys; sys.path.insert(0, "train/sr")          # or wherever this folder is
    from upscaler import Upscaler
    up = Upscaler()                                      # train/sr/checkpoints/sharp_eyes.pt, fp16, CUDA graphs
    hr = up.up(frames)       # frames [B, 3, 96, 168] model space ([-1, 1], any float dtype, any device)
                             # -> [B, 3, 192, 336] float32 in [-1, 1] on the GPU
    hr_u8 = up.up_u8(u8)     # numpy/torch uint8 [B, 96, 168, 3] -> numpy uint8 [B, 192, 336, 3]

Sharpness dial: Upscaler(strength=s) or up.set_strength(s), s in 0..1.
    1.0 (default) = the final model: invents fine detail (trained with the light adversarial critic)
    0.0           = the plain-L1 model it started from: cleaner, softer, never invents texture
    between       = a pixel blend of the two outputs ((1 - s) * soft + s * sharp); costs a second network pass
                    (~+1 ms). (Blending the WEIGHTS instead does not work for this pair: 0.5 came out worse than both
                    ends and no sharper than 0, measured on the validation frames.)
set_strength() can be called any time, also after CUDA graphs were recorded.

How it runs: output = bilinear 2x of the input (float32) + the network's detail (fp16). The first call for each
batch size (and dial mode) records a CUDA graph (~0.2 s, once); after that a batch-1 frame costs ~1.5 ms on the
RTX 5090 (~3 ms with a blend). graphs=False runs it eagerly (~3.5 ms at batch 1, Python launch overhead). It holds
~100 MB of VRAM. The output of a graphed call is a fresh tensor (cloned), safe to keep.
"""
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sr_model import SharpEyes  # noqa: E402

DEFAULT_CKPT = HERE / "checkpoints" / "sharp_eyes.pt"


class Upscaler:
    def __init__(self, ckpt=DEFAULT_CKPT, device="cuda", dtype=torch.float16, graphs=True, strength=1.0):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.device, self.dtype = torch.device(device), dtype

        def build(sd):
            n = SharpEyes(**self.cfg)
            n.load_state_dict(sd)
            n = n.to(self.device, dtype).eval()
            for p in n.parameters():
                p.requires_grad_(False)
            return n
        self.sharp = build(ck["G_ema"])
        self.soft = build(ck["G_l1"]) if "G_l1" in ck else None
        self.step = ck.get("step")
        self.s = torch.ones((), device=self.device)
        self.graphs = {} if graphs else None
        self.set_strength(strength)

    def set_strength(self, strength):
        strength = float(min(max(strength, 0.0), 1.0))
        if self.soft is None and strength != 1.0:
            raise ValueError("this checkpoint has no G_l1 (soft) weights: only strength 1.0")
        self.strength = strength
        self.mode = "sharp" if strength == 1.0 else ("soft" if strength == 0.0 else "blend")
        self.s.fill_(strength)                         # in place: recorded graphs read it

    @torch.no_grad()
    def _run(self, x, mode):
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        xh = x.to(self.dtype)
        if mode == "sharp":
            det = self.sharp.detail(xh).float()
        elif mode == "soft":
            det = self.soft.detail(xh).float()
        else:
            det = torch.lerp(self.soft.detail(xh).float(), self.sharp.detail(xh).float(), self.s)
        return (base + det).clamp_(-1, 1)

    @torch.no_grad()
    def _capture(self, B, mode):
        x = torch.zeros(B, 3, 96, 168, device=self.device)
        st = torch.cuda.Stream(self.device)
        st.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(st):
            for _ in range(3):
                self._run(x, mode)
        torch.cuda.current_stream(self.device).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = self._run(x, mode)
        return g, x, y

    @torch.no_grad()
    def up(self, frames):
        """[B, 3, 96, 168] in [-1, 1] -> [B, 3, 192, 336] float32 in [-1, 1]."""
        x = frames.to(self.device, torch.float32)
        if self.graphs is None:
            return self._run(x, self.mode)
        key = (x.shape[0], self.mode)
        if key not in self.graphs:
            self.graphs[key] = self._capture(*key)
        g, xin, y = self.graphs[key]
        xin.copy_(x)
        g.replay()
        return y.clone()

    @torch.no_grad()
    def up_u8(self, frames_u8):
        """uint8 [B, 96, 168, 3] (numpy or torch) -> numpy uint8 [B, 192, 336, 3]."""
        t = torch.as_tensor(np.asarray(frames_u8)).to(self.device)
        x = t.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        y = self.up(x)
        return ((y + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def export(final_ckpt, l1_ckpt, out=DEFAULT_CKPT):
    """Small play-ready file from training saves: G_ema of the final save (sharp) + G_ema of the L1-only save (soft)."""
    f = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    l = torch.load(l1_ckpt, map_location="cpu", weights_only=False)
    torch.save({"cfg": f["cfg"], "G_ema": f["G_ema"], "G_l1": l["G_ema"], "step": f["step"], "l1_step": l["step"],
                "colour_fit": f["colour_fit"], "from": [str(final_ckpt), str(l1_ckpt)]}, out)
    return out
