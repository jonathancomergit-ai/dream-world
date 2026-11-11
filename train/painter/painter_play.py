"""Painter v1 in play (2026-10-24): paints each 96x168 dream frame at 192x336, remembering its own last frame.

    from painter_play import PainterPlay
    pp = PainterPlay()            # train/checkpoints/painter_v1/painter_pick.pt (else latest.pt), EMA weights, bf16
    hr = pp.paint(frame)          # [1, 3, 96, 168] model space -> [1, 3, 192, 336] float in [-1, 1]
    pp.reset()                    # new start / mode switch: forget the previous painted frame
Display only: the dreamer keeps dreaming its own 96x168 frames. The whole few-step paint (all Euler steps) is
recorded as ONE CUDA graph on the first call (the starting noise is drawn outside it), like the dreamer's sampler."""
import pathlib

import torch

from painter_model import Painter, est_shift, slide

CKS = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "painter_v1"


def default_ckpt():
    for name in ("painter_pick.pt", "latest.pt"):
        if (CKS / name).exists():
            return CKS / name
    return None


class PainterPlay:
    def __init__(self, ckpt=None, device="cuda", steps=4, graphs=True):
        ckpt = ckpt or default_ckpt()
        if ckpt is None:
            raise FileNotFoundError(f"no painter save in {CKS}")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = Painter().to(device).eval()
        self.model.load_state_dict(ck["ema"])
        self.step, self.steps, self.dev, self.use_graphs = ck.get("step"), steps, device, graphs
        self.prev_hr = self.prev_lr = None
        self.g = None

    def reset(self):
        self.prev_hr = self.prev_lr = None

    def _run(self):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            return self.model.sample(self.st["lr"], self.st["prev"], self.st["has"], steps=self.steps,
                                     noise=self.st["noise"]).float()

    def _record(self):
        d = self.dev
        self.st = {"lr": torch.zeros(1, 3, 96, 168, device=d), "prev": torch.zeros(1, 3, 192, 336, device=d),
                   "has": torch.zeros(1, device=d), "noise": torch.zeros(1, 3, 192, 336, device=d)}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream().wait_stream(side)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = self._run()

    @torch.no_grad()
    def paint(self, lr):
        lr = lr.float()
        if self.prev_hr is None:
            prev, has = torch.zeros(1, 3, 192, 336, device=self.dev), torch.zeros(1, device=self.dev)
        else:
            prev, has = slide(self.prev_hr, est_shift(self.prev_lr, lr) * 2), torch.ones(1, device=self.dev)
        noise = torch.randn(1, 3, 192, 336, device=self.dev)
        if self.use_graphs:
            if self.g is None:
                self._record()
            for k, v in (("lr", lr), ("prev", prev), ("has", has), ("noise", noise)):
                self.st[k].copy_(v)
            self.g.replay()
            out = self.out.clone()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model.sample(lr, prev, has, steps=self.steps, noise=noise).float()
        self.prev_hr, self.prev_lr = out, lr
        return out
