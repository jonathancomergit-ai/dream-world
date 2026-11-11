"""Melt critic: p(this dreamed frame has melted), a learned meter (2026-10-21). Never trained against.

Trained by train/_exp/v9_melt_critic.py on EARLY (0-2 s) vs LATE (25-40 s) frames of 40 v8-final dreams from
the training part of the footage. Both sides carry v8's own look, so the only difference it can learn is
melting (mush, smears, glitch bars). (A plain real-vs-dream critic just learned v8's fingerprint.)
Checks it passed (all on things it never saw): real held-out video reads ~0.04 for the whole 40 s; v8 on held-out
walks rises 0.1 -> 0.5 (10 s) -> 0.8 (20 s) -> 0.96 (40 s) like the eye says; v7 dreams (another model),
epsilon-scaling colour glitch (marked worse, earlier) and ancestral / churn mush (worst) rank like the eye.
Caveat: a NEW kind of failure it has never seen might slip past it, so it is always used next to contact sheets.
  melt_score(walk) = mean p over the walk (0 = never melts, 1 = melted from the start)
  melt_time(walk)  = seconds until the 1 s average p stays above 0.5 for 3 s
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = pathlib.Path(__file__).resolve().parent / "_exp" / "v9" / "melt_critic.pt"


class Critic(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        ch = [3, w, 2 * w, 4 * w, 4 * w, 8 * w]
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1) for i in range(5)])
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(ch[-1], 1)

    def forward(self, x):                       # x uint8 [B, H, W, 3]
        h = x.permute(0, 3, 1, 2).float() / 127.5 - 1
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
        return self.head(self.drop(h.mean((2, 3))))[:, 0]


class MeltCritic:
    def __init__(self, device="cuda", weights=WEIGHTS):
        self.net = Critic().to(device).eval()
        self.net.load_state_dict(torch.load(weights, map_location=device))
        self.dev = device

    @torch.no_grad()
    def p(self, frames_u8):
        out = []
        for i in range(0, len(frames_u8), 512):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + 512])).to(self.dev)
            out.append(torch.sigmoid(self.net(x)).cpu().numpy())
        return np.concatenate(out)

    def report(self, frames_u8, fps=10, hold_s=3):
        p = self.p(frames_u8)
        ma = np.convolve(p, np.ones(fps) / fps, mode="valid")
        run, t = 0, None
        for i, b in enumerate(ma > 0.5):
            run = run + 1 if b else 0
            if run >= hold_s * fps:
                t = round((i - run + 1 + fps) / fps, 1)
                break
        return {"melt_score": round(float(p.mean()), 3), "melt_time": t if t is not None else len(p) / fps, "p": p}
