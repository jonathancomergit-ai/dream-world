"""Panorama memory: a 360-degree strip of everything the dream has seen, by direction (2026-10-21).

WHY: since v7, turning is a SLIDE of known size (the "shift" start: the previous frame slid by the turn,
then static added). So we know exactly which compass direction every column of every dreamed frame was
facing. Keep a long strip that wraps around 360 degrees and paint each new frame into it at the
direction you're facing. When you turn, the next frame's starting "ghost" is cut from the strip at the
new direction:
  - direction seen before   -> the ghost IS the old view, so the model (which follows its ghost at
                               heavy noise, the discovery behind every fix this week) brings it back:
                               turn away and back and the same place is there
  - direction never seen    -> no memory there yet: use the slid previous frame as v7 does (the model
                               invents it), and paint the invention in, so the world gets BUILT as
                               you look around
  - walking forward         -> what's ahead of you changes as you move, so remembered columns slowly
                               fade ("confidence" decays per forward step); faded columns are treated
                               as unseen again, the dream reinvents them
This is a flat, 1-D version of the pose-indexed memory used by WorldMem / WorldCam / AnchorWeave
(memories tagged by camera direction, pulled up when you look that way again); flat is fine for an
abstract dream world.

Units: one panorama column = one pixel of sideways slide. The slide for a turn of y degrees is
f * tan(y) pixels (train_v7.motion_shift_px, f = the labeler's focal length, ~120 px at 168 wide), so for
small steps 1 degree ~ f * pi/180 px, and the full circle is 360 of those (~754 columns).
"""
import math

import numpy as np
import torch

from train_v7 import shift_frames


class PanoramaMemory:
    def __init__(self, B, H, fpx, device, fade_per_forward=0.985, min_conf=0.35):
        """B dreams at once, frames H high. fade_per_forward: how much a remembered column's confidence
        keeps per step of walking forward (0.985 -> half-life ~46 steps ~ 4.6 s). min_conf: below this a
        column counts as 'not seen' and the slid previous frame is used there instead."""
        self.P = int(round(360 * fpx * math.pi / 180))      # columns in the full circle
        self.B, self.H, self.dev = B, H, device
        self.img = torch.zeros(B, 3, H, self.P, device=device)
        self.conf = torch.zeros(B, self.P, device=device)
        self.heading = torch.zeros(B, device=device)       # in columns; + = turned right
        self.fade, self.min_conf = fade_per_forward, min_conf

    def _cols(self, W):
        """Panorama column of each of the W frame columns at the current heading [B, W]."""
        # whole-number offsets (-W/2 .. W/2-1): half-pixel offsets + torch.round's round-half-to-even
        # mapped some columns onto the same spot and skipped others (caught by the unit test)
        j = torch.arange(W, device=self.dev) - W // 2
        return (self.heading[:, None].round().long() + j[None]) % self.P

    def write(self, frames):
        """Paint frames [B, 3, H, W] (model space) in at the current heading, at full confidence."""
        B, _, H, W = frames.shape
        cols = self._cols(W)                                              # [B, W]
        for b in range(B):
            self.img[b][:, :, cols[b]] = frames[b].float()
            self.conf[b, cols[b]] = 1.0

    def write_centre(self, frames, turning):
        """OPT-IN alternative to write() (added 2026-10-21 by the v8 work; write() is unchanged).
        turning [B] bool. On TURNING steps a column is only repainted if this frame sees it at least as close to
        the middle of the view as the stored version was (or the stored one is forgotten, conf < min_conf);
        on walking steps everything is repainted, like write(). Why: with write(), the view before a turn gets
        repainted ~34 times by drifting re-dreams while it slides out of sight, so turning back finds a collage.
        Measured on v7 best (train_v8.py --mem-write centre, 5 walks x 2): turn-back return 0.88 -> 0.97, detail
        SSIM before-vs-back 0.37 -> 0.61, full 360 0.87 -> 0.95."""
        B, _, H, W = frames.shape
        if not hasattr(self, "cdist"):
            self.cdist = torch.zeros(B, self.P, device=self.dev)
        cols = self._cols(W)
        jd = (torch.arange(W, device=self.dev) - W // 2).abs().float()
        for b in range(B):
            c = cols[b]
            ok = (~turning[b]) | (jd <= self.cdist[b, c]) | (self.conf[b, c] < self.min_conf)
            cc = c[ok]
            self.img[b][:, :, cc] = frames[b][:, :, ok].float()
            self.conf[b, cc] = 1.0
            self.cdist[b, cc] = jd[ok]

    def turn(self, slide_px):
        """Advance the heading by the (whole-pixel) sideways slide of this step [B]."""
        self.heading = self.heading + torch.round(slide_px.float())

    def forward(self, amount):
        """Fade every remembered column by how much this step moved forward [B] (0 = standing / turning)."""
        self.conf = self.conf * (self.fade ** amount.clamp(min=0).float())[:, None]

    def ghost(self, slid_prev):
        """The next frame's starting ghost [B, 3, H, W]: remembered columns where confidence is high
        enough, else the slid previous frame (v7's behaviour). Also returns the share of columns that
        came from memory (for logging)."""
        B, _, H, W = slid_prev.shape
        cols = self._cols(W)
        mem = torch.stack([self.img[b][:, :, cols[b]] for b in range(B)])        # [B, 3, H, W]
        c = torch.stack([self.conf[b, cols[b]] for b in range(B)])               # [B, W]
        use = (c >= self.min_conf)[:, None, None, :].float()
        return use * mem + (1 - use) * slid_prev.float(), float(use.mean())


class MapMemory:
    """Map memory (2026-10-22): a panorama PER SPOT on the ground, on top of a working PanoramaMemory.
    The dream's position is added up from the asked motion (walk keys only: a turn counts as on the spot, as the
    panorama treats it); the ground is cut into square cells `cell` W-steps wide. Every dreamed frame is also
    painted into its cell's own strip (centre rule, never fades). Stepping into a cell you've been in before swaps
    that cell's strip in as the working panorama, so the views you saw from there come back as ghosts.
    Tested (train/_exp/v9_mapmem.py, walk 4 s out, turn round, walk back, turn round): the start view returns
    0.82-0.94 (panorama alone 0.70-0.72, no memory 0.08-0.19). With no revisits it dreams exactly like the panorama."""

    def __init__(self, pano, cell=12.0):
        self.pano, self.cell = pano, cell
        if not hasattr(pano, "cdist"):
            pano.cdist = torch.zeros(pano.B, pano.P, device=pano.dev)
        self.pos = torch.zeros(pano.B, 2, device=pano.dev)
        self.cur = [(0, 0)] * pano.B
        self.cells = [{(0, 0): self._snap(b)} for b in range(pano.B)]
        self.swaps = 0

    def _snap(self, b):
        return (self.pano.img[b].clone(), self.pano.conf[b].clone(), self.pano.cdist[b].clone())

    def move(self, fwd, side):
        """fwd, side [B]: this step's forward / sideways motion in W-steps (0 for turn keys). Swaps strips."""
        ang = self.pano.heading * (2 * math.pi / self.pano.P)
        self.pos = self.pos + torch.stack([fwd * torch.sin(ang) + side * torch.cos(ang),
                                           fwd * torch.cos(ang) - side * torch.sin(ang)], 1)
        for b in range(self.pano.B):
            c = (int(round(float(self.pos[b, 0]) / self.cell)), int(round(float(self.pos[b, 1]) / self.cell)))
            if c == self.cur[b]:
                continue
            if c in self.cells[b]:
                im, cf, cd = self.cells[b][c]
                self.pano.img[b], self.pano.conf[b], self.pano.cdist[b] = im.clone(), cf.clone(), cd.clone()
                self.swaps += 1
            else:
                self.cells[b][c] = (torch.zeros_like(self.pano.img[b]), torch.zeros_like(self.pano.conf[b]),
                                    torch.zeros_like(self.pano.cdist[b]))
            self.cur[b] = c

    def write(self, frames, turning):
        """Also paint frames [B, 3, H, W] into each row's current cell strip (centre rule, no fading)."""
        B, _, H, W = frames.shape
        cols = self.pano._cols(W)
        jd = (torch.arange(W, device=self.pano.dev) - W // 2).abs().float()
        for b in range(B):
            im, cf, cd = self.cells[b][self.cur[b]]
            c = cols[b]
            ok = (~turning[b]) | (jd <= cd[c]) | (cf[c] < self.pano.min_conf)
            cc = c[ok]
            im[:, :, cc] = frames[b][:, :, ok].float()
            cf[cc] = 1.0
            cd[cc] = jd[ok]


def zoom_frames(x, z, cy=0.0):
    """Zoom frames x [B, 3, H, W] IN by z [B] (>= 1) around height cy (0 = middle), bicubic: the 'zoom ghost' for
    walking forward (real footage zooms ~1.5% per W-step: train/_exp/v9_zoom_fit.py). Same as train_v9.zoom_frames."""
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0] = 1.0 / z
    th[:, 1, 1] = 1.0 / z
    th[:, 1, 2] = cy * (1.0 - 1.0 / z)
    grid = torch.nn.functional.affine_grid(th, x.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(x.float(), grid, mode="bicubic", padding_mode="border", align_corners=False)
    return torch.where((z > 1.0)[:, None, None, None], out.clamp(-1, 1), x.float())
