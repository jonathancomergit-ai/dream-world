"""deface's CenterFace face detector, run on the GPU with PyTorch (2026-10-22 night, for train/build_cache_v11.py).

WHY: the v11 cache blurs faces exactly like train/sr/build_hr.py (deface 1.5.0's bundled centerface.onnx, threshold
0.2, detection on the 1280x720 frame at 1280x720, 640x360 and 256x144). On the CPU (OpenCV DNN) that costs ~1 s per
frame, so the full 518k-frame cache would take 7+ hours. This file runs the SAME network (the very same weights,
read straight out of deface's centerface.onnx - nothing downloaded, no new packages) on the GPU, with deface's own
pre-processing (cv2.dnn.blobFromImage resize to multiples of 32) and its own decode + NMS-free box rule
(deface.centerface.CenterFace.decode), in float32 with TF32 off. Checked against OpenCV on real frames by
train/_exp/v11/v11_face_gpu_check.py (see face_gpu_check.json).

The ONNX file is read with a tiny protobuf reader below (the `onnx` package is not installed), and the graph is
executed node by node with torch ops.
"""
import struct

import numpy as np
import torch
import torch.nn.functional as F

ONNX_PATH = None


def _default_path():
    import deface.centerface as dc
    return dc.default_onnx_path


# ---------------------------------------------------------------- minimal protobuf wire reader

def _varint(b, i):
    r, s = 0, 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b):
    """bytes -> list of (field number, wire type, value) (value: int for varint/fixed, bytes for length-delimited)."""
    out, i, n = [], 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        elif w == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"wire type {w}")
        out.append((f, w, v))
    return out


def _packed_varints(v):
    r, i = [], 0
    while i < len(v):
        x, i = _varint(v, i)
        r.append(x)
    return r


def _signed(x):
    return x - (1 << 64) if x >= (1 << 63) else x


def _tensor(b):
    dims, dtype, raw, fdata, idata, name = [], 1, None, [], [], ""
    for f, w, v in _fields(b):
        if f == 1:
            dims += [_signed(x) for x in (_packed_varints(v) if w == 2 else [v])]
        elif f == 2:
            dtype = v
        elif f == 4:
            fdata += list(struct.unpack(f"<{len(v) // 4}f", v)) if w == 2 else [struct.unpack("<f", v)[0]]
        elif f == 7:
            idata += [_signed(x) for x in (_packed_varints(v) if w == 2 else [v])]
        elif f == 8:
            name = bytes(v).decode()
        elif f == 9:
            raw = bytes(v)
    if raw is not None:
        np_t = {1: np.float32, 7: np.int64, 6: np.int32, 11: np.float64}[dtype]
        arr = np.frombuffer(raw, np_t).copy()
    elif dtype == 1:
        arr = np.array(fdata, np.float32)
    else:
        arr = np.array(idata, np.int64)
    return name, arr.reshape(dims) if dims else arr.reshape(())


def _attr(b):
    name, val = "", None
    ints, floats = [], []
    for f, w, v in _fields(b):
        if f == 1:
            name = bytes(v).decode()
        elif f == 2:
            val = struct.unpack("<f", v)[0]
        elif f == 3:
            val = _signed(v)
        elif f == 4:
            val = bytes(v).decode(errors="replace")
        elif f == 5:
            val = _tensor(v)[1]
        elif f == 7:
            floats += list(struct.unpack(f"<{len(v) // 4}f", v)) if w == 2 else [struct.unpack("<f", v)[0]]
        elif f == 8:
            ints += [_signed(x) for x in (_packed_varints(v) if w == 2 else [v])]
    if ints:
        val = ints
    elif floats:
        val = floats
    return name, val


def read_onnx(path):
    """-> (nodes [(op, inputs, outputs, attrs)], initializers {name: np.ndarray}, graph inputs, graph outputs)."""
    b = memoryview(open(path, "rb").read())
    graph = next(v for f, w, v in _fields(b) if f == 7)
    nodes, inits, gin, gout = [], {}, [], []
    for f, w, v in _fields(graph):
        if f == 1:
            ins, outs, op, attrs = [], [], "", {}
            for f2, w2, v2 in _fields(v):
                if f2 == 1:
                    ins.append(bytes(v2).decode())
                elif f2 == 2:
                    outs.append(bytes(v2).decode())
                elif f2 == 4:
                    op = bytes(v2).decode()
                elif f2 == 5:
                    k, a = _attr(v2)
                    attrs[k] = a
            nodes.append((op, ins, outs, attrs))
        elif f == 5:
            k, t = _tensor(v)
            inits[k] = t
        elif f == 11:
            gin.append(next(bytes(v2).decode() for f2, w2, v2 in _fields(v) if f2 == 1))
        elif f == 12:
            gout.append(next(bytes(v2).decode() for f2, w2, v2 in _fields(v) if f2 == 1))
    return nodes, inits, gin, gout


# ---------------------------------------------------------------- the network as torch ops

class CenterFaceTorch:
    """CenterFace on the GPU. __call__(frames_rgb_uint8 [B, H, W, 3] numpy, in_shape (w, h) or None, threshold)
    -> list of B arrays [n, 5] (x1, y1, x2, y2, score) in the frame's pixel coordinates, exactly as
    deface.centerface.CenterFace(in_shape=...)(frame, threshold)[0] computes them (same resize, same decode)."""

    def __init__(self, path=None, device="cuda"):
        self.dev = device
        nodes, inits, gin, gout = read_onnx(path or _default_path())
        self.nodes = nodes
        self.w = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in inits.items()}
        self.inp = next(n for n in gin if n not in inits)
        self.outs = ["537", "538", "539", "540"]
        self.ops = sorted({n[0] for n in nodes})
        # free every intermediate tensor right after its last use (keeps big batches at 1280x736 small)
        last = {}
        for k, (op, ins, outs, a) in enumerate(nodes):
            for t in ins:
                last[t] = k
        self.free_after = [[t for t in ins if t and last.get(t) == k and t not in self.w and t not in self.outs]
                           for k, (op, ins, outs, a) in enumerate(nodes)]

    @torch.no_grad()
    def forward(self, x):
        env = dict(self.w)
        env[self.inp] = x
        for k, (op, ins, outs, a) in enumerate(self.nodes):
            g = lambda j: env[ins[j]] if j < len(ins) and ins[j] else None
            if op == "Conv":
                pads = a.get("pads", [0, 0, 0, 0])
                assert pads[0] == pads[2] and pads[1] == pads[3], pads
                y = F.conv2d(g(0), g(1), g(2), stride=a.get("strides", [1, 1]), padding=(pads[0], pads[1]),
                             dilation=a.get("dilations", [1, 1]), groups=a.get("group", 1))
            elif op == "ConvTranspose":
                pads = a.get("pads", [0, 0, 0, 0])
                assert pads[0] == pads[2] and pads[1] == pads[3], pads
                op_ = a.get("output_padding", [0, 0])
                y = F.conv_transpose2d(g(0), g(1), g(2), stride=a.get("strides", [1, 1]), padding=(pads[0], pads[1]),
                                       output_padding=tuple(op_), groups=a.get("group", 1),
                                       dilation=a.get("dilations", [1, 1]))
            elif op == "BatchNormalization":
                y = F.batch_norm(g(0), g(3), g(4), g(1), g(2), training=False, eps=a.get("epsilon", 1e-5))
            elif op == "Relu":
                y = F.relu(g(0))
            elif op == "Clip":
                lo = a.get("min", None)
                hi = a.get("max", None)
                lo = float(g(1)) if g(1) is not None else lo
                hi = float(g(2)) if g(2) is not None else hi
                y = g(0).clamp(lo, hi)
            elif op == "Add":
                y = g(0) + g(1)
            elif op == "Mul":
                y = g(0) * g(1)
            elif op == "Sigmoid":
                y = torch.sigmoid(g(0))
            elif op == "Concat":
                y = torch.cat([env[i] for i in ins], dim=a.get("axis", 1))
            elif op == "MaxPool":
                pads = a.get("pads", [0, 0, 0, 0])
                y = F.max_pool2d(g(0), a["kernel_shape"], a.get("strides", a["kernel_shape"]), (pads[0], pads[1]))
            elif op in ("Upsample", "Resize"):
                sc = None
                for t in ins[1:]:
                    if t and t in env and env[t].numel() == 4:
                        sc = env[t]
                y = F.interpolate(g(0), scale_factor=(float(sc[2]), float(sc[3])), mode="nearest")
            elif op == "Identity" or op == "Dropout":
                y = g(0)
            else:
                raise NotImplementedError(op)
            env[outs[0]] = y
            for t in self.free_after[k]:
                env.pop(t, None)
        return [env[o] for o in self.outs]

    @staticmethod
    def shape_transform(in_shape, orig_shape):
        h_orig, w_orig = orig_shape
        w_new, h_new = in_shape
        w_new, h_new = int(np.ceil(w_new / 32) * 32), int(np.ceil(h_new / 32) * 32)
        return w_new, h_new, w_new / w_orig, h_new / h_orig

    def blobs(self, frames, size):
        """deface's cv2.dnn.blobFromImage(img, 1.0, size, (0,0,0), swapRB=False, crop=False): an INTER_LINEAR resize
        (on the CPU, with cv2, so the input is bit-identical to OpenCV's) -> float32 NCHW."""
        import cv2
        if (frames.shape[2], frames.shape[1]) == size:
            arr = frames
        else:
            arr = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_LINEAR) for f in frames])
        return torch.from_numpy(arr).to(self.dev).permute(0, 3, 1, 2).float()

    @torch.no_grad()
    def __call__(self, frames, in_shape=None, threshold=0.2):
        B, H, W = frames.shape[:3]
        ins = (W, H) if in_shape is None else in_shape
        w_new, h_new, sw, sh = self.shape_transform(ins, (H, W))
        x = self.blobs(frames, (w_new, h_new))
        heat, scale, offset, lms = self.forward(x)
        heat, scale, offset = heat.float().cpu().numpy(), scale.float().cpu().numpy(), offset.float().cpu().numpy()
        out = []
        for b in range(B):
            out.append(decode(heat[b:b + 1], scale[b:b + 1], offset[b:b + 1], (h_new, w_new), threshold, sw, sh))
        return out


def decode(heatmap, scale, offset, size, threshold, scale_w, scale_h):
    """deface.centerface.CenterFace.decode (boxes only) + the rescale in __call__, word for word."""
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    boxes = []
    for i in range(len(c0)):
        s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
        o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
        s = heatmap[c0[i], c1[i]]
        x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
        x1, y1 = min(x1, size[1]), min(y1, size[0])
        boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
    if not boxes:
        return np.empty((0, 5), np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    keep = nms(boxes[:, :4], boxes[:, 4], 0.3)
    dets = boxes[keep, :]
    dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / scale_w, dets[:, 1:4:2] / scale_h
    return dets


def nms(boxes, scores, nms_thresh):
    """deface.centerface.CenterFace.nms, word for word."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.uint8)
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        ix1, iy1, ix2, iy2, iarea = x1[i], y1[i], x2[i], y2[i], areas[i]
        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue
            xx1, yy1, xx2, yy2 = max(ix1, x1[j]), max(iy1, y1[j]), min(ix2, x2[j]), min(iy2, y2[j])
            w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True
    return np.nonzero(suppressed == 0)[0]
