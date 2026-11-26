"""The 'scribble' look: pencil shading + ink outlines + cross-hatching on paper.

Pure image code (OpenCV/numpy), no I/O. Used by cli.py.
"""
import math
import threading

import cv2
import numpy as np


class Sketcher:
    """Converts BGR (or gray) frames of a fixed size into sketch-style BGR frames.

    Paper grain and hatch textures are generated once per instance so a video
    doesn't "boil" from frame to frame. Safe to call from several threads at once.
    """

    def __init__(self, width, height, seed=7):
        self.w, self.h = width, height
        rng = np.random.default_rng(seed)
        paper = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 0.8)
        self.paper = 1.0 - 0.035 * (paper - paper.min()) / (paper.max() - paper.min())
        # (texture, tone threshold): darker regions get more hatch layers
        # hatch jitter seeds are fixed (1-3) to reproduce the original look
        self.hatches = [(self._hatch(45, 6, 1), 0.55),
                        (self._hatch(-45, 6, 2), 0.35),
                        (self._hatch(0, 5, 3), 0.18)]
        # how dark each hatch layer makes the paper where it is fully on
        self.hatch_dark = [(((1.0 - tex) * 0.75).astype(np.float32), thr) for tex, thr in self.hatches]
        # pencil (colour-dodge of gray with its blurred inverse, then a gamma) depends only
        # on two uint8 values, so precompute it as a 256x256 table indexed gray*256 + blur
        g = np.arange(256, dtype=np.float32)[:, None] / 255.0
        b = np.arange(256, dtype=np.float32)[None, :] / 255.0
        self.pencil_lut = (np.clip(g / np.maximum(1.0 - b, 1e-3), 0, 1) ** 1.6).astype(np.float32).ravel()
        self._local = threading.local()  # a CLAHE object keeps scratch buffers: one per thread

    @property
    def clahe(self):
        c = getattr(self._local, "clahe", None)
        if c is None:
            c = self._local.clahe = cv2.createCLAHE(2.0, (8, 8))
        return c

    def _hatch(self, angle, spacing, seed):
        r = np.random.default_rng(seed)
        W, H = self.w, self.h
        img = np.ones((H * 2, W * 2), np.float32)
        for y in range(0, H * 2, spacing):
            off = int(r.integers(-1, 2))
            cv2.line(img, (0, y + off), (W * 2, y + off + int(r.integers(-3, 4))), 0.0, 1, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((W, H), angle, 1.0)
        img = cv2.warpAffine(img, M, (W * 2, H * 2), borderValue=1.0)
        return img[H // 2:H // 2 + H, W // 2:W // 2 + W]

    def gray_and_edges(self, frame):
        """Contrast-equalised gray and dilated ink edges (uint8): the steps the GPU
        sketcher also leaves on the CPU, as CuPy has no CLAHE, bilateral or Canny.
        `frame` is BGR, or already gray (2-D): the look only uses brightness."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        edges = cv2.Canny(cv2.bilateralFilter(gray, 7, 50, 7), 40, 110)
        return gray, cv2.dilate(edges, np.ones((2, 2), np.uint8))

    @staticmethod
    def wobble(t):
        """Slight hand wobble that drifts slowly over time: an affine matrix for warpAffine."""
        k = math.sin(t * 0.7) * 0.6
        return np.float32([[1, 0.004 * k, 0], [0.003 * k, 1, 0]])

    # warm paper tint (BGR): each channel is out * 255 * a + 255 * b
    TINT = ((0.93, 0.04), (0.96, 0.025), (1.0, 0.0))

    def __call__(self, frame, t=0.0):
        """Sketch of one frame (BGR or gray uint8) at time t seconds, as BGR uint8."""
        # Every factor below is in [0, 1], multiplied together in place into `out`;
        # in-place float32 math keeps this fast (numpy temporaries dominated the time).
        gray, edges = self.gray_and_edges(frame)
        # pencil: table lookup on (gray, blurred inverse of gray)
        idx = gray.astype(np.int32)
        idx <<= 8
        idx += cv2.GaussianBlur(255 - gray, (0, 0), 6)
        out = self.pencil_lut.take(idx)
        # ink outlines: 1 - 0.85 * soft edges
        ink = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 0.7)
        ink *= np.float32(-0.85 / 255.0)
        ink += 1.0
        out *= ink
        # cross-hatching in the shadows: each layer fades in below its tone threshold
        tone = cv2.GaussianBlur(gray.astype(np.float32) * np.float32(1 / 255.0), (0, 0), 3)
        dark = np.zeros_like(tone)
        layer = np.empty_like(tone)
        for weight, thr in self.hatch_dark:
            np.multiply(tone, np.float32(-1 / 0.12), out=layer)  # (thr - tone) / 0.12
            layer += np.float32(thr / 0.12)
            np.clip(layer, 0, 1, out=layer)
            layer *= weight
            np.maximum(dark, layer, out=dark)
        np.subtract(1.0, dark, out=dark)
        out *= dark
        out *= self.paper
        out = cv2.warpAffine(out, self.wobble(t), (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
        # scale each channel straight to uint8
        return cv2.merge([(out * np.float32(255 * a) + np.float32(255 * b)).astype(np.uint8)
                          for a, b in self.TINT])


def contact_sheet(thumbs, title, cols=6, scale=1 / 3):
    """Grid of (timestamp_seconds, bgr_image) pairs with a title and time labels."""
    if not thumbs:
        raise ValueError("no frames for the contact sheet")
    h, w = thumbs[0][1].shape[:2]
    tw, th = int(w * scale), int(h * scale)
    pad, lab, head = 14, 22, 50
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((head + pad + rows * (th + lab + pad), pad + cols * (tw + pad), 3),
                    (235, 242, 246), np.uint8)
    # Hershey fonts are ASCII-only
    title = title.encode("ascii", "replace").decode()
    cv2.putText(sheet, title, (pad, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    for k, (t, img) in enumerate(thumbs):
        r, c = divmod(k, cols)
        x = pad + c * (tw + pad)
        y = head + pad + r * (th + lab + pad)
        sheet[y:y + th, x:x + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (x - 1, y - 1), (x + tw, y + th), (60, 60, 60), 1)
        cv2.putText(sheet, f"#{k + 1}  {fmt_time(t)}", (x, y + th + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    return sheet


def fmt_time(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"
