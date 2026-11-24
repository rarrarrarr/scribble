"""The 'scribble' look: pencil shading + ink outlines + cross-hatching on paper.

Pure image code (OpenCV/numpy), no I/O. Used by make_scribble.py.
"""
import math

import cv2
import numpy as np


class Sketcher:
    """Converts BGR frames of a fixed size into sketch-style BGR frames.

    Paper grain and hatch textures are generated once per instance so a video
    doesn't "boil" from frame to frame.
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
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

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

    def __call__(self, bgr, t=0.0):
        gray = self.clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        g = gray.astype(np.float32) / 255.0
        # pencil: colour-dodge of gray with its blurred inverse
        blur = cv2.GaussianBlur(255 - gray, (0, 0), 6).astype(np.float32) / 255.0
        pencil = np.clip(g / np.maximum(1.0 - blur, 1e-3), 0, 1) ** 1.6
        # ink outlines
        edges = cv2.Canny(cv2.bilateralFilter(gray, 7, 50, 7), 40, 110)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        ink = 1.0 - cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 0.7) * 0.85
        # cross-hatching in the shadows
        tone = cv2.GaussianBlur(g, (0, 0), 3)
        shade = np.ones_like(g)
        for tex, thr in self.hatches:
            mask = np.clip((thr - tone) / 0.12, 0, 1)
            shade = np.minimum(shade, 1.0 - mask * (1.0 - tex) * 0.75)
        out = pencil * ink * shade * self.paper
        # slight hand wobble that drifts slowly over time
        k = math.sin(t * 0.7) * 0.6
        M = np.float32([[1, 0.004 * k, 0], [0.003 * k, 1, 0]])
        out = cv2.warpAffine(out, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
        # warm paper tint (BGR)
        tinted = np.dstack([out * 0.93 + 0.04, out * 0.96 + 0.025, out])
        return (np.clip(tinted, 0, 1) * 255).astype(np.uint8)


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
