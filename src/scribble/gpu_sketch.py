"""The scribble look with most of the per-frame work on an NVIDIA GPU (CuPy).

Optional: needs the `gpu` extra (see README). Importing this module raises
ImportError when CuPy is missing; GpuSketcher() raises when no GPU can run it.

CuPy has no CLAHE, bilateral filter or Canny, so those still run on the CPU
(Sketcher.gray_and_edges); the blurs, pencil lookup, hatching, paper, wobble and
tint run here as a few small CUDA kernels.

Output matches the CPU Sketcher except for scattered single pixels (about 0.1-0.2%
of them): OpenCV's 8-bit blur rounds in fixed point, the float blur here can land
one level away, and the pencil table is steep enough in places to turn that into
a visible step. Side by side the frames look the same.
"""
import threading
import warnings

import cv2
import numpy as np

# Expected with the pip-installed CUDA libraries (the `gpu` extra), which CuPy still
# finds on its own; it is raised when kernels first compile, so filter it for good.
warnings.filterwarnings("ignore", message="CUDA path could not be detected")
import cupy as cp  # noqa: E402

from .sketch import Sketcher

_SRC = r"""
__device__ __forceinline__ int reflect101(int i, int n) {
    return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i);
}

// One pass of a separable blur on a uint8 image: dst = blur(src * scale + offset).
// (dx, dy) is (1, 0) for along rows, (0, 1) for along columns.
extern "C" __global__ void blur_u8(const unsigned char* src, float* dst, const float* k, int r,
                                   int w, int h, int dx, int dy, float scale, float offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = 0.f;
    for (int i = -r; i <= r; i++)
        s += k[i + r] * src[reflect101(y + i * dy, h) * w + reflect101(x + i * dx, w)];
    dst[y * w + x] = s * scale + offset;
}

extern "C" __global__ void blur_f32(const float* src, float* dst, const float* k, int r,
                                    int w, int h, int dx, int dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = 0.f;
    for (int i = -r; i <= r; i++)
        s += k[i + r] * src[reflect101(y + i * dy, h) * w + reflect101(x + i * dx, w)];
    dst[y * w + x] = s;
}

// pencil * ink * (1 - strongest hatch layer) * paper, as Sketcher.__call__ does it
extern "C" __global__ void combine(const unsigned char* gray, const float* inv_blur,
                                   const float* ink, const float* tone, const float* lut,
                                   const float* hatch, const float* paper, float* out,
                                   int n, float t0, float t1, float t2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = __float2int_rn(inv_blur[i]);  // the CPU blur returns uint8, rounded
    float v = lut[gray[i] * 256 + min(max(b, 0), 255)];
    v *= 1.f + ink[i] * (-0.85f / 255.f);
    float tn = tone[i] * (-1.f / 0.12f), dark = 0.f;
    dark = fmaxf(dark, fminf(fmaxf(tn + t0, 0.f), 1.f) * hatch[i]);
    dark = fmaxf(dark, fminf(fmaxf(tn + t1, 0.f), 1.f) * hatch[n + i]);
    dark = fmaxf(dark, fminf(fmaxf(tn + t2, 0.f), 1.f) * hatch[2 * n + i]);
    out[i] = v * (1.f - dark) * paper[i];
}

__device__ __forceinline__ int reflect(int i, int n) {  // OpenCV BORDER_REFLECT
    return i < 0 ? -i - 1 : (i >= n ? 2 * n - 1 - i : i);
}

// Wobble (bilinear warp by the inverse affine m) and paper tint, straight to BGR uint8.
extern "C" __global__ void warp_tint(const float* src, unsigned char* bgr, int w, int h,
                                     float m0, float m1, float m2, float m3, float m4, float m5,
                                     float a0, float b0, float a1, float b1, float a2, float b2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float sx = m0 * x + m1 * y + m2, sy = m3 * x + m4 * y + m5;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0;
    int xa = reflect(x0, w), xb = reflect(x0 + 1, w), ya = reflect(y0, h), yb = reflect(y0 + 1, h);
    float v = (src[ya * w + xa] * (1.f - fx) + src[ya * w + xb] * fx) * (1.f - fy)
            + (src[yb * w + xa] * (1.f - fx) + src[yb * w + xb] * fx) * fy;
    unsigned char* p = bgr + 3 * (y * w + x);
    p[0] = (unsigned char)(v * a0 + b0);  // truncates, like numpy's astype(uint8)
    p[1] = (unsigned char)(v * a1 + b1);
    p[2] = (unsigned char)(v * a2 + b2);
}
"""


def _gauss(sigma, for_uint8):
    """OpenCV's kernel for GaussianBlur(ksize=(0, 0), sigma), whose size depends on the depth."""
    ksize = int(round(sigma * (3 if for_uint8 else 4) * 2 + 1)) | 1
    return cp.asarray(cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel()), ksize // 2


class GpuSketcher(Sketcher):
    """Drop-in for Sketcher. Safe to call from several threads at once: each thread
    gets its own CUDA stream, so one frame's copies overlap another's kernels."""

    def __init__(self, width, height, seed=7):
        super().__init__(width, height, seed)
        mod = cp.RawModule(code=_SRC)
        self._blur_u8, self._blur_f32, self._combine, self._warp_tint = (
            mod.get_function(n) for n in ("blur_u8", "blur_f32", "combine", "warp_tint"))
        self._lut = cp.asarray(self.pencil_lut)
        self._paper = cp.asarray(self.paper.astype(np.float32))
        self._hatch = cp.asarray(np.stack([w for w, _ in self.hatch_dark]))
        self._thr = [np.float32(thr / 0.12) for _, thr in self.hatch_dark]
        self._k6, self._k3, self._k07 = _gauss(6, True), _gauss(3, False), _gauss(0.7, False)
        self._tls = threading.local()
        # compile and run once now, so a GPU that can't do this fails here, not mid-video
        self(np.zeros((height, width, 3), np.uint8))

    def _blur(self, src, kr, scale=1.0, offset=0.0):
        """Separable blur of a uint8 image, scaled after; returns float32."""
        k, r = kr
        w, h = self.w, self.h
        tmp = cp.empty((h, w), cp.float32)
        out = cp.empty((h, w), cp.float32)
        grid = ((w + 31) // 32, (h + 7) // 8)
        self._blur_u8(grid, (32, 8), (src, tmp, k, np.int32(r), np.int32(w), np.int32(h),
                                     np.int32(1), np.int32(0), np.float32(scale), np.float32(offset)))
        self._blur_f32(grid, (32, 8), (tmp, out, k, np.int32(r), np.int32(w), np.int32(h),
                                      np.int32(0), np.int32(1)))
        return out

    def __call__(self, frame, t=0.0):
        gray, edges = self.gray_and_edges(frame)
        stream = getattr(self._tls, "stream", None)
        if stream is None:
            stream = self._tls.stream = cp.cuda.Stream(non_blocking=True)
        w, h, n = self.w, self.h, self.w * self.h
        with stream:
            g = cp.asarray(gray)
            e = cp.asarray(edges)
            # blur(255 - gray) == 255 - blur(gray): the kernel sums to 1
            inv_blur = self._blur(g, self._k6, -1.0, 255.0)
            ink = self._blur(e, self._k07)
            tone = self._blur(g, self._k3, 1 / 255.0)
            out = cp.empty((h, w), cp.float32)
            self._combine(((n + 255) // 256,), (256,),
                          (g, inv_blur, ink, tone, self._lut, self._hatch, self._paper, out,
                           np.int32(n), *self._thr))
            m = cv2.invertAffineTransform(self.wobble(t)).ravel()
            tint = [np.float32(v) for a, b in self.TINT for v in (255 * a, 255 * b)]
            res = cp.empty((h, w, 3), cp.uint8)
            self._warp_tint(((w + 31) // 32, (h + 7) // 8), (32, 8),
                            (out, res, np.int32(w), np.int32(h), *map(np.float32, m), *tint))
            host = res.get(stream=stream)
        stream.synchronize()
        return host
