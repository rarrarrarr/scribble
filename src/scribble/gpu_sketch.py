"""The scribble look with most of the per-frame work on an NVIDIA GPU (CuPy).

Optional: needs the `gpu` extra (see README). Importing this module raises
ImportError when CuPy is missing; GpuSketcher() raises when no GPU can run it.

Every step runs here as small CUDA kernels; the CPU only uploads the gray frame and
downloads the sketch. CuPy has no CLAHE, bilateral filter or Canny, so this file has
its own, written to follow OpenCV's (tiles, rounding, borders and thresholds alike):
on real frames they match OpenCV exactly but for about 0.02% of pixels.

Output matches the CPU Sketcher except for scattered single pixels (about 0.1-0.2%
of them): OpenCV's 8-bit blur rounds in fixed point, the float blur here can land
one level away, and the pencil table is steep enough in places to turn that into
a visible step. Side by side the frames look the same. A change to the look in
sketch.py needs the same change here.
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

__device__ __forceinline__ unsigned char round_u8(float v) {
    return (unsigned char)min(max(__float2int_rn(v), 0), 255);
}

// ---- CLAHE, as OpenCV does it: a clipped histogram per tile, turned into a lookup
// table, then each pixel blends the tables of its four nearest tiles.

// One block of 256 threads per tile. Tiles of tw x th may run past the image, where
// OpenCV pads with BORDER_REFLECT_101.
extern "C" __global__ void clahe_lut(const unsigned char* src, int w, int h, int tiles_x,
                                     int tw, int th, int clip, float scale, unsigned char* lut) {
    __shared__ int hist[256];
    __shared__ int excess;
    int t = threadIdx.x, tile = blockIdx.x;
    int x0 = (tile % tiles_x) * tw, y0 = (tile / tiles_x) * th;
    hist[t] = 0;
    if (t == 0) excess = 0;
    __syncthreads();
    for (int i = t; i < tw * th; i += 256)
        atomicAdd(&hist[src[reflect101(y0 + i / tw, h) * w + reflect101(x0 + i % tw, w)]], 1);
    __syncthreads();
    int v = hist[t];
    if (v > clip) {
        atomicAdd(&excess, v - clip);
        v = clip;
    }
    __syncthreads();
    // spread the clipped excess evenly, and the remainder one each over every step-th bin
    int batch = excess / 256, residual = excess - batch * 256;
    v += batch;
    if (residual > 0) {
        int step = max(256 / residual, 1);
        if (t % step == 0 && t / step < residual) v++;
    }
    hist[t] = v;
    __syncthreads();
    if (t == 0) {
        int sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += hist[i];
            lut[tile * 256 + i] = round_u8(sum * scale);
        }
    }
}

extern "C" __global__ void clahe_apply(const unsigned char* src, unsigned char* dst,
                                       const unsigned char* lut, int w, int h, int tiles_x,
                                       int tiles_y, float inv_tw, float inv_th) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float tyf = y * inv_th - 0.5f, txf = x * inv_tw - 0.5f;
    int ty1 = (int)floorf(tyf), tx1 = (int)floorf(txf);
    float ya = tyf - ty1, xa = txf - tx1;
    int ty2 = min(ty1 + 1, tiles_y - 1), tx2 = min(tx1 + 1, tiles_x - 1);
    ty1 = max(ty1, 0);
    tx1 = max(tx1, 0);
    int v = src[y * w + x];
    float top = lut[(ty1 * tiles_x + tx1) * 256 + v] * (1.f - xa) + lut[(ty1 * tiles_x + tx2) * 256 + v] * xa;
    float bot = lut[(ty2 * tiles_x + tx1) * 256 + v] * (1.f - xa) + lut[(ty2 * tiles_x + tx2) * 256 + v] * xa;
    dst[y * w + x] = round_u8(top * (1.f - ya) + bot * ya);
}

// ---- bilateral filter (8-bit gray): `space` holds (2r+1)^2 weights, negative outside
// the circle of radius r; `color` is the weight for each brightness difference 0-255.
extern "C" __global__ void bilateral(const unsigned char* src, unsigned char* dst, int w, int h,
                                     const float* space, const float* color, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int c = src[y * w + x], d = 2 * r + 1;
    float sum = 0.f, wsum = 0.f;
    for (int dy = -r; dy <= r; dy++) {
        const unsigned char* row = src + reflect101(y + dy, h) * w;
        for (int dx = -r; dx <= r; dx++) {
            float sw = space[(dy + r) * d + dx + r];
            if (sw < 0.f) continue;
            int v = row[reflect101(x + dx, w)];
            float wt = sw * color[abs(v - c)];
            sum += v * wt;
            wsum += wt;
        }
    }
    dst[y * w + x] = round_u8(sum / wsum);
}

// ---- Canny (aperture 3, L1 magnitude), as OpenCV does it.

// 3x3 Sobel with BORDER_REPLICATE, and |dx| + |dy|.
extern "C" __global__ void sobel(const unsigned char* src, int* dxo, int* dyo, int* mag, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    const unsigned char* a = src + max(y - 1, 0) * w;
    const unsigned char* b = src + y * w;
    const unsigned char* c = src + min(y + 1, h - 1) * w;
    int dx = (a[xp] + 2 * b[xp] + c[xp]) - (a[xm] + 2 * b[xm] + c[xm]);
    int dy = (c[xm] + 2 * c[x] + c[xp]) - (a[xm] + 2 * a[x] + a[xp]);
    int i = y * w + x;
    dxo[i] = dx;
    dyo[i] = dy;
    mag[i] = abs(dx) + abs(dy);
}

__device__ __forceinline__ int mag_at(const int* mag, int x, int y, int w, int h) {
    return (x < 0 || y < 0 || x >= w || y >= h) ? 0 : mag[y * w + x];  // zero outside
}

// Non-maximum suppression along the gradient, then the two thresholds:
// map = 2 for a sure edge, 1 for a maybe (kept if it touches a sure edge), 0 for none.
extern "C" __global__ void canny_nms(const int* dxa, const int* dya, const int* mag,
                                     unsigned char* map, int w, int h, int low, int high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x, m = mag[i];
    unsigned char out = 0;
    if (m > low) {
        const int TG22 = 13573;  // tan(22.5 degrees) in 15-bit fixed point
        int dx = dxa[i], dy = dya[i], xs = abs(dx), ys = abs(dy) << 15;
        int tg22x = xs * TG22;
        bool keep;
        if (ys < tg22x) {  // near-horizontal gradient: compare left and right
            keep = m > mag_at(mag, x - 1, y, w, h) && m >= mag_at(mag, x + 1, y, w, h);
        } else if (ys > tg22x + ((xs + xs) << 15)) {  // near-vertical: above and below
            keep = m > mag_at(mag, x, y - 1, w, h) && m >= mag_at(mag, x, y + 1, w, h);
        } else {  // diagonal
            int s = (dx ^ dy) < 0 ? -1 : 1;
            keep = m > mag_at(mag, x - s, y - 1, w, h) && m > mag_at(mag, x + s, y + 1, w, h);
        }
        if (keep) out = m > high ? 2 : 1;
    }
    map[i] = out;
}

// Hysteresis: turn maybes touching a sure edge into sure edges. Each 32x32 block repeats
// in shared memory until nothing changes, then flags `changed` so the host runs another
// pass (an edge can cross into the next block).
extern "C" __global__ void canny_grow(unsigned char* map, int w, int h, int* changed) {
    __shared__ unsigned char s[34][34];
    __shared__ int any, block_changed;
    int lx = threadIdx.x, ly = threadIdx.y;
    int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int i = ly * 32 + lx; i < 34 * 34; i += 1024) {
        int gx = bx + i % 34 - 1, gy = by + i / 34 - 1;
        s[i / 34][i % 34] = (gx < 0 || gy < 0 || gx >= w || gy >= h) ? 0 : map[gy * w + gx];
    }
    if (lx == 0 && ly == 0) block_changed = 0;
    do {
        __syncthreads();
        if (lx == 0 && ly == 0) any = 0;
        __syncthreads();
        if (s[ly + 1][lx + 1] == 1 &&
            (s[ly][lx] == 2 || s[ly][lx + 1] == 2 || s[ly][lx + 2] == 2 ||
             s[ly + 1][lx] == 2 || s[ly + 1][lx + 2] == 2 ||
             s[ly + 2][lx] == 2 || s[ly + 2][lx + 1] == 2 || s[ly + 2][lx + 2] == 2)) {
            s[ly + 1][lx + 1] = 2;
            any = 1;
            block_changed = 1;
        }
        __syncthreads();
    } while (any);
    int x = bx + lx, y = by + ly;
    if (x < w && y < h) map[y * w + x] = s[ly + 1][lx + 1];
    if (lx == 0 && ly == 0 && block_changed) *changed = 1;
}

// Sure edges -> 255, dilated by OpenCV's 2x2 kernel (anchor at its centre, so each
// pixel takes the max of itself, left, up and up-left).
extern "C" __global__ void edges_out(const unsigned char* map, unsigned char* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    bool e = map[y * w + x] == 2 || (x > 0 && map[y * w + x - 1] == 2) ||
             (y > 0 && (map[(y - 1) * w + x] == 2 || (x > 0 && map[(y - 1) * w + x - 1] == 2)));
    dst[y * w + x] = e ? 255 : 0;
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

    GROW_BATCH = 6  # Canny hysteresis passes per check (see _gray_and_edges)

    def __init__(self, width, height, seed=7):
        super().__init__(width, height, seed)
        mod = cp.RawModule(code=_SRC)
        self._k = {n: mod.get_function(n) for n in (
            "clahe_lut", "clahe_apply", "bilateral", "sobel", "canny_nms", "canny_grow", "edges_out",
            "blur_u8", "blur_f32", "combine", "warp_tint")}
        # CLAHE(clipLimit=2, 8x8 tiles) as Sketcher.clahe: OpenCV pads the image to whole
        # tiles, by a full extra tile per axis when either size doesn't divide evenly
        self._tiles = (8, 8)
        tx, ty = self._tiles
        pad = width % tx or height % ty
        tw = (width + (tx - width % tx if pad else 0)) // tx
        th = (height + (ty - height % ty if pad else 0)) // ty
        self._tile = (tw, th)
        self._clip = max(int(2.0 * tw * th / 256), 1)
        # bilateralFilter(d=7, sigmaColor=50, sigmaSpace=7) weights, built as OpenCV builds them
        r = 3
        ii, jj = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
        dist = np.sqrt(ii * ii + jj * jj)
        space = np.exp(dist * dist * (-0.5 / 7 ** 2))
        self._bf_space = cp.asarray(np.where(dist > r, -1.0, space).astype(np.float32).ravel())
        self._bf_color = cp.asarray(np.exp(np.arange(256.0) ** 2 * (-0.5 / 50 ** 2)).astype(np.float32))
        self._bf_r = r
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
        self._k["blur_u8"](self._grid, (32, 8), (src, tmp, k, np.int32(r), np.int32(w), np.int32(h),
                                                np.int32(1), np.int32(0), np.float32(scale), np.float32(offset)))
        self._k["blur_f32"](self._grid, (32, 8), (tmp, out, k, np.int32(r), np.int32(w), np.int32(h),
                                                 np.int32(0), np.int32(1)))
        return out

    @property
    def _grid(self):
        return ((self.w + 31) // 32, (self.h + 7) // 8)

    def _gray_and_edges(self, g, stream, flag):
        """GPU version of Sketcher.gray_and_edges, on a uint8 gray image already on the GPU."""
        k, w, h = self._k, self.w, self.h
        W, H = np.int32(w), np.int32(h)
        tx, ty = self._tiles
        tw, th = self._tile
        lut = cp.empty((tx * ty, 256), cp.uint8)
        k["clahe_lut"]((tx * ty,), (256,), (g, W, H, np.int32(tx), np.int32(tw), np.int32(th),
                                           np.int32(self._clip), np.float32(255.0 / (tw * th)), lut))
        gray = cp.empty((h, w), cp.uint8)
        k["clahe_apply"](self._grid, (32, 8), (g, gray, lut, W, H, np.int32(tx), np.int32(ty),
                                               np.float32(1.0 / tw), np.float32(1.0 / th)))
        smooth = cp.empty((h, w), cp.uint8)
        k["bilateral"](self._grid, (32, 8), (gray, smooth, W, H, self._bf_space, self._bf_color,
                                             np.int32(self._bf_r)))
        dx, dy, mag = (cp.empty((h, w), cp.int32) for _ in range(3))
        k["sobel"](self._grid, (32, 8), (smooth, dx, dy, mag, W, H))
        cmap = cp.empty((h, w), cp.uint8)
        k["canny_nms"](self._grid, (32, 8), (dx, dy, mag, cmap, W, H, np.int32(40), np.int32(110)))
        blocks = ((w + 31) // 32, (h + 31) // 32)
        # Edges usually settle in about 10 passes. Asking the GPU whether a pass changed
        # anything makes this thread wait, so run passes in batches and only ask about
        # the last one of each: no change there means done.
        while True:
            for _ in range(self.GROW_BATCH - 1):
                k["canny_grow"](blocks, (32, 32), (cmap, W, H, flag))
            flag.fill(0)
            k["canny_grow"](blocks, (32, 32), (cmap, W, H, flag))
            if not int(flag.get(stream=stream)[0]):
                break
        edges = cp.empty((h, w), cp.uint8)
        k["edges_out"](self._grid, (32, 8), (cmap, edges, W, H))
        return gray, edges

    def __call__(self, frame, t=0.0):
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tls = self._tls
        if not hasattr(tls, "stream"):
            tls.stream = cp.cuda.Stream(non_blocking=True)
            with tls.stream:
                tls.flag = cp.zeros(1, cp.int32)
        stream = tls.stream
        w, h, n = self.w, self.h, self.w * self.h
        with stream:
            g, e = self._gray_and_edges(cp.asarray(frame), stream, tls.flag)
            # blur(255 - gray) == 255 - blur(gray): the kernel sums to 1
            inv_blur = self._blur(g, self._k6, -1.0, 255.0)
            ink = self._blur(e, self._k07)
            tone = self._blur(g, self._k3, 1 / 255.0)
            out = cp.empty((h, w), cp.float32)
            self._k["combine"](((n + 255) // 256,), (256,),
                               (g, inv_blur, ink, tone, self._lut, self._hatch, self._paper, out,
                                np.int32(n), *self._thr))
            m = cv2.invertAffineTransform(self.wobble(t)).ravel()
            tint = [np.float32(v) for a, b in self.TINT for v in (255 * a, 255 * b)]
            res = cp.empty((h, w, 3), cp.uint8)
            self._k["warp_tint"](self._grid, (32, 8),
                                 (out, res, np.int32(w), np.int32(h), *map(np.float32, m), *tint))
            host = res.get(stream=stream)
        stream.synchronize()
        return host
