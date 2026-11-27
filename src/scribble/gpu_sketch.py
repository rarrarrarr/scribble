"""The scribble look with most of the per-frame work on an NVIDIA GPU (CuPy).

Optional: needs the `gpu` extra (see README). Importing this module raises
ImportError when CuPy is missing; GpuSketcher() raises when no GPU can run it.

Every step runs here as small CUDA kernels, replayed per frame as CUDA graphs. With
ffmpeg decoding and encoding, the CPU only uploads the gray frame and downloads the
sketch, already converted to the YUV the encoder takes (half the bytes of BGR, and
no conversion left for ffmpeg); with nv_video (keep_on_gpu), frames arrive and leave
on the GPU. Either way the BGR image is downloaded only for the frames that become
shots. CuPy has no CLAHE, bilateral filter or Canny, so this file has its own,
written to follow OpenCV's (tiles, rounding, borders and thresholds alike): on real
frames they match OpenCV exactly but for about 0.02% of pixels.

Output matches the CPU Sketcher except for scattered single pixels (about 0.1-0.2%
of them): OpenCV's 8-bit blur rounds in fixed point, the float blur here can land
one level away, and the pencil table is steep enough in places to turn that into
a visible step. Side by side the frames look the same. A change to the look in
sketch.py needs the same change here.
"""
import threading
import warnings
from functools import lru_cache

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
// Blocks of 32x8 threads; r is at most BF_R. The block's pixels and their r-wide border
// are loaded into shared memory once, rather than every thread reading its (2r+1)^2
// neighbours from global memory.
#define BF_R 3
extern "C" __global__ void bilateral(const unsigned char* src, unsigned char* dst, int w, int h,
                                     const float* space, const float* color, int r) {
    __shared__ unsigned char tile[8 + 2 * BF_R][32 + 2 * BF_R];
    __shared__ float ctab[256];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int x0 = blockIdx.x * 32 - r, y0 = blockIdx.y * 8 - r, tw = 32 + 2 * r, th = 8 + 2 * r;
    for (int i = tid; i < tw * th; i += 256)
        tile[i / tw][i % tw] = src[reflect101(y0 + i / tw, h) * w + reflect101(x0 + i % tw, w)];
    ctab[tid] = color[tid];
    __syncthreads();
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 8 + threadIdx.y;
    if (x >= w || y >= h) return;
    int c = tile[threadIdx.y + r][threadIdx.x + r], d = 2 * r + 1;
    float sum = 0.f, wsum = 0.f;
    for (int dy = -r; dy <= r; dy++) {
        const unsigned char* row = tile[threadIdx.y + r + dy];
        for (int dx = -r; dx <= r; dx++) {
            float sw = space[(dy + r) * d + dx + r];
            if (sw < 0.f) continue;
            int v = row[threadIdx.x + r + dx];
            float wt = sw * ctab[abs(v - c)];
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
// Pass number `pass` runs every block if it is 0, otherwise only the blocks the previous
// pass stamped in `stamp`: a block whose neighbourhood didn't change in the last pass
// can't change now (it settled then). A block that changes stamps itself and its eight
// neighbours with this pass's number. (A stale stamp from an earlier frame only costs
// a block some needless work, never skips one, so stamps are never cleared.)
extern "C" __global__ void canny_grow(unsigned char* map, int w, int h, int* changed,
                                      int* stamp, int pass) {
    int nbx = gridDim.x, nby = gridDim.y;
    if (pass > 0 && stamp[blockIdx.y * nbx + blockIdx.x] != pass - 1) return;
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
    if (block_changed && ly < 3 && lx < 3) {  // 9 threads mark the 3x3 blocks around this one
        int ax = (int)blockIdx.x + lx - 1, ay = (int)blockIdx.y + ly - 1;
        if (ax >= 0 && ay >= 0 && ax < nbx && ay < nby) stamp[ay * nbx + ax] = pass;
        if (lx == 1 && ly == 1) *changed = 1;
    }
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

__device__ __forceinline__ float row_blur(const unsigned char* row, int x, int w, const float* k, int r) {
    float s = 0.f;
    for (int i = -r; i <= r; i++) s += k[i + r] * row[reflect101(x + i, w)];
    return s;
}

__device__ __forceinline__ float col_blur(const float* src, int x, int y, int w, int h, const float* k, int r) {
    float s = 0.f;
    for (int i = -r; i <= r; i++) s += k[i + r] * src[reflect101(y + i, h) * w + x];
    return s;
}

// Row passes of the three Gaussian blurs, in one launch: blur(255 - gray) (as
// 255 - blur(gray), the kernel summing to 1), blur(gray / 255) and blur(edges).
// (Staging rows in shared memory measured no faster here: the cache does it already.)
extern "C" __global__ void blur_rows(const unsigned char* gray, const unsigned char* edges,
                                     const float* k_inv, int r_inv, const float* k_tone, int r_tone,
                                     const float* k_ink, int r_ink,
                                     float* inv, float* tone, float* ink, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    inv[i] = row_blur(gray + y * w, x, w, k_inv, r_inv) * -1.f + 255.f;
    tone[i] = row_blur(gray + y * w, x, w, k_tone, r_tone) * (1.f / 255.f);
    ink[i] = row_blur(edges + y * w, x, w, k_ink, r_ink);
}

// The blurs' column passes, then pencil * ink * (1 - strongest hatch layer) * paper,
// as Sketcher.__call__ does it.
extern "C" __global__ void combine(const unsigned char* gray, const float* inv_rows,
                                   const float* tone_rows, const float* ink_rows,
                                   const float* k_inv, int r_inv, const float* k_tone, int r_tone,
                                   const float* k_ink, int r_ink, const float* lut,
                                   const float* hatch, const float* paper, float* out,
                                   int w, int h, float t0, float t1, float t2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x, n = w * h;
    int b = __float2int_rn(col_blur(inv_rows, x, y, w, h, k_inv, r_inv));  // the CPU blur returns uint8
    float v = lut[gray[i] * 256 + min(max(b, 0), 255)];
    v *= 1.f + col_blur(ink_rows, x, y, w, h, k_ink, r_ink) * (-0.85f / 255.f);
    float tn = col_blur(tone_rows, x, y, w, h, k_tone, r_tone) * (-1.f / 0.12f), dark = 0.f;
    dark = fmaxf(dark, fminf(fmaxf(tn + t0, 0.f), 1.f) * hatch[i]);
    dark = fmaxf(dark, fminf(fmaxf(tn + t1, 0.f), 1.f) * hatch[n + i]);
    dark = fmaxf(dark, fminf(fmaxf(tn + t2, 0.f), 1.f) * hatch[2 * n + i]);
    out[i] = v * (1.f - dark) * paper[i];
}

// BGR -> YUV 4:2:0, BT.709 limited range, the format the video encoder takes: I420
// (all Y, then U, then V) or, with nv12, NV12 (all Y, then U and V interleaved).
// One thread per 2x2 block; chroma from the block's average. Truncating, as ffmpeg's
// own conversion does: Y matches it exactly; chroma is within one level (ffmpeg
// filters chroma a little differently), invisible on a pale tint.
extern "C" __global__ void to_yuv420(const unsigned char* bgr, unsigned char* yuv, int w, int h, int nv12) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y * blockDim.y + threadIdx.y;
    int cw = w / 2, ch = h / 2;
    if (cx >= cw || cy >= ch) return;
    float sb = 0.f, sg = 0.f, sr = 0.f;
    for (int j = 0; j < 4; j++) {
        int x = 2 * cx + (j & 1), y = 2 * cy + (j >> 1);
        const unsigned char* p = bgr + 3 * (y * w + x);
        float l = 0.0722f * p[0] + 0.7152f * p[1] + 0.2126f * p[2];
        yuv[y * w + x] = (unsigned char)(16.f + l * (219.f / 255.f));  // truncated, as ffmpeg does
        sb += p[0]; sg += p[1]; sr += p[2];
    }
    sb *= 0.25f; sg *= 0.25f; sr *= 0.25f;
    float l = 0.0722f * sb + 0.7152f * sg + 0.2126f * sr;
    unsigned char u = (unsigned char)fminf(fmaxf(128.f + (sb - l) * (224.f / 255.f / 1.8556f), 0.f), 255.f);
    unsigned char v = (unsigned char)fminf(fmaxf(128.f + (sr - l) * (224.f / 255.f / 1.5748f), 0.f), 255.f);
    if (nv12) {
        yuv[w * h + cy * w + 2 * cx] = u;
        yuv[w * h + cy * w + 2 * cx + 1] = v;
    } else {
        yuv[w * h + cy * cw + cx] = u;
        yuv[w * h + cw * ch + cy * cw + cx] = v;
    }
}

__device__ __forceinline__ int reflect(int i, int n) {  // OpenCV BORDER_REFLECT
    return i < 0 ? -i - 1 : (i >= n ? 2 * n - 1 - i : i);
}

// Wobble (bilinear warp by the inverse affine m) and paper tint, straight to BGR uint8.
// p: the inverse affine m0..m5, then the tint a0, b0, a1, b1, a2, b2 (in GPU memory, so a
// recorded CUDA graph can replay this with each frame's wobble).
extern "C" __global__ void warp_tint(const float* src, unsigned char* bgr, int w, int h, const float* p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float m0 = p[0], m1 = p[1], m2 = p[2], m3 = p[3], m4 = p[4], m5 = p[5];
    float a0 = p[6], b0 = p[7], a1 = p[8], b1 = p[9], a2 = p[10], b2 = p[11];
    float sx = m0 * x + m1 * y + m2, sy = m3 * x + m4 * y + m5;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0;
    int xa = reflect(x0, w), xb = reflect(x0 + 1, w), ya = reflect(y0, h), yb = reflect(y0 + 1, h);
    float v = (src[ya * w + xa] * (1.f - fx) + src[ya * w + xb] * fx) * (1.f - fy)
            + (src[yb * w + xa] * (1.f - fx) + src[yb * w + xb] * fx) * fy;
    unsigned char* o = bgr + 3 * (y * w + x);
    o[0] = (unsigned char)(v * a0 + b0);  // truncates, like numpy's astype(uint8)
    o[1] = (unsigned char)(v * a1 + b1);
    o[2] = (unsigned char)(v * a2 + b2);
}
"""


_KERNELS = ("clahe_lut", "clahe_apply", "bilateral", "sobel", "canny_nms", "canny_grow",
            "edges_out", "blur_rows", "combine", "warp_tint", "to_yuv420")


@lru_cache(maxsize=None)
def kernels():
    """The compiled CUDA kernels by name. Compiling takes a moment (CuPy caches the
    result on disk), so the CLI calls this early, alongside its other startup work."""
    mod = cp.RawModule(code=_SRC)
    return {n: mod.get_function(n) for n in _KERNELS}


def _gauss(sigma, for_uint8):
    """OpenCV's kernel for GaussianBlur(ksize=(0, 0), sigma), whose size depends on the depth."""
    ksize = int(round(sigma * (3 if for_uint8 else 4) * 2 + 1)) | 1
    return cp.asarray(cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel()), np.int32(ksize // 2)


class GpuSketch:
    """One sketched frame from GpuSketcher.

    yuv: the frame as ffmpeg's video encoder takes it (I420, BT.709 limited range,
    uint8, in pinned memory, which copies off the GPU faster), or None if the frame's
    width or height is odd, which 4:2:0 can't represent, or with keep_on_gpu.
    nv12: with keep_on_gpu, the frame as NVENC takes it, left on the GPU: a CuPy array
    of height * 3/2 rows, Y then interleaved UV, BT.709 limited range.
    bgr: the frame as a BGR image like Sketcher returns, copied off the GPU only when
    first asked for (only shots need it).
    """
    __slots__ = ("yuv", "nv12", "_dev", "_bgr")

    def __init__(self, yuv, dev, nv12=None):
        self.yuv, self.nv12, self._dev, self._bgr = yuv, nv12, dev, None

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr, self._dev = cp.asnumpy(self._dev), None
        return self._bgr


class _Work:
    """One thread's CUDA stream, fixed buffers, and the kernel launches recorded as two
    CUDA graphs over them (see GpuSketcher.__call__)."""


class GpuSketcher(Sketcher):
    """Like Sketcher, but calling it returns a GpuSketch (see there). Safe to call from
    several threads at once: each thread gets its own CUDA stream and buffers, so one
    frame's copies overlap another's kernels.

    The kernels themselves take a fraction of a millisecond per frame; launching some
    40 of them one by one from Python took longer, so each thread records its launches
    once as CUDA graphs and replays them per frame with a single call each.
    """

    GROW_IN_GRAPH = 12  # Canny hysteresis passes recorded in the graph: most frames settle in ~10
    GROW_BATCH = 6      # passes per check when a frame needs more

    def __init__(self, width, height, seed=7, keep_on_gpu=False):
        # keep_on_gpu: frames come in as CuPy arrays (from nv_video.NvDecoder) and go
        # out as GpuSketch.nv12 for nv_video.NvEncoder; nothing is copied to the CPU.
        super().__init__(width, height, seed)
        self._k = kernels()
        self.keep_on_gpu = keep_on_gpu
        even = width % 2 == 0 and height % 2 == 0
        self.pix_fmt = "nv12" if keep_on_gpu else "yuv420p" if even else "bgr24"
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
        assert self._bf_r <= 3, "raise BF_R in the kernels"
        self._tls = threading.local()
        # compile and run once now, so a GPU that can't do this fails here, not mid-video
        self(np.zeros((height, width, 3), np.uint8))

    @property
    def _grid(self):
        return ((self.w + 31) // 32, (self.h + 7) // 8)

    def _work(self):
        """This thread's _Work, set up (buffers allocated, graphs recorded) on first use."""
        ws = getattr(self._tls, "work", None)
        if ws is not None:
            return ws
        ws = self._tls.work = _Work()
        h, w = self.h, self.w
        ws.stream = cp.cuda.Stream(non_blocking=True)
        with ws.stream:
            u8 = lambda *shape: cp.empty(shape or (h, w), cp.uint8)
            f32 = lambda: cp.empty((h, w), cp.float32)
            ws.g, ws.gray, ws.smooth, ws.cmap, ws.edges = u8(), u8(), u8(), u8(), u8()
            ws.lut = u8(self._tiles[0] * self._tiles[1], 256)
            ws.dx, ws.dy, ws.mag = (cp.empty((h, w), cp.int32) for _ in range(3))
            ws.inv, ws.tone, ws.ink, ws.out = f32(), f32(), f32(), f32()
            ws.bgr = u8(h, w, 3)
            ws.yuv = u8(h * 3 // 2, w)
            ws.params = cp.zeros(12, cp.float32)  # wobble and tint, see warp_tint
            ws.flag = cp.zeros(1, cp.int32)
            n_blocks = ((w + 31) // 32) * ((h + 31) // 32)
            ws.stamp = cp.full(n_blocks, -1, cp.int32)  # see canny_grow
            ws.stream.synchronize()
            # Thread-local capture: other threads keep allocating while this records.
            mode = cp.cuda.runtime.streamCaptureModeThreadLocal
            ws.stream.begin_capture(mode)
            self._front(ws)
            ws.front = ws.stream.end_capture()
            ws.stream.begin_capture(mode)
            self._back(ws)
            ws.back = ws.stream.end_capture()
        return ws

    def _front(self, ws):
        """GPU version of Sketcher.gray_and_edges, from ws.g up to Canny's hysteresis
        (GROW_IN_GRAPH passes, the last one setting ws.flag if it still changed anything)."""
        k, w, h = self._k, self.w, self.h
        W, H = np.int32(w), np.int32(h)
        tx, ty = self._tiles
        tw, th = self._tile
        k["clahe_lut"]((tx * ty,), (256,), (ws.g, W, H, np.int32(tx), np.int32(tw), np.int32(th),
                                           np.int32(self._clip), np.float32(255.0 / (tw * th)), ws.lut))
        k["clahe_apply"](self._grid, (32, 8), (ws.g, ws.gray, ws.lut, W, H, np.int32(tx), np.int32(ty),
                                               np.float32(1.0 / tw), np.float32(1.0 / th)))
        k["bilateral"](self._grid, (32, 8), (ws.gray, ws.smooth, W, H, self._bf_space, self._bf_color,
                                             np.int32(self._bf_r)))
        k["sobel"](self._grid, (32, 8), (ws.smooth, ws.dx, ws.dy, ws.mag, W, H))
        k["canny_nms"](self._grid, (32, 8), (ws.dx, ws.dy, ws.mag, ws.cmap, W, H, np.int32(40), np.int32(110)))
        self._grow(ws, self.GROW_IN_GRAPH)

    def _grow(self, ws, passes):
        """Canny hysteresis passes; ws.flag ends up set if the last one changed anything.
        The first pass looks at every block; later ones only at blocks next to a change."""
        W, H = np.int32(self.w), np.int32(self.h)
        blocks = ((self.w + 31) // 32, (self.h + 31) // 32)
        for i in range(passes):
            if i == passes - 1:
                cp.cuda.runtime.memsetAsync(ws.flag.data.ptr, 0, 4, ws.stream.ptr)
            self._k["canny_grow"](blocks, (32, 32), (ws.cmap, W, H, ws.flag, ws.stamp, np.int32(i)))

    def _back(self, ws):
        """From the finished Canny map to the sketch: ws.bgr, and ws.yuv unless BGR is wanted."""
        k, w, h = self._k, self.w, self.h
        W, H = np.int32(w), np.int32(h)
        k["edges_out"](self._grid, (32, 8), (ws.cmap, ws.edges, W, H))
        blurs = (*self._k6, *self._k3, *self._k07)  # (weights, radius) for inv, tone, ink
        k["blur_rows"](self._grid, (32, 8), (ws.gray, ws.edges, *blurs, ws.inv, ws.tone, ws.ink, W, H))
        k["combine"](self._grid, (32, 8), (ws.gray, ws.inv, ws.tone, ws.ink, *blurs, self._lut, self._hatch,
                                           self._paper, ws.out, W, H, *self._thr))
        k["warp_tint"](self._grid, (32, 8), (ws.out, ws.bgr, W, H, ws.params))
        if self.pix_fmt in ("yuv420p", "nv12"):
            k["to_yuv420"](((w // 2 + 31) // 32, (h // 2 + 7) // 8), (32, 8),
                           (ws.bgr, ws.yuv, W, H, np.int32(self.pix_fmt == "nv12")))

    def __call__(self, frame, t=0.0):
        """frame: uint8 gray (2-D) or BGR, a NumPy array, or a CuPy one already on the GPU."""
        if frame.ndim == 3:
            frame = cv2.cvtColor(cp.asnumpy(frame), cv2.COLOR_BGR2GRAY)
        ws = self._work()
        s = ws.stream
        m = cv2.invertAffineTransform(self.wobble(t)).ravel()
        params = np.array([*m, *(v for a, b in self.TINT for v in (255 * a, 255 * b))], np.float32)
        with s:
            if isinstance(frame, cp.ndarray):
                ws.g[...] = frame
            else:
                ws.g.set(np.ascontiguousarray(frame), s)
            ws.params.set(params, s)
            ws.front.launch(s)
            # Edges usually settle within the recorded passes. Asking the GPU whether the
            # last one changed anything makes this thread wait, so when more are needed,
            # run them in batches and only ask about the last of each.
            while int(ws.flag.get(stream=s)[0]):
                self._grow(ws, self.GROW_BATCH)
            ws.back.launch(s)
            # copies the caller keeps, so the buffers are free for this thread's next frame
            bgr = ws.bgr.copy()
            yuv = nv12 = None
            if self.pix_fmt == "nv12":
                nv12 = ws.yuv.copy()
            elif self.pix_fmt == "yuv420p":
                yuv = np.frombuffer(cp.cuda.alloc_pinned_memory(ws.yuv.nbytes), np.uint8, ws.yuv.size)
                ws.yuv.ravel().get(stream=s, out=yuv)
        s.synchronize()
        return GpuSketch(yuv, bgr, nv12)
