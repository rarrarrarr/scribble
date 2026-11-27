"""Decode and encode on NVIDIA's video engines (NVDEC, NVENC) with frames kept in GPU
memory the whole way, via NVIDIA's PyNvVideoCodec. Part of the `gpu` extra.

cli.py uses these in place of ffmpeg_tools' VideoDecoder, SceneDetector and
VideoEncoder when nv_problem() finds nothing in the way (the codec, the GPU, ...):
then no frame goes to the CPU except the few that become shot pictures, and ffmpeg
only writes the .mp4 from the compressed video, adding the audio.

They reproduce what the ffmpeg pipeline produces:
- gray frames: ffmpeg's `-pix_fmt gray` goes through RGB (BT.709) and BT.601 luma
  weights rather than using Y, and nv12_gray does the same; ffmpeg's fixed-point
  arithmetic still lands one level away now and then (about a quarter of pixels,
  0.26 levels on average);
- scene cuts: ffmpeg's `select` scene score (the mean absolute difference of the Y
  plane from the previous frame, against the one before), exactly: the same cuts.
"""
import os
import queue
import subprocess
import threading
from fractions import Fraction
from functools import lru_cache

import numpy as np

from .ffmpeg_tools import CutTracker, audio_args, ffmpeg_exe
from .gpu_sketch import cp  # imports CuPy with its warning filter

# PyNvVideoCodec's own logger (read at import) warns about seek accuracy on every
# .mkv/.webm opened; nothing here seeks. Errors still show.
os.environ.setdefault("LOGGER_LEVEL", "ERROR")
import PyNvVideoCodec as nvc  # noqa: E402

_SRC = r"""
// NV12 -> gray as ffmpeg's `-pix_fmt gray` makes it: limited-range BT.709 YUV to RGB,
// then BT.601 luma weights, truncated; chroma taken from the nearest sample.
extern "C" __global__ void nv12_gray(const unsigned char* y, int ypitch, const unsigned char* uv,
                                     int uvpitch, unsigned char* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || r >= h) return;
    float Y = (y[r * ypitch + x] - 16.f) * (255.f / 219.f);
    const unsigned char* c = uv + (r / 2) * uvpitch + (x / 2) * 2;
    float cb = (c[0] - 128.f) * (255.f / 224.f), cr = (c[1] - 128.f) * (255.f / 224.f);
    float red = Y + 1.5748f * cr, green = Y - 0.1873f * cb - 0.4681f * cr, blue = Y + 1.8556f * cb;
    float v = floorf(0.299f * red + 0.587f * green + 0.114f * blue);
    dst[r * w + x] = (unsigned char)fminf(fmaxf(v, 0.f), 255.f);
}

// Copy a decoded Y plane (rows `pitch` apart) to `dst`, adding the sum of |Y - prev| over
// all pixels to *sad: the input to ffmpeg's scene score. 256 threads per block.
extern "C" __global__ void sad_copy(const unsigned char* src, int pitch, const unsigned char* prev,
                                    unsigned char* dst, int w, int h, unsigned long long* sad) {
    __shared__ unsigned long long part[256];
    unsigned long long s = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < w * h; i += gridDim.x * blockDim.x) {
        unsigned char v = src[(i / w) * pitch + i % w];
        dst[i] = v;
        s += abs((int)v - (int)prev[i]);
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if (threadIdx.x < k) part[threadIdx.x] += part[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(sad, part[0]);
}
"""


@lru_cache(maxsize=None)
def _module():
    return cp.RawModule(code=_SRC)


def _kernel(name):
    return _module().get_function(name)


def _gray_kernel():
    return _kernel("nv12_gray")


def _context():
    """CuPy's CUDA context, so the decoder and encoder use the same one."""
    cp.cuda.Device(0).use()
    cp.zeros(1)  # make sure the context exists
    return cp.cuda.driver.ctxGetCurrent()


def _batches(video, stream, size=16):
    """Decoded frames in lists of up to `size`, each (Y address, Y pitch, UV address,
    UV pitch, pts in seconds from the first frame), valid until the next list.

    NVDEC runs in PyNvVideoCodec's own C++ thread (ThreadedDecoder), ahead of this
    one. Python time is what limits the render, not the GPU: decoding one frame per
    call from a Python thread, competing for the GIL, took longer than all the rest.
    """
    dmx = nvc.CreateDemuxer(str(video))
    tb = dmx.GetTimebaseNum() / dmx.GetTimebaseDen()
    del dmx
    dec = nvc.ThreadedDecoder(str(video), buffer_size=2 * size, cuda_context=_context(),
                              cuda_stream=stream.ptr, use_device_memory=True)
    pts0 = None
    try:
        while batch := dec.get_batch_frames(size):
            out = []
            for f in batch:
                y, uv = (p.__cuda_array_interface__ for p in f.cuda())
                pts0 = f.getPTS() if pts0 is None else pts0
                # rounded as ffmpeg prints it (%.6g), which the ffmpeg pipeline's cut times
                # come from: shot middles are computed from cut times, and a 1e-15
                # difference can move one to the neighbouring frame
                t = float(f"{(f.getPTS() - pts0) * tb:.6g}")
                out.append((y["data"][0], y["strides"][0], uv["data"][0], uv["strides"][0], t))
            yield out
    finally:
        dec.end()


def nv_problem(video, width, height):
    """None if `video` can go through NvDecoder/NvEncoder here at width x height (as
    ffmpeg decodes it), else why not, in a few words."""
    if width % 2 or height % 2:
        return "odd frame size"
    try:
        dmx = nvc.CreateDemuxer(str(video))
        if (dmx.Width(), dmx.Height()) != (width, height):  # e.g. rotated: ffmpeg turns those
            return f"NVDEC frame size {dmx.Width()}x{dmx.Height()} differs from {width}x{height}"
        stream = cp.cuda.Stream(non_blocking=True)
        batches = _batches(video, stream, 1)
        next(batches)
        batches.close()
        enc = nvc.CreateEncoder(width, height, "NV12", False, codec="h264", cudacontext=_context())
        enc.EndEncode()
        _module()
    except StopIteration:
        return "NVDEC decoded no frames"
    except Exception as e:  # unsupported codec, no NVDEC/NVENC, driver too old, ...
        # PyNvVideoCodec's messages run to several lines of source locations: keep the first
        first = str(e).strip().splitlines()[0].strip(" :") if str(e).strip() else type(e).__name__
        return f"this video's codec or size isn't supported ({first[:60]})"
    return None


class NvDecoder:
    """Iterate over a video's frames as gray uint8 CuPy arrays (height, width) on the
    GPU, matching ffmpeg_tools.VideoDecoder(gray=True) (see the module docstring).
    Decodes in a background thread. Use as a context manager."""

    def __init__(self, video, width, height):
        self.video, self.w, self.h = video, width, height
        self._frames = queue.Queue(maxsize=8)
        self._error = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            stream = cp.cuda.Stream(non_blocking=True)
            kernel = _gray_kernel()
            w, h = self.w, self.h
            grid = ((w + 31) // 32, (h + 7) // 8)
            W, H = np.int32(w), np.int32(h)
            for batch in _batches(self.video, stream):
                if self._stop.is_set():
                    return
                grays = []
                with stream:
                    for y, ypitch, uv, uvpitch, _ in batch:
                        grays.append(cp.empty((h, w), cp.uint8))
                        kernel(grid, (32, 8), (np.uint64(y), np.int32(ypitch), np.uint64(uv),
                                               np.int32(uvpitch), grays[-1], W, H))
                stream.synchronize()  # done before the decoder reuses its surfaces, and before use
                for gray in grays:
                    self._frames.put(gray)
        except Exception as e:
            self._error = e
        finally:
            self._frames.put(None)

    def __iter__(self):
        while (frame := self._frames.get()) is not None:
            yield frame
        if self._error:
            raise RuntimeError(f"NVDEC decoding failed: {self._error}") from self._error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._stop.set()
        while not self._frames.empty():  # unblock the thread if it waits to hand over a frame
            self._frames.get_nowait()


class NvSceneDetector(CutTracker):
    """ffmpeg_tools.SceneDetector's job (the same cuts), with its own decode on the GPU,
    in a background thread. Use as a context manager."""

    BATCH = 64  # frames scored per copy of results to the CPU

    def __init__(self, video, width, height, threshold=0.3):
        super().__init__(threshold)
        self.video, self.size = video, (width, height)
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            stream = cp.cuda.Stream(non_blocking=True)
            kernel = _kernel("sad_copy")
            ys = None  # two Y-plane buffers, the current frame and the previous one
            sads = None
            first = True
            prev_mafd = 0.0
            for batch in _batches(self.video, stream, self.BATCH):
                if self._stop.is_set():
                    return
                if ys is None:
                    w, h = self.size
                    with stream:
                        ys = [cp.empty(w * h, cp.uint8) for _ in range(2)]
                        sads = cp.zeros(self.BATCH, cp.uint64)
                with stream:
                    sads[:len(batch)] = 0
                    for j, (y, ypitch, _, _, _) in enumerate(batch):
                        cur, prev = ys[j % 2], ys[1 - j % 2]
                        kernel((128,), (256,), (np.uint64(y), np.int32(ypitch), prev, cur,
                                                np.int32(w), np.int32(h), sads[j:]))
                    if len(batch) % 2:  # keep "previous" in ys[1] for the next batch's first frame
                        ys.reverse()
                    values = sads[:len(batch)].get().tolist()
                scores = []
                for (*_, t), s in zip(batch, values):
                    score = 0.0  # the first frame has nothing to compare with
                    if not first:
                        # as ffmpeg's select filter: mafd, and its change from the previous frame
                        mafd = s / (w * h)
                        diff = abs(mafd - prev_mafd)
                        score = float(np.float32(min(max(min(mafd, diff) / 100.0, 0.0), 1.0)))  # av_clipf: float
                        prev_mafd = mafd
                    first = False
                    scores.append((t, score))
                self._scored(scores)
            self._finished()
        except Exception as e:
            self._finished(RuntimeError(f"scene detection on the GPU failed: {e}"))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._stop.set()


class _NV12:
    """What NVENC's Encode() takes: an object whose cuda() lists the planes."""

    def __init__(self, frame, height):
        w = frame.shape[1]
        self.planes = [frame[:height].reshape(height, w, 1), frame[height:].reshape(height // 2, w // 2, 2)]

    def cuda(self):
        return self.planes


class NvEncoder:
    """H.264 on NVENC from frames on the GPU (GpuSketch.nv12), into an .mp4 with the
    audio of `audio_from`, like ffmpeg_tools.VideoEncoder(encoder='nvenc'): the same
    quality setting for `crf`, and labelled BT.709. Encodes in a background thread;
    ffmpeg only puts the compressed video and the audio in the file. Use as a
    context manager, calling write(frame) for each frame."""

    def __init__(self, out, width, height, fps, audio_from=None, crf=24):
        rate = Fraction(fps).limit_denominator(100000)  # e.g. 29.97002997 -> 30000/1001
        cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "h264", "-framerate", str(rate), "-i", "-"]
        cmd += audio_args(audio_from, out)
        # raw H.264 carries no timestamps (without them the .mp4 muxer drops the audio):
        # number the frames; and label the colours, as VideoEncoder does
        cmd += ["-c:v", "copy", "-bsf:v",
                f"setts=ts=N*{rate.denominator}/({rate.numerator}*TB),"
                "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:"
                "video_full_range_flag=0", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # As ffmpeg_tools.VideoEncoder's NVENC settings: preset p4, high quality tuning,
        # constant quality crf - 3 (the same quality as x264 at crf), no B-frames.
        self._opts = dict(codec="h264", preset="P4", tuning_info="high_quality", rc="vbr",
                          cq=str(max(crf - 3, 0)), bitrate="0", fps=f"{fps:.6f}", gop="250", bf="0")
        self.w, self.h = width, height
        self._frames = queue.Queue(maxsize=8)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        enc = None
        while True:
            frame = self._frames.get()
            if self._error is not None:  # after a failure, keep draining so write() never blocks
                if frame is None:
                    return
                continue
            try:
                if enc is None:
                    enc = nvc.CreateEncoder(self.w, self.h, "NV12", False, cudacontext=_context(), **self._opts)
                packets = enc.EndEncode() if frame is None else enc.Encode(_NV12(frame, self.h))
                self.proc.stdin.write(b"".join(p["data"] for p in packets))
            except Exception as e:  # includes a broken pipe if ffmpeg stopped
                self._error = e
            if frame is None:
                return

    def write(self, frame):
        if self._error is not None:
            self.close()
        self._frames.put(frame)

    def close(self):
        if self._thread.is_alive():
            self._frames.put(None)
            self._thread.join()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg muxer exited with code {rc}")
        if self._error is not None:
            raise RuntimeError(f"NVENC encoding failed: {self._error}") from self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:  # don't mask the original error; just stop
            self.proc.kill()
            self.proc.wait()
            self._frames.put(None)
