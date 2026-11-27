"""The GPU versions of OpenCV's and ffmpeg's steps must match them (up to the small,
measured differences gpu_sketch.py and nv_video.py describe). Skipped without a GPU."""
import subprocess

import cv2
import numpy as np
import pytest

from scribble.ffmpeg_tools import VideoDecoder, ffmpeg_exe, pick_decoder
from scribble.sketch import Sketcher

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def frames(clip):
    """Gray frames from each of the clip's four segments, decoded by ffmpeg."""
    _, _, w, h = pick_decoder(clip)
    with VideoDecoder(clip, w, h, (), gray=True) as d:
        return [f.copy() for i, f in enumerate(d) if i % 40 == 20]


def diff(a, b):
    d = np.abs(a.astype(int) - b.astype(int))
    return d.max(), (d > 0).mean() * 100


def gpu_gray_and_edges(sketcher, frame):
    """The GPU's CLAHE output and Canny edges for a frame (left in this thread's buffers)."""
    sketcher(frame)
    ws = sketcher._work()
    return ws.gray.get(), ws.edges.get()


def test_clahe_matches_opencv(gpu, frames):
    h, w = frames[0].shape
    s = gpu.GpuSketcher(w, h)
    for f in frames:
        gray, _ = gpu_gray_and_edges(s, f)
        worst, share = diff(gray, cv2.createCLAHE(2.0, (8, 8)).apply(f))
        assert worst <= 1 and share < 0.1, (worst, share)


def test_bilateral_matches_opencv(gpu, frames):
    h, w = frames[0].shape
    s = gpu.GpuSketcher(w, h)
    for f in frames:
        f = cv2.createCLAHE(2.0, (8, 8)).apply(f)
        out = gpu.cp.empty((h, w), gpu.cp.uint8)
        s._k["bilateral"](s._grid, (32, 8), (gpu.cp.asarray(f), out, np.int32(w), np.int32(h),
                                             s._bf_space, s._bf_color, np.int32(s._bf_r)))
        worst, share = diff(out.get(), cv2.bilateralFilter(f, 7, 50, 7))
        assert worst <= 1 and share < 0.01, (worst, share)


def test_edges_match_opencv(gpu, frames):
    h, w = frames[0].shape
    s, cpu = gpu.GpuSketcher(w, h), Sketcher(w, h)
    for f in frames:
        _, edges = gpu_gray_and_edges(s, f)
        _, share = diff(edges, cpu.gray_and_edges(f)[1])
        assert share < 0.05, share


def test_gpu_sketch_matches_cpu(gpu, frames):
    h, w = frames[0].shape
    s, cpu = gpu.GpuSketcher(w, h), Sketcher(w, h)
    for i, f in enumerate(frames):
        d = np.abs(s(f, t=i).bgr.astype(int) - cpu(f, t=i).astype(int))
        assert d.mean() < 0.15 and (d > 8).mean() * 100 < 0.5, (d.mean(), (d > 8).mean() * 100)


def test_yuv_matches_ffmpeg(gpu, frames):
    """The encoder-ready YUV the GPU makes vs ffmpeg's own BT.709 conversion of the
    same BGR sketch: luma exactly, chroma within one level."""
    h, w = frames[0].shape
    s = gpu.GpuSketcher(w, h)
    sketches = [s(f, t=i) for i, f in enumerate(frames)]
    r = subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                        "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                        "-f", "rawvideo", "-"], input=b"".join(k.bgr.tobytes() for k in sketches),
                       capture_output=True, check=True)
    ref = np.frombuffer(r.stdout, np.uint8).reshape(len(sketches), -1)
    for k, want in zip(sketches, ref):
        assert np.array_equal(k.yuv[:w * h], want[:w * h])
        assert np.abs(k.yuv[w * h:].astype(int) - want[w * h:]).max() <= 1


def test_nv12_is_i420_interleaved(gpu, frames):
    h, w = frames[0].shape
    i420 = gpu.GpuSketcher(w, h)(frames[1], t=1.0).yuv
    nv12 = gpu.GpuSketcher(w, h, keep_on_gpu=True)(frames[1], t=1.0).nv12.get().ravel()
    n, q = w * h, w * h // 4
    assert np.array_equal(nv12[:n], i420[:n])
    assert np.array_equal(nv12[n::2], i420[n:n + q]) and np.array_equal(nv12[n + 1::2], i420[n + q:])


def test_nvdec_gray_matches_ffmpeg(nv, labelled_clip):
    """NVDEC's frames, made gray as ffmpeg does (nv12_gray, following the video's colour
    labels), vs ffmpeg's gray: ffmpeg's fixed-point rounding can't be matched exactly,
    so a small difference is expected."""
    _, _, w, h = pick_decoder(labelled_clip)
    with VideoDecoder(labelled_clip, w, h, (), gray=True) as d:
        want = [f.copy() for i, f in enumerate(d) if i % 40 == 20]
    with nv.NvDecoder(labelled_clip, w, h) as d:
        ours = [f.get() for i, f in enumerate(d) if i % 40 == 20]
    assert len(ours) == len(want)
    # measured: means 0.5 (BT.601) to 0.8 (BT.709), at most 2 levels apart; before the
    # matrix followed the labels and BT.709 clipped, these were 19 and 4 levels apart
    for a, b in zip(ours, want):
        d = np.abs(a.astype(int) - b.astype(int))
        assert d.mean() < 1.0 and d.max() <= 3, (d.mean(), d.max())
