"""Shared test setup: a generated test clip, and skipping GPU tests where they can't run.

Run everything with `uv run pytest`. Tests marked `gpu` need an NVIDIA GPU and the
`gpu` extra (`uv sync --extra gpu`); without them they are skipped, with the reason
listed at the end of the run.
"""
import re
import subprocess

import pytest

from scribble.ffmpeg_tools import ffmpeg_exe

W, H, FPS = 640, 360, 25
SEGMENT = 1.6  # seconds per segment: four segments, so three scene cuts


def pytest_addoption(parser):
    parser.addoption("--update-look", action="store_true",
                     help="rewrite the reference images in tests/look/ (after changing the look on purpose)")


def make_clip(path, audio=True):
    """A short H.264 .mp4 of four very different test patterns (three hard cuts),
    with a 440 Hz tone unless audio=False. Made by ffmpeg, so nothing is downloaded."""
    seg = f"s={W}x{H}:r={FPS}"
    sources = [f"testsrc2={seg}:d={SEGMENT}", f"mandelbrot={seg}", f"life={seg}:mold=10:ratio=0.5",
               f"smptehdbars={seg}:d={SEGMENT}"]
    cmd = [ffmpeg_exe(), "-v", "error", "-y"]
    for s in sources:
        cmd += ["-f", "lavfi", "-i", s]
    if audio:  # inputs all come before the output options
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={SEGMENT * len(sources)}"]
    parts = "".join(f"[{i}]trim=duration={SEGMENT},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[v{i}];"
                    for i in range(len(sources)))
    cmd += ["-filter_complex", parts + "".join(f"[v{i}]" for i in range(len(sources)))
            + f"concat=n={len(sources)}:v=1:a=0[v]", "-map", "[v]"]
    if audio:
        cmd += ["-map", f"{len(sources)}:a", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("clip") / "patterns.mp4")


def relabel(src, dst, params):
    """A copy of `src` labelled with other colour properties (setparams), pixels unchanged."""
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-vf", f"setparams={params}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "copy", str(dst)], check=True)
    return dst


@pytest.fixture(scope="session", params=["unlabelled", "bt709", "full range"])
def labelled_clip(request, clip, tmp_path_factory):
    """The clip as it is (no colour labels, so ffmpeg assumes BT.601, limited range),
    labelled BT.709 (as most HD video is), or labelled full range."""
    params = {"bt709": "colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
              "full range": "range=pc"}.get(request.param)
    if params is None:
        return clip
    return relabel(clip, tmp_path_factory.mktemp("clip") / f"{request.param.replace(' ', '_')}.mp4", params)


@pytest.fixture(scope="session")
def silent_clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("clip") / "silent.mp4", audio=False)


def probe(video):
    """ffmpeg's description of a file's streams (its stderr for `ffmpeg -i`)."""
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)],
                          capture_output=True, text=True, errors="replace").stderr


def video_streams(video):
    return re.findall(r"Stream #\S+.*?: Video: .*", probe(video))


def audio_streams(video):
    return re.findall(r"Stream #\S+.*?: Audio: .*", probe(video))


@pytest.fixture(scope="session")
def gpu():
    """The gpu_sketch module, or skip: no CuPy, or no GPU it can use."""
    try:
        from scribble import gpu_sketch
        gpu_sketch.GpuSketcher(64, 64)
    except ImportError:
        pytest.skip("the gpu extra isn't installed (uv sync --extra gpu)")
    except Exception as e:
        pytest.skip(f"no usable NVIDIA GPU ({type(e).__name__}: {e})")
    return gpu_sketch


@pytest.fixture(scope="session")
def nv(gpu, clip):
    """The nv_video module, or skip if the test clip can't be decoded and encoded on
    the GPU here (no PyNvVideoCodec, no NVDEC/NVENC, ...)."""
    from scribble.cli import gpu_video_problem
    why = gpu_video_problem(clip)
    if why:
        pytest.skip(f"GPU video pipeline unavailable: {why}")
    from scribble import nv_video
    return nv_video
