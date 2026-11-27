"""Whole renders of the generated test clip, through each pipeline: the files it makes,
and that the GPU pipeline (nv_video) agrees with the ffmpeg one."""
import cv2
import numpy as np
import pytest

from scribble import cli
from conftest import audio_streams, video_streams

pytestmark = pytest.mark.slow


def render(clip, out, *args):
    cli.main([str(clip), "-o", str(out), *args])
    return out


def shots(out):
    return {p.name: cv2.imread(str(p)) for p in sorted((out / "frames").glob("*.png"))}


def frame_count(video):
    return int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))


def check_outputs(out, clip):
    """What every full render must produce, from the 160-frame, four-pattern clip."""
    cuts = (out / "cuts.txt").read_text().split()
    for planned in ("1.600", "3.200", "4.800"):  # where the patterns change
        assert planned in cuts
    assert len(shots(out)) == 4  # (the life pattern adds a cut at 3.28 s: too short a shot)
    assert (out / "contact_sheet.png").is_file()
    video = out / "patterns_scribble.mp4"
    assert frame_count(video) == frame_count(clip) == 160
    assert len(audio_streams(video)) == 1
    assert "tv, bt709" in video_streams(video)[0]  # labelled with the matrix it was made with


def test_cpu_pipeline(clip, tmp_path):
    out = render(clip, tmp_path / "cpu", "--device", "cpu", "--encoder", "x264", "--preset", "ultrafast")
    check_outputs(out, clip)


@pytest.mark.gpu
def test_gpu_pipeline_agrees_with_ffmpeg_pipeline(nv, clip, tmp_path, capsys):
    gpu = render(clip, tmp_path / "gpu")
    assert "decoding on the GPU (NVDEC)" in capsys.readouterr().out
    check_outputs(gpu, clip)
    # the ffmpeg pipeline, still sketching on the GPU (x264 isn't NVENC, so nv_video stays out)
    ffm = render(clip, tmp_path / "ffmpeg", "--encoder", "x264", "--preset", "ultrafast")
    assert "decoding with ffmpeg" in capsys.readouterr().out
    check_outputs(ffm, clip)
    assert (gpu / "cuts.txt").read_text() == (ffm / "cuts.txt").read_text()
    a, b = shots(gpu), shots(ffm)
    assert a.keys() == b.keys()
    for name in a:  # the decoders' gray differs a little (see nv_video.py), the sketch amplifies it
        d = np.abs(a[name].astype(int) - b[name].astype(int))
        assert d.mean() < 2, (name, d.mean())  # measured 0.6 to 0.9


@pytest.mark.parametrize("pipeline", ["gpu", "ffmpeg"])
def test_no_video_picks_the_same_shots(pipeline, clip, tmp_path, request):
    """--no-video finds the shots before sketching; a full render while detecting."""
    if pipeline == "gpu":
        request.getfixturevalue("nv")
        args = []
    else:
        args = ["--device", "cpu", "--encoder", "x264", "--preset", "ultrafast"]
    full = render(clip, tmp_path / "full", *args)
    only = render(clip, tmp_path / "shots", "--no-video", *args)
    a, b = shots(full), shots(only)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
