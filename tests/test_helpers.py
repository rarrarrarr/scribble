"""Small helpers: folder names, times, frame rates, audio options."""
from fractions import Fraction

import pytest

from scribble.cli import slugify
from scribble.ffmpeg_tools import audio_args, audio_codec, color_info, exact_rate
from scribble.sketch import fmt_time


@pytest.mark.parametrize("title, slug", [
    ("Coheed and Cambria - A Favor House Atlantic (Official Video)", "coheed-and-cambria-a-favor-house-atlantic"),
    ("La Femme - Sacatela", "la-femme-sacatela"),
    ("Song [HD] (Lyrics)", "song"),
    ("Café Tacvba", "cafe-tacvba"),  # accents folded to ASCII
    ("東京", ""),  # nothing ASCII left: the caller falls back to the file name
])
def test_slugify(title, slug):
    assert slugify(title) == slug


def test_slugify_keeps_whole_words_when_shortening():
    s = slugify("word " * 30, max_len=20)
    assert len(s) <= 20 and not s.endswith("-") and s.split("-") == ["word"] * len(s.split("-"))


@pytest.mark.parametrize("t, text", [(0, "0:00.00"), (61.5, "1:01.50"), (3599.99, "59:59.99")])
def test_fmt_time(t, text):
    assert fmt_time(t) == text


@pytest.mark.parametrize("fps, rate", [
    (30000 / 1001, Fraction(30000, 1001)),
    (24000 / 1001, Fraction(24000, 1001)),
    (25.0, Fraction(25)),
    (12.5, Fraction(25, 2)),
    (29.970029, Fraction(30000, 1001)),  # stored rounded
    (119.88012, Fraction(120000, 1001)),
])
def test_exact_rate(fps, rate):
    assert exact_rate(fps) == rate


@pytest.mark.parametrize("labelled_clip, info", [
    ("unlabelled", (None, False)), ("bt709", ("bt709", False)), ("full range", (None, True)),
], indirect=["labelled_clip"])
def test_color_info(labelled_clip, info):
    assert color_info(labelled_clip) == info


def test_color_info_of_downloaded_style_video(tmp_path, clip):
    # YouTube's labels: matrix, primaries and transfer all bt709 (shown as one "bt709")
    from conftest import relabel
    v = relabel(clip, tmp_path / "yt.mp4", "colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv")
    assert color_info(v) == ("bt709", False)


def test_audio_is_copied_into_mp4(clip):
    assert audio_codec(clip) == "aac"
    args = audio_args(clip, "out.mp4")
    assert args[:2] == ["-i", str(clip)] and args[-2:] == ["-c:a", "copy"]


def test_no_audio_means_no_audio_options(silent_clip):
    assert audio_codec(silent_clip) is None
    assert audio_args(silent_clip, "out.mp4") == []
