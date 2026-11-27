"""The sketch look must not change by accident.

Two synthetic images (made here, the same on every machine) are sketched and compared
with reference images in tests/look/. After changing the look on purpose, rewrite the
references with `uv run pytest tests/test_look.py --update-look` and commit them.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from scribble.sketch import Sketcher

LOOK = Path(__file__).parent / "look"
W, H = 480, 270  # 270 isn't a multiple of 8: the GPU's CLAHE padding gets tested too


def scene(seed):
    """A BGR test image with what the look reacts to: smooth gradients (pencil shading),
    dark areas (hatching), hard shapes and lines (ink edges), text, and some noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.dstack([x / W * 200 + 30, y / H * 180 + 40, (x + y) / (W + H) * 150 + 60]).astype(np.uint8)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        center = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if rng.random() < 0.5:
            cv2.circle(img, center, int(rng.integers(10, 70)), c, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, center, (center[0] + int(rng.integers(20, 120)), center[1] + int(rng.integers(20, 90))), c, -1)
    for _ in range(6):
        p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.line(img, p1, p2, (20, 20, 20), int(rng.integers(1, 4)), cv2.LINE_AA)
    cv2.putText(img, "scribble", (40, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (15, 15, 15), 3, cv2.LINE_AA)
    noisy = img.astype(np.float32) + rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def compare(got, ref):
    d = np.abs(got.astype(int) - ref.astype(int))
    return d.mean(), (d > 8).mean() * 100


@pytest.fixture(params=[1, 2], ids=["scene1", "scene2"])
def case(request):
    seed = request.param
    return scene(seed), LOOK / f"scene{seed}.png"


def test_cpu_look(case, request):
    img, ref_path = case
    got = Sketcher(W, H)(img, t=1.0)  # t=1: the hand wobble is in effect
    if request.config.getoption("--update-look"):
        LOOK.mkdir(exist_ok=True)
        cv2.imwrite(str(ref_path), got)
        pytest.skip(f"rewrote {ref_path.name}")
    ref = cv2.imread(str(ref_path))
    assert ref is not None, f"no reference image {ref_path}: run with --update-look"
    mean, big = compare(got, ref)
    # identical with the same OpenCV; a little slack for other builds of it (other CPUs'
    # code paths can round differently); a real change to the look moves far more
    assert mean < 0.1 and big < 0.1, f"the look changed: mean |diff| {mean:.3f}, {big:.3f}% of pixels off by more than 8"


@pytest.mark.gpu
def test_gpu_look(case, gpu):
    img, ref_path = case
    ref = cv2.imread(str(ref_path))
    assert ref is not None, f"no reference image {ref_path}: run with --update-look"
    got = gpu.GpuSketcher(W, H)(img, t=1.0).bgr
    mean, big = compare(got, ref)
    # the GPU sketch differs from the CPU one in scattered pixels (see gpu_sketch.py)
    assert mean < 0.15 and big < 0.5, f"GPU look differs: mean |diff| {mean:.3f}, {big:.3f}% of pixels off by more than 8"
