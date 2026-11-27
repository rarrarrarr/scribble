"""ShotPicker picks each shot's middle frame while scene detection is still running.
It must pick exactly what picking after detection finishes would, however far ahead
detection is, and hold few frames while it waits."""
import random

import pytest

from scribble.cli import ShotPicker


def shot_frames(cuts, duration, fps, min_shot):
    """The simple version, once all cuts are known (scribble 0.3.1 worked this way):
    {frame index: shot number} for the middle of each shot of at least min_shot seconds."""
    bounds = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    mids = [(a + b) / 2 for a, b in zip(bounds, bounds[1:]) if b - a >= min_shot]
    return {int(m * fps): i + 1 for i, m in enumerate(mids)}


def run_picker(fps, total, cuts, min_shot, lead, rng=None):
    """Feed frames 0..total-1 to a ShotPicker the way cli.main does, with detection
    `lead`ing as given: 'done' (finished before the render), 'barely' (just past each
    frame) or 'random'. Returns ({frame: shot}, most frames held at once)."""
    duration = total / fps
    frame_times = [round(i / fps, 3) for i in range(total)]  # as ffmpeg prints pts_time
    got = {}
    p = ShotPicker(fps, duration, min_shot, lambda s, i, img: got.__setitem__(i, s))
    k = 0  # detection has scored frames up to k
    most_held = 0
    for i in range(total):
        t = i / fps
        if lead == "done":
            h, done = float("inf"), True
        else:
            k = max(k, i if lead == "barely" else min(total - 1, i + rng.randint(0, 60)))
            while frame_times[k] < t and k + 1 < total:  # cli waits for detection to reach t
                k += 1
            h, done = frame_times[k], k == total - 1 and rng.random() < 0.5
        p.update([c for c in cuts if c <= h], h, done)
        if len(p.kept) >= p.KEEP_MAX and not done:  # as cli.main: wait for detection instead
            need = p.horizon_to_skip(i)
            while frame_times[k] < need and k + 1 < total:
                k += 1
            h = frame_times[k]
            done = k == total - 1 and h < need
            p.update([c for c in cuts if c <= h], h, done)
        p.feed(i, f"img{i}")
        most_held = max(most_held, len(p.kept))
    p.update(cuts, float("inf"), True)
    return got, most_held


def expected(cuts, total, fps, min_shot):
    picks = shot_frames(cuts, total / fps, fps, min_shot)
    return {i: s for i, s in picks.items() if i < total}  # middles past the end never render


@pytest.mark.parametrize("lead", ["done", "barely", "random"])
def test_picks_match_picking_after_detection(lead):
    rng = random.Random(1)
    for _ in range(1000):
        fps = rng.choice([23.976, 25.0, 29.97, 30.0, 59.94])
        total = rng.randint(1, 900)
        frame_times = [round(i / fps, 3) for i in range(total)]
        cuts = sorted(set(rng.sample(frame_times, min(rng.randint(0, 40), total))))
        min_shot = rng.choice([0.0, 0.25, 1.0])
        got, most_held = run_picker(fps, total, cuts, min_shot, lead, rng)
        assert got == expected(cuts, total, fps, min_shot), (fps, total, cuts, min_shot)
        assert most_held <= ShotPicker.KEEP_MAX


def test_no_cuts_is_one_shot():
    got, _ = run_picker(25.0, 100, [], 0.25, "barely", random.Random(0))
    assert got == {50: 1}


def test_short_shots_are_skipped():
    # shots 0-2 s, 2-2.04 s (one frame) and 2.04-8 s: the middle one is too short;
    # the others' middles are 1 s (frame 25) and 5.02 s (frame 125.5, so 125)
    got, _ = run_picker(25.0, 200, [2.0, 2.04], 0.25, "done")
    assert got == {25: 1, 125: 2}


def test_middle_frame_just_before_its_cut():
    # Timestamps are rounded to the millisecond, so a frame can look a hair earlier than
    # the cut that starts its shot. This case once lost the short shot's picture.
    cuts = [0.2, 1.201, 1.401, 2.202, 2.369, 2.536, 3.337, 3.37, 3.837]
    got, _ = run_picker(29.97, 150, cuts, 0.0, "barely", random.Random(0))
    assert got == expected(cuts, 150, 29.97, 0.0)
    assert 100 in got  # the middle of the 3.337-3.37 shot
