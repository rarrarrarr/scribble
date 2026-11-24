"""One command: video -> scribbled shot frames + contact sheet + sketch video with audio.

usage:
    python scripts/make_scribble.py VIDEO_OR_LINK [-o OUTDIR] [options]

A link (e.g. https://www.youtube.com/watch?v=...) is downloaded with yt-dlp into
./source first ('<title> [<id>].mkv' + .info.json); later runs reuse that file.

Each video gets its own OUTDIR, default ./scribble/<slug>, where <slug> is the
video's title in lowercase-with-dashes (tags like "(Official Video)" dropped):
    frames/frame_NNN_<m>m<ss.ss>s.png   one sketch per shot (middle of the shot)
    contact_sheet.png                   all shot sketches on one page
    <slug>_scribble.mp4                 the whole video in sketch style, original audio
    cuts.txt                            detected scene-cut timestamps (seconds)
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from contextlib import nullcontext
from pathlib import Path

# Sibling modules live next to this file; add it explicitly so `python -I` works too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from download import download, is_url  # noqa: E402
from ffmpeg_tools import VideoEncoder, detect_cuts  # noqa: E402
from scribble import Sketcher, contact_sheet, fmt_time  # noqa: E402


def guess_title(video):
    """Title from a yt-dlp `<name>.info.json` beside the video, else the file name."""
    info = video.with_suffix(".info.json")
    if info.is_file():
        try:
            return json.loads(info.read_text(encoding="utf-8"))["title"]
        except (ValueError, KeyError):
            pass
    return video.stem


# Bracketed tags that add nothing to a folder name, e.g. "(Official Music Video)", "[HD]".
_TAG_RE = re.compile(r"[\(\[][^\)\]]*\b(official|video|audio|lyrics?|hd|hq|4k|remaster(ed)?|visualizer)\b[^\)\]]*[\)\]]", re.I)


def slugify(title, max_len=60):
    """'Coheed and Cambria - A Favor House Atlantic (Official Video)' -> 'coheed-and-cambria-a-favor-house-atlantic'."""
    s = _TAG_RE.sub("", title)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0]
    return s  # may be "" for e.g. all-CJK titles; caller falls back


def save_png(path, img):
    # cv2.imwrite silently fails on non-ASCII paths on Windows; encode in memory instead.
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    Path(path).write_bytes(buf.tobytes())


def shot_frames(cuts, duration, fps, min_shot):
    """{frame_index: shot_number} for the middle frame of every shot >= min_shot seconds."""
    bounds = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    mids = [(a + b) / 2 for a, b in zip(bounds, bounds[1:]) if b - a >= min_shot]
    return {int(m * fps): i + 1 for i, m in enumerate(mids)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="video file, or a link (e.g. YouTube) to download first")
    p.add_argument("-o", "--out", type=Path,
                   help="output folder for this video (default: ./scribble/<slug of title>)")
    p.add_argument("--download-dir", type=Path, default=Path("source"),
                   help="where links are downloaded to (default: ./source); re-runs reuse the file")
    p.add_argument("--max-height", type=int, default=1080, help="max video height to download (default 1080)")
    p.add_argument("--title", help="contact sheet title (default: from .info.json or file name)")
    p.add_argument("--threshold", type=float, default=0.3, help="scene-change sensitivity 0-1; lower = more shots (default 0.3)")
    p.add_argument("--min-shot", type=float, default=0.25, help="ignore shots shorter than this many seconds (default 0.25)")
    p.add_argument("--cols", type=int, default=6, help="contact sheet columns (default 6)")
    p.add_argument("--crf", type=int, default=24, help="x264 quality, lower = bigger/better (default 24)")
    p.add_argument("--no-video", action="store_true", help="only make frames + contact sheet (much faster)")
    a = p.parse_args(argv)
    # Don't crash printing non-ASCII titles to a legacy console / redirected output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if is_url(a.video):
        print(f"downloading {a.video} ...")
        a.video = download(a.video, a.download_dir, a.max_height)
        print(f"  -> {a.video}")
    else:
        a.video = Path(a.video)
        if not a.video.is_file():
            p.error(f"no such file: {a.video}")
    title = a.title or guess_title(a.video)
    # downloaded files are '<title> [<id>]', so a non-Latin title still yields the video id
    slug = slugify(title) or slugify(a.video.stem) or "video"
    a.out = a.out or Path("scribble") / slug
    frames_dir = a.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):  # stale shots from an earlier run
        old.unlink()

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        sys.exit(f"OpenCV could not open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps
    print(f"{a.video.name}: {W}x{H} @ {fps:.3f} fps, {fmt_time(duration)}")

    print("detecting scene cuts...")
    cuts = detect_cuts(a.video, a.threshold)
    (a.out / "cuts.txt").write_text("\n".join(f"{c:.3f}" for c in cuts) + "\n")
    picks = shot_frames(cuts, duration, fps, a.min_shot)
    print(f"  {len(cuts)} cuts -> {len(picks)} shots")

    sketch = Sketcher(W, H)
    video_out = a.out / f"{slug}_scribble.mp4"
    thumbs = []
    start = time.time()
    with (nullcontext() if a.no_video
          else VideoEncoder(video_out, W, H, fps, audio_from=a.video, crf=a.crf)) as encoder:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            shot = picks.get(i)
            if encoder or shot:
                s = sketch(frame, t)
                if encoder:
                    encoder.write(s)
                if shot:
                    name = f"frame_{shot:03d}_{int(t // 60)}m{t % 60:05.2f}s.png"
                    save_png(frames_dir / name, s)
                    thumbs.append((t, s))
            i += 1
            if i % 300 == 0:
                print(f"\r  rendering {i}/{total} ({time.time() - start:.0f}s)", end="", flush=True)
        print(f"\r  read {i} frames in {time.time() - start:.0f}s" + ("" if a.no_video else "; finishing encode..."))
    cap.release()

    save_png(a.out / "contact_sheet.png", contact_sheet(thumbs, title, cols=a.cols))
    print(f"done -> {a.out.resolve()}")
    print(f"  {len(thumbs)} frames, contact_sheet.png" + ("" if a.no_video else f", {video_out.name}"))


if __name__ == "__main__":
    main()
