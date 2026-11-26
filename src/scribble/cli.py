"""One command: video -> scribbled shot frames + contact sheet + sketch video with audio.

usage:
    scribble VIDEO_OR_LINK [-o OUTDIR] [options]      (installed as a tool)
    uv run scribble VIDEO_OR_LINK ...                 (from a clone of the repo)

Where files go: run from the repo (`uv run scribble`), into the repo's scribble/
and source/ folders; installed as a tool, into ~/scribble/. Set SCRIBBLE_HOME to
use another folder instead (outputs in it, downloads in its source/ subfolder).

A link (e.g. https://www.youtube.com/watch?v=...) is downloaded with yt-dlp into
the source folder first ('<title> [<id>].mkv' + .info.json); later runs reuse that file.

Each video gets its own OUTDIR, named <slug>: the video's title in
lowercase-with-dashes (tags like "(Official Video)" dropped):
    frames/frame_NNN_<m>m<ss.ss>s.png   one sketch per shot (middle of the shot)
    contact_sheet.png                   all shot sketches on one page
    <slug>_scribble.mp4                 the whole video in sketch style, original audio
    cuts.txt                            detected scene-cut timestamps (seconds)
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2

from .download import download, is_url
from .ffmpeg_tools import VideoEncoder, detect_cuts, pick_encoder
from .sketch import Sketcher, contact_sheet, fmt_time


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


def default_dirs():
    """(output root, download dir) when -o / --download-dir aren't given.

    SCRIBBLE_HOME wins. Otherwise, running from this repo's own venv (`uv run scribble`)
    writes into the repo, whatever the current folder; an installed tool (`uv tool
    install`, which gets its own venv) writes into ~/scribble.
    """
    home = os.environ.get("SCRIBBLE_HOME")
    if home:
        home = Path(home).expanduser()
        return home, home / "source"
    repo = Path(__file__).resolve().parents[2]  # <repo>/src/scribble/cli.py
    if (repo / "pyproject.toml").is_file() and repo in Path(sys.prefix).resolve().parents:
        return repo / "scribble", repo / "source"
    home = Path.home() / "scribble"
    return home, home / "source"


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


def make_sketcher(device, width, height):
    """(sketcher, 'gpu' or 'cpu'): 'auto' falls back to the CPU, an explicit 'gpu' must work."""
    if device != "cpu":
        try:
            from .gpu_sketch import GpuSketcher
            return GpuSketcher(width, height), "gpu"
        except ImportError:
            why = "CuPy is not installed (install scribble with the gpu extra, see README)"
        except Exception as e:  # no NVIDIA GPU, driver too old, kernel compile failed, ...
            why = f"CuPy could not use the GPU: {type(e).__name__}: {e}"
        if device == "gpu":
            raise SystemExit(f"--device gpu: {why}")
    return Sketcher(width, height), "cpu"


def main(argv=None):
    p = argparse.ArgumentParser(prog="scribble", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    try:
        ver = version("scribble-video")  # from pyproject.toml, via the installed package's metadata
    except PackageNotFoundError:
        ver = "unknown"
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {ver}")
    p.add_argument("video", help="video file, or a link (e.g. YouTube) to download first")
    p.add_argument("-o", "--out", type=Path,
                   help="output folder for this video (default: <output root>/<slug of title>, see above)")
    p.add_argument("--download-dir", type=Path,
                   help="where links are downloaded to (default: the source folder, see above); re-runs reuse the file")
    p.add_argument("--max-height", type=int, default=1080, help="max video height to download (default 1080)")
    p.add_argument("--title", help="contact sheet title (default: from .info.json or file name)")
    p.add_argument("--threshold", type=float, default=0.3, help="scene-change sensitivity 0-1; lower = more shots (default 0.3)")
    p.add_argument("--min-shot", type=float, default=0.25, help="ignore shots shorter than this many seconds (default 0.25)")
    p.add_argument("--cols", type=int, default=6, help="contact sheet columns (default 6)")
    p.add_argument("--crf", type=int, default=24, help="video quality, lower = bigger/better (default 24)")
    p.add_argument("--encoder", choices=("auto", "nvenc", "x264"), default="auto",
                   help="nvenc = NVIDIA GPU, x264 = CPU; auto uses the GPU when it works (default auto)")
    p.add_argument("--preset", default="slow",
                   help="x264 speed preset, e.g. medium, fast, veryfast: faster files are a little bigger (default slow)")
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                   help="where to sketch: gpu = NVIDIA GPU via CuPy (needs the gpu extra, see README), "
                        "cpu = CPU only; auto uses the GPU when it works (default auto)")
    p.add_argument("--no-video", action="store_true", help="only make frames + contact sheet (much faster)")
    p.add_argument("--workers", type=int, default=0,
                   help="frames sketched in parallel (default: CPU threads, at most 12; 1 = serial)")
    a = p.parse_args(argv)
    out_root, download_dir = default_dirs()
    a.download_dir = a.download_dir or download_dir
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
    a.out = a.out or out_root / slug
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

    workers = max(1, a.workers or min(12, os.cpu_count() or 1))
    if workers > 1:
        # parallelism comes from sketching whole frames side by side; OpenCV's own
        # per-call threads on top of that only oversubscribe the cores
        cv2.setNumThreads(1)
    sketch, device = make_sketcher(a.device, W, H)
    video_out = a.out / f"{slug}_scribble.mp4"
    if a.no_video:
        print(f"sketching shots on the {device.upper()}")
    else:
        a.encoder = pick_encoder(a.encoder)
        print(f"sketching on the {device.upper()} with {workers} threads, encoding with {a.encoder}")
    thumbs = []
    start = time.time()

    def finish(t, shot, s):
        if encoder:
            encoder.write(s)
        if shot:
            name = f"frame_{shot:03d}_{int(t // 60)}m{t % 60:05.2f}s.png"
            save_png(frames_dir / name, s)
            thumbs.append((t, s))

    with (nullcontext() if a.no_video
          else VideoEncoder(video_out, W, H, fps, audio_from=a.video, crf=a.crf,
                                preset=a.preset, encoder=a.encoder)) as encoder, \
            ThreadPoolExecutor(workers) as pool:
        # frames are sketched out of order on the pool and written back in order;
        # `pending` caps how many decoded frames are held in memory at once
        pending = deque()
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            shot = picks.get(i)
            if encoder or shot:
                pending.append((t, shot, pool.submit(sketch, frame, t)))
                if len(pending) > 2 * workers:
                    t0, shot0, fut = pending.popleft()
                    finish(t0, shot0, fut.result())
            i += 1
            if i % 300 == 0:
                print(f"\r  rendering {i}/{total} ({time.time() - start:.0f}s)", end="", flush=True)
        while pending:
            t0, shot0, fut = pending.popleft()
            finish(t0, shot0, fut.result())
        elapsed = time.time() - start
        print(f"\r  read {i} frames in {elapsed:.0f}s ({i / max(elapsed, 1e-9):.0f} fps, {workers} threads)"
              + ("" if a.no_video else "; finishing encode..."))
    cap.release()

    save_png(a.out / "contact_sheet.png", contact_sheet(thumbs, title, cols=a.cols))
    print(f"done -> {a.out.resolve()}")
    print(f"  {len(thumbs)} frames, contact_sheet.png" + ("" if a.no_video else f", {video_out.name}"))

