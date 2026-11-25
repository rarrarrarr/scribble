"""Download a video (e.g. a YouTube link) with yt-dlp for make_scribble.py."""
import re
from pathlib import Path

from ffmpeg_tools import ffmpeg_exe

URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


def is_url(s):
    return bool(URL_RE.match(str(s)))


def download(url, dest_dir, max_height=1080):
    """Download best video (<= max_height) + best audio, merged to .mkv, plus a
    yt-dlp .info.json beside it. Returns the video's Path.

    Files are named '<title> [<id>].mkv'; an already-downloaded video is reused.
    """
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("yt-dlp is needed for links: uv lock --upgrade-package yt-dlp")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mkv",
        "outtmpl": str(dest_dir / "%(title).80B [%(id)s].%(ext)s"),
        "writeinfojson": True,
        "noplaylist": True,  # a watch?v=...&list=... link means just that one video
        "ffmpeg_location": ffmpeg_exe(),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist":
            raise SystemExit("that link is a playlist; pass a single video link")
        downloads = info.get("requested_downloads") or []
        path = Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))
    if not path.is_file():
        raise SystemExit(f"download finished but the file wasn't found: {path}")
    return path
