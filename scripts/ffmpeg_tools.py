"""Thin helpers around the ffmpeg binary (bundled by imageio-ffmpeg, or found on PATH)."""
import re
import shutil
import subprocess
from functools import lru_cache

# Audio codecs that can be stream-copied into an .mp4 without re-encoding.
MP4_SAFE_AUDIO = {"aac", "mp3", "opus", "flac", "alac", "ac3", "eac3"}


@lru_cache(maxsize=None)
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):  # RuntimeError: no bundled binary for this platform
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise SystemExit("ffmpeg not found: pip install imageio-ffmpeg, or install ffmpeg on PATH")


def _probe(video):
    # `ffmpeg -i` with no output exits non-zero but prints stream info to stderr.
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)],
                       capture_output=True, text=True, errors="replace")
    return r.stderr


def audio_codec(video):
    """Name of the first audio stream's codec, or None if the file has no audio."""
    m = re.search(r"Stream #\S+.*?: Audio: (\w+)", _probe(video))
    return m.group(1) if m else None


def detect_cuts(video, threshold=0.3):
    """Timestamps (seconds) where ffmpeg's scene-change score exceeds `threshold`."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(video), "-an",
           "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"scene detection failed:\n{r.stderr[-2000:]}")
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", r.stderr)]


@lru_cache(maxsize=None)
def nvenc_works():
    """True if ffmpeg can open NVIDIA's H.264 encoder here (needs an NVIDIA GPU and a
    recent enough driver; ffmpeg lists the encoder even when it cannot use it)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
           "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pick_encoder(choice):
    """'auto' -> 'nvenc' when it works, else 'x264'; an explicit choice is checked."""
    if choice == "auto":
        return "nvenc" if nvenc_works() else "x264"
    if choice == "nvenc" and not nvenc_works():
        raise SystemExit("--encoder nvenc: ffmpeg could not open the NVIDIA encoder "
                         "(needs an NVIDIA GPU and an up-to-date driver)")
    return choice


class VideoEncoder:
    """Pipe raw BGR frames into ffmpeg (H.264), optionally taking audio from another file.

    encoder is 'x264' (CPU; crf and preset apply) or 'nvenc' (NVIDIA GPU; crf maps
    to about the same quality as x264's slow preset, and preset is ignored).
    Audio is stream-copied when the codec fits in the container, otherwise encoded to AAC.
    Use as a context manager, calling write(frame) for each frame.
    """

    def __init__(self, out, width, height, fps, audio_from=None, crf=24, preset="slow", encoder="x264"):
        cmd = [ffmpeg_exe(), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-framerate", f"{fps:.6f}", "-i", "-"]
        codec = audio_codec(audio_from) if audio_from else None
        if codec:
            copy = codec in MP4_SAFE_AUDIO or not str(out).lower().endswith((".mp4", ".m4v", ".mov"))
            cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a:0", "-shortest"]
            cmd += ["-c:a", "copy"] if copy else ["-c:a", "aac", "-b:a", "256k"]
        if encoder == "nvenc":
            # NVENC's -cq 21 measured the same size and PSNR as x264 -crf 24 -preset slow
            # on sketch frames, so offset it to keep --crf meaning one quality on both
            cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                    "-rc", "vbr", "-cq", str(max(crf - 3, 0)), "-b:v", "0"]
        else:
            cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
        cmd += ["-pix_fmt", "yuv420p", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self.close()  # surfaces ffmpeg's exit code
            raise

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder exited with code {rc}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:  # don't mask the original error; just stop ffmpeg
            self.proc.kill()
            self.proc.wait()
