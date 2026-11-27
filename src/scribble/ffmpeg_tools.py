"""Thin helpers around the ffmpeg binary (bundled by imageio-ffmpeg, or found on PATH)."""
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from fractions import Fraction
from functools import lru_cache

import numpy as np

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
        raise SystemExit("ffmpeg not found: run `uv sync`, or install ffmpeg on PATH")


def _probe(video):
    # `ffmpeg -i` with no output exits non-zero but prints stream info to stderr.
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)],
                       capture_output=True, text=True, errors="replace")
    return r.stderr


@lru_cache(maxsize=None)
def color_info(video):
    """(colour matrix, full range?) of the first video stream as ffmpeg reads its labels,
    e.g. ('bt709', False); the matrix is None when unlabelled. ffmpeg describes the
    stream as e.g. 'yuv420p(tv, bt709, progressive)', 'yuvj420p(pc, progressive)' or
    'yuv420p(tv, bt470bg/bt709/bt709)' (matrix/primaries/transfer when they differ)."""
    m = re.search(r"Stream #\S+.*?: Video: [^,]+, (\w+)(?:\(([^)]*)\))?", _probe(video))
    if not m:
        return None, False
    pix_fmt, details = m.group(1), [d.strip() for d in (m.group(2) or "").split(",")]
    full = "pc" in details or pix_fmt.startswith("yuvj")
    known = ("bt709", "bt470bg", "smpte170m", "bt2020nc", "bt2020c", "smpte240m", "fcc", "gbr", "ycgco")
    matrix = next((d.split("/")[0] for d in details if d.split("/")[0] in known), None)
    return matrix, full


@lru_cache(maxsize=None)
def audio_codec(video):
    """Name of the first audio stream's codec, or None if the file has no audio."""
    m = re.search(r"Stream #\S+.*?: Audio: (\w+)", _probe(video))
    return m.group(1) if m else None


@lru_cache(maxsize=None)
def pick_decoder(video):
    """(ffmpeg input options, label, width, height) for decoding `video`.

    Tries NVIDIA's decoder, then whatever GPU decoder ffmpeg finds on its own (AMD and
    Intel on Windows, VideoToolbox on macOS, VAAPI on Linux), then the CPU. Even with a
    GPU decoder picked, ffmpeg decodes on the CPU any codec that decoder can't handle,
    so the label says what was asked for, not what ran. Width and height come from
    ffmpeg itself, so they match the frames it outputs (after rotation).
    """
    for hw in (["-hwaccel", "cuda"], ["-hwaccel", "auto"], []):
        cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-v", "verbose", *hw, "-i", str(video),
               "-an", "-sn", "-frames:v", "3", "-f", "null", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        out = r.stderr.partition("Output #0")[2]
        size = re.search(r"Video: .*?[ ,](\d{2,5})x(\d{2,5})[ ,\n]", out)
        if r.returncode != 0 or not size:
            continue
        if hw == ["-hwaccel", "auto"]:
            # 'auto' quietly decodes on the CPU when it finds no GPU decoder: report that
            m = re.search(r"Using auto hwaccel type (\w+)", r.stderr)
            if not m:
                continue
            hw = ["-hwaccel", m.group(1)]
        return hw, (hw[1] if hw else "cpu"), int(size.group(1)), int(size.group(2))
    raise RuntimeError(f"ffmpeg could not decode {video}:\n{r.stderr[-2000:]}")


class VideoDecoder:
    """Iterate over a video's frames as uint8 arrays: (height, width, 3) BGR, or
    (height, width) with gray=True.

    ffmpeg decodes (with `hwaccel`, the options from pick_decoder) and a background
    thread reads its output, so decoding overlaps whatever the caller does with each
    frame. Every decoded frame is returned once, like OpenCV's VideoCapture. Gray is
    a third of the data and skips ffmpeg's colour conversion; it matches OpenCV's
    BGR-to-gray of ffmpeg's BGR to within one level. Use as a context manager.
    """

    def __init__(self, video, width, height, hwaccel=(), gray=False):
        cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", *hwaccel, "-i", str(video), "-an", "-sn",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
        self.shape = (height, width) if gray else (height, width, 3)
        self._err = tempfile.TemporaryFile()  # a stderr pipe nobody reads could fill and stall ffmpeg
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err)
        self._frames = queue.Queue(maxsize=8)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        size = int(np.prod(self.shape))
        try:
            while True:
                # one frame per read: much faster through a Windows pipe than big reads
                buf = self.proc.stdout.read(size)
                if len(buf) < size:
                    break
                self._frames.put(np.frombuffer(buf, np.uint8).reshape(self.shape))
        finally:
            self._frames.put(None)

    def __iter__(self):
        while (frame := self._frames.get()) is not None:
            yield frame
        if self.proc.wait() != 0:
            self._err.seek(0)
            msg = self._err.read().decode(errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg decoder exited with code {self.proc.returncode}:\n{msg}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.proc.poll() is None:  # stopped early: don't leave ffmpeg running
            self.proc.kill()
        self.proc.wait()
        self._err.close()


class CutTracker:
    """What a scene detector shares with the render: cuts found so far (timestamps,
    in seconds), and how far detection has got. A detector runs in a background
    thread, calling _scored() per frame (or per batch) and _finished() at the end."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.cuts = []
        self.horizon = float("-inf")  # timestamp of the last frame scored
        self.done = False
        self._error = None
        self._cond = threading.Condition()

    def _scored(self, scores):
        """scores: (timestamp, scene score) of the next frames, in order."""
        with self._cond:
            for t, score in scores:
                if score > self.threshold:
                    self.cuts.append(t)
                self.horizon = t
            self._cond.notify_all()

    def _finished(self, error=None):
        with self._cond:
            self._error = error
            self.done = True
            self._cond.notify_all()

    def wait_until(self, t):
        """(cuts so far, horizon, done) once every cut at or before time t is known."""
        with self._cond:
            self._cond.wait_for(lambda: self.done or self.horizon >= t)
            if self._error:
                raise self._error
            return list(self.cuts), self.horizon, self.done

    def result(self):
        """All the cuts, once detection has finished."""
        return self.wait_until(float("inf"))[0]


class SceneDetector(CutTracker):
    """Finds scene cuts (where ffmpeg's scene-change score exceeds `threshold`) in a
    background ffmpeg, so the video can render meanwhile.

    ffmpeg prints every frame's score as it goes, so besides the cuts so far this
    knows how far it has got: wait_until(t) returns once every cut up to t is known.
    `hwaccel` (from pick_decoder) decodes on the GPU; the cuts found are the same.
    Use as a context manager.
    """

    def __init__(self, video, threshold=0.3, hwaccel=()):
        super().__init__(threshold)
        cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-nostdin", *hwaccel, "-i", str(video), "-an", "-sn",
               "-vf", "select='gte(scene,0)',metadata=print:key=lavfi.scene_score", "-f", "null", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True, errors="replace")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        # per frame: "[Parsed_metadata_1 @ ...] frame:12 pts:400 pts_time:0.4"
        # then        "[Parsed_metadata_1 @ ...] lavfi.scene_score=0.012345"
        tail = deque(maxlen=40)  # for the error message
        t = None
        for line in self.proc.stderr:
            tail.append(line)
            if "Parsed_metadata" not in line:
                continue
            if m := re.search(r"pts_time:(-?[0-9.]+)", line):
                t = float(m.group(1))
            elif (m := re.search(r"lavfi\.scene_score=([0-9.]+)", line)) and t is not None:
                self._scored([(t, float(m.group(1)))])
        rc = self.proc.wait()
        self._finished(RuntimeError(f"scene detection failed:\n{''.join(tail)[-2000:]}") if rc else None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.proc.poll() is None:  # stopped early: don't leave ffmpeg running
            self.proc.kill()
        self.proc.wait()


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


def exact_rate(fps):
    """A frame rate as an exact fraction: 29.97002997... -> 30000/1001, 25.0 -> 25.
    Denominators up to 1001 cover the usual rates (the NTSC ones are n * 1000/1001),
    and snap a rate stored rounded, like 29.970029, back to 30000/1001."""
    return Fraction(fps).limit_denominator(1001)


def audio_args(audio_from, out):
    """ffmpeg options adding the first audio stream of `audio_from` (input 1) to the
    video (input 0), if it has one: stream-copied when the codec fits the container,
    otherwise encoded to AAC."""
    codec = audio_codec(audio_from) if audio_from else None
    if not codec:
        return []
    copy = codec in MP4_SAFE_AUDIO or not str(out).lower().endswith((".mp4", ".m4v", ".mov"))
    return (["-i", str(audio_from), "-map", "0:v", "-map", "1:a:0", "-shortest"]
            + (["-c:a", "copy"] if copy else ["-c:a", "aac", "-b:a", "256k"]))


class VideoEncoder:
    """Pipe raw BGR frames into ffmpeg (H.264), optionally taking audio from another file.

    encoder is 'x264' (CPU; crf and preset apply) or 'nvenc' (NVIDIA GPU; crf maps
    to about the same quality as x264's slow preset, and preset is ignored).
    Audio is stream-copied when the codec fits in the container, otherwise encoded to AAC.
    Use as a context manager, calling write(frame) for each frame. Frames are piped to
    ffmpeg from a background thread, so write() returns before ffmpeg has taken them.
    """

    def __init__(self, out, width, height, fps, audio_from=None, crf=24, preset="slow", encoder="x264",
                 pix_fmt="bgr24"):
        # pix_fmt: what write() gets, 'bgr24' frames (H, W, 3) or 'yuv420p' (I420, already
        # BT.709 limited range, as GpuSketch.yuv). BGR is converted with that same
        # matrix, and the video is labelled with it: HD players assume BT.709 for
        # unlabelled video, while ffmpeg would convert with BT.601 by default.
        cmd = [ffmpeg_exe(), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}",
               "-framerate", f"{fps:.6f}", "-i", "-"]
        cmd += audio_args(audio_from, out)
        if encoder == "nvenc":
            # NVENC's -cq 21 measured the same size and PSNR as x264 -crf 24 -preset slow
            # on sketch frames, so offset it to keep --crf meaning one quality on both.
            # Preset p4 encodes about 1.3x faster than p5 for the same size and 0.1 dB less PSNR.
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                    "-rc", "vbr", "-cq", str(max(crf - 3, 0)), "-b:v", "0"]
        else:
            cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
        # label the frames (ffmpeg now takes colour tags from frames, not output options)
        vf = "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
        if pix_fmt != "yuv420p":
            vf = f"scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,{vf}"
        cmd += ["-vf", vf, "-pix_fmt", "yuv420p", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._frames = queue.Queue(maxsize=8)
        self._error = None  # set by the writer thread if the pipe breaks
        self._writer = threading.Thread(target=self._write_all, daemon=True)
        self._writer.start()

    def _write_all(self):
        while (frame := self._frames.get()) is not None:
            if self._error is None:  # after a failure, keep draining so write() never blocks
                try:
                    self.proc.stdin.write(np.ascontiguousarray(frame).data)
                except (BrokenPipeError, OSError) as e:
                    self._error = e

    def write(self, frame):
        if self._error is not None:
            self.close()  # surfaces ffmpeg's exit code
            raise self._error
        self._frames.put(frame)

    def close(self):
        if self._writer.is_alive():
            self._frames.put(None)
            self._writer.join()
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder exited with code {rc}")
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:  # don't mask the original error; just stop ffmpeg and the writer
            self.proc.kill()
            self.proc.wait()
            self._frames.put(None)
