# Security

## Reporting a problem

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/rarrarrarr/scribble/security/advisories/new),
not in a public issue. I maintain scribble on my own, so I can't promise a response time,
but I'll reply as soon as I can and credit you in the fix unless you'd rather I didn't.

Only the latest release on [PyPI](https://pypi.org/project/scribble-video/) gets fixes.

## What counts

Problems in scribble's own code or its release process, for example:

- a video title, file name or link that makes scribble write files outside its output folder
- a way to make scribble run commands it shouldn't
- anything that could let someone else publish a scribble release to PyPI

## Problems in the tools scribble uses

scribble hands videos to ffmpeg, OpenCV, yt-dlp and (with the `gpu` extra) NVIDIA's
video libraries. Please report a flaw in one of those to that project. If scribble needs
a newer version because of it, open an issue here.

The ffmpeg that scribble runs comes from imageio-ffmpeg, and it only gets newer when
imageio-ffmpeg releases a new version. To use a different ffmpeg, for example one with a
security fix, set the `IMAGEIO_FFMPEG_EXE` environment variable to its path.
