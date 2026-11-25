# scribble

Turn a video (or a YouTube link) into a pencil-and-ink sketch: one drawing per shot, a contact sheet of every shot, and the whole video redrawn in sketch style with its original audio.

![Original frame beside its sketch, and four more sketched shots](docs/sample.jpg)

## Requirements

- [uv](https://docs.astral.sh/uv/) (Windows, macOS or Linux), which also fetches Python 3.10+ if you don't have it

Dependencies go in a virtual environment inside this repo (`.venv`), not your global Python. `uv run` creates it on first use; to set it up ahead of time:

```shell
uv sync
```

This installs the versions pinned in `uv.lock`: OpenCV, NumPy, [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), which downloads an ffmpeg binary for you. A system ffmpeg on your PATH is used as a fallback.

## Usage

```shell
uv run scribble "https://www.youtube.com/watch?v=aqz-KE-bpKQ"   # a link: downloaded first
uv run scribble path/to/video.mp4                                # a local file
uv run scribble path/to/video.mp4 --no-video                     # just the drawings, much faster
```

Run it from the repo folder; output and downloads go in the current folder. Quote links, since `&` is special to most shells.

Each video gets its own folder, named after its title:

```text
scribble/<title-slug>/
  frames/frame_001_0m01.63s.png   one drawing per shot, from the middle of the shot
  contact_sheet.png               every shot on one page
  <title-slug>_scribble.mp4       the full video in sketch style, with original audio
  cuts.txt                        detected scene-cut times, in seconds
```

Links are downloaded into `source/`, and running again reuses that file.

### Options

| Option | Default | |
| --- | --- | --- |
| `-o, --out DIR` | `scribble/<title-slug>` | output folder |
| `--no-video` | | skip the sketch video; only make the drawings and contact sheet |
| `--threshold N` | `0.3` | scene-change sensitivity, 0 to 1; lower finds more shots |
| `--min-shot SECONDS` | `0.25` | ignore shots shorter than this |
| `--crf N` | `24` | video quality; lower is better and bigger |
| `--encoder NAME` | `auto` | `nvenc` encodes on an NVIDIA GPU (needs a recent driver), `x264` on the CPU; `auto` uses the GPU when it works |
| `--preset NAME` | `slow` | x264 speed preset; `veryfast` renders about 20% sooner |
| `--workers N` | CPU threads, at most 12 | frames sketched in parallel; `1` for one at a time |
| `--cols N` | `6` | contact sheet columns |
| `--title TEXT` | video title | contact sheet title (also sets the folder name) |
| `--download-dir DIR` | `source` | where links are downloaded |
| `--max-height N` | `1080` | highest resolution to download |

Each run replaces that folder's previous drawings.

## Troubleshooting

- **Downloads fail:** YouTube changes often and older yt-dlp versions stop working. Update it with `uv lock --upgrade-package yt-dlp`; the next run installs the new version.
- **yt-dlp warns about a JavaScript runtime:** install [deno](https://deno.com) to silence it and make more formats available.

## Code layout

- `scripts/make_scribble.py`: the `scribble` command; runs each step in order
- `scripts/scribble.py`: the sketch look and the contact sheet layout; edit this to change the style
- `scripts/ffmpeg_tools.py`: scene detection and video encoding
- `scripts/download.py`: downloading links with yt-dlp

## Use responsibly

Only download and process videos you have the right to use. YouTube's Terms of Service restrict downloading content without permission, and a sketched copy of someone else's video is still derived from their work.

## License

[MIT](LICENSE). The sample image above is from *Big Buck Bunny*, © 2008 Blender Foundation, [www.bigbuckbunny.org](https://www.bigbuckbunny.org), used under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
