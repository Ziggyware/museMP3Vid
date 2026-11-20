# museMP3Vid

Turn a song into vertical shorts built around its best hook.

```
Song
 ↓
Whisper transcription        (faster-whisper, word-level timestamps)
 ↓
Best-hook detector           (lyric repetition × energy × vocal coverage)
 ↓
Generate captions            (SRT + karaoke ASS + JSON)
 ↓
WebGL visualizer             (per-song themed shader, deterministic frames)
 ↓
Render shorts                (headless Chromium capture + ffmpeg)
 ↓
Export TikTok/Reels/Shorts  (1080×1920 H.264/AAC, one file per platform)
```

## Install

```powershell
# Windows (run from the repo root; note the .\ prefix - PowerShell
# will not run executables from a bare relative path)
winget install Gyan.FFmpeg          # ffmpeg + ffprobe on PATH
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # also installs muse-mp3vid itself (-e .)
.\.venv\Scripts\python -m playwright install chromium
```

Linux/macOS: install ffmpeg via your package manager, then the same pip steps.

## Usage

```powershell
.\.venv\Scripts\muse-mp3vid "My Song.mp3"
```

Outputs land in `out/<song name>/`:

| File | What it is |
|---|---|
| `transcript.json` | Whisper segments + word timestamps |
| `hook.json` | Chosen hook window + ranked alternates, with scores |
| `captions.srt` / `captions.ass` / `captions.json` | Captions for the hook (ASS = karaoke word-sweep, styled for 1080×1920) |
| `theme.json` | The song's visual identity: shader mode, palette, spin, cover |
| `frames.json` | Per-frame spectrum/RMS/beat-pulse data that drives the visualizer |
| `hook_audio.m4a` | The extracted hook audio |
| `visualizer.html` / `preview.html` | The WebGL visualizer (see below) |
| `<song>_hook_tiktok.mp4` etc. | Finished shorts, one per platform, + a `.post.txt` caption stub |

### Options

```
--hook-len 30        hook length in seconds (15/30/60 typical)
--hook-start 42.5    override detection, start the hook here
--model small        Whisper model: tiny/base/small/medium/large-v3
--language en        force a language (default: auto-detect)
--device cuda        use GPU for Whisper if available
--platforms tiktok,reels,shorts
--record             render via real-time MediaRecorder capture instead of
                     per-frame screenshots (needs a GPU-fast machine; the
                     screenshot default is exact on any hardware)
--no-render          stop after captions + visualizer data
--skip-transcribe    reuse transcript.json already in the output dir
--caption-words 4    max words per caption cue
--cover art.jpg      cover art (default: auto-detect in the song's folder)
--no-cover           ignore cover art entirely
--viz auto           auto = a per-section effect schedule derived from
                     the song; or force any single effect by name
--render-workers 3   parallel browsers for the fallback frame capture
--no-direct          skip the direct-to-MP4 encode (WebCodecs) and use
                     per-frame capture instead
```

Rendering writes **directly to MP4 by default**: the visualizer encodes
H.264 in the browser (WebCodecs) as fast as it can draw — not real-time,
not screenshots — and streams it to disk; ffmpeg only muxes. Per-frame
screenshot capture remains as the automatic fallback (it reuses frames
already captured if a run is resumed).

## Per-song themes: 32 effects, scheduled by the music

No two songs get the same video — and a single video doesn't sit on one
effect either. Each run derives a theme (`theme.json`):

- **Palette** — the dominant colors of the song's cover art (median-cut
  quantization), or generated from the song's spectral balance when no
  cover is found.
- **Effect schedule** — the hook is cut into musical sections (lyric
  segment starts + energy change points), and each section gets its own
  effect from a catalog of **32 shader modes**, crossfaded at the cuts:

  radial, tunnel, ribbon, kaleido, sunburst, vinyl, orbital, mandala,
  scope, waterfall, terrain, eqbars, ledwall, plasma, voronoi, moire,
  starfield, spirograph, gridwave, skyline, rings, helix, iris, liquid,
  glitch, halftone, contour, fire, rain, bubbles, laserfan, gears

- **Music picks the family, the song picks the effects** — bass-heavy
  songs draw from tunnel/vinyl/fire/rings/laserfan; dense + bright songs
  from kaleido/glitch/moire/starfield/halftone; sparse songs from
  ribbon/contour/liquid/bubbles/scope; everything else from the full
  balanced pool. The specific sequence is seeded by the song's stats
  (bass share, brightness, onset rate, spectral flux, dynamic range), so
  different songs get different sequences — deterministically.
- **Every effect is parameterized by the music** — kaleido fold counts
  from brightness, sunburst ray counts from onset density, voronoi cell
  scale from spectral flux, fire intensity from bass weight, waterfall
  and terrain read the song's actual spectrum history. Motion speed
  (`spin`) follows onset density throughout.
- **Cover** — the cover art itself appears inside the effects that have
  a natural home for it (radial/kaleido/sunburst discs, vinyl label,
  iris pupil, tunnel/ribbon backdrops).

Force one effect for the whole hook with `--viz <name>`; override the art
with `--cover` / `--no-cover`.

Platform exports share one encode (all three platforms accept the same
1080×1920 H.264/AAC), but each platform's app UI covers a different part of
the frame — so each export burns its captions at that platform's safe-zone
height, and the files are no longer identical.

## How the hook detector scores

Each candidate window is scored `0.45·repetition + 0.30·energy + 0.25·coverage`:

- **Repetition** — lyric lines inside the window that repeat elsewhere in the
  song (Jaccard similarity ≥ 0.7 against every other line). Choruses win.
- **Energy** — mean RMS loudness in the window, normalized to the song peak.
- **Coverage** — fraction of the window covered by sung words (penalizes
  intros, solos, dead air).

Windows snap to segment starts so a short never begins mid-word. Instrumental
tracks fall back to energy over a 2-second start grid.

## The visualizer

`visualizer/visualizer.html` is a standalone WebGL page with two modes:

- **Live** (`visualizer.html`) — pick an audio file; a WebAudio analyser
  drives a radial spectrum + interference shader in real time. Add
  `?capdata=captions.json` to preview the karaoke captions over it.
- **Render** (`visualizer.html?mode=render&data=frames.json`) — deterministic:
  frame *N* is a pure function of `frames.json`, exposed to the renderer as
  `window.__drawFrame(i)`, plus `window.__record(seconds)` for capture.

Rendering uses Playwright + headless Chromium (SwiftShader WebGL), capturing
one screenshot per frame so timing is exact even on CPU-only machines.
`--record` switches to real-time MediaRecorder capture for GPU machines.
If Chromium is unavailable, the pipeline falls back to an ffmpeg
`showspectrum` visual so you still get a finished short — the WebGL page
remains the primary visualizer.

## Layout

```
src/muse_mp3vid/
  transcribe.py   faster-whisper wrapper (word timestamps)
  analyze.py      ffmpeg decode, RMS series, per-frame spectrum + beat pulse
  hook.py         best-hook detector (pure functions, unit-tested)
  theme.py        per-song palette / shader mode / cover derivation
  captions.py     cue grouping -> SRT / karaoke ASS / JSON
  render.py       Playwright capture, ffmpeg compose, platform presets
  cli.py          the pipeline as one command
visualizer/visualizer.html
tests/            unit tests for the pure stages (no Whisper/ffmpeg needed)
```

## Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```
