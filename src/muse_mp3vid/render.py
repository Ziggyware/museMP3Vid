"""Stages 4-6: WebGL visualizer rendering + ffmpeg composition + platform export.

The visualizer page renders deterministically from frames.json. Capture
paths, fastest first:
  - direct (default): the page encodes H.264 itself via WebCodecs, as fast
    as it can draw, streaming chunks to visual.h264 over HTTP; ffmpeg then
    re-times the stream (frame-index timestamps at the page's fps) into
    visual.mp4, and the frame count is verified against frames.json.
    Falls back to frames on any failure.
  - frames: one screenshot per frame -> visual.mp4. Exact frame timing on
    any machine, including CPU-only SwiftShader rendering.
  - record (opt-in): canvas.captureStream + MediaRecorder inside headless
    Chromium -> visual.webm. Real-time, so only correct when the machine
    can actually draw 1080x1920 WebGL at speed (GPU); on slow software
    rendering the capture starves and the duration comes out wrong.
If Playwright/Chromium is unavailable, falls back to an ffmpeg showspectrum
visual so the pipeline still produces a short (the WebGL page remains the
primary visualizer and can be opened in any browser).
"""
from __future__ import annotations

import base64
import functools
import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920

# All three platforms accept the same 1080x1920 H.264/AAC file; what actually
# differs is the UI each one draws over the bottom of the frame, so each
# platform gets its own caption safe-zone (ASS MarginV) and post text.
PLATFORM_PRESETS = {
    "tiktok": {"suffix": "tiktok", "label": "TikTok", "margin_v": 560},
    "reels": {"suffix": "reels", "label": "Instagram Reels", "margin_v": 680},
    "shorts": {"suffix": "shorts", "label": "YouTube Shorts", "margin_v": 470},
}


def _run(cmd: list[str], cwd: str | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:3])}...\n{proc.stderr[-800:]}")


def extract_hook_audio(song: str, start: float, end: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Output-side seek (after -i): sample-accurate. Input-side -ss on an
    # MP3 lands on a frame boundary up to ~26 ms before the requested start
    # and AAC priming pads the tail, so the hook audio drifted long and
    # started early; the export then inherited the wrong length.
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-i", song, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-vn", "-c:a", "aac", "-b:a", "192k", str(out_path),
    ])
    return out_path


class _ChunkHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the job directory; also accepts POST /chunk appends from the
    visualizer page's WebCodecs encoder (direct-to-video path). The target
    file + lock live on the server instance (see _Server)."""

    def do_POST(self):
        if self.path != "/chunk":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        target = getattr(self.server, "chunk_path", None)
        lock = getattr(self.server, "chunk_lock", None)
        if target is not None and lock is not None and body:
            with lock:
                with open(target, "ab") as fh:
                    fh.write(body)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _Server:
    def __init__(self, directory: Path, chunk_path: Path | None = None):
        handler = functools.partial(_ChunkHandler, directory=str(directory))
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.chunk_path = chunk_path
        self.httpd.chunk_lock = threading.Lock()
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()


def _launch_browser(pw, swiftshader: bool):
    """Chromium launch, aiming at the REAL GPU.

    Playwright's default headless launch uses the headless-shell build,
    which is software-only (SwiftShader) no matter what ANGLE flags it
    gets - on a machine with a GPU that silently turns every render into
    a CPU crawl. The full Chromium build (channel="chromium") runs the
    new headless mode, which can reach the GPU (ANGLE D3D11 on Windows)
    and the hardware H.264 encoder. Try it first; if that build is not
    installed, fall back to the default launch and let the page's
    software-GL detection cheapen the heavy shaders instead."""
    args = ["--mute-audio", "--disable-dev-shm-usage"]
    if swiftshader:
        args += ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
        return pw.chromium.launch(args=args)
    # Real hardware: unblock the GPU and the accelerated video encoder
    # (Media Foundation on Windows). Without these, headless Chromium
    # software-encodes H.264 no matter what the page requests.
    args += ["--ignore-gpu-blocklist", "--enable-accelerated-video-encode"]
    import sys as _sys

    if _sys.platform == "win32":
        args.append("--use-angle=d3d11")
    try:
        return pw.chromium.launch(channel="chromium", args=args,
                                  ignore_default_args=["--disable-gpu"])
    except Exception:  # noqa: BLE001 - full build missing: shell fallback
        print("  [render] full Chromium build unavailable; using the "
              "headless shell (software rendering likely)")
        return pw.chromium.launch(args=args)


def _capture_chunk(url: str, frames_dir: Path, indices: list[int],
                   swiftshader: bool, progress: dict) -> None:
    """Draw + screenshot `indices` in this thread's own browser instance."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = _launch_browser(pw, swiftshader)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.set_default_timeout(120000)
        page.goto(url)
        page.wait_for_function("window.__ready === true", timeout=20000)
        if not progress.get("sw_note") and page.evaluate(
                "window.__softwareGL === true"):
            progress["sw_note"] = True
            print("  [render] WebGL is SOFTWARE (SwiftShader) - the GPU "
                  "was not reached; heavy effects run in cheap mode")
        for i in indices:
            page.evaluate(f"window.__drawFrame({i})")
            page.screenshot(path=str(frames_dir / f"frame_{i:05d}.jpg"),
                            type="jpeg", quality=90)
            progress["done"] += 1
            if progress["done"] % 60 == 0:
                print(f"  [render] frame {progress['done']}/{progress['total']}")
        browser.close()


def _probe_stream(path: Path) -> tuple[int, float]:
    """(frame count, duration in seconds) of a video stream via ffprobe.
    Duration is NaN for raw streams that carry no timing."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr[-300:]}")
    stream = json.loads(proc.stdout)["streams"][0]
    raw_dur = stream.get("duration")
    dur = float(raw_dur) if raw_dur not in (None, "N/A") else float("nan")
    return int(stream["nb_read_frames"]), dur


def _direct_visual(job_dir: Path, url: str, duration: float, fps: int,
                   n_expected: int) -> Path:
    """WebCodecs direct encode: the page encodes H.264 itself (as fast as it
    can draw, not real-time) and streams chunks to visual.h264 via
    POST /chunk. Raises RuntimeError on any failure so the caller can
    fall back to frame capture.

    The mux does NOT stream-copy: a raw Annex B stream has no timestamps
    of its own, so a copy-mux lets ffmpeg guess each frame's timing from
    whatever the encoder wrote into the SPS - and a hardware encoder's
    guess can be wildly wrong (one user's stream played 10s of frames in
    ~1s, after which the composed short held the last frame for the
    remaining 9). Instead the frames are re-timed from their index
    (setpts=N) at the page's fps and re-encoded, and the result is
    verified against the frame count in frames.json before it is
    trusted."""
    from playwright.sync_api import sync_playwright

    h264_path = job_dir / "visual.h264"
    if h264_path.exists():
        h264_path.unlink()
    error = None
    nbytes = 0
    with sync_playwright() as pw:
        browser = None
        page = None
        last_exc: Exception | None = None
        for swiftshader in (False, True):
            try:
                browser = _launch_browser(pw, swiftshader)
                page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
                page.set_default_timeout(30000)
                page.goto(url)
                page.wait_for_function("window.__ready === true", timeout=20000)
                break
            except Exception as exc:  # noqa: BLE001 - retried under SwiftShader
                last_exc = exc
                if browser is not None:
                    browser.close()
                    browser = None
                    page = None
        if browser is None or page is None:
            raise RuntimeError(f"direct encode: browser setup failed: {last_exc}")
        try:
            # fire-and-forget: __encodeDirect never rejects (it records its
            # own error), but attach a catch in JS for absolute safety
            page.evaluate("() => { window.__encodeDirect().catch(() => {}); }")
            # Poll with progress prints instead of one silent wait: a long
            # encode must never look like a hang. The deadline is generous -
            # while bytes keep arriving the encode is alive, not stuck.
            import time as _time

            deadline = _time.monotonic() + max(600.0, duration * 20.0)
            started = _time.monotonic()
            last_print = 0.0
            announced = False
            last_bytes = 0
            last_change = started
            while True:
                if page.evaluate("window.__directDone === true"):
                    break
                if not announced and page.evaluate(
                        "window.__directHwAccel !== undefined"):
                    hw = page.evaluate("window.__directHwAccel")
                    renderer = page.evaluate("window.__glRenderer || 'unknown'")
                    software_gl = page.evaluate("window.__softwareGL === true")
                    # The encoder config check alone lies: Chromium will
                    # accept prefer-hardware while WebGL is SwiftShader
                    # and everything still crawls. HARDWARE is claimed
                    # only when the GL renderer is real silicon too.
                    if hw and not software_gl:
                        print(f"  [render] direct encode: HARDWARE H.264 "
                              f"(WebGL: {renderer})")
                    else:
                        print("  [render] direct encode: SOFTWARE H.264 - "
                              f"the GPU was not reached (WebGL: {renderer}); "
                              "heavy effects run in cheap mode, expect slow")
                    announced = True
                now = _time.monotonic()
                if now > deadline:
                    raise RuntimeError("direct encode timed out")
                # Pace watchdog: never again burn 20+ minutes discovering
                # the encode is crawling. Once enough frames are in,
                # project the total; if it is hopeless, bail to frame
                # capture while that is still cheap.
                elapsed = now - started
                frames_done = page.evaluate("window.__directFrames || 0")
                if elapsed > 45.0 and frames_done >= 30:
                    projected = elapsed / frames_done * n_expected
                    if projected > max(240.0, duration * 10.0):
                        raise RuntimeError(
                            f"direct encode too slow ({frames_done} frames "
                            f"in {elapsed:.0f}s, ~{projected / 60.0:.0f} min "
                            "projected); falling back to frame capture")
                sent_now = page.evaluate("window.__directBytes || 0")
                if sent_now != last_bytes:
                    last_bytes = sent_now
                    last_change = now
                elif now - last_change > 90.0 and elapsed > 90.0:
                    raise RuntimeError(
                        "direct encode stalled (no data for 90s); "
                        "falling back to frame capture")
                if now - last_print >= 30.0:
                    print(f"  [render] direct encode streaming... {last_bytes} bytes")
                    last_print = now
                _time.sleep(2.0)
            error = page.evaluate("window.__directError")
            nbytes = page.evaluate("window.__directBytes || 0")
        finally:
            browser.close()
    if error:
        raise RuntimeError(f"direct encode failed in page: {error}")
    if not h264_path.exists() or h264_path.stat().st_size == 0:
        raise RuntimeError("direct encode produced no H.264 data")
    print(f"  [render] direct encode: {nbytes} bytes streamed from the page; muxing...")
    # The page encoded n_expected frames; if the stream on disk holds
    # materially fewer, the encoder dropped them (seen with hardware
    # encoders fed faster than real time) and no mux can invent them
    # back - fail over to frame capture instead of shipping a stub.
    got, _ = _probe_stream(h264_path)
    if got < int(n_expected * 0.98):
        raise RuntimeError(
            f"direct encode: stream holds {got} of {n_expected} frames; "
            "falling back to frame capture")
    visual = job_dir / "visual.mp4"
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(fps), "-i", str(h264_path),
        "-vf", f"setpts=N/({fps}*TB)",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-movflags", "+faststart", str(visual),
    ])
    got, dur = _probe_stream(visual)
    if got < int(n_expected * 0.98) or not (abs(dur - duration) <= 0.6):
        raise RuntimeError(
            f"direct encode: muxed visual is {got} frames / {dur:.2f}s, "
            f"expected {n_expected} / {duration:.2f}s; "
            "falling back to frame capture")
    return visual


def render_visualizer(job_dir: Path, duration: float, fps: int = 30,
                      frames_mode: bool = True, workers: int = 3,
                      direct: bool = True) -> Path:
    """Render the WebGL visualizer for `duration` seconds. Returns the visual
    video path (no audio, no captions - those are composed later).

    Default path is the WebCodecs direct encode (see _direct_visual); on
    any failure it falls back to frames mode: one screenshot per frame,
    split across `workers` parallel browser instances (frames are
    independent). GPU rendering is tried first; chunks that fail retry
    under SwiftShader.
    """
    from concurrent.futures import ThreadPoolExecutor

    frames_meta = json.loads((job_dir / "frames.json").read_text(encoding="utf-8"))
    n_frames = len(frames_meta["spectrum"])
    url = "http://127.0.0.1:{port}/visualizer.html?mode=render&data=frames.json"
    if (job_dir / "theme.json").exists():
        url += "&theme=theme.json"
    with _Server(job_dir, chunk_path=job_dir / "visual.h264") as srv:
        full_url = url.format(port=srv.port)
        if not frames_mode:
            return _record_visual(job_dir, full_url, duration)
        if direct:
            try:
                return _direct_visual(job_dir, full_url, duration, fps, n_frames)
            except Exception as exc:  # noqa: BLE001 - frames path is the fallback
                print(f"  [render] direct encode unavailable ({exc}); "
                      "falling back to frame capture")
        frames_dir = job_dir / "frames_jpg"
        frames_dir.mkdir(exist_ok=True)
        workers = max(1, min(workers, n_frames))
        chunks = [list(range(w, n_frames, workers)) for w in range(workers)]
        progress = {"done": 0, "total": n_frames}
        for swiftshader in (False, True):
            # a chunk still needs work if any of its frames are missing
            todo = [c for c in chunks if c and
                    any(not (frames_dir / f"frame_{i:05d}.jpg").exists() for i in c)]
            if not todo:
                break
            errors = []

            def _work(chunk, _sw=swiftshader):
                try:
                    missing = [i for i in chunk
                               if not (frames_dir / f"frame_{i:05d}.jpg").exists()]
                    _capture_chunk(full_url, frames_dir, missing, _sw, progress)
                except Exception as exc:  # noqa: BLE001 - collected, retried/raised by caller
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                list(pool.map(_work, todo))
            missing = [i for i in range(n_frames)
                       if not (frames_dir / f"frame_{i:05d}.jpg").exists()]
            if not missing:
                break
            if swiftshader:
                raise RuntimeError(
                    f"frame capture failed ({len(missing)} frames missing): "
                    f"{errors[0] if errors else 'unknown error'}")
            print("  [render] GPU capture failed; retrying with software rendering...")
        print(f"  [render] {n_frames} frames captured")
        visual = job_dir / "visual.mp4"
        _run([
            "ffmpeg", "-y", "-v", "error",
            "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.jpg"),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(visual),
        ])
        return visual


def _record_visual(job_dir: Path, url: str, duration: float) -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            browser = _launch_browser(pw, swiftshader=False)
        except Exception:
            browser = _launch_browser(pw, swiftshader=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.set_default_timeout(int((duration + 60) * 1000))
        page.goto(url)
        page.wait_for_function("window.__ready === true")
        ok = page.evaluate(f"window.__record({duration:.3f})")
        recorded = page.evaluate("window.__recorded")
        error = page.evaluate("window.__recordError")
        browser.close()
        if not ok or not recorded:
            raise RuntimeError(f"MediaRecorder capture failed: {error}")
        visual = job_dir / "visual.webm"
        visual.write_bytes(base64.b64decode(recorded))
        return visual


def fallback_visual(hook_audio: Path, duration: float, out_path: Path) -> Path:
    """ffmpeg-native spectrum visual, used only when Chromium is unavailable."""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(hook_audio),
        "-filter_complex",
        f"[0:a]showspectrum=s={WIDTH}x{HEIGHT}:fps=30:mode=combined:color=viridis:scale=log,format=yuv420p[v]",
        "-map", "[v]", "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(out_path),
    ])
    return out_path


# ---- user overlays (overlays.json in the job dir, written by the UI) ----
# Each overlay: {"type": "image"|"text", "start": s, "end": s (hook-relative
# seconds), "position": preset, ...}. Images carry "path" (relative to the
# job dir) and "width_frac" (of the 1080-wide frame); text carries "text",
# "size" (ASS px) and "color" (#RRGGBB). compose() burns them over the
# visual, under nothing else - they are the topmost layer.
OVERLAY_POSITIONS = ("top-left", "top-center", "top-right", "center",
                     "bottom-left", "bottom-center", "bottom-right")
_ASS_ALIGN = {"top-left": 7, "top-center": 8, "top-right": 9, "center": 5,
              "bottom-left": 1, "bottom-center": 2, "bottom-right": 3}
_OVERLAY_XY = {
    "top-left": ("W*0.04", "H*0.06"), "top-center": ("(W-w)/2", "H*0.06"),
    "top-right": ("W*0.96-w", "H*0.06"), "center": ("(W-w)/2", "(H-h)/2"),
    "bottom-left": ("W*0.04", "H*0.94-h"), "bottom-center": ("(W-w)/2", "H*0.94-h"),
    "bottom-right": ("W*0.96-w", "H*0.94-h"),
}


def _ass_ts(seconds: float) -> str:
    cs = int(round(seconds * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def write_text_overlays_ass(overlays: list[dict], path: Path) -> Path:
    """Render the text overlays as an ASS file at the captions' own
    resolution (1080x1920), one Dialogue per overlay, hook-relative times."""
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
        "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Overlay,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
        "1,0,0,0,100,100,0,0,1,4,0,8,60,60,110,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for o in overlays:
        pos = o.get("position", "top-center")
        align = _ASS_ALIGN.get(pos, 8)
        size = int(o.get("size", 72))
        hexcol = str(o.get("color", "#FFFFFF")).lstrip("#")
        bgr = hexcol[4:6] + hexcol[2:4] + hexcol[0:2] if len(hexcol) == 6 else "FFFFFF"
        text = str(o.get("text", "")).replace("\\", "").replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\r\n", "\\N").replace("\n", "\\N")
        lines.append(
            f"Dialogue: 1,{_ass_ts(float(o['start']))},{_ass_ts(float(o['end']))},"
            f"Overlay,,0,0,0,,{{\\an{align}\\fs{size}\\c&H{bgr}&}}{text}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def validate_overlays(raw, job_dir: Path) -> list[dict]:
    """Clean an overlays list (from the UI or a hand-written file) into
    the schema compose() consumes. Bad entries are dropped, values are
    clamped; image paths must resolve inside the job dir."""
    out = []
    for o in raw or []:
        if not isinstance(o, dict):
            continue
        try:
            start = max(0.0, float(o.get("start", 0.0)))
            end = float(o.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        pos = o.get("position") if o.get("position") in OVERLAY_POSITIONS else "top-center"
        if o.get("type") == "image":
            rel = str(o.get("path", ""))
            full = (job_dir / rel).resolve()
            if not full.is_file() or job_dir.resolve() not in full.parents:
                continue
            out.append({"type": "image", "path": rel, "start": start, "end": end,
                        "position": pos,
                        "width_frac": min(1.0, max(0.05, float(o.get("width_frac", 0.35))))})
        elif o.get("type") == "text":
            text = str(o.get("text", "")).strip()
            if not text:
                continue
            out.append({"type": "text", "text": text[:200], "start": start, "end": end,
                        "position": pos, "size": min(240, max(24, int(o.get("size", 72)))),
                        "color": str(o.get("color", "#FFFFFF"))})
    return out


def compose(visual: Path, hook_audio: Path, ass_path: Path | None, out_path: Path,
            duration: float | None = None, overlays: list[dict] | None = None) -> Path:
    """Burn captions (ASS) over the visual and mux the hook audio.

    When `duration` (the hook window length) is known, the output is
    capped at exactly that length with -t. -shortest alone trusts the
    streams' own durations, which is how MP3 seek slop, AAC priming and
    duration-less MediaRecorder WebMs used to stretch the exported
    short past the hook it was supposed to be."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The ass filter needs a bare filename + cwd; resolve the media paths
    # to absolute first so the cwd switch cannot break them.
    visual = Path(visual).resolve()
    hook_audio = Path(hook_audio).resolve()
    out_path = Path(out_path).resolve()
    base_dir = visual.parent
    overlays = overlays or []
    images = [o for o in overlays if o.get("type") == "image"]
    texts = [o for o in overlays if o.get("type") == "text"]
    text_ass = None
    if texts:
        text_ass = write_text_overlays_ass(texts, base_dir / "overlays_text.ass")
    have_ass = ass_path is not None and Path(ass_path).exists()
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(visual), "-i", str(hook_audio)]
    for o in images:
        cmd += ["-i", str((base_dir / o["path"]).resolve())]
    cwd = None
    if images:
        chains = []
        cur = "0:v"
        for i, o in enumerate(images):
            w = max(2, int(round(WIDTH * float(o.get("width_frac", 0.35)) / 2)) * 2)
            x, y = _OVERLAY_XY.get(o.get("position", "top-center"),
                                   _OVERLAY_XY["top-center"])
            chains.append(f"[{2 + i}:v]scale={w}:-2[ov{i}]")
            chains.append(
                f"[{cur}][ov{i}]overlay={x}:{y}:"
                f"enable='between(t,{float(o['start']):.3f},{float(o['end']):.3f})'[v{i}]")
            cur = f"v{i}"
        ass_names = ([Path(ass_path).name] if have_ass else []) + \
                    ([text_ass.name] if text_ass else [])
        for j, name in enumerate(ass_names):
            chains.append(f"[{cur}]ass={name}[a{j}]")
            cur = f"a{j}"
        cmd += ["-filter_complex", ";".join(chains), "-map", f"[{cur}]", "-map", "1:a"]
        cwd = str(base_dir)
    else:
        vf = []
        if have_ass:
            vf.append(f"ass={Path(ass_path).name}")
        if text_ass:
            vf.append(f"ass={text_ass.name}")
        if vf:
            cmd += ["-vf", ",".join(vf)]
            cwd = str(base_dir)
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "192k",
    ]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    else:
        cmd += ["-shortest"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    _run(cmd, cwd=cwd)
    return out_path


def export_shorts(
    song_stem: str,
    job_dir: Path,
    out_dir: Path,
    duration: float,
    platforms: list[str],
    fps: int = 30,
    frames_mode: bool = True,
    workers: int = 3,
    direct: bool = True,
) -> list[Path]:
    """Full render+export. job_dir must contain visualizer.html, frames.json
    and hook_audio.m4a; out_dir receives one mp4 (+ post text) per platform."""
    hook_audio = job_dir / "hook_audio.m4a"
    try:
        visual = render_visualizer(job_dir, duration, fps=fps,
                                   frames_mode=frames_mode, workers=workers,
                                   direct=direct)
    except Exception as exc:  # Playwright/Chromium missing or WebGL failed
        print(f"  [render] WebGL capture unavailable ({exc}); using ffmpeg fallback visual")
        visual = fallback_visual(hook_audio, duration, job_dir / "visual.mp4")

    overlays = None
    ov_path = job_dir / "overlays.json"
    if ov_path.exists():
        try:
            overlays = validate_overlays(
                json.loads(ov_path.read_text(encoding="utf-8")), job_dir)
        except Exception:  # noqa: BLE001 - a bad overlays file must not kill the export
            overlays = None
    results = []
    for platform in platforms:
        preset = PLATFORM_PRESETS[platform]
        platform_ass = job_dir / f"captions_{platform}.ass"
        ass_path = platform_ass if platform_ass.exists() else job_dir / "captions.ass"
        out_path = out_dir / f"{song_stem}_hook_{preset['suffix']}.mp4"
        compose(visual, hook_audio, ass_path, out_path, duration=duration,
                overlays=overlays)
        post = out_dir / f"{song_stem}_hook_{preset['suffix']}.post.txt"
        post.write_text(
            f"{song_stem} - the hook\n#music #newmusic #hook #shorts\n",
            encoding="utf-8",
        )
        results.append(out_path)
    return results


def stage_visualizer_assets(job_dir: Path, repo_root: Path) -> None:
    """Copy the visualizer page into a job/preview directory."""
    src = repo_root / "visualizer" / "visualizer.html"
    if src.exists():
        shutil.copy2(src, job_dir / "visualizer.html")