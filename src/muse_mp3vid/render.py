"""Stages 4-6: WebGL visualizer rendering + ffmpeg composition + platform export.

The visualizer page renders deterministically from frames.json. Capture
paths, fastest first:
  - direct (default): the page encodes H.264 itself via WebCodecs, as fast
    as it can draw, streaming chunks to visual.h264 over HTTP; ffmpeg only
    muxes (-c copy). Falls back to frames on any failure.
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
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", song,
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
    """Chromium launch. GPU (default ANGLE) first on real machines - forcing
    SwiftShader everywhere made renders crawl on hardware that has a GPU.
    SwiftShader remains the fallback for GPU-less environments."""
    args = ["--mute-audio", "--disable-dev-shm-usage"]
    if swiftshader:
        args += ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
    return pw.chromium.launch(args=args)


def _capture_chunk(url: str, frames_dir: Path, indices: list[int],
                   swiftshader: bool, progress: dict) -> None:
    """Draw + screenshot `indices` in this thread's own browser instance."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = _launch_browser(pw, swiftshader)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.set_default_timeout(30000)
        page.goto(url)
        page.wait_for_function("window.__ready === true", timeout=20000)
        for i in indices:
            page.evaluate(f"window.__drawFrame({i})")
            page.screenshot(path=str(frames_dir / f"frame_{i:05d}.jpg"),
                            type="jpeg", quality=90)
            progress["done"] += 1
            if progress["done"] % 60 == 0:
                print(f"  [render] frame {progress['done']}/{progress['total']}")
        browser.close()


def _direct_visual(job_dir: Path, url: str, duration: float, fps: int) -> Path:
    """WebCodecs direct encode: the page encodes H.264 itself (as fast as it
    can draw, not real-time) and streams chunks to visual.h264 via
    POST /chunk; ffmpeg then muxes with -c copy - no re-encode, no
    screenshot files. Raises RuntimeError on any failure so the caller can
    fall back to frame capture."""
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

            deadline = _time.monotonic() + max(300.0, duration * 45.0)
            last_print = 0.0
            while True:
                if page.evaluate("window.__directDone === true"):
                    break
                now = _time.monotonic()
                if now > deadline:
                    raise RuntimeError("direct encode timed out")
                if now - last_print >= 30.0:
                    sent = page.evaluate("window.__directBytes || 0")
                    print(f"  [render] direct encode streaming... {sent} bytes")
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
    visual = job_dir / "visual.mp4"
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(fps), "-i", str(h264_path),
        "-c", "copy", "-movflags", "+faststart", str(visual),
    ])
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
                return _direct_visual(job_dir, full_url, duration, fps)
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


def compose(visual: Path, hook_audio: Path, ass_path: Path | None, out_path: Path) -> Path:
    """Burn captions (ASS) over the visual and mux the hook audio."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(visual), "-i", str(hook_audio)]
    cwd = None
    if ass_path is not None and ass_path.exists():
        cmd += ["-vf", f"ass={ass_path.name}"]
        cwd = str(ass_path.parent)
    cmd += [
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(out_path),
    ]
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

    results = []
    for platform in platforms:
        preset = PLATFORM_PRESETS[platform]
        platform_ass = job_dir / f"captions_{platform}.ass"
        ass_path = platform_ass if platform_ass.exists() else job_dir / "captions.ass"
        out_path = out_dir / f"{song_stem}_hook_{preset['suffix']}.mp4"
        compose(visual, hook_audio, ass_path, out_path)
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
