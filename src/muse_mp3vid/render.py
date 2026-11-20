"""Stages 4-6: WebGL visualizer rendering + ffmpeg composition + platform export.

The visualizer page renders deterministically from frames.json. Two capture
paths:
  - frames (default): one screenshot per frame -> visual.mp4. Exact frame
    timing on any machine, including CPU-only SwiftShader rendering.
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

PLATFORM_PRESETS = {
    "tiktok": {"suffix": "tiktok", "label": "TikTok"},
    "reels": {"suffix": "reels", "label": "Instagram Reels"},
    "shorts": {"suffix": "shorts", "label": "YouTube Shorts"},
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


class _Server:
    def __init__(self, directory: Path):
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()


def render_visualizer(job_dir: Path, duration: float, fps: int = 30, frames_mode: bool = True) -> Path:
    """Render the WebGL visualizer for `duration` seconds. Returns the visual
    video path (no audio, no captions - those are composed later)."""
    from playwright.sync_api import sync_playwright

    frames_meta = json.loads((job_dir / "frames.json").read_text(encoding="utf-8"))
    n_frames = len(frames_meta["spectrum"])
    url = f"http://127.0.0.1:{{port}}/visualizer.html?mode=render&data=frames.json"
    with _Server(job_dir) as srv, sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--mute-audio",
            "--disable-dev-shm-usage",
        ])
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.set_default_timeout(int((duration + 60) * 1000))
        page.goto(url.format(port=srv.port))
        page.wait_for_function("window.__ready === true")
        if frames_mode:
            frames_dir = job_dir / "frames_jpg"
            frames_dir.mkdir(exist_ok=True)
            for i in range(n_frames):
                page.evaluate(f"window.__drawFrame({i})")
                page.screenshot(path=str(frames_dir / f"frame_{i:05d}.jpg"),
                                type="jpeg", quality=90)
            browser.close()
            visual = job_dir / "visual.mp4"
            _run([
                "ffmpeg", "-y", "-v", "error",
                "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.jpg"),
                "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                "-pix_fmt", "yuv420p", "-r", str(fps), str(visual),
            ])
            return visual
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
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
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
) -> list[Path]:
    """Full render+export. job_dir must contain visualizer.html, frames.json
    and hook_audio.m4a; out_dir receives one mp4 (+ post text) per platform."""
    hook_audio = job_dir / "hook_audio.m4a"
    ass_path = job_dir / "captions.ass"
    try:
        visual = render_visualizer(job_dir, duration, fps=fps, frames_mode=frames_mode)
    except Exception as exc:  # Playwright/Chromium missing or WebGL failed
        print(f"  [render] WebGL capture unavailable ({exc}); using ffmpeg fallback visual")
        visual = fallback_visual(hook_audio, duration, job_dir / "visual.mp4")

    results = []
    for platform in platforms:
        preset = PLATFORM_PRESETS[platform]
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
