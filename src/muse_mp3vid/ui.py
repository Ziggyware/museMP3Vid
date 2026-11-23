"""Local web UI for museMP3Vid.

    muse-mp3vid SONG.mp3 --ui [--ui-port 8765]

One page, four steps: pick the hook on the waveform (audio analysis only,
no transcription yet), transcribe that window - once, under the same
once-only rule as the CLI - edit the lyric lines, add image/text overlays,
then generate. Generation runs the normal CLI pipeline in a thread with
--hook-start set to the chosen window, so everything downstream (theme,
captions, render, the overlays.json this page writes) is exactly the CLI's
own machinery.
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import analyze, hook as hook_mod, render as render_mod
from .transcribe import (apply_lyric_edits, load_transcript, refine_window,
                         save_transcript, song_fingerprint, window_needs_words)

PAGE = Path(__file__).with_name("ui_page.html")


def _segments_in(transcript: dict, start: float, end: float) -> list[dict]:
    return [
        {"start": s["start"], "end": s["end"], "text": s.get("text", "")}
        for s in transcript.get("segments", [])
        if s["end"] > start and s["start"] < end
    ]


class _State:
    def __init__(self, song: Path, out_dir: Path, cfg: dict):
        self.song = song
        self.out_dir = out_dir
        self.cfg = cfg
        self.samples = analyze.decode_audio(str(song))
        self.rms = analyze.rms_series(self.samples)
        self.duration = len(self.samples) / analyze.SAMPLE_RATE
        self._feats = None
        self.gen = {"running": False, "log": [], "results": [], "error": None}
        self.gen_lock = threading.Lock()

    def feats(self) -> np.ndarray:
        if self._feats is None:
            coarse = analyze.frame_features(self.samples, fps=2)
            chroma = analyze.chroma_features(self.samples, fps=2)
            spec = np.asarray(coarse["spectrum"], dtype=np.float32)
            m = min(len(chroma), len(spec))
            self._feats = np.hstack([chroma[:m] * 2.3, spec[:m]])
        return self._feats

    def candidates(self, hook_len: float) -> list[dict]:
        cands = hook_mod.detect_hooks_audio(self.feats(), self.rms, self.duration,
                                            fps=2, hook_len=hook_len, top_n=3)
        out = []
        for c in cands:
            e = hook_mod.enforce_min_window(c, self.duration) or c
            out.append({"start": e["start"], "end": e["end"],
                        "score": e.get("score")})
        return out

    @property
    def transcript_path(self) -> Path:
        return self.out_dir / "transcript.json"

    def load_transcript(self) -> dict | None:
        if self.transcript_path.exists():
            return load_transcript(self.transcript_path)
        return None

    def transcribe(self, start: float, end: float, retranscribe: bool) -> dict:
        """The song's one transcription, on the user's chosen window.
        Refuses to decode a second window unless retranscribe is set -
        the same once-only rule the CLI enforces."""
        fp = song_fingerprint(self.song)
        existing = self.load_transcript()
        if existing is not None and not retranscribe:
            if existing.get("source_fingerprint") in (None, fp) \
                    and not window_needs_words(existing, start, end):
                return {"segments": _segments_in(existing, start, end),
                        "reused": True}
            if existing.get("segments"):
                raise RuntimeError(
                    "this song already has its one transcription, for a "
                    "different window. Redo it explicitly if the new hook "
                    "is the right one.")
        transcript = {
            "language": self.cfg.get("language"), "language_probability": None,
            "duration": round(self.duration, 3), "model": self.cfg["model"],
            "vad_fallback": False, "windowed": True, "refined_windows": [],
            "segments": [],
        }
        transcript = refine_window(
            self.song, transcript, start, end,
            model_size=self.cfg["model"], device=self.cfg["device"],
            compute_type=self.cfg["compute_type"],
            language=self.cfg.get("language"), pad=3.0,
            fast=self.cfg.get("fast", False))
        transcript["duration"] = round(self.duration, 3)
        transcript["source_fingerprint"] = fp
        save_transcript(transcript, self.transcript_path)
        return {"segments": _segments_in(transcript, start, end), "reused": False}


class _LogWriter:
    def __init__(self, state: _State):
        self.state = state
        self.buf = ""

    def write(self, text: str) -> int:
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.state.gen["log"].append(line)
        return len(text)

    def flush(self) -> None:
        if self.buf:
            self.state.gen["log"].append(self.buf)
            self.buf = ""


def _run_generate(state: _State, start: float, end: float, platforms: list[str]) -> None:
    from . import cli as cli_mod

    writer = _LogWriter(state)
    try:
        argv = [str(state.song), "--out", str(state.out_dir),
                "--hook-start", f"{start:.3f}", "--hook-len", f"{end - start:.3f}",
                "--platforms", ",".join(platforms) or "tiktok",
                "--model", state.cfg["model"], "--fps", str(state.cfg.get("fps", 30))]
        if not state.cfg.get("direct", True):
            argv.append("--no-direct")
        with redirect_stdout(writer), redirect_stderr(writer):
            code = cli_mod.main(argv)
        writer.flush()
        stem = state.song.stem.replace(" ", "_")
        state.gen["results"] = sorted(
            p.name for p in state.out_dir.glob(f"{stem}_hook_*.mp4"))
        state.gen["error"] = None if code == 0 else f"pipeline exited {code}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the page
        state.gen["error"] = str(exc)
    finally:
        state.gen["running"] = False


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console clean
            pass

        # -- helpers ------------------------------------------------------
        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def _serve_file(self, path: Path):
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            status = 200
            range_hdr = self.headers.get("Range")
            if range_hdr:
                m = re.match(r"bytes=(\d*)-(\d*)", range_hdr)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                    status = 206
            length = max(0, end - start + 1)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        # -- routes -------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == "/":
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/state":
                    tr = state.load_transcript()
                    self._json({
                        "song": state.song.name, "duration": state.duration,
                        "min_window": hook_mod.MIN_VIDEO_SECONDS,
                        "transcribed_windows": (tr or {}).get("refined_windows", []),
                        "positions": list(render_mod.OVERLAY_POSITIONS),
                    })
                elif url.path == "/api/waveform":
                    peaks = [round(float(r["rms"]), 4) for r in state.rms]
                    self._json({"hop": 0.25, "peaks": peaks})
                elif url.path == "/api/candidates":
                    hook_len = float(q.get("len", ["30"])[0])
                    self._json({"candidates": state.candidates(hook_len)})
                elif url.path == "/api/lyrics":
                    tr = state.load_transcript()
                    if tr is None:
                        self._json({"segments": []})
                    else:
                        start = float(q.get("start", ["0"])[0])
                        end = float(q.get("end", [str(state.duration)])[0])
                        self._json({"segments": _segments_in(tr, start, end)})
                elif url.path == "/api/overlays":
                    p = state.out_dir / "overlays.json"
                    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
                    self._json({"overlays": render_mod.validate_overlays(raw, state.out_dir)})
                elif url.path == "/api/status":
                    self._json(state.gen)
                elif url.path == "/media/song":
                    self._serve_file(state.song)
                elif url.path.startswith("/media/out/"):
                    name = Path(url.path.removeprefix("/media/out/")).name
                    self._serve_file(state.out_dir / name)
                else:
                    self.send_error(404)
            except Exception as exc:  # noqa: BLE001 - reported to the page
                self._json({"error": str(exc)}, status=500)

        def do_POST(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == "/api/transcribe":
                    body = self._body()
                    out = state.transcribe(float(body["start"]), float(body["end"]),
                                           bool(body.get("retranscribe")))
                    self._json(out)
                elif url.path == "/api/lyrics":
                    body = self._body()
                    tr = state.load_transcript()
                    if tr is None:
                        self._json({"error": "no transcript yet"}, status=400)
                        return
                    tr = apply_lyric_edits(tr, body.get("segments", []),
                                           float(body["start"]), float(body["end"]))
                    save_transcript(tr, state.transcript_path)
                    self._json({"ok": True,
                                "segments": _segments_in(tr, float(body["start"]),
                                                         float(body["end"]))})
                elif url.path == "/api/overlays":
                    body = self._body()
                    cleaned = render_mod.validate_overlays(body.get("overlays", []),
                                                           state.out_dir)
                    (state.out_dir / "overlays.json").write_text(
                        json.dumps(cleaned, indent=2), encoding="utf-8")
                    self._json({"ok": True, "overlays": cleaned})
                elif url.path == "/api/upload":
                    name = re.sub(r"[^A-Za-z0-9._-]", "_",
                                  Path(q.get("name", ["image.png"])[0]).name)
                    data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    dest_dir = state.out_dir / "assets"
                    dest_dir.mkdir(exist_ok=True)
                    (dest_dir / name).write_bytes(data)
                    self._json({"path": f"assets/{name}"})
                elif url.path == "/api/generate":
                    body = self._body()
                    with state.gen_lock:
                        if state.gen["running"]:
                            self._json({"error": "generation already running"},
                                       status=409)
                            return
                        state.gen = {"running": True, "log": [], "results": [],
                                     "error": None}
                    t = threading.Thread(
                        target=_run_generate,
                        args=(state, float(body["start"]), float(body["end"]),
                              body.get("platforms", ["tiktok"])),
                        daemon=True)
                    t.start()
                    self._json({"ok": True})
                else:
                    self.send_error(404)
            except Exception as exc:  # noqa: BLE001 - reported to the page
                self._json({"error": str(exc)}, status=500)

    return Handler


def serve(song: Path, out_dir: Path, port: int = 8765, model: str = "small",
          device: str = "cpu", compute_type: str = "int8",
          language: str | None = None, fast: bool = False,
          fps: int = 30, direct: bool = True) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"model": model, "device": device, "compute_type": compute_type,
           "language": language, "fast": fast, "fps": fps, "direct": direct}
    print(f"[ui] analyzing {song.name} ...", flush=True)
    state = _State(song, out_dir, cfg)
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(state))
    print(f"[ui] open http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
