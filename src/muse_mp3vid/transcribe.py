"""Stage 1: Whisper transcription via faster-whisper (word-level timestamps).

Audio is decoded with ffmpeg (already a pipeline dependency) and handed to
Whisper as raw PCM, instead of letting faster-whisper decode via PyAV -
PyAV's av.open() signature changes between releases and has broken
faster-whisper's file path before.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np

# HF's Xet storage backend gets its connections reset on some networks
# (WinError 10054 on the xet-read-token request, then the download dies
# before any model bytes move). Force the classic HTTPS download path,
# which is resumable and battle-tested. Must be set before
# huggingface_hub is imported (it is only imported lazily below).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WHISPER_SR = 16000

_CONN_MARKERS = ("ConnectError", "10054", "10038", "Connection",
                 "timed out", "Timeout", "LocalEntryNotFoundError")


def cached_model_path(model_size: str) -> str | None:
    """Local snapshot path if this model is already in the HF cache.

    Some networks reset connections to huggingface.co outright (WinError
    10054 on the very first API call). When the model is already cached,
    loading it by path skips the network entirely - no repo_info check,
    no download attempt, nothing to reset.
    """
    if os.path.isdir(model_size):
        return model_size
    # Look in the cache directly first: faster-whisper's own download skips
    # repo metadata files (.gitattributes, README.md), which makes
    # snapshot_download(local_files_only=True) reject the snapshot as
    # "incomplete" even though every file Whisper needs is present.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = Path(HF_HUB_CACHE) / f"models--Systran--faster-whisper-{model_size}"
        snaps = repo / "snapshots"
        if snaps.is_dir():
            candidates = [p for p in snaps.iterdir() if p.is_dir()]
            ref = repo / "refs" / "main"
            if ref.exists():
                want = ref.read_text(encoding="utf-8").strip()
                candidates.sort(key=lambda p: p.name != want)
            for snap in candidates:
                if (snap / "model.bin").exists() and (snap / "config.json").exists():
                    return str(snap)
    except Exception:
        pass
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            f"Systran/faster-whisper-{model_size}", local_files_only=True
        )
    except Exception:
        return None


def _cpu_threads(device: str) -> int:
    """CTranslate2 defaults to 4 CPU threads whatever the machine has;
    use every logical core on CPU. 0 = library default (used for GPU)."""
    return (os.cpu_count() or 4) if device == "cpu" else 0


@functools.lru_cache(maxsize=2)
def _load_model(model_size: str, device: str, compute_type: str):
    """Cache-first WhisperModel load with retries on connection resets.
    Memoized so the full-song pass and the hook-window pass share one load."""
    import time

    from faster_whisper import WhisperModel

    threads = _cpu_threads(device)
    cached = cached_model_path(model_size)
    if cached:
        return WhisperModel(cached, device=device, compute_type=compute_type,
                            cpu_threads=threads)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return WhisperModel(model_size, device=device, compute_type=compute_type,
                                cpu_threads=threads)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
            last_exc = exc
            if not any(m in str(exc) or m in type(exc).__name__ for m in _CONN_MARKERS):
                raise
            if attempt < 2:
                time.sleep(2 + attempt * 3)
    raise RuntimeError(
        f"could not download Whisper model '{model_size}': the connection to "
        "huggingface.co was reset (this network drops HF connections). "
        "If you have run another model before, rerun with that one - e.g. "
        "--model medium - since cached models now load fully offline. "
        f"Original error: {last_exc}"
    )


def _decode_for_whisper(audio_path: str | Path, start: float = 0.0,
                        end: float | None = None) -> np.ndarray:
    span = ["-ss", f"{start:.3f}"] if start > 0 else []
    if end is not None:
        span += ["-t", f"{max(0.0, end - start):.3f}"]
    cmd = [
        "ffmpeg", "-v", "error", *span, "-i", str(audio_path),
        "-f", "f32le", "-ac", "1", "-ar", str(WHISPER_SR), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for {audio_path}: {proc.stderr.decode(errors='replace')[:400]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def segment_coverage(segments: list[dict], duration: float) -> float:
    """Fraction of the song covered by transcribed segments."""
    if duration <= 0:
        return 0.0
    covered = sum(max(0.0, min(s["end"], duration) - s["start"]) for s in segments)
    return min(1.0, covered / duration)


def _transcribe_once(model, samples, language, beam_size, vad_filter,
                     word_timestamps=True, fallback=True):
    kw = {}
    if not fallback:
        # Default temperature ladder re-decodes any low-confidence segment up
        # to 5 more times (best_of=5); sung vocals trip it constantly.
        kw["temperature"] = 0.0
    segments_iter, info = model.transcribe(
        samples,
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        **kw,
    )
    segments = []
    for seg in segments_iter:
        words = [
            {
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "word": w.word,
                "probability": round(w.probability, 4),
            }
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words,
            }
        )
    return segments, info


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    beam_size: int = 5,
    word_timestamps: bool = True,
    fallback: bool = True,
) -> dict:
    """Transcribe an audio file. Returns a JSON-serializable dict:

    {
      "language": str, "language_probability": float, "duration": float,
      "model": str,
      "segments": [{"id", "start", "end", "text",
                    "words": [{"start", "end", "word", "probability"}]}]
    }

    faster-whisper's VAD is tuned for speech and can strip sung vocals out
    of a song entirely (near-empty transcript -> no captions, no sections).
    If the VAD pass covers < 25% of a song-length file, retry without VAD
    and keep whichever pass heard more of the song.
    """
    model = _load_model(model_size, device, compute_type)
    samples = _decode_for_whisper(audio_path)
    segments, info = _transcribe_once(model, samples, language, beam_size, True,
                                      word_timestamps, fallback)
    duration = float(info.duration)
    coverage = segment_coverage(segments, duration)
    vad_fallback = False
    if duration >= 20.0 and coverage < 0.25:
        print(f"  [transcribe] VAD pass covered only {coverage:.0%} of the song; "
              "retrying without VAD (VAD strips sung vocals)...")
        segments2, info2 = _transcribe_once(model, samples, language, beam_size, False,
                                            word_timestamps, fallback)
        coverage2 = segment_coverage(segments2, float(info2.duration))
        if coverage2 > coverage:
            segments, info, coverage = segments2, info2, coverage2
            vad_fallback = True
        print(f"  [transcribe] no-VAD coverage {coverage2:.0%}; "
              f"using {'no-VAD' if vad_fallback else 'VAD'} pass")
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "model": model_size,
        "vad_fallback": vad_fallback,
        "segments": segments,
    }


def window_needs_words(transcript: dict, start: float, end: float) -> bool:
    """True if [start, end] has no word timings yet. Windowed transcripts
    (hook-only mode) record which windows were transcribed; legacy
    whole-song transcripts are checked segment by segment."""
    if transcript.get("windowed"):
        return not any(a <= start + 1e-6 and end <= b + 1e-6
                       for a, b in transcript.get("refined_windows", []))
    return any(
        not s.get("words")
        for s in transcript.get("segments", [])
        if s["end"] > start and s["start"] < end
    )


def refine_window(
    audio_path: str | Path,
    transcript: dict,
    start: float,
    end: float,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    pad: float = 1.0,
    language: str | None = None,
) -> dict:
    """Re-transcribe only the hook window with word timestamps and beam
    search, then splice it over the coarse segments it replaces. Captions
    need word timings for ~30 s, not for the whole song. Returns a new
    transcript dict; the input is not mutated."""
    lo = max(0.0, start - pad)
    hi = end + pad
    model = _load_model(model_size, device, compute_type)
    samples = _decode_for_whisper(audio_path, lo, hi)
    lang = transcript.get("language") or language
    vad = not transcript.get("vad_fallback", False)
    segs, info = _transcribe_once(model, samples, lang, beam_size, vad,
                                  word_timestamps=True, fallback=True)
    span = hi - lo
    if vad and span >= 20.0 and segment_coverage(segs, span) < 0.25:
        # VAD can strip sung vocals entirely; keep whichever pass heard more
        segs2, info2 = _transcribe_once(model, samples, lang, beam_size, False,
                                        word_timestamps=True, fallback=True)
        if segment_coverage(segs2, span) > segment_coverage(segs, span):
            segs, info, vad = segs2, info2, False
    for s in segs:  # window-relative -> song-absolute
        s["start"] = round(s["start"] + lo, 3)
        s["end"] = round(s["end"] + lo, 3)
        for w in s["words"]:
            w["start"] = round(w["start"] + lo, 3)
            w["end"] = round(w["end"] + lo, 3)
    kept = [s for s in transcript["segments"] if s["end"] <= lo or s["start"] >= hi]
    merged = sorted(kept + segs, key=lambda s: s["start"])
    for i, s in enumerate(merged):
        s["id"] = i
    out = dict(transcript)
    out["segments"] = merged
    out["refined_window"] = [round(lo, 3), round(hi, 3)]
    out["refined_windows"] = list(transcript.get("refined_windows", [])) + [[round(lo, 3), round(hi, 3)]]
    out["vad_fallback"] = not vad
    if not out.get("language"):
        out["language"] = info.language
        out["language_probability"] = round(info.language_probability, 4)
    return out


def save_transcript(transcript: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(transcript, indent=2), encoding="utf-8")


def load_transcript(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_words(transcript: dict) -> list[dict]:
    """All words across segments, absolute times, text stripped. Falls back to
    segment-level pseudo-words when word timestamps are absent."""
    words: list[dict] = []
    for seg in transcript.get("segments", []):
        if seg.get("words"):
            for w in seg["words"]:
                text = w["word"].strip()
                if text:
                    words.append({"start": w["start"], "end": w["end"], "text": text})
        elif seg.get("text"):
            words.append(
                {"start": seg["start"], "end": seg["end"], "text": seg["text"], "segment_level": True}
            )
    words.sort(key=lambda w: w["start"])
    return words