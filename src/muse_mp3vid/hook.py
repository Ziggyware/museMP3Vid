"""Stage 2: best-hook detector.

Scores candidate windows of `hook_len` seconds on three signals:

  repetition  - lyric lines inside the window that repeat elsewhere in the
                song (the chorus effect). The single strongest hook signal.
  energy      - mean RMS loudness in the window (hooks are rarely the
                quietest part of the song).
  coverage    - fraction of the window actually covered by sung words
                (penalizes intros, solos and dead air).

score = 0.45*repetition + 0.30*energy + 0.25*coverage

Pure functions over plain dicts -> unit-testable without audio or Whisper.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def line_repetition(segments: list[dict], threshold: float = 0.7) -> list[float]:
    """Per-segment repetition score in 0..1: how strongly this segment's line
    is repeated by *other* segments (0 = unique, 1 = repeated many times)."""
    token_sets = [set(_tokens(s.get("text", ""))) for s in segments]
    scores: list[float] = []
    for i, toks in enumerate(token_sets):
        if len(toks) < 2:
            scores.append(0.0)
            continue
        repeats = 0
        for j, other in enumerate(token_sets):
            if i == j or len(other) < 2:
                continue
            if _jaccard(toks, other) >= threshold:
                repeats += 1
        # 1 repeat -> 0.5, 2 -> 0.67, 3 -> 0.75 ... saturating
        scores.append(repeats / (repeats + 1) if repeats else 0.0)
    return scores


def _window_energy(rms: list[dict], start: float, end: float, peak: float) -> float:
    vals = [p["rms"] for p in rms if start <= p["t"] < end]
    if not vals or peak <= 0:
        return 0.0
    return min(1.0, (sum(vals) / len(vals)) / peak)


def _window_coverage(segments: list[dict], start: float, end: float) -> float:
    span = end - start
    if span <= 0:
        return 0.0
    covered = 0.0
    for s in segments:
        covered += max(0.0, min(s["end"], end) - max(s["start"], start))
    return min(1.0, covered / span)


def _window_repetition(segments: list[dict], rep: list[float], start: float, end: float) -> float:
    total = 0.0
    weight = 0.0
    for s, r in zip(segments, rep):
        overlap = max(0.0, min(s["end"], end) - max(s["start"], start))
        if overlap > 0:
            total += r * overlap
            weight += overlap
    return total / weight if weight > 0 else 0.0


def detect_hooks(
    transcript: dict,
    rms: list[dict],
    hook_len: float = 30.0,
    top_n: int = 3,
) -> list[dict]:
    """Ranked hook candidates: [{"start", "end", "score",
    "reasons": {"repetition", "energy", "coverage"}}] best first."""
    duration = float(transcript.get("duration") or 0.0)
    segments = transcript.get("segments", [])
    if duration <= 0 and segments:
        duration = max(s["end"] for s in segments)
    if duration <= 0:
        return []

    hook_len = min(hook_len, duration)
    peak_rms = max((p["rms"] for p in rms), default=0.0)
    rep = line_repetition(segments)

    # Candidate starts: every segment start that fits, plus 0. Snapping to a
    # segment start means the short never begins mid-word. A 2s grid is added
    # so instrumental tracks (no segments) still get energy-driven candidates.
    starts = {0.0}
    for s in segments:
        if s["start"] + 1.0 <= duration:
            starts.add(round(s["start"], 3))
    t = 0.0
    while t + 1.0 <= duration:
        starts.add(round(t, 3))
        t += 2.0

    candidates: list[dict] = []
    for start in sorted(starts):
        end = min(duration, start + hook_len)
        if end - start < min(5.0, hook_len):
            continue
        r = _window_repetition(segments, rep, start, end)
        e = _window_energy(rms, start, end, peak_rms)
        c = _window_coverage(segments, start, end)
        score = 0.45 * r + 0.30 * e + 0.25 * c
        candidates.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(float(score), 4),
                "reasons": {
                    "repetition": round(float(r), 4),
                    "energy": round(float(e), 4),
                    "coverage": round(float(c), 4),
                },
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]
