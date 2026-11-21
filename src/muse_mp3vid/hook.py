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

# No exported video may be shorter than this. The CLI enforces it on the
# chosen hook window (see enforce_min_window) and verifies it on the
# rendered files, so a short can never slip out under this length.
MIN_VIDEO_SECONDS = 10.0


def enforce_min_window(chosen: dict, duration: float,
                       min_seconds: float = MIN_VIDEO_SECONDS) -> dict | None:
    """Return `chosen` guaranteed to span at least `min_seconds`, or None
    when the song itself is shorter than the minimum (no honest window
    exists). A short window is extended forward first - the start is the
    detector's best guess and worth keeping - and shifted back only when
    it runs into the end of the song. An already-long-enough window is
    returned unchanged (the same object)."""
    if duration < min_seconds:
        return None
    start = min(max(0.0, float(chosen["start"])), duration - min_seconds)
    end = float(chosen["end"])
    if end - start < min_seconds - 1e-6:
        end = min(duration, start + min_seconds)
    if end - start < min_seconds - 1e-6:
        start = end - min_seconds
    if start == float(chosen["start"]) and end == float(chosen["end"]):
        return chosen
    out = dict(chosen)
    out["start"] = round(start, 3)
    out["end"] = round(end, 3)
    reasons = dict(chosen.get("reasons") or {})
    reasons["min_length"] = f"window adjusted to the {min_seconds:.0f}s minimum video length"
    out["reasons"] = reasons
    return out


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
        # Full-length windows only: a start too close to the end of
        # the song would truncate the window. (The minimum video
        # length itself is enforced later, on the chosen window -
        # see enforce_min_window.)
        if end - start < min(hook_len, duration) - 1e-6:
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


def detect_hooks_audio(
    features,
    rms: list[dict],
    duration: float,
    fps: int = 2,
    hook_len: float = 30.0,
    top_n: int = 3,
) -> list[dict]:
    """Hook candidates from audio alone (no transcript): a window scores high
    when its feature sequence recurs elsewhere in the song (self-similarity
    along diagonals of the frame-by-frame cosine matrix) and is loud.

        score = 0.6 * repetition + 0.4 * energy

    Every candidate is exactly `hook_len` long (or the whole song if shorter).
    `features`: [frames, dims] at `fps` frames/second."""
    import numpy as np

    n = len(features)
    if duration <= 0 or n == 0:
        return []
    hook_len = min(hook_len, duration)
    w = max(1, int(round(hook_len * fps)))
    if n < w:
        return [{"start": 0.0, "end": round(duration, 3), "score": 0.0,
                 "reasons": {"repetition": 0.0, "energy": 0.0}}]

    X = np.asarray(features, dtype=np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    S = X @ X.T
    rep = np.zeros(n - w + 1)
    for lag in range(w, n - w + 1):          # partner window must not overlap
        d = np.diagonal(S, offset=lag)
        cs = np.concatenate([[0.0], np.cumsum(d)])
        m = (cs[w:] - cs[:-w]) / w           # mean similarity, window start j
        rep[: len(m)] = np.maximum(rep[: len(m)], m)
        rep[lag: lag + len(m)] = np.maximum(rep[lag: lag + len(m)], m)
    rep = np.clip(rep, 0.0, 1.0)

    peak = max((p["rms"] for p in rms), default=0.0)
    cands = []
    for i in range(len(rep)):
        start = i / fps
        end = start + hook_len
        e = _window_energy(rms, start, end, peak)
        cands.append({
            "start": round(start, 3), "end": round(min(end, duration), 3),
            "score": round(float(0.6 * rep[i] + 0.4 * e), 4),
            "reasons": {"repetition": round(float(rep[i]), 4), "energy": round(float(e), 4)},
        })
    cands.sort(key=lambda c: c["score"], reverse=True)
    picked: list[dict] = []
    for c in cands:                           # greedy, keep candidates apart
        if all(abs(c["start"] - p["start"]) >= hook_len / 2 for p in picked):
            picked.append(c)
        if len(picked) >= top_n:
            break
    return picked


def snap_start(segments: list[dict], start: float, hook_len: float,
               duration: float, max_shift: float = 2.5) -> float:
    """Move `start` to the nearest sung-line start within +-max_shift so the
    short does not begin mid-word. Unchanged if no segment starts nearby."""
    best = None
    for s in segments:
        t = float(s["start"])
        if abs(t - start) <= max_shift and t + min(hook_len, duration) <= duration + 1e-6:
            if best is None or abs(t - start) < abs(best - start):
                best = t
    return round(best, 3) if best is not None else start