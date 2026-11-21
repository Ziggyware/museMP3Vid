"""Audio decoding + feature extraction (RMS energy, log-band spectrum, beat pulse).

Pure-numpy features over ffmpeg-decoded PCM, so the feature functions are
unit-testable without any audio files.
"""
from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 22050
BANDS = 64
F_MIN = 40.0


def decode_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono float32 PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {proc.stderr.decode(errors='replace')[:400]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def probe_duration(path: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}")
    return float(proc.stdout.strip())


def rms_series(samples: np.ndarray, sr: int = SAMPLE_RATE, hop_seconds: float = 0.25) -> list[dict]:
    """Coarse RMS energy over time: [{"t": seconds, "rms": 0..~1}, ...]."""
    hop = max(1, int(sr * hop_seconds))
    out = []
    for i in range(0, max(1, len(samples) - hop + 1), hop):
        frame = samples[i : i + hop]
        if len(frame) == 0:
            break
        out.append({"t": round(i / sr, 3), "rms": round(float(np.sqrt(np.mean(frame**2))), 5)})
    return out


def _band_edges(sr: int, bands: int = BANDS, f_min: float = F_MIN) -> np.ndarray:
    return np.logspace(np.log10(f_min), np.log10(sr / 2), bands + 1)


def frame_features(
    samples: np.ndarray,
    sr: int = SAMPLE_RATE,
    fps: int = 30,
    bands: int = BANDS,
    start: float = 0.0,
    end: float | None = None,
) -> dict:
    """Per-frame spectrum + RMS + beat pulse for the visualizer.

    Returns {"fps", "bands", "duration", "rms": [...], "pulse": [...],
             "spectrum": [[band values 0..1] x frames]}
    Times are relative to `start` (seconds into the source audio).
    """
    lo = int(start * sr)
    hi = int(end * sr) if end is not None else len(samples)
    clip = samples[lo:hi]
    duration = len(clip) / sr
    hop = max(1, int(round(sr / fps)))
    win = 2048
    window = np.hanning(win).astype(np.float32)
    edges = _band_edges(sr, bands)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    band_idx = np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, bands - 1)

    # round, not truncate: int() systematically dropped the final partial
    # frame, so every visual came out up to one frame shorter than its audio
    n_frames = max(1, int(round(duration * fps)))
    spectrum = np.zeros((n_frames, bands), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    for f in range(n_frames):
        center = f * hop
        seg = clip[center : center + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        rms[f] = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        mag = np.abs(np.fft.rfft(seg * window))
        # mean magnitude per log band
        sums = np.bincount(band_idx, weights=mag, minlength=bands)
        counts = np.bincount(band_idx, minlength=bands)
        spectrum[f] = sums / np.maximum(counts, 1)

    # log-scale, then normalize by a robust global peak so quiet songs still move
    spectrum = np.log1p(spectrum * 50.0)
    peak = np.percentile(spectrum, 99.5) if spectrum.size else 1.0
    if peak > 0:
        spectrum = np.clip(spectrum / peak, 0.0, 1.0)
    # Per-band stretch: mastered songs are loudness-compressed, so globally
    # normalized bands sit pegged near one value and the visual barely moves.
    # Rescale each band by its own p95 over the window (bands that are
    # essentially silent keep the global values - no noise amplification).
    if spectrum.shape[0] > 4:
        band_p95 = np.percentile(spectrum, 95, axis=0)
        live = band_p95 > 0.10
        gain = np.where(live, np.minimum(0.92 / np.maximum(band_p95, 1e-6), 3.0), 1.0)
        stretched = np.clip(spectrum * gain, 0.0, 1.0)
        spectrum = np.where(live, stretched, spectrum)
    rms_peak = float(np.percentile(rms, 99.5)) if rms.size else 1.0
    if rms_peak > 0:
        rms = np.clip(rms / rms_peak, 0.0, 1.0)

    # Beat pulse from RMS jumps AND spectral flux - flux keeps reacting on
    # compressed masters where loudness itself hardly changes.
    pulse = _beat_pulse(rms, fps)
    if spectrum.shape[0] > 4:
        flux = np.maximum(np.diff(spectrum, axis=0, prepend=spectrum[:1]), 0.0).mean(axis=1)
        fpeak = np.percentile(flux, 99.5)
        if fpeak > 0:
            flux = np.clip(flux / fpeak, 0.0, 1.0)
        pulse = np.maximum(pulse, _beat_pulse(flux, fps, thresh_k=0.5))
    return {
        "fps": fps,
        "bands": bands,
        "duration": round(duration, 3),
        "rms": [round(float(v), 4) for v in rms],
        "pulse": [round(float(v), 4) for v in pulse],
        "spectrum": [[round(float(v), 4) for v in row] for row in spectrum],
    }


def _beat_pulse(rms: np.ndarray, fps: int, thresh_k: float = 1.0) -> np.ndarray:
    """Crude onset pulse: decaying spikes where the signal jumps
    (kick/onset proxy; also used on spectral flux)."""
    pulse = np.zeros_like(rms)
    if len(rms) < 3:
        return pulse
    smooth = np.convolve(rms, np.ones(3) / 3, mode="same")
    onset = np.maximum(np.diff(smooth, prepend=smooth[0]), 0.0)
    thresh = onset.mean() + thresh_k * onset.std()
    last = -10**9
    min_gap = int(0.24 * fps)
    for i in range(len(rms)):
        if onset[i] > thresh and i - last >= min_gap:
            pulse[i] = min(1.0, onset[i] / (thresh + 1e-6) * 0.7)
            last = i
    # exponential decay after each spike
    decay = np.exp(-1.0 / (0.18 * fps))
    for i in range(1, len(pulse)):
        pulse[i] = max(pulse[i], pulse[i - 1] * decay)
    return np.clip(pulse, 0.0, 1.0)


def chroma_features(samples: np.ndarray, sr: int = SAMPLE_RATE, fps: int = 2,
                    win: int = 8192) -> np.ndarray:
    """12-bin pitch-class energy per frame, rows sum to 1: [frames, 12].
    Used for audio-only repetition (chorus) detection; harmony repeats
    where timbre alone may not."""
    hop = max(1, int(round(sr / fps)))
    n = max(1, int(len(samples) / hop))
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    sel = (freqs >= 65.0) & (freqs <= 2000.0)
    pc = (np.round(12.0 * np.log2(freqs[sel] / 440.0) + 69.0).astype(int)) % 12
    out = np.zeros((n, 12), dtype=np.float32)
    for f in range(n):
        c = f * hop
        seg = samples[max(0, c - win // 2): c + win // 2]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        mag = np.abs(np.fft.rfft(seg * window))[sel]
        out[f] = np.bincount(pc, weights=mag, minlength=12)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-9)