"""Per-song visual identity: theme derivation.

Every song gets its own look, derived deterministically from the song itself:

  palette - dominant colors of the cover art when a cover is found
            (median-cut quantization); otherwise generated from the song's
            spectral balance.
  mode    - one of four shader modes, picked from the audio's character
            (bass weight, onset density, brightness, vocal coverage).
  spin    - motion speed from onset density (tempo proxy).
  cover   - the cover art itself, shown in the visualizer's center disc /
            as a backdrop, so the video is branded per song.

Same song in -> same theme out. Override with --viz / --cover / --no-cover.
"""
from __future__ import annotations

import colorsys
import subprocess
from pathlib import Path

import numpy as np

MODES = {"radial": 0, "tunnel": 1, "ribbon": 2, "kaleido": 3}

_COVER_NAMES = ("cover", "folder", "front", "album", "artwork")
_COVER_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_cover(song_path: str | Path) -> Path | None:
    """Look for cover art in the song's own folder."""
    folder = Path(song_path).parent
    images = [p for p in folder.iterdir()
              if p.is_file() and p.suffix.lower() in _COVER_EXTS]
    if not images:
        return None
    for name in _COVER_NAMES:
        for p in images:
            if p.stem.lower() == name:
                return p
    for p in images:
        if "cover" in p.stem.lower():
            return p
    return images[0]


def palette_from_pixels(pixels: np.ndarray, n: int = 5) -> list[list[float]]:
    """Median-cut quantization of an (h, w, 3) uint8 image -> n RGB colors
    in 0..1, sorted dark -> bright."""
    px = pixels.reshape(-1, 3).astype(np.float32)
    if len(px) > 20000:
        px = px[:: len(px) // 20000 + 1]
    boxes = [px]
    while len(boxes) < n:
        # split the box with the widest channel range at its median
        ranges = [b.max(axis=0) - b.min(axis=0) for b in boxes]
        widest = max(range(len(boxes)), key=lambda i: ranges[i].max())
        box = boxes.pop(widest)
        if len(box) < 2:
            boxes.append(box)
            break
        channel = int(np.argmax(ranges[widest]))
        order = np.argsort(box[:, channel])
        mid = len(box) // 2
        boxes.append(box[order[:mid]])
        boxes.append(box[order[mid:]])
    colors = [b.mean(axis=0) / 255.0 for b in boxes if len(b)]
    colors.sort(key=lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
    return [[round(float(v), 4) for v in c] for c in colors]


def palette_from_image(path: str | Path, n: int = 5) -> list[list[float]]:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", "scale=64:64", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < 64 * 64 * 3:
        raise RuntimeError(f"could not decode cover image: {path}")
    pixels = np.frombuffer(proc.stdout[: 64 * 64 * 3], dtype=np.uint8).reshape(64, 64, 3)
    return palette_from_pixels(pixels, n)


def palette_from_audio(mean_spectrum: np.ndarray) -> list[list[float]]:
    """Fallback palette from spectral balance: bass weight warms the base
    hue, brightness cools the top of the ramp."""
    spec = np.asarray(mean_spectrum, dtype=np.float64)
    total = spec.sum() + 1e-9
    bass = spec[:8].sum() / total
    bright = spec[40:].sum() / total
    base_hue = (0.02 + 0.55 * float(np.clip(bass, 0, 1))) % 1.0   # red..cyan by bass
    top_hue = (base_hue + 0.25 + 0.35 * float(np.clip(bright * 3, 0, 1))) % 1.0
    colors = []
    for i in range(5):
        t = i / 4
        hue = (base_hue + (top_hue - base_hue) * t) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.22 + 0.38 * t, 0.85)
        colors.append([round(r, 4), round(g, 4), round(b, 4)])
    return colors


def audio_stats(coarse_frames: dict, rms: list[dict]) -> dict:
    """Character stats from the full-song coarse spectrum + RMS series."""
    spec = np.asarray(coarse_frames["spectrum"], dtype=np.float64)
    mean_spec = spec.mean(axis=0) if spec.size else np.zeros(coarse_frames["bands"])
    total = mean_spec.sum() + 1e-9
    bass_share = float(mean_spec[:8].sum() / total)
    bright_share = float(mean_spec[40:].sum() / total)
    duration = coarse_frames.get("duration") or (rms[-1]["t"] if rms else 0.0)
    onsets = 0
    if len(rms) > 2:
        vals = np.array([p["rms"] for p in rms])
        diffs = np.maximum(np.diff(vals), 0.0)
        onsets = int((diffs > diffs.mean() + diffs.std()).sum())
    onset_rate = onsets / max(duration / 60.0, 1e-6)  # per minute
    return {
        "bass_share": round(bass_share, 4),
        "bright_share": round(bright_share, 4),
        "onset_rate": round(float(onset_rate), 2),
        "mean_spectrum": mean_spec,
    }


def pick_mode(stats: dict, coverage: float) -> tuple[str, str]:
    if stats["bass_share"] >= 0.42:
        return "tunnel", f"bass-heavy (bass share {stats['bass_share']:.2f})"
    if stats["onset_rate"] >= 90 and stats["bright_share"] >= 0.20:
        return "kaleido", f"dense + bright ({stats['onset_rate']:.0f} onsets/min)"
    if coverage < 0.25:
        return "ribbon", f"sparse vocals (coverage {coverage:.2f})"
    return "radial", "balanced mix"


def build_theme(
    coarse_frames: dict,
    rms: list[dict],
    coverage: float,
    cover_path: str | Path | None = None,
    viz: str = "auto",
) -> dict:
    """Assemble the theme dict consumed by the visualizer (theme.json)."""
    stats = audio_stats(coarse_frames, rms)
    if viz != "auto":
        mode, mode_reason = viz, "manual (--viz)"
    else:
        mode, mode_reason = pick_mode(stats, coverage)
    if cover_path is not None:
        try:
            palette = palette_from_image(cover_path)
            palette_source = f"cover art ({Path(cover_path).name})"
        except RuntimeError:
            palette = palette_from_audio(stats["mean_spectrum"])
            palette_source = "audio (cover unreadable)"
    else:
        palette = palette_from_audio(stats["mean_spectrum"])
        palette_source = "audio (no cover found)"
    spin = round(min(2.2, max(0.6, 0.6 + stats["onset_rate"] / 120.0)), 3)
    return {
        "mode": MODES[mode],
        "mode_name": mode,
        "mode_reason": mode_reason,
        "palette": palette,
        "palette_source": palette_source,
        "spin": spin,
        "glow": 1.05,
        "cover": Path(cover_path).name if cover_path else None,
        "stats": {k: v for k, v in stats.items() if k != "mean_spectrum"},
    }
