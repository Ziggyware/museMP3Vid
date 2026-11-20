"""Per-song visual identity: theme derivation.

Every song gets its own look, derived deterministically from the song itself:

  palette  - dominant colors of the cover art when a cover is found
             (median-cut quantization); otherwise generated from the song's
             spectral balance.
  schedule - the hook is cut into musical sections (lyric segment starts +
             energy change points), and each section gets its own effect
             from a catalog of 32 shader modes. The family of effects is
             chosen from the song's character (bass weight, onset density,
             brightness, vocal coverage, spectral flux); the specific picks
             are seeded by the song's stats, so two different songs land on
             different effect sequences while the same song always lands
             on the same one.
  params   - every effect is parameterized from the music (ray counts,
             fold counts, flow speeds, zoom...) - no two renders share
             motion parameters either.
  cover    - the cover art itself, shown inside the visualizer.

Override with --viz <mode> (single effect), --cover / --no-cover.
"""
from __future__ import annotations

import colorsys
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Effect catalog: 32 shader modes. `family` pools drive auto-selection.
# ---------------------------------------------------------------------------
MODE_CATALOG: list[dict] = [
    {"name": "radial",    "family": "balanced"},
    {"name": "tunnel",    "family": "bass"},
    {"name": "ribbon",    "family": "sparse"},
    {"name": "kaleido",   "family": "bright"},
    {"name": "sunburst",  "family": "balanced"},
    {"name": "vinyl",     "family": "bass"},
    {"name": "orbital",   "family": "balanced"},
    {"name": "mandala",   "family": "balanced"},
    {"name": "scope",     "family": "sparse"},
    {"name": "waterfall", "family": "balanced"},
    {"name": "terrain",   "family": "balanced"},
    {"name": "eqbars",    "family": "balanced"},
    {"name": "ledwall",   "family": "balanced"},
    {"name": "plasma",    "family": "balanced"},
    {"name": "voronoi",   "family": "balanced"},
    {"name": "moire",     "family": "bright"},
    {"name": "starfield", "family": "bright"},
    {"name": "spirograph", "family": "balanced"},
    {"name": "gridwave",  "family": "balanced"},
    {"name": "skyline",   "family": "balanced"},
    {"name": "rings",     "family": "bass"},
    {"name": "helix",     "family": "balanced"},
    {"name": "iris",      "family": "balanced"},
    {"name": "liquid",    "family": "sparse"},
    {"name": "glitch",    "family": "bright"},
    {"name": "halftone",  "family": "bright"},
    {"name": "contour",   "family": "sparse"},
    {"name": "fire",      "family": "bass"},
    {"name": "rain",      "family": "balanced"},
    {"name": "bubbles",   "family": "sparse"},
    {"name": "laserfan",  "family": "bass"},
    {"name": "gears",     "family": "balanced"},
]
MODES = {m["name"]: i for i, m in enumerate(MODE_CATALOG)}
FAMILIES: dict[str, list[str]] = {}
for _m in MODE_CATALOG:
    FAMILIES.setdefault(_m["family"], []).append(_m["name"])
# every family can also borrow the balanced pool - variety beats purity
FAMILIES["bass"] += ["radial", "mandala", "orbital"]
FAMILIES["bright"] += ["radial", "sunburst", "ledwall"]
FAMILIES["sparse"] += ["radial", "waterfall", "terrain"]

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


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
def palette_from_pixels(pixels: np.ndarray, n: int = 5) -> list[list[float]]:
    """Median-cut quantization of an (h, w, 3) uint8 image -> n RGB colors
    in 0..1, sorted dark -> bright."""
    px = pixels.reshape(-1, 3).astype(np.float32)
    if len(px) > 20000:
        px = px[:: len(px) // 20000 + 1]
    boxes = [px]
    while len(boxes) < n:
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
    base_hue = (0.02 + 0.55 * float(np.clip(bass, 0, 1))) % 1.0
    top_hue = (base_hue + 0.25 + 0.35 * float(np.clip(bright * 3, 0, 1))) % 1.0
    colors = []
    for i in range(5):
        t = i / 4
        hue = (base_hue + (top_hue - base_hue) * t) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.22 + 0.38 * t, 0.85)
        colors.append([round(r, 4), round(g, 4), round(b, 4)])
    return colors


# ---------------------------------------------------------------------------
# Audio character stats
# ---------------------------------------------------------------------------
def audio_stats(coarse_frames: dict, rms: list[dict]) -> dict:
    """Character stats from the full-song coarse spectrum + RMS series."""
    spec = np.asarray(coarse_frames["spectrum"], dtype=np.float64)
    mean_spec = spec.mean(axis=0) if spec.size else np.zeros(coarse_frames["bands"])
    total = mean_spec.sum() + 1e-9
    bass_share = float(mean_spec[:8].sum() / total)
    mid_share = float(mean_spec[8:40].sum() / total)
    bright_share = float(mean_spec[40:].sum() / total)
    flux = float(np.abs(np.diff(spec, axis=0)).mean()) if spec.shape[0] > 1 else 0.0
    duration = coarse_frames.get("duration") or (rms[-1]["t"] if rms else 0.0)
    onsets = 0
    dyn_range = 1.0
    if len(rms) > 2:
        vals = np.array([p["rms"] for p in rms])
        diffs = np.maximum(np.diff(vals), 0.0)
        onsets = int((diffs > diffs.mean() + diffs.std()).sum())
        p50 = float(np.percentile(vals, 50))
        p95 = float(np.percentile(vals, 95))
        dyn_range = p95 / p50 if p50 > 1e-6 else 1.0
    onset_rate = onsets / max(duration / 60.0, 1e-6)
    return {
        "bass_share": round(bass_share, 4),
        "mid_share": round(mid_share, 4),
        "bright_share": round(bright_share, 4),
        "flux": round(flux, 5),
        "dyn_range": round(dyn_range, 3),
        "onset_rate": round(float(onset_rate), 2),
        "mean_spectrum": mean_spec,
    }


def pick_family(stats: dict, coverage: float) -> tuple[str, str]:
    if stats["bass_share"] >= 0.42:
        return "bass", f"bass-heavy (bass share {stats['bass_share']:.2f})"
    if stats["onset_rate"] >= 90 and stats["bright_share"] >= 0.20:
        return "bright", f"dense + bright ({stats['onset_rate']:.0f} onsets/min)"
    if coverage < 0.25:
        return "sparse", f"sparse vocals (coverage {coverage:.2f})"
    return "balanced", "balanced mix"


def _seed(stats: dict) -> int:
    """Stable per-song seed from quantized stats."""
    key = (
        f"{stats['bass_share']:.2f}|{stats['bright_share']:.2f}|"
        f"{stats['onset_rate']:.0f}|{stats['flux']:.3f}|{stats['dyn_range']:.1f}"
    )
    h = 2166136261
    for ch in key:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def mode_params(mode_name: str, stats: dict, spin: float) -> list[float]:
    """Per-effect parameters derived from the music. Meaning of the four
    slots is per-mode (documented in the shader); ranges are safe."""
    bass = stats["bass_share"]
    bright = stats["bright_share"]
    onset = stats["onset_rate"]
    flux = stats["flux"]
    table = {
        "sunburst":  [24 + onset / 6, spin, bass, bright],          # ray count
        "kaleido":   [6 + bright * 24, spin, bass, flux * 8],       # folds 6..12ish
        "mandala":   [5 + bass * 10, spin, onset / 60, bright],     # petals
        "vinyl":     [spin * 0.7, bass, 24 + flux * 60, bright],    # grooves
        "orbital":   [3 + onset / 40, spin, bass, bright],          # rings
        "eqbars":    [32 + bright * 32, spin, bass, dyn(stats)],    # bar count
        "ledwall":   [18 + flux * 40, 26 + flux * 40, bass, bright],
        "voronoi":   [3 + flux * 22, spin, bass, bright],           # cell scale
        "moire":     [30 + bright * 50, spin, bass, flux * 8],
        "starfield": [0.5 + onset / 90, spin, bass, bright],        # warp speed
        "spirograph": [3 + bass * 9, 5 + bright * 11, spin, flux * 8],
        "gridwave":  [spin, bass, bright, 6 + onset / 30],          # mountain rows
        "skyline":   [spin, bass, bright, 14 + flux * 30],
        "rings":     [0.4 + onset / 150, spin, bass, bright],       # emit rate
        "helix":     [2 + bass * 4, spin, bright, flux * 8],        # turns
        "iris":      [40 + bright * 60, spin, bass, flux * 8],      # fibers
        "liquid":    [1.5 + flux * 8, spin * 0.6, bass, bright],    # warp scale
        "glitch":    [8 + flux * 30, spin, bright, bass],           # block rows
        "halftone":  [26 + bright * 30, spin, bass, flux * 8],      # dot grid
        "contour":   [6 + bass * 10, spin * 0.7, bright, flux * 8], # contour count
        "fire":      [0.7 + bass * 1.2, spin, bright, flux * 8],    # intensity
        "rain":      [24 + bright * 40, 0.5 + onset / 120, bass, flux * 8],
        "bubbles":   [10 + bright * 22, spin * 0.7, bass, flux * 8],
        "laserfan":  [5 + onset / 25, spin, bass, bright],          # beams
        "gears":     [6 + bass * 8, spin, bright, flux * 8],        # teeth
        "waterfall": [spin, bass, bright, 40 + flux * 60],          # scroll rows
        "terrain":   [spin, bass, bright, 5 + flux * 20],
        "plasma":    [4 + bass * 4, spin, bright, flux * 8],        # blobs
        "scope":     [spin, bass, bright, 2 + flux * 30],           # traces
        "tunnel":    [spin, bass, bright, 3 + flux * 10],
        "ribbon":    [spin, bass, bright, flux * 8],
        "radial":    [spin, bass, bright, flux * 8],
    }
    raw = table.get(mode_name, [spin, bass, bright, flux * 8])
    return [round(float(v), 4) for v in raw]


def dyn(stats: dict) -> float:
    return stats.get("dyn_range", 1.0)


# ---------------------------------------------------------------------------
# Section schedule: cut the hook into musical sections, one effect each
# ---------------------------------------------------------------------------
def section_boundaries(
    segments: list[dict],
    rms: list[dict],
    hook_start: float,
    hook_end: float,
    min_len: float = 3.5,
    max_sections: int = 5,
) -> list[float]:
    """Section start times relative to the hook start, always beginning at 0.
    Boundaries come from lyric segment starts inside the hook plus strong
    energy change points, thinned to >= min_len apart."""
    length = hook_end - hook_start
    if length < min_len * 2:
        return [0.0]
    points = {0.0}
    for s in segments:
        rel = s["start"] - hook_start
        if 0.5 < rel < length - 0.5:
            points.add(round(rel, 3))
    if len(rms) > 8:
        vals = np.array([p["rms"] for p in rms])
        diffs = np.abs(np.diff(vals))
        thresh = diffs.mean() + 1.5 * diffs.std()
        for i in np.argsort(diffs)[::-1][:8]:
            if diffs[i] > thresh:
                rel = rms[i]["t"] - hook_start
                if 0.5 < rel < length - 0.5:
                    points.add(round(float(rel), 3))
    ordered = sorted(points)
    kept = [ordered[0]]
    for p in ordered[1:]:
        if p - kept[-1] >= min_len:
            kept.append(p)
    if len(kept) > max_sections:
        # keep the strongest spread: first, last-ish, evenly sampled
        idx = np.linspace(0, len(kept) - 1, max_sections).round().astype(int)
        kept = [kept[i] for i in sorted(set(idx))]
    # final section must have room to play
    if len(kept) > 1 and length - kept[-1] < min_len:
        kept.pop()
    return kept


def build_schedule(
    family_pool: list[str],
    seed: int,
    boundaries: list[float],
    hook_len: float,
    stats: dict,
    spin: float,
) -> list[dict]:
    """Assign effects to sections: seeded walk through the family pool,
    never repeating the previous section's effect."""
    pool = list(family_pool)
    rng_state = seed
    schedule = []
    prev = None
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else hook_len
        for _ in range(len(pool)):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            name = pool[rng_state % len(pool)]
            if name != prev or len(pool) == 1:
                break
        prev = name
        schedule.append({
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "mode": MODES[name],
            "mode_name": name,
            "params": mode_params(name, stats, spin),
        })
    return schedule


def build_theme(
    coarse_frames: dict,
    rms: list[dict],
    coverage: float,
    cover_path: str | Path | None = None,
    viz: str = "auto",
    hook: dict | None = None,
    segments: list[dict] | None = None,
) -> dict:
    """Assemble the theme dict consumed by the visualizer (theme.json).

    hook: {"start", "end"} of the chosen hook window - when given (and viz
    is auto), the theme carries a per-section effect schedule; otherwise a
    single-effect theme.
    """
    stats = audio_stats(coarse_frames, rms)
    family, family_reason = pick_family(stats, coverage)
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
    seed = _seed(stats)

    if viz != "auto":
        schedule = [{
            "start": 0.0,
            "end": round(hook["end"] - hook["start"], 3) if hook else 0.0,
            "mode": MODES[viz],
            "mode_name": viz,
            "params": mode_params(viz, stats, spin),
        }]
        family_reason = "manual (--viz)"
    elif hook is not None:
        bounds = section_boundaries(segments or [], rms,
                                    hook["start"], hook["end"])
        schedule = build_schedule(FAMILIES[family], seed, bounds,
                                  hook["end"] - hook["start"], stats, spin)
    else:
        name = FAMILIES[family][seed % len(FAMILIES[family])]
        schedule = [{
            "start": 0.0, "end": 0.0, "mode": MODES[name], "mode_name": name,
            "params": mode_params(name, stats, spin),
        }]

    first = schedule[0]
    return {
        "mode": first["mode"],
        "mode_name": first["mode_name"],
        "mode_reason": family_reason,
        "family": family,
        "schedule": schedule,
        "palette": palette,
        "palette_source": palette_source,
        "spin": spin,
        "glow": 1.05,
        "cover": Path(cover_path).name if cover_path else None,
        "stats": {k: v for k, v in stats.items() if k != "mean_spectrum"},
    }
