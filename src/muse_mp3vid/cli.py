"""museMP3Vid CLI: the full pipeline in one command.

  muse-mp3vid SONG.mp3 [--hook-len 30] [--platforms tiktok,reels,shorts]

Stages: transcribe -> analyze -> hook detect -> captions -> visualizer
data -> render -> export. Every stage writes its artifact to the output
directory, and stages can be re-run individually via the flags.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import analyze, captions as captions_mod, hook as hook_mod, render as render_mod, theme as theme_mod
from .transcribe import flatten_words, load_transcript, save_transcript, transcribe_audio

REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def run(args: argparse.Namespace) -> int:
    song = Path(args.song).resolve()
    if not song.exists():
        print(f"song not found: {song}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).resolve() if args.out else Path.cwd() / "out" / song.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = song.stem.replace(" ", "_")

    # ---- Stage 1: Whisper transcription
    transcript_path = out_dir / "transcript.json"
    if args.skip_transcribe and transcript_path.exists():
        transcript = load_transcript(transcript_path)
        _log("transcribe", f"reused {transcript_path}")
    else:
        _log("transcribe", f"Whisper '{args.model}' on {song.name} ...")
        transcript = transcribe_audio(
            song,
            model_size=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
        )
        save_transcript(transcript, transcript_path)
    duration = transcript["duration"] or analyze.probe_duration(str(song))
    _log("transcribe", f"language={transcript['language']} duration={duration:.1f}s "
                        f"segments={len(transcript['segments'])}")

    # ---- Stage 2: audio analysis (energy for hook scoring)
    _log("analyze", "decoding + RMS energy ...")
    samples = analyze.decode_audio(str(song))
    rms = analyze.rms_series(samples)

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    bad = [p for p in platforms if p not in render_mod.PLATFORM_PRESETS]
    if bad:
        print(f"unknown platforms: {bad}", file=sys.stderr)
        return 2

    # ---- Stage 2b: inputs for the visual theme (built after the hook is
    # chosen, since the effect schedule follows the hook's sections)
    coverage = 0.0
    if duration > 0:
        covered = sum(max(0.0, min(s["end"], duration) - s["start"])
                      for s in transcript["segments"])
        coverage = min(1.0, covered / duration)
    if args.no_cover:
        cover_path = None
    elif args.cover:
        cover_path = Path(args.cover)
    else:
        cover_path = theme_mod.find_cover(song)
    coarse = analyze.frame_features(samples, fps=2)

    # ---- Stage 3: best-hook detection
    if args.hook_start is not None:
        start = max(0.0, args.hook_start)
        chosen = {
            "start": round(start, 3),
            "end": round(min(duration, start + args.hook_len), 3),
            "score": None,
            "reasons": {"manual": "start overridden by --hook-start"},
        }
        candidates = [chosen]
    else:
        candidates = hook_mod.detect_hooks(transcript, rms, hook_len=args.hook_len, top_n=3)
        if not candidates:
            print("no hook candidates (empty audio?)", file=sys.stderr)
            return 1
        chosen = candidates[0]
    (out_dir / "hook.json").write_text(
        json.dumps({"chosen": chosen, "candidates": candidates}, indent=2), encoding="utf-8"
    )
    _log("hook", f"chosen {chosen['start']:.1f}s -> {chosen['end']:.1f}s "
                 f"score={chosen['score']} reasons={chosen['reasons']}")
    for alt in candidates[1:]:
        _log("hook", f"  alt {alt['start']:.1f}s -> {alt['end']:.1f}s score={alt['score']}")

    # ---- Stage 3b: per-song visual theme + effect schedule for this hook
    theme = theme_mod.build_theme(
        coarse, rms, coverage,
        cover_path=cover_path, viz=args.viz,
        hook={"start": chosen["start"], "end": chosen["end"]},
        segments=transcript["segments"],
    )
    if cover_path is not None and cover_path.exists():
        dest = out_dir / cover_path.name
        if cover_path.resolve() != dest.resolve():
            shutil.copy2(cover_path, dest)
        theme["cover"] = cover_path.name
    else:
        theme["cover"] = None
    (out_dir / "theme.json").write_text(json.dumps(theme, indent=2), encoding="utf-8")
    _log("theme", f"family={theme['family']} ({theme['mode_reason']}), "
                  f"palette from {theme['palette_source']}, spin={theme['spin']}, "
                  f"cover={theme['cover']}")
    for entry in theme["schedule"]:
        _log("theme", f"  {entry['start']:5.1f}s-{entry['end']:5.1f}s  {entry['mode_name']}")

    # ---- Stage 4: captions
    words = flatten_words(transcript)
    cues = captions_mod.build_cues(words, chosen["start"], chosen["end"],
                                   max_words=args.caption_words)
    if not cues:
        _log("captions", "WARNING: no lyric words inside the chosen hook - "
                         "the videos will have no captions (instrumental "
                         "stretch, or the transcription heard no singing)")
    (out_dir / "captions.srt").write_text(captions_mod.cues_to_srt(cues), encoding="utf-8")
    (out_dir / "captions.ass").write_text(captions_mod.cues_to_ass(cues), encoding="utf-8")
    for platform in platforms:
        margin = render_mod.PLATFORM_PRESETS[platform]["margin_v"]
        (out_dir / f"captions_{platform}.ass").write_text(
            captions_mod.cues_to_ass(cues, margin_v=margin), encoding="utf-8")
    (out_dir / "captions.json").write_text(json.dumps(cues, indent=2), encoding="utf-8")
    _log("captions", f"{len(cues)} cues -> captions.srt / captions.ass "
                     f"(+ per-platform safe zones) / captions.json")

    # ---- Stage 5: visualizer frame data for the hook window
    frames = analyze.frame_features(samples, fps=args.fps,
                                    start=chosen["start"], end=chosen["end"])
    (out_dir / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
    _log("visualizer", f"{len(frames['spectrum'])} frames @ {args.fps}fps -> frames.json")

    # ---- hook audio + job dir (visualizer assets live alongside the data)
    render_mod.extract_hook_audio(str(song), chosen["start"], chosen["end"],
                                  out_dir / "hook_audio.m4a")
    render_mod.stage_visualizer_assets(out_dir, REPO_ROOT)

    # browser preview bundle: open out/preview.html?mode=live&capdata=captions.json
    preview = out_dir / "preview.html"
    if (out_dir / "visualizer.html").exists():
        preview.write_text(
            (out_dir / "visualizer.html").read_text(encoding="utf-8"), encoding="utf-8"
        )

    if args.no_render:
        _log("render", "skipped (--no-render). Artifacts ready in " + str(out_dir))
        return 0

    # ---- Stages 6-7: render + export
    _log("render", f"rendering visualizer + exporting {platforms} ...")
    results = render_mod.export_shorts(
        stem, out_dir, out_dir,
        duration=chosen["end"] - chosen["start"],
        platforms=platforms, fps=args.fps, frames_mode=not args.record,
        workers=args.render_workers, direct=not args.no_direct,
    )
    for path in results:
        _log("export", str(path))
    _log("done", f"hook {chosen['start']:.1f}s-{chosen['end']:.1f}s of '{song.name}' -> {len(results)} short(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="muse-mp3vid", description=__doc__)
    ap.add_argument("song", help="input song (mp3/wav/flac/m4a...)")
    ap.add_argument("--out", help="output directory (default: ./out/<song name>)")
    ap.add_argument("--model", default="small", help="Whisper model size (default: small)")
    ap.add_argument("--language", default=None, help="force language (default: auto-detect)")
    ap.add_argument("--device", default="cpu", help="faster-whisper device (cpu/cuda)")
    ap.add_argument("--compute-type", default="int8", help="faster-whisper compute type")
    ap.add_argument("--skip-transcribe", action="store_true",
                    help="reuse transcript.json from the output directory")
    ap.add_argument("--hook-len", type=float, default=30.0, help="hook length in seconds (default: 30)")
    ap.add_argument("--hook-start", type=float, default=None,
                    help="override hook start (seconds); skips detection")
    ap.add_argument("--caption-words", type=int, default=4, help="max words per caption cue")
    ap.add_argument("--cover", default=None,
                    help="cover art image (default: auto-detect in the song's folder)")
    ap.add_argument("--no-cover", action="store_true", help="do not use cover art")
    ap.add_argument("--viz", default="auto",
                    choices=["auto"] + sorted(theme_mod.MODES),
                    help="force one visualizer effect (default: auto = a per-section "
                         "effect schedule derived from the song)")
    ap.add_argument("--fps", type=int, default=30, help="visualizer frames per second")
    ap.add_argument("--platforms", default="tiktok,reels,shorts",
                    help="comma list: tiktok,reels,shorts")
    ap.add_argument("--record", action="store_true",
                    help="render via real-time MediaRecorder capture (needs a GPU-fast "
                         "machine; default is exact per-frame screenshots)")
    ap.add_argument("--render-workers", type=int, default=3,
                    help="parallel browsers capturing frames (default: 3)")
    ap.add_argument("--no-direct", action="store_true",
                    help="skip the WebCodecs direct-to-video encode and use "
                         "per-frame screenshot capture instead")
    ap.add_argument("--no-render", action="store_true",
                    help="stop after captions + visualizer data (no video)")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
