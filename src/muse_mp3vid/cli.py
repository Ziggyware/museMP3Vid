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
import time
from pathlib import Path

import numpy as np

from . import analyze, captions as captions_mod, hook as hook_mod, render as render_mod, theme as theme_mod
from .transcribe import (flatten_words, load_transcript, refine_window, save_transcript,
                         song_fingerprint, transcribe_audio, window_needs_words)

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

    if getattr(args, "ui", False):
        from . import ui as ui_mod
        return ui_mod.serve(song, out_dir, port=args.ui_port, model=args.model,
                            device=args.device, compute_type=args.compute_type,
                            language=args.language, fast=args.fast,
                            fps=args.fps, direct=not args.no_direct)

    lap = time.perf_counter()
    t_start = lap

    def _lap(label: str) -> None:
        nonlocal lap
        now = time.perf_counter()
        _log("time", f"{label}: {now - lap:.1f}s")
        lap = now

    # ---- Stage 1: decode + RMS energy (cheap; everything else needs it)
    _log("analyze", "decoding + RMS energy ...")
    samples = analyze.decode_audio(str(song))
    rms = analyze.rms_series(samples)
    song_duration = len(samples) / analyze.SAMPLE_RATE
    _lap("decode + rms")

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    bad = [p for p in platforms if p not in render_mod.PLATFORM_PRESETS]
    if bad:
        print(f"unknown platforms: {bad}", file=sys.stderr)
        return 2

    # ---- Stage 2: transcription - exactly once per song, ever. If
    # transcript.json already holds this song's transcription (matched by
    # file size + mtime fingerprint), it is reused and Whisper never runs
    # again; --retranscribe is the explicit escape hatch. Default mode is
    # hook-only: the one transcription is the hook-window pass in stage 3a.
    # --full-transcribe makes the one transcription a whole-song pass
    # (with word timestamps, so no window pass follows it).
    transcript_path = out_dir / "transcript.json"
    fingerprint = song_fingerprint(song)
    transcript = None
    transcribed_this_run = False
    reused_transcript = False
    if transcript_path.exists() and not args.retranscribe:
        stored = load_transcript(transcript_path)
        fp = stored.get("source_fingerprint")
        if fp is None and abs(float(stored.get("duration") or -1) - song_duration) <= 0.6:
            fp = fingerprint  # legacy transcript of this song: adopt + stamp
        if fp == fingerprint or args.skip_transcribe:
            transcript = stored
            reused_transcript = True
            _log("transcribe", "transcript.json already holds this song's "
                               "transcription - reusing it, Whisper will not "
                               "run again (--retranscribe to redo it)")
    if transcript is None:
        if args.full_transcribe:
            _log("transcribe", f"Whisper '{args.model}' on {song.name} (whole song) ...")
            transcript = transcribe_audio(
                song,
                model_size=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=5 if args.thorough else args.beam,
                word_timestamps=True,
                fast=args.fast,
            )
            transcribed_this_run = True
        else:
            transcript = {
                "language": args.language, "language_probability": None,
                "duration": round(song_duration, 3), "model": args.model,
                "vad_fallback": False, "windowed": True, "refined_windows": [],
                "segments": [],
            }
            _log("transcribe", "hook-only mode: Whisper runs on the chosen hook, not the song")
        transcript["source_fingerprint"] = fingerprint
        if args.full_transcribe:
            save_transcript(transcript, transcript_path)
    # The decoded PCM length is the authority on how long the song is. A
    # transcript's duration comes from container/Whisper metadata and can
    # disagree (MP3 padding estimates, a stale reused transcript); windows
    # clamped to it could run past the real audio and come out short.
    duration = song_duration
    transcript["duration"] = round(song_duration, 3)
    _lap("transcribe (whole song)")

    # ---- Stage 2b: inputs for the visual theme
    if args.no_cover:
        cover_path = None
    elif args.cover:
        cover_path = Path(args.cover)
    else:
        cover_path = theme_mod.find_cover(song)
    coarse = analyze.frame_features(samples, fps=2)

    # ---- Stage 3: best-hook detection
    use_lyrics = bool(transcript["segments"]) and not transcript.get("windowed")
    source = "manual" if args.hook_start is not None else ("lyrics" if use_lyrics else "audio")
    if source == "manual":
        start = max(0.0, args.hook_start)
        chosen = {
            "start": round(start, 3),
            "end": round(min(duration, start + args.hook_len), 3),
            "score": None,
            "reasons": {"manual": "start overridden by --hook-start"},
        }
        candidates = [chosen]
    else:
        if source == "lyrics":
            candidates = hook_mod.detect_hooks(transcript, rms, hook_len=args.hook_len, top_n=3)
        else:
            chroma = analyze.chroma_features(samples, fps=2)
            spec = np.asarray(coarse["spectrum"], dtype=np.float32)
            m = min(len(chroma), len(spec))
            feats = np.hstack([chroma[:m] * 2.3, spec[:m]])  # balance 12 vs 64 dims
            candidates = hook_mod.detect_hooks_audio(
                feats, rms, duration, fps=2, hook_len=args.hook_len, top_n=3)
        if not candidates:
            print("no hook candidates (empty audio?)", file=sys.stderr)
            return 1
        chosen = candidates[0]
    # ---- Minimum video length: no exported short may be under
    # MIN_VIDEO_SECONDS. Short windows (a small --hook-len, a --hook-start
    # near the end of the song) are extended/shifted to the minimum here,
    # before anything transcribes or renders the window. A song shorter
    # than the minimum cannot produce a video at all.
    enforced = hook_mod.enforce_min_window(chosen, duration)
    if enforced is None:
        if not args.no_render:
            print(f"song is {duration:.1f}s - shorter than the "
                  f"{hook_mod.MIN_VIDEO_SECONDS:.0f}s minimum video length; "
                  "no video created", file=sys.stderr)
            return 1
        _log("hook", f"WARNING: song ({duration:.1f}s) is shorter than the "
                     f"{hook_mod.MIN_VIDEO_SECONDS:.0f}s minimum; artifacts only")
    elif enforced is not chosen:
        _log("hook", f"window {chosen['start']:.1f}s-{chosen['end']:.1f}s adjusted to "
                     f"{enforced['start']:.1f}s-{enforced['end']:.1f}s "
                     f"({hook_mod.MIN_VIDEO_SECONDS:.0f}s minimum)")
        chosen = enforced
        candidates[0] = chosen
    _lap("hook detection")

    # ---- Stage 3a: word timings for the hook window - this is the one
    # transcription in hook-only mode, and it may only run if no
    # transcription exists yet (not this run, not a previous one).
    if window_needs_words(transcript, chosen["start"], chosen["end"]):
        if transcribed_this_run or reused_transcript:
            _log("transcribe", "WARNING: this window has no word timings, but "
                               "the song's one transcription already exists - "
                               "captions use the stored transcript as-is "
                               "(--retranscribe to redo transcription)")
        else:
            _log("transcribe", f"Whisper on hook {chosen['start']:.1f}s-{chosen['end']:.1f}s "
                               f"({'fast profile' if args.fast else 'quality profile'}) ...")
            transcript = refine_window(song, transcript, chosen["start"], chosen["end"],
                                       model_size=args.model, device=args.device,
                                       compute_type=args.compute_type, language=args.language,
                                       pad=3.0 if source == "audio" else 1.0,
                                       fast=args.fast)
            save_transcript(transcript, transcript_path)
            _lap("transcribe (hook window)")
    if source == "audio":
        # audio-detected starts land on a 0.5 s grid: pull to a sung-line start
        # (the fit check uses the enforced window length, not the raw
        # --hook-len, so snapping can never re-truncate the window)
        win_len = chosen["end"] - chosen["start"]
        s2 = hook_mod.snap_start(transcript["segments"], chosen["start"],
                                 win_len, duration)
        if s2 != chosen["start"]:
            chosen = {**chosen, "start": s2,
                      "end": round(min(duration, s2 + win_len), 3),
                      "reasons": {**chosen["reasons"], "snapped_from": chosen["start"]}}
            candidates[0] = chosen
    (out_dir / "hook.json").write_text(
        json.dumps({"source": source, "chosen": chosen, "candidates": candidates}, indent=2),
        encoding="utf-8",
    )
    _log("hook", f"[{source}] chosen {chosen['start']:.1f}s -> {chosen['end']:.1f}s "
                 f"score={chosen['score']} reasons={chosen['reasons']}")
    for alt in candidates[1:]:
        _log("hook", f"  alt {alt['start']:.1f}s -> {alt['end']:.1f}s score={alt['score']}")

    # fraction of the (hook window | song) covered by sung words
    lo, hi = (chosen["start"], chosen["end"]) if transcript.get("windowed") else (0.0, duration)
    span = hi - lo
    covered = sum(max(0.0, min(s["end"], hi) - max(s["start"], lo))
                  for s in transcript["segments"])
    coverage = min(1.0, covered / span) if span > 0 else 0.0

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
    # Verify what actually landed on disk: an exported file under the
    # minimum is deleted (never created, as far as the user is concerned)
    # and the run fails; a length that disagrees with the hook window by
    # more than a beat is called out instead of silently shipped.
    expected = chosen["end"] - chosen["start"]
    short = []
    for path in results:
        try:
            actual = analyze.probe_duration(str(path))
        except RuntimeError:
            _log("export", f"{path} (duration unverified)")
            continue
        if actual < hook_mod.MIN_VIDEO_SECONDS - 0.05:
            _log("export", f"ERROR: {path.name} came out {actual:.2f}s - under the "
                           f"{hook_mod.MIN_VIDEO_SECONDS:.0f}s minimum; deleting it")
            path.unlink(missing_ok=True)
            short.append(path)
        elif abs(actual - expected) > 0.75:
            _log("export", f"WARNING: {path.name} is {actual:.2f}s, "
                           f"expected {expected:.2f}s")
        else:
            _log("export", f"{path} ({actual:.2f}s)")
    if short:
        return 1
    _lap("render + export")
    _log("time", f"total: {time.perf_counter() - t_start:.1f}s")
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
    ap.add_argument("--beam", type=int, default=1,
                    help="beam size for the full-song pass (default 1 = greedy; the "
                         "hook window always uses beam 5)")
    ap.add_argument("--thorough", action="store_true",
                    help="beam 5 (instead of greedy) for the whole-song pass")
    ap.add_argument("--full-transcribe", action="store_true",
                    help="transcribe the whole song and score hooks on lyric repetition "
                         "(old behavior; slow). Default: pick the hook from audio "
                         "repetition and transcribe only the hook")
    ap.add_argument("--skip-transcribe", action="store_true",
                    help="reuse transcript.json from the output directory")
    ap.add_argument("--retranscribe", action="store_true",
                    help="redo the song's one transcription even though "
                         "transcript.json already holds it")
    ap.add_argument("--fast", action="store_true",
                    help="fast transcription: greedy decoding (beam 1), no "
                         "temperature-fallback re-decodes, tighter hook "
                         "window pad. Several times faster on CPU; captions "
                         "may mishear a few more words")
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
    ap.add_argument("--ui", action="store_true",
                    help="open the local web UI: pick the hook on a waveform, "
                         "transcribe it, edit lyrics, add image/text overlays, "
                         "then generate")
    ap.add_argument("--ui-port", type=int, default=8765,
                    help="port for --ui (default: 8765)")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())