"""Unit tests for the pure-logic stages (hook detection, captions, features).
Run:  python -m unittest discover -s tests -v
No Whisper, no ffmpeg, no browser needed.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muse_mp3vid import analyze, captions, hook, render, theme, transcribe  # noqa: E402


def _seg(i, start, end, text):
    return {"id": i, "start": start, "end": end, "text": text, "words": []}


def _chorus_song():
    """Synthetic 60s song: unique verses, a chorus line repeated 3x at 20-35s."""
    segs = [
        _seg(0, 0.0, 5.0, "walking down a lonely road tonight"),
        _seg(1, 5.0, 10.0, "neon signs and empty streets"),
        _seg(2, 10.0, 15.0, "counting all my old mistakes"),
        _seg(3, 15.0, 20.0, "waiting for the morning light"),
        _seg(4, 20.0, 25.0, "we duct tape the problems away"),
        _seg(5, 25.0, 30.0, "we duct tape the problems away"),
        _seg(6, 30.0, 35.0, "we duct tape the problems away"),
        _seg(7, 35.0, 40.0, "save the day save the day"),
        _seg(8, 40.0, 45.0, "a quiet bridge with soft guitars"),
        _seg(9, 45.0, 50.0, "whispered words and falling rain"),
        _seg(10, 50.0, 55.0, "we duct tape the problems away"),
        _seg(11, 55.0, 60.0, "fade out on a single note"),
    ]
    rms = [{"t": t * 0.25, "rms": 0.9 if 20 <= t * 0.25 < 40 else 0.3}
           for t in range(240)]
    return {"duration": 60.0, "segments": segs}, rms


class HookTests(unittest.TestCase):
    def test_repetition_scores(self):
        transcript, _ = _chorus_song()
        rep = hook.line_repetition(transcript["segments"])
        self.assertGreater(rep[4], 0.5)   # chorus line repeats
        self.assertEqual(rep[0], 0.0)     # verse line is unique

    def test_detects_chorus_window(self):
        transcript, rms = _chorus_song()
        cands = hook.detect_hooks(transcript, rms, hook_len=15.0, top_n=3)
        self.assertTrue(cands)
        best = cands[0]
        # best 15s window must overlap the chorus block (20-35s) heavily
        overlap = min(best["end"], 35.0) - max(best["start"], 20.0)
        self.assertGreaterEqual(overlap, 10.0)
        self.assertGreater(best["reasons"]["repetition"], 0.3)

    def test_empty_transcript_falls_back_to_energy(self):
        transcript = {"duration": 40.0, "segments": []}
        rms = [{"t": t * 0.25, "rms": 0.95 if t * 0.25 >= 20 else 0.2} for t in range(160)]
        cands = hook.detect_hooks(transcript, rms, hook_len=10.0, top_n=1)
        self.assertEqual(len(cands), 1)
        self.assertGreaterEqual(cands[0]["start"], 15.0)


class CaptionTests(unittest.TestCase):
    def _words(self):
        texts = ["we", "duct", "tape", "the", "problems", "away", "now"]
        return [{"start": 10.0 + i * 0.3, "end": 10.25 + i * 0.3, "text": t}
                for i, t in enumerate(texts)]

    def test_cues_group_and_rebase(self):
        cues = captions.build_cues(self._words(), 10.0, 12.5, max_words=4)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["start"], 0.0)
        self.assertEqual(cues[0]["text"], "we duct tape the")
        self.assertAlmostEqual(cues[1]["start"], 1.2, places=2)

    def test_srt_format(self):
        cues = captions.build_cues(self._words(), 10.0, 12.5)
        srt = captions.cues_to_srt(cues)
        self.assertIn("00:00:00,000 -->", srt)
        self.assertIn("duct", srt)

    def test_ass_karaoke(self):
        cues = captions.build_cues(self._words(), 10.0, 12.5)
        ass = captions.cues_to_ass(cues)
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("{\\kf", ass)
        self.assertIn("Dialogue:", ass)

    def test_ass_margin_per_platform(self):
        cues = captions.build_cues(self._words(), 10.0, 12.5)
        self.assertIn(",680,1", captions.cues_to_ass(cues, margin_v=680))
        self.assertIn(",470,1", captions.cues_to_ass(cues, margin_v=470))


class TranscribeCacheTests(unittest.TestCase):
    def test_local_dir_is_used_as_is(self):
        self.assertEqual(transcribe.cached_model_path("."), ".")

    def test_uncached_model_returns_none(self):
        self.assertIsNone(transcribe.cached_model_path("definitely-not-a-real-size-xyz"))

    def test_segment_coverage(self):
        segs = [_seg(0, 0.0, 25.0, "a"), _seg(1, 50.0, 75.0, "b")]
        self.assertAlmostEqual(transcribe.segment_coverage(segs, 100.0), 0.5)
        self.assertEqual(transcribe.segment_coverage([], 100.0), 0.0)
        self.assertEqual(transcribe.segment_coverage(segs, 0.0), 0.0)


class SectionBoundaryTests(unittest.TestCase):
    def test_flat_mastered_song_still_gets_sections(self):
        # loudness-compressed song: no energy swings, no lyric segments -
        # the phrase grid must still cut the hook into multiple sections
        rms = [{"t": i * 0.25, "rms": 0.8} for i in range(160)]
        bounds = theme.section_boundaries([], rms, 0.0, 30.0)
        self.assertGreaterEqual(len(bounds), 4)
        self.assertEqual(bounds[0], 0.0)

    def test_energy_jump_becomes_a_boundary(self):
        rms = [{"t": i * 0.25, "rms": 0.2 if i < 24 else 0.9} for i in range(48)]
        bounds = theme.section_boundaries([], rms, 0.0, 12.0)
        self.assertTrue(any(5.0 <= b <= 6.5 for b in bounds), bounds)


class PerBandStretchTests(unittest.TestCase):
    def test_quiet_high_tone_still_moves(self):
        sr = analyze.SAMPLE_RATE
        t = np.arange(sr * 2) / sr
        # loud bass throughout + a quiet high tone that switches on at 1s
        sig = 0.9 * np.sin(2 * np.pi * 110 * t)
        sig[sr:] += 0.12 * np.sin(2 * np.pi * 5000 * t[sr:])
        feats = analyze.frame_features(sig.astype(np.float32), fps=30)
        spec = np.array(feats["spectrum"])
        edges = analyze._band_edges(sr)
        hi_band = int(np.searchsorted(edges, 5000) - 1)
        self.assertGreater(spec[45:, hi_band].max(), 0.7)
        self.assertGreater(max(feats["pulse"][30:]), 0.0)


class ThemeTests(unittest.TestCase):
    def test_palette_from_pixels_two_colors(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[:, :8] = (200, 30, 30)     # red half
        img[:, 8:] = (30, 30, 200)     # blue half
        pal = theme.palette_from_pixels(img, n=4)
        self.assertGreaterEqual(len(pal), 2)
        reds = [c for c in pal if c[0] > 0.6 and c[1] < 0.3]
        blues = [c for c in pal if c[2] > 0.6 and c[0] < 0.3]
        self.assertTrue(reds and blues)

    def test_palette_from_audio_shape(self):
        pal = theme.palette_from_audio(np.random.RandomState(1).rand(64))
        self.assertEqual(len(pal), 5)
        self.assertTrue(all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in pal))

    def test_pick_family_rules(self):
        bass = {"bass_share": 0.6, "bright_share": 0.1, "onset_rate": 40.0}
        self.assertEqual(theme.pick_family(bass, 0.9)[0], "bass")
        dense = {"bass_share": 0.2, "bright_share": 0.3, "onset_rate": 120.0}
        self.assertEqual(theme.pick_family(dense, 0.9)[0], "bright")
        sparse = {"bass_share": 0.2, "bright_share": 0.1, "onset_rate": 40.0}
        self.assertEqual(theme.pick_family(sparse, 0.1)[0], "sparse")
        self.assertEqual(theme.pick_family(sparse, 0.9)[0], "balanced")

    def test_catalog_is_large_and_contiguous(self):
        self.assertGreaterEqual(len(theme.MODE_CATALOG), 30)
        ids = sorted(theme.MODES.values())
        self.assertEqual(ids, list(range(len(ids))))
        names = [m["name"] for m in theme.MODE_CATALOG]
        self.assertEqual(len(names), len(set(names)))

    def _coarse(self, seed=7):
        rng = np.random.RandomState(seed)
        return {"fps": 2, "bands": 64, "duration": 60.0,
                "spectrum": rng.rand(120, 64).tolist(),
                "rms": [0.5] * 120, "pulse": [0.0] * 120}

    def test_build_theme_deterministic(self):
        rms = [{"t": i * 0.25, "rms": 0.5} for i in range(240)]
        t1 = theme.build_theme(self._coarse(), rms, 0.9)
        t2 = theme.build_theme(self._coarse(), rms, 0.9)
        self.assertEqual(t1, t2)
        self.assertIn(t1["mode_name"], theme.MODES)

    def test_schedule_sections_for_long_hook(self):
        rms = [{"t": i * 0.25, "rms": 0.5} for i in range(240)]
        segs = [_seg(i, s, s + 5.0, f"line {i}") for i, s in
                enumerate([0.0, 6.0, 12.0, 18.0, 24.0, 30.0])]
        th = theme.build_theme(self._coarse(), rms, 0.9,
                               hook={"start": 0.0, "end": 30.0}, segments=segs)
        sched = th["schedule"]
        self.assertGreaterEqual(len(sched), 2)
        self.assertEqual(sched[0]["start"], 0.0)
        self.assertAlmostEqual(sched[-1]["end"], 30.0)
        for a, b in zip(sched, sched[1:]):
            self.assertNotEqual(a["mode_name"], b["mode_name"])
            self.assertAlmostEqual(a["end"], b["start"])
        for entry in sched:
            self.assertEqual(len(entry["params"]), 4)

    def test_schedule_single_for_short_hook(self):
        rms = [{"t": i * 0.25, "rms": 0.5} for i in range(240)]
        th = theme.build_theme(self._coarse(), rms, 0.9,
                               hook={"start": 0.0, "end": 5.0}, segments=[])
        self.assertEqual(len(th["schedule"]), 1)


class FeatureTests(unittest.TestCase):
    def test_frame_features_shape(self):
        sr = analyze.SAMPLE_RATE
        t = np.arange(sr * 2) / sr
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        feats = analyze.frame_features(samples, fps=30)
        self.assertEqual(feats["fps"], 30)
        self.assertEqual(len(feats["spectrum"]), 60)
        self.assertEqual(len(feats["spectrum"][0]), 64)
        self.assertTrue(all(0.0 <= v <= 1.0 for row in feats["spectrum"] for v in row))
        self.assertGreater(max(feats["rms"]), 0.5)

    def test_rms_series(self):
        samples = np.ones(analyze.SAMPLE_RATE, dtype=np.float32) * 0.5
        series = analyze.rms_series(samples)
        self.assertTrue(series)
        self.assertAlmostEqual(series[0]["rms"], 0.5, places=3)


class MinWindowTests(unittest.TestCase):
    """The 10s floor: no exported video may be shorter than
    hook.MIN_VIDEO_SECONDS, whatever path chose the window."""

    def test_long_window_untouched(self):
        chosen = {"start": 20.0, "end": 50.0, "score": 0.9, "reasons": {}}
        self.assertIs(hook.enforce_min_window(chosen, 70.0), chosen)

    def test_short_window_extended_forward(self):
        out = hook.enforce_min_window({"start": 10.5, "end": 16.5, "reasons": {}}, 70.0)
        self.assertEqual((out["start"], out["end"]), (10.5, 20.5))
        self.assertIn("min_length", out["reasons"])

    def test_tail_window_shifted_back(self):
        out = hook.enforce_min_window({"start": 65.0, "end": 70.0, "reasons": {}}, 70.0)
        self.assertEqual((out["start"], out["end"]), (60.0, 70.0))

    def test_start_past_the_song_is_pulled_back(self):
        out = hook.enforce_min_window({"start": 999.0, "end": 70.0, "reasons": {}}, 70.0)
        self.assertEqual((out["start"], out["end"]), (60.0, 70.0))

    def test_song_shorter_than_minimum_refused(self):
        self.assertIsNone(hook.enforce_min_window({"start": 0.0, "end": 8.0}, 8.0))

    def test_detect_hooks_small_hook_len_still_finds_candidates(self):
        # regression: detect_hooks used to discard EVERY candidate when
        # hook_len < 10, so the CLI died with "no hook candidates"
        tr, rms = _chorus_song()
        cands = hook.detect_hooks(tr, rms, hook_len=6.0, top_n=3)
        self.assertTrue(cands)
        for c in cands:
            self.assertAlmostEqual(c["end"] - c["start"], 6.0, places=3)


class FastProfileTests(unittest.TestCase):
    """--fast transcription: greedy decoding in the hook-window pass;
    the quality profile must stay untouched."""

    def _run_refine(self, fast):
        calls = []

        class FakeInfo:
            language = "en"
            language_probability = 1.0
            duration = 12.0

        def fake_once(model, samples, language, beam_size, vad_filter,
                      word_timestamps=True):
            calls.append({"beam": beam_size, "vad": vad_filter})
            return ([{"id": 0, "start": 1.0, "end": 2.0, "text": "la",
                      "words": []}], FakeInfo())

        tr = {"windowed": True, "segments": [], "language": "en",
              "vad_fallback": False, "refined_windows": []}
        with mock.patch.object(transcribe, "_load_model", return_value=object()), \
             mock.patch.object(transcribe, "_decode_for_whisper",
                               return_value=np.zeros(16000, dtype=np.float32)), \
             mock.patch.object(transcribe, "_transcribe_once", side_effect=fake_once):
            transcribe.refine_window("x.mp3", tr, 10.0, 20.0, fast=fast)
        return calls

    def test_fast_profile_is_greedy(self):
        calls = self._run_refine(True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["beam"], 1)

    def test_quality_profile_unchanged(self):
        calls = self._run_refine(False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["beam"], 5)


class OnceOnlyTests(unittest.TestCase):
    """Transcription happens exactly once: no VAD retry, no temperature
    ladder, no second pass - even when the first pass hears nothing."""

    class FakeInfo:
        language = "en"
        language_probability = 1.0
        duration = 60.0

    def _patches(self, fake_once):
        return (
            mock.patch.object(transcribe, "_load_model", return_value=object()),
            mock.patch.object(transcribe, "_decode_for_whisper",
                              return_value=np.zeros(16000, dtype=np.float32)),
            mock.patch.object(transcribe, "_transcribe_once", side_effect=fake_once),
        )

    def test_refine_window_decodes_once_even_with_zero_coverage(self):
        calls = []

        def fake_once(model, samples, language, beam_size, vad_filter,
                      word_timestamps=True):
            calls.append(beam_size)
            return ([], self.FakeInfo())  # heard nothing: old code retried

        tr = {"windowed": True, "segments": [], "language": "en",
              "vad_fallback": False, "refined_windows": []}
        p1, p2, p3 = self._patches(fake_once)
        with p1, p2, p3:
            # 10s-40s window: span >= 20s, the old VAD-retry trigger zone
            transcribe.refine_window("x.mp3", tr, 10.0, 40.0)
        self.assertEqual(len(calls), 1)

    def test_transcribe_audio_decodes_once_even_with_zero_coverage(self):
        calls = []

        def fake_once(model, samples, language, beam_size, vad_filter,
                      word_timestamps=True):
            calls.append(beam_size)
            return ([], self.FakeInfo())

        p1, p2, p3 = self._patches(fake_once)
        with p1, p2, p3:
            out = transcribe.transcribe_audio("x.mp3")
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["segments"], [])

    def test_single_decode_pins_temperature_to_zero(self):
        seen = {}

        class FakeModel:
            def transcribe(self, samples, **kwargs):
                seen.update(kwargs)
                return iter([]), OnceOnlyTests.FakeInfo()

        segments, _ = transcribe._transcribe_once(
            FakeModel(), np.zeros(16000, dtype=np.float32), None, 5, False)
        self.assertEqual(segments, [])
        self.assertEqual(seen["temperature"], 0.0)


class LyricEditTests(unittest.TestCase):
    """UI lyric edits: pure data - segments in the window are replaced,
    word timings redistributed, Whisper never involved."""

    def _tr(self):
        return {"segments": [
            {"id": 0, "start": 0.0, "end": 4.0, "text": "before", "words": []},
            {"id": 1, "start": 10.0, "end": 14.0, "text": "old line here",
             "words": []},
            {"id": 2, "start": 30.0, "end": 34.0, "text": "after", "words": []},
        ], "source_fingerprint": {"size": 1, "mtime_ns": 2}}

    def test_edits_replace_only_the_window(self):
        out = transcribe.apply_lyric_edits(
            self._tr(), [{"start": 10.0, "end": 14.0, "text": "brand new words"}],
            10.0, 14.0)
        texts = [s["text"] for s in out["segments"]]
        self.assertEqual(texts, ["before", "brand new words", "after"])
        self.assertEqual(out["source_fingerprint"], {"size": 1, "mtime_ns": 2})

    def test_words_redistributed_over_the_line(self):
        out = transcribe.apply_lyric_edits(
            self._tr(), [{"start": 10.0, "end": 14.0, "text": "aa bbbb cc"}],
            10.0, 14.0)
        words = out["segments"][1]["words"]
        self.assertEqual([w["word"].strip() for w in words], ["aa", "bbbb", "cc"])
        self.assertAlmostEqual(words[0]["start"], 10.0)
        self.assertAlmostEqual(words[-1]["end"], 14.0)
        # the 4-char word gets twice the time of the 2-char words
        durs = [w["end"] - w["start"] for w in words]
        self.assertAlmostEqual(durs[1], 2 * durs[0], places=2)


class OverlayTests(unittest.TestCase):
    def test_validate_drops_bad_and_clamps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            job = Path(td)
            (job / "assets").mkdir()
            (job / "assets" / "ok.png").write_bytes(b"x")
            out = render.validate_overlays([
                {"type": "image", "path": "assets/ok.png", "start": 1, "end": 4,
                 "width_frac": 5.0},
                {"type": "image", "path": "assets/missing.png", "start": 0, "end": 2},
                {"type": "image", "path": "../escape.png", "start": 0, "end": 2},
                {"type": "text", "text": "  hello  ", "start": -1, "end": 2,
                 "size": 9999, "position": "nowhere"},
                {"type": "text", "text": "", "start": 0, "end": 2},
                {"type": "text", "text": "backwards", "start": 3, "end": 2},
            ], job)
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0]["width_frac"], 1.0)
            self.assertEqual(out[1]["text"], "hello")
            self.assertEqual(out[1]["start"], 0.0)
            self.assertEqual(out[1]["size"], 240)
            self.assertEqual(out[1]["position"], "top-center")

    def test_text_ass_events(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ov.ass"
            render.write_text_overlays_ass(
                [{"type": "text", "text": "Hi {there}", "start": 1.0, "end": 3.5,
                  "position": "center", "size": 96, "color": "#FF8000"}], path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("PlayResX: 1080", text)
            self.assertIn("Dialogue: 1,0:00:01.00,0:00:03.50", text)
            self.assertIn("\\an5", text)
            self.assertIn("\\fs96", text)
            self.assertIn("&H0080FF&", text)  # #FF8000 -> BGR
            self.assertIn("Hi \\{there\\}", text)

    def test_compose_burns_image_and_text_overlays(self):
        import shutil as _sh
        import subprocess as _sp
        import tempfile
        if not _sh.which("ffmpeg"):
            self.skipTest("ffmpeg unavailable")
        with tempfile.TemporaryDirectory() as td:
            job = Path(td)
            _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                     "-i", "color=c=0x202028:s=270x480:d=1.2", "-pix_fmt", "yuv420p",
                     str(job / "visual.mp4")], check=True)
            _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                     "-i", "sine=frequency=440:duration=1.2",
                     str(job / "hook.wav")], check=True)
            _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                     "-i", "color=c=red:s=80x80:d=0.1", "-frames:v", "1",
                     str(job / "dot.png")], check=True)
            out = job / "final.mp4"
            render.compose(job / "visual.mp4", job / "hook.wav", None, out,
                           duration=1.2, overlays=[
                               {"type": "image", "path": "dot.png", "start": 0.2,
                                "end": 1.0, "position": "center", "width_frac": 0.3},
                               {"type": "text", "text": "TITLE", "start": 0.0,
                                "end": 1.2, "position": "top-center", "size": 90,
                                "color": "#FFFFFF"},
                           ])
            self.assertTrue(out.exists())
            self.assertAlmostEqual(analyze.probe_duration(str(out)), 1.2, delta=0.25)


if __name__ == "__main__":
    unittest.main()