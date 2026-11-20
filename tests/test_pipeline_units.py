"""Unit tests for the pure-logic stages (hook detection, captions, features).
Run:  python -m unittest discover -s tests -v
No Whisper, no ffmpeg, no browser needed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muse_mp3vid import analyze, captions, hook, theme, transcribe  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
