"""Tests for the hook-only transcription path (audio hook detection, snap,
windowed-transcript bookkeeping). Pure functions; no audio or Whisper."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from muse_mp3vid import hook, transcribe  # noqa: E402


def _song(order, sec_frames=20, dims=16, seed=0):
    rng = np.random.default_rng(seed)
    proto = {k: rng.standard_normal(dims) for k in set(order)}
    rows = []
    for k in order:
        rows.append(proto[k] + 0.05 * rng.standard_normal((sec_frames, dims)))
    return np.vstack(rows)


class AudioHookTests(unittest.TestCase):
    def test_picks_repeated_loud_section(self):
        order = "ABCBCDE"          # 10 s sections at fps=2 -> 20 frames each
        feats = _song(order)
        loud = {"C"}
        rms = []
        for i, k in enumerate(order):
            for j in range(40):    # 0.25 s hop, 10 s per section
                rms.append({"t": i * 10 + j * 0.25, "rms": 0.8 if k in loud else 0.2})
        cands = hook.detect_hooks_audio(feats, rms, duration=70.0, fps=2, hook_len=10.0, top_n=3)
        best = cands[0]
        self.assertIn(best["start"], (20.0, 40.0))
        for c in cands:
            self.assertAlmostEqual(c["end"] - c["start"], 10.0, places=3)

    def test_never_returns_short_tail_window(self):
        feats = _song("ABAB")
        rms = [{"t": i * 0.25, "rms": 0.5} for i in range(160)]
        for c in hook.detect_hooks_audio(feats, rms, duration=40.0, fps=2, hook_len=10.0, top_n=5):
            self.assertAlmostEqual(c["end"] - c["start"], 10.0, places=3)

    def test_song_shorter_than_hook(self):
        feats = _song("A", sec_frames=6)
        c = hook.detect_hooks_audio(feats, [], duration=3.0, fps=2, hook_len=30.0)
        self.assertEqual((c[0]["start"], c[0]["end"]), (0.0, 3.0))


class SnapAndWindowTests(unittest.TestCase):
    def test_snap_to_nearest_line_start(self):
        segs = [{"start": 18.0}, {"start": 21.2}, {"start": 30.0}]
        self.assertEqual(hook.snap_start(segs, 20.0, 10.0, 100.0), 21.2)

    def test_snap_noop_when_far_or_overruns_song(self):
        self.assertEqual(hook.snap_start([{"start": 50.0}], 20.0, 10.0, 100.0), 20.0)
        self.assertEqual(hook.snap_start([{"start": 95.0}], 94.0, 10.0, 100.0), 94.0)

    def test_windowed_transcript_tracks_refined_windows(self):
        tr = {"windowed": True, "refined_windows": [[17.0, 33.0]], "segments": []}
        self.assertFalse(transcribe.window_needs_words(tr, 20.0, 30.0))
        self.assertTrue(transcribe.window_needs_words(tr, 40.0, 50.0))

    def test_legacy_transcript_checks_segments(self):
        tr = {"segments": [{"start": 0, "end": 10, "words": []}]}
        self.assertTrue(transcribe.window_needs_words(tr, 2.0, 8.0))
        tr = {"segments": [{"start": 0, "end": 10, "words": [{"w": 1}]}]}
        self.assertFalse(transcribe.window_needs_words(tr, 2.0, 8.0))


if __name__ == "__main__":
    unittest.main()