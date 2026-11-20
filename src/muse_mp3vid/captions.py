"""Stage 3: caption generation from word timestamps.

Builds short TikTok-style cues (<= 4 words) and exports:
  - SRT  (plain subtitles)
  - ASS  (karaoke word-sweep, styled for 1080x1920 vertical video)
  - JSON (for the WebGL visualizer's live preview overlay)
All times are rebased to the hook window (0 = hook start).
"""
from __future__ import annotations


def build_cues(
    words: list[dict],
    window_start: float,
    window_end: float,
    max_words: int = 4,
    max_dur: float = 1.6,
    max_gap: float = 0.35,
) -> list[dict]:
    """Group words inside [window_start, window_end] into caption cues,
    rebased so the window starts at 0."""
    inside = [
        w for w in words
        if w["end"] > window_start and w["start"] < window_end
    ]
    cues: list[dict] = []
    current: list[dict] = []

    def flush():
        if not current:
            return
        cues.append(
            {
                "start": round(current[0]["start"] - window_start, 3),
                "end": round(current[-1]["end"] - window_start, 3),
                "text": " ".join(w["text"] for w in current),
                "words": [
                    {
                        "start": round(w["start"] - window_start, 3),
                        "end": round(w["end"] - window_start, 3),
                        "text": w["text"],
                    }
                    for w in current
                ],
            }
        )
        current.clear()

    for w in inside:
        if current:
            gap = w["start"] - current[-1]["end"]
            span = w["end"] - current[0]["start"]
            if len(current) >= max_words or gap > max_gap or span > max_dur:
                flush()
        current.append(w)
    flush()
    # keep cue ends monotonic and inside the window
    limit = round(window_end - window_start, 3)
    for i, cue in enumerate(cues):
        cue["end"] = min(cue["end"], limit)
        if i + 1 < len(cues):
            cue["end"] = min(cue["end"], cues[i + 1]["start"])
        cue["end"] = max(cue["end"], cue["start"] + 0.1)
    return cues


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cues_to_srt(cues: list[dict]) -> str:
    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(f"{i}\n{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n{cue['text']}")
    return "\n\n".join(blocks) + "\n"


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEADER = """[Script Info]
Title: museMP3Vid hook captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,Arial Black,86,&H0000FFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,2,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def cues_to_ass(cues: list[dict], margin_v: int = 560) -> str:
    """ASS with \\kf karaoke sweep: words start white (SecondaryColour) and
    fill to yellow (PrimaryColour) as they are sung. margin_v lifts the
    captions clear of each platform's UI overlay (safe zone)."""
    lines = [ASS_HEADER.replace("{margin_v}", str(margin_v))]
    for cue in cues:
        parts = []
        words = cue["words"]
        for i, w in enumerate(words):
            nxt = words[i + 1]["start"] if i + 1 < len(words) else w["end"]
            dur_cs = max(1, int(round((nxt - w["start"]) * 100)))
            parts.append(f"{{\\kf{dur_cs}}}{w['text']}")
        text = " ".join(parts)
        lines.append(
            f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},Hook,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"
