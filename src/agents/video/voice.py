"""Node 6, voice_music: speak the script, then cut the picture to the speech.

Every script line is spoken on its own (the speech service picks the voice:
Gemini, a voice on this machine, or a silent track timed to the words), so
each line's real length is known. The storyboard was planned from word counts;
here its shots are re-timed to those lengths.

THE TIMELINE. Lines are laid end to end with a breath before the first (the
lead), a pause between them (the gap) and a tail after the last. A shot
starts a moment before the first line it carries (the pre-roll) and runs to
the start of the next shot's line; shots that only continue a line share its
time. The closing card starts just before the call to action is spoken.

FITTING. The total stays at the length asked for (within 0.5 s):

    narration shorter than the video   the spare time is shared out between the
                                       lead, the gaps and the tail
    narration longer                   the lead and the gaps shrink to their
                                       minimums, then the closing card to 1.5 s
    still too long                     the video runs over and a defect says by
                                       how much

Speech is never sped up: a narrator rushed to fit is worse than a long video
that says so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.config import load_yaml_config
from src.data.video_schemas import LineTiming, Storyboard, VideoBrief, VideoScript
from src.models.music import get_music_provider
from src.models.speech import SpeechService, get_speech_service

logger = logging.getLogger(__name__)

EPSILON = 1e-6


@dataclass
class Timing:
    starts: list[float]               # when each line begins, seconds from the start of the video
    shot_seconds: list[float]         # each storyboard shot, the closing card last
    total_s: float
    notes: list[str] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)


@dataclass
class VoiceResult:
    storyboard: Storyboard
    line_timings: list[LineTiming]
    voice_tracks: list[str]
    voice_name: str
    music_track: Optional[str]
    beat_times: list[float]
    total_s: float
    notes: list[str] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)


def _layout(durations: list[float], lead: float, gap: float, tail: float, pre: float) -> tuple[list[float], float, float]:
    """(line starts, where the closing card begins, the total length)."""
    starts, t = [], lead
    for i, length in enumerate(durations):
        starts.append(t)
        t += length + (gap if i < len(durations) - 1 else 0.0)
    return starts, max(0.0, starts[-1] - pre), t + tail


def _split(storyboard: Storyboard, starts: list[float], card_start: float, pre: float) -> list[float]:
    """Seconds for every shot: a shot begins `pre` before the first line it is
    the first to carry; shots that only continue a line share that line's time;
    the closing card takes what is left."""
    photo = storyboard.shots[:-1]
    cta = len(starts) - 1
    owned: set[int] = set()
    segments: list[list] = []                               # [start time, [shot positions]]
    for position, shot in enumerate(photo):
        new = [i for i in shot.line_ids if i not in owned and i != cta and i < len(starts)]
        owned.update(shot.line_ids)
        if not segments:
            segments.append([0.0, [position]])
        elif new:
            segments.append([max(segments[-1][0], starts[min(new)] - pre), [position]])
        else:
            segments[-1][1].append(position)
    seconds = [0.0] * len(photo)
    for index, (start, members) in enumerate(segments):
        end = segments[index + 1][0] if index + 1 < len(segments) else card_start
        span = max(0.1 * len(members), end - start)
        for position in members:
            seconds[position] = span / len(members)
    return seconds


def retime(durations: list[float], storyboard: Storyboard, duration_s: float, config: Optional[dict] = None) -> Timing:
    """Where each line and each shot goes, for these spoken lengths."""
    config = config or load_yaml_config("video/pacing.yaml")
    v, end_card_s = config["voice"], config["end_card_s"]
    if not durations:
        return Timing([], [s.seconds for s in storyboard.shots], float(duration_s))
    n, pre, d_cta = len(durations), v["pre_roll_s"], durations[-1]
    lead, gap = v["lead_s"], v["gap_s"]
    tail = max(v["tail_s"], end_card_s - d_cta - pre)               # the card is at least its planned length
    natural = lead + sum(durations) + (n - 1) * gap + tail
    notes: list[str] = []
    defects: list[str] = []

    if natural <= duration_s + EPSILON:
        share = (duration_s - natural) / (n + 1)
        lead, gap, tail = lead + share, gap + share, tail + share
        if share > 0.6:
            notes.append(f"The narration is {natural:.1f} s for a {duration_s:g} s video: pauses are "
                         f"{gap:.1f} s long. A longer script would fill the time better.")
    else:
        over = natural - duration_s
        room = (lead - v["min_lead_s"]) + (n - 1) * (gap - v["min_gap_s"])
        if room > 0:
            take = min(over, room)
            f = take / room
            lead -= (lead - v["min_lead_s"]) * f
            gap -= (gap - v["min_gap_s"]) * f
            over -= take
        if over > EPSILON:
            tail_min = max(0.1, v["min_end_card_s"] - d_cta - pre)
            take = min(over, max(0.0, tail - tail_min))
            tail -= take
            over -= take
        starts, card_start, total = _layout(durations, lead, gap, tail, pre)
        if total < natural - EPSILON:
            notes.append(f"The narration is long for {duration_s:g} s: pauses are as short as "
                         f"{max(gap, 0):.2f} s and the closing card is {total - card_start:.1f} s.")
        if over > v["fit_tolerance_s"]:
            defects.append(f"The narration needs {total:.1f} s but the video is {duration_s:g} s: it runs "
                           f"{over:.1f} s over. The speech was not sped up; shorten the script or ask for "
                           "a longer video.")
        elif over > EPSILON:
            notes.append(f"The narration runs {over:.1f} s over the {duration_s:g} s asked for.")

    starts, card_start, total = _layout(durations, lead, gap, tail, pre)
    seconds = _split(storyboard, starts, card_start, pre) + [total - card_start]
    return Timing(starts, [round(s, 3) for s in seconds], round(total, 3), notes, defects)


def _voice_name(audios: list) -> str:
    spoken = [a for a in audios if a.mode == "speech"]
    if not spoken:
        return "silent"
    first = spoken[0]
    name = f"{first.provider}/{first.model_id}" + (f" ({first.voice})" if first.voice else "")
    silent = len(audios) - len(spoken)
    return name + (f"; {silent} line(s) silent" if silent else "")


def run(script: VideoScript, storyboard: Storyboard, brief: VideoBrief, out_dir: Path,
        service: Optional[SpeechService] = None) -> VoiceResult:
    service = service or get_speech_service()
    audio_dir = out_dir / "audio"
    audios = [service.synthesize(line.text, brief.voice, audio_dir / f"line_{i:02d}.wav")
              for i, line in enumerate(script.lines)]
    timing = retime([a.duration_s for a in audios], storyboard, brief.duration_s)

    board = storyboard.model_copy(deep=True)
    for shot, seconds in zip(board.shots, timing.shot_seconds):
        shot.seconds = seconds
    timings = [LineTiming(line_index=i, start_s=round(start, 3), end_s=round(start + a.duration_s, 3),
                          path=str(a.path), provider=a.provider, voice=a.voice, mode=a.mode)
               for i, (a, start) in enumerate(zip(audios, timing.starts))]

    notes = list(dict.fromkeys(n for a in audios for n in a.notes))
    notes += timing.notes
    if audios and all(a.mode == "silent" for a in audios):
        notes.append("No voice was available, so the video has no narration (the captions file has the lines).")
    music = get_music_provider()
    track = music.track(brief.music_mood, timing.total_s, audio_dir / "music.wav")
    return VoiceResult(board, timings, [str(a.path) for a in audios], _voice_name(audios),
                       str(track) if track else None, music.beat_times(track), timing.total_s, notes, timing.defects)
