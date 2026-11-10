"""Node 6, voice_music: speak the script, then cut the picture to the speech.

Every script line is spoken on its own (the speech service picks the voice:
Gemini, a voice on this machine, or a silent track timed to the words), so
each line's real length is known. The whole script is voiced by ONE provider: if
the first one fails partway, every line is spoken again by the next, and the
notes say why. The storyboard was planned from word counts;
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

THE HOOK. The opening shot lasts at least `min_hook_s` (pacing.yaml). The hook line is
short, so its shot would end as the next line begins; the shot is kept to the minimum
and the time is taken from the shots after it (never below `min_other_shot_s`), so the
total does not change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.config import load_yaml_config
from src.data.video_schemas import LineTiming, MusicInfo, Storyboard, VideoBrief, VideoScript
from src.models.music import MusicService, get_music_service
from src.models.speech import SpeechService, get_speech_service
from src.models.speech.pronunciation import merge, respell

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
    music_info: MusicInfo
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
        # The spare time goes to the photograph shots (the silence before the first word and between the
        # lines is where a shot lengthens), not to the closing card: the card takes only what keeps it
        # within `max_end_card_s`.
        spare = duration_s - natural
        card_room = max(0.0, v.get("max_end_card_s", 3.0) - pre - d_cta - tail)
        tail_share = min(spare / (n + 1), card_room)
        share = (spare - tail_share) / n                                     # the lead and the n - 1 gaps
        lead, gap, tail = lead + share, gap + share, tail + tail_share
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
    photo_seconds = _split(storyboard, starts, card_start, pre)
    floor_note = _lift_to_floor(photo_seconds, v.get("min_other_shot_s", 1.0))
    if floor_note:
        notes.append(floor_note)
    hook_note = _give_the_hook_its_minimum(photo_seconds, config.get("min_hook_s", 0.0), v.get("min_other_shot_s", 1.0))
    if hook_note:
        notes.append(hook_note)
    seconds = photo_seconds + [total - card_start]
    return Timing(starts, [round(s, 3) for s in seconds], round(total, 3), notes, defects)


def _lift_to_floor(seconds: list[float], floor: float) -> str:
    """Raise every shot under `floor` to it (in place), taking the time from the shots that have more, in
    proportion to what each has above it. The sum is unchanged. A short line that carries a shot of its own,
    or a line shared by several shots, can leave one a fraction of a second long. A note if it was done, or
    if the shots had no time to give."""
    short = [i for i, s in enumerate(seconds) if s < floor - EPSILON]
    if not short:
        return ""
    need = sum(floor - seconds[i] for i in short)
    donors = {i: seconds[i] - floor for i, s in enumerate(seconds) if s > floor + EPSILON}
    spare = sum(donors.values())
    take = min(need, spare)
    if take <= EPSILON:
        return f"{len(short)} shot(s) are under {floor:g} s and the others have no time to give."
    lack = {i: floor - seconds[i] for i in short}
    for i, extra in donors.items():
        seconds[i] -= take * extra / spare
    for i, wanted in lack.items():
        seconds[i] += take * wanted / need
    left = " The others had no more to give." if take < need - EPSILON else ""
    return f"{len(short)} shot(s) were under {floor:g} s and were lengthened to it, the time taken from longer shots.{left}"


def _give_the_hook_its_minimum(seconds: list[float], minimum: float, floor: float) -> str:
    """Lengthen the first shot to `minimum` (in place), taking the time from the shots after it
    in proportion to what each has above `floor`. The sum is unchanged. A note if it fell short
    or if it was done."""
    if len(seconds) < 2 or seconds[0] >= minimum - EPSILON:
        return ""
    need = minimum - seconds[0]
    spare = [max(0.0, s - floor) for s in seconds[1:]]
    take = min(need, sum(spare))
    if take <= EPSILON:
        return f"The hook shot is {seconds[0]:.1f} s, under its minimum of {minimum:g} s: the other shots have no time to give."
    for i, extra in enumerate(spare, start=1):
        seconds[i] -= take * extra / sum(spare)
    seconds[0] += take
    if need - take > EPSILON:
        return (f"The hook shot is {seconds[0]:.1f} s, under its minimum of {minimum:g} s: the other shots "
                "had no more time to give.")
    return f"The hook shot was kept to {minimum:g} s; the time came from the shots after it."


def _voice_name(audios: list) -> str:
    spoken = [a for a in audios if a.mode == "speech"]
    if not spoken:
        return "silent"
    first = spoken[0]
    name = f"{first.provider}/{first.model_id}" + (f" ({first.voice})" if first.voice else "")
    silent = len(audios) - len(spoken)
    return name + (f"; {silent} line(s) silent" if silent else "")


def pronunciation_map(brief: VideoBrief) -> dict[str, str]:
    """The words speech.yaml (and the property's own video_input.json) respells for the voice."""
    return merge(load_yaml_config("video/speech.yaml").get("pronunciations"), brief.pronunciations)


def spoken_text(text: str, brief: VideoBrief) -> str:
    """What a voice with no phonetic spelling of its own is given for a line: the text, with the
    words of the map respelled. The line itself, as printed, is untouched."""
    return respell(text, pronunciation_map(brief))


def run(script: VideoScript, storyboard: Storyboard, brief: VideoBrief, out_dir: Path,
        service: Optional[SpeechService] = None, music_service: Optional[MusicService] = None) -> VoiceResult:
    service = service or get_speech_service()
    audio_dir = out_dir / "audio"
    lines = [line.text for line in script.lines]
    audios = service.synthesize_all(lines, brief.voice, [audio_dir / f"line_{i:02d}.wav" for i in range(len(lines))],
                                    respellings=pronunciation_map(brief), local=list(brief.pronunciations))
    spoken = [a.spoken or text for a, text in zip(audios, lines)]
    timing = retime([a.duration_s for a in audios], storyboard, brief.duration_s)

    board = storyboard.model_copy(deep=True)
    for shot, seconds in zip(board.shots, timing.shot_seconds):
        shot.seconds = seconds
    timings = [LineTiming(line_index=i, start_s=round(start, 3), end_s=round(start + a.duration_s, 3),
                          path=str(a.path), provider=a.provider, voice=a.voice, mode=a.mode, spoken=text,
                          cost_usd=0.0 if a.from_cache else a.cost_usd)
               for i, (a, start, text) in enumerate(zip(audios, timing.starts, spoken))]

    notes = list(dict.fromkeys(n for a in audios for n in a.notes))
    notes += timing.notes
    if audios and all(a.mode == "silent" for a in audios):
        notes.append("No voice was available, so the video has no narration (the captions file has the lines).")
    music = (music_service or get_music_service()).make(
        brief.music_mood, brief.tone, brief.objective, brief.creative_direction, timing.total_s,
        audio_dir / "music.wav")
    notes += [f"Music ({music.info.provider}): {n}" for n in music.info.notes]
    return VoiceResult(board, timings, [str(a.path) for a in audios], _voice_name(audios),
                       str(music.path) if music.path else None, music.info, [], timing.total_s, notes,
                       timing.defects)
