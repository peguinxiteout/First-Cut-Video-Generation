"""Where the crossfades between shots go, in whole frames.

A crossfade is centred on the cut, so the video is as long as the shots add up
to: the fade starts half its length before the cut and ends half after. That
needs each clip to run a little past its own shot at both ends (the "head"
before its first frame's time, the "tail" after its last). The motion step
makes its clips that much longer; compose cuts and joins them with xfade.

    clip i   head_i = the first half of the fade into it, tail_i = the second
             half of the fade out of it; the first clip has no head and the
             last no tail.

A fade never takes more than half of either shot it joins, so two fades in one
shot cannot meet.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fade:
    """The crossfade into shot i + 1."""
    frames: int          # its length
    lead: int            # frames of it before the cut

    @property
    def tail(self) -> int:
        return self.frames - self.lead


def transition_frames(transition_s: float, fps: int) -> int:
    """The configured fade in frames; 0 for hard cuts."""
    return 0 if transition_s <= 0 else max(2, round(transition_s * fps))


def fades(frames: list[int], fade_frames: int) -> list[Fade]:
    """One fade for each cut between consecutive shots of these lengths."""
    if fade_frames <= 0:
        return []
    out = []
    for before, after in zip(frames, frames[1:]):
        length = max(1, min(fade_frames, before // 2, after // 2))
        out.append(Fade(length, length // 2))
    return out


def margins(frames: list[int], fade_frames: int) -> list[tuple[int, int]]:
    """(head, tail) in frames for every clip."""
    found = fades(frames, fade_frames)
    if not found:
        return [(0, 0)] * len(frames)
    return [(found[i - 1].lead if i > 0 else 0, found[i].tail if i < len(found) else 0)
            for i in range(len(frames))]
