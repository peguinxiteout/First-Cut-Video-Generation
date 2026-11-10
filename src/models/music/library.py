"""Music from a folder of tracks the team has the right to use.

    data/music/<mood>/<track>/<audio file>
                              licence.txt

`<mood>` is the brief's music_mood ("warm acoustic" is read from a folder
named warm-acoustic, warm_acoustic or Warm Acoustic). A track without a
licence.txt that has something in it is not used: the file is how anyone later
sees where the track came from and what it allows. The licence's first line is
copied to the record.

Of the tracks for the mood, the first (by name) that is long enough for the
video is used; if none is, the longest, and the result says it is short.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.core.exceptions import RenderingError
from src.models.music.base import MusicError, MusicProvider, MusicRequest, MusicTrack
from src.models.speech.base import wav_duration
from src.rendering import ffmpeg


def mood_slug(mood: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (mood or "").lower()).strip("-")


class LibraryMusicProvider(MusicProvider):
    name = "library"
    cacheable = False

    def __init__(self, root: Path, extensions: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".ogg", ".flac")):
        self._root = Path(root)
        self._extensions = tuple(e.lower() for e in extensions)

    def is_available(self) -> bool:
        return self._root.is_dir()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else f"there is no music library at {self._root}"

    def _tracks(self, mood: str) -> tuple[list[tuple[Path, Path]], list[str]]:
        """(audio file, licence file) for each usable track of the mood, and
        what was set aside."""
        slug = mood_slug(mood)
        folders = [d for d in sorted(self._root.iterdir()) if d.is_dir() and mood_slug(d.name) == slug] if slug else []
        found, skipped = [], []
        for folder in folders:
            for track in sorted(d for d in folder.iterdir() if d.is_dir()):
                audio = next((f for f in sorted(track.iterdir()) if f.suffix.lower() in self._extensions), None)
                licence = track / "licence.txt"
                if audio is None:
                    continue
                if not licence.is_file() or not licence.read_text(encoding="utf-8", errors="replace").strip():
                    skipped.append(f"{track.name} has no licence.txt")
                    continue
                found.append((audio, licence))
        return found, skipped

    def generate(self, request: MusicRequest, dest: Path) -> MusicTrack:
        if not self.is_available():
            raise MusicError(self.unavailable_reason())
        found, skipped = self._tracks(request.mood)
        if not found:
            why = f"there is no track for the mood '{request.mood}' in {self._root}"
            raise MusicError(why + (f" ({'; '.join(skipped)})" if skipped else ""))
        measured = [(ffmpeg.probe(audio).duration_s, audio, licence) for audio, licence in found]
        long_enough = [m for m in measured if m[0] >= request.duration_s]
        length, audio, licence = long_enough[0] if long_enough else max(measured, key=lambda m: m[0])
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            ffmpeg.run(["-i", str(audio), "-vn", "-ar", "44100", "-ac", "2", str(dest)])
        except RenderingError as exc:
            raise MusicError(f"the track {audio.name} could not be read: {exc}") from exc
        notes = list(skipped)
        if not long_enough:
            notes.append(f"the longest track for '{request.mood}' is {length:.0f} s, shorter than the video")
        first_line = licence.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
        return MusicTrack(path=dest, provider=self.name, duration_s=wav_duration(dest), title=audio.parent.name,
                          licence=first_line, notes=notes)
