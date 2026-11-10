"""Background music service.

    from src.models.music import get_music_service
    result = get_music_service().make(mood, tone, objective, direction, total_s, dest_path)

The chain comes from configs/video/music.yaml (Beatoven, Lyria, then a licensed
library folder, then none). `make` never raises for want of a key or a track: "none"
is always the last link. What comes back says which provider made the track,
the prompt and model, what it cost, and why the providers ahead of it did not
(`info.notes`).

The track is cut to the video's length and fades out at the end. A generated
track is cached on its prompt and model, so the same brief is not paid for
twice; a library track is read from the folder each time. One service is one
RUN: a provider that answers with an auth or quota error is skipped for the
rest of it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Optional, Sequence

from src.core.config import load_yaml_config
from src.core.exceptions import ConfigError, RenderingError
from src.data.video_schemas import MusicInfo
from src.models.music.base import (
    MusicAuthError, MusicError, MusicProvider, MusicQuotaError, MusicRequest, MusicTrack,
)
from src.models.music.beatoven import BeatovenProvider
from src.models.music.library import LibraryMusicProvider
from src.models.music.lyria import LyriaProvider
from src.models.speech.base import wav_duration
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

__all__ = ["MusicResult", "MusicService", "build_prompt", "get_music_service"]

CONFIG_FILE = "video/music.yaml"
NONE = "none"


@dataclass
class MusicResult:
    path: Optional[Path]               # the track as mixed: cut to the video, faded out; None = no music
    info: MusicInfo


def build_prompt(template: str, objectives: dict, mood: str, tone: str, objective: str, direction: str) -> str:
    """The generator's prompt. Always instrumental: the template says so."""
    text = Template(template).safe_substitute(
        mood=(mood or "calm and warm").strip(), tone=(tone or "warm").strip(),
        objective=objectives.get(objective, "to present the place well"), direction=(direction or "").strip())
    return " ".join(text.split())


class MusicService:
    def __init__(self, providers: Sequence[MusicProvider], cache_dir: Path, template: str = "", objectives=None,
                 fade_out_s: float = 1.5, fade_in_s: float = 0.4):
        self.providers = list(providers)
        self.cache_dir = cache_dir
        self.template = template or "Instrumental only, no vocals. $mood."
        self.objectives = objectives or {}
        self.fade_out_s, self.fade_in_s = fade_out_s, fade_in_s
        self._dropped: dict[str, str] = {}

    @property
    def dropped(self) -> dict[str, str]:
        return dict(self._dropped)

    # -- cache ----------------------------------------------------------------

    @staticmethod
    def _key(provider: MusicProvider, request: MusicRequest) -> str:
        payload = json.dumps({"provider": provider.name, **provider.cache_key_parts(request)}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _lookup(self, key: str, dest: Path, provider: MusicProvider) -> Optional[MusicTrack]:
        wav, meta_path = self.cache_dir / f"{key}.wav", self.cache_dir / f"{key}.json"
        if not (wav.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shutil.copyfile(wav, dest)
            return MusicTrack(path=dest, provider=provider.name, model_id=meta.get("model_id", ""),
                              prompt=meta.get("prompt", ""), duration_s=meta["duration_s"], cost_usd=0.0,
                              from_cache=True, track_id=meta.get("track_id", ""), licence=meta.get("licence", ""),
                              notes=list(meta.get("notes", [])))
        except (OSError, ValueError, KeyError):
            logger.warning("Ignoring an unreadable music cache entry: %s", key)
            return None

    def _store(self, key: str, track: MusicTrack) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(track.path, self.cache_dir / f"{key}.wav")
        meta = {"model_id": track.model_id, "prompt": track.prompt, "duration_s": track.duration_s,
                "cost_usd": track.cost_usd, "track_id": track.track_id, "licence": track.licence,
                "notes": track.notes}
        (self.cache_dir / f"{key}.json").write_text(json.dumps(meta), encoding="utf-8")

    # -- the track ----------------------------------------------------------------

    def _fit(self, source: Path, total_s: float, dest: Path) -> None:
        """Cut to the video's length with a short fade in and a fade out. A
        track shorter than the video fades at its own end."""
        length = min(total_s, wav_duration(source))
        out = min(self.fade_out_s, length / 2)
        fade_in = min(self.fade_in_s, length / 4)
        ffmpeg.run(["-i", str(source), "-af",
                    f"atrim=0:{length:.3f},afade=t=in:st=0:d={fade_in:.3f},"
                    f"afade=t=out:st={length - out:.3f}:d={out:.3f}", str(dest)])

    def make(self, mood: str, tone: str, objective: str, direction: str, total_s: float, dest: Path) -> MusicResult:
        """The music for a video `total_s` long, written to `dest` (a WAV)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = MusicRequest(prompt=build_prompt(self.template, self.objectives, mood, tone, objective, direction),
                               mood=mood, duration_s=total_s, tone=tone, objective=objective,
                               creative_direction=direction)
        notes: list[str] = []
        raw = dest.with_name(dest.stem + "_raw.wav")
        for provider in self.providers:
            if provider.name in self._dropped:
                notes.append(f"{provider.name}: skipped for this run ({self._dropped[provider.name]})")
                continue
            if not provider.is_available():
                notes.append(f"{provider.name}: {provider.unavailable_reason()}")
                continue
            key = self._key(provider, request) if provider.cacheable else ""
            try:
                track = self._lookup(key, raw, provider) if key else None
                if track is None:
                    track = provider.generate(request, raw)
                    if key:
                        self._store(key, track)
                self._fit(raw, total_s, dest)
            except (MusicAuthError, MusicQuotaError) as exc:
                logger.warning("%s dropped for the rest of this run: %s", provider.name, exc)
                self._dropped[provider.name] = str(exc)
                notes.append(f"{provider.name}: {exc}")
                continue
            except (MusicError, RenderingError) as exc:
                logger.warning("%s could not make a track, trying the next provider: %s", provider.name, exc)
                notes.append(f"{provider.name}: {exc}")
                continue
            finally:
                raw.unlink(missing_ok=True)
            if track.duration_s < total_s - 0.5:
                track.notes.append(f"the track is {track.duration_s:.0f} s, shorter than the video "
                                   f"({total_s:.0f} s): it fades out where it ends")
            return MusicResult(dest, MusicInfo(
                provider=track.provider, model_id=track.model_id, track_id=track.track_id, prompt=track.prompt,
                path=str(dest), title=track.title, licence=track.licence, cost_usd=round(track.cost_usd, 4),
                from_cache=track.from_cache, duration_s=track.duration_s, notes=[*notes, *track.notes]))
        return MusicResult(None, MusicInfo(provider=NONE, prompt=request.prompt, notes=notes))

    def beat_times(self, track: Optional[Path]) -> list[float]:
        return []


def _build(name: str, settings, cfg: dict) -> MusicProvider:
    if name == "beatoven":
        b = cfg.get("beatoven") or {}
        return BeatovenProvider(
            api_key=settings.beatoven_api_key, base_url=b.get("url", ""), model_id=b.get("model_id", "beatoven-compose"),
            extra_s=float(b.get("extra_s", 1.0)), cost_usd=float(b.get("cost_usd") or 0.0),
            licence_note=" ".join(str(b.get("licence_note", "")).split()), poll_s=float(b.get("poll_s", 5.0)),
            poll_timeout_s=float(b.get("poll_timeout_s", 300.0)))
    if name == "lyria":
        lyria = cfg.get("lyria") or {}
        return LyriaProvider(api_key=settings.gemini_api_key, url=lyria.get("url", ""),
                             models=list(lyria.get("models") or []), api_revision=lyria.get("api_revision", ""))
    if name == "library":
        library = cfg.get("library") or {}
        return LibraryMusicProvider(settings.data_dir_resolved / library.get("folder", "music"),
                                    tuple(library.get("extensions") or (".wav", ".mp3")))
    raise ConfigError(f"Unknown music provider '{name}' (configs/{CONFIG_FILE}). Use beatoven, lyria, library or none.")


def get_music_service(settings=None) -> MusicService:
    """The service for the configured chain. A link named "none" ends it (the
    service falls to no music anyway). The cache folder is worked out now."""
    if settings is None:
        from src.core.config import settings as _settings

        settings = _settings
    cfg = load_yaml_config(CONFIG_FILE)
    names = []
    for name in cfg.get("chain") or ["beatoven", "lyria", "library", NONE]:
        if name == NONE:
            break
        names.append(name)
    return MusicService([_build(n, settings, cfg) for n in dict.fromkeys(names)],
                        settings.data_dir_resolved / "music_cache", template=cfg.get("prompt", ""),
                        objectives=cfg.get("objectives"), fade_out_s=float(cfg.get("fade_out_s", 1.5)),
                        fade_in_s=float(cfg.get("fade_in_s", 0.4)))
