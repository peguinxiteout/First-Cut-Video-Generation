"""Image-to-video service.

    from src.models.video_gen import get_video_service
    clip = get_video_service("auto", duration_s=15).generate(request, dest_path)

get_video_service() never fails for want of a provider: every chain ends
with pan-and-zoom, which needs no key. generate() raises VideoGenerationError
only when even that could not make the clip (an unreadable picture).

One service is one RUN. A provider that answers with an auth or quota error
is dropped for the rest of the run, so the remaining shots do not each wait
for the same refusal; any other failure costs that shot only, which falls to
the next provider with the reason recorded on the clip.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from src.models.video_gen import registry, selector
from src.models.video_gen.base import (
    ImageToVideoRequest, ProviderSpec, VideoClip, VideoContentRefusedError, VideoGenerationAuthError,
    VideoGenerationError, VideoGenerationQuotaError, VideoGenerationTransportError, VideoProvider, check_request,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ImageToVideoRequest", "ProviderSpec", "VideoClip", "VideoContentRefusedError", "VideoGenerationAuthError",
    "VideoGenerationError", "VideoGenerationQuotaError", "VideoGenerationTransportError", "VideoProvider",
    "VideoService", "get_video_service",
]


def _file_digest(path: Optional[Path]) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoService:
    """A chain of providers with a content-addressed cache: the same picture,
    prompt, camera and size is paid for once, which matters on a metered API
    and makes re-running a generation cheap and repeatable."""

    def __init__(self, providers: list[VideoProvider], cache_dir: Path, settings=None, note: str = "",
                 estimated_cost_usd: Optional[float] = None):
        self.providers = list(providers)
        self.cache_dir = cache_dir
        self.note = note
        self.estimated_cost_usd = estimated_cost_usd
        self._settings = settings
        self._dropped: dict[str, str] = {}

    @property
    def dropped(self) -> dict[str, str]:
        """Providers given up on for this run, with the reason."""
        return dict(self._dropped)

    # -- cache ----------------------------------------------------------------

    def _key(self, provider: VideoProvider, request: ImageToVideoRequest) -> str:
        payload = json.dumps({
            "provider": provider.id, "model": provider.model_id,
            "image": _file_digest(request.image_path), "end_image": _file_digest(request.end_image_path),
            "prompt": request.prompt, "camera": request.camera, "seconds": request.duration_s,
            "w": request.width, "h": request.height, "fps": request.fps, "focal": request.focal,
            "seed": request.seed,
            **({"negative": request.negative_prompt} if request.negative_prompt else {}),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _lookup(self, key: str, provider: VideoProvider) -> Optional[VideoClip]:
        video, meta_path = self.cache_dir / f"{key}.mp4", self.cache_dir / f"{key}.json"
        if not (video.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return VideoClip(path=video, provider=meta["provider"], model_id=meta["model_id"],
                             duration_s=meta["duration_s"], width=meta["width"], height=meta["height"],
                             fps=meta["fps"], from_cache=True, cost_usd=0.0)
        except (OSError, ValueError, KeyError):
            logger.warning("Ignoring an unreadable cache entry for %s: %s", provider.id, key)
            return None

    def _store(self, key: str, clip: VideoClip) -> None:
        meta = {"provider": clip.provider, "model_id": clip.model_id, "duration_s": clip.duration_s,
                "width": clip.width, "height": clip.height, "fps": clip.fps, "cost_usd": clip.cost_usd}
        (self.cache_dir / f"{key}.json").write_text(json.dumps(meta), encoding="utf-8")

    # -- one shot ---------------------------------------------------------------

    def generate(self, request: ImageToVideoRequest, dest_path: Path) -> VideoClip:
        check_request(request)       # a fault in the request fails every provider alike: say so once
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        reasons: list[str] = [self.note] if self.note else []
        for provider in self.providers:
            if provider.id in self._dropped:
                reasons.append(f"{provider.id}: dropped for this run ({self._dropped[provider.id]})")
                continue
            try:
                clip = self._try(provider, request, dest_path)
            except VideoGenerationQuotaError as exc:
                self._drop(provider, f"allowance spent: {exc}", reasons)
                selector.mark_quota(provider.id, self._settings)
                continue
            except VideoGenerationAuthError as exc:
                self._drop(provider, f"key rejected: {exc}", reasons)
                continue
            except VideoGenerationError as exc:
                logger.warning("%s could not make this clip, trying the next provider: %s", provider.id, exc)
                reasons.append(f"{provider.id}: {exc}")
                continue
            if clip is None:
                reasons.append(f"{provider.id}: not available")
                continue
            clip.reason = "; ".join(reasons)
            return clip
        raise VideoGenerationError("no provider could make this clip: " + "; ".join(reasons))

    def _drop(self, provider: VideoProvider, why: str, reasons: list[str]) -> None:
        logger.warning("%s dropped for the rest of this run: %s", provider.id, why)
        self._dropped[provider.id] = why
        reasons.append(f"{provider.id}: {why}")

    def _try(self, provider: VideoProvider, request: ImageToVideoRequest, dest_path: Path) -> Optional[VideoClip]:
        key = self._key(provider, request) if provider.cacheable else ""
        if key:
            cached = self._lookup(key, provider)
            if cached is not None:
                shutil.copyfile(cached.path, dest_path)
                cached.path = dest_path
                return cached
        if not provider.is_available():
            return None
        if not key:
            return provider.generate(request, dest_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        clip = provider.generate(request, self.cache_dir / f"{key}.mp4")
        self._store(key, clip)
        shutil.copyfile(clip.path, dest_path)
        clip.path = dest_path
        return clip


def get_video_service(choice: Optional[str] = None, duration_s: float = 15, settings=None) -> VideoService:
    """The service for a model choice: "auto", a provider id, or "kenburns".
    The cache folder is worked out now, not at import."""
    if settings is None:
        from src.core.config import settings as _settings

        settings = _settings
    plan = selector.resolve(choice, duration_s, settings)
    catalog = selector.load_catalog()
    providers = []
    for provider_id in plan.ids:
        spec = catalog[provider_id]
        providers.append(registry.ADAPTERS[spec.adapter](spec, selector.resolve_key(spec.key_env, settings)))
    return VideoService(providers, settings.data_dir_resolved / "video_cache", settings, note=plan.note,
                        estimated_cost_usd=plan.est_cost_usd)
