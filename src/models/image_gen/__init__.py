"""Image generation service.

    from src.models.image_gen import get_image_service
    img = get_image_service().generate(ImageRequest(prompt=..., width=1600, height=900))

get_image_service() never raises at import or construction time. If no
provider is usable it returns a service whose generate() raises
ImageGenerationError, which the agents treat as "skip this image".
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from src.models.image_gen.base import (
    GeneratedImage, ImageGenerationAuthError, ImageGenerationError,
    ImageGenerationQuotaError, ImageGenerationTransportError, ImageProvider, ImageRequest,
)
from src.models.image_gen.chain import ChainImageProvider
from src.models.image_gen.cloudflare import CloudflareImageProvider
from src.models.image_gen.huggingface import HuggingFaceImageProvider
from src.models.image_gen.mock import MockImageProvider
from src.models.image_gen.openrouter import OpenRouterImageProvider

logger = logging.getLogger(__name__)

__all__ = [
    "GeneratedImage", "ImageGenerationError", "ImageGenerationAuthError",
    "ImageGenerationQuotaError", "ImageGenerationTransportError",
    "ImageProvider", "ImageRequest", "ImageService",
    "get_image_service", "build_provider",
]


def _split(models: str) -> list[str]:
    return [m.strip() for m in (models or "").split(",") if m.strip()]


def build_provider(kind: str, settings) -> Optional[ImageProvider]:
    """The provider for this configuration.

    "auto" builds the chain: Cloudflare Workers AI first because its free
    allowance renews daily and costs nothing, OpenRouter behind it because
    it is a different vendor and so does not share the thing that failed.
    Hugging Face is deliberately NOT in the automatic chain — this account's
    credits are exhausted, so including it would only add a 402 and a delay
    to every request. It stays selectable by name for anyone who buys more.

    With nothing configured this returns the mock provider: offline
    previews, so the pipelines still run end to end on a laptop with no
    keys at all."""
    kind = (kind or "auto").lower()
    retries, timeout = settings.image_gen_max_retries, settings.image_gen_timeout_s

    def cloudflare() -> Optional[ImageProvider]:
        return CloudflareImageProvider(
            account_id=settings.cloudflare_account_id, api_token=settings.cloudflare_api_token,
            model_ids=_split(settings.cloudflare_image_models),
            max_retries=retries, timeout_s=timeout,
            budget_s=getattr(settings, 'image_gen_budget_s', 210.0))

    def openrouter() -> Optional[ImageProvider]:
        return OpenRouterImageProvider(
            api_key=settings.openrouter_api_key,
            model_ids=_split(settings.openrouter_image_models),
            max_retries=max(1, retries - 1), timeout_s=max(timeout, 180.0),
            budget_s=getattr(settings, 'image_gen_budget_s', 210.0))

    if kind == "none":
        return None
    if kind == "mock":
        return MockImageProvider()
    if kind == "cloudflare":
        return cloudflare()
    if kind == "openrouter":
        return openrouter()
    if kind == "huggingface":
        # No budget argument: the Hugging Face path is the retired one (its
        # free tier ended in 402s), kept only for anyone who buys credits.
        return HuggingFaceImageProvider(token=settings.hf_token,
                                        model_ids=_split(settings.image_gen_models),
                                        max_retries=retries, timeout_s=timeout)

    usable = [p for p in (cloudflare(), openrouter()) if p is not None and p.is_available()]
    if not usable:
        return MockImageProvider()      # nothing configured -> offline previews
    return ChainImageProvider(usable) if len(usable) > 1 else usable[0]


class ImageService:
    """Wraps a provider with a content-addressed cache: the same prompt at
    the same size is generated once, which matters on a credit-limited
    free tier and makes re-running a generation cheap and reproducible."""

    def __init__(self, provider: Optional[ImageProvider], cache_dir: Path):
        self.provider = provider
        self.cache_dir = cache_dir

    @property
    def enabled(self) -> bool:
        return self.provider is not None and self.provider.is_available()

    @property
    def is_mock(self) -> bool:
        return isinstance(self.provider, MockImageProvider)

    def _key(self, request: ImageRequest) -> str:
        payload = json.dumps({
            "provider": getattr(self.provider, "name", "none"),
            "models": getattr(self.provider, "_model_ids", []),
            "prompt": request.prompt, "neg": request.negative_prompt,
            "w": request.width, "h": request.height, "seed": request.seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def generate(self, request: ImageRequest, dest_path: Optional[Path] = None) -> GeneratedImage:
        if not self.enabled:
            raise ImageGenerationError("Image generation is disabled or not configured.")
        key = self._key(request)
        cached = self.cache_dir / f"{key}.png"
        meta_path = self.cache_dir / f"{key}.json"
        if cached.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            result = GeneratedImage(path=cached, model_id=meta["model_id"], prompt=request.prompt,
                                    width=meta["width"], height=meta["height"], from_cache=True)
        else:
            result = self.provider.generate(request, cached)
            meta_path.write_text(json.dumps({"model_id": result.model_id, "width": result.width,
                                             "height": result.height}), encoding="utf-8")
        if dest_path is not None:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result.path, dest_path)
            result.path = dest_path
        return result


def get_image_service(settings=None) -> ImageService:
    if settings is None:
        from src.core.config import settings as _settings
        settings = _settings
    provider = build_provider(settings.image_gen_provider, settings)
    return ImageService(provider, settings.data_dir_resolved / "image_cache")
