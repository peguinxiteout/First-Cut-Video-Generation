"""OpenRouter Images backend — the fallback provider.

WHY THIS ONE AS THE FALLBACK

A fallback is only worth having if it fails independently of the primary.
OpenRouter is an aggregator that routes to Google, Black Forest Labs,
ByteDance and OpenAI backends and re-routes around a single upstream
outage itself, so it shares no infrastructure with Cloudflare Workers AI.
The project also already holds an OpenRouter key for the text fallback,
so this costs no new account.

It is not free. Every paid image request checks for more than $1 of
credit before it runs, and a request without that balance comes back 402
— the same signal that ended the Hugging Face path. That is why this is
the fallback and not the primary: the system stays working with no money
spent at all, and reaches for this only when the day's free allowance is
gone.

SIZE

Images are requested by aspect ratio and a resolution tier rather than by
exact pixels, which suits the caller: the layout cares that a blog hero
is 16:9, not that it is exactly 1600x900, and every model here renders at
its own native scale anyway.
"""
from __future__ import annotations

import base64
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

from src.models.image_gen import transport
from src.models.image_gen.base import (
    GeneratedImage, ImageGenerationAuthError, ImageGenerationError,
    ImageGenerationQuotaError, ImageGenerationTransportError, ImageProvider, ImageRequest,
)

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/images"

# Aspect ratios the API accepts, as (label, width/height).
_RATIOS = (
    ("21:9", 21 / 9), ("16:9", 16 / 9), ("3:2", 3 / 2), ("4:3", 4 / 3),
    ("1:1", 1.0), ("3:4", 3 / 4), ("2:3", 2 / 3), ("9:16", 9 / 16),
)


def nearest_aspect_ratio(width: int, height: int) -> str:
    """The supported ratio closest to what was asked for.

    Compared in log space so 2:1 is judged as far from 1:1 as 1:2 is —
    in linear space the wide side dominates and a portrait request can
    quietly come back landscape."""
    if width <= 0 or height <= 0:
        return "1:1"
    wanted = math.log(width / height)
    return min(_RATIOS, key=lambda pair: abs(math.log(pair[1]) - wanted))[0]


def resolution_tier(width: int, height: int) -> str:
    longest = max(width, height, 1)
    if longest <= 768:
        return "512"
    if longest <= 1536:
        return "1K"
    return "2K"


class OpenRouterImageProvider(ImageProvider):
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        model_ids: list[str],
        max_retries: int = 3,
        timeout_s: float = 180.0,
        post: Optional[Callable[..., object]] = None,
        sleep: Callable[[float], None] = time.sleep,
        budget_s: float = transport.DEFAULT_BUDGET_S,
    ):
        self._api_key = (api_key or "").strip()
        self._model_ids = [m.strip() for m in model_ids if m.strip()]
        self._max_retries = max(1, max_retries)
        self._timeout_s = timeout_s
        self._post = post
        self._sleep = sleep
        self._budget_s = budget_s

    def is_available(self) -> bool:
        return bool(self._api_key and self._model_ids)

    def _request(self, model_id: str, request: ImageRequest, at: Optional[float] = None):
        post = self._post
        if post is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImageGenerationError(
                    "requests is not installed; run `pip install -r requirements.txt`") from exc
            post = requests.post
        body = {
            "model": model_id,
            "prompt": request.prompt,
            "aspect_ratio": nearest_aspect_ratio(request.width, request.height),
            "resolution": resolution_tier(request.width, request.height),
        }
        # Through transport.call, so a read timeout or a dropped connection
        # arrives as an ImageGenerationError rather than escaping every
        # handler built to absorb a failed image.
        return transport.call(post, ENDPOINT,
                              headers={"Authorization": f"Bearer {self._api_key}",
                                       "Content-Type": "application/json"},
                              json=body, read_timeout=self._timeout_s, at=at)

    @staticmethod
    def _image_bytes(response) -> bytes:
        body = response.json()
        entries = body.get("data") or []
        if not entries:
            raise ImageGenerationError(f"OpenRouter returned no image: {str(body)[:200]}")
        encoded = entries[0].get("b64_json")
        if not encoded:
            raise ImageGenerationError("OpenRouter returned an image without data")
        return base64.b64decode(encoded)

    def _generate_with_model(self, model_id: str, request: ImageRequest,
                             at: Optional[float] = None) -> bytes:
        last_error: Optional[str] = None
        for attempt in range(self._max_retries):
            if transport.expired(at):
                raise ImageGenerationTransportError(
                    f"{model_id}: gave up after {self._budget_s:.0f}s "
                    f"({last_error or 'no response'})")
            try:
                response = self._request(model_id, request, at=at)
            except ImageGenerationTransportError as exc:
                last_error = str(exc)
                if attempt >= self._max_retries - 1 or transport.expired(at):
                    raise
                delay = 3 + attempt * 2 + random.uniform(0, 1)
                logger.warning("OpenRouter %s on %s (attempt %d/%d); retrying in %.1fs",
                               exc, model_id, attempt + 1, self._max_retries, delay)
                self._sleep(delay)
                continue
            status = getattr(response, "status_code", 0)

            if status == 200:
                return self._image_bytes(response)

            if status == 401:
                raise ImageGenerationAuthError(
                    "OpenRouter rejected the key (401). Check OPENROUTER_API_KEY in .env.")
            if status == 402:
                raise ImageGenerationQuotaError(
                    "OpenRouter needs more than $1 of credit before it will run an image "
                    "request (402). Top up at https://openrouter.ai/credits, or leave image "
                    "generation to the free Cloudflare allowance.")
            if status == 403:
                # A moderation block. Rewording might pass, retrying will not.
                raise ImageGenerationError(
                    f"OpenRouter refused this prompt ({model_id}): content guardrail.")
            if status == 429:
                delay = float(getattr(response, "headers", {}).get("Retry-After", 0) or 0)
                delay = delay or (3 + attempt * 3 + random.uniform(0, 2))
            elif status in (500, 502, 503, 504):
                delay = 2 + attempt + random.uniform(0, 1)
            else:
                last_error = f"HTTP {status}: {getattr(response, 'text', '')[:200]}"
                break

            last_error = f"HTTP {status}"
            if attempt < self._max_retries - 1:
                logger.warning("OpenRouter %s on %s (attempt %d/%d); retrying in %.1fs",
                               status, model_id, attempt + 1, self._max_retries, delay)
                self._sleep(delay)
        raise ImageGenerationError(f"{model_id} failed: {last_error}")

    def generate(self, request: ImageRequest, output_path: Path) -> GeneratedImage:
        if not self.is_available():
            raise ImageGenerationAuthError(
                "OpenRouter image generation is not configured; set OPENROUTER_API_KEY.")
        failures: list[str] = []
        at = transport.deadline(self._budget_s)   # one budget for every model and retry
        for model_id in self._model_ids:
            try:
                data = self._generate_with_model(model_id, request, at=at)
            except (ImageGenerationAuthError, ImageGenerationQuotaError):
                raise      # the key and the balance are shared by every model
            except ImageGenerationError as exc:
                failures.append(str(exc))
                logger.warning("Image model %s unavailable, trying next: %s", model_id, exc)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
            return GeneratedImage(path=output_path, model_id=model_id, prompt=request.prompt,
                                  width=request.width, height=request.height)
        raise ImageGenerationError("All image models failed: " + " | ".join(failures))
