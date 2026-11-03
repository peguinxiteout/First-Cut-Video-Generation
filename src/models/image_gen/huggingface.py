"""Hugging Face Inference Providers backend.

Adapted from the team's realestate-api-gen/generate.py. Kept: the 503
(cold start) / 429 (rate limit) / 401 (bad token) retry handling, which
worked. Changed for integration:

* No work at import time. The original raised ValueError on import when
  HF_TOKEN was unset, which would have crashed the entire app — including
  brochure and blog generation that don't need images at all. The client
  is now created on first use.
* The model is chosen per call from a fallback chain instead of being
  bound once at module load.
* Per-model parameters (see profiles.py) instead of one setting for all.
* `e.response` can be None on some transport errors; the original read
  `e.response.status_code` unconditionally.
* Output filenames are chosen by the caller (content-addressed), fixing
  the original's second-resolution timestamps that overwrote each other.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional

from src.models.image_gen.base import (
    GeneratedImage, ImageGenerationAuthError, ImageGenerationError, ImageProvider, ImageRequest,
)
from src.models.image_gen.profiles import fit_dimensions, profile_for

logger = logging.getLogger(__name__)


def _status_of(exc: Exception) -> Optional[int]:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


class HuggingFaceImageProvider(ImageProvider):
    name = "huggingface"

    def __init__(
        self,
        token: str,
        model_ids: list[str],
        max_retries: int = 4,
        timeout_s: float = 120.0,
        client_factory: Optional[Callable[..., object]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._token = token
        self._model_ids = [m.strip() for m in model_ids if m.strip()]
        self._max_retries = max(1, max_retries)
        self._timeout_s = timeout_s
        self._client_factory = client_factory  # injectable for tests
        self._sleep = sleep
        self._clients: dict[str, object] = {}

    def is_available(self) -> bool:
        return bool(self._token and self._model_ids)

    def _client(self, model_id: str):
        if model_id not in self._clients:
            factory = self._client_factory
            if factory is None:
                try:
                    from huggingface_hub import InferenceClient
                except ImportError as exc:  # pragma: no cover - dependency guard
                    raise ImageGenerationError(
                        "huggingface_hub is not installed; run `pip install -r requirements.txt`"
                    ) from exc
                factory = InferenceClient
            self._clients[model_id] = factory(model=model_id, token=self._token, timeout=self._timeout_s)
        return self._clients[model_id]

    def _generate_with_model(self, model_id: str, request: ImageRequest):
        profile = profile_for(model_id)
        width, height = fit_dimensions(request.width, request.height, profile)
        kwargs = dict(prompt=request.prompt, width=width, height=height,
                      num_inference_steps=profile.steps)
        if profile.guidance_scale is not None:
            kwargs["guidance_scale"] = profile.guidance_scale
        if profile.supports_negative_prompt and request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            kwargs["seed"] = request.seed

        client = self._client(model_id)
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                return client.text_to_image(**kwargs), width, height
            except Exception as exc:  # HfHubHTTPError and transport errors
                last_error = exc
                status = _status_of(exc)
                if status in (401, 403):
                    raise ImageGenerationAuthError(
                        f"Hugging Face rejected the token ({status}). Create a new token at "
                        "https://huggingface.co/settings/tokens and set HF_TOKEN in .env."
                    ) from exc
                if status == 503:
                    delay = (2 ** attempt) + random.uniform(0, 1)       # cold start
                elif status == 429:
                    delay = 5 + attempt * 3 + random.uniform(0, 2)      # rate limit
                elif status in (500, 502, 504) or status is None:
                    delay = 2 + attempt + random.uniform(0, 1)          # transient
                else:
                    break  # 400/404/422 etc: this model won't work; try the next one
                if attempt < self._max_retries - 1:
                    logger.warning("HF %s on %s (attempt %d/%d); retrying in %.1fs",
                                   status, model_id, attempt + 1, self._max_retries, delay)
                    self._sleep(delay)
        raise ImageGenerationError(f"{model_id} failed: {last_error}")

    def generate(self, request: ImageRequest, output_path: Path) -> GeneratedImage:
        if not self.is_available():
            raise ImageGenerationAuthError("HF_TOKEN is not set; image generation is disabled.")
        failures: list[str] = []
        for model_id in self._model_ids:
            try:
                image, width, height = self._generate_with_model(model_id, request)
            except ImageGenerationAuthError:
                raise  # same token for every model; no point trying the rest
            except ImageGenerationError as exc:
                failures.append(str(exc))
                logger.warning("Image model %s unavailable, trying next: %s", model_id, exc)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(output_path, format="PNG")
            return GeneratedImage(path=output_path, model_id=model_id, prompt=request.prompt,
                                  width=width, height=height)
        raise ImageGenerationError("All image models failed: " + " | ".join(failures))
