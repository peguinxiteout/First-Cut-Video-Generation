"""Cloudflare Workers AI backend — the default image provider.

WHY THIS ONE

Hugging Face Inference Providers stopped serving this project: the
account's monthly included credits ran out and every model started
returning 402 Payment Required. That is not a transient failure to retry
around, it is the free tier ending, so the fix is a different vendor
rather than better error handling.

Workers AI is the only provider surveyed with a *recurring, card-free*
allowance rather than a one-off trial: 10,000 neurons a day on the default
Workers Free plan, which is roughly 170 FLUX.1-schnell images a day, and
SDXL is currently billed at $0.00 per step on top of that. It also runs
the same FLUX.1-schnell and SDXL weights the project already targeted, so
prompts that were tuned against Hugging Face behave the same here.

TWO THINGS THIS API DOES THAT THE REST DO NOT

1. `flux-1-schnell` has no width/height in its input schema at all. It
   returns a square image, full stop. Asking it for a 16:9 blog hero
   silently gets you a square one. So a request whose aspect ratio is not
   square is routed to SDXL, which does take width/height — see
   `_model_for`. Getting this wrong is invisible until someone looks at
   the page and finds the hero cropped to a circle-ish square.

2. The two models answer in different encodings. FLUX returns JSON with
   base64 in `result.image`; SDXL streams raw PNG bytes. Both come back
   200 OK, so sniffing the content type is not optional.

QUOTA, WHICH IS THE WHOLE POINT

The daily neuron budget is reported as HTTP 429 with error code 3036 in
the body, and capacity pressure as 429 with code 3040. They mean opposite
things — one is "come back tomorrow", the other is "retry in a moment" —
and treating them alike would either burn the retry budget against a wall
or give up on a request that would have worked. 3036 is this provider's
version of the 402 that ended the Hugging Face path, so it raises
ImageGenerationQuotaError, which the service uses to fail over to the
fallback provider rather than to the next model.
"""
from __future__ import annotations

import base64
import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional

from src.models.image_gen import transport
from src.models.image_gen.base import (
    GeneratedImage, ImageGenerationAuthError, ImageGenerationError,
    ImageGenerationQuotaError, ImageGenerationTransportError, ImageProvider, ImageRequest,
)
from src.models.image_gen.profiles import fit_dimensions, profile_for

logger = logging.getLogger(__name__)

API_ROOT = "https://api.cloudflare.com/client/v4/accounts"

# Cloudflare error codes that mean something specific to us.
CODE_DAILY_LIMIT = 3036      # the free allowance is spent until tomorrow
CODE_CAPACITY = 3040         # busy; worth retrying

# Models whose input schema has no width/height. A non-square request must
# not be sent to one of these.
FIXED_SQUARE = frozenset({
    "@cf/black-forest-labs/flux-1-schnell",
    "@cf/black-forest-labs/flux-2-klein-4b",
})

# How far from square counts as square. A hero at 1024x1000 is square
# enough to take the better model; 16:9 plainly is not.
SQUARE_TOLERANCE = 0.12


def _is_square(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return True
    return abs(width - height) / max(width, height) <= SQUARE_TOLERANCE


class CloudflareImageProvider(ImageProvider):
    """Workers AI text-to-image over plain HTTPS.

    No SDK: this is one POST with a bearer token, and adding a dependency
    for that would be a dependency to keep current for no gain."""

    name = "cloudflare"

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model_ids: list[str],
        max_retries: int = 4,
        timeout_s: float = 120.0,
        post: Optional[Callable[..., object]] = None,
        sleep: Callable[[float], None] = time.sleep,
        budget_s: float = transport.DEFAULT_BUDGET_S,
    ):
        self._account_id = (account_id or "").strip()
        self._api_token = (api_token or "").strip()
        self._model_ids = [m.strip() for m in model_ids if m.strip()]
        self._max_retries = max(1, max_retries)
        self._timeout_s = timeout_s
        self._post = post          # injectable for tests
        self._sleep = sleep
        self._budget_s = budget_s

    def is_available(self) -> bool:
        return bool(self._account_id and self._api_token and self._model_ids)

    # -- model choice -------------------------------------------------------

    def _model_for(self, request: ImageRequest) -> list[str]:
        """The configured models, with any that cannot honour this request's
        shape moved out of the way.

        A square-only model asked for a 16:9 hero returns a square image and
        no error at all, which is the worst kind of failure: the page looks
        wrong and nothing in the logs says why."""
        if _is_square(request.width, request.height):
            return list(self._model_ids)
        usable = [m for m in self._model_ids if m not in FIXED_SQUARE]
        skipped = [m for m in self._model_ids if m in FIXED_SQUARE]
        if skipped and usable:
            logger.debug("Skipping square-only model(s) %s for a %dx%d request",
                         ", ".join(skipped), request.width, request.height)
        # If every configured model is square-only, use them anyway rather
        # than failing outright — a square image beats no image.
        return usable or list(self._model_ids)

    # -- one call -----------------------------------------------------------

    def _payload(self, model_id: str, request: ImageRequest) -> tuple[dict, int, int]:
        profile = profile_for(model_id)
        width, height = fit_dimensions(request.width, request.height, profile)
        if model_id in FIXED_SQUARE:
            # The model decides the size; report what it actually returns.
            body = {"prompt": request.prompt, "steps": min(profile.steps, 8)}
            if request.seed is not None:
                body["seed"] = request.seed
            return body, 1024, 1024
        body = {
            "prompt": request.prompt,
            "width": width,
            "height": height,
            # Workers AI caps SDXL at 20 steps and rejects more.
            "num_steps": min(profile.steps, 20),
        }
        if profile.guidance_scale is not None:
            body["guidance"] = profile.guidance_scale
        if profile.supports_negative_prompt and request.negative_prompt:
            body["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            body["seed"] = request.seed
        return body, width, height

    def _request(self, model_id: str, body: dict, at: Optional[float] = None):
        post = self._post
        if post is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImageGenerationError(
                    "requests is not installed; run `pip install -r requirements.txt`") from exc
            post = requests.post
        # Through transport.call, so a read timeout or a dropped connection
        # arrives as an ImageGenerationError and is retried and failed over
        # like any other failure — see src/models/image_gen/transport.py.
        return transport.call(
            post,
            f"{API_ROOT}/{self._account_id}/ai/run/{model_id}",
            headers={"Authorization": f"Bearer {self._api_token}"},
            json=body,
            read_timeout=self._timeout_s,
            at=at,
        )

    @staticmethod
    def _error_codes(response) -> set[int]:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON error body is not a crash
            return set()
        return {e.get("code") for e in (body.get("errors") or []) if isinstance(e, dict)}

    @staticmethod
    def _image_bytes(response) -> bytes:
        """FLUX answers with base64 inside JSON; SDXL streams raw bytes."""
        content_type = (response.headers or {}).get("content-type", "")
        if "application/json" not in content_type:
            return response.content
        body = response.json()
        if not body.get("success", True):
            raise ImageGenerationError(f"Workers AI returned an error: {body.get('errors')}")
        encoded = (body.get("result") or {}).get("image")
        if not encoded:
            raise ImageGenerationError("Workers AI returned no image in its response")
        return base64.b64decode(encoded)

    def _generate_with_model(self, model_id: str, request: ImageRequest,
                             at: Optional[float] = None) -> tuple[bytes, int, int]:
        body, width, height = self._payload(model_id, request)
        last_error: Optional[str] = None
        for attempt in range(self._max_retries):
            if transport.expired(at):
                raise ImageGenerationTransportError(
                    f"{model_id}: gave up after {self._budget_s:.0f}s "
                    f"({last_error or 'no response'})")
            try:
                response = self._request(model_id, body, at=at)
            except ImageGenerationTransportError as exc:
                # The most retryable condition there is: an image model
                # answering slowly is the normal case.
                last_error = str(exc)
                if attempt >= self._max_retries - 1 or transport.expired(at):
                    raise
                delay = 2 + attempt * 2 + random.uniform(0, 1)
                logger.warning("Workers AI %s on %s (attempt %d/%d); retrying in %.1fs",
                               exc, model_id, attempt + 1, self._max_retries, delay)
                self._sleep(delay)
                continue
            status = getattr(response, "status_code", 0)

            if status == 200:
                return self._image_bytes(response), width, height

            if status in (401, 403):
                raise ImageGenerationAuthError(
                    f"Cloudflare rejected the credentials ({status}). Check CLOUDFLARE_ACCOUNT_ID "
                    "and that CLOUDFLARE_API_TOKEN has the 'Workers AI' permission.")

            if status == 429:
                codes = self._error_codes(response)
                if CODE_DAILY_LIMIT in codes:
                    raise ImageGenerationQuotaError(
                        "Cloudflare Workers AI daily free allowance is used up (error 3036). "
                        "It resets every day; set an OpenRouter key to fail over in the meantime.")
                delay = 2 + attempt * 2 + random.uniform(0, 1)     # capacity: 3040
            elif status in (500, 502, 503, 504):
                delay = 2 + attempt + random.uniform(0, 1)
            else:
                # 400/404/413/422: this model will not serve this request.
                last_error = f"HTTP {status}: {getattr(response, 'text', '')[:200]}"
                break

            last_error = f"HTTP {status}"
            if attempt < self._max_retries - 1:
                logger.warning("Workers AI %s on %s (attempt %d/%d); retrying in %.1fs",
                               status, model_id, attempt + 1, self._max_retries, delay)
                self._sleep(delay)
        raise ImageGenerationError(f"{model_id} failed: {last_error}")

    # -- the provider contract ---------------------------------------------

    def generate(self, request: ImageRequest, output_path: Path) -> GeneratedImage:
        if not self.is_available():
            raise ImageGenerationAuthError(
                "Cloudflare Workers AI is not configured; set CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN in .env.")
        failures: list[str] = []
        # One budget for the whole provider — every model, every retry.
        # Counting retries instead of timing them meant four attempts across
        # two models at a 120s read timeout: sixteen minutes of a progress
        # bar going nowhere before the chain even considered the other vendor.
        at = transport.deadline(self._budget_s)
        for model_id in self._model_for(request):
            try:
                data, width, height = self._generate_with_model(model_id, request, at=at)
            except (ImageGenerationAuthError, ImageGenerationQuotaError):
                raise      # every model shares the token and the daily budget
            except ImageGenerationError as exc:
                failures.append(str(exc))
                logger.warning("Image model %s unavailable, trying next: %s", model_id, exc)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
            return GeneratedImage(path=output_path, model_id=model_id, prompt=request.prompt,
                                  width=width, height=height)
        raise ImageGenerationError("All image models failed: " + " | ".join(failures))
