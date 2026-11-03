"""Two vendors, one provider: try the free one, fall over to the paid one.

The failure that prompted this was not a bug. Hugging Face's monthly
credits ran out and every model returned 402, so the whole image feature
stopped at once — a single account's allowance was a single point of
failure, and walking a longer list of models on that same account could
never have helped.

So the fallback here is between VENDORS, not models. The primary is
Cloudflare Workers AI, whose free allowance renews daily; when that day's
budget is gone (or the account is not configured at all) the request goes
to OpenRouter instead, which shares no infrastructure with it.

WHAT COUNTS AS A REASON TO FAIL OVER

Only a condition the second vendor would not also hit:

    quota spent        the day's allowance, or a balance too low to run
    not configured     no account for this vendor
    auth rejected      wrong or missing credentials
    every model failed the vendor is up but could not serve the request

A content-guardrail refusal is deliberately NOT in that list — a prompt
one vendor declines the other will usually decline too, and spending the
paid budget to be told the same thing is waste.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from src.models.image_gen.base import (
    GeneratedImage, ImageGenerationAuthError, ImageGenerationError,
    ImageGenerationQuotaError, ImageProvider, ImageRequest,
)

logger = logging.getLogger(__name__)


class ChainImageProvider(ImageProvider):
    """Providers in order of preference; the first that can serve wins."""

    name = "chain"

    def __init__(self, providers: Sequence[ImageProvider]):
        self._providers = [p for p in providers if p is not None]

    @property
    def providers(self) -> list[ImageProvider]:
        return list(self._providers)

    def is_available(self) -> bool:
        return any(p.is_available() for p in self._providers)

    @property
    def _model_ids(self) -> list[str]:
        """Everything this chain could use — part of the cache key, so a
        configuration change produces new images rather than serving ones
        made by a provider that is no longer in play."""
        out: list[str] = []
        for provider in self._providers:
            out.extend(getattr(provider, "_model_ids", []) or [])
        return out

    def generate(self, request: ImageRequest, output_path: Path) -> GeneratedImage:
        failures: list[str] = []
        for provider in self._providers:
            if not provider.is_available():
                failures.append(f"{provider.name}: not configured")
                continue
            try:
                return provider.generate(request, output_path)
            except ImageGenerationQuotaError as exc:
                failures.append(f"{provider.name}: {exc}")
                logger.warning("%s is out of allowance; trying the next provider", provider.name)
            except ImageGenerationAuthError as exc:
                failures.append(f"{provider.name}: {exc}")
                logger.warning("%s rejected our credentials; trying the next provider", provider.name)
            except ImageGenerationError as exc:
                failures.append(f"{provider.name}: {exc}")
                logger.warning("%s could not serve this request; trying the next provider: %s",
                               provider.name, exc)
        raise ImageGenerationError("; ".join(failures) or "no image provider is configured")
