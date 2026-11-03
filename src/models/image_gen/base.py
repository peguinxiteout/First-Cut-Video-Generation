"""Provider-agnostic image generation interface.

Architecture:  ImageGenerationAgent -> ImageProvider -> concrete model.
Nothing outside src/models/image_gen knows which model or vendor produced
an image, so swapping FLUX for SDXL (or for a paid API later) is a config
change, not a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ImageGenerationError(Exception):
    """Raised when every configured model failed for a request. Callers
    treat this as a soft failure: the brochure/blog is still produced,
    just without that generated image."""


class ImageGenerationAuthError(ImageGenerationError):
    """Token missing/invalid. Not worth retrying other models for — they
    all share the same token."""


class ImageGenerationQuotaError(ImageGenerationError):
    """The account's allowance is spent — Cloudflare's daily neuron budget,
    or an OpenRouter balance below the $1 an image request pre-authorises.

    Distinct from a plain failure because it says something different about
    what to do next. Another MODEL on the same account will fail the same
    way, so there is no point walking the chain; another VENDOR will not,
    so this is exactly the signal to fail over to one. Hugging Face's 402
    was this condition, and having no name for it is why every model in the
    chain was tried and reported separately when the answer was always
    going to be the same.

    Not a hard failure either: the brochure or blog is still produced,
    without that image, and the reason is surfaced to the person."""


class ImageGenerationTransportError(ImageGenerationError):
    """The request never got an answer: a read timeout, a dropped
    connection, a DNS or TLS failure.

    This has a name because of a real generation that died on

        HTTPSConnectionPool(host='api.cloudflare.com', port=443):
        Read timed out. (read timeout=120.0)

    A timeout is raised by the HTTP library, not by the vendor, so it was
    not an ImageGenerationError and nothing downstream recognised it: the
    provider's retry loop never saw it, the next model was never tried,
    the chain never failed over to the second vendor, and the agent's
    `except ImageGenerationError` — the soft failure that is supposed to
    let a brochure finish without its generated images — let it straight
    through. One slow request took down the whole brochure.

    It is a subclass, so every existing handler treats it as the soft
    failure it is. Providers raise it instead of letting a transport
    exception escape, and retry it: a read timeout is the most retryable
    condition there is, because an image model answering slowly is the
    normal case, not an error."""


@dataclass
class ImageRequest:
    prompt: str
    width: int = 1024
    height: int = 1024
    # Only sent to models that actually support it (see ModelProfile).
    negative_prompt: str = ""
    seed: Optional[int] = None


@dataclass
class GeneratedImage:
    path: Path
    model_id: str
    prompt: str
    width: int
    height: int
    from_cache: bool = False
    # Always True for AI output; carried through to the page so every
    # generated visual is labelled as an artistic impression.
    is_ai_generated: bool = True
    meta: dict = field(default_factory=dict)


class ImageProvider(ABC):
    """One vendor/transport (Hugging Face Inference, a mock, a future paid
    API). A provider may try several models internally."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: ImageRequest, output_path: Path) -> GeneratedImage:
        """Generate one image and write it to output_path (PNG)."""

    def is_available(self) -> bool:
        return True
