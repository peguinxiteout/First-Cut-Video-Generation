from __future__ import annotations

import base64
from typing import Any, Optional

from src.core.config import settings
from src.core.exceptions import ModelProviderError
from src.models.base import LLMProvider
from src.models.gemini_provider import _extract_json  # reuse the same forgiving JSON parser

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _extract_message_content(response, context: str) -> str:
    """Defensively pull the text out of a chat-completion response.

    Free-tier/newer models occasionally return a response shaped
    differently than expected — empty `choices`, a `None` message, a
    refusal instead of content, etc. Accessing `response.choices[0]...`
    directly on a malformed response raises a raw TypeError/AttributeError
    that with_retries() can't catch (it only retries ModelProviderError),
    so it escapes uncaught all the way to the user as an opaque
    "'NoneType' object is not subscriptable"-style message. This raises a
    clear ModelProviderError instead, for every step of the way."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelProviderError(f"OpenRouter {context}: response had no choices ({response!r})")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ModelProviderError(f"OpenRouter {context}: response choice had no message ({response!r})")

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ModelProviderError(f"OpenRouter {context}: model refused — {refusal}")

    content = getattr(message, "content", None)
    if not content:
        raise ModelProviderError(f"OpenRouter {context}: empty response content ({response!r})")
    return content


class OpenRouterProvider(LLMProvider):
    """Fallback provider. Uses the OpenAI-compatible chat completions API that
    OpenRouter exposes, so any OpenRouter model slug works here — including
    the rotating set of free ones. Not every free model supports vision;
    generate_json_with_image will raise ModelProviderError if the chosen
    model rejects image input."""

    name = "openrouter"

    def __init__(self, model: str):
        if not settings.openrouter_api_key:
            raise ModelProviderError(
                "OPENROUTER_API_KEY is not set. Get a free key at https://openrouter.ai/keys"
            )
        try:
            from openai import OpenAI  # noqa: WPS433
        except ImportError as exc:
            raise ModelProviderError(
                f"Could not import the openai package (used for OpenRouter's OpenAI-"
                f"compatible API). Original error: {exc}. This is imported lazily, so it "
                "only surfaces on the first real fallback call. Run `pip show openai` to "
                "check; if it's missing, `pip install openai` (see requirements.txt for the "
                "pinned version). On a hosted platform, trigger a clean rebuild/redeploy so "
                "requirements.txt is reinstalled from scratch."
            ) from exc

        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            # Generous timeout: a batched multi-image request to a slower
            # free model can legitimately take a while. max_retries=0
            # because with_retries() (src/agents/base.py) already retries
            # with its own backoff — stacking the SDK's internal retries on
            # top would make wait times unpredictable and hard to reason
            # about (this is deliberately the ONLY place retries happen).
            timeout=120.0,
            max_retries=0,
        )
        self._model = model

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.4) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=temperature
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(f"OpenRouter generate() failed: {exc}") from exc

        return _extract_message_content(response, "generate()")

    def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        json_instruction = (
            "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble, no commentary."
        )
        text = self.generate(prompt + json_instruction, system=system, temperature=temperature)
        return _extract_json(text)

    def generate_json_with_image(
        self,
        prompt: str,
        image_path: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        return self.generate_json_with_images(prompt, [image_path], system=system, temperature=temperature)

    def generate_json_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        content: list[dict] = []
        for idx, image_path in enumerate(image_paths, start=1):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            content.append({"type": "text", "text": f"Image {idx}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
        content.append({"type": "text", "text": prompt + "\n\nRespond with ONLY valid JSON."})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=temperature
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(
                f"OpenRouter generate_json_with_images() failed (model may not support vision): {exc}"
            ) from exc

        content_out = _extract_message_content(response, "generate_json_with_images()")
        return _extract_json(content_out)
