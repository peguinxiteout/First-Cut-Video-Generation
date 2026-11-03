from __future__ import annotations

import json
import re
from typing import Any, Optional

from src.core.config import settings
from src.core.exceptions import ModelOutputParseError, ModelProviderError
from src.models.base import LLMProvider

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in markdown fences or add stray text.
    Try a few increasingly forgiving strategies before giving up."""
    candidates = [text.strip()]

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    # Fallback: take the substring between the first '{' and the last '}'.
    first_brace, last_brace = text.find("{"), text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    raise ModelOutputParseError(f"Could not parse JSON from model output: {text[:300]!r}")


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str):
        if not settings.gemini_api_key:
            raise ModelProviderError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://aistudio.google.com/apikey"
            )
        # Imported lazily so the rest of the app can be imported/tested
        # without the google-genai package installed if someone only wants
        # e.g. the deterministic evaluators.
        try:
            from google import genai  # noqa: WPS433
        except ImportError as exc:
            # This exact failure has a specific, well-known cause worth
            # naming rather than letting Python's raw (and genuinely
            # confusing) "cannot import name 'genai' from 'google' (unknown
            # location)" reach the user: `google` is a namespace package
            # shared by several unrelated PyPI packages (google-genai,
            # google-generativeai, google-cloud-*, protobuf, ...) with no
            # single install location of its own. That message means
            # SOMETHING contributes to the `google` namespace, but the
            # specific google-genai package — which is what provides the
            # `genai` submodule — isn't actually installed/importable in
            # this environment, even though other google-* packages are.
            raise ModelProviderError(
                "Could not import google-genai (the Gemini SDK) in this environment. "
                f"Original error: {exc}. This usually means google-genai isn't actually "
                "installed here, even if the app otherwise starts fine — it's imported "
                "lazily, so this only surfaces on the first real Gemini call. Run "
                "`pip show google-genai` to check; if it's missing, `pip install "
                "google-genai==2.20.0` (see requirements.txt). On a hosted platform, this "
                "usually means the deployment's environment is stale — trigger a clean "
                "rebuild/redeploy so requirements.txt is reinstalled from scratch, rather "
                "than reusing a cached environment from before this dependency existed."
            ) from exc

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = model

    @staticmethod
    def _config(temperature: float, system: Optional[str]) -> dict[str, Any]:
        """The request config shared by every Gemini call in this class.

        `automatic_function_calling.disable` is deliberate. This codebase
        never passes tools to Gemini — every agent asks for text or JSON
        back — but google-genai turns AFC ON by default regardless, which
        routes each call through its function-calling loop and logs
        "Direct use of automatic function calling (AFC) in
        Models.generate_content is not recommended" once per process. That
        warning showed up in a user's terminal and read like a real
        problem; it isn't, but the honest fix is to turn off a feature we
        don't use rather than to filter the log line. Note the bare
        {"disable": True} with no maximum_remote_calls — setting both makes
        the SDK log a *different* warning about the combination."""
        return {
            "temperature": temperature,
            "automatic_function_calling": {"disable": True},
            **({"system_instruction": system} if system else {}),
        }

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.4) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config(temperature, system),
            )
        except Exception as exc:  # noqa: BLE001 - surface as our own error type
            raise ModelProviderError(f"Gemini generate() failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise ModelProviderError("Gemini returned an empty response")
        return text

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
        from google.genai import types  # noqa: WPS433

        json_instruction = (
            "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble, no commentary."
        )

        # Interleave a "Image N:" text label before each image so the model
        # can reliably correspond its per-image output back to input order,
        # then the prompt (with instructions) last.
        contents: list = []
        for idx, image_path in enumerate(image_paths, start=1):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            contents.append(f"Image {idx}:")
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        contents.append(prompt + json_instruction)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._config(temperature, system),
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(f"Gemini generate_json_with_images() failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise ModelProviderError("Gemini returned an empty response")
        return _extract_json(text)
