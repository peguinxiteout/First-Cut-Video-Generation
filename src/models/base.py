"""
Every model provider (Gemini, OpenRouter, future Ollama) implements this same
interface. Agents only ever call methods on an LLMProvider — never a
provider-specific SDK directly — so swapping providers is a config change,
not a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMProvider(ABC):
    """Provider-agnostic text/vision generation interface."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.4) -> str:
        """Plain text generation. Returns the raw text response."""

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Generation where the model is instructed to return JSON. The
        provider is responsible for parsing it into a dict; raises
        ModelOutputParseError if parsing fails after retry."""

    @abstractmethod
    def generate_json_with_image(
        self,
        prompt: str,
        image_path: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Vision + JSON generation for a single local image file."""

    @abstractmethod
    def generate_json_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Vision + JSON generation for MULTIPLE local image files in a
        single API call. Used by the Vision Agent to classify every photo
        in one request instead of one-per-photo — free-tier daily quotas
        can be as low as ~20 requests/day for a given model, and a dozen
        individual per-image calls would alone consume most of that budget
        for a single brochure generation."""
