"""Per-model generation parameters.

The original test script sent every model the same settings (30 steps +
a negative prompt). That's wrong for FLUX.1-schnell: it is a timestep-
distilled model built for ~4 steps and does not use a negative prompt, so
30 steps costs ~7x the time/credits for no quality gain. Each model gets
parameters that match how it was trained.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    steps: int
    guidance_scale: float | None
    supports_negative_prompt: bool
    size_multiple: int          # width/height must be a multiple of this
    max_pixels: int             # total pixel budget the model handles well
    max_side: int


PROFILES: dict[str, ModelProfile] = {
    "black-forest-labs/FLUX.1-schnell": ModelProfile(
        model_id="black-forest-labs/FLUX.1-schnell", steps=4, guidance_scale=None,
        supports_negative_prompt=False, size_multiple=16, max_pixels=1_400_000, max_side=1440,
    ),
    "black-forest-labs/FLUX.1-dev": ModelProfile(
        model_id="black-forest-labs/FLUX.1-dev", steps=28, guidance_scale=3.5,
        supports_negative_prompt=False, size_multiple=16, max_pixels=1_400_000, max_side=1440,
    ),
    "stabilityai/stable-diffusion-xl-base-1.0": ModelProfile(
        model_id="stabilityai/stable-diffusion-xl-base-1.0", steps=30, guidance_scale=6.5,
        supports_negative_prompt=True, size_multiple=8, max_pixels=1_050_000, max_side=1536,
    ),
    "stabilityai/stable-diffusion-2-1": ModelProfile(
        model_id="stabilityai/stable-diffusion-2-1", steps=30, guidance_scale=7.0,
        supports_negative_prompt=True, size_multiple=8, max_pixels=600_000, max_side=768,
    ),

    # Cloudflare Workers AI serves the same weights under its own ids, but
    # with its own limits: SDXL is capped at 20 steps there and rejects
    # more, and flux-1-schnell has no width/height in its schema at all
    # (the provider routes non-square requests away from it).
    "@cf/black-forest-labs/flux-1-schnell": ModelProfile(
        model_id="@cf/black-forest-labs/flux-1-schnell", steps=4, guidance_scale=None,
        supports_negative_prompt=False, size_multiple=16, max_pixels=1_100_000, max_side=1024,
    ),
    "@cf/stabilityai/stable-diffusion-xl-base-1.0": ModelProfile(
        model_id="@cf/stabilityai/stable-diffusion-xl-base-1.0", steps=20, guidance_scale=7.5,
        supports_negative_prompt=True, size_multiple=8, max_pixels=1_050_000, max_side=1536,
    ),
    "@cf/stabilityai/stable-diffusion-xl-lightning": ModelProfile(
        model_id="@cf/stabilityai/stable-diffusion-xl-lightning", steps=8, guidance_scale=2.0,
        supports_negative_prompt=True, size_multiple=8, max_pixels=1_050_000, max_side=1536,
    ),
    "@cf/leonardo/phoenix-1.0": ModelProfile(
        model_id="@cf/leonardo/phoenix-1.0", steps=25, guidance_scale=2.0,
        supports_negative_prompt=True, size_multiple=8, max_pixels=1_050_000, max_side=2048,
    ),
}

# Conservative defaults for a model id we have no profile for.
_DEFAULT = dict(steps=28, guidance_scale=6.0, supports_negative_prompt=True,
                size_multiple=16, max_pixels=1_000_000, max_side=1280)


def profile_for(model_id: str) -> ModelProfile:
    if model_id in PROFILES:
        return PROFILES[model_id]
    return ModelProfile(model_id=model_id, **_DEFAULT)


def fit_dimensions(width: int, height: int, profile: ModelProfile) -> tuple[int, int]:
    """Scale a requested size into the model's comfortable range while
    preserving the aspect ratio, then snap to the model's size multiple.

    The aspect ratio is what matters to the layout (a 16:9 blog hero, a
    3:4 portrait slot) — the absolute size is only a budget. Generating
    at the model's native scale and letting the page scale the image is
    better than asking a 1MP model for a 4MP image."""
    width, height = max(1, int(width)), max(1, int(height))
    scale = min(1.0, (profile.max_pixels / (width * height)) ** 0.5,
                profile.max_side / max(width, height))
    w, h = width * scale, height * scale
    m = profile.size_multiple
    w = max(m * 16, int(round(w / m)) * m)
    h = max(m * 16, int(round(h / m)) * m)
    return w, h
