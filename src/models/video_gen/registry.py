"""Adapter id -> class. A providers.yaml entry names an adapter here; one
adapter can serve several entries (the fal adapter will serve every model
that goes through fal, each entry carrying its own endpoint and params).

To add a vendor: write a VideoProvider subclass taking (spec, api_key) and
add it below. To add a model on a vendor that already has an adapter, add a
YAML entry and nothing else."""
from __future__ import annotations

from typing import Type

from src.models.video_gen.base import VideoProvider
from src.models.video_gen.kenburns import KenBurnsProvider
from src.models.video_gen.mock import MockVideoProvider

ADAPTERS: dict[str, Type[VideoProvider]] = {
    "kenburns": KenBurnsProvider,
    "mock": MockVideoProvider,
}
