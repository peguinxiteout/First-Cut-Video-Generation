"""Cached access to the blog YAML configs (configs/blog/*.yaml).

`load_yaml_config` reads from disk on every call; the normaliser and the
rubric consult these lexicons hundreds of times per article, so they are
read once per process here."""
from __future__ import annotations

from functools import lru_cache

from src.core.config import load_yaml_config


@lru_cache(maxsize=1)
def editorial() -> dict:
    return load_yaml_config("blog/editorial.yaml")


@lru_cache(maxsize=1)
def archetypes() -> dict:
    return load_yaml_config("blog/archetypes.yaml")


@lru_cache(maxsize=1)
def rubric() -> dict:
    return load_yaml_config("blog/evaluation.yaml")


@lru_cache(maxsize=1)
def sections() -> dict:
    return load_yaml_config("blog/sections.yaml")["sections"]


def limits() -> dict:
    return rubric().get("limits", {})


def clear() -> None:
    for fn in (editorial, archetypes, rubric, sections):
        fn.cache_clear()
