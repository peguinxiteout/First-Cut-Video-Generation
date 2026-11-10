"""The words sent with a keyframe to an image-to-video model.

The model is asked for camera movement and nothing else: "the scene stays
exactly as photographed". A model with a negative field (Kling) also gets the
list of what must not appear; one without (MiniMax H3 Max) gets positive
wording only, because a prompt that says "no people" to such a model puts
people in. The style is the model's, from providers.yaml (params.negative_prompt),
and the words are in configs/video/motion_prompts.yaml.

A transition is one continuous move between two real views (a first and a last
frame); its prompt says both places stay as photographed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from string import Template
from typing import Optional

from src.core.config import load_yaml_config
from src.data.video_schemas import CAMERA_MOVES

CONFIG_FILE = "video/motion_prompts.yaml"


@dataclass(frozen=True)
class MotionPrompt:
    prompt: str
    negative_prompt: str = ""
    style: str = "positive_only"


def _cfg() -> dict:
    return load_yaml_config(CONFIG_FILE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def negations(text: str) -> list[str]:
    """The negation words of the config found in `text` (whole words, any case)."""
    words = re.findall(r"[a-z']+", text.lower())
    banned = {w.lower() for w in _cfg()["negation_words"]}
    return [w for w in words if w in banned]


def build(camera: str = "push_in", *, takes_negative: bool = False, scene: str = "any",
          transition: bool = False, subject: Optional[str] = None) -> MotionPrompt:
    """The prompt for one shot (or a transition). Raises ValueError for an unknown
    camera move, and for a positive-only prompt that would carry a negation."""
    cfg = _cfg()
    if transition:
        text = _clean(cfg["transition"])
    else:
        if camera not in CAMERA_MOVES:
            raise ValueError(f"unknown camera move {camera!r}; use one of {list(CAMERA_MOVES)}")
        subject = subject or cfg["subjects"].get(scene, cfg["subjects"]["any"])
        move = Template(cfg["camera"][camera]).substitute(subject=subject)
        text = _clean(Template(cfg["shot"]).substitute(move=move))
    if takes_negative:
        return MotionPrompt(prompt=text, negative_prompt=_clean(cfg["negative"]), style="with_negative")
    found = negations(text)
    if found:
        raise ValueError(f"a positive-only prompt carries negation words {found}: {text!r}")
    return MotionPrompt(prompt=text, style="positive_only")


def for_provider(params: dict, camera: str = "push_in", **kwargs) -> MotionPrompt:
    """build() in the style a providers.yaml entry's params ask for."""
    return build(camera, takes_negative=bool((params or {}).get("negative_prompt")), **kwargs)
