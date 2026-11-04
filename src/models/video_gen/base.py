"""Provider-agnostic image-to-video interface.

Architecture:  motion node -> VideoService -> VideoProvider -> concrete model.
Nothing outside src/models/video_gen knows which model or vendor moved a
photograph, so adding one is an adapter and a YAML entry, not a change to the
workflow. A provider only ADDS CAMERA MOTION to a still it is given; it is
never asked to invent the property.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.data.video_schemas import CAMERA_MOVES


class VideoGenerationError(Exception):
    """Raised when a provider could not make a clip. The caller treats it as
    a soft failure: the shot is filled by pan-and-zoom and the reason is
    recorded."""


class VideoGenerationAuthError(VideoGenerationError):
    """Key missing or rejected. Every shot would fail the same way, so the
    provider is dropped for the rest of the run."""


class VideoGenerationQuotaError(VideoGenerationError):
    """The account's allowance or balance is spent. Dropped for the rest of
    the run and remembered across runs for a while (see selector.mark_quota),
    because the next run would hit the same wall."""


class VideoGenerationTransportError(VideoGenerationError):
    """No answer: a timeout, a dropped connection, a 5xx. This shot falls
    back; the provider is tried again on the next one."""


class VideoContentRefusedError(VideoGenerationError):
    """The vendor's guardrails declined this picture or prompt. A different
    vendor may differ, so it is not a reason to drop the provider."""


@dataclass(frozen=True)
class ProviderSpec:
    """One entry of configs/video/providers.yaml."""
    id: str
    label: str
    adapter: str
    key_env: Optional[str] = None
    auto_priority: Optional[int] = None
    cost_per_second: Optional[float] = None
    max_duration_s: float = 8.0
    supports_first_last_frame: bool = False
    hidden: bool = False
    endpoint: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class ImageToVideoRequest:
    image_path: Path
    end_image_path: Optional[Path] = None        # a closing keyframe, for providers that take one
    prompt: str = ""                             # the motion, in words
    camera: str = "push_in"                      # one of CAMERA_MOVES
    duration_s: float = 3.0
    width: int = 1080
    height: int = 1920
    fps: int = 25
    focal: str = "50% 50%"                       # where the subject sits, as focal.focal_point returns it
    seed: Optional[int] = None


@dataclass
class VideoClip:
    path: Path
    provider: str
    model_id: str
    duration_s: float
    width: int
    height: int
    fps: int
    from_cache: bool = False
    cost_usd: float = 0.0
    # Why the provider that was asked for is not the one that answered.
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class VideoProvider(ABC):
    """One vendor or local method. Built from its ProviderSpec, so limits and
    prices are read from YAML and never repeated in code."""

    id: str = "base"
    model_id: str = ""
    max_duration_s: float = 8.0
    supports_first_last_frame: bool = False
    # False for providers whose clip costs nothing to make again: caching
    # them would spend disk to save less time than reading the file takes.
    cacheable: bool = True

    def __init__(self, spec: Optional[ProviderSpec] = None, api_key: str = ""):
        self.spec = spec
        self._api_key = (api_key or "").strip()
        if spec is not None:
            self.id = spec.id
            self.max_duration_s = spec.max_duration_s
            self.supports_first_last_frame = spec.supports_first_last_frame

    def is_available(self) -> bool:
        """A provider whose entry names a key is available only with one."""
        if self.spec is not None and self.spec.key_env:
            return bool(self._api_key)
        return True

    def estimate_cost(self, seconds: float) -> Optional[float]:
        """USD for `seconds` of video, or None when the price is not set."""
        rate = self.spec.cost_per_second if self.spec is not None else 0.0
        return None if rate is None else round(rate * seconds, 4)

    @abstractmethod
    def generate(self, request: ImageToVideoRequest, output_path: Path) -> VideoClip:
        """Make one clip and write it to output_path (MP4)."""


def check_request(request: ImageToVideoRequest) -> None:
    """The faults no provider can work around, as one readable error."""
    if request.camera not in CAMERA_MOVES:
        raise VideoGenerationError(f"unknown camera move {request.camera!r}; use one of {list(CAMERA_MOVES)}")
    for picture in (request.image_path, request.end_image_path):
        if picture is not None and not Path(picture).is_file():
            raise VideoGenerationError(f"the input picture does not exist: {picture}")
    if request.duration_s <= 0 or request.width <= 0 or request.height <= 0 or request.fps <= 0:
        raise VideoGenerationError("duration, size and fps must be positive")
