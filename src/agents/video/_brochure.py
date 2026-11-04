"""The brochure's private helpers the video agents reuse, imported in one place.

They are underscore-named in their own modules (copywriter, image_plan) and
are reused here rather than copied. If the brochure renames or changes one,
this is the only video file that has to follow."""
from __future__ import annotations

from src.agents.brochure.copywriter import _short_item as short_item
from src.agents.brochure.image_plan import _score as photo_score

__all__ = ["photo_score", "short_item"]
