"""Synthetic properties for the video tests: a real property folder as the
Streamlit form writes it (tests/brochure/helpers.py), with a logo, and the
helpers to run the plan half on it."""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.video import assets as asset_audit
from src.agents.video import brief as brief_agent
from src.agents.video import fact_ledger
from src.agents.video import script as script_agent
from src.data import property_form
from tests.brochure import helpers as brochure


def make_video_property(root: Path, category: str = "Mid-Tier", photos: int = len(brochure.PHOTOS),
                        logo: bool = True, video_input: dict | None = None, **extra) -> Path:
    """A property folder with `photos` photographs, a floor plan and a logo."""
    uploads = {"photo": [(name, brochure.photo_bytes(size, colour, i))
                         for i, (name, size, colour) in enumerate(brochure.PHOTOS[:photos])],
               "floorplan": [("plan_0.png", brochure.plan_bytes())]}
    if logo:
        uploads["logo"] = [("developer_logo.png", brochure.plan_bytes((400, 400)))]
    folder = property_form.write_property(brochure.form(category, **extra), uploads, root)
    if video_input is not None:
        (folder / "video_input.json").write_text(json.dumps(video_input), encoding="utf-8")
    return folder


class Plan:
    """The first four nodes' results, built the way the workflow builds them
    (offline), for tests that need to start from a brief, a ledger, assets and a script."""

    def __init__(self, folder: Path, overrides: dict | None = None, draft: bool = False):
        loaded = brief_agent.run(folder, overrides, draft=draft)
        self.loaded = loaded
        self.prop, self.brief, self.images = loaded.property_input, loaded.brief, loaded.images
        self.knowledge, _ = fact_ledger.run(self.prop, loaded.documents_text, offline=True)
        self.analyses, self.assets, self.image_plan, self.asset_notes = asset_audit.run(
            self.images, offline=True, photo_shots=self.brief.n_shots - 1)
        self.result = script_agent.run(self.prop, self.brief, self.knowledge, self.assets)
        self.script = self.result.script
