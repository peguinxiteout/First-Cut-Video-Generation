"""Visual enrichment (brochure): supporting visuals where the uploads leave
a real gap — and only there.

Gap analysis (src/agents/image/gap_analysis.py) decides what is missing —
a lifestyle scene, the neighbourhood's mood, an amenity the property lists
but has no photograph of — and the image service generates it. Each
generated visual joins the book as an ordinary picture that the planner
places like any other, under three rules it enforces itself
(src/agents/brochure/planner.py):

    * it is never the cover, the back page, the contents or the overview
    * it never depicts the property: the briefs only ask for mood and
      context (RERA's truthful-advertising rule), never a facade or a room
    * it is never captioned or labelled as an "artistic impression"

Soft by contract: no supporting visual is worth a failed brochure, and the
offline placeholder images the mock service produces are never placed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.agents.image import gap_analysis
from src.agents.image import generator as image_generator
from src.agents.image.generator import GenerationReport
from src.agents.image.prompting import derive_style
from src.core.types import ImageCategory
from src.data.schemas import ImageAnalysis, ImageAsset, PropertyInput, PropertyKnowledge
from src.models.image_gen import ImageService

logger = logging.getLogger(__name__)

# A design element is page furniture the planner has no use for; the other
# kinds are pictures a page can be built around.
_KIND_CATEGORY = {"lifestyle": ImageCategory.LIFESTYLE, "amenity_concept": ImageCategory.AMENITY,
                  "neighbourhood": ImageCategory.LIFESTYLE}
_SHAPES = ("wide", "portrait", "page")


def run(prop: PropertyInput, knowledge: Optional[PropertyKnowledge], images: list[ImageAsset],
        analyses: dict[str, ImageAnalysis], service: ImageService,
        out_dir: Path) -> tuple[list[ImageAsset], dict[str, ImageAnalysis], GenerationReport]:
    analysis_list = list(analyses.values())
    briefs = [b for b in gap_analysis.run(prop, knowledge, analysis_list) if b.kind in _KIND_CATEGORY]
    # Different shapes give the planner different pages to build.
    for index, brief in enumerate(briefs):
        brief.aspect = _SHAPES[index % len(_SHAPES)]
    from src.core.config import settings

    style = derive_style(analysis_list, region=prop.location.city if prop.location else "",
                         medium_override=settings.image_gen_medium)
    report = image_generator.run(briefs, style, service, out_dir)

    new_images = list(images)
    new_analyses = dict(analyses)
    taken = {img.filename for img in images}
    for index, visual in enumerate(report.visuals, start=1):
        path = Path(visual.path)
        filename = path.name if path.name not in taken else f"ai_{index:02d}_{path.name}"
        image_id = f"AI_{index:02d}"
        new_images.append(ImageAsset(image_id=image_id, filename=filename, path=str(path),
                                     is_ai_generated=True))
        subject = (visual.brief.subject or visual.brief.kind).lower()
        new_analyses[image_id] = ImageAnalysis(
            image_id=image_id, category=_KIND_CATEGORY[visual.brief.kind],
            quality=8, composition=8, hero_score=0,
            scene_tag=f"ai_{visual.brief.kind}_{subject}"[:60].replace(" ", "_"),
            notes="AI-generated supporting visual; not a depiction of the property.",
        )
    return new_images, new_analyses, report
