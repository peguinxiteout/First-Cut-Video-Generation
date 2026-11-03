"""Image Generation Agent — the one place briefs become images.

Used by both pipelines:
  brochure: gap_analysis briefs  -> generator.run -> layout
  blog:     visual_director brief -> generator.run -> hero image

Failures are soft by design. One image failing (rate limit, a model
being retired, no token) must never fail a brochure or blog — the output
is produced with whatever images did succeed, and the reason is recorded
so the UI and MLflow can show it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.agents.image.prompting import NEGATIVE_PROMPT, StyleProfile, build_prompt, is_allowed, size_for
from src.data.schemas import GeneratedVisual, ImageBrief
from src.models.image_gen import ImageGenerationAuthError, ImageGenerationError, ImageRequest, ImageService

logger = logging.getLogger(__name__)


@dataclass
class GenerationReport:
    visuals: list[GeneratedVisual] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # human-readable reasons
    provider: str = ""
    is_mock: bool = False


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "image"


def run(briefs: list[ImageBrief], style: StyleProfile, service: ImageService,
        out_dir: Path, seed: Optional[int] = None) -> GenerationReport:
    report = GenerationReport(provider=getattr(service.provider, "name", "none"), is_mock=service.is_mock)
    if not service.enabled:
        report.skipped.append("Image generation is not configured (set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env to enable).")
        return report

    for index, brief in enumerate(briefs):
        if not is_allowed(brief):
            # Defence in depth: gap_analysis already filters, but nothing
            # that depicts the actual property is ever generated.
            report.skipped.append(f"Rejected brief '{brief.subject[:50]}': would depict the property.")
            continue
        width, height = size_for(brief.aspect)
        prompt = build_prompt(brief, style)
        dest = out_dir / f"ai_{index + 1:02d}_{brief.kind}_{_slug(brief.subject)}.png"
        try:
            result = service.generate(
                ImageRequest(prompt=prompt, width=width, height=height,
                             negative_prompt=NEGATIVE_PROMPT, seed=seed),
                dest_path=dest,
            )
        except ImageGenerationAuthError as exc:
            report.skipped.append(str(exc))
            break  # every later image would fail the same way
        except ImageGenerationError as exc:
            logger.warning("Image %d (%s) failed: %s", index + 1, brief.kind, exc)
            report.skipped.append(f"{brief.kind}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - see below
            # The backstop, and it earned its place. A generated image is an
            # ENHANCEMENT: the brochure is built from the person's own
            # photographs and finishes perfectly well without one. Every
            # layer below is supposed to guarantee that, and one of them
            # did not — a `requests` read timeout is not an
            # ImageGenerationError, so it walked past the handler above and
            # took down a whole generation with
            #
            #     1 step(s) failed: visual_enrichment
            #
            # That specific hole is fixed at source (see
            # src/models/image_gen/transport.py). This is here so the NEXT
            # one — a provider SDK raising its own class, a JSON decode
            # error, a disk that fills — costs the person a missing
            # illustration and a line in the report instead of their
            # brochure. It is deliberately wide for that reason, and it
            # logs the traceback so the hole is still visible.
            logger.exception("Image %d (%s) failed unexpectedly", index + 1, brief.kind)
            report.skipped.append(
                f"{brief.kind}: image generation failed unexpectedly ({type(exc).__name__}). "
                "The brochure was produced from the uploaded photographs.")
            continue
        report.visuals.append(GeneratedVisual(
            brief=brief, path=str(result.path), model_id=result.model_id, prompt=prompt,
            width=result.width, height=result.height, from_cache=result.from_cache,
        ))
    return report
