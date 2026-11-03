"""Smoke-test AI image generation end to end.

    python scripts/test_image_generation.py                # uses .env settings
    python scripts/test_image_generation.py --mock         # offline, no token needed
    python scripts/test_image_generation.py --model stabilityai/stable-diffusion-xl-base-1.0

Generates one blog-hero-style image and one brochure lifestyle image
through the SAME code path the pipelines use (brief -> prompt -> service
-> provider), and prints which model answered and where the files are.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.image import generator  # noqa: E402
from src.agents.image.prompting import StyleProfile  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.data.schemas import ImageBrief  # noqa: E402
from src.models.image_gen import get_image_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="use the offline mock provider")
    parser.add_argument("--model", help="override the model chain with a single model id")
    parser.add_argument("--out", default="data/image_smoke_test")
    args = parser.parse_args()

    if args.mock:
        settings.image_gen_provider = "mock"
    if args.model:
        settings.image_gen_models = args.model

    service = get_image_service()
    print(f"Provider : {getattr(service.provider, 'name', 'none')}  (mock={service.is_mock})")
    print(f"Models   : {settings.image_gen_models}")
    if not service.enabled:
        print("Image generation is disabled. Set cloudflare_account_id and cloudflare_api_token in .env, or pass --mock.")
        return 1
    if service.is_mock and not args.mock:
        print("NOTE: no image credentials are set, so this run uses the offline mock provider.")

    briefs = [
        ImageBrief(kind="blog_hero", aspect="wide", purpose="smoke test",
                   subject="a quiet residential street lined with mature trees at golden hour"),
        ImageBrief(kind="lifestyle", aspect="landscape", purpose="smoke test",
                   subject="a couple sharing coffee on a spacious private terrace, seen from behind"),
    ]
    style = StyleProfile(materials=["brick", "oak"], palette=["warm"], light=["golden hour"], region="Noida")

    started = time.time()
    report = generator.run(briefs, style, service, Path(args.out))
    print(f"\nFinished in {time.time() - started:.1f}s")
    for v in report.visuals:
        cached = " (cached)" if v.from_cache else ""
        print(f"  OK  {v.brief.kind:10s} {v.width}x{v.height}  {v.model_id}{cached}\n      -> {v.path}")
    for reason in report.skipped:
        print(f"  SKIPPED  {reason}")
    return 0 if report.visuals else 2


if __name__ == "__main__":
    raise SystemExit(main())
