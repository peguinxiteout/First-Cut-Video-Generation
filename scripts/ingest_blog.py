#!/usr/bin/env python
"""
Usage:
    python scripts/ingest_blog.py \\
        --blog-id BLOG-001 \\
        --topic "5 tips for first-time home buyers" \\
        --tone "friendly and encouraging" \\
        --sections strengths weaknesses location_analysis conclusion \\
        --facts "Metro station 1.2 km away" "Listed at 11,000-13,000 per sq ft" \\
        --cta-text "Book a site visit" --cta-phone "+91 98100 00000" \\
        --target-word-count 900 \\
        --images "/path/to/some/images"   # optional

Writes data/blogs/<blog-id>/blog_input.json (+ copies any --images in).
Deliberately minimal — topic, tone, key points, and target length are the
only inputs the platform exposes (matches the Streamlit Blog Generation form).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import IMAGE_EXTENSIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blog-id", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--tone", default="")
    parser.add_argument("--sections", nargs="*", default=[],
                        help="Section ids from configs/blog/sections.yaml, e.g. strengths weaknesses")
    parser.add_argument("--facts", nargs="*", default=[],
                        help="Facts you can vouch for; the ONLY source of figures the writer may use")
    parser.add_argument("--cta-text", default="")
    parser.add_argument("--cta-url", default="")
    parser.add_argument("--cta-phone", default="")
    parser.add_argument("--key-points", nargs="*", default=[], help="(legacy) extra points to cover")
    parser.add_argument("--target-word-count", type=int, default=None)
    parser.add_argument("--images", default=None, help="Optional folder of images to copy in")
    args = parser.parse_args()

    blog_dir = PROJECT_ROOT / "data" / "blogs" / args.blog_id
    blog_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        source_dir = Path(args.images).expanduser().resolve()
        images_dir = blog_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img in sorted(source_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(img, images_dir / img.name)
                copied += 1
        print(f"Copied {copied} image(s) to {images_dir}")

    from src.agents.blog.brief import section_catalog

    unknown = [s for s in args.sections if s not in section_catalog()]
    if unknown:
        parser.error(f"Unknown section id(s): {', '.join(unknown)}. "
                     f"Valid: {', '.join(section_catalog())}")

    blog_input = {
        "blog_id": args.blog_id,
        "topic": args.topic,
        "sections": args.sections,
        "local_facts": args.facts,
        "cta_text": args.cta_text,
        "cta_url": args.cta_url,
        "cta_phone": args.cta_phone,
        "key_points": args.key_points,
        "tone": args.tone,
        "target_word_count": args.target_word_count,
    }
    blog_input_path = blog_dir / "blog_input.json"
    blog_input_path.write_text(json.dumps(blog_input, indent=2), encoding="utf-8")
    print(f"blog_input.json written to {blog_input_path}")
    print(f"Next: python scripts/generate_blog.py --blog-id {args.blog_id}")


if __name__ == "__main__":
    main()
