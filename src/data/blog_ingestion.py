from __future__ import annotations

import json
from pathlib import Path

from src.core.exceptions import IngestionError
from src.data.ingestion import IMAGE_EXTENSIONS
from src.data.schemas import BlogImageAsset, BlogInput


def load_blog_input(blog_dir: Path) -> BlogInput:
    blog_input_path = blog_dir / "blog_input.json"
    if not blog_input_path.exists():
        raise IngestionError(f"Missing blog_input.json in {blog_dir}")
    with blog_input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return BlogInput.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"blog_input.json failed validation: {exc}") from exc


def discover_blog_images(blog_dir: Path) -> list[BlogImageAsset]:
    """Finds every image under <blog_dir>/images/. Unlike brochures, a blog
    post can have zero images — that's valid, not an error."""
    images_dir = blog_dir / "images"
    if not images_dir.exists():
        return []

    assets: list[BlogImageAsset] = []
    for idx, path in enumerate(sorted(images_dir.iterdir()), start=1):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        assets.append(
            BlogImageAsset(image_id=f"BIMG{idx:03d}", filename=path.name, path=str(path.resolve()))
        )
    return assets
