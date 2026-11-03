"""Reading a property folder: property.json and the images (with the
slot each was uploaded in, from images/roles.json)."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.exceptions import IngestionError
from src.data.schemas import ImageAsset, PropertyInput

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def load_property_input(property_dir: Path) -> PropertyInput:
    property_json_path = property_dir / "property.json"
    if not property_json_path.exists():
        raise IngestionError(
            f"Missing property.json in {property_dir}. Use the Streamlit form or "
            f"scripts/ingest_property.py first."
        )
    with property_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return PropertyInput.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"property.json failed validation: {exc}") from exc


ROLES_FILE = "roles.json"
VALID_ROLES = ("photo", "floorplan", "masterplan", "map", "logo")


def discover_images(property_dir: Path) -> list[ImageAsset]:
    """Every image under <property_dir>/images/.

    The brochure form uploads each kind of picture into its own slot —
    project photos, floor plans, the master plan, the location map, the
    developer's logo — and records which was which in images/roles.json
    ({"filename": "floorplan", ...}). That is a stronger signal than any
    classifier, so it travels with the image as `ImageAsset.role`. Images
    with no recorded role are left for the Vision Agent to identify."""
    images_dir = property_dir / "images"
    if not images_dir.exists():
        raise IngestionError(f"Missing images/ folder in {property_dir}")

    roles: dict[str, str] = {}
    subjects: dict[str, str] = {}
    roles_path = images_dir / ROLES_FILE
    if roles_path.exists():
        try:
            raw = json.loads(roles_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        for key, value in (raw or {}).items():
            # "floorplan", or "photo:pool" — a photograph and what it shows
            role, _, subject = str(value).strip().lower().partition(":")
            if role in VALID_ROLES:
                roles[str(key)] = role
            if subject:
                subjects[str(key)] = subject.strip()

    assets: list[ImageAsset] = []
    for idx, path in enumerate(sorted(images_dir.iterdir()), start=1):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        assets.append(
            ImageAsset(image_id=f"IMG{idx:03d}", filename=path.name, path=str(path.resolve()),
                       role=roles.get(path.name, ""), subject=subjects.get(path.name, ""))
        )

    if not assets:
        raise IngestionError(f"No images found in {images_dir}")

    return assets
