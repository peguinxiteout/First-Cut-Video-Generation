#!/usr/bin/env python
"""Create a property folder from the command line — the same folder the
Streamlit form writes.

    python scripts/ingest_property.py \\
        --source ./photos --property-id green-acres --name "Green Acres" \\
        --city Pune --category Mid-Tier --config "2 BHK|680 sq ft" --config "3 BHK|940 sq ft" \\
        --phone "+91 98765 43210" --floorplan plans/2bhk.png --masterplan site.png

Photographs come from --source; floor plans, the master plan, the location
map and the logo are passed one by one so each is known for what it is.
Everything else in property.json can be edited afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings  # noqa: E402
from src.data import property_form  # noqa: E402
from src.data.ingestion import IMAGE_EXTENSIONS  # noqa: E402


def _files(paths) -> list[tuple[str, bytes]]:
    out = []
    for raw in paths or []:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        out.append((path.name, path.read_bytes()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder of project photographs")
    parser.add_argument("--property-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--locality", default="")
    parser.add_argument("--category", required=True, choices=property_form.CATEGORIES)
    parser.add_argument("--config", action="append", default=[],
                        help='"2 BHK|680 sq ft|Rs 52 Lakh" — repeat for each configuration')
    parser.add_argument("--price", default="")
    parser.add_argument("--developer", default="")
    parser.add_argument("--phone", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--website", default="")
    parser.add_argument("--floorplan", action="append", default=[])
    parser.add_argument("--masterplan", action="append", default=[])
    parser.add_argument("--map", action="append", default=[])
    parser.add_argument("--logo", action="append", default=[])
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    photos = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS] if source.is_dir() else []
    rows = []
    for raw in args.config:
        parts = [p.strip() for p in raw.split("|")] + ["", ""]
        rows.append(property_form.ConfigRow(configuration=parts[0], carpet_area=parts[1], price=parts[2]))
    form = property_form.BrochureForm(name=args.name, city=args.city, locality=args.locality,
                                      category=args.category, configurations=rows, price=args.price,
                                      developer=args.developer, phone=args.phone, email=args.email,
                                      website=args.website)
    gaps = property_form.missing(form, len(photos))
    if gaps:
        raise SystemExit("Missing: " + ", ".join(gaps))
    uploads = {"photo": _files(photos), "floorplan": _files(args.floorplan),
               "masterplan": _files(args.masterplan), "map": _files(args.map), "logo": _files(args.logo)}
    folder = property_form.write_property(form, uploads, settings.data_dir_resolved / "properties",
                                          property_id=args.property_id)
    print(f"Wrote {folder}")
    print(f"Next: python scripts/generate_brochure.py --property-id {args.property_id}")


if __name__ == "__main__":
    main()
