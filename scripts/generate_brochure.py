#!/usr/bin/env python
"""Generate a brochure from a property folder.

    python scripts/generate_brochure.py --property-id sobha-neopolis
    python scripts/generate_brochure.py --property-id sobha-neopolis --template "Luxury - Portrait"
    python scripts/generate_brochure.py --property-id sobha-neopolis --orientation portrait

The template is one of the nine (affordable / mid / luxury x landscape /
portrait / digital), by id ("mid_portrait") or name ("Mid-Tier - Portrait"). Without
one, the property's own category is used with the chosen orientation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.brochure import catalog  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.data.ingestion import load_property_input  # noqa: E402
from src.workflows.brochure import run_brochure_workflow  # noqa: E402


def main() -> None:
    names = ", ".join(f"{tid} ({name})" for tid, name in catalog.choices())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--property-id", required=True)
    parser.add_argument("--template", default=None, help=f"One of: {names}")
    parser.add_argument("--orientation", default="landscape", choices=["landscape", "portrait", "digital"])
    args = parser.parse_args()

    property_dir = settings.data_dir_resolved / "properties" / args.property_id
    if not property_dir.exists():
        raise SystemExit(f"No property found at {property_dir}. Use the Streamlit form or "
                         f"scripts/ingest_property.py first.")
    prop = load_property_input(property_dir)
    spec = catalog.resolve(args.template, prop.category, args.orientation)
    record = run_brochure_workflow(property_dir, template_name=spec.id)

    print(f"Template: {spec.name}")
    print(f"Status:   {record.status}")
    if record.workflow_evaluation:
        for step in record.workflow_evaluation.steps:
            print(f"  [{'OK  ' if step.status == 'success' else 'FAIL'}] {step.node:<22} {step.latency_ms:>8.0f} ms")
    if record.status == "failed":
        print(f"Error: {record.error}")
        raise SystemExit(1)
    print(f"PDF:      {record.pdf_path}")
    if record.evaluation:
        from src.evaluation.tiers import quality_tier

        e = record.evaluation
        print(f"Quality:  {e.quality_score:.0f} ({quality_tier(e.quality_score)}, gate {e.gate_result.value}, "
              f"judged by {e.judge})")
        for check, result in e.checks.items():
            print(f"  {check:<16} {result}")


if __name__ == "__main__":
    main()
