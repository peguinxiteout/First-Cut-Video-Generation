#!/usr/bin/env python
"""
Usage:
    python scripts/generate_blog.py --blog-id BLOG-001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings  # noqa: E402
from src.workflows.blog import run_blog_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blog-id", required=True)
    args = parser.parse_args()

    blog_dir = settings.data_dir_resolved / "blogs" / args.blog_id
    if not blog_dir.exists():
        raise SystemExit(f"No blog found at {blog_dir}. Run scripts/ingest_blog.py first.")

    record = run_blog_workflow(blog_dir)

    print(f"Status: {record.status}")
    print("\n--- Workflow Evaluation ---")
    if record.workflow_evaluation and record.workflow_evaluation.steps:
        for step in record.workflow_evaluation.steps:
            icon = "OK  " if step.status == "success" else "FAIL"
            print(f"  [{icon}] {step.node:<12} {step.latency_ms:>8.0f} ms")
            if step.error:
                print(f"         -> {step.error}")

    if record.status == "failed":
        print(f"\nError: {record.error}")
        raise SystemExit(1)

    print(f"\nGeneration ID: {record.generation_id}")
    print(f"Markdown: {record.markdown_path}")
    print(f"HTML preview: {record.html_path}")

    if record.evaluation:
        from src.evaluation.tiers import quality_tier

        e = record.evaluation
        print(f"\nQuality score: {e.quality_score} ({quality_tier(e.quality_score)}, gate: {e.gate_result.value})")
        print(f"  grounding={e.grounding} writing_quality={e.writing_quality} "
              f"structural_completeness={e.structural_completeness}")


if __name__ == "__main__":
    main()
