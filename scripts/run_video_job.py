#!/usr/bin/env python
"""One background video job: the plan, the production of a planned video, or both.

    python scripts/run_video_job.py plan vid_0123456789
    python scripts/run_video_job.py produce vid_0123456789
    python scripts/run_video_job.py full vid_0123456789     plan and produce, no stop for approval

Started by src/workflows/video_jobs.py, not by hand. Everything the job needs
is read from the record in the database (what was asked for, including
--offline and draft) and, for produce, from the video's plan_state.json; the
job saves its progress in the database after every node and its final result
in the record. Output goes to <video folder>/work/job.log.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_video import force_offline  # noqa: E402
from src.data import video_repository as repository  # noqa: E402
from src.data.video_schemas import ACTIVE_STATUSES, VideoRequest  # noqa: E402

logger = logging.getLogger("run_video_job")


def _fail(video_id: str, reason: str) -> None:
    """Records a job that stopped before the workflow could, if it is still running."""
    record = repository.get_generation(video_id)
    if record is not None and record.status in (*ACTIVE_STATUSES, "running"):
        record.status, record.error = "failed", reason
        repository.save_generation(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one background video job.")
    parser.add_argument("mode", choices=["plan", "produce", "full"])
    parser.add_argument("video_id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    repository.init_db()
    record = repository.get_generation(args.video_id)
    if record is None:
        logger.error("There is no video %s", args.video_id)
        return 2
    request = record.request or VideoRequest(property_id=record.property_id)
    if request.offline:
        force_offline()

    # Imported after force_offline(): nothing here may read a key before it is blanked.
    from src.workflows import video_jobs
    from src.workflows.video import run_plan_workflow, run_produce_workflow, run_video_workflow

    try:
        if args.mode in ("plan", "full"):
            run = run_plan_workflow if args.mode == "plan" else run_video_workflow
            record = run(video_jobs.resolve_property_dir(request.property_id), request.overrides,
                         offline=request.offline, draft=request.draft, refresh_ledger=request.refresh_ledger,
                         video_id=args.video_id)
        else:
            record = run_produce_workflow(args.video_id)
    except Exception as exc:  # noqa: BLE001 - the workflow saves its own failures; this is for what came before it
        logger.exception("The %s job of %s failed", args.mode, args.video_id)
        _fail(args.video_id, str(exc))
        return 1
    logger.info("%s job of %s finished: %s", args.mode, args.video_id, record.status)
    return 1 if record.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
