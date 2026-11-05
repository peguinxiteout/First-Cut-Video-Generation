#!/usr/bin/env python
"""Plan, or plan and render, a promotional video from a property folder.

    python scripts/generate_video.py --property-id ANAMIKA-HIGH-POINT --plan-only
    python scripts/generate_video.py --property-id ANAMIKA-HIGH-POINT --duration 15 --aspect 9:16 --offline
    python scripts/generate_video.py --property-id ANAMIKA-HIGH-POINT --offline --draft

Without --plan-only the video is made: voice, motion, overlays and compose,
into data/generated_videos/<video_id>/ (video.mp4, thumbnail.jpg, captions.srt,
script.json, storyboard.json, with the clips, audio and overlays beside them),
and a record with the status "rendered" is saved.

--plan-only stops after the script and the storyboard: it writes script.json
and storyboard.json and saves a record with the status "planned".

--draft renders small and fast (540x960 at 12 fps for 9:16) to look at a plan
before spending the time on the full size.

--offline takes every fallback and makes no network call, whatever keys
.env holds: the keys are blanked in this process, the fact ledger is built
from the form, pictures are read from the uploads, narration would come from
a voice on this machine and motion from pan-and-zoom.

Without --offline a run reaches only what configs/video/policy.yaml allows
(Gemini: vision, Lyria; Google Cloud TTS: the voice; Beatoven: the music). Any call that would go elsewhere is refused
and listed under "Blocked" at the end.

Options not given here are read from data/properties/<id>/video_input.json
(duration_s, aspect, audience, objective, highlights, tone, voice,
music_mood, model, cta, creative_direction, allow_generated, hero_image,
burn_captions, grade, pronunciations); without that file the video is 15 s,
9:16, with burned captions and the light grade.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings  # noqa: E402
from src.core.ids import brief_dir  # noqa: E402
from src.data.video_schemas import VIDEO_ASPECTS, VIDEO_DURATIONS  # noqa: E402

# Settings that reach a model, a vendor or the network. --offline blanks them
# in this process, so nothing later in the run can use them.
_ONLINE_SETTINGS = ("gemini_api_key", "openrouter_api_key", "hf_token", "cloudflare_account_id",
                    "cloudflare_api_token", "tavily_api_key", "google_cloud_tts_api_key", "beatoven_api_key")


def force_offline() -> None:
    for name in _ONLINE_SETTINGS:
        setattr(settings, name, "")
    settings.image_gen_provider = "mock"
    settings.tools_enabled = False
    settings.speech_provider = "offline"
    settings.video_gen_provider = "kenburns"
    settings.mlflow_enabled = False          # an offline run leaves no trace in mlflow.db


def print_plan(record) -> None:
    script, board = record.script, record.storyboard
    print("\n--- Script ---")
    for i, line in enumerate(script.lines):
        print(f"  {i}  {line.role:<8} {line.words:>2} words  {line.text}")
    print(f"  {len(script.lines)} lines, {script.word_count} words "
          f"(target {record.brief.target_words}, {script.mode} script)")
    print("\n--- On screen ---")
    print(f"  title: {' | '.join(script.on_screen)}")
    print(f"  chips: {' | '.join(script.chips) or '(none)'}")
    print(f"  call to action: {script.cta}")
    print(f"  end card: {' | '.join(script.end_card) or '(none)'}")
    print("\n--- Storyboard ---")
    print(f"  {'shot':<5}{'kind':<6}{'photo':<24}{'shows':<12}{'camera':<10}{'secs':>5}  lines  reuse")
    for shot in board.shots:
        photo = Path(shot.source_path).name if shot.source_path else "-"
        print(f"  {shot.index + 1:<5}{shot.kind:<6}{photo[:22]:<24}{shot.subject[:10]:<12}{shot.camera:<10}"
              f"{shot.seconds:>5.1f}  {','.join(map(str, shot.line_ids)) or '-':<6} {'yes' if shot.reuse else ''}")
    print(f"  total {board.total_seconds} s, {board.planner_mode} planner")
    print("\n--- What each line plays over ---")
    for shot in board.shots:
        if shot.kind != "real":
            continue
        said = " / ".join(script.lines[i].text for i in shot.line_ids) or "-"
        print(f"  {shot.index + 1}  {Path(shot.source_path).name}{' (shown again)' if shot.reuse else ''}")
        print(f"       says:   {said}")
        print(f"       shows:  {shot.caption or '(no vision reading)'}")
        print(f"       tags:   {', '.join(shot.tags) or '-'}")


def music_line(info) -> str:
    if info is None or info.provider == "none":
        return "none" + (f" ({'; '.join(info.notes)})" if info and info.notes else "")
    where = info.title or info.model_id
    cost = "cached, $0.00" if info.from_cache else f"${info.cost_usd:.2f}"
    return f"{info.provider} {where} ({cost}){'; licence: ' + info.licence if info.licence else ''}"


def print_result(record, wall_s: float) -> None:
    m = record.measurements
    print("\n--- Result ---")
    print(f"  video:     {record.video_path}")
    print(f"  thumbnail: {record.thumbnail_path}")
    print(f"  captions:  {record.captions_path}")
    if m:
        print(f"  probe:     {m.vcodec}/{m.acodec or 'no audio'}, {m.width}x{m.height}, {m.fps:g} fps, "
              f"{m.duration_s:.2f} s (asked for {record.duration_s} s), {m.size_bytes / 1e6:.2f} MB")
        if m.loudness_lufs is not None:
            peak = f", true peak {m.true_peak_dbfs:.1f} dBFS" if m.true_peak_dbfs is not None else ""
            print(f"  loudness:  {m.loudness_lufs:.1f} LUFS{peak}")
    print(f"  voice:     {record.voice or '-'}")
    print(f"  music:     {music_line(record.music_info)}")
    print(f"  clips:     {', '.join(f'{c.shot_index + 1}:{c.provider or c.mode}' for c in record.clips)}")
    speech = sum(t.cost_usd for t in record.line_timings)
    print(f"  cost:      ${record.cost_usd:.2f} (speech ${speech:.4f}, music ${record.music_info.cost_usd if record.music_info else 0:.2f})")
    hook = record.storyboard.shots[0] if record.storyboard and record.storyboard.shots else None
    if hook and hook.source_path:
        print(f"  hook:      {Path(hook.source_path).name} ({hook.subject or 'photo'}, {hook.seconds:.1f} s)")
    print("\n--- Layout ---")
    print(f"  {'shot':<5}{'photo':<24}{'layout':<8}{'peak enlargement':>17}   (at the final frame size)")
    for shot, clip in zip(record.storyboard.shots, record.clips):
        photo = Path(shot.source_path).name if shot.source_path else "-"
        print(f"  {shot.index + 1:<5}{photo[:22]:<24}{clip.layout or clip.mode:<8}"
              f"{(f'x{clip.peak_upscale:.2f}' if clip.layout else '-'):>17}")
    print(f"\n  total wall time {wall_s:.1f} s")
    print("\n--- Defects ---")
    for defect in record.defects or ["none"]:
        print(f"  - {defect}")


def print_gate(record) -> None:
    ev = record.evaluation
    if ev is None:
        return
    print(f"\n--- Quality gate: {ev.gate_result.value.upper()}, score {ev.quality_score:.0f}/100 "
          f"(status {record.status}) ---")
    for item in ev.items:
        mark = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL", "skipped": "skip"}[item.status]
        print(f"  [{mark}] {item.name:<19}{item.detail}")
    print(f"  details: {Path(record.output_dir) / 'qa.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--property-id", required=True)
    parser.add_argument("--duration", type=int, choices=VIDEO_DURATIONS, default=None)
    parser.add_argument("--aspect", choices=VIDEO_ASPECTS, default=None)
    parser.add_argument("--model", default=None, help="auto, kenburns, or a provider id from configs/video/providers.yaml")
    parser.add_argument("--plan-only", action="store_true", help="stop after the script and the storyboard")
    parser.add_argument("--offline", action="store_true", help="take every fallback; make no network call")
    parser.add_argument("--draft", action="store_true", help="render small and fast, to look at a plan")
    parser.add_argument("--refresh-ledger", action="store_true",
                        help="ask the model for the fact ledger again instead of reusing the one kept for this property")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):          # a Windows console is not UTF-8: a price with a rupee sign must still print
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.offline:
        force_offline()
    from src.workflows.video import run_plan_workflow, run_video_workflow     # after the settings are fixed

    properties = settings.data_dir_resolved / "properties"
    try:
        property_dir = brief_dir(properties, args.property_id)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not property_dir.exists() and properties.is_dir():          # the folder name is matched without regard to case
        property_dir = next((d for d in sorted(properties.iterdir())
                             if d.is_dir() and d.name.lower() == args.property_id.strip().lower()), property_dir)
    if not property_dir.exists():
        raise SystemExit(f"No property found at {property_dir}. Use the Streamlit form or "
                         f"scripts/ingest_property.py first.")

    overrides = {"duration_s": args.duration, "aspect": args.aspect,
                 "model": "kenburns" if args.offline else args.model}
    started = time.perf_counter()
    run = run_plan_workflow if args.plan_only else run_video_workflow
    record = run(property_dir, overrides, offline=args.offline, draft=args.draft, refresh_ledger=args.refresh_ledger)
    wall_s = time.perf_counter() - started

    print(f"Status:   {record.status}")
    if record.workflow_evaluation:
        for step in record.workflow_evaluation.steps:
            print(f"  [{'OK  ' if step.status == 'success' else 'FAIL'}] {step.node:<14} {step.latency_ms:>8.0f} ms")
    if record.status == "failed" and not record.video_path:
        print(f"Error: {record.error}")
        raise SystemExit(1)
    print(f"Video id: {record.video_id}")
    print(f"Folder:   {record.output_dir}")
    print_plan(record)
    if not args.plan_only:
        print_result(record, wall_s)
        print_gate(record)
    blocked = [n for n in record.notes if n.startswith("Blocked by the run's policy")]
    if blocked:
        print("\n--- Blocked ---")
        for note in blocked:
            print(f"  - {note}")
    if record.music_info and record.music_info.provider != "none":
        print(f"\n--- Music prompt ---\n  {record.music_info.prompt}")
    if record.notes:
        print("\n--- Notes ---")
        for note in record.notes:
            print(f"  - {note}")
    if record.status == "failed":
        print(f"\nError: {record.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
