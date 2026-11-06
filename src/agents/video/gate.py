"""Node 10, gate: measurements on the finished video and on the plan it came from.

No model judges anything. Each check reads a number off the MP4, the storyboard, the
script or the voice timings and says pass, warn or fail against the limits in
configs/video/evaluation.yaml:

    format          codec, pixel format, size, frame rate, an audio track, a file of some size
    duration        within tolerance of the length asked for
    loudness        integrated loudness near -16 LUFS, true peak under the ceiling
    black_frames    stretches of black picture
    frozen          stretches where the picture does not change (the still closing card excepted)
    narration_gaps  silence between spoken lines, and before the first
    safe_zones      every piece of printed text inside the platform-safe area
    numbers         every figure on screen or spoken is a fact of the property
    thumbnail       exists and is not blank
    upscale         how far each photograph's pixels are enlarged, per shot
    one_voice       every line spoken by the same voice
    hook            the opening shot lasts at least min_hook_s
    chips           a chip is only printed over a photograph whose tags show it
    narration_picture  every amenity a spoken line names is a tag of a photograph that line plays over
    repeated_photo  the same photograph in two shots one after the other (a warning)

A check that cannot be measured is "skipped", with the reason, and the score is worked out
from the rest. The score is the weighted share of passes (a warn counts half), out of 100.
The gate FAILS when a hard check fails or the score is under `fail_below`, WARNS when any
check warns or the score is under `pass_at`, and PASSES otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageStat

from src.agents.brochure import facts, strategy
from src.agents.video import photo_tags
from src.agents.video.guard import VideoGuard
from src.core.config import load_yaml_config
from src.data.video_schemas import (
    ClipResult, GateCheck, LineTiming, MusicInfo, OverlayItem, Storyboard, VideoBrief, VideoEvaluationResult,
    VideoGate, VideoMeasurements, VideoScript,
)
from src.rendering import ffmpeg, video_overlay

logger = logging.getLogger(__name__)

CONFIG_FILE = "video/evaluation.yaml"
VALUE = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


@dataclass
class GateInputs:
    video_path: Path
    thumbnail_path: Optional[Path]
    brief: VideoBrief
    script: VideoScript
    storyboard: Storyboard
    clips: list[ClipResult] = field(default_factory=list)
    line_timings: list[LineTiming] = field(default_factory=list)
    overlay_items: list[OverlayItem] = field(default_factory=list)
    music_info: Optional[MusicInfo] = None
    guard: Optional[VideoGuard] = None


def _check(name: str, status: str, detail: str, shots: Optional[list[int]] = None, **values) -> GateCheck:
    return GateCheck(name=name, status=status, detail=detail, shots=sorted(set(shots or [])), values=values)


def _worst(*statuses: str) -> str:
    return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"


def _by_limits(value: float, warn_above: float, fail_above: float) -> str:
    return "fail" if value > fail_above else "warn" if value > warn_above else "pass"


# ---------------------------------------------------------------------------
# Measuring the file
# ---------------------------------------------------------------------------

def measure(video_path: Path, cfg: Optional[dict] = None) -> VideoMeasurements:
    """Everything read off the finished file: its streams, loudness, black and frozen stretches."""
    cfg = cfg or load_yaml_config(CONFIG_FILE)
    info = ffmpeg.probe(video_path)
    loud = ffmpeg.measure_loudness(video_path)
    measured = VideoMeasurements(
        duration_s=info.duration_s, width=info.width, height=info.height, fps=info.fps, vcodec=info.vcodec,
        acodec=info.acodec, has_audio=info.has_audio, size_bytes=info.size_bytes, pix_fmt=info.pix_fmt,
        loudness_lufs=loud.integrated_lufs if loud else None, true_peak_dbfs=loud.true_peak_dbfs if loud else None)
    black, frozen = ffmpeg.detect_black_and_frozen(
        video_path, black_min_s=cfg["black"]["min_s"], black_pixel=cfg["black"]["pixel_threshold"],
        freeze_noise_db=cfg["frozen"]["noise_db"], freeze_min_s=cfg["frozen"]["min_s"])
    measured.black_segments = len(black)
    measured.black_s = round(sum(s.length_s for s in black), 3)
    measured.frozen_s = round(sum(s.length_s for s in frozen), 3)
    measured.black_spans = [[s.start_s, s.end_s] for s in black]
    measured.frozen_spans = [[s.start_s, s.end_s] for s in frozen]
    return measured


def _shots_at(board: Storyboard, start_s: float, end_s: float) -> list[int]:
    """The shots (1-based) a stretch of the video overlaps."""
    out, t = [], 0.0
    for shot in board.shots:
        if start_s < t + shot.seconds and end_s > t:
            out.append(shot.index + 1)
        t += shot.seconds
    return out


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_format(m: VideoMeasurements, brief: VideoBrief, cfg: dict) -> GateCheck:
    problems = []
    if m.vcodec != cfg["vcodec"]:
        problems.append(f"the video codec is {m.vcodec or 'unknown'}, not {cfg['vcodec']}")
    if m.pix_fmt and m.pix_fmt != cfg["pix_fmt"]:
        problems.append(f"the pixel format is {m.pix_fmt}, not {cfg['pix_fmt']}")
    if (m.width, m.height) != (brief.width, brief.height):
        problems.append(f"the size is {m.width}x{m.height}, not {brief.width}x{brief.height}")
    if abs(m.fps - brief.fps) > cfg["fps_tolerance"]:
        problems.append(f"the frame rate is {m.fps:g}, not {brief.fps}")
    if not m.has_audio:
        problems.append("there is no audio track")
    elif m.acodec != cfg["acodec"]:
        problems.append(f"the audio codec is {m.acodec}, not {cfg['acodec']}")
    if m.size_bytes < cfg["min_size_bytes"]:
        problems.append(f"the file is only {m.size_bytes} bytes")
    detail = "; ".join(problems) or (f"{m.vcodec}/{m.acodec} {m.pix_fmt}, {m.width}x{m.height}, {m.fps:g} fps, "
                                    f"{m.size_bytes / 1e6:.1f} MB")
    return _check("format", "fail" if problems else "pass", detail, vcodec=m.vcodec, acodec=m.acodec,
                  width=m.width, height=m.height, fps=m.fps, size_bytes=m.size_bytes)


def check_duration(m: VideoMeasurements, brief: VideoBrief, cfg: dict) -> GateCheck:
    off = abs(m.duration_s - brief.duration_s)
    status = _by_limits(off, cfg["warn_s"], cfg["fail_s"])
    return _check("duration", status, f"{m.duration_s:.2f} s against {brief.duration_s} s asked for "
                  f"({off:.2f} s off)", duration_s=m.duration_s, off_s=round(off, 3))


def check_loudness(m: VideoMeasurements, expected: bool, cfg: dict) -> GateCheck:
    if not expected:
        return _check("loudness", "skipped", "no narration and no music: nothing to measure")
    if m.loudness_lufs is None or m.loudness_lufs <= -69.0:
        return _check("loudness", "fail", "the audio track is silent although narration or music was made")
    off = abs(m.loudness_lufs - cfg["target_lufs"])
    level = "pass" if off <= cfg["pass_lu"] else "fail" if off > cfg["fail_lu"] else "warn"
    peak = m.true_peak_dbfs
    ceiling = cfg["max_true_peak_dbfs"]
    peak_status = "pass" if peak is None or peak <= ceiling else "fail" if peak >= 0 else "warn"
    parts = [f"{m.loudness_lufs:.1f} LUFS against {cfg['target_lufs']} ({off:.1f} LU off)"]
    if peak is not None:
        parts.append(f"true peak {peak:.1f} dBFS" + ("" if peak_status == "pass" else f" over the {ceiling:g} ceiling"))
    return _check("loudness", _worst(level, peak_status), ", ".join(parts), lufs=m.loudness_lufs, true_peak_dbfs=peak)


def check_black(m: VideoMeasurements, board: Storyboard, cfg: dict) -> GateCheck:
    status = "pass" if m.black_s <= cfg["warn_s"] else _by_limits(m.black_s, cfg["warn_s"], cfg["fail_s"])
    if not m.black_segments:
        return _check("black_frames", "pass", "no black frames")
    shots = [n for a, b in m.black_spans for n in _shots_at(board, a, b)]
    where = ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in m.black_spans)
    return _check("black_frames", status, f"{m.black_segments} black stretch(es), {m.black_s:.2f} s in all ({where})",
                  shots, black_s=m.black_s, spans=m.black_spans)


def check_frozen(m: VideoMeasurements, board: Storyboard, cfg: dict) -> GateCheck:
    """Stretches that do not change, less the closing card (still by design)."""
    card_from = round(sum(s.seconds for s in board.shots[:-1]), 3) if board.shots and board.shots[-1].kind == "card" \
        else m.duration_s
    spans = []
    for a, b in m.frozen_spans:
        b = min(b, card_from)
        if b - a >= cfg["min_s"] - 1e-6:
            spans.append((a, b))
    if not spans:
        return _check("frozen", "pass", "nothing but the closing card stands still")
    longest = max(b - a for a, b in spans)
    where = ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in spans)
    shots = [n for a, b in spans for n in _shots_at(board, a, b)]
    return _check("frozen", _by_limits(longest, cfg["warn_s"] - 1e-6, cfg["fail_s"] - 1e-6),
                  f"the picture does not change for {longest:.1f} s ({where})", shots, longest_s=round(longest, 3))


def check_narration(timings: list[LineTiming], cfg: dict) -> GateCheck:
    spoken = sorted((t for t in timings if t.mode == "speech"), key=lambda t: t.start_s)
    if not spoken:
        return _check("narration_gaps", "warn", "no line was spoken: the video has no narration")
    problems, status = [], "pass"
    if spoken[0].start_s > cfg["max_lead_s"]:
        problems.append(f"{spoken[0].start_s:.1f} s of silence before the first word")
        status = "warn"
    gaps = [(b.start_s - a.end_s, i + 1) for i, (a, b) in enumerate(zip(spoken, spoken[1:]))]
    for gap, after in gaps:
        if gap > cfg["max_gap_s"]:
            problems.append(f"{gap:.1f} s of silence after line {after}")
            status = _worst(status, _by_limits(gap, cfg["max_gap_s"], cfg["fail_gap_s"]))
    widest = max((g for g, _ in gaps), default=0.0)
    return _check("narration_gaps", status, "; ".join(problems) or f"the longest pause is {widest:.2f} s",
                  longest_gap_s=round(widest, 3))


def check_safe_zones(items: list[OverlayItem], brief: VideoBrief, cfg: dict) -> GateCheck:
    safe = video_overlay.safe_rect(brief.width, brief.height, video_overlay.safe_zone(brief.aspect))
    x0, y0, x1, y1 = safe
    tolerance = cfg["tolerance_px"]
    outside, measured, unmeasured = [], 0, 0
    for item in items:
        if not item.rects:
            unmeasured += 1 if item.texts else 0
            continue
        for rect in item.rects:
            measured += 1
            left, top = float(rect["x"]), float(rect["y"])
            right, bottom = left + float(rect["w"]), top + float(rect["h"])
            if left < x0 - tolerance or top < y0 - tolerance or right > x1 + tolerance or bottom > y1 + tolerance:
                outside.append((item.shot_index + 1, str(rect.get("text", item.key))))
    if outside:
        names = "; ".join(f"'{t}' in shot {n}" for n, t in outside[:4])
        return _check("safe_zones", "fail", f"{len(outside)} piece(s) of text outside the safe zone: {names}",
                      [n for n, _ in outside])
    if not measured:
        return _check("safe_zones", "skipped" if not items else "warn",
                      "no text was drawn" if not items else "where the text landed was not measured")
    note = f" ({unmeasured} piece(s) drawn by the fallback renderer were not measured)" if unmeasured else ""
    return _check("safe_zones", "pass", f"{measured} piece(s) of text, all inside the safe zone{note}")


def printed_texts(inputs: GateInputs) -> list[str]:
    """Everything a viewer sees written or hears: the overlay pictures' text and the script's lines."""
    texts = [t for item in inputs.overlay_items for t in item.texts]
    s = inputs.script
    texts += [line.text for line in s.lines] + list(s.on_screen) + list(s.chips) + list(s.end_card) + [s.cta]
    return [t for t in dict.fromkeys(texts) if t and t.strip()]


def check_numbers(inputs: GateInputs) -> GateCheck:
    if inputs.guard is None:
        return _check("numbers", "skipped", "no fact ledger to check the figures against")
    texts = printed_texts(inputs)
    bad = [(t, inputs.guard.unknown_figure(t)) for t in texts]
    bad = [(t, f) for t, f in bad if f]
    if bad:
        names = "; ".join(f"'{f}' in '{t}'" for t, f in bad[:4])
        return _check("numbers", "fail", f"{len(bad)} figure(s) not in the property's facts: {names}")
    return _check("numbers", "pass", f"every figure in {len(texts)} printed or spoken texts is a fact of the property")


def check_thumbnail(path: Optional[Path], brief: VideoBrief, cfg: dict) -> GateCheck:
    if path is None or not Path(path).exists():
        return _check("thumbnail", "fail", "there is no thumbnail")
    path = Path(path)
    size = path.stat().st_size
    try:
        with Image.open(path) as handle:
            grey = handle.convert("L")
            width, height = grey.size
            spread = float(ImageStat.Stat(grey).stddev[0])
    except Exception as exc:  # noqa: BLE001 - an unreadable thumbnail is a failed check, not an error
        return _check("thumbnail", "fail", f"the thumbnail cannot be read ({type(exc).__name__})")
    problems = []
    status = _worst("pass", "fail" if spread < cfg["fail_stddev"] else "warn" if spread < cfg["warn_stddev"] else "pass")
    if status != "pass":
        problems.append(f"it is almost blank (brightness spread {spread:.1f})")
    if size < cfg["min_size_bytes"]:
        problems.append(f"it is only {size} bytes")
        status = "fail"
    if abs(width / height - brief.width / brief.height) > 0.02:
        problems.append(f"its shape is {width}x{height}, not the video's")
        status = _worst(status, "warn")
    return _check("thumbnail", status, "; ".join(problems) or f"{width}x{height}, brightness spread {spread:.0f}",
                  stddev=round(spread, 1), size_bytes=size)


def check_upscale(clips: list[ClipResult], cfg: dict) -> GateCheck:
    measured = [c for c in clips if c.layout and c.peak_upscale]
    if not measured:
        return _check("upscale", "skipped", "no photograph shot was laid out")
    worst = max(measured, key=lambda c: c.peak_upscale)
    status = _by_limits(worst.peak_upscale, cfg["warn_above"], cfg["fail_above"])
    per_shot = ", ".join(f"{c.shot_index + 1}: {c.layout} x{c.peak_upscale:.2f}" for c in measured)
    over = [c.shot_index + 1 for c in measured if c.peak_upscale > cfg["warn_above"]]
    detail = f"the most a photograph is enlarged is x{worst.peak_upscale:.2f} (shot {worst.shot_index + 1}); {per_shot}"
    return _check("upscale", status, detail, over, peak_upscale=worst.peak_upscale)


def check_one_voice(timings: list[LineTiming]) -> GateCheck:
    if not timings:
        return _check("one_voice", "skipped", "there are no spoken lines")
    spoken = [t for t in timings if t.mode == "speech"]
    voices = {(t.provider, t.voice) for t in spoken}
    if len(voices) > 1:
        return _check("one_voice", "fail", "the lines were spoken by different voices: "
                      + ", ".join(sorted(f"{p} {v}".strip() for p, v in voices)))
    if spoken and len(spoken) < len(timings):
        return _check("one_voice", "fail", f"{len(timings) - len(spoken)} of {len(timings)} lines are silent "
                      "and the rest are spoken")
    if not spoken:
        return _check("one_voice", "pass", "no line was spoken (one silent track)")
    p, v = next(iter(voices))
    return _check("one_voice", "pass", f"all {len(spoken)} lines were spoken by {p} {v}".strip())


def check_hook(board: Storyboard, minimum: float, tolerance: float = 0.05) -> GateCheck:
    if not board.shots:
        return _check("hook", "fail", "the storyboard has no shots")
    first = board.shots[0]
    status = "pass" if first.seconds >= minimum - tolerance else "fail"
    return _check("hook", status, f"the opening shot lasts {first.seconds:.1f} s; the minimum is {minimum:g} s",
                  [1] if status == "fail" else [], hook_s=round(first.seconds, 3))


def check_chips(board: Storyboard, script: VideoScript) -> GateCheck:
    shown = []
    for shot in board.shots:
        for text in shot.overlay_text:
            if text in script.chips and shot.kind == "real":
                shown.append((shot, text))
    if not shown:
        return _check("chips", "pass", "no chip is printed")
    wrong, unchecked = [], []
    for shot, chip in shown:
        if not shot.tags:
            unchecked.append((shot, chip))
        elif not photo_tags.supports(chip, shot.tags):
            wrong.append((shot, chip))
    if wrong:
        names = "; ".join(f"'{c}' over a photo showing {', '.join(s.tags)} (shot {s.index + 1})" for s, c in wrong)
        return _check("chips", "fail", f"a chip is over a photograph that does not show it: {names}",
                      [s.index + 1 for s, _ in wrong])
    if unchecked:
        names = "; ".join(f"'{c}' (shot {s.index + 1})" for s, c in unchecked)
        return _check("chips", "warn", f"a chip is over a photograph with no tags to check it against: {names}",
                      [s.index + 1 for s, _ in unchecked])
    return _check("chips", "pass", f"{len(shown)} chip(s), each over a photograph tagged with what it names")


def check_narration_picture(script: VideoScript, board: Storyboard, cfg: dict) -> GateCheck:
    """What the narrator says against what is on screen: each amenity a spoken line names must be a
    tag of at least one of the photographs under that line."""
    real = [s for s in board.shots if s.kind == "real"]
    if not any(s.tags for s in real):
        return _check("narration_picture", "skipped", "the photographs carry no tags to check the narration against")
    missing: list[tuple[int, str]] = []
    unverified: list[int] = []
    checked = 0
    for index, line in enumerate(script.lines):
        named = sorted(photo_tags.needs(line.text))
        if not named:
            continue
        under = [s for s in real if index in s.line_ids]
        if not under:
            missing += [(index + 1, name) for name in named]
            checked += len(named)
            continue
        shown = {t for s in under for t in s.tags}
        checked += len(named)
        if not shown:
            unverified.append(index + 1)
        else:
            missing += [(index + 1, name) for name in named if name not in shown]
    shots = sorted({s.index + 1 for s in real for i, _ in missing if i - 1 in s.line_ids})
    if missing:
        names = "; ".join(f"line {i} names {n}" for i, n in missing[:4])
        status = "fail" if len(missing) >= cfg["fail_missing"] else "warn"
        return _check("narration_picture", status, f"{len(missing)} amenity(ies) named and not shown by the pictures "
                      f"under the line: {names}", shots, missing=len(missing))
    if unverified:
        return _check("narration_picture", "warn", "lines naming an amenity play over photographs with no tags: "
                      + ", ".join(f"line {i}" for i in unverified))
    return _check("narration_picture", "pass", f"every one of the {checked} amenity name(s) in the spoken lines is "
                  "shown by a photograph under its line" if checked else "no spoken line names an amenity")


def check_narration_evidence(script: VideoScript, board: Storyboard) -> GateCheck:
    """An audit of the plan, from the stored shots: each amenity a spoken line names, and each chip, is
    over a photograph whose tags AND caption both say it (photo_tags.evidence). The planner only picks
    such photographs, so this warns for a plan made another way: an older one, or one a person edited."""
    real = [s for s in board.shots if s.kind == "real"]
    if not any(s.tags for s in real):
        return _check("narration_evidence", "skipped", "the photographs carry no tags to check the narration against")
    said = {s.index: set(photo_tags.from_caption(s.caption)) for s in real}
    weak: list[str] = []
    checked = 0
    for index, line in enumerate(script.lines):
        for name in sorted(photo_tags.needs(line.text)):
            under = [s for s in real if index in s.line_ids]
            checked += 1
            if under and not any(name in s.tags and name in said[s.index] for s in under):
                weak.append(f"line {index + 1} names {name}")
    for shot in real:
        for text in shot.overlay_text:
            for name in sorted(photo_tags.needs(text)) if text in script.chips else []:
                checked += 1
                if not (name in shot.tags and name in said[shot.index]):
                    weak.append(f"the chip '{text}' (shot {shot.index + 1})")
    if weak:
        return _check("narration_evidence", "warn", f"{len(weak)} of {checked} amenity mention(s) are over a photograph "
                      "whose tags and caption do not both say it: " + "; ".join(weak[:4]), weak=len(weak))
    return _check("narration_evidence", "pass", f"every one of the {checked} amenity mention(s) is over a photograph "
                  "whose tags and caption both say it" if checked else "no line or chip names an amenity")


def check_repeated_photo(board: Storyboard, cfg: dict) -> GateCheck:
    repeats = [(a, b) for a, b in zip(board.shots, board.shots[1:])
               if a.kind == "real" and b.kind == "real" and a.image_id and a.image_id == b.image_id]
    if len(repeats) < cfg["warn_at"]:
        return _check("repeated_photo", "pass", "no photograph is in two shots one after the other")
    where = ", ".join(f"{a.index + 1} and {b.index + 1}" for a, b in repeats)
    return _check("repeated_photo", "warn", f"the same photograph is in two shots one after the other (shots {where})",
                  [n for a, b in repeats for n in (a.index + 1, b.index + 1)], repeats=len(repeats))


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def verdict(items: list[GateCheck], cfg: dict) -> VideoEvaluationResult:
    """Weights, score, result and the lists, from the checks."""
    weights, hard = cfg["weights"], set(cfg.get("hard") or [])
    for item in items:
        item.weight = float(weights.get(item.name, 0.0))
        item.hard = item.name in hard
    counted = [i for i in items if i.status in VALUE and i.weight > 0]
    total = sum(i.weight for i in counted)
    score = round(100 * sum(i.weight * VALUE[i.status] for i in counted) / total, 1) if total else 0.0
    failed_hard = [i for i in items if i.hard and i.status == "fail"]
    if failed_hard or score < cfg["fail_below"]:
        gate = VideoGate.FAIL
    elif any(i.status == "warn" or i.status == "fail" for i in items) or score < cfg["pass_at"]:
        gate = VideoGate.WARN
    else:
        gate = VideoGate.PASS
    defects = [f"[{i.status.upper()}] {i.name}: {i.detail}" for i in items if i.status in ("warn", "fail")]
    skipped = [f"{i.name} was not measured: {i.detail}" for i in items if i.status == "skipped"]
    return VideoEvaluationResult(
        checks={i.name: "pass" if i.status == "pass" else f"{i.status}: {i.detail}" for i in items},
        items=items, defects=defects, feedback=skipped,
        failed_shots=sorted({n for i in items if i.status in ("warn", "fail") for n in i.shots}),
        quality_score=score, gate_result=gate, judge="measured")


def build_guard(prop, knowledge, brief: VideoBrief) -> VideoGuard:
    return VideoGuard(prop, knowledge, facts.build(prop, brief.category), strategy.for_category(brief.category))


def evaluate(inputs: GateInputs, measurements: Optional[VideoMeasurements] = None,
             cfg: Optional[dict] = None) -> tuple[VideoEvaluationResult, VideoMeasurements]:
    """Run every check. A check that raises is recorded as skipped with the reason: the gate
    never fails a run by itself failing."""
    cfg = cfg or load_yaml_config(CONFIG_FILE)
    pacing = load_yaml_config("video/pacing.yaml")
    brief = inputs.brief
    items: list[GateCheck] = []
    try:
        m = measurements or measure(inputs.video_path, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("The finished file could not be measured: %s", exc)
        m = VideoMeasurements()
        items.append(_check("format", "fail", f"the file could not be read ({type(exc).__name__}: {str(exc)[:120]})"))
    music = inputs.music_info
    audio_expected = any(t.mode == "speech" for t in inputs.line_timings) or bool(music and music.provider != "none")
    steps: list[tuple[str, Callable[[], GateCheck]]] = []
    if not items:
        steps += [("format", lambda: check_format(m, brief, cfg["format"])),
                  ("duration", lambda: check_duration(m, brief, cfg["duration"])),
                  ("loudness", lambda: check_loudness(m, audio_expected, cfg["loudness"])),
                  ("black_frames", lambda: check_black(m, inputs.storyboard, cfg["black"])),
                  ("frozen", lambda: check_frozen(m, inputs.storyboard, cfg["frozen"]))]
    steps += [("narration_gaps", lambda: check_narration(inputs.line_timings, cfg["narration"])),
              ("safe_zones", lambda: check_safe_zones(inputs.overlay_items, brief, cfg["safe_zones"])),
              ("numbers", lambda: check_numbers(inputs)),
              ("thumbnail", lambda: check_thumbnail(inputs.thumbnail_path, brief, cfg["thumbnail"])),
              ("upscale", lambda: check_upscale(inputs.clips, cfg["upscale"])),
              ("one_voice", lambda: check_one_voice(inputs.line_timings)),
              ("hook", lambda: check_hook(inputs.storyboard, pacing["min_hook_s"])),
              ("chips", lambda: check_chips(inputs.storyboard, inputs.script)),
              ("narration_picture", lambda: check_narration_picture(inputs.script, inputs.storyboard,
                                                                    cfg["narration_picture"])),
              ("narration_evidence", lambda: check_narration_evidence(inputs.script, inputs.storyboard)),
              ("repeated_photo", lambda: check_repeated_photo(inputs.storyboard, cfg["repeated_photo"]))]
    for name, step in steps:
        try:
            items.append(step())
        except Exception as exc:  # noqa: BLE001 - one check failing to run is not the video failing
            logger.warning("Gate check %s could not run: %s", name, exc)
            items.append(_check(name, "skipped", f"could not be measured ({type(exc).__name__}: {str(exc)[:120]})"))
    return verdict(items, cfg), m
