"""Node 5, storyboard: which photograph, which camera move, how long.

The rule planner. The shots are the script's lines laid on a timeline:

    * the hook gets the best exterior, and at least `min_hook_s`
    * every other shot gets the best unused photograph of what its line talks
      about (the line's topic against the picture's topic), else the best
      unused photograph there is. When the pictures carry tags (a vision model
      read them), a line that needs something shown gets a picture tagged with
      it (the one the script set aside, else another), and never an unrelated one;
      a general line gets a picture tagged exterior if one is left
    * a chip is printed only over a picture tagged with what the chip names
    * a line that spans two shots gets two DIFFERENT photographs (both tagged with what it
      needs, when it needs something); when no other photograph will do, the two shots become
      one longer shot. The same photograph is never in two shots one after the other
    * pictures that look alike (image_plan) are not shown twice; when there
      are fewer pictures than shots, a picture is shown again with a different
      crop and the opposite move, and the shot is marked `reuse`
    * the last shot is a two second closing card (kind="card"): palette, logo,
      call to action, RERA and contact are drawn on it by our code
    * a shot with no usable picture to give it is a card too
    * nothing is generated: this planner never makes a "generated" shot

`validate_plan` is the check any plan has to pass, this one included, and the
one a model's plan will be put through before it is used.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.agents.brochure import focal
from src.agents.video import photo_tags
from src.agents.brochure.image_plan import Allocator, Plan
from src.core.config import load_yaml_config
from src.core.exceptions import WorkflowError
from src.data.schemas import ImageAnalysis
from src.data.video_schemas import CAMERA_MOVES, SHOT_KINDS, RankedAsset, Shot, Storyboard, VideoBrief, VideoScript

logger = logging.getLogger(__name__)

OPPOSITE = {"push_in": "pull_out", "pull_out": "push_in", "pan_left": "pan_right", "pan_right": "pan_left",
            "tilt_up": "tilt_down", "tilt_down": "tilt_up", "still": "still"}
WIDE_RATIO = 1.3          # a picture this much wider than the frame has room to pan
SECONDS_TOLERANCE = 0.1   # the shots add up to the video's length, give or take this


def allowed_generated_kinds() -> list[str]:
    styles = load_yaml_config("video/styles.yaml")
    return list(styles["sectors"][styles["default_sector"]]["allowed_generated_kinds"])


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def photo_seconds(duration_s: float, count: int, end_card_s: float, min_hook_s: float) -> list[float]:
    """Seconds for each photograph shot: the hook first, the rest equal, the
    whole adding up to the video less its closing card."""
    total = round(duration_s - end_card_s, 1)
    if count <= 1:
        return [total]
    hook = round(max(min_hook_s, total / count), 1)
    rest = round((total - hook) / (count - 1), 1)
    seconds = [hook] + [rest] * (count - 1)
    seconds[-1] = round(total - sum(seconds[:-1]), 1)
    return seconds


def assign_lines(script: VideoScript, seconds: list[float], words_per_second: float) -> list[list[int]]:
    """Which script lines each photograph shot plays under. The lines (the
    call to action apart: the closing card takes that) are laid end to end,
    squeezed to the time the photograph shots have, and a shot gets the lines
    whose middle falls inside it. A shot no line's middle falls in carries on
    the line that is playing when it begins."""
    body = [i for i, line in enumerate(script.lines) if line.role != "cta"]
    if not body:
        return [[] for _ in seconds]
    lengths = [max(1.0, script.lines[i].words / words_per_second) for i in body]
    scale = sum(seconds) / sum(lengths)
    spans, t = [], 0.0
    for i, length in zip(body, lengths):
        spans.append((i, t, t + length * scale))
        t += length * scale
    out, start = [], 0.0
    for sec in seconds:
        end = start + sec
        ids = [i for i, a, b in spans if start <= (a + b) / 2 < end]
        out.append(ids or [i for i, a, b in spans if a <= start < b][:1] or [spans[-1][0]])
        start = end
    return out


def hook_asset(usable: list[RankedAsset], hero_name: str, notes: list[str]) -> Optional[RankedAsset]:
    """The picture the video opens on when pictures are tagged: the hero the person named, else the
    plainest picture tagged exterior (the one with the fewest amenity tags: a picture that also
    shows a pool is kept for the line about the pool), the better picture winning a tie
    (`usable` is best first); None when there is no such picture (the planner then falls back
    to the picture's topic)."""
    hero = resolve_hero(hero_name, usable, notes)
    if hero is not None:
        return hero
    amenities = set(photo_tags.names())
    exteriors = [a for a in usable if "exterior" in a.tags]
    return min(exteriors, key=lambda a: sum(t in amenities for t in a.tags), default=None)


def resolve_hero(name: str, assets: list[RankedAsset], notes: list[str]) -> Optional[RankedAsset]:
    """The picture video_input.json asks to open on (by image id or file name,
    with or without the extension), if it exists and can carry a shot; else
    None, with the reason in the notes."""
    key = (name or "").strip().lower()
    if not key:
        return None
    found = next((a for a in assets if key in (a.image_id.lower(), a.filename.lower(),
                                               Path(a.filename).stem.lower())), None)
    if found is None:
        notes.append(f"hero_image '{name}' matches no uploaded picture; ignored.")
    elif not found.usable:
        notes.append(f"hero_image '{name}' cannot be used ({found.note}); ignored.")
    else:
        return found
    return None


def one_tagged_line_per_shot(script: VideoScript, lines_by_shot: list[list[int]]) -> list[list[int]]:
    """A shot shows one picture, so it can carry only one line that needs a picture shown.
    When the timing put two such lines in one shot, the later one moves to the front of the
    next shot (the order of the lines is kept). The last photograph shot keeps what it has."""
    shots = [list(ids) for ids in lines_by_shot]
    for k in range(len(shots) - 1):
        needing = [i for i in shots[k] if script.lines[i].tags]
        if len(needing) > 1:
            at = shots[k].index(needing[1])
            moved, shots[k] = shots[k][at:], shots[k][:at]
            shots[k + 1] = moved + [i for i in shots[k + 1] if i not in moved]
    return shots


def _by_tags(lines: list, fresh: list, candidates: list[str], by_id: dict) -> Optional[str]:
    """The picture for a shot when pictures are tagged (`candidates` are the unused pictures,
    best first). `fresh` are the lines of the shot that begin here and need something shown:
    they get the picture the script set aside, else one tagged with what they need. A line
    that carries on from an earlier shot takes a picture of the same kind. If no candidate
    shows what a line needs the answer is None (never an unrelated picture: the planner then
    shows a matching picture again). Lines that need nothing take a picture tagged exterior
    if there is one, else the best."""
    for line in fresh:
        if line.photo and line.photo in candidates:
            return line.photo
    carrying = [line for line in lines if line.tags and line not in fresh]
    for line in [*fresh, *carrying]:
        hit = next((c for c in candidates if set(line.tags) & set(by_id[c].tags)), None)
        if hit:
            return hit
    if fresh or carrying:
        return None
    return next((c for c in candidates if "exterior" in by_id[c].tags), candidates[0] if candidates else None)


def _shift(focal_point: str, wide: bool) -> str:
    """The other side of the picture, for the second showing of a photograph."""
    try:
        x, y = (float(p.rstrip("%")) for p in focal_point.split())
    except ValueError:
        x, y = 50.0, 50.0
    if wide:
        x = 20.0 if x >= 50 else 80.0
    else:
        y = 25.0 if y >= 50 else 75.0
    return f"{x:.0f}% {y:.0f}%"


def run(brief: VideoBrief, script: VideoScript, assets: list[RankedAsset],
        analyses: dict[str, ImageAnalysis], plan: Plan) -> Storyboard:
    pacing = load_yaml_config("video/pacing.yaml")
    words_per_second = load_yaml_config("video/speech.yaml")["words_per_second"]
    end_card_s, notes = pacing["end_card_s"], []
    usable = [a for a in assets if a.usable]                      # best first
    by_id = {a.image_id: a for a in usable}
    tagged = any(a.tags for a in usable)                          # read by a vision model
    desired = brief.n_shots - 1
    allowance = pacing["max_reuse_per_photo"]
    count = 1 if not usable else min(desired, len(usable) + min(len(usable) * allowance, len(usable) - 1))
    if not usable:
        notes.append("No usable photograph: the video is title cards only.")
    elif count < desired:
        notes.append(f"{len(usable)} usable photograph(s) allow {count} photograph shot(s), not {desired}: "
                     "the shots are a little longer.")

    seconds = photo_seconds(brief.duration_s, count, end_card_s, pacing["min_hook_s"])
    lines_by_shot = assign_lines(script, seconds, words_per_second)
    if tagged:
        lines_by_shot = one_tagged_line_per_shot(script, lines_by_shot)
    out_aspect = brief.width / brief.height
    allocator = Allocator(plan, by_id, analyses)
    hero = resolve_hero(brief.hero_image, assets, notes)
    first_camera, uses, shown_chips = {}, {}, set()
    chip_why: dict[str, str] = {}                                  # chip -> why a picture did not carry it
    shots: list[Shot] = []
    cycle_at, matched, asked = 0, 0, 0
    owned: set[int] = set()                                        # lines already shown in an earlier shot

    for k in range(count):
        n = len(shots) + 1                                         # this shot's place in the finished video
        line_ids = lines_by_shot[k]
        lines = [script.lines[i] for i in line_ids]
        fresh = [script.lines[i] for i in line_ids if i not in owned and script.lines[i].tags]
        carry = [script.lines[i] for i in line_ids if i in owned and script.lines[i].tags]
        continuing = bool(shots) and shots[-1].kind == "real" and all(i in owned for i in line_ids)
        owned.update(line_ids)
        wanted = "exterior" if k == 0 else next((l.topic for l in lines if l.topic), "")
        asset: Optional[RankedAsset] = None
        reuse = False
        candidates: list[str] = []
        if usable:
            if k == 0 and hero:
                choice = hero.image_id
                allocator.note(choice, 0)
            elif k == 0 and tagged and hook_asset(usable, "", []):
                choice = hook_asset(usable, "", []).image_id
                allocator.note(choice, 0)
            elif k == 0:
                choice = allocator.take(0, ("exterior",))
            else:
                candidates = allocator.candidates(k)
                if tagged:
                    choice = _by_tags(lines, fresh, candidates, by_id)
                else:
                    choice = next((c for c in candidates if wanted and by_id[c].topic == wanted),
                                  candidates[0] if candidates else None)
                if choice:
                    allocator.note(choice, k)
            asset = by_id.get(choice) if choice else None
            if asset is None and not continuing:                   # every picture is used: show one again
                previous = shots[-1].image_id if shots else ""
                needing = fresh + carry if tagged else []
                again = [a for a in usable if 1 <= uses.get(a.image_id, 0) <= allowance
                         and a.image_id != previous]                # never the picture of the shot before
                again.sort(key=lambda a: (not any(set(l.tags) & set(a.tags) for l in lines), a.topic != wanted))
                if needing:                                        # never an unrelated picture for a line that needs one
                    again = [a for a in again if any(set(l.tags) & set(a.tags) for l in needing)]
                asset, reuse = (again[0] if again else None), True
        if asset is None and continuing:
            # No new photograph will do for the rest of a line already shown (a photograph is not shown
            # again for it): one longer shot.
            before = shots[-1]
            before.seconds = round(before.seconds + seconds[k], 3)
            before.line_ids = [*before.line_ids, *[i for i in line_ids if i not in before.line_ids]]
            what = ", ".join(sorted({t for l in lines for t in l.tags})) or "the line"
            notes.append(f"Shot {n} would have shown {before.source_path and Path(before.source_path).name or 'the same picture'} "
                         f"again: no other photograph shows {what}, so it is joined to the shot before, which is longer.")
            continue
        needed = fresh
        if asset is not None and asset.tags and needed:
            asked += 1
            matched += any(set(l.tags) & set(asset.tags) for l in needed)
            if not any(set(l.tags) & set(asset.tags) for l in needed):
                notes.append(f"Shot {n} plays '{needed[0].text}' over {asset.filename}, which shows "
                             f"{', '.join(asset.tags) or 'nothing tagged'}, not {', '.join(needed[0].tags)}.")
        elif wanted and wanted != "exterior" and asset is not None and not (tagged and asset.tags):
            asked += 1
            matched += asset.topic == wanted

        overlay = list(script.on_screen) if k == 0 else []
        for chip, line_index in zip(script.chips, script.chip_lines):
            # A chip names something (a pool, a clubhouse); it is only printed over a
            # photograph known to show that, so it never labels the wrong picture.
            topic = script.lines[line_index].topic
            if line_index not in line_ids or chip in shown_chips or asset is None:
                continue
            if asset.tags:                       # read by a vision model: the chip's tag must be among the picture's
                if photo_tags.supports(chip, asset.tags):
                    overlay.append(chip)
                    shown_chips.add(chip)
                else:
                    named = sorted(photo_tags.needs(chip))
                    chip_why.setdefault(chip, f"{asset.filename} (the picture under its line) shows "
                                              f"{', '.join(asset.tags)}, not {', '.join(named) if named else 'what it names'}")
            elif topic and asset.topic == topic:
                overlay.append(chip)
                shown_chips.add(chip)
        if asset is None:
            shots.append(Shot(index=len(shots), kind="card", seconds=seconds[k], camera="still", subject="card",
                              overlay_text=overlay or [brief.name], line_ids=line_ids))
            continue

        wide = asset.aspect > out_aspect * WIDE_RATIO
        cycle = pacing["camera_cycles"]["wide" if wide else "tall"]
        previous_camera = shots[-1].camera if shots else ""
        if k == 0:
            camera = pacing["hook_camera"]
        elif reuse:
            camera = OPPOSITE[first_camera[asset.image_id]]
        else:
            camera = next(c for c in cycle[cycle_at:] + cycle[:cycle_at] if c != previous_camera)
            cycle_at = (cycle.index(camera) + 1) % len(cycle)
        if reuse and camera == previous_camera:
            camera = next(c for c in cycle if c != previous_camera)
        first_camera.setdefault(asset.image_id, camera)
        uses[asset.image_id] = uses.get(asset.image_id, 0) + 1

        point = focal.focal_point(asset.path, asset.aspect, out_aspect, asset.topic)
        shots.append(Shot(
            index=len(shots), kind="real", image_id=asset.image_id, source_path=asset.path, subject=asset.topic,
            caption=asset.caption, tags=list(asset.tags),
            camera=camera, motion_prompt=f"{pacing['motion_phrases'][camera]}. {pacing['motion_suffix']}",
            seconds=seconds[k], focal=_shift(point, wide) if reuse else point, overlay_text=overlay,
            line_ids=line_ids, reuse=reuse))
        if reuse:
            notes.append(f"Shot {n} shows {asset.filename} again, cropped from the other side.")

    for chip, line_index in zip(script.chips, script.chip_lines):
        if chip not in shown_chips:
            topic = script.lines[line_index].topic
            if chip in chip_why:
                notes.append(f"Chip '{chip}' dropped: {chip_why[chip]}.")
            else:
                notes.append(f"Chip '{chip}' not shown: no photograph under its line is known to show "
                             f"{topic or 'what it names'}.")
    cta_ids = [i for i, line in enumerate(script.lines) if line.role == "cta"]
    shots.append(Shot(index=len(shots), kind="card", seconds=end_card_s, camera="still", subject="end card",
                      overlay_text=[t for t in [script.cta, *script.end_card] if t], line_ids=cta_ids))
    if asked:
        notes.append(f"{matched} of {asked} shots show what their line talks about"
                     + ("." if matched == asked else "; the rest use the best photograph left."))

    storyboard = Storyboard(shots=shots, planner_mode="rules", notes=notes)
    problems = validate_plan(storyboard, brief, assets, plan)
    if problems:
        raise WorkflowError("The rule planner produced a plan that fails its own checks: " + "; ".join(problems))
    return storyboard


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------

def validate_plan(storyboard: Storyboard, brief: VideoBrief, assets: list[RankedAsset], plan: Plan,
                  guard=None) -> list[str]:
    """What is wrong with a storyboard, in words, or [] when it can be used.

    Checks that every photograph exists and is usable, that no picture (or
    its twin) is shown twice unless the shot says so, that the shots add up
    to the video's length, that the camera moves and shot kinds are known,
    that a generated shot is allowed by the brief and the sector, that the
    video ends on a card, and (with a guard) that no overlay text states a
    figure that is not a fact."""
    problems: list[str] = []
    shots = storyboard.shots
    if not shots:
        return ["the storyboard has no shots"]
    by_id = {a.image_id: a for a in assets}
    shown: set = set()
    for before, shot in zip(shots, shots[1:]):
        if shot.kind == "real" and before.kind == "real" and shot.image_id == before.image_id:
            by_name = by_id.get(shot.image_id)
            problems.append(f"Shots {before.index + 1} and {shot.index + 1} show the same photograph "
                            f"({by_name.filename if by_name else shot.image_id}) one after the other")
    for shot in shots:
        label = f"Shot {shot.index + 1}"
        if shot.kind not in SHOT_KINDS:
            problems.append(f"{label} has an unknown kind '{shot.kind}'")
        if shot.camera not in CAMERA_MOVES:
            problems.append(f"{label} has an unknown camera move '{shot.camera}'")
        if shot.seconds <= 0:
            problems.append(f"{label} has no length")
        if shot.kind == "real":
            asset = by_id.get(shot.image_id)
            if asset is None:
                problems.append(f"{label} uses a photograph that does not exist ({shot.image_id or 'none named'})")
                continue
            if not asset.usable:
                problems.append(f"{label} uses {asset.filename}, which is not usable ({asset.note})")
            key = plan.cluster(shot.image_id)
            key = ("cluster", key) if key >= 0 else ("image", shot.image_id)
            if key in shown and not shot.reuse:
                problems.append(f"{label} shows {asset.filename} (or a picture like it) again without being marked reuse")
            if key not in shown and shot.reuse:
                problems.append(f"{label} is marked reuse but is the first showing of {asset.filename}")
            shown.add(key)
        elif shot.kind == "generated":
            if not brief.allow_generated:
                problems.append(f"{label} is generated, but this video does not allow generated pictures")
            if shot.generated_kind not in allowed_generated_kinds():
                problems.append(f"{label} is a generated '{shot.generated_kind}'; "
                                f"only {', '.join(allowed_generated_kinds())} are allowed")
            if not shot.image_prompt:
                problems.append(f"{label} is generated but has no image prompt")
        if guard is not None:
            for text in shot.overlay_text:
                reason = guard.why(text, repeat=False)
                if reason:
                    problems.append(f"{label} prints '{text}', but {reason}")
    if shots[-1].kind != "card":
        problems.append("the video does not end on a closing card")
    total = storyboard.total_seconds
    if abs(total - brief.duration_s) > SECONDS_TOLERANCE:
        problems.append(f"the shots add up to {total} s, not {brief.duration_s} s")
    return problems
