"""Node 9, compose: the clips, the text and the voice, as one MP4.

    picture   every clip at one frame rate and size, cut to its exact frame
              count, given the light grade (photographs only) and joined by
              crossfades centred on the cuts (rendering/transitions.py); each
              piece of text laid over with its own fade, and a slide up where it
              has one; encoded once, as H.264 yuv420p
    voice     each line placed at its start time on a silent track of the full
              length; a music track, if there is one, goes under it and is
              ducked while the voice speaks
    level     two-pass EBU R128 loudness to -16 LUFS: ffmpeg measures the mix,
              then applies exactly the gain that measurement says
    mux       the picture is copied, the audio encoded as AAC, the moov atom
              moved to the front so the file plays while it downloads

The only encoder is the ffmpeg that imageio-ffmpeg
bundles (rendering/ffmpeg.py). If no MP4 comes out, this raises and the run
fails: it is the one step with no fallback, because there is nothing to fall
back to.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.config import load_yaml_config
from src.core.exceptions import RenderingError
from src.rendering import ffmpeg, transitions

logger = logging.getLogger(__name__)

SILENT_FLOOR_LUFS = -60.0       # below this the mix is silence: there is nothing to normalise
HOLD_S = 30                     # a clip shorter than its shot holds its last frame this long, then is cut
AUDIO_RATE = 48000
MONO = f"aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=mono"


@dataclass
class ClipPart:
    path: Path
    frames: int                       # the shot's own length; a clip runs past it for its crossfades
    grade: bool = False               # a photograph is graded; a card keeps the brand's exact colours


@dataclass
class OverlayPart:
    path: Path
    start_s: float
    end_s: float
    fade_s: float = 0.3
    slide_px: int = 0                 # slides up from this far below over slide_s while it fades in


@dataclass
class VoicePart:
    path: Path
    start_s: float


@dataclass
class ComposeResult:
    video_path: Path
    thumbnail_path: Path
    loudness: Optional[ffmpeg.Loudness] = None
    notes: list[str] = field(default_factory=list)


def thread_caps() -> tuple[int, int]:
    """(decoder threads per clip, per overlay picture) from configs/video/styles.yaml."""
    section = load_yaml_config("video/styles.yaml").get("compose") or {}
    return max(1, int(section.get("decode_threads", 2))), max(1, int(section.get("overlay_threads", 1)))


def output_caps() -> tuple[int, int]:
    """(filter graph threads, x264 threads) from configs/video/styles.yaml."""
    section = load_yaml_config("video/styles.yaml").get("compose") or {}
    return max(1, int(section.get("filter_threads", 2))), max(1, int(section.get("encode_threads", 4)))


def shot_frames(seconds: list[float], fps: int) -> list[int]:
    """Whole frames for every shot, so the total never drifts: each shot ends on
    the frame its cumulative time falls on."""
    frames, edge, used = [], 0.0, 0
    for s in seconds:
        edge += s
        end = round(edge * fps)
        frames.append(max(1, end - used))
        used += frames[-1]
    return frames


def grade_filter(config: dict) -> str:
    """ffmpeg's eq for the light grade, or "" when it is off."""
    if not config.get("enabled"):
        return ""
    return (f"eq=contrast={config['contrast']}:saturation={config['saturation']}:"
            f"brightness={config['brightness']}")


def _render_picture(clips: list[ClipPart], overlays: list[OverlayPart], width: int, height: int, fps: int,
                    dest: Path, crf: int, preset: str, grade: str, fade_frames: int, slide_s: float) -> None:
    frames = [clip.frames for clip in clips]
    found = transitions.fades(frames, fade_frames)
    head_tail = transitions.margins(frames, fade_frames)
    decode_threads, overlay_threads = thread_caps()
    inputs: list[str] = []
    for clip in clips:
        inputs += ["-threads", str(decode_threads), "-i", str(clip.path)]   # every clip is open at once: keep each decoder small
    for ov in overlays:
        inputs += ["-threads", str(overlay_threads), "-loop", "1", "-framerate", str(fps),
                   "-t", f"{ov.end_s - ov.start_s:.3f}", "-i", str(ov.path)]
    parts = []
    for i, (clip, (head, tail)) in enumerate(zip(clips, head_tail)):
        eq = f"{grade}," if grade and clip.grade else ""
        parts.append(f"[{i}:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                     f"crop={width}:{height},setsar=1,{eq}format=yuv420p,tpad=stop_mode=clone:stop_duration={HOLD_S},"
                     f"trim=end_frame={clip.frames + head + tail},setpts=PTS-STARTPTS[c{i}]")
    if found:
        label, edge = "c0", 0
        for i, fade in enumerate(found):
            edge += frames[i]
            parts.append(f"[{label}][c{i + 1}]xfade=transition=fade:duration={fade.frames / fps:.4f}:"
                         f"offset={(edge - fade.lead) / fps:.4f}[x{i}]")
            label = f"x{i}"
        base = label
    else:
        parts.append("".join(f"[c{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[base]")
        base = "base"
    label = base
    for j, ov in enumerate(overlays):
        seconds = ov.end_s - ov.start_s
        f = min(ov.fade_s, seconds / 2)
        parts.append(f"[{len(clips) + j}:v]format=rgba,fade=t=in:st=0:d={f:.3f}:alpha=1,"
                     f"fade=t=out:st={seconds - f:.3f}:d={f:.3f}:alpha=1,setpts=PTS+{ov.start_s:.3f}/TB[o{j}]")
        move = (f"y='{ov.slide_px}*pow(1-min(1,max(0,(t-{ov.start_s:.3f})/{slide_s})),3)':" if ov.slide_px else "")
        parts.append(f"[{label}][o{j}]overlay={move}enable='between(t,{ov.start_s:.3f},{ov.end_s:.3f})':"
                     f"eof_action=pass:format=auto[b{j}]")
        label = f"b{j}"
    parts.append(f"[{label}]format=yuv420p[vout]")
    filter_threads, encode_threads = output_caps()
    ffmpeg.run([*inputs, "-filter_complex_threads", str(filter_threads), "-filter_complex", ";".join(parts),
                "-map", "[vout]", "-r", str(fps), "-frames:v", str(sum(frames)), "-threads", str(encode_threads),
                "-c:v", "libx264", "-preset", preset,
                "-crf", str(crf), "-pix_fmt", "yuv420p", str(dest)])


def _mix_voice(voices: list[VoicePart], music: Optional[Path], total_s: float, gain: float, dest: Path) -> None:
    if not voices and not music:
        ffmpeg.run(["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=mono", "-t", f"{total_s:.3f}", str(dest)])
        return
    if not voices:                                  # music alone: nothing to duck it for
        ffmpeg.run(["-i", str(music), "-af", f"{MONO},apad=whole_dur={total_s:.3f},atrim=0:{total_s:.3f}",
                    "-ar", str(AUDIO_RATE), "-ac", "1", str(dest)])
        return
    inputs = [arg for v in voices for arg in ("-i", str(v.path))]
    parts = [f"[{k}:a]{MONO},adelay={round(v.start_s * 1000)}:all=1[a{k}]" for k, v in enumerate(voices)]
    n = len(voices)
    joined = "".join(f"[a{k}]" for k in range(n))
    parts.append(f"{joined}amix=inputs={n}:normalize=0:duration=longest[mix]" if n > 1 else "[a0]anull[mix]")
    parts.append(f"[mix]apad=whole_dur={total_s:.3f},atrim=0:{total_s:.3f}[voice]")
    if music:
        inputs += ["-i", str(music)]
        parts += [f"[{n}:a]{MONO},volume={gain},apad=whole_dur={total_s:.3f},atrim=0:{total_s:.3f}[mus]",
                  "[voice]asplit=2[v1][v2]",
                  "[mus][v2]sidechaincompress=threshold=0.03:ratio=10:attack=20:release=400[duck]",
                  "[v1][duck]amix=inputs=2:normalize=0:duration=first[out]"]
    else:
        parts.append("[voice]anull[out]")
    ffmpeg.run([*inputs, "-filter_complex", ";".join(parts), "-map", "[out]", "-ar", str(AUDIO_RATE), "-ac", "1",
                str(dest)])


def _normalise(source: Path, dest: Path, target: float, peak: float, lra: float) -> Optional[str]:
    """Two-pass loudnorm. Returns a note when nothing was done (silence)."""
    base = f"loudnorm=I={target}:TP={peak}:LRA={lra}"
    report = ffmpeg.run_info(["-i", str(source), "-af", f"{base}:print_format=json", "-f", "null", "-"])
    found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", report, re.S)
    if not found:
        raise RenderingError("ffmpeg's loudness measurement could not be read")
    measured = json.loads(found.group(0))
    if float(measured["input_i"]) < SILENT_FLOOR_LUFS:
        dest.write_bytes(source.read_bytes())
        return "The audio is silent, so it was not loudness-normalised."
    ffmpeg.run(["-i", str(source), "-af",
                f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
                f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
                f"offset={measured['target_offset']}:linear=true,aresample={AUDIO_RATE}",
                "-ar", str(AUDIO_RATE), "-ac", "1", str(dest)])
    return None


def compose(*, clips: list[ClipPart], overlays: list[OverlayPart], voices: list[VoicePart], out_path: Path,
            thumbnail_path: Path, width: int, height: int, fps: int, draft: bool, work_dir: Path,
            hook_seconds: float, music_path: Optional[Path] = None, grade: bool = True) -> ComposeResult:
    if not clips:
        raise RenderingError("There are no clips to compose.")
    config = load_yaml_config("video/pacing.yaml")
    c, quality = config["compose"], "draft" if draft else "final"
    total_s = sum(clip.frames for clip in clips) / fps
    work_dir.mkdir(parents=True, exist_ok=True)
    silent, mix, level = work_dir / "picture.mp4", work_dir / "voice_mix.wav", work_dir / "voice_level.wav"

    _render_picture(clips, overlays, width, height, fps, silent, c["crf"][quality], c["preset"][quality],
                    grade_filter(c["grade"]) if grade else "", transitions.transition_frames(c["transition_s"], fps),
                    config["overlays"]["slide_s"])
    _mix_voice(voices, music_path, total_s, c["music_gain"], mix)
    notes = []
    note = _normalise(mix, level, c["loudness_lufs"], c["true_peak_db"], c["loudness_range"])
    if note:
        notes.append(note)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(["-i", str(silent), "-i", str(level), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                "-b:a", c["audio_bitrate"], "-ac", "2", "-ar", str(AUDIO_RATE), "-movflags", "+faststart",
                "-t", f"{total_s:.3f}", str(out_path)])
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RenderingError("ffmpeg finished but no MP4 was written")

    at = min(c["thumbnail_s"], max(0.2, hook_seconds * 0.7))
    ffmpeg.run(["-ss", f"{at:.2f}", "-i", str(out_path), "-frames:v", "1", "-q:v", "2", str(thumbnail_path)])
    return ComposeResult(out_path, thumbnail_path, ffmpeg.measure_loudness(out_path), notes)
