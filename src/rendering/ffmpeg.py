"""The one place the video workflow runs ffmpeg.

The binary comes from imageio-ffmpeg, so nothing has to be installed on the
machine. That package ships ffmpeg and no ffprobe, so a file is read by
parsing what `ffmpeg -i` prints about it, and loudness by the `ebur128`
filter. ffmpeg runs as a subprocess, like Chromium does for the PDFs.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.core.exceptions import RenderingError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - a missing binary is reported once, in words
        raise RenderingError(
            "ffmpeg is not available. Install the video requirements: pip install -r requirements.txt "
            f"(imageio-ffmpeg). {type(exc).__name__}: {exc}") from exc


def _execute(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([ffmpeg_exe(), "-hide_banner", *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RenderingError(f"ffmpeg did not finish in {timeout:.0f}s ({' '.join(args[:6])} ...)") from exc


def run(args: list[str], *, timeout: float = DEFAULT_TIMEOUT_S) -> subprocess.CompletedProcess:
    """ffmpeg with `-y -loglevel error` ahead of `args`. Raises RenderingError
    with the end of ffmpeg's own message when it exits non-zero."""
    result = _execute(["-y", "-loglevel", "error", *args], timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-600:] or f"exit code {result.returncode}"
        raise RenderingError(f"ffmpeg failed: {detail}")
    return result


def encode_frames(frames: Iterable[bytes], dest: Path | str, *, width: int, height: int, fps: int,
                  crf: int = 14, preset: str = "fast", timeout: float = DEFAULT_TIMEOUT_S) -> None:
    """Raw RGB24 frames of `width` x `height` in, an H.264 yuv420p MP4 out.

    Frames are piped to ffmpeg's stdin as they are made, so a clip of
    full-size frames is never held in memory. ffmpeg's messages go to a
    temp file, not a pipe nobody reads, so a chatty failure cannot block it."""
    args = [ffmpeg_exe(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", str(dest)]
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            try:
                for frame in frames:
                    proc.stdin.write(frame)
            except BrokenPipeError:
                pass        # ffmpeg stopped early; its own message below says why
            finally:
                proc.stdin.close()
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RenderingError(f"ffmpeg did not finish encoding in {timeout:.0f}s") from exc
        except BaseException:
            proc.kill()
            raise
        if code != 0:
            log.seek(0)
            detail = log.read().decode("utf-8", "replace").strip()[-600:] or f"exit code {code}"
            raise RenderingError(f"ffmpeg failed: {detail}")


def run_info(args: list[str], *, timeout: float = DEFAULT_TIMEOUT_S) -> str:
    """ffmpeg at its normal log level, returning what it printed. For filters
    that report their findings in the log (loudnorm's measurement); raises
    RenderingError, with the end of the log, when ffmpeg exits non-zero."""
    result = _execute(["-y", "-nostats", *args], timeout)
    if result.returncode != 0:
        raise RenderingError(f"ffmpeg failed: {(result.stderr or '').strip()[-600:]}")
    return result.stderr


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaInfo:
    duration_s: float = 0.0
    size_bytes: int = 0
    has_video: bool = False
    vcodec: str = ""
    pix_fmt: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    acodec: str = ""
    sample_rate: int = 0
    channels: str = ""


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO = re.compile(r"Stream #\S+: Video: (\w+)[^\n]*")
_AUDIO = re.compile(r"Stream #\S+: Audio: (\w+)[^\n]*?, (\d+) Hz, ([^,\n]+)")
# "0x31637661" in a codec tag must not read as a size.
_SIZE = re.compile(r"(?<![\w])(\d{2,5})x(\d{2,5})(?![\w])")
_PIX_FMT = re.compile(r"Video: \w+[^,\n]*, (\w+)")
_FPS = re.compile(r"(\d+(?:\.\d+)?) fps")
_TBR = re.compile(r"(\d+(?:\.\d+)?k?) tbr")


def probe(path: Path | str, *, timeout: float = 60) -> MediaInfo:
    """Streams, size and duration of a media file. Raises RenderingError if
    the file is missing or ffmpeg cannot read it."""
    path = Path(path)
    if not path.exists():
        raise RenderingError(f"Cannot probe {path}: the file does not exist")
    # `ffmpeg -i` with no output exits 1 by design; the report is on stderr.
    info = _execute(["-i", str(path)], timeout).stderr
    duration = _DURATION.search(info)
    video, audio = _VIDEO.search(info), _AUDIO.search(info)
    if not duration and not video and not audio:
        raise RenderingError(f"ffmpeg could not read {path.name} as media")

    h, m, s = duration.groups() if duration else (0, 0, 0)
    fields = {"duration_s": round(int(h) * 3600 + int(m) * 60 + float(s), 3),
              "size_bytes": path.stat().st_size}
    if video:
        line = video.group(0)
        size = _SIZE.search(line)
        fps = _FPS.search(line) or _TBR.search(line)
        pix = _PIX_FMT.search(line)
        fields.update(has_video=True, vcodec=video.group(1), pix_fmt=pix.group(1) if pix else "",
                      width=int(size.group(1)) if size else 0, height=int(size.group(2)) if size else 0,
                      fps=_number(fps.group(1)) if fps else 0.0)
    if audio:
        fields.update(has_audio=True, acodec=audio.group(1), sample_rate=int(audio.group(2)),
                      channels=audio.group(3).strip())
    return MediaInfo(**fields)


def _number(text: str) -> float:
    return float(text[:-1]) * 1000 if text.endswith("k") else float(text)


# ---------------------------------------------------------------------------
# Loudness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Loudness:
    integrated_lufs: float
    true_peak_dbfs: Optional[float]
    range_lu: Optional[float]


_LUFS = re.compile(r"\bI:\s+(-?inf|-?\d+(?:\.\d+)?) LUFS")
_LRA = re.compile(r"\bLRA:\s+(-?\d+(?:\.\d+)?) LU\b")
_PEAK = re.compile(r"\bPeak:\s+(-?inf|-?\d+(?:\.\d+)?) dBFS")


def measure_loudness(path: Path | str, *, timeout: float = 120) -> Optional[Loudness]:
    """EBU R128 integrated loudness and true peak of a file's audio, or None
    when it has no audio. Silence reads as -70 LUFS, ffmpeg's floor."""
    path = Path(path)
    if not probe(path).has_audio:
        return None
    report = _execute(["-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
                      timeout).stderr
    summary = report[report.rfind("Summary:"):] if "Summary:" in report else ""
    lufs, lra, peak = _LUFS.search(summary), _LRA.search(summary), _PEAK.search(summary)
    if not lufs:
        logger.warning("Could not read loudness from ffmpeg's report for %s", path.name)
        return None
    return Loudness(integrated_lufs=_db(lufs.group(1)), true_peak_dbfs=_db(peak.group(1)) if peak else None,
                    range_lu=float(lra.group(1)) if lra else None)


def _db(text: str) -> float:
    return float("-inf") if text.endswith("inf") else float(text)
