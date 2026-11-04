"""The ffmpeg wrapper, against real (tiny) files made by the bundled ffmpeg."""
from __future__ import annotations

import math

import pytest

from src.core.exceptions import RenderingError
from src.rendering import ffmpeg


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """One second of 160x284 test picture and a 440 Hz tone: H.264 + AAC."""
    path = tmp_path_factory.mktemp("media") / "tiny.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", "testsrc=duration=1:size=160x284:rate=25",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=44100",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)])
    return path


def test_probe_reads_video_and_audio_streams(clip):
    info = ffmpeg.probe(clip)
    assert info.has_video and info.vcodec == "h264" and info.pix_fmt == "yuv420p"
    assert (info.width, info.height) == (160, 284)
    assert info.fps == 25
    assert info.has_audio and info.acodec == "aac" and info.sample_rate == 44100
    assert info.duration_s == pytest.approx(1.0, abs=0.1)
    assert info.size_bytes == clip.stat().st_size > 0


def test_probe_of_a_silent_video_says_there_is_no_audio(tmp_path):
    path = tmp_path / "silent.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", "color=c=gray:size=64x64:duration=0.5:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    info = ffmpeg.probe(path)
    assert info.has_video and not info.has_audio and info.acodec == ""
    assert ffmpeg.measure_loudness(path) is None


def test_loudness_of_a_tone_is_a_finite_reading(clip):
    loud = ffmpeg.measure_loudness(clip)
    assert loud is not None and math.isfinite(loud.integrated_lufs)
    assert -40 < loud.integrated_lufs < 0
    assert loud.true_peak_dbfs is not None and loud.true_peak_dbfs <= 0.5


def test_silence_reads_at_the_floor(tmp_path):
    path = tmp_path / "quiet.m4a"
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", "-c:a", "aac", str(path)])
    loud = ffmpeg.measure_loudness(path)
    assert loud is not None and loud.integrated_lufs <= -69


def test_encode_frames_turns_raw_rgb_into_an_h264_clip(tmp_path):
    frames = (bytes([i * 25 % 256, 80, 160]) * (64 * 48) for i in range(10))
    ffmpeg.encode_frames(frames, tmp_path / "raw.mp4", width=64, height=48, fps=10)
    info = ffmpeg.probe(tmp_path / "raw.mp4")
    assert (info.width, info.height, info.vcodec, info.pix_fmt) == (64, 48, "h264", "yuv420p")
    assert info.duration_s == pytest.approx(1.0, abs=0.1)


def test_encode_frames_surfaces_a_failure_in_the_frame_source(tmp_path):
    def frames():
        yield bytes(64 * 48 * 3)
        raise ValueError("a frame could not be cut")

    with pytest.raises(ValueError, match="could not be cut"):
        ffmpeg.encode_frames(frames(), tmp_path / "raw.mp4", width=64, height=48, fps=10)


def test_encode_frames_reports_ffmpegs_message_when_it_refuses(tmp_path):
    with pytest.raises(RenderingError, match="ffmpeg failed"):
        ffmpeg.encode_frames([bytes(64 * 48 * 3)], tmp_path / "no_such_dir" / "raw.mp4", width=64, height=48, fps=10)


def test_probe_of_a_missing_file_raises(tmp_path):
    with pytest.raises(RenderingError, match="does not exist"):
        ffmpeg.probe(tmp_path / "nope.mp4")


def test_probe_of_a_file_that_is_not_media_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a video", encoding="utf-8")
    with pytest.raises(RenderingError, match="could not read"):
        ffmpeg.probe(path)


def test_a_failing_run_raises_with_the_end_of_ffmpegs_message(tmp_path):
    with pytest.raises(RenderingError, match="ffmpeg failed"):
        ffmpeg.run(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4")])
