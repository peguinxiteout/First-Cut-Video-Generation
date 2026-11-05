"""Node 9: the clips, the text and the voice as one MP4."""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from src.core.config import load_yaml_config
from src.core.exceptions import RenderingError
from src.models.speech.mock import MockSpeechProvider
from src.rendering import ffmpeg, video_compose
from src.rendering.video_compose import ClipPart, OverlayPart, VoicePart

W, H, FPS = 270, 480, 10


def _clip(path, seconds, colour, size=(W, H), fps=FPS):
    ffmpeg.run(["-f", "lavfi", "-i", f"color=c={colour}:s={size[0]}x{size[1]}:r={fps}:d={seconds}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    return path


def _overlay(path, box=(40, 300, 230, 380)):
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=(255, 255, 255, 255))
    image.save(path)
    return path


def _voice(path, text="one two three four five six seven eight nine ten"):
    MockSpeechProvider().synthesize(text, None, path)
    return path


def _frame(video, at, tmp_path, name):
    out = tmp_path / f"{name}.png"
    ffmpeg.run(["-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1", str(out)])
    return Image.open(out).convert("RGB")


def _compose(tmp_path, *, voices=None, overlays=None, music=None, clips=None, **kw):
    if clips is None:
        clips = [ClipPart(_clip(tmp_path / "a.mp4", 2, "red"), 20), ClipPart(_clip(tmp_path / "b.mp4", 3, "green"), 30),
                 ClipPart(_clip(tmp_path / "c.mp4", 2, "blue"), 20)]
    voices = [VoicePart(_voice(tmp_path / "v1.wav"), 0.3), VoicePart(_voice(tmp_path / "v2.wav"), 3.0)] if voices is None else voices
    overlays = [OverlayPart(_overlay(tmp_path / "o.png"), 0.5, 1.8)] if overlays is None else overlays
    args = dict(clips=clips, overlays=overlays, voices=voices, out_path=tmp_path / "out" / "video.mp4",
                thumbnail_path=tmp_path / "out" / "thumbnail.jpg", width=W, height=H, fps=FPS, draft=True,
                work_dir=tmp_path / "work", hook_seconds=2.0, music_path=music)
    args.update(kw)
    return video_compose.compose(**args)


# -- the file -----------------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_file_is_h264_yuv420p_with_aac_at_the_asked_size_fps_and_length(tmp_path):
    result = _compose(tmp_path)

    info = ffmpeg.probe(result.video_path)
    assert (info.vcodec, info.pix_fmt, info.acodec) == ("h264", "yuv420p", "aac")
    assert (info.width, info.height, info.fps) == (W, H, FPS)
    assert info.duration_s == pytest.approx(7.0, abs=0.3)
    assert info.has_audio and info.sample_rate == 48000 and info.channels == "stereo"


def test_the_voice_is_normalised_to_minus_16_lufs(tmp_path):
    result = _compose(tmp_path)
    assert result.loudness is not None
    assert result.loudness.integrated_lufs == pytest.approx(-16.0, abs=1.5)
    assert result.loudness.true_peak_dbfs <= -1.0


@pytest.mark.slow
def test_a_quiet_voice_and_a_loud_one_end_at_the_same_level(tmp_path):
    loud = _voice(tmp_path / "loud.wav")
    quiet = tmp_path / "quiet.wav"
    ffmpeg.run(["-i", str(loud), "-af", "volume=0.1", str(quiet)])
    levels = []
    for name, voice in (("loud", loud), ("quiet", quiet)):
        folder = tmp_path / name
        folder.mkdir()
        levels.append(_compose(folder, voices=[VoicePart(voice, 0.3)]).loudness.integrated_lufs)
    assert all(level == pytest.approx(-16.0, abs=1.5) for level in levels)


def test_every_clip_is_cut_to_its_frames_so_the_length_is_exact(tmp_path):
    result = _compose(tmp_path)
    info = ffmpeg.probe(result.video_path)
    assert info.duration_s == pytest.approx(70 / FPS, abs=0.15)


def test_a_clip_shorter_than_its_shot_holds_its_last_frame(tmp_path):
    clips = [ClipPart(_clip(tmp_path / "a.mp4", 1, "red"), 30), ClipPart(_clip(tmp_path / "b.mp4", 1, "blue"), 10)]
    result = _compose(tmp_path, clips=clips, overlays=[], voices=[])
    assert ffmpeg.probe(result.video_path).duration_s == pytest.approx(4.0, abs=0.2)
    held = _frame(result.video_path, 2.5, tmp_path, "held")
    assert held.getpixel((W // 2, H // 2))[0] > 200                                    # still red at 2.5 s


def test_a_clip_of_another_size_and_rate_is_conformed(tmp_path):
    odd = _clip(tmp_path / "odd.mp4", 2, "green", size=(160, 284), fps=25)
    result = _compose(tmp_path, clips=[ClipPart(odd, 20)], overlays=[], voices=[])
    info = ffmpeg.probe(result.video_path)
    assert (info.width, info.height, info.fps) == (W, H, FPS)


def test_the_cuts_land_where_the_frame_counts_say(tmp_path):
    result = _compose(tmp_path, overlays=[], voices=[])
    before, after = _frame(result.video_path, 1.8, tmp_path, "b"), _frame(result.video_path, 2.2, tmp_path, "a")
    assert before.getpixel((5, 5))[0] > 200 and before.getpixel((5, 5))[1] < 60               # red
    assert after.getpixel((5, 5))[1] > 100 and after.getpixel((5, 5))[0] < 60                 # green


# -- the text ------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_an_overlay_is_burned_in_for_its_window_only(tmp_path):
    result = _compose(tmp_path)
    inside, before, after = (_frame(result.video_path, t, tmp_path, f"t{i}") for i, t in enumerate((1.2, 0.2, 2.4)))
    probe_point = (135, 340)
    assert inside.getpixel(probe_point)[0] > 200 and inside.getpixel(probe_point)[1] > 200        # the white box
    assert before.getpixel(probe_point)[1] < 60                                                     # red clip, no box yet
    assert after.getpixel(probe_point)[0] < 60                                                      # green clip, box gone


@pytest.mark.slow
def test_an_overlay_fades_in_and_out(tmp_path):
    result = _compose(tmp_path)
    at_start = _frame(result.video_path, 0.6, tmp_path, "start")        # 0.1 s into a 0.3 s fade
    full = _frame(result.video_path, 1.2, tmp_path, "full")
    assert at_start.getpixel((135, 340))[1] < full.getpixel((135, 340))[1] - 30


# -- the voice -------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_each_line_is_placed_at_its_start_time(tmp_path):
    late = VoicePart(_voice(tmp_path / "late.wav"), 4.5)
    result = _compose(tmp_path, voices=[late])
    out = tmp_path / "early.wav"
    ffmpeg.run(["-i", str(result.video_path), "-t", "4.0", "-vn", "-ar", "8000", "-ac", "1", str(out)])
    assert ffmpeg.measure_loudness(out).integrated_lufs < -60                                 # nothing before 4.5 s


def test_with_no_voice_the_file_still_has_a_silent_audio_track(tmp_path):
    result = _compose(tmp_path, voices=[])
    info = ffmpeg.probe(result.video_path)
    assert info.has_audio and info.acodec == "aac"
    assert result.loudness.integrated_lufs < -60
    assert result.notes == ["The audio is silent, so it was not loudness-normalised."]


@pytest.mark.slow
def test_a_music_track_goes_under_the_voice(tmp_path):
    music = tmp_path / "music.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "sine=frequency=330:duration=7:sample_rate=44100", str(music)])
    result = _compose(tmp_path, music=music)
    assert ffmpeg.probe(result.video_path).has_audio
    assert result.loudness.integrated_lufs == pytest.approx(-16.0, abs=2.0)


def test_music_alone_is_normalised_too(tmp_path):
    music = tmp_path / "music.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "sine=frequency=330:duration=7:sample_rate=44100", str(music)])
    result = _compose(tmp_path, music=music, voices=[])
    assert result.loudness.integrated_lufs == pytest.approx(-16.0, abs=2.0)


# -- the thumbnail ---------------------------------------------------------------------------------------

def test_the_thumbnail_is_a_frame_of_the_hook_with_its_title(tmp_path):
    result = _compose(tmp_path, overlays=[OverlayPart(_overlay(tmp_path / "o.png"), 0.35, 1.8)], hook_seconds=2.0)
    thumb = Image.open(result.thumbnail_path)
    assert thumb.format == "JPEG" and thumb.size == (W, H)
    assert thumb.convert("RGB").getpixel((135, 340))[1] > 200                                # the overlay is in it


def test_a_short_hook_gives_a_thumbnail_from_inside_it(tmp_path):
    result = _compose(tmp_path, hook_seconds=0.5)
    assert result.thumbnail_path.exists()


# -- the failures -------------------------------------------------------------------------------------------

def test_no_clips_is_an_error(tmp_path):
    with pytest.raises(RenderingError, match="no clips"):
        _compose(tmp_path, clips=[])


def test_a_clip_that_cannot_be_read_fails_the_run_with_ffmpegs_message(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video", encoding="utf-8")
    with pytest.raises(RenderingError, match="ffmpeg failed"):
        _compose(tmp_path, clips=[ClipPart(bad, 10)], overlays=[], voices=[])


# -- frame counts ------------------------------------------------------------------------------------------

def test_shot_frames_never_drift_from_the_total():
    seconds = [2.327, 3.594, 2.642, 2.642, 3.795]
    frames = video_compose.shot_frames(seconds, 25)
    assert sum(frames) == round(sum(seconds) * 25)
    assert all(f >= 1 for f in frames)


def test_shot_frames_round_each_cut_to_its_own_frame_not_each_shot_to_its_own_length():
    assert video_compose.shot_frames([0.3, 0.3, 0.3], 10) == [3, 3, 3]
    assert sum(video_compose.shot_frames([0.34] * 10, 25)) == round(3.4 * 25)


def test_every_clip_input_caps_its_decoder_threads_because_all_of_them_are_open_at_once(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(video_compose.ffmpeg, "run", lambda args, **kw: seen.append(list(args)))
    clips = [ClipPart(tmp_path / f"c{i}.mp4", 10) for i in range(3)]
    overlays = [OverlayPart(tmp_path / f"o{i}.png", 1.0, 2.0) for i in range(4)]
    video_compose._render_picture(clips, overlays, W, H, FPS, tmp_path / "out.mp4", 26, "veryfast", "", 3, 0.4)
    args = seen[0]
    inputs = [i for i, a in enumerate(args) if a == "-i"]
    assert len(inputs) == 7
    for i in inputs[:3]:
        assert args[i - 2:i] == ["-threads", "2"]
    for i in inputs[3:]:                                    # each overlay picture: threads, then the loop options, then -i
        assert args[i - 8:i - 6] == ["-threads", "1"]
    assert args[args.index("-filter_complex_threads") + 1] == "2"          # the graph's own threads, before the graph
    assert args.index("-filter_complex_threads") < args.index("-filter_complex")
    at = args.index("-c:v")
    assert args[at - 2:at] == ["-threads", "4"]                            # x264's, an output option


def test_the_thread_caps_come_from_the_config_and_default_to_two_and_one(monkeypatch):
    assert load_yaml_config("video/styles.yaml")["compose"] == {"decode_threads": 2, "overlay_threads": 1,
                                                                 "filter_threads": 2, "encode_threads": 4}
    assert video_compose.thread_caps() == (2, 1) and video_compose.output_caps() == (2, 4)
    real = video_compose.load_yaml_config
    monkeypatch.setattr(video_compose, "load_yaml_config",
                        lambda name: {**real(name), "compose": {"decode_threads": 4, "overlay_threads": 3,
                                                                "filter_threads": 5, "encode_threads": 6}}
                        if name == "video/styles.yaml" else real(name))
    assert video_compose.thread_caps() == (4, 3) and video_compose.output_caps() == (5, 6)
    monkeypatch.setattr(video_compose, "load_yaml_config",
                        lambda name: {**real(name), "compose": {"decode_threads": 0}} if name == "video/styles.yaml" else real(name))
    assert video_compose.thread_caps() == (1, 1) and video_compose.output_caps() == (2, 4)   # never below one thread
    monkeypatch.setattr(video_compose, "load_yaml_config", lambda name: {} if name == "video/styles.yaml" else real(name))
    assert video_compose.thread_caps() == (2, 1) and video_compose.output_caps() == (2, 4)   # no section: the defaults
