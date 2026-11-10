"""The whole workflow: plan -> approve -> produce, on a synthetic property in draft mode."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.core.config import settings
from src.data import video_repository as repository
from src.data.video_schemas import Storyboard
from src.models.speech import mock as speech_mock
from src.models.video_gen import VideoGenerationTransportError, VideoService
from src.models.video_gen.kenburns import KenBurnsProvider
from src.rendering import ffmpeg, video_overlay
from src.workflows import video as workflow
from tests.video.fakes import ScriptedVideoProvider
from tests.video.helpers import make_video_property


@pytest.fixture(autouse=True)
def _database(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "videos.db"), raising=False)


@pytest.fixture
def refuse_models(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was asked")

    for target in ("src.agents.brochure.property_intelligence.call_with_fallback",
                   "src.agents.brochure.vision.call_with_fallback"):
        monkeypatch.setattr(target, refuse)


@pytest.fixture
def folder(tmp_path):
    return make_video_property(tmp_path / "properties")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One 15 s draft render, shared by the tests that only read its result."""
    root = tmp_path_factory.mktemp("render")
    saved = settings.data_dir, settings.database_path, settings.speech_provider
    settings.data_dir, settings.database_path, settings.speech_provider = str(root / "data"), str(root / "v.db"), "mock"
    try:
        record = workflow.run_video_workflow(make_video_property(root / "properties"), offline=True, draft=True)
    finally:
        settings.data_dir, settings.database_path, settings.speech_provider = saved
    return record


# -- the result -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_video_is_rendered_gated_and_every_step_ran(rendered):
    assert rendered.status in ("passed", "warned") and rendered.error is None
    assert rendered.evaluation is not None and rendered.evaluation.gate_result.value in ("pass", "warn")
    assert (Path(rendered.output_dir) / "qa.json").exists()
    assert re.fullmatch(r"vid_[0-9a-f]{10}", rendered.video_id)
    assert [(s.node, s.status) for s in rendered.workflow_evaluation.steps] == [(n, "success") for n in workflow._STEP_NAMES]
    assert len(workflow._STEP_NAMES) == 10 and workflow._STEP_NAMES[-1] == "gate"


@pytest.mark.slow
def test_the_files_are_where_the_record_says(rendered):
    out = Path(rendered.output_dir)
    for name in ("video.mp4", "thumbnail.jpg", "captions.srt", "script.json", "storyboard.json"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert Path(rendered.video_path) == out / "video.mp4" and Path(rendered.thumbnail_path) == out / "thumbnail.jpg"
    assert Path(rendered.captions_path) == out / "captions.srt"
    assert len(list((out / "shots").glob("shot_*.mp4"))) == len(rendered.storyboard.shots)
    assert len(list((out / "audio").glob("line_*.wav"))) == len(rendered.script.lines)
    assert list((out / "overlays").glob("*.png"))


@pytest.mark.slow
def test_the_file_is_what_was_asked_for(rendered):
    info = ffmpeg.probe(rendered.video_path)
    assert (info.vcodec, info.pix_fmt, info.acodec) == ("h264", "yuv420p", "aac")
    assert (info.width, info.height, info.fps) == (540, 960, 12)
    assert info.duration_s == pytest.approx(15.0, abs=0.5)
    m = rendered.measurements
    assert (m.width, m.height, m.vcodec, m.acodec, m.has_audio) == (540, 960, "h264", "aac", True)
    assert m.loudness_lufs == pytest.approx(-16.0, abs=1.5) and m.size_bytes == info.size_bytes
    assert rendered.actual_duration_s == m.duration_s


@pytest.mark.slow
def test_the_record_says_who_spoke_what_each_clip_was_and_what_it_cost(rendered):
    assert rendered.voice == "mock/mock"
    assert rendered.music == "" and rendered.cost_usd == 0
    assert [c.shot_index for c in rendered.clips] == list(range(len(rendered.storyboard.shots)))
    assert rendered.providers_used == {"kenburns": len(rendered.clips) - 1}
    assert [c.mode for c in rendered.clips][-1] == "card"
    assert len(rendered.line_timings) == len(rendered.script.lines)
    assert rendered.defects == []


@pytest.mark.slow
def test_the_storyboard_in_the_record_is_cut_to_the_voice(rendered):
    out = Path(rendered.output_dir)
    saved = Storyboard.model_validate_json((out / "storyboard.json").read_text(encoding="utf-8"))
    assert saved == rendered.storyboard
    assert rendered.storyboard.total_seconds == pytest.approx(15.0, abs=0.5)
    starts = [t.start_s for t in rendered.line_timings]
    assert starts == sorted(starts) and starts[0] > 0


@pytest.mark.slow
def test_the_record_reads_back_whole(rendered, tmp_path_factory):
    # the module fixture's database is gone with its settings; read the JSON the way the repository does
    clone = type(rendered).model_validate_json(rendered.model_dump_json())
    assert clone == rendered


@pytest.mark.slow
def test_every_agent_leaves_a_report_including_the_new_four(rendered):
    assert {"voice_music", "motion", "overlays", "compose"} <= set(rendered.agent_reports)
    assert "540x960 h264/aac at 12 fps" in rendered.agent_reports["compose"][0]
    assert rendered.agent_reports["voice_music"][0] == "voice: mock/mock"


# -- soft failures -----------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_clip_that_fails_falls_to_pan_and_zoom_and_the_run_still_renders(folder, refuse_models, monkeypatch, tmp_path):
    paid = ScriptedVideoProvider("paid", script=[VideoGenerationTransportError("timed out")])
    service = VideoService([paid, KenBurnsProvider()], tmp_path / "video_cache")
    monkeypatch.setattr("src.agents.video.motion.get_video_service", lambda *a, **k: service)
    # these small test photographs would all be laid out as fit (made here, never by a model); let them fill
    monkeypatch.setattr("src.agents.video.layout.settings", lambda: {
        "fill_max_upscale": 9.0, "fit": {"dim": 0.6, "blur_px": 30, "centre_y": 0.47, "zoom": 0.06}})

    record = workflow.run_video_workflow(folder, offline=True, draft=True)

    assert record.status == "warned" and record.defects == []
    # the forced fill layouts enlarge the photographs, and these flat test pictures barely change under a pan
    assert {i.name for i in record.evaluation.items if i.status in ("warn", "fail")} == {"upscale", "frozen"}
    assert not [i for i in record.evaluation.items if i.hard and i.status == "fail"]
    first = record.clips[0]
    assert (first.provider, first.ok) == ("kenburns", True) and "paid: timed out" in first.reason
    assert record.providers_used["paid"] >= 1 and record.cost_usd > 0
    assert Path(record.video_path).exists()


@pytest.mark.slow
def test_when_chromium_cannot_run_the_text_is_still_drawn_by_the_fallback(folder, refuse_models, monkeypatch, tmp_path):
    monkeypatch.setattr(video_overlay, "_WORKER", tmp_path / "missing_worker.py")
    record = workflow.run_video_workflow(folder, offline=True, draft=True)
    assert record.status == "passed"
    assert any("fallback renderer" in n for n in record.notes)
    assert "fallback" in record.agent_reports["overlays"][0]


@pytest.mark.slow
def test_a_narration_too_long_for_the_video_is_a_defect_and_the_video_runs_over(folder, refuse_models, monkeypatch):
    monkeypatch.setattr("src.models.speech.MockSpeechProvider", lambda: speech_mock.MockSpeechProvider(words_per_second=0.7))

    record = workflow.run_video_workflow(folder, offline=True, draft=True)

    assert record.status == "failed" and Path(record.video_path).exists()      # the gate fails it, and the MP4 is kept
    assert len(record.defects) == 1 and "not sped up" in record.defects[0]
    assert record.actual_duration_s > 15.5
    assert next(i for i in record.evaluation.items if i.name == "duration").status == "fail"
    assert "the quality gate failed" in record.error.lower()


@pytest.mark.slow
def test_with_no_voice_the_video_is_made_silent_and_says_so(folder, refuse_models, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.speech_provider", "silent", raising=False)
    record = workflow.run_video_workflow(folder, offline=True, draft=True)
    assert record.status == "warned" and record.voice == "silent"
    assert next(i for i in record.evaluation.items if i.name == "narration_gaps").status == "warn"
    assert any("No voice was available" in n for n in record.notes)
    assert record.measurements.loudness_lufs <= -60
    assert any("not loudness-normalised" in n for n in record.notes)


# -- the approval step -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_storyboard_is_shown_for_approval_before_it_is_cut_to_the_voice(folder, refuse_models):
    seen = []

    def approve(board: Storyboard) -> Storyboard:
        seen.append(board.model_copy(deep=True))
        edited = board.model_copy(deep=True)
        edited.shots[1].camera = "tilt_up"
        return edited

    record = workflow.run_video_workflow(folder, offline=True, draft=True, approve=approve)

    assert len(seen) == 1 and seen[0].total_seconds == pytest.approx(15.0, abs=0.05)
    assert record.status == "passed" and record.storyboard.shots[1].camera == "tilt_up"
    assert [s.seconds for s in record.storyboard.shots] != [s.seconds for s in seen[0].shots]      # re-timed afterwards


def test_a_bad_approval_stops_before_anything_is_rendered(folder, refuse_models):
    def approve(board: Storyboard) -> Storyboard:
        edited = board.model_copy(deep=True)
        edited.shots[0].image_id = "IMG999"
        return edited

    record = workflow.run_video_workflow(folder, offline=True, draft=True, approve=approve)

    assert record.status == "failed" and "The approved storyboard is not valid" in record.error
    assert record.video_path is None
    assert [s.node for s in record.workflow_evaluation.steps] == workflow._PLAN_STEPS


# -- the one hard failure -------------------------------------------------------------------------------

@pytest.mark.slow
def test_if_no_mp4_can_be_made_the_run_fails_but_keeps_the_plan_and_the_pieces(folder, refuse_models, monkeypatch):
    from src.core.exceptions import RenderingError

    def boom(**kwargs):
        raise RenderingError("ffmpeg failed: out of disk")

    monkeypatch.setattr("src.rendering.video_compose.compose", boom)

    record = workflow.run_video_workflow(folder, offline=True, draft=True)

    assert record.status == "failed" and record.error == "ffmpeg failed: out of disk"
    assert record.video_path is None and record.script is not None
    out = Path(record.output_dir)
    assert (out / "script.json").exists() and (out / "storyboard.json").exists()
    assert list((out / "audio").glob("*.wav")) and list((out / "shots").glob("*.mp4"))
    assert [(s.node, s.status) for s in record.workflow_evaluation.steps][-1] == ("compose", "failed")
    assert repository.get_generation(record.video_id).status == "failed"


# -- shapes -------------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("aspect, size", [("16:9", (960, 540)), ("1:1", (540, 540))])
def test_other_shapes_render_at_their_own_size(folder, refuse_models, aspect, size):
    record = workflow.run_video_workflow(folder, {"aspect": aspect, "duration_s": 10}, offline=True, draft=True)
    info = ffmpeg.probe(record.video_path)
    assert (info.width, info.height) == size and info.duration_s == pytest.approx(10.0, abs=0.5)
    assert record.defects == []


def test_a_plan_only_run_still_stops_at_planned(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, offline=True)
    assert record.status == "planned" and record.video_path is None and not record.clips


# -- the finish: layout, captions, music ---------------------------------------------------------------------------

@pytest.mark.slow
def test_every_photograph_clip_records_its_layout_and_peak_upscale_in_the_record_and_the_plan_file(rendered):
    photos = [c for c in rendered.clips if c.mode != "card"]
    assert photos and all(c.layout in ("fill", "fit") and c.peak_upscale > 0 for c in photos)
    saved = Storyboard.model_validate_json((Path(rendered.output_dir) / "storyboard.json").read_text(encoding="utf-8"))
    assert [(s.layout, s.peak_upscale) for s in saved.shots if s.kind == "real"] == [(c.layout, c.peak_upscale) for c in photos]
    assert any(line.startswith("layout and peak enlargement: shot 1: ") for line in rendered.agent_reports["motion"])


@pytest.mark.slow
def test_the_spoken_lines_are_burned_in_and_the_overlays_were_drawn_a_piece_at_a_time(rendered):
    overlays = Path(rendered.output_dir) / "overlays"
    assert len(list(overlays.glob("caption_*.png"))) == len(rendered.script.lines)
    assert list(overlays.glob("overlay_00_heading.png")) and list(overlays.glob("overlay_00_scrim.png"))
    assert (Path(rendered.output_dir) / "captions.srt").read_text(encoding="utf-8").count("-->") == len(rendered.script.lines)
    assert all(t.spoken for t in rendered.line_timings)


@pytest.mark.slow
def test_with_no_music_provider_able_the_record_says_none_and_why(rendered):
    assert rendered.music_info.provider == "none" and rendered.music == ""
    assert "Music (none): lyria: GEMINI_API_KEY is not set" in rendered.notes


@pytest.mark.slow
def test_music_from_a_provider_is_mixed_in_recorded_and_paid_for_in_the_record(folder, refuse_models, monkeypatch, tmp_path):
    from src.models.music import MusicService
    from tests.video.fakes import FakeMusic

    music = FakeMusic("lyria", seconds=20.0, cost_usd=0.04)
    monkeypatch.setattr("src.agents.video.voice.get_music_service", lambda: MusicService([music], tmp_path / "music_cache"))
    record = workflow.run_video_workflow(folder, offline=True, draft=True)

    assert record.status == "passed" and record.defects == []
    assert (record.music_info.provider, record.music_info.model_id, record.music_info.cost_usd) == ("lyria", "fake-model", 0.04)
    assert Path(record.music).is_file() and record.cost_usd == 0.04
    assert record.music_info.prompt.startswith("Instrumental only, no vocals")
    assert any(line.startswith("music: lyria (fake-model, $0.04)") for line in record.agent_reports["voice_music"])
    assert music.requests[0].duration_s == pytest.approx(record.measurements.duration_s, abs=0.6)
    assert record.measurements.loudness_lufs == pytest.approx(-16.0, abs=1.5)


@pytest.mark.slow
def test_a_run_with_the_grade_and_captions_off_still_renders(folder, refuse_models):
    (folder / "video_input.json").write_text('{"grade": false, "burn_captions": false}', encoding="utf-8")
    record = workflow.run_video_workflow(folder, offline=True, draft=True)
    assert record.status == "passed" and record.defects == []
    assert not list((Path(record.output_dir) / "overlays").glob("caption_*.png"))
    assert record.brief.grade is False and record.brief.burn_captions is False
