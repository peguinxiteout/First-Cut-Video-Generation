"""scripts/generate_video.py --plan-only --offline, run in-process."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.core.config import settings
from src.data import video_repository as repository
from tests.video.helpers import make_video_property

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_video.py"


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    """Run the CLI with these arguments, against a temporary data folder and
    database. force_offline() changes global settings, so every one it touches
    is put back afterwards."""
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "videos.db"), raising=False)
    for name in ("gemini_api_key", "openrouter_api_key", "hf_token", "cloudflare_account_id",
                 "cloudflare_api_token", "tavily_api_key", "image_gen_provider", "tools_enabled",
                 "speech_provider", "video_gen_provider", "mlflow_enabled"):
        monkeypatch.setattr(settings, name, getattr(settings, name), raising=False)
    spec = importlib.util.spec_from_file_location("generate_video", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_force_offline = module.force_offline

    def force_offline_but_keep_the_mock_voice():
        real_force_offline()
        settings.speech_provider = "mock"           # the test must not speak through this machine's SAPI

    module.force_offline = force_offline_but_keep_the_mock_voice

    def run(*args: str) -> str:
        monkeypatch.setattr(sys, "argv", ["generate_video.py", *args])
        module.main()
        return capsys.readouterr().out

    run.module = module
    run.real_force_offline = real_force_offline
    return run


@pytest.fixture
def property_id():
    folder = make_video_property(settings.data_dir_resolved / "properties")
    return folder.name


@pytest.fixture
def refuse_models(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was asked")

    for target in ("src.agents.brochure.property_intelligence.call_with_fallback",
                   "src.agents.brochure.vision.call_with_fallback"):
        monkeypatch.setattr(target, refuse)


def test_plan_only_offline_prints_the_plan_and_writes_the_files(cli, property_id, refuse_models):
    out = cli("--property-id", property_id, "--duration", "15", "--aspect", "9:16", "--plan-only", "--offline")

    assert "Status:   planned" in out
    assert out.count("[OK  ]") == 5
    for heading in ("--- Script ---", "--- On screen ---", "--- Storyboard ---"):
        assert heading in out
    assert "hook" in out and "cta" in out and "total 15.0 s, rules planner" in out
    assert "From ₹85 Lakh" in out                                   # the rupee sign prints on any console

    record = repository.list_generations(property_id)[0]
    assert record.status == "planned"
    folder = repository.video_output_dir(record.video_id)
    assert json.loads((folder / "script.json").read_text(encoding="utf-8"))["mode"] == "template"
    assert len(json.loads((folder / "storyboard.json").read_text(encoding="utf-8"))["shots"]) == 5


def test_offline_blanks_every_key_in_this_process_and_asks_no_model(cli, property_id, refuse_models, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "live-looking-key", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "live-looking-key", raising=False)
    monkeypatch.setattr(settings, "cloudflare_api_token", "live-looking-token", raising=False)
    monkeypatch.setattr(settings, "google_cloud_tts_api_key", "live-looking-key", raising=False)
    monkeypatch.setattr(settings, "beatoven_api_key", "live-looking-key", raising=False)

    cli("--property-id", property_id, "--plan-only", "--offline")

    assert (settings.gemini_api_key, settings.openrouter_api_key, settings.cloudflare_api_token) == ("", "", "")
    assert (settings.google_cloud_tts_api_key, settings.beatoven_api_key) == ("", "")
    assert settings.video_gen_provider == "kenburns"
    assert settings.image_gen_provider == "mock" and settings.tools_enabled is False
    assert settings.mlflow_enabled is False


def test_offline_asks_for_the_voice_on_this_machine_and_for_pan_and_zoom(cli):
    cli.real_force_offline()
    assert settings.speech_provider == "offline" and settings.video_gen_provider == "kenburns"


def test_options_given_on_the_command_line_reach_the_plan(cli, property_id, refuse_models):
    out = cli("--property-id", property_id, "--duration", "10", "--aspect", "1:1", "--plan-only", "--offline")
    record = repository.list_generations(property_id)[0]
    assert (record.duration_s, record.aspect) == (10, "1:1") and "total 10.0 s" in out


def test_offline_forces_the_pan_and_zoom_model_whatever_was_asked(cli, property_id, refuse_models):
    cli("--property-id", property_id, "--model", "minimax_h3_max", "--plan-only", "--offline")
    assert repository.list_generations(property_id)[0].provider == "kenburns"


def test_video_input_json_is_used_for_what_the_command_line_leaves_out(cli, property_id, refuse_models):
    folder = settings.data_dir_resolved / "properties" / property_id
    (folder / "video_input.json").write_text(json.dumps({"duration_s": 30, "highlights": ["Rooftop deck"]}),
                                            encoding="utf-8")
    out = cli("--property-id", property_id, "--aspect", "16:9", "--plan-only", "--offline")
    record = repository.list_generations(property_id)[0]
    assert (record.duration_s, record.aspect) == (30, "16:9") and "Rooftop deck." in out


@pytest.mark.slow
def test_without_plan_only_the_video_is_rendered_and_the_result_is_printed(cli, property_id, refuse_models):
    out = cli("--property-id", property_id, "--duration", "10", "--offline", "--draft")

    assert ("Status:   passed" in out or "Status:   warned" in out) and out.count("[OK  ]") == 10
    for heading in ("--- Script ---", "--- Storyboard ---", "--- Result ---", "--- Defects ---", "--- Quality gate: "):
        assert heading in out
    assert "h264/aac, 540x960, 12 fps" in out and "LUFS" in out and "voice:     mock/mock" in out
    assert "total wall time" in out and "  - none" in out
    assert "--- Layout ---" in out and "peak enlargement" in out and "  music:     none" in out
    assert "  cost:      $0.00" in out and "  hook:      " in out
    record = repository.list_generations(property_id)[0]
    assert record.status in ("passed", "warned") and Path(record.video_path).exists() and record.duration_s == 10
    assert (Path(record.output_dir) / "qa.json").exists() and "details:" in out
    assert all(c.layout in ("fill", "fit") and c.peak_upscale > 0 for c in record.clips if c.mode != "card")
    assert record.music_info.provider == "none"
    assert Path(record.thumbnail_path).exists() and Path(record.captions_path).exists()


def test_draft_is_ignored_for_a_plan_and_a_full_size_render_is_the_default(cli, property_id, refuse_models):
    cli("--property-id", property_id, "--plan-only", "--offline", "--draft")
    assert repository.list_generations(property_id)[0].brief.quality == "draft"


def test_an_unknown_property_is_named(cli):
    with pytest.raises(SystemExit, match="No property found at"):
        cli("--property-id", "NO-SUCH-PROPERTY", "--plan-only", "--offline")


def test_an_id_that_escapes_the_data_folder_is_refused(cli):
    with pytest.raises(SystemExit, match="not a valid id"):
        cli("--property-id", "../../etc", "--plan-only", "--offline")


@pytest.mark.parametrize("args", [("--duration", "20"), ("--aspect", "4:3")])
def test_a_bad_duration_or_aspect_is_refused_before_anything_runs(cli, property_id, args):
    with pytest.raises(SystemExit) as caught:
        cli("--property-id", property_id, *args, "--plan-only", "--offline")
    assert caught.value.code == 2


def test_a_failed_plan_prints_the_reason_and_exits_non_zero(cli, property_id, refuse_models, capsys):
    folder = settings.data_dir_resolved / "properties" / property_id
    for picture in (folder / "images").iterdir():
        picture.unlink()
    with pytest.raises(SystemExit) as caught:
        cli("--property-id", property_id, "--plan-only", "--offline")
    assert caught.value.code == 1
    assert "Error: No images found" in capsys.readouterr().out


def test_the_music_line_says_what_was_used_what_it_cost_and_why_not_when_there_is_none(cli):
    from src.data.video_schemas import MusicInfo

    line = cli.module.music_line
    assert line(None) == "none"
    assert line(MusicInfo(provider="none", notes=["lyria: no key", "library: no track"])) == "none (lyria: no key; library: no track)"
    assert line(MusicInfo(provider="lyria", model_id="lyria-3-clip-preview", cost_usd=0.04)) == "lyria lyria-3-clip-preview ($0.04)"
    assert line(MusicInfo(provider="lyria", model_id="m", from_cache=True)) == "lyria m (cached, $0.00)"
    assert line(MusicInfo(provider="library", title="morning", licence="CC0")) == "library morning ($0.00); licence: CC0"


def test_the_property_folder_is_found_whatever_the_case_of_the_id(cli, property_id, refuse_models):
    out = cli("--property-id", property_id.swapcase(), "--plan-only", "--offline")
    assert "Status:   planned" in out and "--- Script ---" in out
