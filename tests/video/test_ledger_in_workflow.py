"""The fact ledger cache as the workflow and the command line see it: a hit is in the record's notes,
--refresh-ledger asks again."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.agents.video import fact_ledger
from src.core.config import settings
from src.workflows import video as workflow
from tests.video.helpers import make_video_property

ANSWER = {"facts": {"name": "Aurelia Heights"}, "selling_points": ["Metro nearby"], "specifications": [],
          "legal_notes": [], "rera_number": ""}


@pytest.fixture
def model(tmp_path, monkeypatch):
    """A ledger model that answers and counts, and a vision model that is down (the pictures are read from their files)."""
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "v.db"), raising=False)
    monkeypatch.setattr(settings, "gemini_api_key", "k", raising=False)
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)
    calls = []

    def answer(role, fn_name, prompt):
        calls.append(role)
        return {**ANSWER, "selling_points": [f"Metro nearby ({len(calls)})"]}

    def down(images):
        raise RuntimeError("vision is down")

    monkeypatch.setattr("src.agents.brochure.property_intelligence.call_with_fallback", answer)
    monkeypatch.setattr("src.agents.video.vision_cache.classify_batch", down)
    return calls


def _plan(folder, **kwargs):
    return workflow.run_plan_workflow(folder, draft=True, **kwargs)


def test_a_cache_hit_is_in_the_records_notes_and_the_model_is_not_asked_again(tmp_path, model):
    folder = make_video_property(tmp_path / "properties")
    first, second = _plan(folder), _plan(folder)

    assert model == ["property_intelligence"]
    assert not any(n.startswith(fact_ledger.CACHE_NOTE) for n in first.notes)
    assert any(n.startswith(fact_ledger.CACHE_NOTE) for n in second.notes)
    assert second.agent_reports["fact_ledger"][-1].startswith(fact_ledger.CACHE_NOTE)


def test_the_same_ledger_gives_the_same_script_every_time(tmp_path, model):
    folder = make_video_property(tmp_path / "properties")
    a, b = _plan(folder), _plan(folder)
    assert a.script == b.script and [l.text for l in a.script.lines] == [l.text for l in b.script.lines]


def test_refreshing_asks_the_model_again_and_says_so(tmp_path, model):
    folder = make_video_property(tmp_path / "properties")
    _plan(folder)
    refreshed = _plan(folder, refresh_ledger=True)
    assert model == ["property_intelligence"] * 2
    assert any(n.startswith(fact_ledger.REFRESH_NOTE) for n in refreshed.notes)
    assert not any(n.startswith(fact_ledger.CACHE_NOTE) for n in refreshed.notes)


def test_an_offline_run_says_nothing_about_the_cache(tmp_path, model):
    record = workflow.run_plan_workflow(make_video_property(tmp_path / "properties"), offline=True, draft=True)
    assert model == [] and not any(n.startswith((fact_ledger.CACHE_NOTE, fact_ledger.REFRESH_NOTE)) for n in record.notes)


# -- the command line ------------------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_video.py"


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "videos.db"), raising=False)
    for name in ("gemini_api_key", "openrouter_api_key", "hf_token", "cloudflare_account_id", "cloudflare_api_token",
                 "tavily_api_key", "google_cloud_tts_api_key", "beatoven_api_key", "image_gen_provider", "tools_enabled",
                 "speech_provider", "video_gen_provider", "mlflow_enabled"):
        monkeypatch.setattr(settings, name, getattr(settings, name), raising=False)
    spec = importlib.util.spec_from_file_location("generate_video_ledger", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(*args: str) -> str:
        monkeypatch.setattr(sys, "argv", ["generate_video.py", *args])
        module.main()
        return capsys.readouterr().out

    return run


@pytest.mark.parametrize("flags, refresh", [([], False), (["--refresh-ledger"], True)])
def test_the_refresh_flag_reaches_the_ledger(cli, monkeypatch, flags, refresh):
    folder = make_video_property(settings.data_dir_resolved / "properties")
    seen = []
    real = fact_ledger.run

    def watching(*args, **kwargs):
        seen.append(kwargs.get("refresh"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fact_ledger, "run", watching)
    out = cli("--property-id", folder.name, "--plan-only", "--offline", *flags)
    assert seen == [refresh] and "Status:   planned" in out
