"""Preflight must be trustworthy: it may not crash, and it may not pass a
machine that cannot actually run a demo."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/preflight.py")


def _module():
    spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runs_to_completion_and_reports_rather_than_raising():
    """Here there is no Chromium and no API key, so it must FAIL cleanly —
    exit 1 with reasons, not a traceback."""
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=300)
    assert result.returncode in (0, 1)
    assert "Traceback" not in result.stderr
    assert "Preflight" in result.stdout
    assert ("READY" in result.stdout) or ("NOT READY" in result.stdout)


def test_a_missing_browser_is_blocking_not_a_warning():
    module = _module()
    module.results.clear()
    module.check_browser()
    status, name, detail = module.results[0]
    if status != module.OK:
        assert status == module.FAIL, "a missing browser must block a demo, not warn"
        assert "playwright install" in detail


def test_missing_optional_keys_warn_but_do_not_block():
    from src.core.config import settings

    module = _module()
    module.results.clear()
    fields = ("cloudflare_account_id", "openrouter_api_key", "gemini_api_key")
    original = tuple(getattr(settings, f) for f in fields)
    try:
        settings.cloudflare_account_id, settings.openrouter_api_key = "", ""
        settings.gemini_api_key = "key"
        module.check_keys()
        statuses = {name: status for status, name, _ in module.results}
        assert statuses["GEMINI_API_KEY"] == module.OK
        # Image generation is optional: with no provider configured the
        # pipelines still run, on offline placeholder images.
        assert statuses["Image generation"] == module.WARN
        assert statuses["OPENROUTER_API_KEY"] == module.WARN
    finally:
        for field, value in zip(fields, original):
            setattr(settings, field, value)


def test_missing_gemini_key_degrades_but_does_not_block():
    from src.core.config import settings

    module = _module()
    module.results.clear()
    original = settings.gemini_api_key
    try:
        settings.gemini_api_key = ""
        module.check_keys()
        assert module.results[0][0] == module.WARN      # every agent has an offline path
    finally:
        settings.gemini_api_key = original


def test_bundled_fonts_are_checked():
    module = _module()
    module.results.clear()
    module.check_assets()
    assert module.results[0][0] == module.OK      # fonts ship with the repo


def test_demo_runbook_covers_what_a_demo_needs():
    text = Path("DEMO.md").read_text(encoding="utf-8")
    for section in ["preflight", "Fallbacks", "Timings", "Do not demo", "invent"]:
        assert section.lower() in text.lower(), section
