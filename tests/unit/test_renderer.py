from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.exceptions import RenderingError
from src.rendering.renderer import group_digits, is_pdf_render_valid, qr_code_data_uri, render_pdf_and_screenshot


def _fake_completed_process(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_render_pdf_and_screenshot_succeeds_on_zero_exit(tmp_path, monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        # A real successful worker run would have written both files —
        # simulate that so the new post-render validity check passes too.
        pdf_path = Path(cmd[3])
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 fake but large enough" + b"0" * 6000)
        Path(cmd[4]).write_bytes(b"fake png bytes")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>")
    pdf_path = tmp_path / "out" / "brochure.pdf"
    screenshot_path = tmp_path / "out" / "screenshot.png"

    render_pdf_and_screenshot(html_path, pdf_path, screenshot_path)  # should not raise

    # Confirms it really does shell out to the standalone worker script,
    # not call Playwright in-process.
    assert "_playwright_worker.py" in calls["cmd"][1]
    assert str(html_path) in calls["cmd"]
    assert str(pdf_path) in calls["cmd"]
    assert str(screenshot_path) in calls["cmd"]
    # Output directories must exist even though the (mocked) worker never ran.
    assert pdf_path.parent.exists()


def test_render_pdf_and_screenshot_raises_clear_error_when_pdf_missing_despite_zero_exit(tmp_path, monkeypatch):
    """Regression test: a subprocess that reports success (exit code 0) but
    doesn't actually produce a valid PDF must not be treated as a
    successful render — this is exactly the 'broken or blank result'
    scenario the person asked to guard against."""

    def fake_run(cmd, **kwargs):
        return _fake_completed_process(returncode=0)  # no PDF file actually written

    monkeypatch.setattr(subprocess, "run", fake_run)

    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>")
    pdf_path = tmp_path / "out" / "brochure.pdf"
    screenshot_path = tmp_path / "out" / "screenshot.png"

    with pytest.raises(RenderingError, match="missing or unexpectedly small"):
        render_pdf_and_screenshot(html_path, pdf_path, screenshot_path)


def test_render_pdf_and_screenshot_raises_clear_error_when_pdf_is_tiny(tmp_path, monkeypatch):
    """Same scenario, but the file exists and is non-empty — just far too
    small to be a genuine multi-page brochure PDF (e.g. a truncated write)."""

    def fake_run(cmd, **kwargs):
        pdf_path = Path(cmd[3])
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 tiny")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>")
    pdf_path = tmp_path / "out" / "brochure.pdf"
    screenshot_path = tmp_path / "out" / "screenshot.png"

    with pytest.raises(RenderingError, match="missing or unexpectedly small"):
        render_pdf_and_screenshot(html_path, pdf_path, screenshot_path)


def test_is_pdf_render_valid_true_for_a_real_sized_pdf(tmp_path):
    pdf_path = tmp_path / "brochure.pdf"
    pdf_path.write_bytes(b"%PDF-1.4" + b"0" * 6000)
    assert is_pdf_render_valid(pdf_path) is True


def test_is_pdf_render_valid_false_for_missing_file(tmp_path):
    assert is_pdf_render_valid(tmp_path / "nope.pdf") is False


def test_is_pdf_render_valid_false_for_tiny_file(tmp_path):
    pdf_path = tmp_path / "brochure.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 tiny")
    assert is_pdf_render_valid(pdf_path) is False


def test_render_pdf_and_screenshot_raises_rendering_error_on_nonzero_exit(tmp_path, monkeypatch):
    from src.core.exceptions import RenderingError

    def fake_run(cmd, **kwargs):
        return _fake_completed_process(returncode=1, stderr="Playwright rendering failed: boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>")

    import pytest

    with pytest.raises(RenderingError, match="boom"):
        render_pdf_and_screenshot(html_path, tmp_path / "brochure.pdf", tmp_path / "screenshot.png")


def test_render_pdf_and_screenshot_raises_rendering_error_on_timeout(tmp_path, monkeypatch):
    from src.core.exceptions import RenderingError

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)

    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>")

    import pytest

    with pytest.raises(RenderingError, match="timed out"):
        render_pdf_and_screenshot(html_path, tmp_path / "brochure.pdf", tmp_path / "screenshot.png")


def test_playwright_worker_script_is_syntactically_valid():
    """Sanity check that the standalone worker script at least parses —
    it's invoked via subprocess so nothing else in the test suite imports
    (and therefore syntax-checks) it."""
    import ast

    from src.rendering.renderer import _PLAYWRIGHT_WORKER

    assert _PLAYWRIGHT_WORKER.exists()
    ast.parse(_PLAYWRIGHT_WORKER.read_text(encoding="utf-8"))


def test_render_pdf_and_screenshot_passes_browser_channel_when_set(tmp_path, monkeypatch):
    """Regression test for the WDAC/AppLocker workaround: when
    PLAYWRIGHT_BROWSER_CHANNEL is set, it must reach the worker subprocess
    as an extra CLI arg so it can use a system-installed browser instead of
    a Playwright-downloaded one."""
    from src.core.config import settings

    monkeypatch.setattr(settings, "playwright_browser_channel", "msedge", raising=False)

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        pdf_path = Path(cmd[3])
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4" + b"0" * 6000)
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>")

    render_pdf_and_screenshot(html_path, tmp_path / "brochure.pdf", tmp_path / "screenshot.png")

    # The channel is no longer the last argument: a render mode follows it.
    assert "msedge" in calls["cmd"]
    assert calls["cmd"][-1] in ("pages", "measure")


def test_qr_code_data_uri_returns_a_valid_png_data_uri():
    uri = qr_code_data_uri("https://example.com")
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")
    # Decodes to real PNG bytes, not garbage.
    import base64
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_qr_code_data_uri_returns_none_for_empty_input():
    assert qr_code_data_uri("") is None
    assert qr_code_data_uri(None) is None


def test_qr_code_data_uri_recolors_to_the_requested_dark_color():
    """A QR code that stays pure black regardless of the template's own
    ink color looks like a bolted-on generic graphic, not part of the
    design — confirms the recolor step actually changes the pixels."""
    default_uri = qr_code_data_uri("https://example.com")
    recolored_uri = qr_code_data_uri("https://example.com", dark_color="#B89B5E")
    assert default_uri != recolored_uri




def test_group_digits_groups_facts_only_once():
    assert group_digits("5789 sq ft") == "5,789 sq ft"
    assert group_digits(group_digits("5789 sq ft")) == "5,789 sq ft"
    assert group_digits("537.81 sq m") == "537.81 sq m"
