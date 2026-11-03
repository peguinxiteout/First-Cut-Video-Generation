from __future__ import annotations

from pathlib import Path

from src.core.config import PROJECT_ROOT, Settings


def test_env_file_is_absolute_and_anchored_at_project_root():
    """Regression test: env_file must be an absolute path anchored at the
    project root, not a bare relative ".env" — a relative path only loads
    correctly when the process's current working directory happens to be
    the project root, and silently (no error) falls back to field defaults
    otherwise. This bit us once already: the OPENROUTER_MODEL field default
    pointed at a since-deprecated free model slug, so a CWD mismatch looked
    identical to the user's .env edit "not taking effect"."""
    env_file = Settings.model_config["env_file"]
    assert Path(env_file).is_absolute()
    assert Path(env_file) == PROJECT_ROOT / ".env"


def test_openrouter_model_default_is_not_the_known_dead_slug():
    """The old default (meta-llama/llama-3.1-8b-instruct:free) was pulled
    from OpenRouter's free tier and now 404s. Whatever the current default
    is, it must not regress back to that exact dead value if someone edits
    this field later without checking."""
    assert Settings.model_fields["openrouter_model"].default != "meta-llama/llama-3.1-8b-instruct:free"
