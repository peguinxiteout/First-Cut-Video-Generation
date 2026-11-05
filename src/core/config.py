"""
Central configuration: environment variables (via pydantic-settings) plus the
YAML files under configs/. Import `settings` and `load_yaml_config(...)` from
here rather than reading .env or YAML directly anywhere else in the codebase.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Anchored to the project root (not a bare relative ".env") so .env
        # loads correctly no matter what directory the app is launched from
        # — a relative path here silently fails to load if the process's
        # working directory isn't the project root, falling back to field
        # defaults with no error or warning.
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"

    # Gemini
    gemini_api_key: str = ""
    gemini_text_model: str = "gemini-3.6-flash"
    # Deliberately FOUR different model names below, not just different-
    # sounding variables — Gemini's free-tier daily quota is tracked per
    # (project, model), so property_intelligence/copywriter (TEXT),
    # vision (VISION), creative_director/blog_seo (FAST), and critic
    # (CRITIC) each land in their own quota bucket instead of stacking
    # onto one. This used to be a stated intent that configs/models.yaml
    # didn't actually implement — vision and critic both pointed at
    # GEMINI_VISION_MODEL, silently sharing one 20-requests/day bucket
    # between two different agents. See the "Free-tier API budgeting"
    # section of the README for how this surfaces (a 429 on critic right
    # after vision succeeded, despite text/fast quota being untouched).
    gemini_vision_model: str = "gemini-3.5-flash"
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    gemini_critic_model: str = "gemini-3.7-flash"

    # OpenRouter
    openrouter_api_key: str = ""
    # openrouter/free is OpenRouter's own maintained router, not a specific
    # model — it randomly selects from whatever's currently free on
    # OpenRouter and filters for the features the request needs (image
    # understanding, for our vision/critic calls). This deliberately
    # replaces pinning to one specific free model slug (e.g.
    # "minimax/minimax-m3:free"), because those rotate out of the free
    # tier without warning — exactly what happened here (OpenRouter began
    # returning 404 "unavailable for free" for that exact slug). Letting
    # OpenRouter pick avoids re-discovering a working slug every time one
    # gets discontinued. If you want a specific model instead, verify it
    # first at https://openrouter.ai/models?max_price=0 — and avoid a
    # Google-hosted model (e.g. a Gemma slug), since OpenRouter's free
    # Gemma models are served via "Google AI Studio" infrastructure and
    # can share the same upstream congestion/quota as the Gemini primary,
    # defeating the point of a fallback.
    openrouter_model: str = "openrouter/free"

    # Only read if configs/models.yaml's vision role is switched to
    # provider: openrouter — see the comment there. Empty by default
    # (an experimental opt-in, not a default behavior change).
    openrouter_vision_model: str = ""

    # Image generation. Read lazily by src/models/image_gen — missing
    # credentials must NEVER break app startup; they only disable image
    # generation, and the pipelines carry on with the uploaded photos alone.
    #
    # Cloudflare Workers AI is the primary: a recurring, card-free daily
    # allowance (10,000 neurons ~= 170 FLUX images a day) running the same
    # FLUX.1-schnell and SDXL weights this project already targeted.
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    # Tried in order; the first that succeeds wins. A request that is not
    # square skips flux-1-schnell automatically, because that model has no
    # width/height in its schema and silently returns a square image.
    cloudflare_image_models: str = (
        "@cf/black-forest-labs/flux-1-schnell,@cf/stabilityai/stable-diffusion-xl-base-1.0")

    # OpenRouter is the fallback, reached only when Cloudflare's allowance
    # for the day is spent. A different vendor rather than another model,
    # because what ended the previous provider was an ACCOUNT running out,
    # which no amount of model-switching survives. Uses the OPENROUTER_API_KEY
    # this project already has for the text fallback; note that image
    # requests there are paid and need >$1 of credit.
    openrouter_image_models: str = "black-forest-labs/flux-1.1-pro"

    # Hugging Face Inference Providers. NO LONGER USED BY DEFAULT: this
    # account's monthly included credits ran out and every model began
    # returning 402 Payment Required. Kept selectable
    # (IMAGE_GEN_PROVIDER=huggingface) for anyone who buys credits, but
    # "auto" will never choose it.
    hf_token: str = ""
    # "auto" -> Cloudflare, then OpenRouter, then offline previews.
    # "cloudflare" / "openrouter" / "huggingface" / "mock" -> force one.
    # "none" -> generation disabled entirely.
    image_gen_provider: str = "auto"
    image_gen_models: str = "black-forest-labs/FLUX.1-schnell,stabilityai/stable-diffusion-xl-base-1.0"
    # auto = match the medium of the property's own images (see
    # prompting.dominant_medium); or force "cgi_render" / "photograph" when
    # a project's assets are all renders and you want generated visuals to
    # match without relying on detection.
    image_gen_medium: str = "auto"
    image_gen_max_retries: int = 4
    image_gen_timeout_s: float = 120.0
    # The whole of ONE provider's attempt — every model, every retry.
    # Retries used to be counted and not timed, so four attempts across two
    # models at a 120s read timeout was sixteen minutes of a progress bar
    # going nowhere before the chain even tried the second vendor.
    image_gen_budget_s: float = 210.0

    # --- Video generation ---------------------------------------------------
    # "auto" -> the chain in configs/video/speech.yaml: Google Cloud TTS, Gemini
    # TTS, a voice on this machine, then a silent track timed to the script.
    # A name forces one.
    speech_provider: str = "auto"
    google_cloud_tts_api_key: str = ""
    # Music: the chain in configs/video/music.yaml; Beatoven first.
    beatoven_api_key: str = ""
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    gemini_tts_voice: str = "en-in-commercial-8"
    # "auto" -> the first available provider in configs/video/providers.yaml,
    # then pan-and-zoom. A provider id forces one; "kenburns" never calls a model.
    video_gen_provider: str = "auto"

    # --- Research tools (blog generation) --------------------------------
    # The master switch. False makes every external tool report itself
    # unavailable instead of reaching the network, which is what the test
    # suite sets: a research tool that quietly called a live API during
    # tests would make the suite slow, flaky and dependent on somebody
    # else's uptime.
    tools_enabled: bool = True
    tools_timeout_s: float = 12.0
    # Optional. Tavily returns cleaner, already-extracted search results,
    # so it is preferred when a key exists; without one the same capability
    # is served by DuckDuckGo, which needs no account.
    tavily_api_key: str = ""

    # Storage
    database_path: str = "./data/app.db"
    data_dir: str = "./data"

    # MLflow
    # sqlite:///mlflow.db, not the legacy "./mlruns" file-based store —
    # MLflow deprecated the file backend (it now refuses to initialize
    # without either migrating existing data via `mlflow migrate-filestore`
    # or explicitly opting back in via MLFLOW_ALLOW_FILE_STORE=true). Even
    # so, every MLflow call in src/observability/tracing.py is defensive —
    # a tracking-backend problem should never be able to block an actual
    # generation, whatever this is pointed at.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "agentic-content-platform"
    mlflow_enabled: bool = True

    # Delay between successive Vision Agent image calls, to stay under
    # free-tier requests-per-minute limits when classifying many photos in
    # one generation. 12 photos with zero pacing can blow through a ~10 RPM
    # free-tier limit in seconds, cascading into fallback-provider rate
    # limits too. Set to 0 on a paid tier / generous rate limits.
    vision_call_delay_seconds: float = 4.0

    # Playwright browser channel for PDF/screenshot rendering. Empty (default)
    # = Playwright's own downloaded Chromium build. Set to "msedge" or
    # "chrome" to use an already-installed SYSTEM browser instead — needed
    # on Windows machines running Application Control policies (WDAC/
    # AppLocker) that block execution of `playwright install`-downloaded
    # binaries but virtually always already allow system-installed Edge.
    playwright_browser_channel: str = ""
    # Rendering is per-page work: a long brochure with many photographs
    # legitimately takes longer than a short one. Raise this rather than
    # dropping pages if a very large brochure times out.
    playwright_timeout_seconds: int = 300

    default_model_provider: str = "gemini"

    @property
    def database_path_resolved(self) -> Path:
        return (PROJECT_ROOT / self.database_path).resolve()

    @property
    def data_dir_resolved(self) -> Path:
        return (PROJECT_ROOT / self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


@lru_cache
def load_yaml_config(relative_path: str) -> dict[str, Any]:
    """Load and cache a YAML file from configs/, e.g. 'models.yaml' or
    'brochure/prompts.yaml'."""
    path = CONFIGS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
