"""
Thin MLflow wrapper.

Every MLflow call in this file is defensive on purpose: tracing is pure
observability, and observability must never be able to block the actual
thing it's observing. A brochure or blog generation succeeding or failing
should depend only on the agents/renderer/critic actually doing their job
— never on whether MLflow's tracking backend happens to be reachable,
correctly configured, or (as actually happened) mid-deprecation-cycle and
now refusing to initialize at all. If any MLflow call fails for any
reason, it's logged as a warning and swallowed; generation proceeds
untraced rather than failing outright. See `_safe()` below, and
`trace_generation()` for the one case that needs a bit more care (the
generation's OWN exceptions — a real agent failure, not a tracing
failure — must still propagate normally).

Uses a local SQLite-backed store by default (MLFLOW_TRACKING_URI=
sqlite:///mlflow.db) so nothing extra needs to be running — `mlflow ui`
just points at it when you want to look at runs. This used to default to
the legacy file-based store (./mlruns); MLflow deprecated that backend and
it now refuses to initialize without either migrating existing data
(`mlflow migrate-filestore`) or explicitly opting back in via
MLFLOW_ALLOW_FILE_STORE=true — see the "Observability" section of the
README for the full story. Every function here is also a no-op if
MLFLOW_ENABLED=false, so you can turn tracing off entirely without
touching workflow code.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from src.core.config import settings

logger = logging.getLogger(__name__)

_mlflow = None
_mlflow_unavailable = False  # set True after a failed init attempt, so we don't retry (and re-log) on every call

# Where we fall back to when the configured tracking URI turns out to be a
# store MLflow refuses to open. Deliberately the same default as
# settings.mlflow_tracking_uri — the point is to land on a backend that is
# known to work rather than to invent a second configuration surface.
_FALLBACK_TRACKING_URI = "sqlite:///mlflow.db"


def _safe(description: str, fn) -> None:
    """Runs one MLflow side-effect defensively. Any failure is logged and
    swallowed — see the module docstring for why this matters."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - tracing must never propagate a failure into real workflow code
        logger.warning("MLflow tracing: %s failed (continuing without it): %s", description, exc)


def _try_backend(mlflow, uri: str) -> None:
    """Points MLflow at `uri` and forces the store to actually open.

    set_tracking_uri() alone proves nothing — it just records a string.
    set_experiment() is the first call that instantiates the backing store,
    which is where a rejected backend (the deprecated ./mlruns file store)
    actually raises."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _get_mlflow():
    """Returns a ready-to-use mlflow module, or None if tracing is off/broken.

    A rejected tracking backend used to mean "no tracing at all for the rest
    of this process" — which is how a stale MLFLOW_TRACKING_URI=./mlruns in
    someone's .env quietly turned into an empty Workflow Evaluation panel in
    the UI ("No workflow step data recorded for this generation"). A
    misconfigured URI is recoverable, so we recover: try what's configured,
    and if the store refuses to open, fall back to the SQLite store and say
    so once. Only if the fallback ALSO fails do we give up on tracing."""
    global _mlflow, _mlflow_unavailable  # noqa: PLW0603
    if _mlflow is not None or _mlflow_unavailable or not settings.mlflow_enabled:
        return _mlflow

    try:
        import mlflow  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001 - mlflow not installed at all
        _mlflow_unavailable = True
        logger.warning("MLflow is not importable; tracing disabled (generation proceeds normally): %s", exc)
        return None

    configured = settings.mlflow_tracking_uri
    try:
        _try_backend(mlflow, configured)
        _mlflow = mlflow
        return _mlflow
    except Exception as exc:  # noqa: BLE001 - see module docstring
        if configured == _FALLBACK_TRACKING_URI:
            _mlflow_unavailable = True
            logger.warning(
                "MLflow tracing disabled for this process (generation will still proceed normally): %s", exc
            )
            return None
        logger.warning(
            "MLflow tracking backend %r could not be opened (%s); falling back to %r. "
            "Set MLFLOW_TRACKING_URI=%s in your .env to silence this.",
            configured, exc, _FALLBACK_TRACKING_URI, _FALLBACK_TRACKING_URI,
        )

    try:
        _try_backend(mlflow, _FALLBACK_TRACKING_URI)
        _mlflow = mlflow
    except Exception as exc:  # noqa: BLE001
        _mlflow_unavailable = True
        logger.warning(
            "MLflow tracing disabled for this process (generation will still proceed normally): %s", exc
        )
    return _mlflow


@contextmanager
def trace_generation(generation_id: str, property_id: str) -> Iterator[None]:
    mlflow = _get_mlflow()
    if mlflow is None:
        yield
        return

    # Only the MLflow calls themselves are guarded here — never the
    # `yield` below. A failure starting/logging to a run means "trace
    # this generation untraced"; a failure INSIDE the generation (an
    # agent actually failing) is a real failure and must propagate
    # normally, not get caught and hidden by this except clause.
    run_cm = None
    try:
        run_cm = mlflow.start_run(run_name=f"brochure-{generation_id}")
        run_cm.__enter__()
        mlflow.log_param("generation_id", generation_id)
        mlflow.log_param("property_id", property_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow tracing: could not start a run (continuing without it): %s", exc)
        run_cm = None

    try:
        yield
    finally:
        if run_cm is not None:
            _safe("closing the MLflow run", lambda: run_cm.__exit__(None, None, None))


def log_step(name: str, *, latency_ms: float, **metrics: Any) -> None:
    """Log one workflow node's timing + any numeric metrics (e.g. agent_score)
    against the currently active MLflow run."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    def _do() -> None:
        if mlflow.active_run() is None:
            return
        mlflow.log_metric(f"{name}_latency_ms", latency_ms)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{name}_{key}", value)

    _safe(f"logging step '{name}'", _do)


def log_artifact_file(path: str) -> None:
    mlflow = _get_mlflow()
    if mlflow is None:
        return
    _safe("logging an artifact file", lambda: mlflow.active_run() is not None and mlflow.log_artifact(path))


def log_final_scores(evaluation, *, step: int = 0) -> None:  # noqa: ANN001 - EvaluationResult or BlogEvaluationResult
    """Logs the Critic's scores against the active run. Works for both the
    brochure's EvaluationResult and the blog's BlogEvaluationResult (or any
    similarly-shaped pydantic model) — logs every numeric field generically
    rather than hardcoding one schema's field names, so this doesn't need
    updating each time a new content type's evaluation shape is added.

    Called once per Critic invocation — which can be MORE than once per
    generation, since the revision loop reruns the Critic after each
    revision attempt. `step` should be the revision count, so repeated
    calls show up as a proper series in the MLflow UI instead of colliding.

    gate_result is logged as a TAG, not a param: params are immutable once
    set within a run (MLflow raises if you log the same key with a
    different value again), but gate_result legitimately changes between
    revision attempts (e.g. "revise" -> "pass") — exactly the mutable,
    "current state of the run" case tags are for."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    def _do() -> None:
        if mlflow.active_run() is None:
            return
        data = evaluation.model_dump(mode="json")
        numeric_metrics = {
            key: value
            for key, value in data.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)

        gate_result = data.get("gate_result")
        if gate_result:
            mlflow.set_tag("gate_result", gate_result)

        # Reasoning strings, not numeric metrics — logged as tags, keyed by
        # step so a later revision's reasoning doesn't overwrite an earlier
        # one (tags are a flat key->value map, unlike metrics' built-in step
        # series). This is what makes "why this score" actually inspectable
        # after the fact, not just asked for in the prompt and discarded.
        for key, value in data.items():
            if key.endswith("_reasoning") and value:
                mlflow.set_tag(f"{key}_step{step}", value)

    _safe("logging final scores", _do)
