from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.observability.tracing as tracing_module
from src.core.types import QualityGateResult
from src.data.schemas import EvaluationResult


def _make_evaluation(gate_result: QualityGateResult, score: float) -> EvaluationResult:
    return EvaluationResult(
        factual_consistency=score,
        visual_design_quality=score,
        content_quality=score,
        quality_score=score,
        gate_result=gate_result,
    )


@pytest.fixture(autouse=True)
def _reset_tracing_module_state():
    """Every test gets a clean slate — these are module-level globals."""
    tracing_module._mlflow = None
    tracing_module._mlflow_unavailable = False
    yield
    tracing_module._mlflow = None
    tracing_module._mlflow_unavailable = False


def test_log_final_scores_survives_a_changing_gate_result_across_revisions(tmp_path, monkeypatch):
    """Regression test: the Critic runs more than once per generation when
    a revision is triggered (see workflows/brochure.py's revision loop),
    and gate_result legitimately changes between calls (e.g.
    "revise" -> "pass"). MLflow params are immutable once set within a run
    — logging gate_result as a param crashed on exactly this sequence.
    Real MLflow against a temp local SQLite store, not a mock, since this
    is MLflow's own validation logic being tested, not our wrapper's."""
    monkeypatch.setattr(tracing_module.settings, "mlflow_tracking_uri", f"sqlite:///{tmp_path}/mlflow.db", raising=False)
    monkeypatch.setattr(tracing_module.settings, "mlflow_experiment_name", "test-experiment", raising=False)
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)

    with tracing_module.trace_generation("gen_test123", "PROP-TEST"):
        # First critic pass: fails the quality gate.
        tracing_module.log_final_scores(_make_evaluation(QualityGateResult.REVISE, 60.0), step=0)
        # Revision loop reruns the critic; this time it passes. This exact
        # sequence (same run, gate_result changes) is what used to crash.
        tracing_module.log_final_scores(_make_evaluation(QualityGateResult.PASS, 92.0), step=1)


# ---------------------------------------------------------------------------
# Resilience: tracing must never be able to block a real generation.
#
# These mock MLflow's failure directly rather than relying on whichever
# MLflow version happens to be installed actually raising — this is a
# regression test for a REAL production incident (MLflow deprecated the
# file-based tracking backend and it now refuses to initialize by default,
# which took down brochure generation entirely because trace_generation()
# wrapped the whole workflow with zero error handling), verified directly
# against a real current MLflow install during development, but pinned
# here with mocks so the test doesn't silently stop testing anything if a
# future MLflow version changes its exact failure behavior again.
# ---------------------------------------------------------------------------

def test_mlflow_init_failure_does_not_block_the_generation(monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)

    fake_mlflow = MagicMock()
    fake_mlflow.set_experiment.side_effect = RuntimeError(
        "The filesystem tracking backend (e.g., './mlruns') is in maintenance mode"
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    generation_ran = False
    with tracing_module.trace_generation("gen_x", "PROP-X"):
        generation_ran = True  # this represents the actual business logic

    assert generation_ran is True
    assert tracing_module._mlflow_unavailable is True


def test_mlflow_init_failure_is_not_retried_on_every_call(monkeypatch):
    """Once MLflow fails to initialize, subsequent calls in the same
    process shouldn't keep retrying (and re-logging) the same failure."""
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)

    fake_mlflow = MagicMock()
    fake_mlflow.set_experiment.side_effect = RuntimeError("boom")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    tracing_module._get_mlflow()
    call_count_after_first = fake_mlflow.set_experiment.call_count
    tracing_module._get_mlflow()
    tracing_module._get_mlflow()

    assert fake_mlflow.set_experiment.call_count == call_count_after_first == 1


def test_start_run_failure_does_not_block_the_generation(monkeypatch):
    """MLflow can initialize fine but still fail when actually starting a
    run (e.g. a permissions issue) — that must be just as non-blocking."""
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)

    fake_mlflow = MagicMock()
    fake_mlflow.start_run.side_effect = RuntimeError("could not start run")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    generation_ran = False
    with tracing_module.trace_generation("gen_y", "PROP-Y"):
        generation_ran = True

    assert generation_ran is True


def test_a_real_exception_inside_the_generation_still_propagates_normally(monkeypatch):
    """The critical distinction: trace_generation must swallow ITS OWN
    (MLflow) failures, but never the generation's real failures — an
    actual agent error must still surface as an actual agent error."""
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", False, raising=False)

    with pytest.raises(ValueError, match="a real agent failure"):
        with tracing_module.trace_generation("gen_z", "PROP-Z"):
            raise ValueError("a real agent failure")


def test_log_step_failure_does_not_propagate(monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)
    fake_mlflow = MagicMock()
    fake_mlflow.active_run.return_value = object()
    fake_mlflow.log_metric.side_effect = RuntimeError("boom")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    tracing_module.log_step("some_step", latency_ms=100.0, agent_score=90)  # must not raise


def test_log_final_scores_failure_does_not_propagate(monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)
    fake_mlflow = MagicMock()
    fake_mlflow.active_run.return_value = object()
    fake_mlflow.log_metrics.side_effect = RuntimeError("boom")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    evaluation = _make_evaluation(QualityGateResult.PASS, 90.0)
    tracing_module.log_final_scores(evaluation)  # must not raise


def test_a_rejected_file_store_uri_falls_back_to_sqlite_instead_of_disabling_tracing(monkeypatch):
    """Regression test for the second half of the same incident.

    Making tracing non-blocking stopped generation from *failing*, but a
    stale `MLFLOW_TRACKING_URI=./mlruns` in someone's .env then silently
    turned tracing off for the whole process — which is what produced an
    empty Workflow Evaluation panel ("No workflow step data recorded for
    this generation") in the UI. A bad URI is recoverable, so the wrapper
    must fall back to the working SQLite store rather than give up."""
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)
    monkeypatch.setattr(tracing_module.settings, "mlflow_tracking_uri", "./mlruns", raising=False)

    attempted_uris: list[str] = []
    fake_mlflow = MagicMock()
    fake_mlflow.set_tracking_uri.side_effect = attempted_uris.append

    def _set_experiment(_name):
        # Only the deprecated file store is rejected; sqlite works.
        if attempted_uris[-1] == "./mlruns":
            raise RuntimeError("The filesystem tracking backend (e.g., './mlruns') is in maintenance mode")

    fake_mlflow.set_experiment.side_effect = _set_experiment
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    assert tracing_module._get_mlflow() is fake_mlflow
    assert tracing_module._mlflow_unavailable is False
    assert attempted_uris == ["./mlruns", "sqlite:///mlflow.db"]


def test_fallback_is_not_attempted_when_the_configured_uri_already_is_the_fallback(monkeypatch):
    """No point retrying the identical URI — that would just log the same
    failure twice and double the startup cost of a genuinely broken store."""
    monkeypatch.setattr(tracing_module.settings, "mlflow_enabled", True, raising=False)
    monkeypatch.setattr(tracing_module.settings, "mlflow_tracking_uri", "sqlite:///mlflow.db", raising=False)

    fake_mlflow = MagicMock()
    fake_mlflow.set_experiment.side_effect = RuntimeError("disk is read-only")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    assert tracing_module._get_mlflow() is None
    assert tracing_module._mlflow_unavailable is True
    assert fake_mlflow.set_experiment.call_count == 1


def test_default_tracking_uri_is_not_the_deprecated_file_store():
    """Regression test for the actual root cause: the default used to be
    './mlruns', the legacy file-based backend MLflow deprecated and now
    refuses to initialize without an explicit opt-in. Must default to a
    database-backed URI instead."""
    from src.core.config import Settings

    default_uri = Settings.model_fields["mlflow_tracking_uri"].default
    assert default_uri != "./mlruns"
    assert default_uri.startswith("sqlite:///")
