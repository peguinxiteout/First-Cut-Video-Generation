"""
Shared Workflow Evaluation instrumentation, used by both
workflows/brochure.py and workflows/blog.py so the "did every agent run
successfully" tracking (see data/schemas.py's WorkflowEvaluation) isn't
duplicated between them.
"""
from __future__ import annotations

import contextvars
import time

from src.data.schemas import AgentStepResult
from src.observability import tracing


def make_step_log_var(name: str) -> contextvars.ContextVar[dict[str, AgentStepResult] | None]:
    """Each workflow gets its OWN ContextVar (not a shared one) so a
    brochure run and a blog run happening concurrently in the same process
    (e.g. two Streamlit sessions) can't cross-contaminate each other's step
    log — see workflows/brochure.py's original docstring for why a
    ContextVar is used here at all instead of a plain module global."""
    return contextvars.ContextVar(name, default=None)


def instrumented(step_name: str, step_log_var: contextvars.ContextVar):
    """Wraps a LangGraph node function to record its outcome (success/
    failure + latency) into both MLflow (tracing.log_step) and the current
    run's Workflow Evaluation step log. Re-raises on failure — this does
    NOT change "a failing node aborts the whole generation" behavior, it
    just records what happened before that exception propagates up to the
    workflow's top-level try/except."""

    def decorator(fn):
        def wrapper(state):
            start = time.perf_counter()
            log = step_log_var.get()
            try:
                result = fn(state)
            except Exception as exc:  # noqa: BLE001 - record, then re-raise unchanged
                latency_ms = (time.perf_counter() - start) * 1000
                if log is not None:
                    log[step_name] = AgentStepResult(
                        node=step_name, status="failed", latency_ms=latency_ms, error=str(exc)
                    )
                tracing.log_step(step_name, latency_ms=latency_ms)
                raise
            latency_ms = (time.perf_counter() - start) * 1000
            if log is not None:
                log[step_name] = AgentStepResult(node=step_name, status="success", latency_ms=latency_ms)
            tracing.log_step(step_name, latency_ms=latency_ms)
            return result

        return wrapper

    return decorator
