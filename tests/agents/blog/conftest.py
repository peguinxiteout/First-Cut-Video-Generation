"""Blog tests never write to the developer's MLflow store or app DB."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_mlflow(monkeypatch, tmp_path):
    monkeypatch.setattr("src.core.config.settings.mlflow_enabled", False, raising=False)
    monkeypatch.setattr("src.observability.tracing._mlflow", None, raising=False)
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "blog_tests.db"), raising=False)
