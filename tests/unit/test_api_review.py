from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data import repository
from src.data.schemas import GenerationRecord


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.core.config.settings.database_path", str(tmp_path / "test_api.db"), raising=False
    )
    repository.init_db()
    repository.save_generation(
        GenerationRecord(generation_id="gen_apitest1", property_id="PROP-TEST", status="completed")
    )
    return TestClient(app)


def test_review_endpoint_approves_a_generation(client):
    response = client.post(
        "/generations/gen_apitest1/review", json={"status": "approved", "comment": "Looks great"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["human_review"]["status"] == "approved"
    assert body["human_review"]["comment"] == "Looks great"


def test_review_endpoint_rejects_invalid_status(client):
    response = client.post("/generations/gen_apitest1/review", json={"status": "not_a_real_status"})
    assert response.status_code == 400


def test_review_endpoint_404s_for_unknown_generation(client):
    response = client.post("/generations/gen_does_not_exist/review", json={"status": "approved"})
    assert response.status_code == 404


def test_get_generation_reflects_review_after_posting(client):
    client.post("/generations/gen_apitest1/review", json={"status": "revision_requested", "comment": "fix logo"})
    response = client.get("/generations/gen_apitest1")
    assert response.status_code == 200
    assert response.json()["human_review"]["status"] == "revision_requested"
