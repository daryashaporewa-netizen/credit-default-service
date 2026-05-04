"""Минимальный набор тестов для Flask API.

Проверяем самое важное: что эндпоинты существуют, отвечают, валидируют
вход и что A/B-сплит детерминирован и более-менее равномерен.

Запуск из корня проекта:
    pytest -q
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import app as flask_app  # noqa: E402
from src.ab_split import assign_ab_group  # noqa: E402


SAMPLE_FEATURES = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


@pytest.fixture()
def client():
    flask_app.testing = True
    return flask_app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert "v1" in body["model_versions"] and "v2" in body["model_versions"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_predict_versions(client, version):
    r = client.post(
        "/predict",
        json={"user_id": "u1", "model_version": version, "features": SAMPLE_FEATURES},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["model_version"] == version
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_default"] <= 1.0


def test_predict_default_version(client):
    r = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 200
    assert r.get_json()["model_version"] == "v1"


def test_predict_missing_features(client):
    r = client.post("/predict", json={"features": {"LIMIT_BAL": 1}})
    assert r.status_code == 400
    assert "missing features" in r.get_json()["error"]


def test_predict_bad_model_version(client):
    r = client.post(
        "/predict",
        json={"model_version": "v9", "features": SAMPLE_FEATURES},
    )
    assert r.status_code == 400


def test_predict_ab_requires_user_id(client):
    r = client.post("/predict_ab", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 400


def test_predict_ab_stable_for_same_user(client):
    seen = set()
    for _ in range(5):
        r = client.post(
            "/predict_ab",
            json={"user_id": "stable_user_42", "features": SAMPLE_FEATURES},
        )
        assert r.status_code == 200
        body = r.get_json()
        seen.add(body["ab_group"])
        assert body["model_version"] == ("v1" if body["ab_group"] == "A" else "v2")
    assert len(seen) == 1, f"A/B group must be stable for the same user_id, got {seen}"


def test_ab_split_distribution_is_balanced():
    """На 1000 разных user_id ожидаем ≈ 500/500 ± 5%."""
    counter = collections.Counter(assign_ab_group(f"u_{i}") for i in range(1000))
    a, b = counter["A"], counter["B"]
    assert abs(a - b) < 100, f"A/B split is too skewed: A={a}, B={b}"
