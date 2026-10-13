
import pytest
from fastapi.testclient import TestClient

from nids.inference import api


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_prediction_rejects_missing_features(client):
    response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422


def test_prediction_with_valid_feature_schema(client):
    assert api.feature_columns

    features = {
        name: 0.0
        for name in api.feature_columns
    }

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_class"] in api.classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_prediction_rejects_unexpected_feature(client):
    response = client.post(
        "/predict",
        json={"features": {"unexpected_feature": 1.0}},
    )

    assert response.status_code == 422
