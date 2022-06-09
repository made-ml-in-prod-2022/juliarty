import starlette.status
import logging

from fastapi.testclient import TestClient
from app.src.main import app
from app.src.utils import sample_data

client = TestClient(app)

logger = logging.getLogger(__name__)


def test_read_main():
    response = client.get("/")
    assert response.status_code == starlette.status.HTTP_200_OK
    assert response.json() == {"msg": "Hello World"}


def test_health():
    response = client.get("/health")
    logger.info(response.content)
    assert response.status_code == starlette.status.HTTP_200_OK


def test_predict():
    response = client.post("/predict", json=sample_data())
    assert response.status_code == starlette.status.HTTP_200_OK


def test_predict_invalid_data():
    data = sample_data()
    negative_value = -1.5  # all features are positive
    request_data = {k: negative_value for k, v in data.items()}
    response = client.post("/predict", json=request_data)
    assert response.status_code == starlette.status.HTTP_400_BAD_REQUEST
