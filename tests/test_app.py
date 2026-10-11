from fastapi.testclient import TestClient

from llm_mitm.app import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert {"id": "test-model", "object": "model", "owned_by": "ollama"} in body["data"]


def test_chat_completion_unknown_model():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_chat_completion_missing_messages():
    response = client.post("/v1/chat/completions", json={"model": "test-model"})
    assert response.status_code == 400
