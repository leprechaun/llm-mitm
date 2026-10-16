from fastapi.testclient import TestClient

from llm_mitm.app import _sse_chunk_has_token, app

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
    assert {"id": "ollama/test-model", "object": "model", "owned_by": "ollama"} in body["data"]


def test_chat_completion_unknown_model():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_chat_completion_missing_messages():
    response = client.post("/v1/chat/completions", json={"model": "ollama/test-model"})
    assert response.status_code == 400


def test_sse_chunk_has_token_false_for_role_only_opener():
    assert not _sse_chunk_has_token('data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n')


def test_sse_chunk_has_token_false_for_done_and_finish_reason():
    assert not _sse_chunk_has_token("data: [DONE]\n\n")
    assert not _sse_chunk_has_token('data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n')


def test_sse_chunk_has_token_true_for_content_delta():
    assert _sse_chunk_has_token('data: {"choices": [{"delta": {"content": "hi"}}]}\n\n')


def test_sse_chunk_has_token_true_for_tool_call_delta():
    assert _sse_chunk_has_token(
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1"}]}}]}\n\n'
    )


def test_sse_chunk_has_token_false_for_empty_content():
    assert not _sse_chunk_has_token('data: {"choices": [{"delta": {"content": ""}}]}\n\n')
