import functools
import json
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

import llm_mitm.app as app_module
from llm_mitm.app import app
from llm_mitm.drivers import bedrock_converse, openai

client = TestClient(app)
_MESSAGES = [{"role": "user", "content": "hi"}]


def _fake_upstream(requests: list[httpx.Request], response: httpx.Response):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert {"id": "local/test-model", "object": "model", "owned_by": "local"} in body["data"]
    assert {"id": "remote/test-model", "object": "model", "owned_by": "remote"} in body["data"]


def test_chat_completion_unknown_model():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_chat_completion_missing_messages():
    response = client.post("/v1/chat/completions", json={"model": "local/test-model"})
    assert response.status_code == 400


def test_same_driver_providers_route_to_their_own_endpoints():
    requests: list[httpx.Request] = []
    response = httpx.Response(200, json={"usage": {}})

    with patch.object(openai.httpx, "AsyncClient", _fake_upstream(requests, response)):
        for model in ("local/test-model", "remote/test-model"):
            assert client.post("/v1/chat/completions", json={"model": model, "messages": _MESSAGES}).status_code == 200

    assert str(requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "test-model"
    assert str(requests[1].url) == "http://remote:11434/v1/chat/completions"
    assert json.loads(requests[1].content)["model"] == "remote-upstream-id"


def test_bedrock_converse_provider_routes_through_its_driver():
    fake_client = MagicMock()
    fake_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "hello"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }

    with patch.object(bedrock_converse.boto3, "client", return_value=fake_client) as make_client:
        response = client.post("/v1/chat/completions", json={"model": "bedrock-us/test-model", "messages": _MESSAGES})

    assert response.status_code == 200
    assert response.json()["model"] == "bedrock-us/test-model"
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    make_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")
    assert fake_client.converse.call_args.kwargs["modelId"] == "bedrock-upstream-id"


def test_usage_event_names_the_configured_provider_and_uses_the_models_pricing():
    events = []
    response = httpx.Response(200, json={"usage": {"prompt_tokens": 2, "completion_tokens": 0}})

    with (
        patch.object(openai.httpx, "AsyncClient", _fake_upstream([], response)),
        patch.object(app_module, "record_usage", events.append),
    ):
        client.post("/v1/chat/completions", json={"model": "remote/test-model", "messages": _MESSAGES})

    assert events[0].provider == "remote"
    assert events[0].driver == "openai"
    assert events[0].model == "remote/test-model"
    # 2 prompt tokens at the fixture's $1,000,000 per 1M tokens.
    assert events[0].cost_usd == 2.0
