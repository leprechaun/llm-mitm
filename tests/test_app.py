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


def _discovering_upstream(model_ids: list[str], requests: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in model_ids]})
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})

    return functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))


def _unreachable_upstream():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("asleep", request=request)

    return functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models():
    with patch.object(openai.httpx, "AsyncClient", _discovering_upstream([])):
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


def _listed_ids(upstream) -> list[str]:
    with patch.object(openai.httpx, "AsyncClient", upstream):
        response = client.get("/v1/models")
    assert response.status_code == 200
    return [model["id"] for model in response.json()["data"]]


def test_list_models_includes_discovered_models_alongside_configured_ones():
    ids = _listed_ids(_discovering_upstream(["gemma:4"]))

    assert "discovering/gemma:4" in ids
    assert "local/test-model" in ids


def test_list_models_hides_upstream_ids_already_configured_under_another_name():
    ids = _listed_ids(_discovering_upstream(["aliased-upstream-id", "test-model"]))

    assert "discovering/aliased-upstream-id" not in ids
    assert ids.count("discovering/test-model") == 1


def test_list_models_still_returns_configured_models_when_upstream_is_unreachable():
    ids = _listed_ids(_unreachable_upstream())

    assert "discovering/test-model" in ids
    assert "local/test-model" in ids


def test_only_discovering_providers_are_asked_for_their_models():
    requests: list[httpx.Request] = []

    _listed_ids(_discovering_upstream([], requests))

    assert [str(request.url) for request in requests] == ["http://discovering:11434/v1/models"]


def test_chat_routes_a_discovered_model_upstream_at_zero_cost():
    requests: list[httpx.Request] = []
    events = []

    with (
        patch.object(openai.httpx, "AsyncClient", _discovering_upstream(["gemma:4"], requests)),
        patch.object(app_module, "record_usage", events.append),
    ):
        response = client.post("/v1/chat/completions", json={"model": "discovering/gemma:4", "messages": _MESSAGES})

    assert response.status_code == 200
    assert response.json()["model"] == "discovering/gemma:4"
    assert json.loads(requests[-1].content)["model"] == "gemma:4"
    assert events[0].model == "discovering/gemma:4"
    assert events[0].provider == "discovering"
    assert events[0].cost_usd == 0


def test_chat_with_a_model_the_upstream_does_not_list_is_not_found_and_never_forwarded():
    requests: list[httpx.Request] = []

    with patch.object(openai.httpx, "AsyncClient", _discovering_upstream(["gemma:4"], requests)):
        response = client.post("/v1/chat/completions", json={"model": "discovering/typo", "messages": _MESSAGES})

    assert response.status_code == 404
    assert [request.method for request in requests] == ["GET"]


def test_chat_cannot_bypass_a_configured_models_pricing_via_its_upstream_id():
    with patch.object(openai.httpx, "AsyncClient", _discovering_upstream(["aliased-upstream-id"])):
        response = client.post(
            "/v1/chat/completions", json={"model": "discovering/aliased-upstream-id", "messages": _MESSAGES}
        )

    assert response.status_code == 404


def test_chat_with_a_discovering_provider_that_is_unreachable_is_not_found():
    with patch.object(openai.httpx, "AsyncClient", _unreachable_upstream()):
        response = client.post("/v1/chat/completions", json={"model": "discovering/gemma:4", "messages": _MESSAGES})

    assert response.status_code == 404


def test_chat_with_a_configured_model_does_not_list_upstream_models():
    requests: list[httpx.Request] = []

    with patch.object(openai.httpx, "AsyncClient", _discovering_upstream([], requests)):
        response = client.post("/v1/chat/completions", json={"model": "discovering/test-model", "messages": _MESSAGES})

    assert response.status_code == 200
    assert [request.method for request in requests] == ["POST"]


def test_chat_with_an_unknown_model_on_a_non_discovering_provider_asks_nobody():
    requests: list[httpx.Request] = []

    with patch.object(openai.httpx, "AsyncClient", _discovering_upstream(["gemma:4"], requests)):
        response = client.post("/v1/chat/completions", json={"model": "local/gemma:4", "messages": _MESSAGES})

    assert response.status_code == 404
    assert requests == []
