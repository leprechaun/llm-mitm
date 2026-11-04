import asyncio
import functools
import json
from unittest.mock import patch

import httpx
import pytest

from llm_mitm.config import ModelConfig, OpenAIOptions, OpenAIProviderConfig
from llm_mitm.drivers import Usage, openai

_BODY = {"model": "virty/llama3", "messages": [{"role": "user", "content": "hi"}]}
_MODEL = ModelConfig(name="llama3", provider="virty", model_id="llama3.1:8b", options=OpenAIOptions())


def _provider(**overrides) -> OpenAIProviderConfig:
    defaults = {"name": "virty", "driver": "openai", "base_url": "http://virty:11434/v1"}
    defaults.update(overrides)
    return OpenAIProviderConfig(**defaults)


def _fake_upstream(requests: list[httpx.Request], response: httpx.Response):
    # The real AsyncClient, so headers it's constructed with still get merged
    # into the outgoing request -- only the network is replaced.
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))


def _chat(provider: OpenAIProviderConfig, response: httpx.Response) -> tuple[list[httpx.Request], dict, Usage]:
    requests: list[httpx.Request] = []
    with patch.object(openai.httpx, "AsyncClient", _fake_upstream(requests, response)):
        data, usage = asyncio.run(openai.chat(_BODY, provider, _MODEL))
    return requests, data, usage


def _chat_stream(provider: OpenAIProviderConfig, response: httpx.Response) -> tuple[list[httpx.Request], list[dict], Usage]:
    requests: list[httpx.Request] = []
    usage = Usage()

    async def run() -> list[dict]:
        return [chunk async for chunk in openai.chat_stream(_BODY, provider, _MODEL, usage)]

    with patch.object(openai.httpx, "AsyncClient", _fake_upstream(requests, response)):
        chunks = asyncio.run(run())
    return requests, chunks, usage


def _sse(*payloads: dict) -> httpx.Response:
    lines = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    return httpx.Response(200, text="".join(lines) + "data: [DONE]\n\n")


def test_chat_posts_upstream_model_id_to_the_providers_base_url():
    response = httpx.Response(200, json={"usage": {}})

    requests, _, _ = _chat(_provider(base_url="https://bedrock-mantle.us-west-2.api.aws/openai/v1"), response)

    assert str(requests[0].url) == "https://bedrock-mantle.us-west-2.api.aws/openai/v1/chat/completions"
    sent = json.loads(requests[0].content)
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    assert sent["messages"] == _BODY["messages"]


def test_chat_returns_client_facing_model_name_and_usage():
    response = httpx.Response(200, json={"model": "llama3.1:8b", "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    _, data, usage = _chat(_provider(), response)

    assert data["model"] == "virty/llama3"
    assert usage == Usage(prompt_tokens=7, completion_tokens=3)


def test_chat_stream_requests_usage_and_posts_to_the_providers_base_url():
    requests, _, _ = _chat_stream(_provider(), _sse())

    assert str(requests[0].url) == "http://virty:11434/v1/chat/completions"
    sent = json.loads(requests[0].content)
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


def test_chat_stream_relays_chunks_under_client_facing_name_and_records_usage():
    response = _sse(
        {"model": "llama3.1:8b", "choices": [{"delta": {"content": "hi"}}]},
        {"model": "llama3.1:8b", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    )

    _, chunks, usage = _chat_stream(_provider(), response)

    assert [chunk["model"] for chunk in chunks] == ["virty/llama3", "virty/llama3"]
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert usage == Usage(prompt_tokens=7, completion_tokens=3)


def test_chat_sends_no_auth_header_when_provider_has_no_api_key_env():
    requests, _, _ = _chat(_provider(), httpx.Response(200, json={"usage": {}}))

    assert "authorization" not in requests[0].headers


def test_chat_sends_key_from_providers_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-test-key")

    requests, _, _ = _chat(_provider(api_key_env="MY_KEY"), httpx.Response(200, json={"usage": {}}))

    assert requests[0].headers["authorization"] == "Bearer sk-test-key"


def test_chat_stream_sends_key_from_providers_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-test-key")

    requests, _, _ = _chat_stream(_provider(api_key_env="MY_KEY"), _sse())

    assert requests[0].headers["authorization"] == "Bearer sk-test-key"


def test_chat_fails_naming_env_var_and_provider_when_key_is_unset(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MY_KEY.*virty"):
        _chat(_provider(api_key_env="MY_KEY"), httpx.Response(200, json={"usage": {}}))


def test_chat_raises_on_upstream_error_status():
    with pytest.raises(httpx.HTTPStatusError):
        _chat(_provider(), httpx.Response(500, json={}))


def test_chat_does_not_mutate_the_clients_request_body():
    _chat(_provider(), httpx.Response(200, json={"usage": {}}))

    assert _BODY["model"] == "virty/llama3"


def test_list_models_returns_upstream_ids_from_the_providers_base_url():
    requests: list[httpx.Request] = []
    response = httpx.Response(200, json={"object": "list", "data": [{"id": "llama3.1:8b"}, {"id": "gemma:4"}]})

    with patch.object(openai.httpx, "AsyncClient", _fake_upstream(requests, response)):
        ids = asyncio.run(openai.list_models(_provider(base_url="http://mac:8000/api/v1")))

    assert ids == ["llama3.1:8b", "gemma:4"]
    assert str(requests[0].url) == "http://mac:8000/api/v1/models"


def test_list_models_sends_the_providers_api_key(monkeypatch):
    monkeypatch.setenv("OMLX_API_KEY", "secret")
    requests: list[httpx.Request] = []
    response = httpx.Response(200, json={"data": []})

    with patch.object(openai.httpx, "AsyncClient", _fake_upstream(requests, response)):
        asyncio.run(openai.list_models(_provider(api_key_env="OMLX_API_KEY")))

    assert requests[0].headers["authorization"] == "Bearer secret"


def test_list_models_raises_on_upstream_error():
    response = httpx.Response(500, request=httpx.Request("GET", "http://virty:11434/v1/models"))

    with (
        patch.object(openai.httpx, "AsyncClient", _fake_upstream([], response)),
        pytest.raises(httpx.HTTPStatusError),
    ):
        asyncio.run(openai.list_models(_provider()))
