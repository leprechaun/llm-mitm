import asyncio
import functools
from unittest.mock import patch

import httpx
import pytest

from llm_mitm.config import OllamaModelConfig
from llm_mitm.providers import Usage, ollama


def _cfg(**overrides) -> OllamaModelConfig:
    defaults = {"model": "llama3.1:8b"}
    defaults.update(overrides)
    return OllamaModelConfig(**defaults)


def _capturing_client(requests: list[httpx.Request], response: httpx.Response):
    # The real AsyncClient, so headers it's constructed with still get merged
    # into the outgoing request -- only the network is replaced.
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))


def test_auth_header_empty_when_no_api_key_env_configured():
    assert ollama._auth_header(_cfg()) == {}


def test_auth_header_reads_key_from_configured_env_var(monkeypatch):
    monkeypatch.setenv("MY_OLLAMA_KEY", "sk-test-key")
    assert ollama._auth_header(_cfg(api_key_env="MY_OLLAMA_KEY")) == {"Authorization": "Bearer sk-test-key"}


def test_auth_header_raises_when_configured_env_var_unset(monkeypatch):
    monkeypatch.delenv("MY_OLLAMA_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MY_OLLAMA_KEY"):
        ollama._auth_header(_cfg(api_key_env="MY_OLLAMA_KEY"))


def test_chat_sends_configured_key(monkeypatch):
    monkeypatch.setenv("MY_OLLAMA_KEY", "sk-test-key")
    requests: list[httpx.Request] = []
    response = httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    with patch.object(ollama.httpx, "AsyncClient", _capturing_client(requests, response)):
        asyncio.run(ollama.chat({"model": "ollama/m"}, _cfg(api_key_env="MY_OLLAMA_KEY")))

    assert requests[0].headers["authorization"] == "Bearer sk-test-key"


def test_chat_stream_sends_configured_key(monkeypatch):
    monkeypatch.setenv("MY_OLLAMA_KEY", "sk-test-key")
    requests: list[httpx.Request] = []
    response = httpx.Response(200, text="data: [DONE]\n\n")

    async def run():
        async for _ in ollama.chat_stream({"model": "ollama/m"}, _cfg(api_key_env="MY_OLLAMA_KEY"), Usage()):
            pass

    with patch.object(ollama.httpx, "AsyncClient", _capturing_client(requests, response)):
        asyncio.run(run())

    assert requests[0].headers["authorization"] == "Bearer sk-test-key"


def test_chat_sends_no_auth_header_by_default():
    requests: list[httpx.Request] = []
    response = httpx.Response(200, json={"usage": {}})

    with patch.object(ollama.httpx, "AsyncClient", _capturing_client(requests, response)):
        asyncio.run(ollama.chat({"model": "ollama/m"}, _cfg()))

    assert "authorization" not in requests[0].headers
