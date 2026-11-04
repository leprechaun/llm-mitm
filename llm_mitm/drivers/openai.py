from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from llm_mitm.config import ModelConfig, OpenAIProviderConfig
from llm_mitm.drivers import Usage

_TIMEOUT = httpx.Timeout(600.0)
# Short, since a client is waiting on this and a local upstream that's asleep
# just doesn't answer.
_LIST_MODELS_TIMEOUT = httpx.Timeout(2.0)


def _auth_header(provider: OpenAIProviderConfig) -> dict[str, str]:
    if provider.api_key_env is None:
        return {}
    key = os.environ.get(provider.api_key_env)
    if not key:
        raise RuntimeError(f"{provider.api_key_env} is not set -- required by provider '{provider.name}'")
    return {"Authorization": f"Bearer {key}"}


def _payload(body: dict, model: ModelConfig, stream: bool) -> dict:
    payload = dict(body)
    payload["model"] = model.model_id
    payload["stream"] = stream
    return payload


async def list_models(provider: OpenAIProviderConfig) -> list[str]:
    async with httpx.AsyncClient(timeout=_LIST_MODELS_TIMEOUT, headers=_auth_header(provider)) as client:
        response = await client.get(f"{provider.base_url}/models")
        response.raise_for_status()
    return [entry["id"] for entry in response.json()["data"]]


async def chat(body: dict, provider: OpenAIProviderConfig, model: ModelConfig) -> tuple[dict, Usage]:
    payload = _payload(body, model, stream=False)
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_auth_header(provider)) as client:
        response = await client.post(f"{provider.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    data["model"] = body["model"]
    response_usage = data.get("usage", {})
    # The OpenAI format has no prompt-caching or service-tier fields we read,
    # so cache_read_tokens/cache_write_tokens/service_tier just keep Usage's
    # defaults (0, 0, None) -- nothing Bedrock-specific to set here.
    usage = Usage(
        prompt_tokens=response_usage.get("prompt_tokens", 0),
        completion_tokens=response_usage.get("completion_tokens", 0),
    )
    return data, usage


async def chat_stream(
    body: dict, provider: OpenAIProviderConfig, model: ModelConfig, usage: Usage
) -> AsyncIterator[dict]:
    # The upstream already speaks OpenAI's streaming format, so this is mostly
    # a passthrough -- we just rewrite the model name back and skim off usage.
    # Yields structured chunks, not wire text -- app.py owns framing and
    # appends [DONE] itself once this generator ends, so the upstream's own
    # "[DONE]" line just stops the loop rather than being relayed.
    payload = _payload(body, model, stream=True)
    payload["stream_options"] = {"include_usage": True}
    async with (
        httpx.AsyncClient(timeout=_TIMEOUT, headers=_auth_header(provider)) as client,
        client.stream("POST", f"{provider.base_url}/chat/completions", json=payload) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            obj = json.loads(data_str)
            obj["model"] = body["model"]
            event_usage = obj.get("usage")
            if event_usage:
                usage.prompt_tokens = event_usage.get("prompt_tokens", 0)
                usage.completion_tokens = event_usage.get("completion_tokens", 0)
            yield obj
