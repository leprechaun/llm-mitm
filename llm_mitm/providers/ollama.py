from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llm_mitm.config import OllamaModelConfig
from llm_mitm.providers import Usage

_TIMEOUT = httpx.Timeout(600.0)


def _payload(body: dict, model_cfg: OllamaModelConfig, stream: bool) -> dict:
    payload = dict(body)
    payload["model"] = model_cfg.model
    payload["stream"] = stream
    return payload


async def chat(body: dict, model_cfg: OllamaModelConfig) -> tuple[dict, Usage]:
    payload = _payload(body, model_cfg, stream=False)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(f"{model_cfg.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    data["model"] = body["model"]
    response_usage = data.get("usage", {})
    # Ollama has no prompt-caching or service-tier concept, so
    # cache_read_tokens/cache_write_tokens/service_tier just keep Usage's
    # defaults (0, 0, None) -- nothing Bedrock-specific to set here.
    usage = Usage(
        prompt_tokens=response_usage.get("prompt_tokens", 0),
        completion_tokens=response_usage.get("completion_tokens", 0),
    )
    return data, usage


async def chat_stream(body: dict, model_cfg: OllamaModelConfig, usage: Usage) -> AsyncIterator[dict]:
    # Ollama already speaks OpenAI's streaming format, so this is mostly a
    # passthrough -- we just rewrite the model name back and skim off usage.
    # Yields structured chunks, not wire text -- app.py owns framing and
    # appends [DONE] itself once this generator ends, so the upstream's own
    # "[DONE]" line just stops the loop rather than being relayed.
    payload = _payload(body, model_cfg, stream=True)
    payload["stream_options"] = {"include_usage": True}
    async with (
        httpx.AsyncClient(timeout=_TIMEOUT) as client,
        client.stream("POST", f"{model_cfg.base_url}/v1/chat/completions", json=payload) as response,
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
