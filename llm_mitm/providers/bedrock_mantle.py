from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from llm_mitm.config import BedrockMantleModelConfig
from llm_mitm.providers import Usage

_TIMEOUT = httpx.Timeout(600.0)


def _auth_header() -> dict[str, str]:
    # The same long-term Bedrock API key boto3 reads from this env var for
    # bearer-token auth to Converse/Invoke -- bedrock-mantle's
    # OpenAI-compatible endpoints take the identical token as a plain
    # Bearer header, so one credential covers both this provider and the
    # Converse-based "bedrock" provider.
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise RuntimeError(
            "AWS_BEARER_TOKEN_BEDROCK is not set -- required to call Bedrock's "
            "bedrock-mantle endpoint (a long-term Bedrock API key, generated "
            "in the Bedrock console)."
        )
    return {"Authorization": f"Bearer {token}"}


def _base_url(model_cfg: BedrockMantleModelConfig) -> str:
    return f"https://bedrock-mantle.{model_cfg.region}.api.aws{model_cfg.base_path}"


def _payload(body: dict, model_cfg: BedrockMantleModelConfig, stream: bool) -> dict:
    payload = dict(body)
    payload["model"] = model_cfg.model_id
    payload["stream"] = stream
    return payload


async def chat(body: dict, model_cfg: BedrockMantleModelConfig) -> tuple[dict, Usage]:
    payload = _payload(body, model_cfg, stream=False)
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_auth_header()) as client:
        response = await client.post(f"{_base_url(model_cfg)}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    data["model"] = body["model"]
    response_usage = data.get("usage", {})
    # bedrock-mantle has no prompt-caching or service-tier concept, so
    # cache_read_tokens/cache_write_tokens/service_tier just keep Usage's
    # defaults (0, 0, None), same as the Ollama provider.
    usage = Usage(
        prompt_tokens=response_usage.get("prompt_tokens", 0),
        completion_tokens=response_usage.get("completion_tokens", 0),
    )
    return data, usage


async def chat_stream(body: dict, model_cfg: BedrockMantleModelConfig, usage: Usage) -> AsyncIterator[dict]:
    # bedrock-mantle speaks OpenAI's streaming format directly, so this is
    # mostly a passthrough -- rewrite the model name back and skim off
    # usage. Yields structured chunks, not wire text -- app.py owns framing.
    payload = _payload(body, model_cfg, stream=True)
    payload["stream_options"] = {"include_usage": True}
    async with (
        httpx.AsyncClient(timeout=_TIMEOUT, headers=_auth_header()) as client,
        client.stream("POST", f"{_base_url(model_cfg)}/chat/completions", json=payload) as response,
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
