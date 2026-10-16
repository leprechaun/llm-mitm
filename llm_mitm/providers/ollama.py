from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llm_mitm.config import OllamaModelConfig

_TIMEOUT = httpx.Timeout(600.0)


def _payload(body: dict, model_cfg: OllamaModelConfig, stream: bool) -> dict:
    payload = dict(body)
    payload["model"] = model_cfg.model
    payload["stream"] = stream
    return payload


async def chat(body: dict, model_cfg: OllamaModelConfig) -> tuple[dict, int, int, int, int, str | None]:
    payload = _payload(body, model_cfg, stream=False)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(f"{model_cfg.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    data["model"] = body["model"]
    usage = data.get("usage", {})
    # Ollama has no prompt-caching concept, so cache token counts are always
    # 0, and no service-tier concept (Bedrock-only), so that's always None.
    return data, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), 0, 0, None


async def chat_stream(body: dict, model_cfg: OllamaModelConfig, usage_holder: dict) -> AsyncIterator[str]:
    # Ollama already speaks OpenAI's streaming format, so this is mostly a
    # passthrough -- we just rewrite the model name back and skim off usage.
    payload = _payload(body, model_cfg, stream=True)
    payload["stream_options"] = {"include_usage": True}
    done_sent = False
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
                done_sent = True
                yield "data: [DONE]\n\n"
                break
            obj = json.loads(data_str)
            obj["model"] = body["model"]
            usage = obj.get("usage")
            if usage:
                usage_holder["prompt_tokens"] = usage.get("prompt_tokens", 0)
                usage_holder["completion_tokens"] = usage.get("completion_tokens", 0)
            yield f"data: {json.dumps(obj)}\n\n"
    if not done_sent:
        yield "data: [DONE]\n\n"
