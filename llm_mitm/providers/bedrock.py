from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import boto3

from llm_mitm.config import BedrockModelConfig

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
}


def _client(region: str | None):
    return boto3.client("bedrock-runtime", region_name=region)


def _to_bedrock_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system: list[dict] = []
    converted: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        else:
            text = content or ""
        if role == "system":
            system.append({"text": text})
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": [{"text": text}]})
        # tool-role messages aren't translated yet (tool calling TBD)
    return system, converted


def _inference_config(body: dict) -> dict:
    cfg: dict[str, Any] = {}
    if body.get("max_tokens") is not None:
        cfg["maxTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        cfg["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        cfg["topP"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        cfg["stopSequences"] = stop if isinstance(stop, list) else [stop]
    return cfg


def _converse_kwargs(body: dict, model_cfg: BedrockModelConfig) -> dict:
    system, messages = _to_bedrock_messages(body["messages"])
    kwargs: dict[str, Any] = {
        "modelId": model_cfg.model_id,
        "messages": messages,
        "inferenceConfig": _inference_config(body),
    }
    if system:
        kwargs["system"] = system
    return kwargs


def _openai_response(
    model_name: str, text: str, prompt_tokens: int, completion_tokens: int, finish_reason: str
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def chat(body: dict, model_cfg: BedrockModelConfig) -> tuple[dict, int, int]:
    client = _client(model_cfg.region)
    kwargs = _converse_kwargs(body, model_cfg)
    response = await asyncio.to_thread(client.converse, **kwargs)
    text = "".join(block.get("text", "") for block in response["output"]["message"]["content"])
    usage = response.get("usage", {})
    prompt_tokens = usage.get("inputTokens", 0)
    completion_tokens = usage.get("outputTokens", 0)
    finish_reason = _STOP_REASON_MAP.get(response.get("stopReason"), "stop")
    data = _openai_response(body["model"], text, prompt_tokens, completion_tokens, finish_reason)
    return data, prompt_tokens, completion_tokens


async def chat_stream(body: dict, model_cfg: BedrockModelConfig, usage_holder: dict) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    q: queue.SimpleQueue = queue.SimpleQueue()
    sentinel = object()

    def producer() -> None:
        try:
            client = _client(model_cfg.region)
            kwargs = _converse_kwargs(body, model_cfg)
            response = client.converse_stream(**kwargs)
            for event in response["stream"]:
                loop.call_soon_threadsafe(q.put, event)
        except Exception as exc:  # noqa: BLE001 -- re-raised on the consumer side
            loop.call_soon_threadsafe(q.put, exc)
        finally:
            loop.call_soon_threadsafe(q.put, sentinel)

    threading.Thread(target=producer, daemon=True).start()

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = body["model"]

    def chunk(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})

    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is sentinel:
            break
        if isinstance(event, Exception):
            raise event
        if "contentBlockDelta" in event:
            text = event["contentBlockDelta"].get("delta", {}).get("text", "")
            if text:
                yield chunk({"content": text})
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            usage_holder["prompt_tokens"] = usage.get("inputTokens", 0)
            usage_holder["completion_tokens"] = usage.get("outputTokens", 0)
        elif "messageStop" in event:
            finish_reason = _STOP_REASON_MAP.get(event["messageStop"].get("stopReason"), "stop")
            yield chunk({}, finish_reason)

    yield "data: [DONE]\n\n"
