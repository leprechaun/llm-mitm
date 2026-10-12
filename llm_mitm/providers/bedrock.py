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


def _content_text(content: str | list[dict] | None) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content or ""


def _to_bedrock_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system: list[dict] = []
    # Flat (role, content_block) pairs, merged into alternating turns below --
    # Bedrock requires strict user/assistant alternation, so e.g. consecutive
    # "tool" result messages must collapse into one user turn.
    turns: list[tuple[str, dict]] = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            text = _content_text(message.get("content"))
            if text:
                system.append({"text": text})
            continue

        if role == "tool":
            text = _content_text(message.get("content"))
            turns.append(
                (
                    "user",
                    {
                        "toolResult": {
                            "toolUseId": message.get("tool_call_id", ""),
                            "content": [{"text": text}],
                        }
                    },
                )
            )
            continue

        if role not in ("user", "assistant"):
            continue

        text = _content_text(message.get("content"))
        if text:
            turns.append((role, {"text": text}))

        if role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                try:
                    tool_input = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                turns.append(
                    (
                        "assistant",
                        {
                            "toolUse": {
                                "toolUseId": call.get("id", ""),
                                "name": fn.get("name", ""),
                                "input": tool_input,
                            }
                        },
                    )
                )

    converted: list[dict] = []
    for role, block in turns:
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].append(block)
        else:
            converted.append({"role": role, "content": [block]})
    return system, converted


def _to_bedrock_tool_config(body: dict) -> dict | None:
    tools = body.get("tools")
    if not tools:
        return None

    tool_specs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        tool_specs.append(
            {
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "inputSchema": {"json": fn.get("parameters") or {"type": "object", "properties": {}}},
                }
            }
        )
    if not tool_specs:
        return None

    tool_choice = body.get("tool_choice")
    if tool_choice == "none":
        return None

    config: dict[str, Any] = {"tools": tool_specs}
    if tool_choice == "required":
        config["toolChoice"] = {"any": {}}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        config["toolChoice"] = {"tool": {"name": tool_choice["function"]["name"]}}
    # "auto" (or unset) -> omit toolChoice, Bedrock defaults to auto
    return config


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


def _apply_prompt_caching(kwargs: dict) -> None:
    # Cache breakpoints mark "everything up to here is worth reusing". Since
    # messages only ever grow by appending, a breakpoint at the end of the
    # current last message will still be a valid prefix match next turn.
    system = kwargs.get("system")
    if system:
        system.append({"cachePoint": {"type": "default"}})

    tool_config = kwargs.get("toolConfig")
    if tool_config and tool_config.get("tools"):
        tool_config["tools"].append({"cachePoint": {"type": "default"}})

    messages = kwargs.get("messages")
    if messages:
        messages[-1]["content"].append({"cachePoint": {"type": "default"}})


def _converse_kwargs(body: dict, model_cfg: BedrockModelConfig) -> dict:
    system, messages = _to_bedrock_messages(body["messages"])
    kwargs: dict[str, Any] = {
        "modelId": model_cfg.model_id,
        "messages": messages,
        "inferenceConfig": _inference_config(body),
    }
    if system:
        kwargs["system"] = system
    tool_config = _to_bedrock_tool_config(body)
    if tool_config:
        kwargs["toolConfig"] = tool_config
    if model_cfg.prompt_caching:
        _apply_prompt_caching(kwargs)
    return kwargs


def _openai_response(
    model_name: str,
    text: str,
    tool_calls: list[dict],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def chat(body: dict, model_cfg: BedrockModelConfig) -> tuple[dict, int, int, int, int]:
    client = _client(model_cfg.region)
    kwargs = _converse_kwargs(body, model_cfg)
    response = await asyncio.to_thread(client.converse, **kwargs)
    content_blocks = response["output"]["message"]["content"]
    text = "".join(block.get("text", "") for block in content_blocks if "text" in block)
    tool_calls = [
        {
            "id": block["toolUse"]["toolUseId"],
            "type": "function",
            "function": {
                "name": block["toolUse"]["name"],
                "arguments": json.dumps(block["toolUse"].get("input", {})),
            },
        }
        for block in content_blocks
        if "toolUse" in block
    ]
    usage = response.get("usage", {})
    prompt_tokens = usage.get("inputTokens", 0)
    completion_tokens = usage.get("outputTokens", 0)
    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
    cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
    finish_reason = _STOP_REASON_MAP.get(response.get("stopReason"), "stop")
    data = _openai_response(body["model"], text, tool_calls, prompt_tokens, completion_tokens, finish_reason)
    return data, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens


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
    # Bedrock's contentBlockIndex also counts text blocks; OpenAI's tool_calls
    # delta index only counts tool calls, so we remap per tool-use block.
    tool_call_indexes: dict[int, int] = {}

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
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]
            tool_use_start = start.get("start", {}).get("toolUse")
            if tool_use_start:
                tool_call_index = len(tool_call_indexes)
                tool_call_indexes[start["contentBlockIndex"]] = tool_call_index
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_call_index,
                                "id": tool_use_start["toolUseId"],
                                "type": "function",
                                "function": {"name": tool_use_start["name"], "arguments": ""},
                            }
                        ]
                    }
                )
        elif "contentBlockDelta" in event:
            block_delta = event["contentBlockDelta"]
            delta = block_delta.get("delta", {})
            if "text" in delta:
                if delta["text"]:
                    yield chunk({"content": delta["text"]})
            elif "toolUse" in delta:
                tool_call_index = tool_call_indexes.get(block_delta.get("contentBlockIndex"), 0)
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_call_index,
                                "function": {"arguments": delta["toolUse"].get("input", "")},
                            }
                        ]
                    }
                )
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            usage_holder["prompt_tokens"] = usage.get("inputTokens", 0)
            usage_holder["completion_tokens"] = usage.get("outputTokens", 0)
            usage_holder["cache_read_tokens"] = usage.get("cacheReadInputTokens", 0)
            usage_holder["cache_write_tokens"] = usage.get("cacheWriteInputTokens", 0)
        elif "messageStop" in event:
            finish_reason = _STOP_REASON_MAP.get(event["messageStop"].get("stopReason"), "stop")
            yield chunk({}, finish_reason)

    yield "data: [DONE]\n\n"
