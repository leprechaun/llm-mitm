from __future__ import annotations

import asyncio
import base64
import json
import queue
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import boto3
import httpx

from llm_mitm.config import BedrockModelConfig
from llm_mitm.providers import Usage

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
}

_DATA_URI_RE = re.compile(r"^data:image/(?P<fmt>[\w.+-]+);base64,(?P<data>.+)$", re.DOTALL)
_IMAGE_FORMAT_ALIASES = {"jpg": "jpeg"}
# Formats Bedrock Converse actually accepts for an "image" content block.
_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "gif", "webp"}
_IMAGE_FETCH_TIMEOUT = httpx.Timeout(30.0)


def _client(region: str | None):
    return boto3.client("bedrock-runtime", region_name=region)


def _content_text(content: str | list[dict] | None) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content or ""


def _normalize_image_format(fmt: str) -> str:
    fmt = _IMAGE_FORMAT_ALIASES.get(fmt.lower(), fmt.lower())
    if fmt not in _SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"unsupported image format '{fmt}' for Bedrock (supported: {sorted(_SUPPORTED_IMAGE_FORMATS)})"
        )
    return fmt


def _image_format_from_content_type(content_type: str) -> str:
    media_type = content_type.split(";")[0].strip()
    return media_type.split("/")[-1] if "/" in media_type else ""


def _image_block_from_url(url: str) -> dict:
    # OpenAI vision clients send either a data: URI (the image embedded
    # inline) or a plain http(s) URL (fetched here, since Bedrock's image
    # block only accepts raw bytes, not a remote URL).
    match = _DATA_URI_RE.match(url)
    if match:
        image_format = _normalize_image_format(match.group("fmt"))
        image_bytes = base64.b64decode(match.group("data"))
    else:
        response = httpx.get(url, timeout=_IMAGE_FETCH_TIMEOUT)
        response.raise_for_status()
        image_bytes = response.content
        fmt_guess = _image_format_from_content_type(response.headers.get("content-type", ""))
        if not fmt_guess:
            path = urlparse(url).path
            fmt_guess = path.rsplit(".", 1)[-1] if "." in path else ""
        image_format = _normalize_image_format(fmt_guess)
    return {"image": {"format": image_format, "source": {"bytes": image_bytes}}}


def _content_blocks(content: str | list[dict] | None) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    blocks: list[dict] = []
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text", "")
            if text:
                blocks.append({"text": text})
        elif part_type == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url:
                blocks.append(_image_block_from_url(url))
    return blocks


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

        for block in _content_blocks(message.get("content")):
            turns.append((role, block))

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

    # toolConfig must stay present whenever tools were defined, even for
    # tool_choice "none" -- Bedrock rejects any request whose message history
    # contains toolUse/toolResult blocks (i.e. a prior tool call) unless
    # toolConfig is defined on *every* subsequent turn, including turns that
    # aren't allowed to call a tool themselves. Bedrock's toolChoice has no
    # "never call a tool" option (only auto/any/a named tool), so "none"
    # degrades to the same as "auto" -- the model could in principle still
    # choose to call a tool, but that's the best available mapping; the
    # alternative (dropping toolConfig) is a hard validation error.
    tool_choice = body.get("tool_choice")
    config: dict[str, Any] = {"tools": tool_specs}
    if tool_choice == "required":
        config["toolChoice"] = {"any": {}}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        config["toolChoice"] = {"tool": {"name": tool_choice["function"]["name"]}}
    # "auto", "none", or unset -> omit toolChoice, Bedrock defaults to auto
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


def _wants_tool_config(body: dict, model_cfg: BedrockModelConfig) -> bool:
    return bool(model_cfg.supports_tools and _to_bedrock_tool_config(body))


def _converse_kwargs(body: dict, model_cfg: BedrockModelConfig) -> dict:
    system, messages = _to_bedrock_messages(body["messages"])
    kwargs: dict[str, Any] = {
        "modelId": model_cfg.model_id,
        "messages": messages,
        "inferenceConfig": _inference_config(body),
    }
    if system:
        kwargs["system"] = system
    if _wants_tool_config(body, model_cfg):
        kwargs["toolConfig"] = _to_bedrock_tool_config(body)
    if model_cfg.prompt_caching:
        _apply_prompt_caching(kwargs)
    if model_cfg.service_tier:
        kwargs["serviceTier"] = {"type": model_cfg.service_tier}
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


async def chat(body: dict, model_cfg: BedrockModelConfig) -> tuple[dict, Usage]:
    client = _client(model_cfg.region)
    # Off the event loop: may do a blocking fetch for a remote image_url.
    kwargs = await asyncio.to_thread(_converse_kwargs, body, model_cfg)
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
    response_usage = response.get("usage", {})
    usage = Usage(
        prompt_tokens=response_usage.get("inputTokens", 0),
        completion_tokens=response_usage.get("outputTokens", 0),
        cache_read_tokens=response_usage.get("cacheReadInputTokens", 0),
        cache_write_tokens=response_usage.get("cacheWriteInputTokens", 0),
        # The tier that actually served the request -- can differ from the
        # requested tier (e.g. "flex" falling back to Standard capacity), so
        # billing must read this instead of trusting model_cfg.service_tier.
        service_tier=response.get("serviceTier", {}).get("type"),
    )
    finish_reason = _STOP_REASON_MAP.get(response.get("stopReason"), "stop")
    data = _openai_response(
        body["model"], text, tool_calls, usage.prompt_tokens, usage.completion_tokens, finish_reason
    )
    return data, usage


async def _fake_stream(body: dict, model_cfg: BedrockModelConfig, usage: Usage) -> AsyncIterator[dict]:
    # Fallback for models that reject tool use combined with ConverseStream:
    # do a normal blocking call and replay it as a single-shot stream of
    # chunks so the client still gets the streaming response shape it asked
    # for. Yields structured chunks, not wire text -- app.py owns framing.
    data, result_usage = await chat(body, model_cfg)
    usage.prompt_tokens = result_usage.prompt_tokens
    usage.completion_tokens = result_usage.completion_tokens
    usage.cache_read_tokens = result_usage.cache_read_tokens
    usage.cache_write_tokens = result_usage.cache_write_tokens
    usage.service_tier = result_usage.service_tier

    choice = data["choices"][0]
    message = choice["message"]

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": data["id"],
            "object": "chat.completion.chunk",
            "created": data["created"],
            "model": data["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield chunk({"role": "assistant"})
    if message.get("content"):
        yield chunk({"content": message["content"]})
    if message.get("tool_calls"):
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": call["function"],
                    }
                    for index, call in enumerate(message["tool_calls"])
                ]
            }
        )
    yield chunk({}, choice["finish_reason"])


async def chat_stream(body: dict, model_cfg: BedrockModelConfig, usage: Usage) -> AsyncIterator[dict]:
    if not model_cfg.supports_tool_streaming and _wants_tool_config(body, model_cfg):
        async for stream_chunk in _fake_stream(body, model_cfg, usage):
            yield stream_chunk
        return

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

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield chunk({"role": "assistant"})

    saw_message_stop = False
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
            event_usage = event["metadata"].get("usage", {})
            usage.prompt_tokens = event_usage.get("inputTokens", 0)
            usage.completion_tokens = event_usage.get("outputTokens", 0)
            usage.cache_read_tokens = event_usage.get("cacheReadInputTokens", 0)
            usage.cache_write_tokens = event_usage.get("cacheWriteInputTokens", 0)
            # The tier that actually served the request -- see the matching
            # comment in chat() for why this must come from the response.
            usage.service_tier = event["metadata"].get("serviceTier", {}).get("type")
        elif "messageStop" in event:
            saw_message_stop = True
            finish_reason = _STOP_REASON_MAP.get(event["messageStop"].get("stopReason"), "stop")
            yield chunk({}, finish_reason)

    if not saw_message_stop:
        # The stream ended with no error and no messageStop -- Bedrock
        # (or something between us and it) silently closed the connection
        # before the model produced any output. Surface this as a real
        # error instead of recording a clean, empty "ok" completion.
        raise RuntimeError(
            "Bedrock stream ended without a messageStop event (no content, "
            "no error) -- likely an idle/read timeout before the model "
            "produced any output."
        )
