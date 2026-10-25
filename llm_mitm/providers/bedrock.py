from __future__ import annotations

import asyncio
import base64
import functools
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


def _image_from_data_uri(match: re.Match[str]) -> tuple[bytes, str]:
    image_format = _normalize_image_format(match.group("fmt"))
    image_bytes = base64.b64decode(match.group("data"))
    return image_bytes, image_format


def _image_from_remote_url(url: str) -> tuple[bytes, str]:
    response = httpx.get(url, timeout=_IMAGE_FETCH_TIMEOUT)
    response.raise_for_status()
    fmt_guess = _image_format_from_content_type(response.headers.get("content-type", ""))
    if not fmt_guess:
        path = urlparse(url).path
        fmt_guess = path.rsplit(".", 1)[-1] if "." in path else ""
    return response.content, _normalize_image_format(fmt_guess)


def _image_block_from_url(url: str) -> dict:
    # OpenAI vision clients send either a data: URI (the image embedded
    # inline) or a plain http(s) URL (fetched here, since Bedrock's image
    # block only accepts raw bytes, not a remote URL).
    match = _DATA_URI_RE.match(url)
    image_bytes, image_format = _image_from_data_uri(match) if match else _image_from_remote_url(url)
    return {"image": {"format": image_format, "source": {"bytes": image_bytes}}}


def _text_block_from_part(part: dict) -> dict | None:
    text = part.get("text", "")
    return {"text": text} if text else None


def _image_block_from_part(part: dict) -> dict | None:
    url = part.get("image_url", {}).get("url", "")
    return _image_block_from_url(url) if url else None


_BLOCK_CONVERTERS = {
    "text": _text_block_from_part,
    "image_url": _image_block_from_part,
}


def _block_from_part(part: dict) -> dict | None:
    converter = _BLOCK_CONVERTERS.get(part.get("type"))
    return converter(part) if converter else None


def _content_blocks(content: str | list[dict] | None) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    return [block for part in content if (block := _block_from_part(part)) is not None]


def _system_block(message: dict) -> dict | None:
    text = _content_text(message.get("content"))
    return {"text": text} if text else None


def _tool_result_turn(message: dict) -> tuple[str, dict]:
    text = _content_text(message.get("content"))
    return (
        "user",
        {
            "toolResult": {
                "toolUseId": message.get("tool_call_id", ""),
                "content": [{"text": text}],
            }
        },
    )


def _tool_use_turn(call: dict) -> tuple[str, dict]:
    fn = call.get("function", {})
    try:
        tool_input = json.loads(fn.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        tool_input = {}
    return (
        "assistant",
        {
            "toolUse": {
                "toolUseId": call.get("id", ""),
                "name": fn.get("name", ""),
                "input": tool_input,
            }
        },
    )


def _turns_for_content_message(message: dict, role: str) -> list[tuple[str, dict]]:
    turns = [(role, block) for block in _content_blocks(message.get("content"))]
    if role == "assistant":
        turns.extend(_tool_use_turn(call) for call in message.get("tool_calls") or [])
    return turns


def _merge_adjacent_turns(turns: list[tuple[str, dict]]) -> list[dict]:
    # Bedrock requires strict user/assistant alternation, so e.g. consecutive
    # "tool" result messages must collapse into one user turn.
    converted: list[dict] = []
    for role, block in turns:
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].append(block)
        else:
            converted.append({"role": role, "content": [block]})
    return converted


def _to_bedrock_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system: list[dict] = []
    turns: list[tuple[str, dict]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            block = _system_block(message)
            if block:
                system.append(block)
        elif role == "tool":
            turns.append(_tool_result_turn(message))
        elif role in ("user", "assistant"):
            turns.extend(_turns_for_content_message(message, role))

    return system, _merge_adjacent_turns(turns)


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


def _extract_usage(container: dict) -> Usage:
    # `container` is either the full Converse response or a ConverseStream
    # "metadata" event -- both carry sibling "usage" and "serviceTier" keys
    # at the same nesting level.
    usage_dict = container.get("usage", {})
    return Usage(
        prompt_tokens=usage_dict.get("inputTokens", 0),
        completion_tokens=usage_dict.get("outputTokens", 0),
        cache_read_tokens=usage_dict.get("cacheReadInputTokens", 0),
        cache_write_tokens=usage_dict.get("cacheWriteInputTokens", 0),
        # The tier that actually served the request -- can differ from the
        # requested tier (e.g. "flex" falling back to Standard capacity), so
        # billing must read this instead of trusting model_cfg.service_tier.
        service_tier=container.get("serviceTier", {}).get("type"),
    )


def _stream_chunk(
    completion_id: str, created: int, model_name: str, delta: dict, finish_reason: str | None = None
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


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
    usage = _extract_usage(response)
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
    usage.update(result_usage)

    choice = data["choices"][0]
    message = choice["message"]
    chunk = functools.partial(_stream_chunk, data["id"], data["created"], data["model"])

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


def _tool_call_start_delta(start: dict, tool_call_indexes: dict[int, int]) -> dict | None:
    tool_use_start = start.get("start", {}).get("toolUse")
    if not tool_use_start:
        return None
    tool_call_index = len(tool_call_indexes)
    tool_call_indexes[start["contentBlockIndex"]] = tool_call_index
    return {
        "tool_calls": [
            {
                "index": tool_call_index,
                "id": tool_use_start["toolUseId"],
                "type": "function",
                "function": {"name": tool_use_start["name"], "arguments": ""},
            }
        ]
    }


def _content_block_delta(block_delta: dict, tool_call_indexes: dict[int, int]) -> dict | None:
    # Bedrock's contentBlockIndex also counts text blocks; OpenAI's tool_calls
    # delta index only counts tool calls, so we remap per tool-use block.
    delta = block_delta.get("delta", {})
    if "text" in delta:
        return {"content": delta["text"]} if delta["text"] else None
    if "toolUse" in delta:
        tool_call_index = tool_call_indexes.get(block_delta.get("contentBlockIndex"), 0)
        return {
            "tool_calls": [
                {
                    "index": tool_call_index,
                    "function": {"arguments": delta["toolUse"].get("input", "")},
                }
            ]
        }
    return None


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
    tool_call_indexes: dict[int, int] = {}

    chunk = functools.partial(_stream_chunk, completion_id, created, model_name)

    yield chunk({"role": "assistant"})

    saw_message_stop = False
    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is sentinel:
            break
        if isinstance(event, Exception):
            raise event
        if "contentBlockStart" in event:
            delta = _tool_call_start_delta(event["contentBlockStart"], tool_call_indexes)
            if delta:
                yield chunk(delta)
        elif "contentBlockDelta" in event:
            delta = _content_block_delta(event["contentBlockDelta"], tool_call_indexes)
            if delta:
                yield chunk(delta)
        elif "metadata" in event:
            usage.update(_extract_usage(event["metadata"]))
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
