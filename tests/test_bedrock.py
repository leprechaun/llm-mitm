import asyncio
import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from llm_mitm.config import BedrockModelConfig
from llm_mitm.providers import bedrock


def _cfg(**overrides) -> BedrockModelConfig:
    defaults = {"model_id": "test-model", "region": "us-east-1"}
    defaults.update(overrides)
    return BedrockModelConfig(**defaults)


def test_to_bedrock_messages_splits_system():
    system, messages = bedrock._to_bedrock_messages(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system == [{"text": "be helpful"}]
    assert messages == [{"role": "user", "content": [{"text": "hi"}]}]


def test_to_bedrock_messages_merges_consecutive_same_role():
    # An assistant tool_calls message followed by two tool results must
    # collapse into a single alternating user/assistant sequence -- Bedrock
    # rejects back-to-back same-role messages.
    _, messages = bedrock._to_bedrock_messages(
        [
            {"role": "user", "content": "do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result b"},
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert len(messages[1]["content"]) == 2
    assert len(messages[2]["content"]) == 2


def test_to_bedrock_messages_tool_call_arguments_parsed_to_dict():
    _, messages = bedrock._to_bedrock_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            }
        ]
    )
    assert messages[0]["content"][0]["toolUse"] == {
        "toolUseId": "call_1",
        "name": "execute",
        "input": {"cmd": "ls"},
    }


def test_to_bedrock_messages_handles_invalid_tool_call_json():
    _, messages = bedrock._to_bedrock_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "execute", "arguments": "not json"}}
                ],
            }
        ]
    )
    assert messages[0]["content"][0]["toolUse"]["input"] == {}


def test_tool_config_omitted_without_tools():
    assert bedrock._to_bedrock_tool_config({"messages": []}) is None


def test_tool_config_auto():
    body = {"tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}]}
    config = bedrock._to_bedrock_tool_config(body)
    assert "toolChoice" not in config
    assert config["tools"][0]["toolSpec"]["name"] == "execute"


def test_tool_config_required_maps_to_any():
    body = {
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}],
        "tool_choice": "required",
    }
    assert bedrock._to_bedrock_tool_config(body)["toolChoice"] == {"any": {}}


def test_tool_config_named_choice():
    body = {
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "execute"}},
    }
    assert bedrock._to_bedrock_tool_config(body)["toolChoice"] == {"tool": {"name": "execute"}}


def test_tool_config_none_keeps_tools_but_omits_tool_choice():
    # Regression test for: "ValidationException: The toolConfig field must
    # be defined when using toolUse and toolResult content blocks." Bedrock
    # requires toolConfig whenever message history has prior tool use, even
    # on a turn with tool_choice "none" -- and Bedrock has no "never call a
    # tool" choice, so "none" degrades to omitting toolChoice (i.e. auto)
    # rather than dropping toolConfig entirely.
    body = {
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}],
        "tool_choice": "none",
    }
    config = bedrock._to_bedrock_tool_config(body)
    assert config is not None
    assert config["tools"][0]["toolSpec"]["name"] == "execute"
    assert "toolChoice" not in config


def test_prompt_caching_inserts_cache_points():
    cfg = _cfg(prompt_caching=True)
    body = {
        "model": "m",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}],
    }
    kwargs = bedrock._converse_kwargs(body, cfg)
    assert kwargs["system"][-1] == {"cachePoint": {"type": "default"}}
    assert kwargs["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
    assert kwargs["messages"][-1]["content"][-1] == {"cachePoint": {"type": "default"}}


def test_prompt_caching_disabled_by_default():
    cfg = _cfg()
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    kwargs = bedrock._converse_kwargs(body, cfg)
    assert "cachePoint" not in json.dumps(kwargs)


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("content_filtered", "content_filter"),
    ],
)
def test_chat_maps_stop_reason(stop_reason, finish_reason):
    cfg = _cfg()
    fake_response = {
        "output": {"message": {"content": [{"text": "hi"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
        "stopReason": stop_reason,
    }
    fake_client = MagicMock()
    fake_client.converse.return_value = fake_response
    with patch.object(bedrock, "_client", return_value=fake_client):
        data, *_ = asyncio.run(bedrock.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, cfg))
    assert data["choices"][0]["finish_reason"] == finish_reason


def test_chat_extracts_tool_calls():
    cfg = _cfg()
    fake_response = {
        "output": {
            "message": {
                "content": [
                    {"text": "Sure, let me check."},
                    {"toolUse": {"toolUseId": "call_abc", "name": "execute", "input": {"cmd": "ls /tmp"}}},
                ]
            }
        },
        "usage": {"inputTokens": 42, "outputTokens": 13},
        "stopReason": "tool_use",
    }
    fake_client = MagicMock()
    fake_client.converse.return_value = fake_response
    with patch.object(bedrock, "_client", return_value=fake_client):
        data, prompt_tokens, completion_tokens, cache_read, cache_write = asyncio.run(
            bedrock.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, cfg)
        )
    message = data["choices"][0]["message"]
    assert message["content"] == "Sure, let me check."
    assert message["tool_calls"] == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "execute", "arguments": json.dumps({"cmd": "ls /tmp"})},
        }
    ]
    assert (prompt_tokens, completion_tokens, cache_read, cache_write) == (42, 13, 0, 0)


def test_chat_extracts_cache_tokens():
    cfg = _cfg()
    fake_response = {
        "output": {"message": {"content": [{"text": "hi"}]}},
        "usage": {
            "inputTokens": 20,
            "outputTokens": 5,
            "cacheReadInputTokens": 5000,
            "cacheWriteInputTokens": 300,
        },
        "stopReason": "end_turn",
    }
    fake_client = MagicMock()
    fake_client.converse.return_value = fake_response
    with patch.object(bedrock, "_client", return_value=fake_client):
        _, prompt_tokens, completion_tokens, cache_read, cache_write = asyncio.run(
            bedrock.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, cfg)
        )
    assert (prompt_tokens, completion_tokens, cache_read, cache_write) == (20, 5, 5000, 300)


def test_chat_stream_translates_text_and_tool_calls():
    cfg = _cfg()
    fake_events = [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Sure, "}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "let me check."}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "call_abc", "name": "execute"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"cmd"'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ': "ls /tmp"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"metadata": {"usage": {"inputTokens": 42, "outputTokens": 13}}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    fake_client = MagicMock()
    fake_client.converse_stream.return_value = {"stream": iter(fake_events)}

    async def run():
        usage_holder = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        chunks = []
        with patch.object(bedrock, "_client", return_value=fake_client):
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            async for sse in bedrock.chat_stream(body, cfg, usage_holder):
                chunks.append(sse)
        return chunks, usage_holder

    chunks, usage_holder = asyncio.run(run())

    assert chunks[-1] == "data: [DONE]\n\n"
    assert usage_holder == {
        "prompt_tokens": 42,
        "completion_tokens": 13,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }

    payloads = [json.loads(c[len("data: ") :]) for c in chunks if c.strip() != "data: [DONE]"]

    tool_call_chunks = [
        p["choices"][0]["delta"]["tool_calls"][0] for p in payloads if p["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_chunks[0]["id"] == "call_abc"
    assert tool_call_chunks[0]["function"]["name"] == "execute"
    arguments = "".join(c["function"].get("arguments", "") for c in tool_call_chunks)
    assert json.loads(arguments) == {"cmd": "ls /tmp"}

    text_chunks = [p["choices"][0]["delta"]["content"] for p in payloads if p["choices"][0]["delta"].get("content")]
    assert "".join(text_chunks) == "Sure, let me check."

    finish_reasons = [p["choices"][0]["finish_reason"] for p in payloads if p["choices"][0]["finish_reason"]]
    assert finish_reasons == ["tool_calls"]


def test_chat_stream_falls_back_to_blocking_call_when_tool_streaming_unsupported():
    # Regression test for: "ValidationException: This model doesn't support
    # tool use in streaming mode" (e.g. Llama 4 Maverick on Bedrock).
    cfg = _cfg(supports_tool_streaming=False)
    body = {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": "list files in /tmp"}],
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {"type": "object"}}}],
    }
    fake_response = {
        "output": {
            "message": {
                "content": [
                    {"text": "Sure."},
                    {"toolUse": {"toolUseId": "call_abc", "name": "execute", "input": {"cmd": "ls /tmp"}}},
                ]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 4},
        "stopReason": "tool_use",
    }
    fake_client = MagicMock()
    fake_client.converse.return_value = fake_response
    fake_client.converse_stream.side_effect = AssertionError("converse_stream must not be called")

    async def run():
        usage_holder = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        chunks = []
        with patch.object(bedrock, "_client", return_value=fake_client):
            async for sse in bedrock.chat_stream(body, cfg, usage_holder):
                chunks.append(sse)
        return chunks, usage_holder

    chunks, usage_holder = asyncio.run(run())

    fake_client.converse.assert_called_once()
    assert usage_holder == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    assert chunks[-1] == "data: [DONE]\n\n"

    payloads = [json.loads(c[len("data: ") :]) for c in chunks if c.strip() != "data: [DONE]"]
    tool_call = next(
        p["choices"][0]["delta"]["tool_calls"][0] for p in payloads if p["choices"][0]["delta"].get("tool_calls")
    )
    assert tool_call["id"] == "call_abc"
    assert json.loads(tool_call["function"]["arguments"]) == {"cmd": "ls /tmp"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tools_dropped_entirely_when_model_does_not_support_them():
    cfg = _cfg(supports_tools=False)
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}],
    }
    kwargs = bedrock._converse_kwargs(body, cfg)
    assert "toolConfig" not in kwargs


def test_tool_choice_none_with_prior_tool_use_still_includes_tool_config():
    # Full regression test for the real-world failure: an agent loop (e.g.
    # opencode) resends history containing a completed tool call/result,
    # with tool_choice "none" on the follow-up turn asking for a final
    # answer. toolConfig must still be present or Bedrock rejects the call.
    cfg = _cfg()
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "list files in /tmp"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "execute", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
        ],
        "tools": [{"type": "function", "function": {"name": "execute", "parameters": {}}}],
        "tool_choice": "none",
    }
    kwargs = bedrock._converse_kwargs(body, cfg)
    assert "toolConfig" in kwargs
    assert "toolChoice" not in kwargs["toolConfig"]


def test_image_data_uri_decoded_to_bedrock_image_block():
    png_bytes = b"\x89PNG\r\n\x1a\nFAKE"
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    cfg = _cfg()
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    kwargs = bedrock._converse_kwargs(body, cfg)
    content = kwargs["messages"][0]["content"]
    assert content[0] == {"text": "what is this?"}
    assert content[1] == {"image": {"format": "png", "source": {"bytes": png_bytes}}}


def test_image_jpg_extension_aliases_to_jpeg():
    data_uri = "data:image/jpg;base64," + base64.b64encode(b"fake").decode()
    cfg = _cfg()
    body = {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]}]}
    kwargs = bedrock._converse_kwargs(body, cfg)
    assert kwargs["messages"][0]["content"][0]["image"]["format"] == "jpeg"


def test_image_unsupported_format_raises():
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode()
    cfg = _cfg()
    body = {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]}]}
    with pytest.raises(ValueError, match="unsupported image format"):
        bedrock._converse_kwargs(body, cfg)


def test_image_remote_url_is_fetched_and_format_detected_from_content_type():
    fake_response = MagicMock()
    fake_response.content = b"JFIFfakejpegbytes"
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.raise_for_status = MagicMock()
    cfg = _cfg()
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}]}
        ],
    }
    with patch("httpx.get", return_value=fake_response) as mock_get:
        kwargs = bedrock._converse_kwargs(body, cfg)
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "https://example.com/photo.jpg"
    image = kwargs["messages"][0]["content"][0]["image"]
    assert image == {"format": "jpeg", "source": {"bytes": b"JFIFfakejpegbytes"}}


def test_text_only_message_unaffected_by_image_handling():
    cfg = _cfg()
    body = {"model": "m", "messages": [{"role": "user", "content": "just text"}]}
    kwargs = bedrock._converse_kwargs(body, cfg)
    assert kwargs["messages"][0]["content"] == [{"text": "just text"}]
