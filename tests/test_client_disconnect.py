import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import llm_mitm.app as app_module


async def _never_finishes_stream(body, model_cfg, usage_holder):
    yield 'data: {"delta": "partial"}\n\n'
    await asyncio.sleep(3600)  # pretend the model is still "thinking"
    yield "data: [DONE]\n\n"  # never reached in this test


def _fake_request(body: dict) -> MagicMock:
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    request.headers = {}
    return request


def test_abandoned_stream_is_not_recorded_as_ok():
    # Regression test for a real production case: a Gateway API HTTPRoute's
    # request timeout fired mid-stream (Envoy response_flags=UT) and closed
    # the downstream connection before the model produced any output.
    # Starlette tears the generator down via GeneratorExit when that
    # happens -- a BaseException that `except Exception` never sees -- so
    # "status" must not default to "ok", or an abandoned request gets
    # misrecorded as a successful one.
    captured = {}

    def fake_record_usage(event):
        captured["event"] = event

    async def run():
        with (
            patch.object(app_module.ollama, "chat_stream", _never_finishes_stream),
            patch.object(app_module, "record_usage", fake_record_usage),
        ):
            request = _fake_request(
                {"model": "ollama/test-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            )
            response = await app_module.chat_completions(request)

            body_iterator = response.body_iterator
            first_chunk = await body_iterator.__anext__()
            if isinstance(first_chunk, bytes):
                first_chunk = first_chunk.decode()
            assert "partial" in first_chunk

            # Simulate what Starlette does when the downstream client/gateway
            # disconnects mid-stream: close the generator without finishing it.
            await body_iterator.aclose()

    asyncio.run(run())

    assert "event" in captured
    assert captured["event"].status != "ok"
