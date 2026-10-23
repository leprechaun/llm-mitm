import json
from unittest.mock import patch

from fastapi.testclient import TestClient

import llm_mitm.app as app_module

client = TestClient(app_module.app)


async def _ok_then_fail_stream(body, model_cfg, usage):
    yield 'data: {"delta": "hello "}\n\n'
    yield 'data: {"delta": "world"}\n\n'
    usage.prompt_tokens = 5
    usage.completion_tokens = 2
    raise RuntimeError("simulated mid-stream failure (e.g. bedrock ThrottlingException)")


def test_mid_stream_failure_closes_cleanly_instead_of_aborting():
    # Regression test: a provider error that happens after chunks have
    # already been sent must not kill the connection (that produced
    # "TransferEncodingError: Not enough data to satisfy transfer length
    # header" on real clients) -- it must end with a clean error event + DONE.
    with (
        patch.object(app_module.ollama, "chat_stream", _ok_then_fail_stream),
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "ollama/test-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response,
    ):
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    assert lines[0] == 'data: {"delta": "hello "}'
    assert lines[1] == 'data: {"delta": "world"}'
    assert lines[-1] == "data: [DONE]"

    error_payload = json.loads(lines[-2][len("data: ") :])
    assert error_payload["error"]["type"] == "upstream_error"
    assert "simulated mid-stream failure" in error_payload["error"]["message"]
