import pytest

from llm_mitm.config import BedrockMantleModelConfig
from llm_mitm.providers import bedrock_mantle


def _cfg(**overrides) -> BedrockMantleModelConfig:
    defaults = {"model_id": "google.gemma-4-e2b", "region": "us-west-2"}
    defaults.update(overrides)
    return BedrockMantleModelConfig(**defaults)


def test_auth_header_reads_bearer_token_from_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "sk-test-token")
    assert bedrock_mantle._auth_header() == {"Authorization": "Bearer sk-test-token"}


def test_auth_header_raises_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
        bedrock_mantle._auth_header()


def test_base_url_uses_region_and_default_path():
    cfg = _cfg()
    assert bedrock_mantle._base_url(cfg) == "https://bedrock-mantle.us-west-2.api.aws/v1"


def test_base_url_honors_custom_base_path():
    # Gemma 4 models are served under /openai/v1, not the usual /v1.
    cfg = _cfg(base_path="/openai/v1")
    assert bedrock_mantle._base_url(cfg) == "https://bedrock-mantle.us-west-2.api.aws/openai/v1"


def test_payload_sets_model_id_and_stream_flag():
    cfg = _cfg(model_id="google.gemma-4-31b")
    body = {"model": "bedrock/gemma4-31b", "messages": [{"role": "user", "content": "hi"}]}
    payload = bedrock_mantle._payload(body, cfg, stream=True)
    assert payload["model"] == "google.gemma-4-31b"
    assert payload["stream"] is True
    assert payload["messages"] == body["messages"]
    # The original body is untouched -- callers may reuse it.
    assert body["model"] == "bedrock/gemma4-31b"
