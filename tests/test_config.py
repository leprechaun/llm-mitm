import pytest

from llm_mitm.config import (
    BedrockMantleModelConfig,
    BedrockModelConfig,
    ModelEntry,
    OllamaModelConfig,
)


def test_model_entry_id_is_provider_prefixed():
    entry = ModelEntry(
        name="llama3",
        provider="ollama",
        ollama=OllamaModelConfig(model="llama3.1:8b"),
    )
    assert entry.id == "ollama/llama3"


def test_bedrock_provider_config_picks_bedrock_block():
    entry = ModelEntry(
        name="claude-haiku-5.5",
        provider="bedrock",
        bedrock=BedrockModelConfig(model_id="anthropic.claude-haiku-5-5", region="us-west-2"),
    )
    assert isinstance(entry.provider_config, BedrockModelConfig)
    # Still shares the "bedrock" namespace regardless of which block is set.
    assert entry.id == "bedrock/claude-haiku-5.5"


def test_bedrock_provider_config_picks_bedrock_mantle_block():
    entry = ModelEntry(
        name="gemma4-e2b",
        provider="bedrock",
        bedrock_mantle=BedrockMantleModelConfig(model_id="google.gemma-4-e2b", region="us-west-2"),
    )
    assert isinstance(entry.provider_config, BedrockMantleModelConfig)
    assert entry.id == "bedrock/gemma4-e2b"


def test_bedrock_provider_config_raises_when_both_blocks_set():
    entry = ModelEntry(
        name="ambiguous",
        provider="bedrock",
        bedrock=BedrockModelConfig(model_id="x", region="us-west-2"),
        bedrock_mantle=BedrockMantleModelConfig(model_id="y", region="us-west-2"),
    )
    with pytest.raises(ValueError, match="both 'bedrock' and 'bedrock_mantle'"):
        _ = entry.provider_config


def test_bedrock_provider_config_raises_when_neither_block_set():
    entry = ModelEntry(name="missing", provider="bedrock")
    with pytest.raises(ValueError, match="missing a 'bedrock' or 'bedrock_mantle'"):
        _ = entry.provider_config


def test_ollama_provider_config_raises_when_block_missing():
    entry = ModelEntry(name="missing", provider="ollama")
    with pytest.raises(ValueError, match="missing its 'ollama'"):
        _ = entry.provider_config
