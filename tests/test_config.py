import textwrap

import pytest
from pydantic import ValidationError

from llm_mitm.config import load_config


def _load(tmp_path, yaml_text: str):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(yaml_text))
    return load_config(str(path))


def test_model_id_combines_configured_provider_name_and_model_name(tmp_path):
    config = _load(
        tmp_path,
        """
        providers:
          - name: virty
            driver: openai
            base_url: http://virty:11434/v1
        models:
          - name: llama3
            provider: virty
            model_id: llama3.1:8b
        """,
    )

    assert config.models[0].id == "virty/llama3"
    assert config.models[0].model_id == "llama3.1:8b"
    assert config.provider_for(config.models[0]).base_url == "http://virty:11434/v1"


def test_model_referencing_unknown_provider_fails_at_load(tmp_path):
    with pytest.raises(ValidationError, match="unknown provider 'nope'"):
        _load(
            tmp_path,
            """
            providers: []
            models:
              - name: llama3
                provider: nope
                model_id: llama3.1:8b
            """,
        )


def test_bedrock_options_are_typed_with_defaults_for_a_converse_provider(tmp_path):
    config = _load(
        tmp_path,
        """
        providers:
          - name: bedrock-us
            driver: bedrock_converse
            region: us-west-2
        models:
          - name: scout
            provider: bedrock-us
            model_id: meta.scout
            options:
              supports_tool_streaming: false
        """,
    )

    options = config.models[0].options
    assert options.supports_tool_streaming is False
    assert options.supports_tools is True


def test_option_the_providers_driver_does_not_support_fails_at_load(tmp_path):
    with pytest.raises(ValidationError, match="prompt_caching"):
        _load(
            tmp_path,
            """
            providers:
              - name: virty
                driver: openai
                base_url: http://virty:11434/v1
            models:
              - name: llama3
                provider: virty
                model_id: llama3.1:8b
                options:
                  prompt_caching: true
            """,
        )


def test_unknown_driver_fails_at_load(tmp_path):
    with pytest.raises(ValidationError, match="driver"):
        _load(
            tmp_path,
            """
            providers:
              - name: virty
                driver: carrier-pigeon
            models: []
            """,
        )


def test_old_per_provider_block_format_fails_at_load(tmp_path):
    with pytest.raises(ValidationError, match="ollama"):
        _load(
            tmp_path,
            """
            providers:
              - name: ollama
                driver: openai
                base_url: http://virty:11434/v1
            models:
              - name: llama3
                provider: ollama
                model_id: llama3.1:8b
                ollama:
                  model: llama3.1:8b
            """,
        )


def test_typo_in_provider_field_fails_at_load(tmp_path):
    with pytest.raises(ValidationError, match="api_key_evn"):
        _load(
            tmp_path,
            """
            providers:
              - name: virty
                driver: openai
                base_url: http://virty:11434/v1
                api_key_evn: OLLAMA_API_KEY
            models: []
            """,
        )


def test_typo_in_pricing_tier_fails_at_load(tmp_path):
    with pytest.raises(ValidationError, match="input_cost_per_1million"):
        _load(
            tmp_path,
            """
            providers:
              - name: virty
                driver: openai
                base_url: http://virty:11434/v1
            models:
              - name: llama3
                provider: virty
                model_id: llama3.1:8b
                tiers:
                  - above_tokens: 100
                    input_cost_per_1m: 1
                    output_cost_per_1m: 1
                    input_cost_per_1million: 1
            """,
        )


def test_duplicate_provider_names_fail_at_load(tmp_path):
    with pytest.raises(ValidationError, match="duplicate provider 'virty'"):
        _load(
            tmp_path,
            """
            providers:
              - name: virty
                driver: openai
                base_url: http://a/v1
              - name: virty
                driver: openai
                base_url: http://b/v1
            models: []
            """,
        )


def test_duplicate_model_ids_fail_at_load(tmp_path):
    with pytest.raises(ValidationError, match="duplicate model 'virty/llama3'"):
        _load(
            tmp_path,
            """
            providers:
              - name: virty
                driver: openai
                base_url: http://virty:11434/v1
            models:
              - name: llama3
                provider: virty
                model_id: a
              - name: llama3
                provider: virty
                model_id: b
            """,
        )


def test_same_model_name_under_two_providers_is_allowed(tmp_path):
    config = _load(
        tmp_path,
        """
        providers:
          - name: mac
            driver: openai
            base_url: http://mac:11434/v1
          - name: virty
            driver: openai
            base_url: http://virty:11434/v1
        models:
          - name: llama3
            provider: mac
            model_id: llama3
          - name: llama3
            provider: virty
            model_id: llama3
        """,
    )

    assert [model.id for model in config.models] == ["mac/llama3", "virty/llama3"]


def test_discovery_is_opt_in_per_provider(tmp_path):
    config = _load(
        tmp_path,
        """
        providers:
          - name: virty
            driver: openai
            base_url: http://virty:11434/v1
            discover: true
          - name: mantle
            driver: openai
            base_url: https://mantle/openai/v1
        models: []
        """,
    )

    assert [provider.discover for provider in config.providers] == [True, False]


def test_discovery_on_a_bedrock_provider_fails_at_load(tmp_path):
    with pytest.raises(ValidationError):
        _load(
            tmp_path,
            """
            providers:
              - name: bedrock-us
                driver: bedrock_converse
                discover: true
            models: []
            """,
        )
