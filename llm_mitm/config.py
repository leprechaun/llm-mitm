from __future__ import annotations

from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    # An unknown field is a config mistake (a typo, or a leftover from an
    # older format), not something to ignore until a request hits it.
    model_config = ConfigDict(extra="forbid")


class PricingTier(_StrictModel):
    # Applies when context size (prompt + cache tokens) exceeds this
    # threshold (e.g. Anthropic's >200k-context pricing tier on Bedrock).
    above_tokens: int
    input_cost_per_1m: float
    output_cost_per_1m: float
    cache_read_cost_per_1m: float = 0.0
    cache_write_cost_per_1m: float = 0.0


class PricingConfig(_StrictModel):
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    # Bedrock prompt caching: cache_read is a cache hit (cheap), cache_write
    # is creating/refreshing a cache entry (pricier than normal input).
    cache_read_cost_per_1m: float = 0.0
    cache_write_cost_per_1m: float = 0.0
    tiers: list[PricingTier] = []
    # Multiplier applied to the whole request's cost when Bedrock resolves
    # it to the "flex" service tier (see BedrockConverseOptions.service_tier).
    # Defaults to AWS's documented flat 50% discount off Standard pricing.
    flex_discount: float = 0.5


class BedrockConverseOptions(_StrictModel):
    # Insert cachePoint breakpoints (system prompt, tool config, last
    # message) so repeated context across turns is served from Bedrock's
    # prompt cache instead of being billed at full input price. Only
    # supported by some models (e.g. recent Anthropic Claude models).
    prompt_caching: bool = False
    # Whether this model accepts tools/tool_choice at all on Converse. If
    # False, tools are never sent even if the client includes them.
    supports_tools: bool = True
    # Some Bedrock models (e.g. Llama 4 Maverick) reject ConverseStream
    # requests that also include toolConfig ("This model doesn't support
    # tool use in streaming mode"). Set False to transparently fall back to
    # a blocking call for streaming requests that include tools, replayed to
    # the client as a single-shot SSE stream so it still gets back the
    # streaming response shape it asked for.
    supports_tool_streaming: bool = True
    # Requests one of Bedrock's Converse service tiers via the "serviceTier"
    # field. "flex" trades latency (minutes rather than seconds, best-effort
    # capacity) for a pricing discount -- good for batch-ish, non-interactive
    # workloads (model evals, summarization, agentic workflows). "priority"
    # trades a price premium for faster responses. Only some models support
    # each tier -- check the model's "Service Tiers" table in its Bedrock
    # model card; requesting an unsupported tier is rejected with a
    # ValidationException.
    service_tier: Literal["default", "priority", "flex"] | None = None


class OpenAIOptions(_StrictModel):
    pass


_OPTIONS_BY_DRIVER: dict[str, type[_StrictModel]] = {
    "openai": OpenAIOptions,
    "bedrock_converse": BedrockConverseOptions,
}


class OpenAIProviderConfig(_StrictModel):
    # Any upstream speaking OpenAI's chat completions format -- Ollama,
    # bedrock-mantle, etc. base_url includes the version prefix, which isn't
    # always /v1: bedrock-mantle serves Gemma 4 under /openai/v1, for one.
    name: str
    driver: Literal["openai"]
    base_url: str
    # Name of the env var holding a bearer token, for an upstream that
    # requires one. The name rather than the key itself, since this file is
    # rendered into a plain ConfigMap by the chart.
    api_key_env: str | None = None


class BedrockConverseProviderConfig(_StrictModel):
    name: str
    driver: Literal["bedrock_converse"]
    region: str | None = None


ProviderConfig = Annotated[OpenAIProviderConfig | BedrockConverseProviderConfig, Field(discriminator="driver")]


class ModelConfig(PricingConfig):
    name: str
    provider: str
    # The model's identifier on the upstream -- what the driver sends.
    model_id: str
    # Validated against the provider's driver by AppConfig, which replaces
    # the raw mapping with that driver's options class.
    options: dict[str, Any] | OpenAIOptions | BedrockConverseOptions = {}

    @property
    def id(self) -> str:
        # The only identifier clients use -- in /v1/models and as the
        # "model" field in requests.
        return f"{self.provider}/{self.name}"


class ServerConfig(_StrictModel):
    host: str = "0.0.0.0"
    port: int = 4000
    # Hash API keys down to a short id before using them anywhere (usage
    # logs, Prometheus key_id label). Disable to see raw keys in those
    # places instead -- not recommended outside trusted/local setups.
    hash_api_keys: bool = True
    # False (default): keep Python/uvicorn's normal human-readable textual
    # logs, for local dev. True: switch every log line (ours + uvicorn's) to
    # compact JSON on stdout with no timestamp field -- a k8s log collector
    # already stamps arrival time, so an app-side one is redundant noise.
    json_logs: bool = False
    # Applies to our own logger and uvicorn's, regardless of json_logs.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AppConfig(_StrictModel):
    server: ServerConfig = ServerConfig()
    providers: list[ProviderConfig]
    models: list[ModelConfig]

    @model_validator(mode="after")
    def _resolve_models_against_providers(self) -> AppConfig:
        drivers: dict[str, str] = {}
        for provider in self.providers:
            if provider.name in drivers:
                raise ValueError(f"duplicate provider '{provider.name}'")
            drivers[provider.name] = provider.driver

        model_ids: set[str] = set()
        for model in self.models:
            if model.provider not in drivers:
                raise ValueError(f"model '{model.name}' references unknown provider '{model.provider}'")
            if model.id in model_ids:
                raise ValueError(f"duplicate model '{model.id}'")
            model_ids.add(model.id)
            model.options = _OPTIONS_BY_DRIVER[drivers[model.provider]].model_validate(model.options)
        return self

    def provider_for(self, model: ModelConfig) -> OpenAIProviderConfig | BedrockConverseProviderConfig:
        return next(provider for provider in self.providers if provider.name == model.provider)


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
