from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel


class PricingTier(BaseModel):
    # Applies when context size (prompt + cache tokens) exceeds this
    # threshold (e.g. Anthropic's >200k-context pricing tier on Bedrock).
    above_tokens: int
    input_cost_per_1m: float
    output_cost_per_1m: float
    cache_read_cost_per_1m: float = 0.0
    cache_write_cost_per_1m: float = 0.0


class PricingConfig(BaseModel):
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    # Bedrock prompt caching: cache_read is a cache hit (cheap), cache_write
    # is creating/refreshing a cache entry (pricier than normal input).
    cache_read_cost_per_1m: float = 0.0
    cache_write_cost_per_1m: float = 0.0
    tiers: list[PricingTier] = []
    # Multiplier applied to the whole request's cost when Bedrock resolves
    # it to the "flex" service tier (see BedrockModelConfig.service_tier).
    # Defaults to AWS's documented flat 50% discount off Standard pricing.
    flex_discount: float = 0.5


class BedrockModelConfig(PricingConfig):
    model_id: str
    region: str | None = None
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


class BedrockMantleModelConfig(PricingConfig):
    model_id: str
    region: str
    # Most bedrock-mantle models serve OpenAI-compatible endpoints under
    # /v1; Gemma 4 models are the exception and use /openai/v1 instead --
    # check the model's Bedrock "Programmatic Access" table before assuming
    # the default. A request to the wrong path is rejected with "model ...
    # isn't supported on this route".
    base_path: str = "/v1"


class OllamaModelConfig(PricingConfig):
    base_url: str = "http://localhost:11434"
    model: str
    # Name of the env var holding a bearer token, for an Ollama that requires
    # one (e.g. behind an authenticating proxy). The name rather than the key
    # itself, since this file is rendered into a plain ConfigMap by the chart.
    api_key_env: str | None = None


class ModelEntry(BaseModel):
    name: str
    provider: Literal["bedrock", "ollama"]
    # Bedrock has two wire protocols behind one "bedrock" namespace: which
    # block is populated -- not a separate provider -- picks Converse vs
    # the bedrock-mantle OpenAI-compatible endpoint. Both share the
    # "bedrock/<name>" model id.
    bedrock: BedrockModelConfig | None = None
    bedrock_mantle: BedrockMantleModelConfig | None = None
    ollama: OllamaModelConfig | None = None

    @property
    def id(self) -> str:
        # The only identifier clients use -- in /v1/models and as the
        # "model" field in requests. "name" in config.yaml is just the
        # short local label; this disambiguates models that share a name
        # across providers and makes the backing provider explicit.
        return f"{self.provider}/{self.name}"

    @property
    def provider_config(self) -> BedrockModelConfig | BedrockMantleModelConfig | OllamaModelConfig:
        if self.provider == "ollama":
            if self.ollama is None:
                raise ValueError(f"model '{self.name}' is missing its 'ollama' config block")
            return self.ollama

        if self.bedrock is not None and self.bedrock_mantle is not None:
            raise ValueError(
                f"model '{self.name}' has both 'bedrock' and 'bedrock_mantle' config blocks -- pick one"
            )
        cfg = self.bedrock or self.bedrock_mantle
        if cfg is None:
            raise ValueError(f"model '{self.name}' is missing a 'bedrock' or 'bedrock_mantle' config block")
        return cfg


class ServerConfig(BaseModel):
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


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: list[ModelEntry]


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
