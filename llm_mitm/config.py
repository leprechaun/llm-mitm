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


class BedrockModelConfig(PricingConfig):
    model_id: str
    region: str | None = None
    # Insert cachePoint breakpoints (system prompt, tool config, last
    # message) so repeated context across turns is served from Bedrock's
    # prompt cache instead of being billed at full input price. Only
    # supported by some models (e.g. recent Anthropic Claude models).
    prompt_caching: bool = False


class OllamaModelConfig(PricingConfig):
    base_url: str = "http://localhost:11434"
    model: str


class ModelEntry(BaseModel):
    name: str
    provider: Literal["bedrock", "ollama"]
    bedrock: BedrockModelConfig | None = None
    ollama: OllamaModelConfig | None = None

    @property
    def provider_config(self) -> BedrockModelConfig | OllamaModelConfig:
        cfg = self.bedrock if self.provider == "bedrock" else self.ollama
        if cfg is None:
            raise ValueError(f"model '{self.name}' is missing its '{self.provider}' config block")
        return cfg


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 4000
    # Hash API keys down to a short id before using them anywhere (usage
    # logs, Prometheus key_id label). Disable to see raw keys in those
    # places instead -- not recommended outside trusted/local setups.
    hash_api_keys: bool = True


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: list[ModelEntry]


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
