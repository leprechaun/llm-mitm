from __future__ import annotations

from llm_mitm.config import PricingConfig


def compute_cost(
    provider_config: PricingConfig,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    input_rate = provider_config.input_cost_per_1m
    output_rate = provider_config.output_cost_per_1m
    cache_read_rate = provider_config.cache_read_cost_per_1m
    cache_write_rate = provider_config.cache_write_cost_per_1m

    # Total context size (not just the non-cached portion) is what actually
    # determines long-context pricing tiers.
    context_tokens = prompt_tokens + cache_read_tokens + cache_write_tokens
    for tier in sorted(provider_config.tiers, key=lambda t: t.above_tokens):
        if context_tokens > tier.above_tokens:
            input_rate = tier.input_cost_per_1m
            output_rate = tier.output_cost_per_1m
            cache_read_rate = tier.cache_read_cost_per_1m
            cache_write_rate = tier.cache_write_cost_per_1m

    return (
        prompt_tokens / 1_000_000 * input_rate
        + completion_tokens / 1_000_000 * output_rate
        + cache_read_tokens / 1_000_000 * cache_read_rate
        + cache_write_tokens / 1_000_000 * cache_write_rate
    )
