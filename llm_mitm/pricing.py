from __future__ import annotations

from llm_mitm.config import PricingConfig, PricingTier


def _active_tier(provider_config: PricingConfig, context_tokens: int) -> PricingTier:
    # The base rate is just the tier that always matches (above_tokens=0) --
    # modeling it as one, instead of a separate set of fields, means there's
    # one rule for "which rate applies" rather than a base case plus an
    # override loop that repeats the same four fields.
    active = PricingTier(
        above_tokens=0,
        input_cost_per_1m=provider_config.input_cost_per_1m,
        output_cost_per_1m=provider_config.output_cost_per_1m,
        cache_read_cost_per_1m=provider_config.cache_read_cost_per_1m,
        cache_write_cost_per_1m=provider_config.cache_write_cost_per_1m,
    )
    for tier in sorted(provider_config.tiers, key=lambda t: t.above_tokens):
        if context_tokens > tier.above_tokens:
            active = tier
    return active


def compute_cost(
    provider_config: PricingConfig,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    service_tier: str | None = None,
) -> float:
    # Total context size (not just the non-cached portion) is what actually
    # determines long-context pricing tiers.
    context_tokens = prompt_tokens + cache_read_tokens + cache_write_tokens
    tier = _active_tier(provider_config, context_tokens)

    cost = (
        prompt_tokens / 1_000_000 * tier.input_cost_per_1m
        + completion_tokens / 1_000_000 * tier.output_cost_per_1m
        + cache_read_tokens / 1_000_000 * tier.cache_read_cost_per_1m
        + cache_write_tokens / 1_000_000 * tier.cache_write_cost_per_1m
    )

    # Bedrock bills whatever tier actually served the request (echoed back
    # in the response), which can differ from the tier that was requested --
    # e.g. a "flex" request falls back to Standard pricing if flex capacity
    # isn't available.
    if service_tier == "flex":
        cost *= provider_config.flex_discount

    return cost
