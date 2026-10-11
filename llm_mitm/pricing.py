from __future__ import annotations

from llm_mitm.config import PricingConfig


def compute_cost(provider_config: PricingConfig, prompt_tokens: int, completion_tokens: int) -> float:
    input_rate = provider_config.input_cost_per_1m
    output_rate = provider_config.output_cost_per_1m
    for tier in sorted(provider_config.tiers, key=lambda t: t.above_tokens):
        if prompt_tokens > tier.above_tokens:
            input_rate = tier.input_cost_per_1m
            output_rate = tier.output_cost_per_1m
    return prompt_tokens / 1_000_000 * input_rate + completion_tokens / 1_000_000 * output_rate
