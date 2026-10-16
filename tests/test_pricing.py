import pytest

from llm_mitm.config import PricingConfig, PricingTier
from llm_mitm.pricing import compute_cost


def test_flat_rate_no_tiers():
    cfg = PricingConfig(input_cost_per_1m=2.0, output_cost_per_1m=10.0)
    cost = compute_cost(cfg, prompt_tokens=1_000_000, completion_tokens=500_000)
    assert cost == 7.0


def test_cache_read_and_write_rates():
    cfg = PricingConfig(
        input_cost_per_1m=1.0,
        output_cost_per_1m=5.0,
        cache_read_cost_per_1m=0.1,
        cache_write_cost_per_1m=1.25,
    )
    cost = compute_cost(
        cfg,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == 1.0 + 0.1 + 1.25


def test_tier_not_applied_at_or_below_threshold():
    cfg = PricingConfig(
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.50,
        tiers=[PricingTier(above_tokens=100_000, input_cost_per_1m=0.50, output_cost_per_1m=2.50)],
    )
    cost = compute_cost(cfg, prompt_tokens=100_000, completion_tokens=0)
    assert cost == 100_000 / 1_000_000 * 0.10


def test_tier_applied_above_threshold():
    cfg = PricingConfig(
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.50,
        tiers=[PricingTier(above_tokens=100_000, input_cost_per_1m=0.50, output_cost_per_1m=2.50)],
    )
    cost = compute_cost(cfg, prompt_tokens=100_001, completion_tokens=0)
    assert cost == 100_001 / 1_000_000 * 0.50


def test_tier_threshold_counts_cache_tokens_toward_context_size():
    # A huge cached conversation should still trip the long-context tier
    # even though "prompt_tokens" (just the fresh/new portion) is tiny --
    # this is the exact bug class flagged earlier ("not a slight error, 5x").
    cfg = PricingConfig(
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.50,
        cache_read_cost_per_1m=0.01,
        cache_write_cost_per_1m=0.125,
        tiers=[
            PricingTier(
                above_tokens=100_000,
                input_cost_per_1m=0.50,
                output_cost_per_1m=2.50,
                cache_read_cost_per_1m=0.05,
                cache_write_cost_per_1m=0.625,
            )
        ],
    )
    cost = compute_cost(cfg, prompt_tokens=200, completion_tokens=0, cache_read_tokens=150_000)
    # Once the tier triggers, it applies uniformly to all rates -- including
    # the "fresh" prompt tokens, not just the cache-read ones that pushed it over.
    expected = 200 / 1_000_000 * 0.50 + 150_000 / 1_000_000 * 0.05
    assert cost == expected


def test_flex_tier_applies_discount():
    cfg = PricingConfig(input_cost_per_1m=1.0, output_cost_per_1m=5.0)
    cost = compute_cost(cfg, prompt_tokens=1_000_000, completion_tokens=0, service_tier="flex")
    assert cost == pytest.approx(0.5)


def test_flex_discount_is_configurable():
    cfg = PricingConfig(input_cost_per_1m=1.0, flex_discount=0.25)
    cost = compute_cost(cfg, prompt_tokens=1_000_000, completion_tokens=0, service_tier="flex")
    assert cost == pytest.approx(0.25)


def test_non_flex_tier_does_not_apply_discount():
    cfg = PricingConfig(input_cost_per_1m=1.0, output_cost_per_1m=5.0)
    cost = compute_cost(cfg, prompt_tokens=1_000_000, completion_tokens=0, service_tier="default")
    assert cost == 1.0


def test_tier_picks_highest_matching_threshold():
    cfg = PricingConfig(
        input_cost_per_1m=0.10,
        tiers=[
            PricingTier(above_tokens=100_000, input_cost_per_1m=0.50, output_cost_per_1m=2.50),
            PricingTier(above_tokens=200_000, input_cost_per_1m=1.00, output_cost_per_1m=5.00),
        ],
    )
    cost = compute_cost(cfg, prompt_tokens=250_000, completion_tokens=0)
    assert cost == 250_000 / 1_000_000 * 1.00
