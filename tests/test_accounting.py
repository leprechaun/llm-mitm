import pytest

from llm_mitm.accounting import RequestAccounting, _chunk_has_token
from llm_mitm.config import PricingConfig
from llm_mitm.providers import Usage


def _fake_clock(*times: float):
    it = iter(times)
    return lambda: next(it)


def test_chunk_has_token_false_for_role_only_opener():
    assert not _chunk_has_token({"choices": [{"delta": {"role": "assistant"}}]})


def test_chunk_has_token_false_for_finish_reason_only():
    assert not _chunk_has_token({"choices": [{"delta": {}, "finish_reason": "stop"}]})


def test_chunk_has_token_true_for_content_delta():
    assert _chunk_has_token({"choices": [{"delta": {"content": "hi"}}]})


def test_chunk_has_token_true_for_tool_call_delta():
    assert _chunk_has_token({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1"}]}}]})


def test_chunk_has_token_false_for_empty_content():
    assert not _chunk_has_token({"choices": [{"delta": {"content": ""}}]})


def test_finish_records_usage_event_with_computed_cost():
    captured = []
    # clock calls: construction (start), finish (duration)
    accounting = RequestAccounting(
        "sk-test",
        "m",
        "bedrock",
        PricingConfig(input_cost_per_1m=1.0, output_cost_per_1m=2.0),
        clock=_fake_clock(100.0, 102.5),
        sink=captured.append,
    )
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    accounting.finish(usage, "ok", stream=False)

    assert len(captured) == 1
    event = captured[0]
    assert event.key_id == "sk-test"
    assert event.model == "m"
    assert event.provider == "bedrock"
    assert event.prompt_tokens == 1_000_000
    assert event.completion_tokens == 500_000
    assert event.cost_usd == pytest.approx(1.0 + 1.0)  # 1M * $1 + 0.5M * $2
    assert event.duration_s == pytest.approx(2.5)
    assert event.ttft_s is None
    assert event.status == "ok"
    assert event.stream is False


def test_observe_chunk_sets_ttft_once_on_first_real_token():
    captured = []
    # clock calls: construction (start), first token chunk (ttft), finish (duration).
    # The clock is only consulted on a chunk that actually has a token, and
    # only the first time -- a second token chunk must not call it again.
    accounting = RequestAccounting(
        "sk-test", "m", "bedrock", PricingConfig(), clock=_fake_clock(0.0, 0.4, 2.0), sink=captured.append
    )
    accounting.observe_chunk({"choices": [{"delta": {"role": "assistant"}}]})
    accounting.observe_chunk({"choices": [{"delta": {"content": "hi"}}]})
    accounting.observe_chunk({"choices": [{"delta": {"content": " there"}}]})
    accounting.finish(Usage(), "ok", stream=True)

    event = captured[0]
    assert event.ttft_s == pytest.approx(0.4)
    assert event.duration_s == pytest.approx(2.0)


def test_observe_chunk_leaves_ttft_none_without_a_token_chunk():
    captured = []
    # Only two clock calls expected: construction and finish -- observe_chunk
    # must never consult the clock when nothing it sees carries a token.
    accounting = RequestAccounting(
        "sk-test", "m", "bedrock", PricingConfig(), clock=_fake_clock(0.0, 1.0), sink=captured.append
    )
    accounting.observe_chunk({"choices": [{"delta": {"role": "assistant"}}]})
    accounting.observe_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    accounting.finish(Usage(), "error", stream=True)

    assert captured[0].ttft_s is None


def test_finish_with_empty_usage_has_zero_cost():
    # The blocking-error path has no Usage from the provider at all --
    # Usage() defaults to all zeros, which must cost $0 regardless of rates.
    captured = []
    accounting = RequestAccounting(
        "sk-test",
        "m",
        "bedrock",
        PricingConfig(input_cost_per_1m=5.0, output_cost_per_1m=10.0),
        clock=_fake_clock(0.0, 0.1),
        sink=captured.append,
    )
    accounting.finish(Usage(), "error", stream=False)

    assert captured[0].cost_usd == 0.0
    assert captured[0].status == "error"
    assert captured[0].stream is False
