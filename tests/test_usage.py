import json
import logging

import pytest

from llm_mitm.logging_setup import JSON_FIELDS, _JsonFormatter
from llm_mitm.usage import UsageEvent, record_usage


def test_record_usage_emits_flat_json_via_extra():
    event = UsageEvent(
        key_id="sk-openwebui",
        model="claude-haiku-5.5",
        provider="bedrock",
        prompt_tokens=395,
        completion_tokens=90,
        cost_usd=8.45e-05,
        duration_s=1.84,
        status="ok",
        stream=False,
    )

    captured: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    usage_logger = logging.getLogger("llm_mitm.usage")
    handler = _CaptureHandler()
    usage_logger.addHandler(handler)
    try:
        record_usage(event)
    finally:
        usage_logger.removeHandler(handler)

    assert len(captured) == 1
    record = captured[0]

    # The flattening formatter promotes fields under this marker attribute
    # (via extra=), not just text buried inside the message string.
    fields = getattr(record, JSON_FIELDS)
    assert fields["model"] == "claude-haiku-5.5"
    assert fields["cost_usd"] == 8.45e-05
    assert fields["key_id"] == "sk-openwebui"

    parsed = json.loads(_JsonFormatter().format(record))
    assert parsed["model"] == "claude-haiku-5.5"
    assert parsed["cost_usd"] == 8.45e-05
    assert "message" not in parsed


def _event(**overrides) -> UsageEvent:
    defaults = {
        "key_id": "sk-test",
        "model": "m",
        "provider": "bedrock",
        "prompt_tokens": 10,
        "completion_tokens": 100,
        "cost_usd": 0.0,
        "duration_s": 2.0,
        "status": "ok",
        "stream": True,
    }
    defaults.update(overrides)
    return UsageEvent(**defaults)


def test_tokens_per_second_excludes_ttft_for_streaming():
    # 100 completion tokens over a 2s request, 0.5s of which was prefill
    # (time to first token) -- generation itself only took 1.5s.
    event = _event(duration_s=2.0, ttft_s=0.5, completion_tokens=100)
    assert event.tokens_per_second == pytest.approx(100 / 1.5)


def test_tokens_per_second_uses_full_duration_when_no_ttft():
    # Non-streaming requests have no ttft_s -- fall back to the full
    # request duration since there's no separate generation phase measured.
    event = _event(duration_s=2.0, ttft_s=None, completion_tokens=100)
    assert event.tokens_per_second == pytest.approx(50.0)


def test_tokens_per_second_none_when_no_completion_tokens():
    event = _event(completion_tokens=0)
    assert event.tokens_per_second is None


def test_tokens_per_second_none_when_ttft_consumes_entire_duration():
    # Degenerate/clock-skew case: ttft_s >= duration_s would divide by zero
    # or go negative -- leave it unset rather than emit a bogus rate.
    event = _event(duration_s=1.0, ttft_s=1.0, completion_tokens=50)
    assert event.tokens_per_second is None


def test_tokens_per_second_explicit_value_is_not_overridden():
    event = _event(duration_s=2.0, completion_tokens=100, tokens_per_second=999.0)
    assert event.tokens_per_second == 999.0


def test_record_usage_observes_ttft_and_tps_metrics():
    from llm_mitm.metrics import OUTPUT_TOKENS_PER_SECOND, TIME_TO_FIRST_TOKEN

    before_ttft = TIME_TO_FIRST_TOKEN.labels("m", "bedrock")._sum.get()
    before_tps = OUTPUT_TOKENS_PER_SECOND.labels("m", "bedrock")._sum.get()

    record_usage(_event(duration_s=2.0, ttft_s=0.5, completion_tokens=100))

    assert TIME_TO_FIRST_TOKEN.labels("m", "bedrock")._sum.get() == pytest.approx(before_ttft + 0.5)
    assert OUTPUT_TOKENS_PER_SECOND.labels("m", "bedrock")._sum.get() == pytest.approx(
        before_tps + 100 / 1.5
    )
