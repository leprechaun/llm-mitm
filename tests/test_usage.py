import json
import logging

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
