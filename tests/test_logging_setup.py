import json
import logging
import sys

import llm_mitm.usage  # noqa: F401 -- import-time side effect: pins usage_logger to INFO
from llm_mitm.logging_setup import JSON_FIELDS, _JsonFormatter, configure_logging


def test_json_formatter_has_no_timestamp_field():
    record = logging.LogRecord("llm_mitm", logging.INFO, "x.py", 1, "hello", None, None)
    parsed = json.loads(_JsonFormatter().format(record))
    assert parsed == {"level": "INFO", "logger": "llm_mitm", "message": "hello"}


def test_json_formatter_includes_exception_traceback():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord("llm_mitm", logging.ERROR, "x.py", 1, "failed", None, sys.exc_info())
    parsed = json.loads(_JsonFormatter().format(record))
    assert "RuntimeError: boom" in parsed["exception"]


def test_json_formatter_promotes_marked_extra_fields_to_top_level():
    # Regression test for: usage events arriving as a nested, escaped JSON
    # string inside "message" instead of flat, queryable top-level fields.
    logger = logging.getLogger("llm_mitm.usage")
    record = logger.makeRecord(
        "llm_mitm.usage",
        logging.INFO,
        "x.py",
        1,
        '{"model": "claude-haiku-5.5", "cost_usd": 0.0001}',
        (),
        None,
        extra={JSON_FIELDS: {"model": "claude-haiku-5.5", "cost_usd": 0.0001}},
    )
    parsed = json.loads(_JsonFormatter().format(record))
    assert parsed == {
        "level": "INFO",
        "logger": "llm_mitm.usage",
        "model": "claude-haiku-5.5",
        "cost_usd": 0.0001,
    }
    assert "message" not in parsed


def test_json_formatter_ignores_unrelated_extra_attributes():
    # uvicorn stashes its own extras (e.g. color_message, for its colorized
    # terminal formatter) that have nothing to do with us -- those must not
    # leak in and must not suppress the real message.
    record = logging.LogRecord("uvicorn.error", logging.INFO, "x.py", 1, "Started server process [%d]", (123,), None)
    record.color_message = "Started server process [\x1b[36m%d\x1b[0m]"
    parsed = json.loads(_JsonFormatter().format(record))
    assert parsed == {"level": "INFO", "logger": "uvicorn.error", "message": "Started server process [123]"}


def _reset_logging_state():
    logging.getLogger().handlers = []
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "llm_mitm", "llm_mitm.usage"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "llm_mitm"):
        logging.getLogger(name).setLevel(logging.NOTSET)
    # llm_mitm.usage's level is a permanent invariant pinned once at import
    # time by usage.py, not something configure_logging() ever touches.
    logging.getLogger("llm_mitm.usage").setLevel(logging.INFO)


def test_configure_logging_installs_one_shared_handler_everywhere():
    try:
        configure_logging(json_logs=False, log_level="INFO")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        shared_handler = root.handlers[0]
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uv_logger = logging.getLogger(name)
            assert uv_logger.handlers == [shared_handler]
            assert uv_logger.propagate is False
        # usage logger isn't touched directly, but propagates up to the
        # same shared root handler -- same format as everything else.
        usage_logger = logging.getLogger("llm_mitm.usage")
        assert usage_logger.handlers == []
        assert usage_logger.propagate is True
    finally:
        _reset_logging_state()


def test_configure_logging_false_uses_textual_formatter():
    try:
        configure_logging(json_logs=False, log_level="INFO")
        formatter = logging.getLogger().handlers[0].formatter
        assert not isinstance(formatter, _JsonFormatter)
    finally:
        _reset_logging_state()


def test_configure_logging_true_uses_json_formatter_everywhere():
    try:
        configure_logging(json_logs=True, log_level="INFO")
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, _JsonFormatter)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            assert logging.getLogger(name).handlers[0].formatter is root.handlers[0].formatter
    finally:
        _reset_logging_state()


def test_configure_logging_applies_log_level_to_root_and_uvicorn():
    try:
        configure_logging(json_logs=True, log_level="ERROR")
        assert logging.getLogger().level == logging.ERROR
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            assert logging.getLogger(name).level == logging.ERROR
    finally:
        _reset_logging_state()


def test_usage_logger_stays_at_info_regardless_of_configured_log_level():
    # Cost-tracking events must never be silenced just because general log
    # verbosity was turned down.
    try:
        configure_logging(json_logs=True, log_level="ERROR")
        usage_logger = logging.getLogger("llm_mitm.usage")
        assert usage_logger.level == logging.INFO
        assert usage_logger.getEffectiveLevel() == logging.INFO
    finally:
        _reset_logging_state()
