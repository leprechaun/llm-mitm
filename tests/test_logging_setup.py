import json
import logging
import sys

from llm_mitm.logging_setup import _JsonFormatter, configure_logging


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


def _reset_logging_state():
    logging.getLogger().handlers = []
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("llm_mitm").setLevel(logging.NOTSET)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).setLevel(logging.NOTSET)
        logging.getLogger(name).propagate = True


def test_configure_logging_false_leaves_handlers_untouched_but_sets_level():
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging(json_logs=False, log_level="DEBUG")
        assert root.handlers == before
        assert root.level == logging.DEBUG
        assert logging.getLogger("llm_mitm").level == logging.DEBUG
    finally:
        _reset_logging_state()


def test_configure_logging_false_returns_uvicorn_config_with_level_override():
    try:
        log_config = configure_logging(json_logs=False, log_level="WARNING")
        assert log_config is not None
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            assert log_config["loggers"][name]["level"] == "WARNING"
        # formatters/handlers are untouched -- still uvicorn's own textual ones
        assert "default" in log_config["formatters"]
    finally:
        _reset_logging_state()


def test_configure_logging_true_installs_json_handler_on_root_and_uvicorn_loggers():
    try:
        log_config = configure_logging(json_logs=True, log_level="INFO")
        assert log_config is None
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, _JsonFormatter)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uv_logger = logging.getLogger(name)
            assert uv_logger.handlers == root.handlers
            assert uv_logger.propagate is False
    finally:
        _reset_logging_state()


def test_configure_logging_true_applies_log_level():
    try:
        configure_logging(json_logs=True, log_level="ERROR")
        assert logging.getLogger().level == logging.ERROR
        assert logging.getLogger("llm_mitm").level == logging.ERROR
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            assert logging.getLogger(name).level == logging.ERROR
    finally:
        _reset_logging_state()
