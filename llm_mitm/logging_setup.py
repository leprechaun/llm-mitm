from __future__ import annotations

import copy
import json
import logging
import sys
from typing import Any

import uvicorn.config

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(json_logs: bool, log_level: str) -> dict[str, Any] | None:
    """Set up stdlib logging and return the log_config to pass to uvicorn.run().

    log_level applies to our own logger and uvicorn's either way.

    json_logs=False: keep uvicorn's normal human-readable textual
    formatters -- returns a copy of uvicorn's own log config with the level
    overridden, since uvicorn.run() would otherwise reset it back to INFO.
    json_logs=True: route every logger (ours and uvicorn's) through one
    JSON-on-stdout handler, deliberately without a timestamp field. Returns
    None so uvicorn.run() skips its own dictConfig entirely, which would
    otherwise overwrite the handlers just set up here.
    """
    level = log_level.upper()

    logging.getLogger().setLevel(level)
    logging.getLogger("llm_mitm").setLevel(level)

    if not json_logs:
        uvicorn_log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
        for name in _UVICORN_LOGGERS:
            uvicorn_log_config["loggers"].setdefault(name, {})["level"] = level
        return uvicorn_log_config

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]

    for name in _UVICORN_LOGGERS:
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = [handler]
        uv_logger.setLevel(level)
        uv_logger.propagate = False

    return None
