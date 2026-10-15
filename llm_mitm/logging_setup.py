from __future__ import annotations

import json
import logging
import sys

import uvicorn.logging

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


# Opt-in marker for extra= fields that should be promoted to top-level JSON
# keys (e.g. usage_logger.info(..., extra={JSON_FIELDS: asdict(event)})).
# Deliberately not "any attribute the record happens to carry" -- uvicorn
# itself stashes its own extras (e.g. color_message, for its colorized
# terminal formatter) that have nothing to do with us and must not leak in.
JSON_FIELDS = "_llm_mitm_json_fields"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name}
        extra = getattr(record, JSON_FIELDS, None)
        if extra is not None:
            payload.update(extra)
        else:
            payload["message"] = record.getMessage()
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(json_logs: bool, log_level: str) -> None:
    """Route every logger (ours, uvicorn's, the usage logger) through one
    shared stdout handler, so every log line -- app logs, uvicorn's access
    log, usage/cost events -- looks the same rather than each picking its
    own format.

    json_logs=False: a plain textual formatter (uvicorn's own style).
    json_logs=True: compact JSON, deliberately with no timestamp field --
    a k8s log collector already stamps arrival time.
    """
    level = log_level.upper()

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(uvicorn.logging.DefaultFormatter("%(levelprefix)s %(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in _UVICORN_LOGGERS:
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = [handler]
        uv_logger.setLevel(level)
        uv_logger.propagate = False
