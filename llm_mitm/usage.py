from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

from llm_mitm.metrics import COST_USD, REQUEST_DURATION, REQUESTS, TOKENS

usage_logger = logging.getLogger("llm_mitm.usage")
# Pinned to INFO regardless of the app's configured log_level -- usage
# events are accounting data, not diagnostic noise, so turning down general
# log verbosity must never silence them. Uses the shared handler/formatter
# set up by logging_setup.configure_logging() (propagates to root), so it
# renders the same way as every other log line instead of its own format.
usage_logger.setLevel(logging.INFO)


@dataclass
class UsageEvent:
    key_id: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float
    status: str
    stream: bool
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = time.time()


def record_usage(event: UsageEvent) -> None:
    usage_logger.info(json.dumps(asdict(event)))

    REQUESTS.labels(event.model, event.provider, event.key_id, event.status).inc()
    TOKENS.labels(event.model, event.provider, event.key_id, "prompt").inc(event.prompt_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "completion").inc(event.completion_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "cache_read").inc(event.cache_read_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "cache_write").inc(event.cache_write_tokens)
    COST_USD.labels(event.model, event.provider, event.key_id).inc(event.cost_usd)
    REQUEST_DURATION.labels(event.model, event.provider).observe(event.duration_s)
