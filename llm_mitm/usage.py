from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass

from llm_mitm.metrics import COST_USD, REQUEST_DURATION, REQUESTS, TOKENS

usage_logger = logging.getLogger("llm_mitm.usage")
usage_logger.setLevel(logging.INFO)
usage_logger.propagate = False
if not usage_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    usage_logger.addHandler(_handler)


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
    ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = time.time()


def record_usage(event: UsageEvent) -> None:
    usage_logger.info(json.dumps(asdict(event)))

    REQUESTS.labels(event.model, event.provider, event.key_id, event.status).inc()
    TOKENS.labels(event.model, event.provider, event.key_id, "prompt").inc(event.prompt_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "completion").inc(event.completion_tokens)
    COST_USD.labels(event.model, event.provider, event.key_id).inc(event.cost_usd)
    REQUEST_DURATION.labels(event.model, event.provider).observe(event.duration_s)
