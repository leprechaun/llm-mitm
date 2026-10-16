from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

from llm_mitm.logging_setup import JSON_FIELDS
from llm_mitm.metrics import (
    COST_USD,
    OUTPUT_TOKENS_PER_SECOND,
    REQUEST_DURATION,
    REQUESTS,
    TIME_TO_FIRST_TOKEN,
    TOKENS,
)

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
    # Bedrock-only: which service tier actually served the request (e.g.
    # "flex", "priority", "default"), or None for providers/models that
    # don't have this concept. Logged for billing reconciliation -- not a
    # Prometheus label, to avoid multiplying metric cardinality.
    service_tier: str | None = None
    # Time from request start to the first streamed token. None for
    # non-streaming requests (there's nothing to measure -- the whole
    # response arrives at once) or if the stream never produced a token
    # (e.g. it errored before generating anything).
    ttft_s: float | None = None
    # Derived in __post_init__ unless passed explicitly -- completion_tokens
    # divided by the generation time. For streaming requests, generation
    # time is duration_s - ttft_s (excludes the prefill/TTFT phase); for
    # non-streaming (ttft_s is None) it's the full duration_s.
    tokens_per_second: float | None = None
    ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = time.time()
        if self.tokens_per_second is None and self.completion_tokens > 0:
            generation_s = self.duration_s - self.ttft_s if self.ttft_s else self.duration_s
            if generation_s > 0:
                self.tokens_per_second = self.completion_tokens / generation_s


def record_usage(event: UsageEvent) -> None:
    fields = asdict(event)
    # extra= attaches the real fields onto the LogRecord so the JSON
    # formatter can promote them to top-level keys instead of nesting them
    # as an escaped string; the JSON message text itself is just for
    # textual/local-mode readability.
    usage_logger.info(json.dumps(fields), extra={JSON_FIELDS: fields})

    REQUESTS.labels(event.model, event.provider, event.key_id, event.status).inc()
    TOKENS.labels(event.model, event.provider, event.key_id, "prompt").inc(event.prompt_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "completion").inc(event.completion_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "cache_read").inc(event.cache_read_tokens)
    TOKENS.labels(event.model, event.provider, event.key_id, "cache_write").inc(event.cache_write_tokens)
    COST_USD.labels(event.model, event.provider, event.key_id).inc(event.cost_usd)
    REQUEST_DURATION.labels(event.model, event.provider).observe(event.duration_s)
    if event.ttft_s is not None:
        TIME_TO_FIRST_TOKEN.labels(event.model, event.provider).observe(event.ttft_s)
    if event.tokens_per_second is not None:
        OUTPUT_TOKENS_PER_SECOND.labels(event.model, event.provider).observe(event.tokens_per_second)
