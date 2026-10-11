from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "llm_mitm_requests_total",
    "Chat completion requests",
    ["model", "provider", "key_id", "status"],
)
TOKENS = Counter(
    "llm_mitm_tokens_total",
    "Tokens processed",
    ["model", "provider", "key_id", "kind"],
)
COST_USD = Counter(
    "llm_mitm_cost_usd_total",
    "Estimated cost in USD",
    ["model", "provider", "key_id"],
)
REQUEST_DURATION = Histogram(
    "llm_mitm_request_duration_seconds",
    "Request duration in seconds",
    ["model", "provider"],
)
