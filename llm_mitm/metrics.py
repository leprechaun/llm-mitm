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
TIME_TO_FIRST_TOKEN = Histogram(
    "llm_mitm_time_to_first_token_seconds",
    "Time from request start to the first streamed token (streaming requests only)",
    ["model", "provider"],
)
OUTPUT_TOKENS_PER_SECOND = Histogram(
    "llm_mitm_output_tokens_per_second",
    "Output token generation rate",
    ["model", "provider"],
    buckets=(5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
)
