from __future__ import annotations

import time
from collections.abc import Callable

from llm_mitm.config import PricingConfig
from llm_mitm.pricing import compute_cost
from llm_mitm.providers import Usage
from llm_mitm.usage import UsageEvent, record_usage


def _chunk_has_token(payload: dict) -> bool:
    # True for a chunk that carries actual generated output -- not the
    # synthetic role-only opener some providers send immediately, and not
    # a finish_reason-only chunk. Used to time TTFT from something a client
    # would actually perceive as "the model responded".
    delta = payload.get("choices", [{}])[0].get("delta", {})
    return bool(delta.get("content")) or bool(delta.get("tool_calls"))


class RequestAccounting:
    """Owns one chat completion request's accounting from start to its
    recorded usage event: the clock, TTFT detection, turning token usage
    into a cost, and building/emitting the UsageEvent. The route handler
    only needs to start one of these and hand it chunks (if streaming) and
    the final outcome -- the status/cost/event-building rules live here
    once instead of being repeated at every call site.

    `clock` and `sink` are injected (defaulting to time.monotonic and
    record_usage) so tests can drive this with a fake clock and capture
    emitted events directly, without going through FastAPI at all.
    """

    def __init__(
        self,
        key_id: str,
        model_name: str,
        provider_name: str,
        provider_cfg: PricingConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sink: Callable[[UsageEvent], None] = record_usage,
    ) -> None:
        self.key_id = key_id
        self.model_name = model_name
        self.provider_name = provider_name
        self.provider_cfg = provider_cfg
        self._clock = clock
        self._sink = sink
        self._start = clock()
        self.ttft_s: float | None = None

    def observe_chunk(self, payload: dict) -> None:
        # Call for every structured chunk a provider yields during a
        # streaming response, before it's serialized and sent. Records the
        # elapsed time on the first chunk that actually carries generated
        # output; a no-op afterwards.
        if self.ttft_s is None and _chunk_has_token(payload):
            self.ttft_s = self._clock() - self._start

    def finish(self, usage: Usage, status: str, *, stream: bool) -> None:
        cost = compute_cost(
            self.provider_cfg,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            usage.service_tier,
        )
        self._sink(
            UsageEvent(
                key_id=self.key_id,
                model=self.model_name,
                provider=self.provider_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                service_tier=usage.service_tier,
                ttft_s=self.ttft_s,
                cost_usd=cost,
                duration_s=self._clock() - self._start,
                status=status,
                stream=stream,
            )
        )
