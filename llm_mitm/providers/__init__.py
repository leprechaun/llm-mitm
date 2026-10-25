from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    # The single value both chat() and chat_stream() deal in at the
    # provider boundary -- returned whole from a blocking call, or mutated
    # in place as a stream progresses. Giving it defaults means a provider
    # with no concept of a field (e.g. Ollama has no prompt caching or
    # service tiers) just never touches it, instead of every provider
    # having to spell out "0, 0, None" to match Bedrock-only features.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    service_tier: str | None = None

    def update(self, other: Usage) -> None:
        # Copy another Usage's fields onto this one in place -- for
        # providers that mutate a caller-supplied Usage as a stream
        # progresses rather than returning a fresh one each time.
        self.prompt_tokens = other.prompt_tokens
        self.completion_tokens = other.completion_tokens
        self.cache_read_tokens = other.cache_read_tokens
        self.cache_write_tokens = other.cache_write_tokens
        self.service_tier = other.service_tier
