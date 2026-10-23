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
