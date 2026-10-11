from __future__ import annotations

import hashlib

from fastapi import Request


def get_key_id(request: Request, hash_api_keys: bool = True) -> str:
    """Identify the caller for cost-tracking purposes only (no authorization).

    Whatever bearer token / api-key header the client sends is used as the
    tracking id (hashed down to a short id by default). Any value is
    accepted -- validating it is the reverse proxy's job, not ours.
    """
    key = None
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        key = auth[len("bearer ") :].strip()
    if not key:
        key = request.headers.get("x-api-key") or request.headers.get("api-key")
    if not key:
        return "anonymous"
    if not hash_api_keys:
        return key
    return hashlib.sha256(key.encode()).hexdigest()[:16]
