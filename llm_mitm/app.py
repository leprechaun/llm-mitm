from __future__ import annotations

import json
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm_mitm.auth import get_key_id
from llm_mitm.config import load_config
from llm_mitm.pricing import compute_cost
from llm_mitm.providers import bedrock, ollama
from llm_mitm.usage import UsageEvent, record_usage

CONFIG_PATH = os.environ.get("LLM_MITM_CONFIG", "config.yaml")
config = load_config(CONFIG_PATH)
MODELS = {m.name: m for m in config.models}
PROVIDERS = {"bedrock": bedrock, "ollama": ollama}

app = FastAPI(title="llm-mitm")
logger = logging.getLogger("llm_mitm")


def _openai_error(message: str, error_type: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": None}},
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": entry.provider}
            for name, entry in MODELS.items()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    model_entry = MODELS.get(model_name)
    if model_entry is None:
        return _openai_error(f"model '{model_name}' not found", "invalid_request_error", 404)
    if not body.get("messages"):
        return _openai_error("field 'messages' is required", "invalid_request_error", 400)

    key_id = get_key_id(request, hash_api_keys=config.server.hash_api_keys)
    provider = PROVIDERS[model_entry.provider]
    provider_cfg = model_entry.provider_config
    stream = bool(body.get("stream", False))
    start = time.monotonic()

    if stream:
        usage_holder = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        async def generate():
            status = "ok"
            try:
                async for sse_chunk in provider.chat_stream(body, provider_cfg, usage_holder):
                    yield sse_chunk
            except Exception as exc:  # closed out as a terminal SSE error event below
                status = "error"
                logger.exception("streaming chat completion failed (model=%s)", model_name)
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'upstream_error'}})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                cost = compute_cost(
                    provider_cfg,
                    usage_holder["prompt_tokens"],
                    usage_holder["completion_tokens"],
                    usage_holder["cache_read_tokens"],
                    usage_holder["cache_write_tokens"],
                )
                record_usage(
                    UsageEvent(
                        key_id=key_id,
                        model=model_name,
                        provider=model_entry.provider,
                        prompt_tokens=usage_holder["prompt_tokens"],
                        completion_tokens=usage_holder["completion_tokens"],
                        cache_read_tokens=usage_holder["cache_read_tokens"],
                        cache_write_tokens=usage_holder["cache_write_tokens"],
                        cost_usd=cost,
                        duration_s=time.monotonic() - start,
                        status=status,
                        stream=True,
                    )
                )

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        data, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens = await provider.chat(
            body, provider_cfg
        )
    except Exception as exc:  # noqa: BLE001 -- turned into a clean upstream_error response below
        record_usage(
            UsageEvent(
                key_id=key_id,
                model=model_name,
                provider=model_entry.provider,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                duration_s=time.monotonic() - start,
                status="error",
                stream=False,
            )
        )
        return _openai_error(str(exc), "upstream_error", 502)

    cost = compute_cost(provider_cfg, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens)
    record_usage(
        UsageEvent(
            key_id=key_id,
            model=model_name,
            provider=model_entry.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost,
            duration_s=time.monotonic() - start,
            status="ok",
            stream=False,
        )
    )
    return JSONResponse(content=data)
