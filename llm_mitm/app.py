from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm_mitm.accounting import RequestAccounting
from llm_mitm.auth import get_key_id
from llm_mitm.config import load_config
from llm_mitm.providers import Usage, bedrock, ollama
from llm_mitm.usage import record_usage

CONFIG_PATH = os.environ.get("LLM_MITM_CONFIG", "config.yaml")
config = load_config(CONFIG_PATH)
MODELS = {m.id: m for m in config.models}
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
            {"id": model_id, "object": "model", "owned_by": entry.provider}
            for model_id, entry in MODELS.items()
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
    # sink=record_usage passed explicitly (rather than relying on
    # RequestAccounting's default) so tests can still patch
    # llm_mitm.app.record_usage, same as before this refactor.
    accounting = RequestAccounting(key_id, model_name, model_entry.provider, provider_cfg, sink=record_usage)

    if stream:
        usage = Usage()

        async def generate():
            # Only ever becomes "ok" once the stream actually finishes below.
            # If the client/gateway disconnects (e.g. a route timeout),
            # Starlette tears this generator down via GeneratorExit --
            # a BaseException, which `except Exception` below does NOT
            # catch -- so status must default to something other than "ok",
            # or an abandoned request would be misrecorded as a success.
            status = "cancelled"
            try:
                async for sse_chunk in provider.chat_stream(body, provider_cfg, usage):
                    accounting.observe_chunk(sse_chunk)
                    yield sse_chunk
                status = "ok"
            except Exception as exc:  # closed out as a terminal SSE error event below
                status = "error"
                logger.exception("streaming chat completion failed (model=%s)", model_name)
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'upstream_error'}})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                accounting.finish(usage, status, stream=True)

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        data, usage = await provider.chat(body, provider_cfg)
    except Exception as exc:  # noqa: BLE001 -- turned into a clean upstream_error response below
        accounting.finish(Usage(), "error", stream=False)
        return _openai_error(str(exc), "upstream_error", 502)

    accounting.finish(usage, "ok", stream=False)
    return JSONResponse(content=data)
