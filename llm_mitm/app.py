from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm_mitm.accounting import RequestAccounting
from llm_mitm.auth import get_key_id
from llm_mitm.config import (
    ModelConfig,
    OpenAIOptions,
    OpenAIProviderConfig,
    load_config,
)
from llm_mitm.drivers import Usage, bedrock_converse, openai
from llm_mitm.usage import record_usage

CONFIG_PATH = os.environ.get("LLM_MITM_CONFIG", "config.yaml")
config = load_config(CONFIG_PATH)
MODELS = {m.id: m for m in config.models}
DRIVERS = {"bedrock_converse": bedrock_converse, "openai": openai}
DISCOVERING_PROVIDERS = {
    p.name: p for p in config.providers if isinstance(p, OpenAIProviderConfig) and p.discover
}

app = FastAPI(title="llm-mitm")
logger = logging.getLogger("llm_mitm")


def _openai_error(message: str, error_type: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": None}},
    )


async def _discover_models(provider: OpenAIProviderConfig) -> dict[str, ModelConfig]:
    # Asked live rather than cached, so a newly pulled model shows up right
    # away and replicas can't disagree. An upstream that doesn't answer just
    # contributes nothing, since it's often a machine that's asleep.
    try:
        upstream_ids = await openai.list_models(provider)
    except Exception:
        logger.warning("model discovery failed (provider=%s)", provider.name, exc_info=True)
        return {}
    # A model that's configured keeps its pricing -- it can't also be reached
    # at $0 through its upstream id.
    configured = {name for m in config.models if m.provider == provider.name for name in (m.name, m.model_id)}
    models = [
        ModelConfig(name=upstream_id, provider=provider.name, model_id=upstream_id, options=OpenAIOptions())
        for upstream_id in upstream_ids
        if upstream_id not in configured
    ]
    return {m.id: m for m in models}


async def _resolve_model(model_name: str) -> ModelConfig | None:
    if model_name in MODELS:
        return MODELS[model_name]
    # Checked against the upstream's list instead of forwarded blindly, because
    # the name becomes a Prometheus label: client typos must not create series.
    provider = DISCOVERING_PROVIDERS.get(str(model_name).partition("/")[0])
    if provider is None:
        return None
    return (await _discover_models(provider)).get(model_name)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models() -> dict:
    models = dict(MODELS)
    for discovered in await asyncio.gather(*map(_discover_models, DISCOVERING_PROVIDERS.values())):
        models.update(discovered)
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": entry.provider}
            for model_id, entry in models.items()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    model_entry = await _resolve_model(model_name)
    if model_entry is None:
        return _openai_error(f"model '{model_name}' not found", "invalid_request_error", 404)
    if not body.get("messages"):
        return _openai_error("field 'messages' is required", "invalid_request_error", 400)

    key_id = get_key_id(request, hash_api_keys=config.server.hash_api_keys)
    provider = config.provider_for(model_entry)
    driver = DRIVERS[provider.driver]
    stream = bool(body.get("stream", False))
    # sink=record_usage passed explicitly (rather than relying on
    # RequestAccounting's default) so tests can still patch
    # llm_mitm.app.record_usage, same as before this refactor.
    accounting = RequestAccounting(
        key_id, model_name, provider.name, provider.driver, model_entry, sink=record_usage
    )

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
                # Providers yield structured chunks, not wire text -- this
                # is the one place that knows about SSE framing ("data: ",
                # json.dumps, [DONE]).
                async for payload in driver.chat_stream(body, provider, model_entry, usage):
                    accounting.observe_chunk(payload)
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
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
        data, usage = await driver.chat(body, provider, model_entry)
    except Exception as exc:  # noqa: BLE001 -- turned into a clean upstream_error response below
        accounting.finish(Usage(), "error", stream=False)
        return _openai_error(str(exc), "upstream_error", 502)

    accounting.finish(usage, "ok", stream=False)
    return JSONResponse(content=data)
