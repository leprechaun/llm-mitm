# llm-mitm -- LLM Man-in-the-Middle

A tiny, self-hosted replacement for the parts of [litellm](https://github.com/BerriAI/litellm)
actually needed here: an OpenAI-compatible `/v1/chat/completions` endpoint that
fans out to **Amazon Bedrock** (via the Converse API) and **Ollama**, with
per-API-key cost/usage tracking exported as structured stdout logs and
Prometheus metrics.

## What it is not

- No authorization/authentication. Any bearer token is accepted and treated
  as an identity to track cost against; validating it is the job of a
  reverse proxy in front of this service.
- No persistent usage database -- usage is a JSON line per request on stdout
  plus Prometheus counters/histograms. Point your log pipeline / Prometheus
  at it.

## Configure

Edit `config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 4000

models:
  - name: claude-sonnet          # the "model" clients ask for
    provider: bedrock
    bedrock:
      model_id: anthropic.claude-3-5-sonnet-20241022-v2:0
      region: us-east-1
      input_cost_per_1m: 3.00    # USD, used for cost tracking only
      output_cost_per_1m: 15.00

  - name: llama3
    provider: ollama
    ollama:
      base_url: http://localhost:11434
      model: llama3
      input_cost_per_1m: 0
      output_cost_per_1m: 0
```

Some models (e.g. long-context Anthropic models on Bedrock) charge a higher
rate once the prompt exceeds a size threshold. Express that with `tiers`:

```yaml
    bedrock:
      model_id: ...
      input_cost_per_1m: 0.10    # base rate, used when prompt_tokens <= 100000
      output_cost_per_1m: 0.50
      tiers:
        - above_tokens: 100000    # applies when prompt_tokens > 100000
          input_cost_per_1m: 0.50
          output_cost_per_1m: 2.50
```

`tiers` works the same way under both `bedrock:` and `ollama:` blocks. The
tier is chosen by context size (prompt + cache tokens), not just prompt
tokens.

Bedrock also supports prompt caching, which reuses repeated context (system
prompt, tool schemas, growing chat history) across turns at a steep discount
instead of billing it as fresh input every time -- a big win for agentic
clients like opencode that resend the whole conversation on every turn. Not
all Bedrock models support it. Enable it per-model and set its (different)
cache rates:

```yaml
    bedrock:
      model_id: ...
      input_cost_per_1m: 0.10
      output_cost_per_1m: 0.50
      cache_read_cost_per_1m: 0.01     # cache hit -- cheap
      cache_write_cost_per_1m: 0.125   # creating/refreshing a cache entry -- pricier than input
      prompt_caching: true
```

Not every Bedrock model handles tool calling the same way. Two per-model
switches handle the quirks:

```yaml
    bedrock:
      model_id: ...
      supports_tools: true           # set false to never send tools to this model at all
      supports_tool_streaming: true  # set false if it rejects ConverseStream + tools together
```

`supports_tool_streaming: false` is for models like Llama 4 Maverick, which
reject streaming requests that also include tools ("This model doesn't
support tool use in streaming mode"). When that happens, a streaming request
with tools transparently falls back to a blocking call replayed as a
single-shot SSE stream, so the client still gets the response shape it asked
for -- streaming without tools is unaffected.

Note: Bedrock has no `tool_choice: "none"` equivalent (only auto / any /
a named tool). If an agent client sends `tool_choice: "none"` on a turn
whose history already contains a tool call, this still sends `toolConfig`
with no `toolChoice` (i.e. behaves like "auto") -- Bedrock rejects the
request outright otherwise ("toolConfig field must be defined when using
toolUse and toolResult content blocks"), so this is the closest available
mapping, not a strict "never call a tool" guarantee.

Bedrock vision-capable models (e.g. Claude Haiku 5.5) accept images: send
them the normal OpenAI way, as an `image_url` content part (either a
`data:image/...;base64,...` URI, or a plain http(s) URL, which gets fetched
and inlined as bytes since Bedrock's image block doesn't accept remote
URLs). Supported formats: PNG, JPEG, GIF, WEBP. Ollama is unaffected --
its requests are passed through to Ollama's own endpoint as-is.

Add as many entries as you like, mixing providers freely. AWS credentials
for Bedrock are picked up the normal boto3 way (env vars, `~/.aws/credentials`,
instance role, SSO profile) -- there's nothing AWS-specific to set in the
config.

Set `LLM_MITM_CONFIG` to point at a different config file path if needed.

## Run

```bash
uv sync
uv run python main.py
```

Point OpenWebUI / opencode / any OpenAI-compatible client at
`http://localhost:4000/v1` as the base URL, with any value as the API key
(it's only used to bucket cost tracking, see below).

## Endpoints

- `POST /v1/chat/completions` -- OpenAI-compatible, streaming and
  non-streaming.
- `GET /v1/models` -- lists the models defined in `config.yaml`.
- `GET /metrics` -- Prometheus metrics.
- `GET /healthz` -- liveness check.

## Cost tracking

Every request's `Authorization: Bearer <key>` (or `x-api-key` /
`api-key` header) is hashed down to a short `key_id` -- nothing is
rejected based on it, it's purely a bucket for accounting. Set
`server.hash_api_keys: false` in `config.yaml` to use the raw key as the
`key_id` instead (e.g. if your keys are already opaque/non-sensitive
labels) -- not recommended if keys are actual secrets, since they'd then
show up as-is in logs and Prometheus labels.

Each request emits:

1. A JSON line on stdout via the `llm_mitm.usage` logger:
   ```json
   {"key_id": "...", "model": "claude-sonnet", "provider": "bedrock", "prompt_tokens": 123, "completion_tokens": 45, "cost_usd": 0.0012, "duration_s": 1.3, "status": "ok", "stream": false, "cache_read_tokens": 0, "cache_write_tokens": 0, "ts": 1760169600.0}
   ```
2. Updates to Prometheus counters/histograms on `/metrics`:
   `llm_mitm_requests_total`, `llm_mitm_tokens_total`,
   `llm_mitm_cost_usd_total`, `llm_mitm_request_duration_seconds` -- all
   labeled by `model`, `provider`, and `key_id` (status too, for requests).

   Note: if you expect a large/unbounded number of distinct API keys, the
   `key_id` label will drive Prometheus cardinality up accordingly.

Nothing is persisted to a database today; wiring either log line or metrics
into a store of your choice is a natural next step if you need historical
querying beyond what Prometheus retains.

## Logging

By default (local dev), general application/uvicorn logs are Python's
normal human-readable textual output. Set `server.json_logs: true` in
`config.yaml` to switch every log line (ours and uvicorn's access/error
logs) to compact JSON on stdout instead -- deliberately with no timestamp
field, since a k8s log collector already stamps arrival time and an
app-side one would just be redundant. The Helm chart sets this to `true`
by default.

`server.log_level` (default `INFO`; one of `DEBUG`/`INFO`/`WARNING`/
`ERROR`/`CRITICAL`) controls the threshold for these same logs -- ours and
uvicorn's -- regardless of `json_logs`.

The `llm_mitm.usage` logger described above goes through this exact same
handler and formatter, so a usage line looks like every other log line
(textual locally, JSON in `json_logs` mode) rather than having its own
separate format -- the JSON payload is just its message text. It's pinned
to `INFO` independent of `server.log_level`, though: cost-tracking events
must never go dark just because you turned down general log verbosity.
