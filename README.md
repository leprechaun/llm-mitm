# llm-mitm

A tiny, self-hosted replacement for the parts of [litellm](https://github.com/BerriAI/litellm)
actually needed here: an OpenAI-compatible `/v1/chat/completions` endpoint that
fans out to **Amazon Bedrock** (via the Converse API) and **Ollama**, with
per-API-key cost/usage tracking exported as structured stdout logs and
Prometheus metrics.

## What it is not

- No authorization/authentication. Any bearer token is accepted and treated
  as an identity to track cost against; validating it is the job of a
  reverse proxy in front of this service.
- No image/vision support yet.
- No tool/function calling translation for Bedrock yet (Ollama gets this for
  free, since its requests are proxied through mostly as-is).
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
tier is chosen by prompt (context) size, not by total tokens.

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
rejected based on it, it's purely a bucket for accounting.

Each request emits:

1. A JSON line on stdout via the `llm_mitm.usage` logger:
   ```json
   {"key_id": "...", "model": "claude-sonnet", "provider": "bedrock", "prompt_tokens": 123, "completion_tokens": 45, "cost_usd": 0.0012, "duration_s": 1.3, "status": "ok", "stream": false, "ts": 1760169600.0}
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
