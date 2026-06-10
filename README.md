# Reverse Proxy Antigravity to Responses

A small FastAPI bridge that exposes OpenAI-compatible `Responses API` and `Chat Completions` endpoints for Antigravity-backed Gemini and Claude models.

It is designed to sit in front of an existing Sub2API deployment and convert client requests into the upstream formats used by Antigravity, Gemini, and Anthropic-compatible routes.

## What it does

- Provides OpenAI-compatible endpoints:
  - `POST /v1/responses`
  - `POST /v1/chat/completions`
  - `GET /v1/models`
- Converts OpenAI Responses / Chat payloads into Gemini `generateContent` style requests.
- Supports Antigravity `v1internal` calls for selected Gemini models.
- Converts Gemini output back into OpenAI Responses or Chat Completions format.
- Preserves final answer text separately from Gemini thought parts.
- Emits reasoning content when upstream Gemini thinking output is requested and returned.
- Supports Claude/Anthropic-style routes through the existing bridge code.
- Can read Antigravity account metadata from the Sub2API database when direct `v1internal` calls are needed.

## Main routing idea

Client request:

```text
OpenAI-compatible client
  -> this bridge /v1/responses or /v1/chat/completions
  -> model routing / request conversion
  -> Sub2API or Antigravity v1internal upstream
  -> response conversion back to OpenAI-compatible JSON/SSE
```

For Gemini 3.5 Flash, the bridge maps user-facing models to Antigravity physical model IDs:

```text
gemini-3.5-flash-high   -> gemini-3-flash-agent
gemini-3.5-flash-medium -> gemini-3.5-flash-low
gemini-3.5-flash-low    -> gemini-3.5-flash-extra-low
gemini-3.5-flash        -> gemini-3.5-flash-medium
```

For Gemini 3.1 Pro:

```text
gemini-3.1-pro-high -> gemini-pro-agent
gemini-3.1-pro-low  -> gemini-3.1-pro-low
```

## Configuration

Configuration is supplied through environment variables.

Common variables:

```env
SUB2API_BASE_URL=http://sub2api:8080
SUB2API_API_KEY=
BRIDGE_API_KEY=
PASSTHROUGH_CLIENT_AUTH=true
REQUEST_TIMEOUT_SECONDS=900

DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_USER=sub2api
DATABASE_PASSWORD=
DATABASE_DBNAME=sub2api

ANTIGRAVITY_GROUP_NAMES=Antigravity
```

Notes:

- `SUB2API_BASE_URL` points to the internal Sub2API service.
- `SUB2API_API_KEY` is used when the bridge should call Sub2API with a fixed upstream key.
- `BRIDGE_API_KEY` can be used as a dedicated bridge key.
- `PASSTHROUGH_CLIENT_AUTH=true` allows client Authorization headers to be passed through when no fixed upstream key is configured.
- Database variables let the bridge read Sub2API account metadata needed for Antigravity OAuth-backed `v1internal` calls.
- Do not commit real `.env` files or credentials.

## Running with Docker

Build:

```bash
docker build -t antigravity-responses-bridge .
```

Run example:

```bash
docker run --rm -p 8090:8090 \
  --env-file .env \
  antigravity-responses-bridge
```

The container starts:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8090 --proxy-headers
```

## Sub2API / reverse proxy integration

A typical deployment puts this bridge on the same Docker network as Sub2API and routes selected paths to it from an nginx/Caddy layer.

Example path split:

```text
/v1/responses          -> bridge:8090/v1/responses
/v1/chat/completions  -> bridge:8090/v1/chat/completions
/v1/models            -> bridge:8090/v1/models
/other paths          -> sub2api:8080
```

The bridge can still call Sub2API for passthrough or non-direct upstream paths.

## Reasoning / thinking output

When Gemini upstream returns thought parts, the bridge keeps them separate from the final answer.

Responses API output uses a reasoning item:

```json
{
  "type": "reasoning",
  "status": "completed",
  "summary": [
    {
      "type": "summary_text",
      "text": "..."
    }
  ]
}
```

Chat Completions output uses:

```json
{
  "role": "assistant",
  "content": "final answer",
  "reasoning_content": "reasoning text"
}
```

## API examples

Responses:

```bash
curl http://localhost:8090/v1/responses \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.5-flash-high",
    "input": "ping",
    "max_output_tokens": 64,
    "stream": false
  }'
```

Chat Completions:

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-pro-high",
    "messages": [
      {"role": "user", "content": "ping"}
    ],
    "max_tokens": 64,
    "stream": false
  }'
```

## Security notes

- This repository should contain code only.
- Do not commit `.env`, API keys, OAuth tokens, database dumps, or account backups.
- The bridge may read account metadata from Sub2API at runtime; those values must stay in the deployment environment or database, not in source control.
