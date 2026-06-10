from __future__ import annotations

import json

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import upstream_auth_headers, validate_bridge_auth
from .config import Settings
from .models import CLAUDE_MODELS, build_model_route

_HEADER_ALLOW = {
    "anthropic-version",
    "anthropic-beta",
    "content-type",
    "accept",
    "user-agent",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-os",
    "x-stainless-arch",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-app",
    "x-client-request-id",
    "x-claude-code-session-id",
}

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def _timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=30.0,
        read=settings.request_timeout_seconds,
        write=30.0,
        pool=30.0,
    )


def _anthropic_error(message: str, status_code: int = 400, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _proxy_headers(request: Request, upstream_auth: dict[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HEADER_ALLOW:
            headers[key] = value
    headers.update(upstream_auth)
    if "Content-Type" not in headers and "content-type" not in {k.lower(): v for k, v in headers.items()}:
        headers["Content-Type"] = "application/json"
    if "anthropic-version" not in {k.lower(): v for k, v in headers.items()}:
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _response_headers(resp: httpx.Response) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in resp.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP:
            continue
        if lower in {"content-type", "cache-control", "x-request-id", "request-id", "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-tokens-limit", "anthropic-ratelimit-tokens-remaining"}:
            out[key] = value
    return out


async def handle_messages(request: Request, settings: Settings):
    auth_error = validate_bridge_auth(request, settings)
    if auth_error is not None:
        return auth_error

    body = await request.body()
    try:
        parsed = json.loads(body)
    except Exception:
        return _anthropic_error("Request body must be valid JSON")
    if not isinstance(parsed, dict):
        return _anthropic_error("Request body must be a JSON object")

    model = str(parsed.get("model") or "")
    if not model:
        return _anthropic_error("model is required")
    route = build_model_route(model)
    if route.route_model not in CLAUDE_MODELS:
        return _anthropic_error(
            f"model {model} is not available on the Anthropic bridge",
            err_type="unsupported_model_protocol",
        )
    if route.route_model != model:
        parsed["model"] = route.route_model
        body = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()

    upstream_auth = upstream_auth_headers(request, settings)
    if isinstance(upstream_auth, JSONResponse):
        return upstream_auth
    headers = _proxy_headers(request, upstream_auth)
    url = f"{settings.sub2api_base_url}/antigravity/v1/messages"
    stream = bool(parsed.get("stream"))

    if stream:
        return StreamingResponse(
            _stream_messages(settings, url, body, headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            resp = await client.post(url, content=body, headers=headers)
        except Exception as exc:
            return _anthropic_error(f"Upstream request failed: {exc}", status_code=502, err_type="upstream_error")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_response_headers(resp),
        media_type=resp.headers.get("content-type", "application/json"),
    )


async def _stream_messages(settings: Settings, url: str, body: bytes, headers: dict[str, str]):
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            async with client.stream("POST", url, content=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    payload = (await resp.aread()).decode("utf-8", "replace")
                    if payload:
                        yield payload
                    else:
                        yield 'event: error\ndata: {"type":"error","error":{"type":"upstream_error","message":"upstream request failed"}}\n\n'
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except Exception as exc:
            msg = json.dumps({"type": "error", "error": {"type": "upstream_error", "message": f"Upstream stream failed: {exc}"}}, ensure_ascii=False)
            yield f"event: error\ndata: {msg}\n\n"
