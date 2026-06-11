from __future__ import annotations

import copy
import hashlib
import json
import secrets
import time
import uuid
from typing import Any

import httpx
import asyncpg
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import client_api_key, upstream_auth_headers, validate_bridge_auth
from .anthropic_responses import (
    anthropic_stream_state,
    stream_open_events as anthropic_stream_open_events,
    anthropic_to_responses,
    parse_anthropic_sse,
    responses_to_anthropic,
    responses_sse,
    stream_close_events as anthropic_stream_close_events,
    stream_event_to_responses,
)
from .config import Settings
from .gemini_converter import (
    BadRequest,
    UnsupportedFeature,
    extract_text,
    extract_thought_text,
    extract_function_calls,
    item_id,
    openai_error,
    response_id,
    responses_json,
    responses_to_gemini,
    stream_delta_event,
    stream_done_events,
    stream_start_events,
    upstream_error_from_body,
    usage_from_gemini,
)
from .models import (
    CLAUDE_MODELS,
    GEMINI_RESPONSES_MODELS,
    RESPONSES_MODELS,
    build_model_route,
    gemini_candidate_upstream_models,
    is_official_v1internal_gemini_model,
    resolve_upstream_model,
    model_list,
)
from .sse import parse_sse_events, sse_event


EMPTY_GEMINI_FALLBACK_TEXT = (
    "上游 Gemini 本次返回了空内容。桥接层已避免请求中断；请重试本轮请求，"
    "或切换到 Claude/Gemini 高配模型。"
)

EMPTY_CLAUDE_FALLBACK_TEXT = (
    "上游 Claude/Antigravity 本次流式响应没有返回可见文本。桥接层已避免空响应中断；"
    "请重试本轮请求。"
)

V1INTERNAL_GENERATE_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:generateContent"
V1INTERNAL_STREAM_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse"


def _gemini_upstream_model(model: str) -> str:
    # Compatibility shim for older call sites. New code passes explicit
    # upstream_model to avoid confusing client-facing and physical model IDs.
    return resolve_upstream_model(model)


def _gemini_thinking_level(gemini_body: dict[str, Any]) -> str | None:
    gen = gemini_body.get("generationConfig") if isinstance(gemini_body, dict) else None
    thinking = gen.get("thinkingConfig") if isinstance(gen, dict) else None
    if not isinstance(thinking, dict):
        return None
    level = thinking.get("thinkingLevel")
    include = thinking.get("includeThoughts")
    parts: list[str] = []
    if level is not None:
        parts.append(f"level={level}")
    if include is not None:
        parts.append(f"includeThoughts={bool(include)}")
    return ",".join(parts) if parts else None


def _log_gemini_upstream(route_model: str, upstream_model: str, method: str, gemini_body: dict[str, Any], *, stream: bool = False, direct: bool = False) -> None:
    suffix = ":streamGenerateContent?alt=sse" if stream else f":{method}"
    if direct:
        endpoint = "official_v1internal"
        path = f"/v1internal{suffix}"
    else:
        endpoint = "sub2api_antigravity"
        path = f"/antigravity/v1beta/models/{upstream_model}{suffix}"
    print(
        "nf_bridge.gemini_upstream "
        f"route_model={route_model} upstream_model={upstream_model} "
        f"endpoint={endpoint} path={path} "
        f"thinkingLevel={_gemini_thinking_level(gemini_body) or '-'}",
        flush=True,
    )


def _gemini_url(settings: Settings, upstream_model: str, method: str, *, stream: bool = False) -> str:
    suffix = ":streamGenerateContent?alt=sse" if stream else f":{method}"
    return f"{settings.sub2api_base_url}/antigravity/v1beta/models/{upstream_model}{suffix}"


def _parse_epoch_seconds(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


async def _load_antigravity_oauth(settings: Settings) -> tuple[str, str]:
    conn = await asyncpg.connect(
        host=settings.database_host,
        port=settings.database_port,
        user=settings.database_user,
        password=settings.database_password,
        database=settings.database_dbname,
    )
    try:
        rows = await conn.fetch(
            """
            select
                id,
                credentials->>'project_id' as project_id,
                credentials->>'access_token' as access_token,
                credentials->>'expires_at' as credential_expires_at,
                extract(epoch from expires_at)::bigint as account_expires_at
            from accounts
            where deleted_at is null
              and status = 'active'
              and lower(platform) = 'antigravity'
              and credentials ? 'project_id'
              and credentials ? 'access_token'
            order by schedulable desc, priority asc, last_used_at nulls first, id asc
            limit 10
            """
        )
    finally:
        await conn.close()
    if not rows:
        raise RuntimeError("No active Antigravity OAuth account with project/access token was found")

    now = int(time.time())
    near_expiry_skew = 60
    expired_candidate: tuple[str, str, Any] | None = None
    for row in rows:
        project_id = str(row.get("project_id") or "").strip()
        access_token = str(row.get("access_token") or "").strip()
        if not project_id or not access_token:
            continue
        expires_at = _parse_epoch_seconds(row.get("credential_expires_at")) or _parse_epoch_seconds(row.get("account_expires_at"))
        if expires_at and expires_at <= now + near_expiry_skew:
            if expired_candidate is None:
                expired_candidate = (project_id, access_token, row.get("id"))
            continue
        return project_id, access_token

    if expired_candidate is not None:
        project_id, access_token, account_id = expired_candidate
        print(
            "nf_bridge.gemini_upstream token_status=expired_or_near_expiry "
            f"account_id={account_id}",
            flush=True,
        )
        # Preserve the previous direct-call behavior: Sub2API owns OAuth refresh;
        # the bridge only warns here and lets the upstream response decide.
        return project_id, access_token

    raise RuntimeError("Antigravity OAuth account is missing project/access token")


def _stable_session_id(gemini_body: dict[str, Any]) -> str:
    contents = gemini_body.get("contents") if isinstance(gemini_body, dict) else None
    if isinstance(contents, list):
        for content in contents:
            if not isinstance(content, dict) or content.get("role") != "user":
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                    digest = hashlib.sha256(part["text"].encode("utf-8")).digest()
                    value = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
                    return f"-{value}"
    return f"-{uuid.uuid4().int % 9_000_000_000_000_000_000}"


def _v1internal_request(gemini_body: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(gemini_body)
    # CLIProxyAPI's Antigravity executor wraps Gemini payloads under request,
    # removes safetySettings, and supplies a per-conversation sessionId.
    request.pop("safetySettings", None)
    request.setdefault("sessionId", _stable_session_id(request))
    return request


def _v1internal_body(project_id: str, upstream_model: str, gemini_body: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": project_id,
        "requestId": f"agent-{uuid.uuid4()}",
        "userAgent": "antigravity",
        "requestType": "agent",
        "model": upstream_model,
        "request": _v1internal_request(gemini_body),
    }


async def _gemini_request_params(
    settings: Settings,
    route_model: str,
    upstream_model: str,
    method: str,
    gemini_body: dict[str, Any],
    upstream_headers: dict[str, str],
    *,
    stream: bool = False,
) -> tuple[str, dict[str, Any], dict[str, str], bool]:
    if is_official_v1internal_gemini_model(route_model):
        project_id, access_token = await _load_antigravity_oauth(settings)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "antigravity",
            "x-goog-api-client": "antigravity",
            "Connection": "close",
        }
        url = V1INTERNAL_STREAM_URL if stream else V1INTERNAL_GENERATE_URL
        return url, _v1internal_body(project_id, upstream_model, gemini_body), headers, True
    return _gemini_url(settings, upstream_model, method, stream=stream), gemini_body, upstream_headers, False


def _is_retryable_model_error(status_code: int, body: str) -> bool:
    if status_code not in {400, 404}:
        return False
    lower = (body or "").lower()
    return any(marker in lower for marker in ["invalid argument", "model", "not found", "unsupported", "not available"])


def _timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=30.0,
        read=settings.request_timeout_seconds,
        write=30.0,
        pool=30.0,
    )


async def _client_group_info(request: Request, settings: Settings) -> dict[str, str | None] | None:
    key = client_api_key(request)
    if not key:
        return None
    try:
        conn = await asyncpg.connect(
            host=settings.database_host,
            port=settings.database_port,
            user=settings.database_user,
            password=settings.database_password,
            database=settings.database_dbname,
        )
        try:
            row = await conn.fetchrow(
                """
                select g.name as group_name,
                       g.platform as group_platform
                from api_keys k
                left join groups g on g.id = k.group_id
                where k.key = $1
                  and k.deleted_at is null
                  and k.status = 'active'
                limit 1
                """,
                key,
            )
        finally:
            await conn.close()
        if row:
            return {
                "group_name": str(row["group_name"]) if row.get("group_name") is not None else None,
                "group_platform": str(row["group_platform"]) if row.get("group_platform") is not None else None,
            }
    except Exception:
        return None
    return None


async def _client_group_name(request: Request, settings: Settings) -> str | None:
    info = await _client_group_info(request, settings)
    if info and info.get("group_name"):
        return str(info["group_name"])
    return None


def _is_antigravity_group(group_name: str | None, settings: Settings, group_platform: str | None = None) -> bool:
    platform = (group_platform or "").strip().lower()
    if platform == "antigravity":
        return True
    if not group_name:
        return False
    allowed_names = {name.strip().lower() for name in settings.antigravity_group_names if name.strip()}
    return group_name.strip().lower() in allowed_names


async def _should_bridge_request(request: Request, settings: Settings) -> tuple[bool, dict[str, Any]]:
    info = await _client_group_info(request, settings) or {}
    group_name = info.get("group_name")
    group_platform = info.get("group_platform")
    bridge_key_used = settings.bridge_api_key is not None and client_api_key(request) == settings.bridge_api_key
    group_is_antigravity = _is_antigravity_group(group_name, settings, group_platform)
    reason = "antigravity_group_or_platform" if group_is_antigravity else "non_antigravity_passthrough"
    if bridge_key_used and not group_is_antigravity:
        # Fail closed: a configured BRIDGE_API_KEY only enables bridge handling
        # when the presented key also resolves to an Antigravity group/platform.
        reason = "bridge_key_not_antigravity_passthrough"
    return group_is_antigravity, {
        "group_name": group_name,
        "group_platform": group_platform,
        "bridge_key_used": bridge_key_used,
        "reason": reason,
    }


def _log_value(value: Any) -> str:
    text = str(value or "-").replace(" ", "_")
    return text[:120] if text else "-"


def _log_route_decision(endpoint: str, model: str | None, route_model: str | None, should_bridge: bool, context: dict[str, Any], *, reason: str | None = None) -> None:
    decision = "bridge" if should_bridge else "passthrough"
    print(
        "nf_bridge.route "
        f"endpoint={endpoint} decision={decision} "
        f"group_name={_log_value(context.get('group_name'))} "
        f"group_platform={_log_value(context.get('group_platform'))} "
        f"bridge_key_used={bool(context.get('bridge_key_used'))} "
        f"model={_log_value(model)} route_model={_log_value(route_model)} "
        f"reason={_log_value(reason or context.get('reason'))}",
        flush=True,
    )


def _passthrough_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": request.headers.get("content-type") or "application/json"}
    for name in ["authorization", "x-api-key", "openai-organization", "openai-project", "idempotency-key"]:
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


async def _passthrough_openai_models(request: Request, settings: Settings):
    url = f"{settings.sub2api_base_url}/v1/models"
    headers = _passthrough_headers(request)
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            resp = await client.get(url, headers=headers)
        except Exception as exc:
            status, body = openai_error(f"Upstream request failed: {exc}", status_code=502, err_type="upstream_error", code="upstream_error")
            return JSONResponse(status_code=status, content=body)
    content_type = resp.headers.get("content-type") or "application/json"
    if "json" in content_type.lower():
        try:
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            pass
    return Response(content=resp.content, status_code=resp.status_code, media_type=content_type)


async def handle_models(request: Request, settings: Settings):
    should_bridge, context = await _should_bridge_request(request, settings)
    _log_route_decision("/v1/models", None, None, should_bridge, context)
    if should_bridge:
        return JSONResponse(status_code=200, content=model_list())
    return await _passthrough_openai_models(request, settings)


async def _passthrough_openai_responses(request: Request, settings: Settings, req: dict[str, Any], *, suffix: str = ""):
    url = f"{settings.sub2api_base_url}/v1/responses{suffix}"
    headers = _passthrough_headers(request)
    if bool(req.get("stream")):
        async def gen():
            model = str(req.get("model") or "openai")
            async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
                try:
                    async with client.stream("POST", url, json=req, headers=headers) as resp:
                        content_type = (resp.headers.get("content-type") or "").lower()
                        if resp.status_code >= 400 or "text/event-stream" not in content_type:
                            body = (await resp.aread()).decode("utf-8", "replace")
                            message = _passthrough_error_message(resp.status_code, body)
                            yield _text_response_sse(model, message).encode()
                            return
                        saw_chunk = False
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                saw_chunk = True
                                yield chunk
                        if not saw_chunk:
                            yield _text_response_sse(model, "Upstream stream ended without returning any data.").encode()
                except Exception as exc:
                    yield _text_response_sse(model, f"Upstream stream failed: {exc}").encode()
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            resp = await client.post(url, json=req, headers=headers)
        except Exception as exc:
            status, body = openai_error(f"Upstream request failed: {exc}", status_code=502, err_type="upstream_error", code="upstream_error")
            return JSONResponse(status_code=status, content=body)
    try:
        content = resp.json()
        return JSONResponse(status_code=resp.status_code, content=content)
    except Exception:
        return JSONResponse(status_code=resp.status_code, content={"error": {"type": "upstream_error", "message": resp.text[:1000], "code": "upstream_error"}})


def _payload_object(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("response")
    if isinstance(nested, dict):
        return nested
    return payload


def _passthrough_error_message(status_code: int, body: str) -> str:
    message = (body or "").strip()[:1000] or f"Upstream returned HTTP {status_code}"
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                message = str(err.get("message") or err.get("code") or message)
            elif parsed.get("message"):
                message = str(parsed.get("message"))
    except Exception:
        pass
    return f"OpenAI upstream returned HTTP {status_code}: {message}"


def _chat_completion_json(model: str, text: str, usage: dict[str, int] | None = None, function_calls: list[dict[str, str]] | None = None, reasoning_text: str | None = None) -> dict[str, Any]:
    now = int(time.time())
    calls = function_calls or []
    message: dict[str, Any] = {"role": "assistant", "content": text if text else (None if calls else "")}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    finish_reason = "stop"
    if calls:
        message["tool_calls"] = [
            {
                "id": call.get("call_id") or ("call_" + secrets.token_hex(8)),
                "type": "function",
                "function": {"name": call.get("name") or "tool", "arguments": call.get("arguments") or "{}"},
            }
            for call in calls
        ]
        finish_reason = "tool_calls"
    body: dict[str, Any] = {
        "id": "chatcmpl_" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        body["usage"] = {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
    return body


def _chat_error_sse(model: str, text: str) -> str:
    cid = "chatcmpl_" + secrets.token_hex(12)
    now = int(time.time())
    chunks = [
        {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return "".join(f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _rewrite_chat_model(payload: Any, model: str) -> Any:
    if isinstance(payload, dict) and payload.get("model"):
        out = dict(payload)
        out["model"] = model
        return out
    return payload


def _rewrite_chat_sse_block(raw: str, model: str) -> str:
    data_lines: list[str] = []
    passthrough_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
        else:
            passthrough_lines.append(line)
    if not data_lines:
        return raw + "\n\n"
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return "data: [DONE]\n\n"
    try:
        payload = json.loads(data)
        payload = _rewrite_chat_model(payload, model)
        return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    except Exception:
        return raw + "\n\n"


async def _passthrough_openai_chat(request: Request, settings: Settings, req: dict[str, Any], *, original_model: str | None = None, headers: dict[str, str] | None = None):
    url = f"{settings.sub2api_base_url}/v1/chat/completions"
    out_model = original_model or str(req.get("model") or "") or "chat"
    send_headers = headers or _passthrough_headers(request)
    send_headers = {**send_headers, "Content-Type": "application/json"}
    if bool(req.get("stream")):
        async def gen():
            async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
                try:
                    async with client.stream("POST", url, json=req, headers=send_headers) as resp:
                        content_type = (resp.headers.get("content-type") or "").lower()
                        if resp.status_code >= 400 or "text/event-stream" not in content_type:
                            body = (await resp.aread()).decode("utf-8", "replace")
                            yield _chat_error_sse(out_model, _passthrough_error_message(resp.status_code, body)).encode()
                            return
                        buffer = ""
                        async for chunk in resp.aiter_text():
                            if not chunk:
                                continue
                            buffer += chunk.replace("\r\n", "\n")
                            while True:
                                idx = buffer.find("\n\n")
                                if idx < 0:
                                    break
                                raw = buffer[:idx]
                                buffer = buffer[idx + 2:]
                                yield _rewrite_chat_sse_block(raw, out_model).encode()
                        if buffer.strip():
                            yield _rewrite_chat_sse_block(buffer.strip(), out_model).encode()
                except Exception as exc:
                    yield _chat_error_sse(out_model, f"Upstream stream failed: {exc}").encode()
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            resp = await client.post(url, json=req, headers=send_headers)
        except Exception as exc:
            status, body = openai_error(f"Upstream request failed: {exc}", status_code=502, err_type="upstream_error", code="upstream_error")
            return JSONResponse(status_code=status, content=body)
    try:
        content = resp.json()
    except Exception:
        return JSONResponse(status_code=resp.status_code, content={"error": {"type": "upstream_error", "message": resp.text[:1000], "code": "upstream_error"}})
    content = _rewrite_chat_model(content, out_model)
    return JSONResponse(status_code=resp.status_code, content=content)


def _chat_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"text", "input_text"}:
                    parts.append(str(part.get("text") or ""))
                elif part.get("text") is not None:
                    parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content)


def _chat_content_to_responses(content: Any) -> Any:
    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    out.append({"type": "input_text", "text": part})
                continue
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                out.append({"type": "input_text", "text": str(part.get("text") or "")})
            elif ptype in {"image_url", "input_image"}:
                image_url = part.get("image_url") or part.get("url")
                out.append({"type": "input_image", "image_url": image_url})
            elif part.get("text") is not None:
                out.append({"type": "input_text", "text": str(part.get("text") or "")})
        return out
    return content if isinstance(content, str) else str(content or "")


def _chat_to_responses_req(req: dict[str, Any], model: str) -> dict[str, Any]:
    messages = req.get("messages")
    if not isinstance(messages, list):
        raise BadRequest("messages must be an array")
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise BadRequest("messages items must be objects")
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role in {"system", "developer"}:
            text = _chat_text_from_content(content).strip()
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            input_items.append({"type": "function_call_output", "call_id": str(msg.get("tool_call_id") or msg.get("call_id") or ""), "output": content or ""})
            continue
        if role not in {"user", "assistant", "model"}:
            role = "user"
        converted_content = _chat_content_to_responses(content)
        if converted_content:
            input_items.append({"type": "message", "role": "assistant" if role in {"assistant", "model"} else "user", "content": converted_content})
        # Chat historical tool_calls do not carry Gemini thoughtSignature, so they
        # cannot be safely replayed as Gemini functionCall parts. The Responses
        # converter intentionally drops missing-signature calls too.
    if not input_items:
        raise BadRequest("messages must contain at least one user/assistant message")
    out: dict[str, Any] = {"model": model, "input": input_items}
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    if req.get("max_tokens") is not None:
        out["max_output_tokens"] = req.get("max_tokens")
    elif req.get("max_completion_tokens") is not None:
        out["max_output_tokens"] = req.get("max_completion_tokens")
    if req.get("temperature") is not None:
        out["temperature"] = req.get("temperature")
    if req.get("top_p") is not None:
        out["top_p"] = req.get("top_p")
    if req.get("stop") is not None:
        out["stop"] = req.get("stop")
    if req.get("tools") is not None:
        out["tools"] = req.get("tools")
    if req.get("tool_choice") is not None:
        out["tool_choice"] = req.get("tool_choice")
    return out


def _chat_chunk(model: str, cid: str, delta: dict[str, Any], finish_reason: str | None = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _chat_done() -> str:
    return "data: [DONE]\n\n"


async def _handle_gemini_chat(request: Request, settings: Settings, req: dict[str, Any], original_model: str, route_model: str, upstream_headers: dict[str, str]):
    try:
        responses_req = _chat_to_responses_req(req, route_model)
        gemini_body = responses_to_gemini(responses_req)
    except UnsupportedFeature as exc:
        status, body = openai_error(str(exc), code="unsupported_feature")
        return JSONResponse(status_code=status, content=body)
    except BadRequest as exc:
        status, body = openai_error(str(exc))
        return JSONResponse(status_code=status, content=body)
    except Exception as exc:
        status, body = openai_error(f"Failed to convert request: {exc}")
        return JSONResponse(status_code=status, content=body)

    headers = {**upstream_headers, "Content-Type": "application/json"}
    if bool(req.get("stream")):
        return StreamingResponse(
            _stream_gemini_chat(settings, original_model, route_model, gemini_body, headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp = None
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        for upstream_model in gemini_candidate_upstream_models(route_model):
            try:
                url, send_body, send_headers, direct = await _gemini_request_params(settings, route_model, upstream_model, "generateContent", gemini_body, headers)
            except Exception as exc:
                status, body = openai_error(f"Upstream auth setup failed: {exc}", status_code=502, err_type="upstream_error", code="upstream_error")
                return JSONResponse(status_code=status, content=body)
            _log_gemini_upstream(route_model, upstream_model, "generateContent", gemini_body, direct=direct)
            try:
                resp = await client.post(url, json=send_body, headers=send_headers)
            except Exception as exc:
                status, body = openai_error(f"Upstream request failed: {exc}", status_code=502, err_type="upstream_error", code="upstream_error")
                return JSONResponse(status_code=status, content=body)
            if resp.status_code < 400:
                break
            if not _is_retryable_model_error(resp.status_code, resp.text):
                break
        if resp is None:
            status, body = openai_error("Upstream request failed", status_code=502, err_type="upstream_error", code="upstream_error")
            return JSONResponse(status_code=status, content=body)
    if resp.status_code >= 400:
        status, body = upstream_error_from_body(resp.status_code, resp.text)
        return JSONResponse(status_code=status, content=body)
    try:
        payload = _payload_object(resp.json())
    except Exception:
        status, body = openai_error("Upstream returned non-JSON response", status_code=502, err_type="upstream_error", code="upstream_error")
        return JSONResponse(status_code=status, content=body)
    text = extract_text(payload)
    reasoning_text = extract_thought_text(payload)
    calls = extract_function_calls(payload)
    usage = usage_from_gemini(payload)
    if not text and not calls:
        text, calls, stream_usage, stream_reasoning = await _collect_gemini_stream_once(settings, route_model, gemini_body, headers)
        if stream_usage:
            usage = stream_usage
        if stream_reasoning and not reasoning_text:
            reasoning_text = stream_reasoning
    if not text and not calls:
        text = EMPTY_GEMINI_FALLBACK_TEXT
    return JSONResponse(status_code=200, content=_chat_completion_json(original_model, text, usage, calls, reasoning_text))


async def _stream_gemini_chat(settings: Settings, original_model: str, route_model: str, gemini_body: dict[str, Any], headers: dict[str, str]):
    upstream_model = resolve_upstream_model(route_model)
    cid = "chatcmpl_" + secrets.token_hex(12)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    yielded_role = False
    try:
        url, send_body, send_headers, direct = await _gemini_request_params(settings, route_model, upstream_model, "streamGenerateContent", gemini_body, headers, stream=True)
    except Exception as exc:
        yield _chat_chunk(original_model, cid, {"role": "assistant"})
        yield _chat_chunk(original_model, cid, {"content": f"Upstream auth setup failed: {exc}"})
        yield _chat_chunk(original_model, cid, {}, "stop")
        yield _chat_done()
        return
    _log_gemini_upstream(route_model, upstream_model, "streamGenerateContent", gemini_body, stream=True, direct=direct)
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            async with client.stream("POST", url, json=send_body, headers=send_headers) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    _, err_body = upstream_error_from_body(resp.status_code, body)
                    message = str(err_body.get("error", {}).get("message") or body[:500] or "Upstream request failed")
                    yield _chat_chunk(original_model, cid, {"role": "assistant"})
                    yield _chat_chunk(original_model, cid, {"content": message})
                    yield _chat_chunk(original_model, cid, {}, "stop")
                    yield _chat_done()
                    return
                yield _chat_chunk(original_model, cid, {"role": "assistant"})
                yielded_role = True
                buffer = ""
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buffer += chunk.replace("\r\n", "\n")
                    events, buffer = parse_sse_events(buffer)
                    for data in events:
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = _payload_object(json.loads(data))
                        except Exception:
                            continue
                        reasoning_delta = extract_thought_text(payload)
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
                            yield _chat_chunk(original_model, cid, {"reasoning_content": reasoning_delta})
                        delta = extract_text(payload)
                        if not delta:
                            continue
                        text_parts.append(delta)
                        yield _chat_chunk(original_model, cid, {"content": delta})
        except Exception as exc:
            if not yielded_role:
                yield _chat_chunk(original_model, cid, {"role": "assistant"})
            yield _chat_chunk(original_model, cid, {"content": f"Upstream stream failed: {exc}"})
            yield _chat_chunk(original_model, cid, {}, "stop")
            yield _chat_done()
            return
    if not text_parts:
        yield _chat_chunk(original_model, cid, {"content": EMPTY_GEMINI_FALLBACK_TEXT})
    yield _chat_chunk(original_model, cid, {}, "stop")
    yield _chat_done()


async def handle_chat_completions(request: Request, settings: Settings):
    try:
        req = await request.json()
    except Exception:
        status, body = openai_error("Request body must be valid JSON")
        return JSONResponse(status_code=status, content=body)
    if not isinstance(req, dict):
        status, body = openai_error("Request body must be a JSON object")
        return JSONResponse(status_code=status, content=body)

    model = str(req.get("model") or "")
    if not model:
        status, body = openai_error("model is required")
        return JSONResponse(status_code=status, content=body)
    route = build_model_route(model)

    should_bridge, context = await _should_bridge_request(request, settings)
    final_bridge = should_bridge and route.route_model in GEMINI_RESPONSES_MODELS
    _log_route_decision(
        "/v1/chat/completions",
        model,
        route.route_model,
        final_bridge,
        context,
        reason=None if final_bridge else ("chat_model_not_bridge_supported" if should_bridge else None),
    )
    if final_bridge:
        if context.get("bridge_key_used"):
            auth_error = validate_bridge_auth(request, settings)
            if auth_error is not None:
                return auth_error
        upstream_headers = upstream_auth_headers(request, settings)
        if isinstance(upstream_headers, JSONResponse):
            return upstream_headers
        return await _handle_gemini_chat(request, settings, req, route.original_model, route.route_model, upstream_headers)

    return await _passthrough_openai_chat(request, settings, req)


def _text_response_sse(model: str, text: str) -> str:
    rid = response_id()
    iid = item_id()
    chunks: list[str] = []
    for event_type, payload in stream_start_events(model, rid, iid):
        chunks.append(sse_event(event_type, payload))
    event_type, payload = stream_delta_event(3, iid, text)
    chunks.append(sse_event(event_type, payload))
    for event_type, payload in stream_done_events(model, rid, iid, text, None, 4):
        chunks.append(sse_event(event_type, payload))
    return "".join(chunks)


def _failed_sse(model: str, message: str, status: str = "failed") -> str:
    rid = response_id()
    payload = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": rid,
            "object": "response",
            "status": status,
            "model": model,
            "output": [],
            "error": {"code": "upstream_error", "message": message},
        },
    }
    return sse_event("response.failed", payload)


async def handle_responses(request: Request, settings: Settings):
    try:
        req = await request.json()
    except Exception:
        status, body = openai_error("Request body must be valid JSON")
        return JSONResponse(status_code=status, content=body)

    if not isinstance(req, dict):
        status, body = openai_error("Request body must be a JSON object")
        return JSONResponse(status_code=status, content=body)

    model = str(req.get("model") or "")
    if not model:
        status, body = openai_error("model is required")
        return JSONResponse(status_code=status, content=body)
    route = build_model_route(model)

    should_bridge, context = await _should_bridge_request(request, settings)
    _log_route_decision("/v1/responses", model, route.route_model, should_bridge, context)
    if not should_bridge:
        # Non-Antigravity groups, e.g. OpenAI, should be handled by Sub2API's
        # native gateway and its own API-key/auth/account routing.
        return await _passthrough_openai_responses(request, settings, req)

    if route.route_model not in RESPONSES_MODELS:
        status, body = openai_error(
            f"model {model} is not available on the Antigravity Responses bridge",
            code="unsupported_model_protocol",
        )
        return JSONResponse(status_code=status, content=body)

    if context.get("bridge_key_used"):
        auth_error = validate_bridge_auth(request, settings)
        if auth_error is not None:
            return auth_error

    upstream_headers = upstream_auth_headers(request, settings)
    if isinstance(upstream_headers, JSONResponse):
        return upstream_headers
    upstream_headers = {**upstream_headers, "Content-Type": "application/json"}

    if route.route_model in CLAUDE_MODELS:
        claude_req = dict(req)
        claude_req["model"] = route.route_model
        return await _handle_claude_responses(request, settings, claude_req, route.original_model, upstream_headers)

    try:
        gemini_req = dict(req)
        gemini_req["model"] = route.route_model
        gemini_body = responses_to_gemini(gemini_req)
    except UnsupportedFeature as exc:
        status, body = openai_error(str(exc), code="unsupported_feature")
        return JSONResponse(status_code=status, content=body)
    except BadRequest as exc:
        status, body = openai_error(str(exc))
        return JSONResponse(status_code=status, content=body)
    except Exception as exc:
        status, body = openai_error(f"Failed to convert request: {exc}")
        return JSONResponse(status_code=status, content=body)

    if bool(req.get("stream")):
        return StreamingResponse(
            _stream_responses(settings, route.original_model, route.route_model, gemini_body, upstream_headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp = None
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        for upstream_model in gemini_candidate_upstream_models(route.route_model):
            try:
                url, send_body, send_headers, direct = await _gemini_request_params(settings, route.route_model, upstream_model, "generateContent", gemini_body, upstream_headers)
            except Exception as exc:
                status, body = openai_error(
                    f"Upstream auth setup failed: {exc}",
                    status_code=502,
                    err_type="upstream_error",
                    code="upstream_error",
                )
                return JSONResponse(status_code=status, content=body)
            _log_gemini_upstream(route.route_model, upstream_model, "generateContent", gemini_body, direct=direct)
            try:
                resp = await client.post(url, json=send_body, headers=send_headers)
            except Exception as exc:
                status, body = openai_error(
                    f"Upstream request failed: {exc}",
                    status_code=502,
                    err_type="upstream_error",
                    code="upstream_error",
                )
                return JSONResponse(status_code=status, content=body)
            if resp.status_code < 400:
                break
            if not _is_retryable_model_error(resp.status_code, resp.text):
                break
        if resp is None:
            status, body = openai_error("Upstream request failed", status_code=502, err_type="upstream_error", code="upstream_error")
            return JSONResponse(status_code=status, content=body)

    if resp.status_code >= 400:
        status, body = upstream_error_from_body(resp.status_code, resp.text)
        return JSONResponse(status_code=status, content=body)

    try:
        payload = _payload_object(resp.json())
    except Exception:
        status, body = openai_error("Upstream returned non-JSON response", status_code=502, err_type="upstream_error", code="upstream_error")
        return JSONResponse(status_code=status, content=body)

    text = extract_text(payload)
    reasoning_text = extract_thought_text(payload)
    calls = extract_function_calls(payload)
    usage = usage_from_gemini(payload)
    if not text and not calls:
        # Some Antigravity Gemini non-stream calls return only hidden thought parts.
        # Collect one stream response as a fallback so non-stream clients still get output.
        text, calls, stream_usage, stream_reasoning = await _collect_gemini_stream_once(settings, route.route_model, gemini_body, upstream_headers)
        if stream_usage:
            usage = stream_usage
        if stream_reasoning and not reasoning_text:
            reasoning_text = stream_reasoning
    if not text and not calls:
        return JSONResponse(status_code=200, content=responses_json(route.original_model, EMPTY_GEMINI_FALLBACK_TEXT, usage, reasoning_text=reasoning_text))
    return JSONResponse(status_code=200, content=responses_json(route.original_model, text, usage, function_calls=calls, reasoning_text=reasoning_text))



async def _collect_gemini_stream_once(settings: Settings, route_model: str, gemini_body: dict[str, Any], upstream_headers: dict[str, str]) -> tuple[str, list[dict[str, str]], dict[str, int] | None, str]:
    upstream_model = resolve_upstream_model(route_model)
    try:
        url, send_body, send_headers, direct = await _gemini_request_params(settings, route_model, upstream_model, "streamGenerateContent", gemini_body, upstream_headers, stream=True)
    except Exception:
        return "", [], None, ""
    _log_gemini_upstream(route_model, upstream_model, "streamGenerateContent", gemini_body, stream=True, direct=direct)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    function_calls: list[dict[str, str]] = []
    seen_calls: set[tuple[str, str]] = set()
    last_usage: dict[str, int] | None = None
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            async with client.stream("POST", url, json=send_body, headers=send_headers) as resp:
                if resp.status_code >= 400:
                    return "", [], None, ""
                buffer = ""
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buffer += chunk.replace("\r\n", "\n")
                    events, buffer = parse_sse_events(buffer)
                    for data in events:
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = _payload_object(json.loads(data))
                        except Exception:
                            continue
                        usage = usage_from_gemini(payload)
                        if usage:
                            last_usage = usage
                        for call in extract_function_calls(payload):
                            key = (call.get("name") or "", call.get("arguments") or "")
                            if key not in seen_calls:
                                seen_calls.add(key)
                                function_calls.append(call)
                        reasoning_delta = extract_thought_text(payload)
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
                        delta = extract_text(payload)
                        if delta:
                            text_parts.append(delta)
        except Exception:
            return "", [], None, ""
    return "".join(text_parts), function_calls, last_usage, "".join(reasoning_parts)


async def _stream_responses(settings: Settings, original_model: str, route_model: str, gemini_body: dict[str, Any], upstream_headers: dict[str, str]):
    upstream_model = resolve_upstream_model(route_model)
    try:
        url, send_body, send_headers, direct = await _gemini_request_params(settings, route_model, upstream_model, "streamGenerateContent", gemini_body, upstream_headers, stream=True)
    except Exception as exc:
        yield _failed_sse(original_model, f"Upstream auth setup failed: {exc}")
        return
    _log_gemini_upstream(route_model, upstream_model, "streamGenerateContent", gemini_body, stream=True, direct=direct)
    rid = response_id()
    iid = item_id()
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    seq = 0
    last_usage: dict[str, int] | None = None
    function_calls: list[dict[str, str]] = []
    seen_calls: set[tuple[str, str]] = set()
    message_opened = False

    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            async with client.stream("POST", url, json=send_body, headers=send_headers) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    _, err_body = upstream_error_from_body(resp.status_code, body)
                    message = err_body.get("error", {}).get("message", body[:500])
                    yield _failed_sse(original_model, str(message))
                    return

                created = __import__("time").time()
                yield sse_event("response.created", {"type": "response.created", "sequence_number": seq, "response": {"id": rid, "object": "response", "created_at": int(created), "status": "in_progress", "model": original_model, "output": []}})
                seq += 1

                buffer = ""
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buffer += chunk.replace("\r\n", "\n")
                    events, buffer = parse_sse_events(buffer)
                    for data in events:
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = _payload_object(json.loads(data))
                        except Exception:
                            continue
                        usage = usage_from_gemini(payload)
                        if usage:
                            last_usage = usage
                        for call in extract_function_calls(payload):
                            key = (call.get("name") or "", call.get("arguments") or "")
                            if key not in seen_calls:
                                seen_calls.add(key)
                                function_calls.append(call)
                        reasoning_delta = extract_thought_text(payload)
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
                        delta = extract_text(payload)
                        if not delta:
                            continue
                        if not message_opened:
                            message_opened = True
                            yield sse_event("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": 0, "item": {"type": "message", "id": iid, "role": "assistant", "status": "in_progress", "content": []}})
                            seq += 1
                            yield sse_event("response.content_part.added", {"type": "response.content_part.added", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})
                            seq += 1
                        text_parts.append(delta)
                        event_type, out = stream_delta_event(seq, iid, delta)
                        seq += 1
                        yield sse_event(event_type, out)

                final_text = "".join(text_parts)
                final_reasoning = "".join(reasoning_parts)
                outputs: list[dict[str, Any]] = []
                if final_reasoning:
                    reasoning_item = {"type": "reasoning", "id": item_id(), "status": "completed", "summary": [{"type": "summary_text", "text": final_reasoning}]}
                    reasoning_output_index = len(outputs)
                    yield sse_event("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": reasoning_output_index, "item": dict(reasoning_item, status="in_progress", summary=[])})
                    seq += 1
                    yield sse_event("response.reasoning_summary_part.added", {"type": "response.reasoning_summary_part.added", "sequence_number": seq, "item_id": reasoning_item["id"], "output_index": reasoning_output_index, "summary_index": 0, "part": {"type": "summary_text", "text": ""}})
                    seq += 1
                    yield sse_event("response.reasoning_summary_text.delta", {"type": "response.reasoning_summary_text.delta", "sequence_number": seq, "item_id": reasoning_item["id"], "output_index": reasoning_output_index, "summary_index": 0, "delta": final_reasoning})
                    seq += 1
                    yield sse_event("response.reasoning_summary_text.done", {"type": "response.reasoning_summary_text.done", "sequence_number": seq, "item_id": reasoning_item["id"], "output_index": reasoning_output_index, "summary_index": 0, "text": final_reasoning})
                    seq += 1
                    yield sse_event("response.reasoning_summary_part.done", {"type": "response.reasoning_summary_part.done", "sequence_number": seq, "item_id": reasoning_item["id"], "output_index": reasoning_output_index, "summary_index": 0, "part": {"type": "summary_text", "text": final_reasoning}})
                    seq += 1
                    yield sse_event("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": reasoning_output_index, "item": reasoning_item})
                    seq += 1
                    outputs.append(reasoning_item)
                if message_opened:
                    message_output_index = len(outputs)
                    output_item = {"type": "message", "id": iid, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": final_text, "annotations": []}]}
                    yield sse_event("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "item_id": iid, "output_index": message_output_index, "content_index": 0, "text": final_text})
                    seq += 1
                    yield sse_event("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq, "item_id": iid, "output_index": message_output_index, "content_index": 0, "part": {"type": "output_text", "text": final_text, "annotations": [], "logprobs": []}})
                    seq += 1
                    yield sse_event("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": message_output_index, "item": output_item})
                    seq += 1
                    outputs.append(output_item)

                for call in function_calls:
                    call_item: dict[str, Any] = {"type": "function_call", "id": call.get("id") or item_id(), "call_id": call.get("call_id") or "call_unknown", "name": call.get("name") or "tool", "arguments": call.get("arguments") or "{}", "status": "completed"}
                    if call.get("thought_signature"):
                        call_item["thought_signature"] = call["thought_signature"]
                    output_index = len(outputs)
                    added_item = dict(call_item)
                    added_item["arguments"] = ""
                    added_item["status"] = "in_progress"
                    yield sse_event("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": output_index, "item": added_item})
                    seq += 1
                    if call_item["arguments"]:
                        yield sse_event("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "sequence_number": seq, "item_id": call_item["id"], "output_index": output_index, "delta": call_item["arguments"]})
                        seq += 1
                    yield sse_event("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "sequence_number": seq, "item_id": call_item["id"], "output_index": output_index, "arguments": call_item["arguments"]})
                    seq += 1
                    yield sse_event("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": output_index, "item": call_item})
                    seq += 1
                    outputs.append(call_item)

                if not outputs:
                    # Some Antigravity Gemini streams occasionally finish with only hidden thought parts.
                    # Retry once using non-stream generateContent so NF does not receive an empty response.
                    try:
                        retry_upstream_model = resolve_upstream_model(route_model)
                        retry_url, retry_body, retry_headers, retry_direct = await _gemini_request_params(settings, route_model, retry_upstream_model, "generateContent", gemini_body, upstream_headers)
                        _log_gemini_upstream(route_model, retry_upstream_model, "generateContent", gemini_body, direct=retry_direct)
                        retry_resp = await client.post(retry_url, json=retry_body, headers=retry_headers)
                        if retry_resp.status_code < 400:
                            retry_payload = _payload_object(retry_resp.json())
                            retry_text = extract_text(retry_payload)
                            retry_reasoning = extract_thought_text(retry_payload)
                            if retry_reasoning and not final_reasoning:
                                final_reasoning = retry_reasoning
                            retry_calls = extract_function_calls(retry_payload)
                            retry_usage = usage_from_gemini(retry_payload)
                            if retry_usage:
                                last_usage = retry_usage
                            if retry_text:
                                if retry_reasoning:
                                    reasoning_item = {"type": "reasoning", "id": item_id(), "status": "completed", "summary": [{"type": "summary_text", "text": retry_reasoning}]}
                                    outputs.append(reasoning_item)
                                output_item = {"type": "message", "id": iid, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": retry_text, "annotations": []}]}
                                yield sse_event("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": 0, "item": {"type": "message", "id": iid, "role": "assistant", "status": "in_progress", "content": []}})
                                seq += 1
                                yield sse_event("response.content_part.added", {"type": "response.content_part.added", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})
                                seq += 1
                                yield sse_event("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "delta": retry_text})
                                seq += 1
                                yield sse_event("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "text": retry_text})
                                seq += 1
                                yield sse_event("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": retry_text, "annotations": [], "logprobs": []}})
                                seq += 1
                                yield sse_event("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": 0, "item": output_item})
                                seq += 1
                                outputs.append(output_item)
                            for call in retry_calls:
                                key = (call.get("name") or "", call.get("arguments") or "")
                                if key in seen_calls:
                                    continue
                                seen_calls.add(key)
                                call_item: dict[str, Any] = {"type": "function_call", "id": call.get("id") or item_id(), "call_id": call.get("call_id") or "call_unknown", "name": call.get("name") or "tool", "arguments": call.get("arguments") or "{}", "status": "completed"}
                                if call.get("thought_signature"):
                                    call_item["thought_signature"] = call["thought_signature"]
                                output_index = len(outputs)
                                added_item = dict(call_item)
                                added_item["arguments"] = ""
                                added_item["status"] = "in_progress"
                                yield sse_event("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": output_index, "item": added_item})
                                seq += 1
                                if call_item["arguments"]:
                                    yield sse_event("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "sequence_number": seq, "item_id": call_item["id"], "output_index": output_index, "delta": call_item["arguments"]})
                                    seq += 1
                                yield sse_event("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "sequence_number": seq, "item_id": call_item["id"], "output_index": output_index, "arguments": call_item["arguments"]})
                                seq += 1
                                yield sse_event("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": output_index, "item": call_item})
                                seq += 1
                                outputs.append(call_item)
                    except Exception:
                        pass
                if not outputs:
                    fallback_text = EMPTY_GEMINI_FALLBACK_TEXT
                    output_item = {"type": "message", "id": iid, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": fallback_text, "annotations": []}]}
                    yield sse_event("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": 0, "item": {"type": "message", "id": iid, "role": "assistant", "status": "in_progress", "content": []}})
                    seq += 1
                    yield sse_event("response.content_part.added", {"type": "response.content_part.added", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})
                    seq += 1
                    yield sse_event("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "delta": fallback_text})
                    seq += 1
                    yield sse_event("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "text": fallback_text})
                    seq += 1
                    yield sse_event("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": fallback_text, "annotations": [], "logprobs": []}})
                    seq += 1
                    yield sse_event("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": 0, "item": output_item})
                    seq += 1
                    outputs.append(output_item)

                response: dict[str, Any] = {"id": rid, "object": "response", "created_at": int(__import__("time").time()), "status": "completed", "model": original_model, "output": outputs}
                if last_usage:
                    response["usage"] = last_usage
                yield sse_event("response.completed", {"type": "response.completed", "sequence_number": seq, "response": response})
        except Exception as exc:
            yield _failed_sse(original_model, f"Upstream stream failed: {exc}")


async def _handle_claude_responses(request: Request, settings: Settings, req: dict[str, Any], model: str, upstream_headers: dict[str, str]):
    try:
        anthropic_body = responses_to_anthropic(req)
    except UnsupportedFeature as exc:
        status, body = openai_error(str(exc), code="unsupported_feature")
        return JSONResponse(status_code=status, content=body)
    except BadRequest as exc:
        status, body = openai_error(str(exc))
        return JSONResponse(status_code=status, content=body)
    except Exception as exc:
        status, body = openai_error(f"Failed to convert request: {exc}")
        return JSONResponse(status_code=status, content=body)

    headers = dict(upstream_headers)
    headers.setdefault("anthropic-version", request.headers.get("anthropic-version") or "2023-06-01")
    if request.headers.get("anthropic-beta"):
        headers["anthropic-beta"] = request.headers["anthropic-beta"]
    url = f"{settings.sub2api_base_url}/antigravity/v1/messages"

    if bool(req.get("stream")):
        return StreamingResponse(
            _stream_claude_responses(settings, url, anthropic_body, headers, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            resp = await client.post(url, json=anthropic_body, headers=headers)
        except Exception as exc:
            status, body = openai_error(
                f"Upstream request failed: {exc}",
                status_code=502,
                err_type="upstream_error",
                code="upstream_error",
            )
            return JSONResponse(status_code=status, content=body)
    if resp.status_code >= 400:
        _, body = upstream_error_from_body(resp.status_code, resp.text)
        message = body.get("error", {}).get("message") or resp.text[:500] or "Upstream request failed"
        return JSONResponse(status_code=200, content=responses_json(model, str(message), None))
    try:
        payload = resp.json()
    except Exception:
        return JSONResponse(status_code=200, content=responses_json(model, "Upstream returned non-JSON response", None))
    converted = anthropic_to_responses(payload, model)
    try:
        has_visible_output = False
        for item in converted.get("output") or []:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        has_visible_output = True
            elif item.get("type") == "function_call":
                has_visible_output = True
        if not has_visible_output:
            converted = responses_json(model, EMPTY_CLAUDE_FALLBACK_TEXT, converted.get("usage"))
    except Exception:
        converted = responses_json(model, EMPTY_CLAUDE_FALLBACK_TEXT, None)
    return JSONResponse(status_code=200, content=converted)


async def _stream_claude_responses(settings: Settings, url: str, anthropic_body: dict[str, Any], headers: dict[str, str], model: str):
    state = anthropic_stream_state(model)
    # Emit an early Responses stream envelope before contacting/receiving upstream.
    # Sub2API may spend tens of seconds in Antigravity smart-retry before the first
    # upstream SSE chunk; without early bytes NF can treat the provider as empty.
    for out in anthropic_stream_open_events(state):
        yield out
    opened = True
    closed = False
    saw_content = False
    async with httpx.AsyncClient(timeout=_timeout(settings)) as client:
        try:
            async with client.stream("POST", url, json=anthropic_body, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    _, err_body = upstream_error_from_body(resp.status_code, body)
                    message = err_body.get("error", {}).get("message", body[:500])
                    yield sse_event("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": state["seq"], "item_id": state["iid"], "output_index": 0, "content_index": 0, "delta": str(message)})
                    state["seq"] += 1
                    state.setdefault("text", []).append(str(message))
                    for out in anthropic_stream_close_events(state):
                        yield out
                    return
                buffer = ""
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buffer += chunk.replace("\r\n", "\n")
                    events, buffer = parse_anthropic_sse(buffer)
                    for event in events:
                        typ = event.get("type")
                        if typ == "message_start":
                            # We already emitted the open envelope. Keep the upstream
                            # id if present, but do not emit duplicate open events.
                            msg = event.get("message") or {}
                            if isinstance(msg, dict) and msg.get("id"):
                                state["rid"] = msg["id"]
                            continue
                        if typ == "message_stop":
                            closed = True
                        before_text = len(state.get("text") or [])
                        for out in stream_event_to_responses(event, state):
                            yield out
                        after_text = len(state.get("text") or [])
                        if after_text > before_text or typ in {"content_block_delta", "content_block_start"}:
                            saw_content = True
                if not closed:
                    if not saw_content and not state.get("text"):
                        yield sse_event("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": state["seq"], "item_id": state["iid"], "output_index": 0, "content_index": 0, "delta": EMPTY_CLAUDE_FALLBACK_TEXT})
                        state["seq"] += 1
                        state.setdefault("text", []).append(EMPTY_CLAUDE_FALLBACK_TEXT)
                    for out in anthropic_stream_close_events(state):
                        yield out
        except Exception as exc:
            # The stream was already opened, so close it with visible text instead
            # of returning a second response.failed envelope with no output item.
            message = f"Upstream stream failed: {exc}"
            yield sse_event("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": state["seq"], "item_id": state["iid"], "output_index": 0, "content_index": 0, "delta": message})
            state["seq"] += 1
            state.setdefault("text", []).append(message)
            for out in anthropic_stream_close_events(state):
                yield out
