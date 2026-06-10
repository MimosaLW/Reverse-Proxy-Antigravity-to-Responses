from __future__ import annotations

import json
from typing import Any

import asyncpg
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .models import IDE_MODELS, MODEL_DISPLAY_NAMES
from .sse import sse_event


_TEXT_DEFAULT_PROMPT = "."


def _admin_headers(request: Request) -> dict[str, str] | JSONResponse:
    auth = request.headers.get("authorization")
    if not auth:
        return JSONResponse(
            status_code=401,
            content={"code": 401, "message": "admin Authorization header is required"},
        )
    return {"Authorization": auth}


async def _validate_admin(request: Request, settings: Settings, account_id: int | None = None) -> JSONResponse | None:
    headers = _admin_headers(request)
    if isinstance(headers, JSONResponse):
        return headers

    if account_id is not None:
        url = f"{settings.sub2api_base_url}/api/v1/admin/accounts/{account_id}/models"
    else:
        url = f"{settings.sub2api_base_url}/api/v1/admin/accounts/antigravity/default-model-mapping"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"code": 502, "message": f"failed to validate admin session: {exc}"},
        )

    if resp.status_code >= 400:
        try:
            content = resp.json()
        except Exception:
            content = {"code": resp.status_code, "message": resp.text[:500] or "admin validation failed"}
        return JSONResponse(status_code=resp.status_code, content=content)
    return None


async def _db(settings: Settings):
    return await asyncpg.connect(
        host=settings.database_host,
        port=settings.database_port,
        user=settings.database_user,
        password=settings.database_password,
        database=settings.database_dbname,
    )


def _ordered_unique(models: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for model in models:
        model = str(model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        out.append(model)
    return out


def _sort_fallback_models(models: list[str]) -> list[str]:
    order = {model: idx for idx, model in enumerate(IDE_MODELS)}
    return sorted(_ordered_unique(models), key=lambda m: (order.get(m, len(order)), m))


def _model_item(model: str) -> dict[str, Any]:
    return {
        "id": model,
        "object": "model",
        "type": "model",
        "display_name": MODEL_DISPLAY_NAMES.get(model, model),
        "owned_by": "antigravity",
        "created_at": "",
    }


async def _configured_models_for_account(settings: Settings, account_id: int) -> list[str]:
    conn = await _db(settings)
    try:
        rows = await conn.fetch(
            """
            select g.models_list_config as models_list_config,
                   a.credentials->'model_mapping' as account_mapping
            from accounts a
            left join account_groups ag on ag.account_id = a.id
            left join groups g on g.id = ag.group_id and g.deleted_at is null
            where a.id = $1
              and a.platform = 'antigravity'
              and a.deleted_at is null
            order by ag.priority asc nulls last, g.id asc nulls last
            """,
            account_id,
        )
        for row in rows:
            cfg = row["models_list_config"] or {}
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {}
            if isinstance(cfg, dict) and cfg.get("enabled") is True and isinstance(cfg.get("models"), list):
                models = _ordered_unique([str(x) for x in cfg.get("models") or []])
                if models:
                    return models

        for row in rows:
            mapping = row["account_mapping"] or {}
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except Exception:
                    mapping = {}
            if isinstance(mapping, dict) and mapping:
                return _sort_fallback_models([str(k) for k in mapping.keys()])
    finally:
        await conn.close()
    return list(IDE_MODELS)


async def _model_mapping_for_account(settings: Settings, account_id: int = 1) -> dict[str, str]:
    conn = await _db(settings)
    try:
        row = await conn.fetchrow(
            """
            select credentials->'model_mapping' as account_mapping
            from accounts
            where id = $1
              and platform = 'antigravity'
              and deleted_at is null
            """,
            account_id,
        )
        mapping = row["account_mapping"] if row else None
        if isinstance(mapping, str):
            try:
                mapping = json.loads(mapping)
            except Exception:
                mapping = {}
        if isinstance(mapping, dict) and mapping:
            return {str(k): str(v) for k, v in mapping.items() if isinstance(v, str)}

        models = await _configured_models_for_account(settings, account_id)
        return {model: model for model in models}
    finally:
        await conn.close()


async def _antigravity_api_key_for_account(settings: Settings, account_id: int) -> str | None:
    conn = await _db(settings)
    try:
        row = await conn.fetchrow(
            """
            select k.key
            from api_keys k
            join account_groups ag on ag.group_id = k.group_id
            where ag.account_id = $1
              and k.deleted_at is null
              and k.status = 'active'
            order by k.id asc
            limit 1
            """,
            account_id,
        )
        if row and row["key"]:
            return str(row["key"])
    finally:
        await conn.close()
    return None


async def admin_account_models(request: Request, settings: Settings, account_id: int) -> JSONResponse:
    auth_error = await _validate_admin(request, settings, account_id)
    if auth_error is not None:
        return auth_error
    models = await _configured_models_for_account(settings, account_id)
    return JSONResponse(
        status_code=200,
        content={"code": 0, "message": "success", "data": [_model_item(model) for model in models]},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "X-Sub2API-Test-Models-Override": "antigravity-dynamic"},
    )


async def admin_default_model_mapping(request: Request, settings: Settings) -> JSONResponse:
    auth_error = await _validate_admin(request, settings, 1)
    if auth_error is not None:
        return auth_error
    mapping = await _model_mapping_for_account(settings, 1)
    return JSONResponse(
        status_code=200,
        content={"code": 0, "message": "success", "data": mapping},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "X-Sub2API-Default-Mapping-Override": "antigravity-dynamic"},
    )


def _extract_responses_text(payload: dict[str, Any]) -> str:
    pieces: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
    if pieces:
        return "".join(pieces)
    direct = payload.get("output_text")
    return str(direct) if isinstance(direct, str) else ""


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or "test failed")
        if payload.get("message"):
            return str(payload["message"])
    return "test failed"


async def admin_account_test(request: Request, settings: Settings, account_id: int) -> StreamingResponse | JSONResponse:
    auth_error = await _validate_admin(request, settings, account_id)
    if auth_error is not None:
        return auth_error

    try:
        req = await request.json()
    except Exception:
        req = {}
    model_id = str(req.get("model_id") or "").strip()
    if not model_id:
        return JSONResponse(status_code=400, content={"code": 400, "message": "model_id is required"})
    prompt = str(req.get("prompt") or "").strip() or _TEXT_DEFAULT_PROMPT

    async def gen():
        yield sse_event("message", {"type": "test_start", "model": model_id})
        key = await _antigravity_api_key_for_account(settings, account_id)
        if not key:
            yield sse_event("message", {"type": "error", "error": "No active API key found for this account group"})
            yield sse_event("message", {"type": "test_complete", "success": False, "error": "No active API key found for this account group"})
            return

        body = {"model": model_id, "input": prompt, "stream": False, "max_output_tokens": 512}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                resp = await client.post(f"http://127.0.0.1:8090/v1/responses", json=body, headers=headers)
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"error": {"message": resp.text[:1000]}}
        except Exception as exc:
            message = f"Bridge test request failed: {exc}"
            yield sse_event("message", {"type": "error", "error": message})
            yield sse_event("message", {"type": "test_complete", "success": False, "error": message})
            return

        if resp.status_code >= 400:
            message = _error_message(payload)
            yield sse_event("message", {"type": "error", "error": message})
            yield sse_event("message", {"type": "test_complete", "success": False, "error": message})
            return

        text = _extract_responses_text(payload)
        if text:
            yield sse_event("message", {"type": "content", "text": text})
        yield sse_event("message", {"type": "test_complete", "success": True})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
