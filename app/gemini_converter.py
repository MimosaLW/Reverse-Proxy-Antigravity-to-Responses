from __future__ import annotations

import json
import secrets
import time
from typing import Any

from .models import get_max_output_tokens, get_thinking_budget, get_thinking_level, is_gemini35_flash_model, is_thinking_model, resolve_client_model


class UnsupportedFeature(ValueError):
    pass


class BadRequest(ValueError):
    pass


def response_id() -> str:
    return "resp_" + secrets.token_hex(12)


def item_id() -> str:
    return "item_" + secrets.token_hex(12)


def _as_text_from_content(content: Any, *, role: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
                continue
            if not isinstance(part, dict):
                raise UnsupportedFeature(f"unsupported content part for role {role}")
            ptype = part.get("type") or "text"
            if ptype in {"input_text", "output_text", "text"}:
                pieces.append(str(part.get("text", "")))
            elif ptype in {"refusal"}:
                pieces.append(str(part.get("refusal", "")))
            elif ptype == "input_image":
                # Images are handled by _content_to_gemini_parts for model/user messages.
                # Text-only extraction is used for system/developer messages, so skip images here.
                continue
            else:
                raise UnsupportedFeature(f"unsupported content part type: {ptype}")
        return "".join(pieces)
    raise UnsupportedFeature(f"unsupported content format for role {role}")


def _append_system(system_parts: list[str], text: str) -> None:
    text = (text or "").strip()
    if text:
        system_parts.append(text)


def _message_to_gemini(role: str, text: str) -> dict[str, Any] | None:
    if text == "":
        return None
    if role in {"assistant", "model"}:
        gemini_role = "model"
    else:
        gemini_role = "user"
    return {"role": gemini_role, "parts": [{"text": text}]}


def _image_url_value(part: dict[str, Any]) -> str | None:
    raw = part.get("image_url") or part.get("url")
    if isinstance(raw, dict):
        raw = raw.get("url") or raw.get("image_url")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _mime_from_url(url: str) -> str:
    lower = url.split("?", 1)[0].lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    return "image/jpeg"


def _data_url_to_mime_data(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if not sep:
        return None
    mime = header[5:].split(";", 1)[0] or "image/jpeg"
    return mime, data.strip()


def _image_part_to_gemini(part: dict[str, Any]) -> dict[str, Any] | None:
    url = _image_url_value(part)
    if not url:
        return None
    data = _data_url_to_mime_data(url)
    if data:
        mime, b64 = data
        return {"inlineData": {"mimeType": mime, "data": b64}}
    if url.startswith("http://") or url.startswith("https://"):
        return {"fileData": {"mimeType": _mime_from_url(url), "fileUri": url}}
    # Raw base64 without a data URL prefix. Prefer explicit mime_type if supplied.
    mime = str(part.get("mime_type") or part.get("media_type") or "image/jpeg")
    return {"inlineData": {"mimeType": mime, "data": url}}


def _content_to_gemini_parts(content: Any, *, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append({"text": part})
                continue
            if not isinstance(part, dict):
                raise UnsupportedFeature(f"unsupported content part for role {role}")
            ptype = part.get("type") or "text"
            if ptype in {"input_text", "output_text", "text"}:
                text = str(part.get("text", ""))
                if text:
                    parts.append({"text": text})
            elif ptype == "refusal":
                text = str(part.get("refusal", ""))
                if text:
                    parts.append({"text": text})
            elif ptype == "input_image":
                img = _image_part_to_gemini(part)
                if img:
                    parts.append(img)
            else:
                raise UnsupportedFeature(f"unsupported content part type: {ptype}")
        return parts
    raise UnsupportedFeature(f"unsupported content format for role {role}")


def _safe_tool_name(raw: Any, fallback: str = "tool") -> str:
    # Antigravity exposes IDE/MCP tools with names like "default_api:Read".
    # Standard Gemini docs are stricter, but this upstream accepts those names;
    # preserving them lets NF match function_call names back to its tools.
    name = str(raw or fallback).strip() or fallback
    return name[:128]


_NETWORKING_TOOL_NAMES = {
    "web_search",
    "google_search",
    "web_search_20250305",
    "google_search_retrieval",
    "builtin_web_search",
}

_SCHEMA_DROP_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "definitions",
    "format",
    "multipleOf",
    "strict",
}


def _tool_name(tool: dict[str, Any]) -> str | None:
    if isinstance(tool.get("function"), dict):
        name = tool.get("function", {}).get("name")
        if name:
            return str(name)
    name = tool.get("name")
    if name:
        return str(name)
    return None


def _is_networking_tool(tool: dict[str, Any]) -> bool:
    typ = str(tool.get("type") or "")
    if typ in _NETWORKING_TOOL_NAMES:
        return True
    name = _tool_name(tool)
    if name in _NETWORKING_TOOL_NAMES:
        return True
    if tool.get("googleSearch") is not None or tool.get("googleSearchRetrieval") is not None:
        return True
    decls = tool.get("functionDeclarations")
    if isinstance(decls, list):
        return any(isinstance(d, dict) and str(d.get("name") or "") in _NETWORKING_TOOL_NAMES for d in decls)
    return False


def _detects_networking_tool(tools: Any) -> bool:
    return isinstance(tools, list) and any(isinstance(t, dict) and _is_networking_tool(t) for t in tools)


def _clean_json_schema(value: Any, depth: int = 0) -> Any:
    if depth > 20:
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SCHEMA_DROP_KEYS:
                continue
            out[key] = _clean_json_schema(item, depth + 1)
        return out
    if isinstance(value, list):
        return [_clean_json_schema(item, depth + 1) for item in value]
    return value


def _default_parameters_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}}


_THOUGHT_SIGNATURES_BY_CALL_ID: dict[str, str] = {}
_THOUGHT_SIGNATURES_BY_KEY: dict[tuple[str, str], str] = {}
_MAX_THOUGHT_SIGNATURES = 4096


def _remember_thought_signature(call_id: str, name: str, arguments: str, signature: Any) -> None:
    if not isinstance(signature, str) or not signature:
        return
    if len(_THOUGHT_SIGNATURES_BY_CALL_ID) > _MAX_THOUGHT_SIGNATURES:
        _THOUGHT_SIGNATURES_BY_CALL_ID.clear()
        _THOUGHT_SIGNATURES_BY_KEY.clear()
    _THOUGHT_SIGNATURES_BY_CALL_ID[call_id] = signature
    _THOUGHT_SIGNATURES_BY_KEY[(name, arguments)] = signature


def _lookup_thought_signature(call_id: Any, name: str, arguments: str) -> str | None:
    if call_id is not None:
        sig = _THOUGHT_SIGNATURES_BY_CALL_ID.get(str(call_id))
        if sig:
            return sig
    return _THOUGHT_SIGNATURES_BY_KEY.get((name, arguments))


def _json_obj(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _function_response_payload(output: Any) -> dict[str, Any]:
    value = _json_obj(output)
    if isinstance(value, dict):
        return value
    return {"result": value}


def _openai_tools_to_gemini(tools: Any) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return declarations
    for tool in tools:
        if not isinstance(tool, dict) or _is_networking_tool(tool):
            continue
        typ = tool.get("type")
        if typ == "function":
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            name = tool.get("name") or fn.get("name")
            description = tool.get("description") or fn.get("description") or ""
            parameters = tool.get("parameters") or fn.get("parameters")
        elif tool.get("name"):
            name = tool.get("name")
            description = tool.get("description") or ""
            parameters = tool.get("parameters")
        else:
            continue
        if not name:
            continue
        clean_params = _clean_json_schema(parameters) if isinstance(parameters, dict) else _default_parameters_schema()
        if not isinstance(clean_params, dict):
            clean_params = _default_parameters_schema()
        clean_params.setdefault("type", "object")
        clean_params.setdefault("properties", {})
        declarations.append({
            "name": _safe_tool_name(name),
            "description": str(description or ""),
            "parameters": clean_params,
        })
    return declarations


def _tool_config(tool_choice: Any, declarations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        choice = tool_choice.lower()
        if choice in {"auto", "default"}:
            return {"mode": "AUTO"}
        if choice == "none":
            return {"mode": "NONE"}
        if choice in {"required", "any"}:
            return {"mode": "ANY"}
        return None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
        name = tool_choice.get("name") or fn.get("name")
        if name:
            return {"mode": "ANY", "allowedFunctionNames": [_safe_tool_name(name)]}
        typ = str(tool_choice.get("type") or "").lower()
        if typ == "function" and declarations:
            return {"mode": "ANY"}
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled", "auto"}
    return False


def _include_thoughts_override(req: dict[str, Any], thinking: dict[str, Any]) -> bool | None:
    for key in ("includeThoughts", "include_thoughts", "include_reasoning"):
        if key in thinking:
            return _truthy(thinking.get(key))
        if key in req:
            return _truthy(req.get(key))
    return None


def _request_wants_thoughts(req: dict[str, Any]) -> bool:
    if _truthy(req.get("include_reasoning")) or _truthy(req.get("include_thoughts")):
        return True
    reasoning = req.get("reasoning")
    if isinstance(reasoning, dict):
        summary = str(reasoning.get("summary") or "").strip().lower()
        if summary and summary not in {"none", "off", "disabled", "false"}:
            return True
        effort = str(reasoning.get("effort") or reasoning.get("reasoning_effort") or "").strip().lower()
        if effort in {"medium", "high", "xhigh"}:
            return True
        if _truthy(reasoning.get("enabled")):
            return True
    elif isinstance(reasoning, str) and reasoning.strip().lower() not in {"", "none", "off", "disabled", "false"}:
        return True
    effort = str(req.get("reasoning_effort") or "").strip().lower()
    return effort in {"medium", "high", "xhigh"}


def _build_generation_config(req: dict[str, Any], route_model: str) -> dict[str, Any]:
    gen: dict[str, Any] = {"topP": 1.0, "topK": 40}
    max_tokens = req.get("max_output_tokens") or req.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        gen["maxOutputTokens"] = max_tokens
    else:
        gen["maxOutputTokens"] = get_max_output_tokens(route_model)

    if isinstance(req.get("temperature"), (int, float)):
        gen["temperature"] = req["temperature"]
    if isinstance(req.get("top_p"), (int, float)):
        gen["topP"] = req["top_p"]

    model_lower = route_model.lower()
    thinking = req.get("thinking") if isinstance(req.get("thinking"), dict) else {}
    thinking_type = str(thinking.get("type") or "").lower() if isinstance(thinking, dict) else ""
    user_budget = (thinking.get("budget_tokens") or thinking.get("budgetTokens")) if isinstance(thinking, dict) else None
    include_override = _include_thoughts_override(req, thinking)
    explicit_enabled = thinking_type == "enabled" or include_override is True
    explicit_disabled = thinking_type == "disabled" or include_override is False
    wants_thoughts = _request_wants_thoughts(req)
    model_level = get_thinking_level(route_model)
    user_level = (thinking.get("level") or thinking.get("thinking_level") or thinking.get("thinkingLevel")) if isinstance(thinking, dict) else None
    thinking_level = str(user_level or model_level or "").lower()

    if is_gemini35_flash_model(route_model):
        # Official v1internal 3.5 tiers are selected by physical model ID.
        # Request thought parts only for explicit reasoning requests or High,
        # and keep them out of final visible text via extract_text().
        include = (not explicit_disabled) and (explicit_enabled or wants_thoughts or model_lower.endswith("-high"))
        config: dict[str, Any] = {"includeThoughts": bool(include)}
        if include and isinstance(user_budget, int) and user_budget >= 0:
            config["thinkingBudget"] = min(user_budget, get_thinking_budget(route_model))
        gen["thinkingConfig"] = config
    elif explicit_disabled:
        gen["thinkingConfig"] = {"includeThoughts": False}
    elif thinking_level in {"minimal", "low", "medium", "high"}:
        # Follow CLIProxyAPI's Antigravity applier: for level-mode models write
        # generationConfig.thinkingConfig.thinkingLevel instead of a fake model ID.
        config = {"thinkingLevel": thinking_level}
        if explicit_enabled or wants_thoughts:
            config["includeThoughts"] = True
        gen["thinkingConfig"] = config
    elif "flash" in model_lower and "thinking" not in model_lower and not explicit_enabled and not wants_thoughts:
        # Older non-level flash aliases still default to no hidden thinking to avoid empty visible output.
        gen.setdefault("thinkingConfig", {"thinkingBudget": 0})
    elif not explicit_disabled and (explicit_enabled or wants_thoughts or is_thinking_model(route_model)):
        cap = get_thinking_budget(route_model)
        budget = int(user_budget) if isinstance(user_budget, int) and user_budget >= 0 else cap
        budget = min(budget, cap)
        gen["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": budget}
        if budget >= 0 and int(gen.get("maxOutputTokens") or 0) <= budget:
            gen["maxOutputTokens"] = min(get_max_output_tokens(route_model), budget + 8192)

    stop = req.get("stop") or req.get("stop_sequences")
    if isinstance(stop, str):
        gen["stopSequences"] = [stop]
    elif isinstance(stop, list) and all(isinstance(x, str) for x in stop):
        gen["stopSequences"] = stop
    return gen

def _inject_google_search_if_needed(out: dict[str, Any], req: dict[str, Any], declarations: list[dict[str, Any]]) -> None:
    tools = req.get("tools")
    route_model = str(req.get("model") or "")
    wants_search = route_model.endswith("-online") or _detects_networking_tool(tools)
    if not wants_search:
        return
    # v1internal currently rejects mixing googleSearch with local functionDeclarations.
    if declarations:
        return
    out.setdefault("tools", [])
    if isinstance(out["tools"], list):
        out["tools"] = [t for t in out["tools"] if not (isinstance(t, dict) and ("googleSearch" in t or "googleSearchRetrieval" in t))]
        out["tools"].append({"googleSearch": {}})


def responses_to_gemini(req: dict[str, Any]) -> dict[str, Any]:
    system_parts: list[str] = []
    _append_system(system_parts, str(req.get("instructions") or ""))
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    textified_call_ids: set[str] = set()

    raw_input = req.get("input")
    if raw_input is None:
        raw_input = ""

    if isinstance(raw_input, str):
        msg = _message_to_gemini("user", raw_input)
        if msg:
            contents.append(msg)
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                raise UnsupportedFeature("responses input array items must be objects")
            itype = item.get("type") or ""
            if itype == "function_call":
                name = _safe_tool_name(item.get("name"))
                args_raw = item.get("arguments")
                args = _json_obj(args_raw)
                if not isinstance(args, dict):
                    args = {"value": args}
                arg_text = args_raw if isinstance(args_raw, str) else json.dumps(args or {}, ensure_ascii=False, separators=(",", ":"))
                call_id = item.get("call_id") or item.get("id")
                thought_signature = item.get("thoughtSignature") or item.get("thought_signature") or _lookup_thought_signature(call_id, name, arg_text)
                if thought_signature:
                    part: dict[str, Any] = {"functionCall": {"name": name, "args": args}, "thoughtSignature": thought_signature}
                    contents.append({"role": "model", "parts": [part]})
                    if call_id:
                        call_names[str(call_id)] = name
                else:
                    # Gemini rejects historical functionCall parts without a
                    # thoughtSignature. Do not serialize missing-signature tool
                    # calls as visible text either, because the model may echo it.
                    if call_id:
                        textified_call_ids.add(str(call_id))
                continue
            if itype == "function_call_output":
                call_id = str(item.get("call_id") or "")
                name = _safe_tool_name(item.get("name") or call_names.get(call_id) or "tool")
                if call_id in textified_call_ids:
                    # The matching function_call was dropped because it lacked a
                    # thoughtSignature, so its output cannot be represented as a
                    # valid Gemini functionResponse. Drop it too to avoid making
                    # stale tool output visible or inducing duplicate tool calls.
                    continue
                else:
                    contents.append({"role": "user", "parts": [{"functionResponse": {"name": name, "response": _function_response_payload(item.get("output"))}}]})
                continue
            if itype == "reasoning":
                # Codex / high reasoning effort can include OpenAI Responses
                # internal reasoning items in conversation history. They are not
                # user-visible content and Gemini cannot consume them directly.
                continue
            if itype in {"computer_call", "web_search_call", "custom_tool_call"}:
                contents.append({"role": "user", "parts": [{"text": json.dumps(item, ensure_ascii=False)}]})
                continue
            role = item.get("role") or ("user" if itype in {"message", "input_text"} else "")
            content = item.get("content")
            if itype == "input_text" and content is None:
                content = item.get("text", "")
            if role in {"system", "developer"}:
                text = _as_text_from_content(content, role=role or itype or "user")
                _append_system(system_parts, text)
            elif role in {"user", "assistant", "model"}:
                parts = _content_to_gemini_parts(content, role=role or itype or "user")
                if parts:
                    contents.append({"role": "model" if role in {"assistant", "model"} else "user", "parts": parts})
            elif content is not None:
                parts = _content_to_gemini_parts(content, role=role or itype or "user")
                if parts:
                    contents.append({"role": "user", "parts": parts})
            else:
                raise UnsupportedFeature(f"unsupported input item type: {itype or '<unknown>'}")
    else:
        raise UnsupportedFeature("responses input must be a string or an array")

    if not contents:
        raise BadRequest("input must contain at least one text message")

    out: dict[str, Any] = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    route_model = resolve_client_model(str(req.get("model") or ""))
    out["generationConfig"] = _build_generation_config(req, route_model)

    declarations = _openai_tools_to_gemini(req.get("tools"))
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    _inject_google_search_if_needed(out, req, declarations)
    config = _tool_config(req.get("tool_choice"), declarations)
    if config:
        out["toolConfig"] = {"functionCallingConfig": config}
    return out


def extract_text(payload: dict[str, Any], *, include_thought: bool = False) -> str:
    texts: list[str] = []
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("thought") is True and not include_thought:
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict):
                data = inline.get("data")
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                if isinstance(data, str) and data:
                    texts.append(f"![image](data:{mime};base64,{data})")
    return "".join(texts)



def extract_thought_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict) or part.get("thought") is not True:
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "".join(texts)

def extract_function_calls(payload: dict[str, Any]) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            fc = part.get("functionCall") or part.get("function_call")
            if not isinstance(fc, dict):
                continue
            name = _safe_tool_name(fc.get("name"))
            args = fc.get("args") or fc.get("arguments") or {}
            if isinstance(args, str):
                arg_text = args
            else:
                arg_text = json.dumps(args or {}, ensure_ascii=False, separators=(",", ":"))
            key = (name, arg_text)
            if key in seen:
                continue
            seen.add(key)
            call_id = "call_" + secrets.token_hex(8)
            signature = part.get("thoughtSignature") or part.get("thought_signature") or fc.get("thoughtSignature") or fc.get("thought_signature")
            _remember_thought_signature(call_id, name, arg_text, signature)
            call: dict[str, str] = {"name": name, "arguments": arg_text, "call_id": call_id, "id": item_id()}
            if isinstance(signature, str) and signature:
                # Non-standard extension retained internally/for debug; OpenAI clients ignore extra fields.
                call["thought_signature"] = signature
            calls.append(call)
    return calls


def usage_from_gemini(payload: dict[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens))
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def finish_reason(payload: dict[str, Any]) -> str | None:
    for cand in payload.get("candidates") or []:
        if isinstance(cand, dict) and cand.get("finishReason"):
            return str(cand["finishReason"])
    return None


def responses_json(model: str, text: str, usage: dict[str, int] | None = None, status: str = "completed", function_calls: list[dict[str, str]] | None = None, reasoning_text: str | None = None) -> dict[str, Any]:
    rid = response_id()
    outputs: list[dict[str, Any]] = []
    calls = function_calls or []
    if reasoning_text:
        outputs.append({
            "type": "reasoning",
            "id": item_id(),
            "status": "completed" if status == "completed" else status,
            "summary": [{"type": "summary_text", "text": reasoning_text}],
        })
    if text or not calls:
        outputs.append({"type": "message", "id": item_id(), "role": "assistant", "status": "completed" if status == "completed" else status, "content": [{"type": "output_text", "text": text, "annotations": []}]})
    for call in calls:
        item = {"type": "function_call", "id": call.get("id") or item_id(), "call_id": call.get("call_id") or ("call_" + secrets.token_hex(8)), "name": call.get("name") or "tool", "arguments": call.get("arguments") or "{}", "status": "completed"}
        if call.get("thought_signature"):
            item["thought_signature"] = call["thought_signature"]
        outputs.append(item)
    body: dict[str, Any] = {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model, "output": outputs}
    if text:
        body["output_text"] = text
    if usage:
        body["usage"] = usage
    return body

def openai_error(message: str, *, status_code: int = 400, err_type: str = "invalid_request_error", code: str | None = None) -> tuple[int, dict[str, Any]]:
    return status_code, {"error": {"type": err_type, "message": message, "code": code or err_type}}


def upstream_error_from_body(status_code: int, body: str) -> tuple[int, dict[str, Any]]:
    message = body.strip()[:1000] or f"upstream returned HTTP {status_code}"
    try:
        parsed = json.loads(body)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            message = str(err.get("message") or message)
            code = str(err.get("code") or err.get("status") or "upstream_error")
        else:
            code = "upstream_error"
    except Exception:
        code = "upstream_error"
    client_status = status_code if status_code < 500 else 502
    return client_status, {"error": {"type": "upstream_error", "message": message, "code": code}}


def stream_start_events(model: str, rid: str, iid: str) -> list[tuple[str, dict[str, Any]]]:
    created = int(time.time())
    return [
        ("response.created", {"type": "response.created", "sequence_number": 0, "response": {"id": rid, "object": "response", "created_at": created, "status": "in_progress", "model": model, "output": []}}),
        ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": {"type": "message", "id": iid, "role": "assistant", "status": "in_progress", "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "sequence_number": 2, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
    ]


def stream_delta_event(seq: int, iid: str, delta: str) -> tuple[str, dict[str, Any]]:
    return "response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "delta": delta}


def stream_done_events(model: str, rid: str, iid: str, text: str, usage: dict[str, int] | None, seq: int, function_calls: list[dict[str, str]] | None = None) -> list[tuple[str, dict[str, Any]]]:
    created = int(time.time())
    output_item = {"type": "message", "id": iid, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}
    outputs: list[dict[str, Any]] = [output_item]
    events: list[tuple[str, dict[str, Any]]] = [
        ("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "text": text}),
        ("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq + 1, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []}}),
        ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq + 2, "output_index": 0, "item": output_item}),
    ]
    seq += 3
    for call in function_calls or []:
        call_item = {"type": "function_call", "id": call.get("id") or item_id(), "call_id": call.get("call_id") or ("call_" + secrets.token_hex(8)), "name": call.get("name") or "tool", "arguments": call.get("arguments") or "{}", "status": "completed"}
        if call.get("thought_signature"):
            call_item["thought_signature"] = call["thought_signature"]
        output_index = len(outputs)
        outputs.append(call_item)
        added_item = dict(call_item)
        added_item["arguments"] = ""
        added_item["status"] = "in_progress"
        events.extend([
            ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": output_index, "item": added_item}),
            ("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "sequence_number": seq + 1, "item_id": call_item["id"], "output_index": output_index, "delta": call_item["arguments"]}),
            ("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "sequence_number": seq + 2, "item_id": call_item["id"], "output_index": output_index, "arguments": call_item["arguments"]}),
            ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq + 3, "output_index": output_index, "item": call_item}),
        ])
        seq += 4
    response = {"id": rid, "object": "response", "created_at": created, "status": "completed", "model": model, "output": outputs}
    if usage:
        response["usage"] = usage
    events.append(("response.completed", {"type": "response.completed", "sequence_number": seq, "response": response}))
    return events
