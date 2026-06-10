from __future__ import annotations

import json
import secrets
import time
from typing import Any

from .gemini_converter import BadRequest, UnsupportedFeature


def _rid() -> str:
    return "resp_" + secrets.token_hex(12)


def _iid() -> str:
    return "item_" + secrets.token_hex(12)


def _call_id(raw: str | None = None) -> str:
    if raw:
        return raw
    return "call_" + secrets.token_hex(8)


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
                continue
            if not isinstance(part, dict):
                raise UnsupportedFeature("unsupported content part")
            typ = part.get("type") or "text"
            if typ in {"text", "input_text", "output_text"}:
                out.append(str(part.get("text", "")))
            elif typ == "refusal":
                out.append(str(part.get("refusal", "")))
            elif typ == "input_image":
                continue
            else:
                raise UnsupportedFeature(f"unsupported content part type: {typ}")
        return "".join(out)
    raise UnsupportedFeature("unsupported content format")


def _anthropic_text_content(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _image_url_value(part: dict[str, Any]) -> str | None:
    raw = part.get("image_url") or part.get("url")
    if isinstance(raw, dict):
        raw = raw.get("url") or raw.get("image_url")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _data_url_to_mime_data(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if not sep:
        return None
    mime = header[5:].split(";", 1)[0] or "image/jpeg"
    return mime, data.strip()


def _anthropic_image_block(part: dict[str, Any]) -> dict[str, Any] | None:
    url = _image_url_value(part)
    if not url:
        return None
    data = _data_url_to_mime_data(url)
    if data:
        mime, b64 = data
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    mime = str(part.get("mime_type") or part.get("media_type") or "image/jpeg")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": url}}


def _anthropic_content(content: Any) -> Any:
    if content is None:
        return None
    if isinstance(content, str):
        return _anthropic_text_content(content)
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        text_only = True
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    blocks.append({"type": "text", "text": part})
                    text_parts.append(part)
                continue
            if not isinstance(part, dict):
                raise UnsupportedFeature("unsupported content part")
            typ = part.get("type") or "text"
            if typ in {"text", "input_text", "output_text"}:
                txt = str(part.get("text", ""))
                if txt:
                    blocks.append({"type": "text", "text": txt})
                    text_parts.append(txt)
            elif typ == "refusal":
                txt = str(part.get("refusal", ""))
                if txt:
                    blocks.append({"type": "text", "text": txt})
                    text_parts.append(txt)
            elif typ == "input_image":
                img = _anthropic_image_block(part)
                if img:
                    blocks.append(img)
                    text_only = False
            else:
                raise UnsupportedFeature(f"unsupported content part type: {typ}")
        if not blocks:
            return None
        if text_only:
            return _anthropic_text_content("".join(text_parts))
        return blocks
    raise UnsupportedFeature("unsupported content format")


def _text_block_from_json_string(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
        if isinstance(decoded, str):
            return {"type": "text", "text": decoded}
    except Exception:
        pass
    return {"type": "text", "text": str(value)}


def _anthropic_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return blocks


def responses_to_anthropic(req: dict[str, Any]) -> dict[str, Any]:
    model = str(req.get("model") or "")
    if not model:
        raise BadRequest("model is required")

    system_parts: list[str] = []
    if req.get("instructions"):
        system_parts.append(str(req["instructions"]))

    messages: list[dict[str, Any]] = []
    raw_input = req.get("input")
    if raw_input is None:
        raw_input = ""

    def append_message(role: str, content: Any) -> None:
        if content is None:
            return
        if messages and messages[-1]["role"] == role:
            old_content = messages[-1]["content"]
            if isinstance(old_content, str) and isinstance(content, str):
                try:
                    old = json.loads(old_content)
                    new = json.loads(content)
                    if isinstance(old, str) and isinstance(new, str):
                        messages[-1]["content"] = _anthropic_text_content(old + "\n\n" + new)
                        return
                except Exception:
                    pass
            old_blocks = old_content if isinstance(old_content, list) else [_text_block_from_json_string(str(old_content))]
            new_blocks = content if isinstance(content, list) else [_text_block_from_json_string(str(content))]
            if isinstance(old_blocks, list) and isinstance(new_blocks, list):
                messages[-1]["content"] = old_blocks + new_blocks
                return
        messages.append({"role": role, "content": content})


    if isinstance(raw_input, str):
        append_message("user", _anthropic_text_content(raw_input))
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                raise UnsupportedFeature("responses input items must be objects")
            typ = item.get("type") or ""
            role = item.get("role") or ""
            if role in {"system", "developer"}:
                text = _extract_text_content(item.get("content"))
                if text:
                    system_parts.append(text)
                continue
            if typ == "function_call":
                args: Any = {}
                if item.get("arguments"):
                    try:
                        args = json.loads(item["arguments"])
                    except Exception:
                        args = item["arguments"]
                block = {"type": "tool_use", "id": _call_id(item.get("call_id") or item.get("id")), "name": item.get("name") or "tool", "input": args}
                append_message("assistant", _anthropic_blocks([block]))
                continue
            if typ == "function_call_output":
                block = {"type": "tool_result", "tool_use_id": _call_id(item.get("call_id")), "content": item.get("output") or ""}
                append_message("user", _anthropic_blocks([block]))
                continue
            if typ == "reasoning":
                # OpenAI Responses internal reasoning history is not portable to
                # Anthropic/Antigravity and should not be sent upstream.
                continue
            if typ in {"computer_call", "web_search_call", "custom_tool_call"}:
                raise UnsupportedFeature(f"unsupported input item type: {typ}")
            content = item.get("content")
            if typ == "input_text" and content is None:
                content = item.get("text", "")
            converted = _anthropic_content(content)
            if role in {"assistant", "model"}:
                append_message("assistant", converted)
            else:
                append_message("user", converted)
    else:
        raise UnsupportedFeature("responses input must be a string or array")

    if not messages:
        raise BadRequest("input must contain at least one message")

    out: dict[str, Any] = {
        "model": model,
        "max_tokens": int(req.get("max_output_tokens") or req.get("max_tokens") or 8192),
        "messages": messages,
        "stream": bool(req.get("stream")),
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    if isinstance(req.get("temperature"), (int, float)):
        out["temperature"] = req["temperature"]
    if isinstance(req.get("top_p"), (int, float)):
        out["top_p"] = req["top_p"]
    stop = req.get("stop") or req.get("stop_sequences")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and all(isinstance(x, str) for x in stop):
        out["stop_sequences"] = stop

    tools = []
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        # OpenAI Responses may include built-in tools such as web_search.
        # Anthropic/Antigravity upstreams here only support function tools, so
        # silently drop non-function built-ins instead of failing the request.
        if tool.get("type") != "function":
            continue
        tools.append({
            "name": tool.get("name") or "tool",
            "description": tool.get("description") or "",
            "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
        })
    if tools:
        out["tools"] = tools
    return out


def _usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    inp = int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def anthropic_to_responses(resp: dict[str, Any], original_model: str) -> dict[str, Any]:
    rid = resp.get("id") or _rid()
    outputs: list[dict[str, Any]] = []
    text_parts: list[dict[str, Any]] = []
    for block in resp.get("content") or []:
        if not isinstance(block, dict):
            continue
        typ = block.get("type")
        if typ == "text":
            text_parts.append({"type": "output_text", "text": str(block.get("text") or ""), "annotations": []})
        elif typ == "tool_use":
            args = block.get("input")
            if not isinstance(args, str):
                args = json.dumps(args or {}, ensure_ascii=False, separators=(",", ":"))
            outputs.append({"type": "function_call", "id": _iid(), "call_id": block.get("id") or _call_id(), "name": block.get("name") or "tool", "arguments": args, "status": "completed"})
    if text_parts:
        outputs.insert(0, {"type": "message", "id": _iid(), "role": "assistant", "status": "completed", "content": text_parts})
    if not outputs:
        outputs.append({"type": "message", "id": _iid(), "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "", "annotations": []}]})
    body: dict[str, Any] = {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": original_model,
        "output": outputs,
    }
    usage = _usage(resp.get("usage"))
    if usage:
        body["usage"] = usage
    return body


def parse_anthropic_sse(buffer: str) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    while True:
        idx = buffer.find("\n\n")
        if idx < 0:
            break
        raw = buffer[:idx]
        buffer = buffer[idx + 2:]
        data_lines = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events, buffer


def anthropic_stream_state(model: str) -> dict[str, Any]:
    return {"rid": _rid(), "iid": _iid(), "model": model, "seq": 0, "text": [], "usage": None, "current": "message", "tool_item": None, "tool_args": []}


def responses_sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def stream_open_events(state: dict[str, Any]) -> list[str]:
    rid, iid, model = state["rid"], state["iid"], state["model"]
    created = int(time.time())
    events = [
        ("response.created", {"type": "response.created", "sequence_number": state["seq"], "response": {"id": rid, "object": "response", "created_at": created, "status": "in_progress", "model": model, "output": []}}),
        ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": state["seq"] + 1, "output_index": 0, "item": {"type": "message", "id": iid, "role": "assistant", "status": "in_progress", "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "sequence_number": state["seq"] + 2, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
    ]
    state["seq"] += 3
    return [responses_sse(t, p) for t, p in events]


def stream_event_to_responses(evt: dict[str, Any], state: dict[str, Any]) -> list[str]:
    typ = evt.get("type")
    out: list[str] = []
    if typ == "message_start":
        msg = evt.get("message") or {}
        if isinstance(msg, dict) and msg.get("id"):
            state["rid"] = msg["id"]
        out.extend(stream_open_events(state))
    elif typ == "content_block_delta":
        delta = evt.get("delta") or {}
        if not isinstance(delta, dict):
            return out
        if delta.get("type") == "text_delta" and delta.get("text"):
            text = str(delta["text"])
            state["text"].append(text)
            payload = {"type": "response.output_text.delta", "sequence_number": state["seq"], "item_id": state["iid"], "output_index": 0, "content_index": 0, "delta": text}
            state["seq"] += 1
            out.append(responses_sse("response.output_text.delta", payload))
    elif typ == "message_delta":
        usage = evt.get("usage")
        if isinstance(usage, dict):
            state["usage"] = _usage(usage)
    elif typ == "message_stop":
        out.extend(stream_close_events(state))
    return out


def stream_close_events(state: dict[str, Any]) -> list[str]:
    text = "".join(state["text"])
    iid, rid, model = state["iid"], state["rid"], state["model"]
    output_item = {"type": "message", "id": iid, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}
    response = {"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed", "model": model, "output": [output_item]}
    if state.get("usage"):
        response["usage"] = state["usage"]
    seq = state["seq"]
    events = [
        ("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "item_id": iid, "output_index": 0, "content_index": 0, "text": text}),
        ("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq + 1, "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []}}),
        ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq + 2, "output_index": 0, "item": output_item}),
        ("response.completed", {"type": "response.completed", "sequence_number": seq + 3, "response": response}),
    ]
    state["seq"] += 4
    return [responses_sse(t, p) for t, p in events]
