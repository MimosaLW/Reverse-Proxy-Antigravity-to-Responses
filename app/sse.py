import json
from typing import Any, Iterable


def sse_event(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def parse_sse_events(buffer: str) -> tuple[list[str], str]:
    events: list[str] = []
    while True:
        marker = "\n\n"
        idx = buffer.find(marker)
        if idx < 0:
            break
        raw = buffer[:idx]
        buffer = buffer[idx + len(marker):]
        data_lines = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            events.append("\n".join(data_lines))
    return events, buffer


def iter_gemini_sse_payloads(text: str) -> Iterable[dict[str, Any]]:
    events, _ = parse_sse_events(text)
    for data in events:
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if isinstance(payload.get("response"), dict):
                payload = payload["response"]
            yield payload
