from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Static specs mirrored from Antigravity-Manager resources/model_specs.json,
# with a few NarraFork-facing virtual IDs layered on top.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gemini-2.0-flash": {"max_output_tokens": 65535, "thinking_budget": 24576, "is_thinking": False},
    "gemini-2.5-flash": {"max_output_tokens": 65535, "thinking_budget": 32768, "is_thinking": True},
    "gemini-3-flash": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    "gemini-3-pro-high": {"max_output_tokens": 65535, "thinking_budget": 10001, "is_thinking": True},
    "gemini-3.1-pro-preview": {"max_output_tokens": 65535, "thinking_budget": 1001, "is_thinking": True},
    "claude-sonnet-4-6": {"max_output_tokens": 64000, "thinking_budget": 32768, "is_thinking": True},
    "claude-sonnet-4-6-thinking": {"max_output_tokens": 64000, "thinking_budget": 32768, "is_thinking": True},
    "claude-opus-4-6-thinking": {"max_output_tokens": 64000, "thinking_budget": 32768, "is_thinking": True},
    "gpt-oss-120b-medium": {"max_output_tokens": 32768, "thinking_budget": 0, "is_thinking": False},
    # Gemini 3.5 Flash virtual tier IDs. Official v1internal physical tiers
    # are selected via UPSTREAM_MODEL_ALIASES; do not emulate tiers with
    # generationConfig.thinkingConfig.thinkingLevel.
    "gemini-3.5-flash": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    "gemini-3.5-flash-high": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    "gemini-3.5-flash-medium": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    "gemini-3.5-flash-low": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    # Physical Antigravity v1internal Gemini 3.5 tier IDs observed from
    # fetchAvailableModels. Keeping specs here prevents any 3.5 path from
    # falling back to the older gemini-3-flash spec.
    "gemini-3.5-flash-extra-low": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    "gemini-3-flash-agent": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
    # Official Antigravity v1internal Gemini 3.1 Pro tiers. High uses the
    # replacement physical ID advertised by fetchAvailableModels.
    "gemini-3.1-pro-low": {"max_output_tokens": 65535, "thinking_budget": 1001, "is_thinking": True},
    "gemini-3.1-pro-high": {"max_output_tokens": 65535, "thinking_budget": 10001, "is_thinking": True},
    "gemini-pro-agent": {"max_output_tokens": 65535, "thinking_budget": 10001, "is_thinking": True},
    "gemini-3-pro-low": {"max_output_tokens": 65535, "thinking_budget": 1001, "is_thinking": True},
    "gemini-3-pro-preview": {"max_output_tokens": 65535, "thinking_budget": 1001, "is_thinking": True},
    "gemini-3-pro-image": {"max_output_tokens": 65536, "thinking_budget": 32768, "is_thinking": True},
}

# User/client-facing aliases. Route model is the bridge's logical model; upstream
# physical path can still be rewritten separately by UPSTREAM_MODEL_ALIASES.
MODEL_ALIASES: dict[str, str] = {
    # Suffixless Gemini 3.5 Flash is a client-facing compatibility alias for
    # the official Medium tier. The bare upstream physical path
    # /models/gemini-3.5-flash 404s on this Antigravity deployment.
    "gemini-3.5-flash": "gemini-3.5-flash-medium",
    "gpt-4": "gemini-2.5-flash",
    "gpt-4-turbo": "gemini-2.5-flash",
    "gpt-4o": "gemini-2.5-flash",
    "gpt-4o-mini": "gemini-2.5-flash",
    "gpt-3.5-turbo": "gemini-2.5-flash",
    "claude-3-5-sonnet": "claude-sonnet-4-6-thinking",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6-thinking",
    "claude-3-7-sonnet": "claude-sonnet-4-6-thinking",
    "claude-3-7-sonnet-thinking": "claude-sonnet-4-6-thinking",
    "claude-sonnet-4-5": "claude-sonnet-4-6-thinking",
    "claude-sonnet-4-5-thinking": "claude-sonnet-4-6-thinking",
    "claude-opus-4": "claude-opus-4-6-thinking",
    "claude-opus-4-5-thinking": "claude-opus-4-6-thinking",
    "claude-opus-4-6": "claude-opus-4-6-thinking",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "gemini-3-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
}

# Antigravity /v1beta Gemini routes do not consistently apply account-level
# model_mapping. These are bridge-local physical rewrites only where the public
# client ID has no directly callable or usable Antigravity physical path.
# Gemini 3.5 Flash tiers are official v1internal physical model IDs. The
# suffixless client alias is intentionally treated as Medium.
GEMINI_35_FLASH_TIER_UPSTREAM_MODELS: dict[str, str] = {
    "gemini-3.5-flash": "gemini-3.5-flash-low",
    "gemini-3.5-flash-high": "gemini-3-flash-agent",
    "gemini-3.5-flash-medium": "gemini-3.5-flash-low",
    "gemini-3.5-flash-low": "gemini-3.5-flash-extra-low",
}

GEMINI_31_PRO_UPSTREAM_MODELS: dict[str, str] = {
    "gemini-3.1-pro-low": "gemini-3.1-pro-low",
    "gemini-3.1-pro-high": "gemini-pro-agent",
    "gemini-3-pro-low": "gemini-3.1-pro-low",
    "gemini-3-pro-high": "gemini-pro-agent",
}

UPSTREAM_MODEL_ALIASES: dict[str, str] = {
    **GEMINI_35_FLASH_TIER_UPSTREAM_MODELS,
    **GEMINI_31_PRO_UPSTREAM_MODELS,
    "gemini-3.1-pro-preview": "gemini-3.1-pro-low",
    "gemini-3-pro-preview": "gemini-3.1-pro-low",
}

GEMINI_35_FLASH_MODELS = set(GEMINI_35_FLASH_TIER_UPSTREAM_MODELS)
GEMINI_31_PRO_MODELS = set(GEMINI_31_PRO_UPSTREAM_MODELS)
GEMINI_OFFICIAL_V1INTERNAL_MODELS = GEMINI_35_FLASH_MODELS | GEMINI_31_PRO_MODELS


GEMINI_RESPONSES_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-3.5-flash-high",
    "gemini-3.5-flash-medium",
    "gemini-3.5-flash-low",
    "gemini-3.1-pro-low",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-low",
    "gemini-3-pro-high",
    "gemini-3-pro-preview",
    "gemini-3-pro-image",
    "gpt-oss-120b-medium",
}

CLAUDE_MODELS = {
    "claude-sonnet-4-6-thinking",
    "claude-opus-4-6-thinking",
}

RESPONSES_MODELS = GEMINI_RESPONSES_MODELS | CLAUDE_MODELS

# What /v1/models advertises. Keep this synchronized with the NarraFork UI list:
# Gemini 3.5 Flash (High/Medium/Low), Gemini 3.1 Pro (Low/High),
# Claude Sonnet/Opus 4.6 Thinking, and GPT-OSS 120B Medium.
IDE_MODELS = [
    "gemini-3.5-flash-high",
    "gemini-3.5-flash-medium",
    "gemini-3.5-flash-low",
    "gemini-3.1-pro-low",
    "gemini-3.1-pro-high",
    "claude-sonnet-4-6-thinking",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gemini-3.5-flash-high": "Gemini 3.5 Flash (High)",
    "gemini-3.5-flash-medium": "Gemini 3.5 Flash (Medium)",
    "gemini-3.5-flash-low": "Gemini 3.5 Flash (Low)",
    "gemini-3.1-pro-low": "Gemini 3.1 Pro (Low)",
    "gemini-3.1-pro-high": "Gemini 3.1 Pro (High)",
    "claude-sonnet-4-6-thinking": "Claude Sonnet 4.6 (Thinking)",
    "claude-opus-4-6-thinking": "Claude Opus 4.6 (Thinking)",
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
}


@dataclass(frozen=True)
class ModelRoute:
    original_model: str
    route_model: str
    upstream_model: str
    protocol: str


def resolve_client_model(model: str) -> str:
    model = (model or "").strip()
    return MODEL_ALIASES.get(model, model)


def resolve_upstream_model(model: str) -> str:
    route_model = resolve_client_model(model)
    return UPSTREAM_MODEL_ALIASES.get(route_model, route_model)


def is_gemini35_flash_model(model: str) -> bool:
    return resolve_client_model(model) in GEMINI_35_FLASH_MODELS


def is_official_v1internal_gemini_model(model: str) -> bool:
    return resolve_client_model(model) in GEMINI_OFFICIAL_V1INTERNAL_MODELS


def is_gemini_bridge_model(model: str) -> bool:
    return resolve_client_model(model) in GEMINI_RESPONSES_MODELS


def is_claude_bridge_model(model: str) -> bool:
    return resolve_client_model(model) in CLAUDE_MODELS


def build_model_route(model: str) -> ModelRoute:
    original = (model or "").strip()
    route = resolve_client_model(original)
    upstream = resolve_upstream_model(route)
    if route in CLAUDE_MODELS:
        protocol = "claude"
    elif route in GEMINI_RESPONSES_MODELS:
        protocol = "gemini"
    else:
        protocol = "passthrough"
    return ModelRoute(original, route, upstream, protocol)


def _spec_id(model: str) -> str:
    route = resolve_client_model(model)
    upstream = resolve_upstream_model(route)
    if route in MODEL_SPECS:
        return route
    if upstream in MODEL_SPECS:
        return upstream
    if route.startswith("gemini-3.5-flash"):
        return "gemini-3.5-flash-medium"
    if "claude" in route and route not in MODEL_SPECS:
        return "claude-sonnet-4-6-thinking"
    return route


def get_max_output_tokens(model: str) -> int:
    spec = MODEL_SPECS.get(_spec_id(model), {})
    return int(spec.get("max_output_tokens") or 65535)


def get_thinking_budget(model: str) -> int:
    spec = MODEL_SPECS.get(_spec_id(model), {})
    return int(spec.get("thinking_budget") or 24576)


def is_thinking_model(model: str) -> bool:
    route = resolve_client_model(model).lower()
    spec = MODEL_SPECS.get(_spec_id(route), {})
    if spec.get("is_thinking") is not None:
        return bool(spec.get("is_thinking"))
    return "thinking" in route or "gemini-3-pro" in route or "gemini-3.1-pro" in route


def get_thinking_level(model: str) -> str | None:
    spec = MODEL_SPECS.get(_spec_id(model), {})
    level = spec.get("thinking_level")
    return str(level) if level else None


def gemini_candidate_upstream_models(model: str) -> list[str]:
    route = resolve_client_model(model)
    first = resolve_upstream_model(route)
    if route not in {
        "gemini-3-pro",
        "gemini-3-pro-preview",
        "gemini-3-pro-high",
        "gemini-3-pro-low",
        "gemini-3.1-pro",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-high",
        "gemini-3.1-pro-low",
    }:
        return [first]

    raw_candidates = [
        first,
        route,
        "gemini-3.1-pro-low",
        "gemini-pro-agent",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-high",
        "gemini-3-pro-high",
        "gemini-3-pro-low",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        physical = UPSTREAM_MODEL_ALIASES.get(candidate, candidate)
        if physical and physical not in seen:
            seen.add(physical)
            out.append(physical)
    return out


def model_list() -> dict:
    seen: set[str] = set()
    data = []
    for model in IDE_MODELS:
        if model in seen:
            continue
        seen.add(model)
        item = {"id": model, "object": "model", "owned_by": "antigravity"}
        if model in MODEL_DISPLAY_NAMES:
            item["display_name"] = MODEL_DISPLAY_NAMES[model]
        data.append(item)
    return {"object": "list", "data": data}
