from fastapi import Request
from fastapi.responses import JSONResponse

from .config import Settings


def _bearer_value(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return None


def client_api_key(request: Request) -> str | None:
    auth = _bearer_value(request.headers.get("authorization"))
    if auth:
        return auth
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key.strip()
    x_goog = request.headers.get("x-goog-api-key")
    if x_goog:
        return x_goog.strip()
    return None


def auth_error(message: str = "Invalid API key") -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"type": "authentication_error", "message": message, "code": "authentication_error"}},
    )


def validate_bridge_auth(request: Request, settings: Settings) -> JSONResponse | None:
    if not settings.bridge_api_key:
        return None
    if client_api_key(request) != settings.bridge_api_key:
        return auth_error()
    return None


def upstream_auth_headers(request: Request, settings: Settings) -> dict[str, str] | JSONResponse:
    if settings.sub2api_api_key:
        return {"Authorization": f"Bearer {settings.sub2api_api_key}"}
    if not settings.passthrough_client_auth:
        return auth_error("No upstream API key configured")
    key = client_api_key(request)
    if not key:
        return auth_error("API key is required")
    return {"Authorization": f"Bearer {key}"}
