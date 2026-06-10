import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    sub2api_base_url: str
    sub2api_api_key: str | None
    bridge_api_key: str | None
    passthrough_client_auth: bool
    request_timeout_seconds: float
    database_host: str
    database_port: int
    database_user: str
    database_password: str | None
    database_dbname: str
    antigravity_group_names: tuple[str, ...]


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def get_settings() -> Settings:
    return Settings(
        sub2api_base_url=os.getenv("SUB2API_BASE_URL", "http://sub2api:8080").rstrip("/"),
        sub2api_api_key=(os.getenv("SUB2API_API_KEY") or "").strip() or None,
        bridge_api_key=(os.getenv("BRIDGE_API_KEY") or "").strip() or None,
        passthrough_client_auth=_bool_env("PASSTHROUGH_CLIENT_AUTH", True),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "900")),
        database_host=os.getenv("DATABASE_HOST", "postgres"),
        database_port=int(os.getenv("DATABASE_PORT", "5432")),
        database_user=os.getenv("DATABASE_USER", "sub2api"),
        database_password=(os.getenv("DATABASE_PASSWORD") or "").strip() or None,
        database_dbname=os.getenv("DATABASE_DBNAME", "sub2api"),
        antigravity_group_names=_csv_env("ANTIGRAVITY_GROUP_NAMES", "Antigravity"),
    )
