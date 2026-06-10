from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .admin_proxy import admin_account_models, admin_account_test, admin_default_model_mapping
from .anthropic_proxy import handle_messages
from .config import get_settings
from .models import model_list
from .responses_bridge import handle_chat_completions, handle_responses

app = FastAPI(title="NF Sub2API Antigravity Bridge", version="0.1.0")
settings = get_settings()


@app.get("/health")
async def health():
    return {"ok": True, "service": "nf-sub2api-bridge"}


@app.get("/v1/models")
async def models():
    return model_list()


@app.post("/v1/responses")
async def responses(request: Request):
    return await handle_responses(request, settings)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await handle_chat_completions(request, settings)


@app.post("/v1/messages")
async def messages(request: Request):
    return await handle_messages(request, settings)


@app.get("/api/v1/admin/accounts/antigravity/default-model-mapping")
async def admin_antigravity_default_mapping(request: Request):
    return await admin_default_model_mapping(request, settings)


@app.get("/api/v1/admin/accounts/{account_id}/models")
async def admin_models(request: Request, account_id: int):
    return await admin_account_models(request, settings, account_id)


@app.post("/api/v1/admin/accounts/{account_id}/test")
async def admin_test(request: Request, account_id: int):
    return await admin_account_test(request, settings, account_id)


@app.get("/")
async def root():
    return {"ok": True, "routes": ["/health", "/v1/models", "/v1/responses", "/v1/chat/completions", "/v1/messages", "/api/v1/admin/accounts/{account_id}/models", "/api/v1/admin/accounts/{account_id}/test"]}


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "server_error", "message": str(exc), "code": "server_error"}},
    )
