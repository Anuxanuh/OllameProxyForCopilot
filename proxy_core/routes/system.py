from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter, FastAPI, Request

from proxy_core.request_utils import safe_text_preview


TRACE_PATHS = {"/api/show", "/api/chat", "/api/generate", "/api/tags", "/api/version", "/v1/chat/completions"}
TRACE_BODY_PATHS = {"/api/show", "/api/chat", "/api/generate", "/v1/chat/completions"}


def register_request_trace_middleware(app: FastAPI, logger) -> None:
    @app.middleware("http")
    async def request_trace_middleware(request: Request, call_next):
        path = request.url.path
        method = request.method
        start = time.perf_counter()

        if method == "POST" and path in TRACE_BODY_PATHS:
            raw = await request.body()
            logger.info("request %s %s body=%s", method, path, safe_text_preview(raw))
        elif path in TRACE_PATHS:
            logger.info("request %s %s", method, path)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("response %s %s status=500 duration_ms=%.2f", method, path, duration_ms)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        if path in TRACE_PATHS:
            logger.info(
                "response %s %s status=%s duration_ms=%.2f",
                method,
                path,
                response.status_code,
                duration_ms,
            )
        return response


def create_system_router(ollama_version: str) -> APIRouter:
    router = APIRouter()

    @router.post("/api/version")
    @router.get("/api/version")
    @router.get("/version")
    async def version() -> Dict[str, str]:
        return {"version": ollama_version}

    return router
