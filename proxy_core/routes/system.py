from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict

from fastapi import APIRouter, FastAPI, Request

from proxy_core.request_utils import safe_text_full, safe_text_preview


TRACE_PATHS = {"/api/show", "/api/chat", "/api/generate", "/api/tags", "/api/version", "/v1/chat/completions"}
TRACE_BODY_PATHS = {"/api/show", "/api/chat", "/api/generate", "/v1/chat/completions"}


def register_request_trace_middleware(app: FastAPI, logger) -> None:
    file_trace_logger = logging.getLogger("ollama_proxy.filetrace")

    @app.middleware("http")
    async def request_trace_middleware(request: Request, call_next):
        path = request.url.path
        method = request.method
        start = time.perf_counter()

        user_agent = request.headers.get("user-agent")
        request_id = request.headers.get("x-request-id") or request.headers.get("x-ms-client-request-id")
        github_request_id = request.headers.get("x-github-request-id")

        if method == "POST" and path in TRACE_BODY_PATHS:
            raw = await request.body()
            body_bytes = len(raw)
            body_sha1 = hashlib.sha1(raw).hexdigest()
            logger.info(
                "request %s %s ua=%s x_request_id=%s x_github_request_id=%s body_bytes=%s body_sha1=%s body_preview=%s",
                method,
                path,
                user_agent,
                request_id,
                github_request_id,
                body_bytes,
                body_sha1,
                safe_text_preview(raw),
            )
            # Keep full request body in file logs for postmortem analysis.
            file_trace_logger.info(
                "request-full %s %s ua=%s x_request_id=%s x_github_request_id=%s body_bytes=%s body_sha1=%s body=%s",
                method,
                path,
                user_agent,
                request_id,
                github_request_id,
                body_bytes,
                body_sha1,
                safe_text_full(raw),
            )
        elif path in TRACE_PATHS:
            logger.info(
                "request %s %s ua=%s x_request_id=%s x_github_request_id=%s",
                method,
                path,
                user_agent,
                request_id,
                github_request_id,
            )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("response %s %s status=500 duration_ms=%.2f", method, path, duration_ms)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        if path in TRACE_PATHS:
            response_content_length = response.headers.get("content-length")
            logger.info(
                "response %s %s status=%s duration_ms=%.2f content_length=%s x_request_id=%s x_github_request_id=%s",
                method,
                path,
                response.status_code,
                duration_ms,
                response_content_length,
                request_id,
                github_request_id,
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
