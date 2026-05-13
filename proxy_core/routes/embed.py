from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from proxy_core.request_utils import flatten_options, parse_request_json
from proxy_core.state import ProxyState
from rule_handlers.base import require_rule_capability


def create_embed_router(state: ProxyState, logger) -> APIRouter:
    router = APIRouter()

    @router.post("/api/embed")
    async def embed(request: Request) -> Any:
        body = await parse_request_json(request, logger)
        model_cfg = await state.resolve_model_config(body.get("model"))
        source_cfg = state.resolve_source_config(model_cfg["source"])
        handler = source_cfg["handler"]
        require_rule_capability(
            source_cfg, "supports_embeddings", "embeddings")
        input_data = body.get("input") or body.get("prompt")
        if input_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="input is required")

        payload: Dict[str, Any] = {
            "model": model_cfg["upstream_model"], "input": input_data}
        if "truncate" in body:
            payload["truncate"] = body["truncate"]
        flatten_options(body, payload)

        data = await handler.proxy_json(source_cfg, "embeddings", payload)
        return JSONResponse(content=data)

    @router.post("/api/embeddings")
    async def embeddings(request: Request) -> Any:
        return await embed(request)

    return router
