from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict

from fastapi import HTTPException, Request, status


def safe_text_preview(raw: bytes, limit: int = 240) -> str:
    text = raw.decode("utf-8", errors="replace").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def flatten_options(body: Dict[str, Any], payload: Dict[str, Any]) -> None:
    options = body.get("options") or {}
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "n",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "seed",
        "stop",
        "best_of",
        "min_p",
        "typical_p",
        "repeat_last_n",
        "num_keep",
        "num_predict",
        "num_ctx",
        "num_thread",
    ]:
        if key in options:
            payload[key] = options[key]
    if "stream" in body:
        payload["stream"] = body["stream"]
    if "keep_alive" in body:
        payload["keep_alive"] = body["keep_alive"]


async def parse_request_json(
    request: Request,
    logger: logging.Logger,
    *,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Parse request body as JSON with a safe fallback for single-quoted dict payloads."""
    raw = await request.body()
    if not raw:
        if allow_empty:
            return {}
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="request body is empty")

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        text = raw.decode("utf-8", errors="replace").strip()
        try:
            alt = ast.literal_eval(text)
        except Exception:
            alt = None
        if isinstance(alt, dict):
            body = alt
            logger.info("json fallback ast.literal_eval accepted path=%s", request.url.path)
        else:
            logger.warning("json parse failed path=%s body=%s", request.url.path, safe_text_preview(raw))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid JSON body",
            )

    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON body must be an object")
    return body
