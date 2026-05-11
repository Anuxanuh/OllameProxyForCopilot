from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from proxy_core.presenters import make_chat_response, make_ollama_response
from proxy_core.request_utils import flatten_options, parse_request_json
from proxy_core.state import ProxyState
from rule_handlers.base import require_rule_capability


def create_chat_router(state: ProxyState, logger) -> APIRouter:
    router = APIRouter()

    @router.post("/api/generate")
    async def generate(request: Request) -> Any:
        body = await parse_request_json(request, logger)
        model_cfg = await state.resolve_model_config(body.get("model"))
        source_cfg = state.resolve_source_config(model_cfg["source"])
        handler = source_cfg["handler"]
        model = model_cfg["name"]
        prompt = body.get("prompt", "")

        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        if body.get("system"):
            messages.insert(0, {"role": "system", "content": body["system"]})

        payload: Dict[str, Any] = {"model": model_cfg["upstream_model"], "messages": messages}
        flatten_options(body, payload)
        payload["stream"] = True

        want_stream = body.get("stream", True)
        logger.info(
            "generate rule=%s model=%s upstream_model=%s want_stream=%s prompt_len=%s",
            source_cfg["rule"],
            state.normalize_model_name(body.get("model")),
            model_cfg["upstream_model"],
            want_stream,
            len(str(prompt)),
        )
        if want_stream:
            return StreamingResponse(
                handler.stream_generate_to_ollama(source_cfg, model, payload),
                media_type="application/x-ndjson",
            )

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        done_reason = "stop"
        async for line_json in handler.stream_generate_to_ollama(source_cfg, model, payload):
            obj = json.loads(line_json)
            full_text += obj.get("response") or ""
            if obj.get("done"):
                prompt_tokens = obj.get("prompt_eval_count", 0)
                completion_tokens = obj.get("eval_count", 0)
                done_reason = str(obj.get("done_reason") or done_reason)

        result = make_ollama_response(
            model,
            full_text,
            {
                "done_reason": done_reason,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 0,
                "eval_count": completion_tokens,
                "eval_duration": 0,
            },
        )
        return JSONResponse(content=result)

    @router.post("/api/chat")
    async def chat(request: Request) -> Any:
        body = await parse_request_json(request, logger)
        model_cfg = await state.resolve_model_config(body.get("model"))
        source_cfg = state.resolve_source_config(model_cfg["source"])
        handler = source_cfg["handler"]
        model = model_cfg["name"]
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages must be a list")

        payload: Dict[str, Any] = {"model": model_cfg["upstream_model"], "messages": messages}
        flatten_options(body, payload)
        payload["stream"] = True
        if "tools" in body:
            require_rule_capability(source_cfg, "supports_tools", "tools")
            payload["tools"] = body["tools"]

        want_stream = body.get("stream", True)
        logger.info(
            "chat rule=%s model=%s upstream_model=%s want_stream=%s messages=%s tools=%s",
            source_cfg["rule"],
            state.normalize_model_name(body.get("model")),
            model_cfg["upstream_model"],
            want_stream,
            len(messages),
            "tools" in body,
        )
        if want_stream:
            return StreamingResponse(
                handler.stream_chat_to_ollama(source_cfg, model, payload),
                media_type="application/x-ndjson",
            )

        full_content = ""
        reasoning_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        tool_calls: List[Dict[str, Any]] = []
        done_reason = "stop"
        async for line_json in handler.stream_chat_to_ollama(source_cfg, model, payload):
            obj = json.loads(line_json)
            message = obj.get("message") or {}
            full_content += message.get("content") or ""
            reasoning_content += message.get("reasoning_content") or ""
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict):
                    tool_calls.append(tool_call)
            if obj.get("done"):
                prompt_tokens = obj.get("prompt_eval_count", 0)
                completion_tokens = obj.get("eval_count", 0)
                done_reason = str(obj.get("done_reason") or done_reason)

        response_message: Dict[str, Any] = {"role": "assistant", "content": full_content}
        if reasoning_content:
            response_message["reasoning_content"] = reasoning_content
        if tool_calls:
            response_message["tool_calls"] = tool_calls

        result = make_chat_response(
            model,
            response_message,
            {
                "done_reason": done_reason,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 0,
                "eval_count": completion_tokens,
                "eval_duration": 0,
            },
        )
        return JSONResponse(content=result)

    @router.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request) -> Any:
        body = await parse_request_json(request, logger)
        model_cfg = await state.resolve_model_config(body.get("model"))
        source_cfg = state.resolve_source_config(model_cfg["source"])
        handler = source_cfg["handler"]
        model = model_cfg["name"]
        if "tools" in body:
            require_rule_capability(source_cfg, "supports_tools", "tools")
        body["model"] = model_cfg["upstream_model"]
        want_stream = body.get("stream", False)

        logger.info(
            "v1/chat/completions rule=%s model=%s want_stream=%s messages=%s",
            source_cfg["rule"],
            model,
            want_stream,
            len(body.get("messages") or []),
        )

        if want_stream:
            return StreamingResponse(
                handler.stream_chat_openai(source_cfg, model, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        data = await handler.chat_json_openai(source_cfg, model, body)
        return JSONResponse(content=data)

    return router
