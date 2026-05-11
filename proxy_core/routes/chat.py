from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from proxy_core.presenters import make_chat_response, make_ollama_response
from proxy_core.request_utils import flatten_options, parse_request_json, safe_text_full
from proxy_core.state import ProxyState
from rule_handlers.base import require_rule_capability


def _is_copilot_request(request: Request) -> bool:
    user_agent = str(request.headers.get("user-agent") or "").lower()
    return "copilot" in user_agent or "vscode" in user_agent


async def _cap_copilot_sse_stream(
    request: Request,
    stream,
    logger,
    model: str,
):
    if not _is_copilot_request(request):
        async for chunk in stream:
            yield chunk
        return

    file_trace_logger = logging.getLogger("ollama_proxy.filetrace")
    request_id = request.headers.get("x-request-id") or request.headers.get("x-ms-client-request-id")
    github_request_id = request.headers.get("x-github-request-id")

    total_bytes = 0
    total_chunks = 0
    emitted_choice_event = False
    emitted_content_chars = 0
    data_events = 0
    done_events = 0
    parse_failures = 0
    total_tool_call_items = 0
    total_function_call_chars = 0
    removed_reasoning_chars = 0
    delta_key_counts: Dict[str, int] = {}
    event_index = 0
    message_id = f"chatcmpl-proxy-cap-{int(time.time() * 1000)}"
    logger.info(
        "v1/chat/completions copilot sse guard enabled model=%s stream_id=%s",
        model,
        message_id,
    )

    def _log_stream_summary(reason: str) -> None:
        logger.info(
            "v1/chat/completions copilot sse summary model=%s stream_id=%s reason=%s bytes=%s chunks=%s data_events=%s done_events=%s content_chars=%s tool_call_items=%s removed_reasoning_chars=%s parse_failures=%s emitted_choice=%s",
            model,
            message_id,
            reason,
            total_bytes,
            total_chunks,
            data_events,
            done_events,
            emitted_content_chars,
            total_tool_call_items,
            removed_reasoning_chars,
            parse_failures,
            emitted_choice_event,
        )
        if delta_key_counts:
            logger.info(
                "v1/chat/completions copilot sse delta-keys model=%s stream_id=%s keys=%s function_call_chars=%s",
                model,
                message_id,
                json.dumps(delta_key_counts, ensure_ascii=False, sort_keys=True),
                total_function_call_chars,
            )

        file_trace_logger.info(
            "sse-summary stream_id=%s model=%s reason=%s x_request_id=%s x_github_request_id=%s bytes=%s chunks=%s data_events=%s done_events=%s content_chars=%s tool_call_items=%s parse_failures=%s",
            message_id,
            model,
            reason,
            request_id,
            github_request_id,
            total_bytes,
            total_chunks,
            data_events,
            done_events,
            emitted_content_chars,
            total_tool_call_items,
            parse_failures,
        )

    def _minify_sse_chunk(raw: bytes | bytearray | str):
        local = {
            "is_data": False,
            "is_done": False,
            "parse_failed": False,
            "choice_event": False,
            "content_chars": 0,
            "tool_call_items": 0,
            "function_call_chars": 0,
            "removed_reasoning_chars": 0,
            "delta_keys": {},
            "tool_calls_signature": None,
            "has_content": False,
            "has_tool_calls": False,
            "has_function_call": False,
            "has_role": False,
            "has_finish_reason": False,
        }
        try:
            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except Exception:
            local["parse_failed"] = True
            return raw, local

        stripped = line.strip()
        if stripped == "data: [DONE]":
            local["is_done"] = True
            return raw, local
        if not stripped.startswith("data: "):
            return raw, local

        local["is_data"] = True

        try:
            payload = json.loads(stripped[6:])
        except Exception:
            local["parse_failed"] = True
            return raw, local

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list):
            return raw, local

        local["choice_event"] = True
        has_meaningful_delta = False

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                for delta_key in delta.keys():
                    key_text = str(delta_key)
                    local_delta_keys = local["delta_keys"]
                    local_delta_keys[key_text] = int(local_delta_keys.get(key_text, 0)) + 1
                content = delta.get("content")
                if isinstance(content, str) and content:
                    has_meaningful_delta = True
                    local["has_content"] = True
                    local["content_chars"] += len(content)
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    local["tool_call_items"] += len(tool_calls)
                    meaningful_tool_call = False
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        tc_id = tool_call.get("id")
                        tc_type = tool_call.get("type")
                        if isinstance(tc_id, str) and tc_id:
                            meaningful_tool_call = True
                        if isinstance(tc_type, str) and tc_type:
                            meaningful_tool_call = True

                        func = tool_call.get("function")
                        if isinstance(func, dict):
                            fn_name = func.get("name")
                            if isinstance(fn_name, str) and fn_name:
                                meaningful_tool_call = True
                            fn_args = func.get("arguments")
                            if isinstance(fn_args, str) and fn_args:
                                meaningful_tool_call = True

                        # Old-style top-level tool call fields
                        top_args = tool_call.get("arguments")
                        if isinstance(top_args, str) and top_args:
                            meaningful_tool_call = True

                    if meaningful_tool_call:
                        has_meaningful_delta = True
                        local["has_tool_calls"] = True

                function_call = delta.get("function_call")
                if isinstance(function_call, dict):
                    fn_args = function_call.get("arguments")
                    fn_name = function_call.get("name")
                    if isinstance(fn_name, str) and fn_name:
                        has_meaningful_delta = True
                        local["has_function_call"] = True
                    if isinstance(fn_args, str) and fn_args:
                        local["function_call_chars"] += len(fn_args)
                        has_meaningful_delta = True
                        local["has_function_call"] = True

                role = delta.get("role")
                if isinstance(role, str) and role:
                    has_meaningful_delta = True
                    local["has_role"] = True
            message = choice.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("content"), str) and message.get("content"):
                    has_meaningful_delta = True

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                # Copilot treats finish_reason="length" as a hard error ("Response too long").
                # Remap to "stop" so truncated responses are accepted instead of rejected.
                if finish_reason == "length":
                    choice["finish_reason"] = "stop"
                    local["remapped_length_finish"] = True
                has_meaningful_delta = True
                local["has_finish_reason"] = True

        if not has_meaningful_delta:
            return raw, local

        return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode(), local

    async for chunk in stream:
        event_index += 1
        raw_chunk_bytes = len(chunk) if isinstance(chunk, (bytes, bytearray)) else len(str(chunk).encode("utf-8", errors="ignore"))

        # Drop bulky fields (tool_calls/reasoning/usage...) from SSE for Copilot compatibility.
        chunk, local_stats = _minify_sse_chunk(chunk)

        if local_stats["is_data"]:
            data_events += 1
        if local_stats["is_done"]:
            done_events += 1
        if local_stats["parse_failed"]:
            parse_failures += 1
        if local_stats["choice_event"]:
            emitted_choice_event = True

        for key, count in (local_stats.get("delta_keys") or {}).items():
            delta_key_counts[str(key)] = int(delta_key_counts.get(str(key), 0)) + int(count)

        emitted_content_chars += int(local_stats["content_chars"])
        total_tool_call_items += int(local_stats["tool_call_items"])
        total_function_call_chars += int(local_stats["function_call_chars"])
        removed_reasoning_chars += int(local_stats["removed_reasoning_chars"])

        out_chunk_bytes = len(chunk) if isinstance(chunk, (bytes, bytearray)) else len(str(chunk).encode("utf-8", errors="ignore"))
        try:
            event_payload = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        except Exception:
            event_payload = "<decode_failed>"

        file_trace_logger.info(
            "sse-event stream_id=%s idx=%s x_request_id=%s x_github_request_id=%s raw_bytes=%s out_bytes=%s is_data=%s is_done=%s parse_failed=%s dropped_empty=%s content_chars=%s tool_call_items=%s function_call_chars=%s delta_keys=%s payload=%s",
            message_id,
            event_index,
            request_id,
            github_request_id,
            raw_chunk_bytes,
            out_chunk_bytes,
            local_stats.get("is_data"),
            local_stats.get("is_done"),
            local_stats.get("parse_failed"),
            False,
            local_stats.get("content_chars"),
            local_stats.get("tool_call_items"),
            local_stats.get("function_call_chars"),
            json.dumps(local_stats.get("delta_keys") or {}, ensure_ascii=False, sort_keys=True),
            safe_text_full(event_payload.encode("utf-8", errors="replace")),
        )

        chunk_len = len(chunk) if isinstance(chunk, (bytes, bytearray)) else len(str(chunk).encode("utf-8", errors="ignore"))
        total_bytes += chunk_len
        total_chunks += 1
        yield chunk

    _log_stream_summary("upstream_done")


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
            "v1/chat/completions rule=%s model=%s want_stream=%s messages=%s ua=%s x_request_id=%s x_github_request_id=%s",
            source_cfg["rule"],
            model,
            want_stream,
            len(body.get("messages") or []),
            request.headers.get("user-agent"),
            request.headers.get("x-request-id") or request.headers.get("x-ms-client-request-id"),
            request.headers.get("x-github-request-id"),
        )

        if want_stream:
            raw_stream = handler.stream_chat_openai(source_cfg, model, body)
            return StreamingResponse(
                _cap_copilot_sse_stream(request, raw_stream, logger, model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        data = await handler.chat_json_openai(source_cfg, model, body)
        return JSONResponse(content=data)

    return router
