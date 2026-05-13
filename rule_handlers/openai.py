from __future__ import annotations

import json
import logging
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

import httpx

from .base import RuleHandler, SourceConfig, make_http_error, source_headers, source_url

RULE_OPENAI = "openai"
RULE_OPENAI_COMPATIBLE_PARTIAL = "openai_compatible_partial"
logger = logging.getLogger("ollama_proxy.openai")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds") + "Z"


def _normalize_done_reason(finish_reason: Any) -> str:
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason
    return "stop"


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _first_positive_int(item: Dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        parsed = _as_positive_int(item.get(key))
        if parsed is not None:
            return parsed
    return None


def _find_positive_int_deep(value: Any, keys: set[str], depth: int = 0) -> int | None:
    if depth > 5:
        return None
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in keys:
                parsed = _as_positive_int(nested)
                if parsed is not None:
                    return parsed
        for nested in value.values():
            parsed = _find_positive_int_deep(nested, keys, depth + 1)
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, list):
        for nested in value:
            parsed = _find_positive_int_deep(nested, keys, depth + 1)
            if parsed is not None:
                return parsed
    return None


def _extract_upstream_model_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"upstream": dict(item)}

    context_length = _find_positive_int_deep(
        item,
        {
            "context_length",
            "context_window",
            "max_context_length",
            "max_input_tokens",
            "input_token_limit",
            "token_limit",
            "max_model_len",
            "max_sequence_length",
            "num_ctx",
            "n_ctx",
            "max_tokens",
        },
    )
    if context_length is not None:
        meta["context_length"] = context_length

    parameter_count = _first_positive_int(
        item, ("parameter_count", "params", "num_parameters"))
    if parameter_count is not None:
        meta["parameter_count"] = parameter_count

    model_size = _first_positive_int(item, ("size", "model_size", "bytes"))
    if model_size is not None:
        meta["size"] = model_size

    caps = item.get("capabilities")
    capabilities: list[str] = []
    if isinstance(caps, list):
        capabilities = [str(value).strip()
                        for value in caps if str(value).strip()]

    modalities = item.get("input_modalities") or item.get("modalities")
    if isinstance(modalities, list):
        text_modalities = [str(value).lower() for value in modalities]
        if any("image" in value for value in text_modalities) and "vision" not in capabilities:
            capabilities.append("vision")
    if capabilities:
        meta["capabilities"] = capabilities

    model_info = item.get("model_info")
    if isinstance(model_info, dict):
        meta["model_info"] = dict(model_info)

    return meta


def _validate_tool_messages(messages: list[Dict[str, Any]]) -> None:
    """
    验证OpenAI API消息格式的工具消息约束。

    规则：每个 role='tool' 的消息必须有对应的前面的 role='assistant' 消息且该消息有 tool_calls。
    如果发现孤立的tool消息（无对应的tool_calls），记录警告。
    """
    has_tool_calls = False
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            has_tool_calls = bool(isinstance(
                tool_calls, list) and len(tool_calls) > 0)
        elif role == "tool":
            if not has_tool_calls:
                logger.warning(
                    "invalid message sequence at index %s: role='tool' without preceding assistant message with tool_calls",
                    i,
                )


def _openai_payload_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []

    role_counts = {"system": 0, "user": 0,
                   "assistant": 0, "tool": 0, "other": 0}
    assistant_reasoning_messages = 0
    tool_call_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role in role_counts:
            role_counts[role] += 1
        else:
            role_counts["other"] += 1

        if role == "assistant":
            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content:
                assistant_reasoning_messages += 1
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                tool_call_count += len(
                    [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)])

    tools = payload.get("tools")
    request_tool_count = len([tool for tool in tools if isinstance(
        tool, dict)]) if isinstance(tools, list) else 0
    return {
        "message_count": len(messages),
        "role_counts": role_counts,
        "request_tool_count": request_tool_count,
        "assistant_tool_calls": tool_call_count,
        "assistant_reasoning_messages": assistant_reasoning_messages,
        "stream": bool(payload.get("stream", False)),
        "max_tokens": payload.get("max_tokens") or payload.get("num_predict"),
    }


class OpenAIRuleHandler(RuleHandler):
    rules = (RULE_OPENAI, RULE_OPENAI_COMPATIBLE_PARTIAL)
    default_paths = {
        "chat_completions": "/chat/completions",
        "embeddings": "/embeddings",
        "models": "/models",
    }
    default_rule_config = {
        "supports_embeddings": True,
        "supports_tools": True,
        "request_headers": {},
    }

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return source_cfg.get("rule") == RULE_OPENAI

    async def proxy_json(self, source_cfg: SourceConfig, path_key: str, body: Dict[str, Any]) -> Any:
        url = source_url(source_cfg, path_key)
        payload_stats = _openai_payload_stats(
            body) if path_key == "chat_completions" else {"path_key": path_key}
        try:
            async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
                response = await client.post(url, headers=source_headers(source_cfg), json=body)
                if response.status_code >= 400:
                    error_text = response.text.strip(
                    ) or f"upstream returned HTTP {response.status_code}"
                    logger.warning(
                        "openai request upstream error source=%s path=%s model=%s status=%s stats=%s error=%s",
                        source_cfg.get("name"),
                        path_key,
                        body.get("model"),
                        response.status_code,
                        payload_stats,
                        error_text,
                    )
                    raise make_http_error(response)
                return response.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "openai request upstream connection failure source=%s path=%s model=%s stats=%s error=%s",
                source_cfg.get("name"),
                path_key,
                body.get("model"),
                payload_stats,
                exc,
            )
            raise

    async def stream_chat_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        url = source_url(source_cfg, "chat_completions")
        payload_stats = _openai_payload_stats(payload)
        emitted_any_chunk = False
        try:
            async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
                async with client.stream("POST", url, headers=source_headers(source_cfg), json=payload) as response:
                    if response.status_code >= 400:
                        error_text = (await response.aread()).decode("utf-8", errors="replace").strip()
                        logger.warning(
                            "openai stream upstream error source=%s model=%s status=%s stats=%s error=%s",
                            source_cfg.get("name"),
                            model,
                            response.status_code,
                            payload_stats,
                            error_text or f"upstream returned HTTP {response.status_code}",
                        )
                        raise make_http_error(response)
                    reasoning_content = ""
                    tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get(
                            "content") or choice.get("text") or ""

                        delta_tool_calls = delta.get("tool_calls")
                        if isinstance(delta_tool_calls, list):
                            for tool_call in delta_tool_calls:
                                if not isinstance(tool_call, dict):
                                    continue
                                index_raw = tool_call.get("index")
                                try:
                                    index = int(index_raw)
                                except (TypeError, ValueError):
                                    index = len(tool_calls_by_index)

                                current = tool_calls_by_index.get(index)
                                if current is None:
                                    current = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                    tool_calls_by_index[index] = current

                                tool_id = tool_call.get("id")
                                if isinstance(tool_id, str) and tool_id:
                                    current["id"] = tool_id

                                tool_type = tool_call.get("type")
                                if isinstance(tool_type, str) and tool_type:
                                    current["type"] = tool_type

                                func = tool_call.get("function")
                                if isinstance(func, dict):
                                    func_name = func.get("name")
                                    if isinstance(func_name, str) and func_name:
                                        current["function"]["name"] = func_name
                                    func_args = func.get("arguments")
                                    if isinstance(func_args, str) and func_args:
                                        current["function"]["arguments"] += func_args

                        chunk_reasoning = delta.get("reasoning_content")
                        if not chunk_reasoning:
                            message = choice.get("message", {})
                            if isinstance(message, dict):
                                chunk_reasoning = message.get(
                                    "reasoning_content")
                        if isinstance(chunk_reasoning, str) and chunk_reasoning:
                            reasoning_content += chunk_reasoning
                        finish_reason = choice.get("finish_reason")
                        done = finish_reason is not None
                        message: Dict[str, Any] = {
                            "role": "assistant", "content": content}
                        if done and reasoning_content:
                            message["reasoning_content"] = reasoning_content
                        if done and tool_calls_by_index:
                            ordered_tool_calls = []
                            for index in sorted(tool_calls_by_index.keys()):
                                tool_call = tool_calls_by_index[index]
                                function_block = tool_call.get(
                                    "function") or {}
                                if not any([
                                    tool_call.get("id"),
                                    function_block.get("name"),
                                    function_block.get("arguments"),
                                ]):
                                    continue
                                ordered_tool_calls.append(tool_call)
                            if ordered_tool_calls:
                                message["tool_calls"] = ordered_tool_calls
                        obj: Dict[str, Any] = {
                            "model": model,
                            "created_at": now_iso(),
                            "message": message,
                            "done": done,
                        }
                        if done:
                            usage = chunk.get("usage") or {}
                            obj["done_reason"] = _normalize_done_reason(
                                finish_reason)
                            obj["total_duration"] = 0
                            obj["load_duration"] = 0
                            obj["prompt_eval_count"] = usage.get(
                                "prompt_tokens", 0)
                            obj["prompt_eval_duration"] = 0
                            obj["eval_count"] = usage.get(
                                "completion_tokens", 0)
                            obj["eval_duration"] = 0
                        emitted_any_chunk = True
                        yield json.dumps(obj, ensure_ascii=False) + "\n"
        except (httpx.HTTPError, httpx.StreamError) as exc:
            logger.warning(
                "openai stream upstream connection failure source=%s model=%s emitted=%s stats=%s error=%s",
                source_cfg.get("name"),
                model,
                emitted_any_chunk,
                payload_stats,
                exc,
            )
            raise

    async def stream_generate_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        url = source_url(source_cfg, "chat_completions")
        payload_stats = _openai_payload_stats(payload)
        emitted_any_chunk = False
        try:
            async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
                async with client.stream("POST", url, headers=source_headers(source_cfg), json=payload) as response:
                    if response.status_code >= 400:
                        error_text = (await response.aread()).decode("utf-8", errors="replace").strip()
                        logger.warning(
                            "openai generate upstream error source=%s model=%s status=%s stats=%s error=%s",
                            source_cfg.get("name"),
                            model,
                            response.status_code,
                            payload_stats,
                            error_text or f"upstream returned HTTP {response.status_code}",
                        )
                        raise make_http_error(response)
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        choice = chunk.get("choices", [{}])[0]
                        content = choice.get("delta", {}).get(
                            "content") or choice.get("text") or ""
                        finish_reason = choice.get("finish_reason")
                        done = finish_reason is not None
                        obj: Dict[str, Any] = {
                            "model": model,
                            "created_at": now_iso(),
                            "response": content,
                            "done": done,
                        }
                        if done:
                            usage = chunk.get("usage") or {}
                            obj["done_reason"] = _normalize_done_reason(
                                finish_reason)
                            obj["total_duration"] = 0
                            obj["load_duration"] = 0
                            obj["prompt_eval_count"] = usage.get(
                                "prompt_tokens", 0)
                            obj["prompt_eval_duration"] = 0
                            obj["eval_count"] = usage.get(
                                "completion_tokens", 0)
                            obj["eval_duration"] = 0
                        emitted_any_chunk = True
                        yield json.dumps(obj, ensure_ascii=False) + "\n"
        except (httpx.HTTPError, httpx.StreamError) as exc:
            logger.warning(
                "openai generate upstream connection failure source=%s model=%s emitted=%s stats=%s error=%s",
                source_cfg.get("name"),
                model,
                emitted_any_chunk,
                payload_stats,
                exc,
            )
            raise

    async def stream_chat_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        url = source_url(source_cfg, "chat_completions")
        payload_stats = _openai_payload_stats(payload)
        messages = payload.get("messages", [])

        # 验证消息序列
        if isinstance(messages, list):
            _validate_tool_messages(messages)

        emitted_any_chunk = False
        message_id = f"chatcmpl-openai-{int(time.time() * 1000)}"
        created = int(time.time())

        def _extract_exception_details(exc: Exception) -> str:
            """提取异常的详细信息，包括类型、errno等"""
            exc_type = type(exc).__name__

            # 优先提取特定的错误信息
            if hasattr(exc, 'errno') and exc.errno:
                return f"{exc_type} (errno={exc.errno})"

            # 尝试获取args中的信息
            if hasattr(exc, 'args') and exc.args:
                msg = str(exc.args[0]) if exc.args else ""
                if msg:
                    return f"{exc_type}: {msg[:300]}"

            # 尝试获取原始异常
            if hasattr(exc, '__cause__') and exc.__cause__:
                cause_type = type(exc.__cause__).__name__
                cause_msg = str(exc.__cause__)
                if cause_msg:
                    return f"{exc_type} (caused by {cause_type}: {cause_msg[:250]})"
                return f"{exc_type} (caused by {cause_type})"

            # 最后尝试str()
            exc_str = str(exc)
            if exc_str:
                return f"{exc_type}: {exc_str[:300]}"

            # 默认值
            return exc_type

        def _safe_error_text(raw: str) -> str:
            text = (raw or "").strip()
            if not text:
                return "upstream request failed"
            return text[:400]

        async def _emit_error_sse(error_text: str) -> AsyncIterator[bytes]:
            # Keep SSE shape valid for Copilot clients: at least one chunk with choices.
            chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": display_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": f"[upstream_error] {error_text}"},
                        "finish_reason": None,
                    }
                ],
            }
            yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
            done_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": display_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield ("data: " + json.dumps(done_chunk, ensure_ascii=False) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

        max_retries = 2
        retry_backoff_seconds = (0.3, 0.8)
        attempt = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
                    async with client.stream("POST", url, headers=source_headers(source_cfg), json=payload) as response:
                        if response.status_code >= 400:
                            error_text = (await response.aread()).decode("utf-8", errors="replace").strip()
                            logger.warning(
                                "openai sse upstream error source=%s model=%s status=%s stats=%s error=%s",
                                source_cfg.get("name"),
                                display_model,
                                response.status_code,
                                payload_stats,
                                error_text or f"upstream returned HTTP {response.status_code}",
                            )
                            async for item in _emit_error_sse(_safe_error_text(error_text)):
                                yield item
                            return
                        async for line in response.aiter_lines():
                            if not line:
                                emitted_any_chunk = True
                                yield b"\n"
                                continue
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    chunk["model"] = display_model
                                    emitted_any_chunk = True
                                    yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
                                    continue
                                except Exception:
                                    pass
                            emitted_any_chunk = True
                            yield (line + "\n\n").encode()
                return
            except (httpx.HTTPError, httpx.StreamError) as exc:
                exc_details = _extract_exception_details(exc)
                should_retry = (
                    isinstance(exc, httpx.RemoteProtocolError)
                    and not emitted_any_chunk
                    and attempt < max_retries
                )
                logger.warning(
                    "openai sse upstream connection failure source=%s model=%s emitted=%s attempt=%s max_retries=%s stats=%s error=%s error_details=%s request_id=%s created=%s",
                    source_cfg.get("name"),
                    display_model,
                    emitted_any_chunk,
                    attempt + 1,
                    max_retries,
                    payload_stats,
                    exc,
                    exc_details,
                    message_id,
                    created,
                )
                if should_retry:
                    backoff = retry_backoff_seconds[min(
                        attempt, len(retry_backoff_seconds) - 1)]
                    logger.info(
                        "openai sse retrying after remote protocol disconnect source=%s model=%s attempt=%s backoff_seconds=%.2f request_id=%s",
                        source_cfg.get("name"),
                        display_model,
                        attempt + 1,
                        backoff,
                        message_id,
                    )
                    attempt += 1
                    await asyncio.sleep(backoff)
                    continue
                if emitted_any_chunk:
                    # Upstream may drop mid-stream and never send terminal chunk.
                    # Emit a synthetic stop + [DONE] so Copilot can finalize choices.
                    done_chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": display_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield ("data: " + json.dumps(done_chunk, ensure_ascii=False) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                    return
                async for item in _emit_error_sse(_safe_error_text(exc_details)):
                    yield item
                return

    async def chat_json_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = await self.proxy_json(source_cfg, "chat_completions", payload)
        data["model"] = display_model
        return data

    async def list_models(self, source_cfg: SourceConfig) -> list[Dict[str, Any]]:
        url = source_url(source_cfg, "models")
        try:
            async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
                response = await client.get(url, headers=source_headers(source_cfg))
                if response.status_code >= 400:
                    error_text = response.text.strip(
                    ) or f"upstream returned HTTP {response.status_code}"
                    logger.warning(
                        "openai models upstream error source=%s status=%s error=%s",
                        source_cfg.get("name"),
                        response.status_code,
                        error_text,
                    )
                    raise make_http_error(response)
        except httpx.HTTPError as exc:
            logger.warning(
                "openai models upstream connection failure source=%s error=%s",
                source_cfg.get("name"),
                exc,
            )
            raise

        data = response.json()
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise RuntimeError("upstream /models response must contain a list")

        models: list[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            models.append({
                "name": model_id,
                "upstream_model": model_id,
                "meta": _extract_upstream_model_meta(item),
            })
        return models
