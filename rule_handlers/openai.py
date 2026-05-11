from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

import httpx

from .base import RuleHandler, SourceConfig, make_http_error, source_headers, source_url

RULE_OPENAI = "openai"
RULE_OPENAI_COMPATIBLE_PARTIAL = "openai_compatible_partial"


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

    parameter_count = _first_positive_int(item, ("parameter_count", "params", "num_parameters"))
    if parameter_count is not None:
        meta["parameter_count"] = parameter_count

    model_size = _first_positive_int(item, ("size", "model_size", "bytes"))
    if model_size is not None:
        meta["size"] = model_size

    caps = item.get("capabilities")
    capabilities: list[str] = []
    if isinstance(caps, list):
        capabilities = [str(value).strip() for value in caps if str(value).strip()]

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
        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            response = await client.post(url, headers=source_headers(source_cfg), json=body)
            if response.status_code >= 400:
                raise make_http_error(response)
            return response.json()

    async def stream_chat_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        url = source_url(source_cfg, "chat_completions")
        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            async with client.stream("POST", url, headers=source_headers(source_cfg), json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
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
                    content = delta.get("content") or choice.get("text") or ""

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
                            chunk_reasoning = message.get("reasoning_content")
                    if isinstance(chunk_reasoning, str) and chunk_reasoning:
                        reasoning_content += chunk_reasoning
                    finish_reason = choice.get("finish_reason")
                    done = finish_reason is not None
                    message: Dict[str, Any] = {"role": "assistant", "content": content}
                    if reasoning_content:
                        message["reasoning_content"] = reasoning_content
                    if done and tool_calls_by_index:
                        ordered_tool_calls = []
                        for index in sorted(tool_calls_by_index.keys()):
                            tool_call = tool_calls_by_index[index]
                            function_block = tool_call.get("function") or {}
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
                        obj["done_reason"] = _normalize_done_reason(finish_reason)
                        obj["total_duration"] = 0
                        obj["load_duration"] = 0
                        obj["prompt_eval_count"] = usage.get("prompt_tokens", 0)
                        obj["prompt_eval_duration"] = 0
                        obj["eval_count"] = usage.get("completion_tokens", 0)
                        obj["eval_duration"] = 0
                    yield json.dumps(obj, ensure_ascii=False) + "\n"

    async def stream_generate_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        url = source_url(source_cfg, "chat_completions")
        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            async with client.stream("POST", url, headers=source_headers(source_cfg), json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
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
                    content = choice.get("delta", {}).get("content") or choice.get("text") or ""
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
                        obj["done_reason"] = _normalize_done_reason(finish_reason)
                        obj["total_duration"] = 0
                        obj["load_duration"] = 0
                        obj["prompt_eval_count"] = usage.get("prompt_tokens", 0)
                        obj["prompt_eval_duration"] = 0
                        obj["eval_count"] = usage.get("completion_tokens", 0)
                        obj["eval_duration"] = 0
                    yield json.dumps(obj, ensure_ascii=False) + "\n"

    async def stream_chat_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        url = source_url(source_cfg, "chat_completions")
        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            async with client.stream("POST", url, headers=source_headers(source_cfg), json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise make_http_error(response)
                async for line in response.aiter_lines():
                    if not line:
                        yield b"\n"
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            chunk["model"] = display_model
                            yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
                            continue
                        except Exception:
                            pass
                    yield (line + "\n\n").encode()

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
        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            response = await client.get(url, headers=source_headers(source_cfg))
            if response.status_code >= 400:
                raise make_http_error(response)

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
