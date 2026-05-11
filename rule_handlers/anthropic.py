from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

import httpx

from .base import RuleHandler, SourceConfig, make_http_error, source_headers, source_url

RULE_ANTHROPIC = "anthropic"
logger = logging.getLogger("ollama_proxy.anthropic")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds") + "Z"


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

    model_info = item.get("model_info")
    if isinstance(model_info, dict):
        meta["model_info"] = dict(model_info)

    return meta


def anthropic_stop_reason(stop_reason: str | None) -> str:
    if stop_reason in {"tool_use", "pause_turn"}:
        return "tool_calls"
    return "stop"


def openai_finish_reason(stop_reason: str | None) -> str:
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "".join(parts)


def parse_tool_input(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"value": text}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {"value": raw}


def stringify_tool_arguments(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False)


def text_content_blocks(content: Any) -> list[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []

    blocks: list[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "text":
            text = str(item.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif item_type in {"tool_use", "tool_result", "thinking", "redacted_thinking"}:
            blocks.append(dict(item))
    return blocks


def openai_tool_call_to_anthropic_block(tool_call: Any, fallback_id: str) -> Dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    tool_id = str(tool_call.get("id") or fallback_id).strip() or fallback_id
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": parse_tool_input(function.get("arguments")),
    }


def split_system_and_messages(messages: Any) -> tuple[str, list[Dict[str, Any]]]:
    if not isinstance(messages, list):
        return "", []

    def previous_assistant_tool_use_ids() -> set[str]:
        if not anthropic_messages:
            return set()
        previous = anthropic_messages[-1]
        if not isinstance(previous, dict) or str(previous.get("role") or "") != "assistant":
            return set()
        content_blocks = previous.get("content")
        if not isinstance(content_blocks, list):
            return set()
        ids: set[str] = set()
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "") != "tool_use":
                continue
            tool_use_id = str(block.get("id") or "").strip()
            if tool_use_id:
                ids.add(tool_use_id)
        return ids

    def dedupe_tool_use_blocks(blocks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        seen_tool_use_ids: set[str] = set()
        deduped: list[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "") != "tool_use":
                deduped.append(block)
                continue
            tool_use_id = str(block.get("id") or "").strip()
            if not tool_use_id:
                deduped.append(block)
                continue
            if tool_use_id in seen_tool_use_ids:
                continue
            seen_tool_use_ids.add(tool_use_id)
            deduped.append(block)
        return deduped

    def collect_following_tool_result_ids(start_index: int) -> set[str]:
        tool_result_ids: set[str] = set()
        index = start_index + 1
        while index < len(messages):
            following = messages[index]
            if not isinstance(following, dict):
                index += 1
                continue
            if str(following.get("role") or "user") != "tool":
                break
            tool_result_id = str(following.get("tool_call_id") or "").strip()
            if tool_result_id:
                tool_result_ids.add(tool_result_id)
            index += 1
        return tool_result_ids

    def filter_assistant_tool_use_blocks(
        blocks: list[Dict[str, Any]],
        allowed_tool_ids: set[str],
    ) -> list[Dict[str, Any]]:
        filtered: list[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "") != "tool_use":
                filtered.append(block)
                continue
            block_id = str(block.get("id") or "").strip()
            if block_id and block_id in allowed_tool_ids:
                filtered.append(block)
        return filtered

    system_parts: list[str] = []
    anthropic_messages: list[Dict[str, Any]] = []
    tool_counter = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            index += 1
            continue

        role = str(message.get("role") or "user")
        content = message.get("content")

        if role == "system":
            text = extract_text_from_content(content)
            if text:
                system_parts.append(text)
            index += 1
            continue

        if role == "tool":
            # Anthropic requires tool_result blocks to be grouped in the immediate next user message.
            allowed_tool_result_ids = previous_assistant_tool_use_ids()
            tool_result_blocks: list[Dict[str, Any]] = []
            while index < len(messages):
                tool_message = messages[index]
                if not isinstance(tool_message, dict):
                    index += 1
                    continue
                if str(tool_message.get("role") or "user") != "tool":
                    break

                tool_use_id = str(tool_message.get("tool_call_id") or "").strip()
                if tool_use_id and tool_use_id in allowed_tool_result_ids:
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": extract_text_from_content(tool_message.get("content")),
                        }
                    )
                index += 1

            if tool_result_blocks:
                anthropic_messages.append({"role": "user", "content": tool_result_blocks})
            continue

        blocks = text_content_blocks(content)
        if role == "assistant":
            following_tool_result_ids = collect_following_tool_result_ids(index)
            blocks = filter_assistant_tool_use_blocks(blocks, following_tool_result_ids)
            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    tool_block = openai_tool_call_to_anthropic_block(tool_call, f"toolu_{tool_counter}")
                    tool_counter += 1
                    if not tool_block:
                        continue
                    if not following_tool_result_ids:
                        continue
                    if str(tool_block.get("id") or "").strip() not in following_tool_result_ids:
                        continue
                    blocks.append(tool_block)

        if not blocks:
            text = extract_text_from_content(content)
            if text:
                blocks.append({"type": "text", "text": text})
        if not blocks:
            continue

        if role == "assistant":
            blocks = dedupe_tool_use_blocks(blocks)
            if not blocks:
                continue

        anthropic_role = "assistant" if role == "assistant" else "user"
        anthropic_messages.append({"role": anthropic_role, "content": blocks})
        index += 1

    return "\n\n".join(part for part in system_parts if part), anthropic_messages


def build_anthropic_tools(tools: Any) -> list[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    anthropic_tools: list[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if str(tool.get("type") or "function") != "function":
            continue
        function = tool.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        input_schema = function.get("parameters")
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        anthropic_tool: Dict[str, Any] = {
            "name": name,
            "input_schema": input_schema,
        }
        description = str(function.get("description") or "").strip()
        if description:
            anthropic_tool["description"] = description
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools


def build_anthropic_tool_choice(tool_choice: Any) -> Dict[str, Any] | None:
    if tool_choice is None or tool_choice == "" or tool_choice == "auto":
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, str):
        lowered = tool_choice.strip().lower()
        if lowered in {"auto", ""}:
            return None
        if lowered == "required":
            return {"type": "any"}
        if lowered == "any":
            return {"type": "any"}
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "").strip().lower()
        if choice_type in {"auto", "any"}:
            return {"type": choice_type}
        if choice_type == "function":
            function = tool_choice.get("function") or {}
            if isinstance(function, dict):
                name = str(function.get("name") or "").strip()
                if name:
                    return {"type": "tool", "name": name}
        if choice_type == "tool":
            name = str(tool_choice.get("name") or "").strip()
            if name:
                return {"type": "tool", "name": name}
    return None


def anthropic_message_to_openai_message(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts: list[str] = []
    tool_calls: list[Dict[str, Any]] = []
    if isinstance(content, list):
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
                continue
            if block_type != "tool_use":
                continue
            name = str(block.get("name") or "").strip()
            if not name:
                continue
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{index}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": stringify_tool_arguments(block.get("input") or {}),
                    },
                }
            )

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def build_anthropic_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    system_text, messages = split_system_and_messages(payload.get("messages"))
    if not messages:
        fallback_text = "Please continue."
        raw_messages = payload.get("messages")
        if isinstance(raw_messages, list):
            for raw_message in reversed(raw_messages):
                if not isinstance(raw_message, dict):
                    continue
                if str(raw_message.get("role") or "") != "user":
                    continue
                candidate = extract_text_from_content(raw_message.get("content")).strip()
                if candidate:
                    fallback_text = candidate
                    break
        messages = [{"role": "user", "content": [{"type": "text", "text": fallback_text}]}]

    anthropic_payload: Dict[str, Any] = {
        "model": payload["model"],
        "messages": messages,
        "max_tokens": int(payload.get("max_tokens") or payload.get("num_predict") or 2048),
        "stream": bool(payload.get("stream", False)),
    }
    if system_text:
        anthropic_payload["system"] = system_text
    for key in ["temperature", "top_p", "top_k", "stop", "metadata"]:
        if key in payload:
            anthropic_payload[key] = payload[key]

    anthropic_tools = build_anthropic_tools(payload.get("tools"))
    if anthropic_tools and payload.get("tool_choice") != "none":
        anthropic_payload["tools"] = anthropic_tools
        tool_choice = build_anthropic_tool_choice(payload.get("tool_choice"))
        if tool_choice:
            anthropic_payload["tool_choice"] = tool_choice

    if "stop" in anthropic_payload and "stop_sequences" not in anthropic_payload:
        stop_value = anthropic_payload.pop("stop")
        if isinstance(stop_value, list):
            anthropic_payload["stop_sequences"] = stop_value
        elif stop_value:
            anthropic_payload["stop_sequences"] = [stop_value]
    return anthropic_payload


def anthropic_text_and_usage(data: Dict[str, Any]) -> tuple[str, Dict[str, Any], str | None]:
    text = extract_text_from_content(data.get("content"))
    usage = data.get("usage") or {}
    return text, usage, data.get("stop_reason")


def _anthropic_payload_stats(anthropic_payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = anthropic_payload.get("messages")
    if not isinstance(messages, list):
        messages = []

    role_counts = {"user": 0, "assistant": 0, "other": 0}
    tool_use_count = 0
    tool_result_count = 0
    thinking_count = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role in {"user", "assistant"}:
            role_counts[role] += 1
        else:
            role_counts["other"] += 1

        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "tool_use":
                tool_use_count += 1
            elif block_type == "tool_result":
                tool_result_count += 1
            elif block_type in {"thinking", "redacted_thinking"}:
                thinking_count += 1

    return {
        "message_count": len(messages),
        "role_counts": role_counts,
        "tool_use_blocks": tool_use_count,
        "tool_result_blocks": tool_result_count,
        "thinking_blocks": thinking_count,
    }


class AnthropicRuleHandler(RuleHandler):
    rules = (RULE_ANTHROPIC,)
    default_paths = {
        "chat_completions": "/messages",
        "embeddings": "/embeddings",
        "models": "/models",
    }
    default_rule_config = {
        "supports_embeddings": False,
        "supports_tools": True,
        "use_bearer_auth": False,
        "request_headers": {
            "x-api-key": "{{api_key}}",
            "anthropic-version": "2023-06-01",
        },
    }

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return True

    def _inject_cached_thinking(self, source_cfg: SourceConfig, payload: Dict[str, Any]) -> None:
        return

    def _remember_cached_thinking(self, source_cfg: SourceConfig, model: str, blocks: Any) -> None:
        return

    async def proxy_json(self, source_cfg: SourceConfig, path_key: str, body: Dict[str, Any]) -> Any:
        if path_key != "chat_completions":
            raise make_http_error(
                httpx.Response(status_code=501, request=httpx.Request("POST", source_url(source_cfg, path_key)), text="unsupported path")
            )
        self._inject_cached_thinking(source_cfg, body)
        anthropic_payload = build_anthropic_payload(body)
        url = source_url(source_cfg, path_key)
        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            response = await client.post(url, headers=source_headers(source_cfg), json=anthropic_payload)
            if response.status_code >= 400:
                raise make_http_error(response)
            data = response.json()
            self._remember_cached_thinking(source_cfg, str(body.get("model") or ""), data.get("content"), body.get("messages"))
            return data

    async def stream_chat_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        self._inject_cached_thinking(source_cfg, payload)
        anthropic_payload = build_anthropic_payload(payload)
        anthropic_payload["stream"] = True
        url = source_url(source_cfg, "chat_completions")
        prompt_tokens = 0
        completion_tokens = 0
        stop_reason: str | None = None
        tool_buffers: Dict[int, Dict[str, Any]] = {}
        content_blocks: Dict[int, Dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            async with client.stream("POST", url, headers=source_headers(source_cfg), json=anthropic_payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise make_http_error(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_text = line[6:].strip()
                    if not data_text:
                        continue
                    try:
                        chunk = json.loads(data_text)
                    except Exception:
                        continue
                    chunk_type = chunk.get("type")
                    if chunk_type == "message_start":
                        usage = chunk.get("message", {}).get("usage") or {}
                        prompt_tokens = usage.get("input_tokens", prompt_tokens)
                    elif chunk_type == "content_block_start":
                        block_index = int(chunk.get("index") or 0)
                        block = chunk.get("content_block") or {}
                        if isinstance(block, dict):
                            content_blocks[block_index] = dict(block)
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            raw_input = block.get("input")
                            tool_buffers[block_index] = {
                                "id": str(block.get("id") or f"toolu_{block_index}"),
                                "name": str(block.get("name") or ""),
                                "input_text": ""
                                if raw_input is None or raw_input == {} or raw_input == ""
                                else stringify_tool_arguments(raw_input),
                            }
                    elif chunk_type == "content_block_delta":
                        block_index = int(chunk.get("index") or 0)
                        delta = chunk.get("delta") or {}
                        block = content_blocks.get(block_index)
                        if isinstance(block, dict):
                            block_type = str(block.get("type") or "")
                            delta_type = str(delta.get("type") or "")
                            if block_type == "text" and delta_type == "text_delta":
                                block["text"] = str(block.get("text") or "") + str(delta.get("text") or "")
                            elif block_type in {"thinking", "redacted_thinking"}:
                                if "thinking" in delta:
                                    key = "thinking"
                                    block[key] = str(block.get(key) or "") + str(delta.get("thinking") or "")
                                if "data" in delta:
                                    key = "data"
                                    block[key] = str(block.get(key) or "") + str(delta.get("data") or "")
                        if delta.get("type") == "text_delta":
                            content = delta.get("text") or ""
                            yield json.dumps(
                                {
                                    "model": model,
                                    "created_at": now_iso(),
                                    "message": {"role": "assistant", "content": content},
                                    "done": False,
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        elif delta.get("type") == "input_json_delta" and block_index in tool_buffers:
                            tool_buffers[block_index]["input_text"] += str(delta.get("partial_json") or "")
                    elif chunk_type == "content_block_stop":
                        block_index = int(chunk.get("index") or 0)
                        tool_buffer = tool_buffers.pop(block_index, None)
                        if tool_buffer and tool_buffer.get("name"):
                            yield json.dumps(
                                {
                                    "model": model,
                                    "created_at": now_iso(),
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "function": {
                                                    "name": tool_buffer["name"],
                                                    "arguments": parse_tool_input(tool_buffer.get("input_text")),
                                                }
                                            }
                                        ],
                                    },
                                    "done": False,
                                },
                                ensure_ascii=False,
                            ) + "\n"
                    elif chunk_type == "message_delta":
                        delta = chunk.get("delta") or {}
                        usage = chunk.get("usage") or {}
                        completion_tokens = usage.get("output_tokens", completion_tokens)
                        stop_reason = delta.get("stop_reason", stop_reason)
                    elif chunk_type == "message_stop":
                        ordered_blocks = [content_blocks[idx] for idx in sorted(content_blocks)]
                        self._remember_cached_thinking(source_cfg, str(payload.get("model") or ""), ordered_blocks, payload.get("messages"))
                        yield json.dumps(
                            {
                                "model": model,
                                "created_at": now_iso(),
                                "message": {"role": "assistant", "content": ""},
                                "done": True,
                                "done_reason": anthropic_stop_reason(stop_reason),
                                "total_duration": 0,
                                "load_duration": 0,
                                "prompt_eval_count": prompt_tokens,
                                "prompt_eval_duration": 0,
                                "eval_count": completion_tokens,
                                "eval_duration": 0,
                            },
                            ensure_ascii=False,
                        ) + "\n"

    async def stream_generate_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        self._inject_cached_thinking(source_cfg, payload)
        anthropic_payload = build_anthropic_payload(payload)
        anthropic_payload["stream"] = True
        url = source_url(source_cfg, "chat_completions")
        prompt_tokens = 0
        completion_tokens = 0
        stop_reason: str | None = None

        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            async with client.stream("POST", url, headers=source_headers(source_cfg), json=anthropic_payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise make_http_error(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_text = line[6:].strip()
                    if not data_text:
                        continue
                    try:
                        chunk = json.loads(data_text)
                    except Exception:
                        continue
                    chunk_type = chunk.get("type")
                    if chunk_type == "message_start":
                        usage = chunk.get("message", {}).get("usage") or {}
                        prompt_tokens = usage.get("input_tokens", prompt_tokens)
                    elif chunk_type == "content_block_delta":
                        delta = chunk.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            content = delta.get("text") or ""
                            yield json.dumps(
                                {
                                    "model": model,
                                    "created_at": now_iso(),
                                    "response": content,
                                    "done": False,
                                },
                                ensure_ascii=False,
                            ) + "\n"
                    elif chunk_type == "message_delta":
                        delta = chunk.get("delta") or {}
                        usage = chunk.get("usage") or {}
                        completion_tokens = usage.get("output_tokens", completion_tokens)
                        stop_reason = delta.get("stop_reason", stop_reason)
                    elif chunk_type == "message_stop":
                        yield json.dumps(
                            {
                                "model": model,
                                "created_at": now_iso(),
                                "response": "",
                                "done": True,
                                "done_reason": anthropic_stop_reason(stop_reason),
                                "total_duration": 0,
                                "load_duration": 0,
                                "prompt_eval_count": prompt_tokens,
                                "prompt_eval_duration": 0,
                                "eval_count": completion_tokens,
                                "eval_duration": 0,
                            },
                            ensure_ascii=False,
                        ) + "\n"

    async def stream_chat_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        self._inject_cached_thinking(source_cfg, payload)
        anthropic_payload = build_anthropic_payload(payload)
        anthropic_payload["stream"] = True
        payload_stats = _anthropic_payload_stats(anthropic_payload)
        url = source_url(source_cfg, "chat_completions")
        message_id = f"chatcmpl-anthropic-{int(time.time() * 1000)}"
        created = int(time.time())
        max_attempts = 2

        async with httpx.AsyncClient(
            timeout=source_cfg["timeout"],
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=0),
        ) as client:
            for attempt in range(1, max_attempts + 1):
                finish_reason: str | None = None
                tool_call_indexes: Dict[int, int] = {}
                content_blocks: Dict[int, Dict[str, Any]] = {}
                emitted_any_chunk = False
                stream_finished = False
                try:
                    async with client.stream("POST", url, headers=source_headers(source_cfg), json=anthropic_payload) as response:
                        if response.status_code >= 400:
                            error_text = (await response.aread()).decode("utf-8", errors="replace")
                            safe_error_text = error_text.strip() or f"upstream returned HTTP {response.status_code}"
                            logger.warning(
                                "anthropic stream upstream error source=%s model=%s status=%s stats=%s error=%s",
                                source_cfg.get("name"),
                                display_model,
                                response.status_code,
                                payload_stats,
                                safe_error_text,
                            )
                            emitted_any_chunk = True
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "id": message_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": display_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant"},
                                                "finish_reason": None,
                                            }
                                        ],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            ).encode()
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "id": message_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": display_model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            ).encode()
                            yield b"data: [DONE]\n\n"
                            return

                        emitted_any_chunk = True
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": display_model,
                                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        ).encode()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_text = line[6:].strip()
                            if not data_text:
                                continue
                            try:
                                chunk = json.loads(data_text)
                            except Exception:
                                continue
                            chunk_type = chunk.get("type")
                            if chunk_type == "content_block_start":
                                block_index = int(chunk.get("index") or 0)
                                block = chunk.get("content_block") or {}
                                if isinstance(block, dict):
                                    content_blocks[block_index] = dict(block)
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    tool_index = len(tool_call_indexes)
                                    tool_call_indexes[block_index] = tool_index
                                    delta_payload = {
                                        "tool_calls": [
                                            {
                                                "index": tool_index,
                                                "id": str(block.get("id") or f"toolu_{tool_index}"),
                                                "type": "function",
                                                "function": {
                                                    "name": str(block.get("name") or ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    }
                                    emitted_any_chunk = True
                                    yield (
                                        "data: "
                                        + json.dumps(
                                            {
                                                "id": message_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": display_model,
                                                "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n\n"
                                    ).encode()
                            elif chunk_type == "content_block_delta":
                                block_index = int(chunk.get("index") or 0)
                                delta = chunk.get("delta") or {}
                                block = content_blocks.get(block_index)
                                if isinstance(block, dict):
                                    block_type = str(block.get("type") or "")
                                    delta_type = str(delta.get("type") or "")
                                    if block_type == "text" and delta_type == "text_delta":
                                        block["text"] = str(block.get("text") or "") + str(delta.get("text") or "")
                                    elif block_type in {"thinking", "redacted_thinking"}:
                                        if "thinking" in delta:
                                            key = "thinking"
                                            block[key] = str(block.get(key) or "") + str(delta.get("thinking") or "")
                                        if "data" in delta:
                                            key = "data"
                                            block[key] = str(block.get(key) or "") + str(delta.get("data") or "")
                                if delta.get("type") == "text_delta":
                                    content = delta.get("text") or ""
                                    emitted_any_chunk = True
                                    yield (
                                        "data: "
                                        + json.dumps(
                                            {
                                                "id": message_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": display_model,
                                                "choices": [
                                                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                                                ],
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n\n"
                                    ).encode()
                                elif delta.get("type") == "input_json_delta" and block_index in tool_call_indexes:
                                    emitted_any_chunk = True
                                    yield (
                                        "data: "
                                        + json.dumps(
                                            {
                                                "id": message_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": display_model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {
                                                            "tool_calls": [
                                                                {
                                                                    "index": tool_call_indexes[block_index],
                                                                    "function": {
                                                                        "arguments": str(delta.get("partial_json") or ""),
                                                                    },
                                                                }
                                                            ]
                                                        },
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n\n"
                                    ).encode()
                            elif chunk_type == "message_delta":
                                delta = chunk.get("delta") or {}
                                finish_reason = openai_finish_reason(delta.get("stop_reason", finish_reason))
                            elif chunk_type == "message_stop":
                                ordered_blocks = [content_blocks[idx] for idx in sorted(content_blocks)]
                                self._remember_cached_thinking(source_cfg, str(payload.get("model") or ""), ordered_blocks, payload.get("messages"))
                                emitted_any_chunk = True
                                yield (
                                    "data: "
                                    + json.dumps(
                                        {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": display_model,
                                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n\n"
                                ).encode()
                                yield b"data: [DONE]\n\n"
                                stream_finished = True
                                return
                except (httpx.HTTPError, httpx.StreamError) as exc:
                    logger.warning(
                        "anthropic stream upstream connection failure source=%s model=%s attempt=%s/%s emitted=%s stats=%s error=%s",
                        source_cfg.get("name"),
                        display_model,
                        attempt,
                        max_attempts,
                        emitted_any_chunk,
                        payload_stats,
                        exc,
                    )

                    if (not emitted_any_chunk) and attempt < max_attempts:
                        continue

                    if stream_finished:
                        return

                    if emitted_any_chunk:
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": display_model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        ).encode()
                        yield b"data: [DONE]\n\n"
                        return

                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": display_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    ).encode()
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": display_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    ).encode()
                    yield b"data: [DONE]\n\n"
                    return

    async def chat_json_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._inject_cached_thinking(source_cfg, payload)
        data = await self.proxy_json(source_cfg, "chat_completions", payload)
        self._remember_cached_thinking(source_cfg, str(payload.get("model") or ""), data.get("content"), payload.get("messages"))
        text, usage, stop_reason = anthropic_text_and_usage(data)
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        return {
            "id": data.get("id") or f"chatcmpl-anthropic-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": display_model,
            "choices": [
                {
                    "index": 0,
                    "message": anthropic_message_to_openai_message(data.get("content")),
                    "finish_reason": openai_finish_reason(stop_reason),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

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
            models.append(
                {
                    "name": model_id,
                    "upstream_model": model_id,
                    "aliases": [str(item.get("display_name") or "").strip()] if item.get("display_name") else [],
                    "meta": _extract_upstream_model_meta(item),
                }
            )
        return models
