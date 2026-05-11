from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import httpx

from .anthropic import AnthropicRuleHandler
from .base import SourceConfig
from .base import make_http_error, source_headers

RULE_DEEPSEEK_ANTHROPIC = "deepseek_anthropic"
_THINKING_CACHE_LIMIT = 256
_DEEPSEEK_THINKING_CACHE: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
_DEEPSEEK_LAST_TRIM_STATE: Dict[tuple[str, str, str], tuple[int, int, int]] = {}
logger = logging.getLogger("ollama_proxy.deepseek_anthropic")
_DEEPSEEK_CACHE_PATH = Path(__file__).resolve().parents[1] / "Logs" / "deepseek_thinking_cache.json"

_DEEPSEEK_CONTEXT_FALLBACK = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
}


def _get_messages_context_hash(messages: Any) -> str:
    """Build a stable context hash for conversation isolation."""
    if not isinstance(messages, list):
        return "empty"

    last_user_msg = ""
    assistant_count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str):
                last_user_msg = content[:100]  # 取前100字符
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and str(block.get("type") or "") == "text":
                        parts.append(str(block.get("text") or ""))
                last_user_msg = "".join(parts)[:100]
        elif role == "assistant":
            assistant_count += 1

    context_str = f"{last_user_msg}|{assistant_count}"
    return hashlib.sha1(context_str.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _cache_key(source_cfg: SourceConfig, model: str, messages: Any = None) -> tuple[str, str, str]:
    """Cache key: (source_name, model, message_context_hash)."""
    source_name = str(source_cfg.get("name") or "")
    model_name = str(model)
    context_hash = _get_messages_context_hash(messages) if messages else "empty"
    return (source_name, model_name, context_hash)


def _serialize_cache_key(cache_key: tuple[str, str, str]) -> str:
    """Serialize cache key tuple to string."""
    return f"{cache_key[0]}::{cache_key[1]}::{cache_key[2]}"


def _deserialize_cache_key(text: str) -> tuple[str, str, str] | None:
    """Deserialize string to cache key tuple."""
    parts = text.split("::", 2)
    if len(parts) != 3:
        return None
    return (parts[0], parts[1], parts[2])


def _load_persistent_cache() -> None:
    if not _DEEPSEEK_CACHE_PATH.exists():
        return
    try:
        raw = json.loads(_DEEPSEEK_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("deepseek anthropic thinking cache load failed path=%s error=%s", _DEEPSEEK_CACHE_PATH, exc)
        return

    if not isinstance(raw, dict):
        return

    for raw_key, entries in raw.items():
        cache_key = _deserialize_cache_key(str(raw_key))
        if cache_key is None or not isinstance(entries, list):
            continue
        normalized_entries: list[Dict[str, Any]] = []
        for entry in entries[-_THINKING_CACHE_LIMIT:]:
            if not isinstance(entry, dict):
                continue
            assistant_text = str(entry.get("assistant_text") or "")
            content = entry.get("content")
            if not isinstance(content, list):
                continue
            normalized_entries.append({"assistant_text": assistant_text, "content": content})
        if normalized_entries:
            _DEEPSEEK_THINKING_CACHE[cache_key] = normalized_entries


def _save_persistent_cache() -> None:
    _DEEPSEEK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable: Dict[str, Any] = {}
    for cache_key, entries in _DEEPSEEK_THINKING_CACHE.items():
        serializable[_serialize_cache_key(cache_key)] = entries[-_THINKING_CACHE_LIMIT:]

    temp_path = _DEEPSEEK_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(_DEEPSEEK_CACHE_PATH)


def _extract_text_from_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def _has_thinking_block(blocks: Any) -> bool:
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") in {"thinking", "redacted_thinking"}:
            return True
    return False


def _message_has_thinking(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if str(message.get("role") or "") != "assistant":
        return False
    return _has_thinking_block(message.get("content"))


def _assistant_without_thinking_indices(messages: list[Any]) -> list[int]:
    indices: list[int] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "assistant":
            continue
        if not _message_has_thinking(message):
            indices.append(idx)
    return indices


def _trim_unreplayable_history(
    payload: Dict[str, Any],
    source_cfg: SourceConfig,
    model: str,
    unresolved_indices: list[int],
) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not unresolved_indices:
        return

    preserved_system: list[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") == "system":
            preserved_system.append(message)

    cut_from = max(unresolved_indices) + 1
    safe_suffix: list[Dict[str, Any]] = []
    for message in messages[cut_from:]:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") == "assistant" and not _message_has_thinking(message):
            continue
        safe_suffix.append(message)

    if not safe_suffix:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "") == "user":
                safe_suffix = [message]
                break

    payload["messages"] = preserved_system + safe_suffix
    state_key = _cache_key(source_cfg, model, payload.get("messages"))
    trim_state = (len(unresolved_indices), cut_from, len(payload["messages"]))
    if _DEEPSEEK_LAST_TRIM_STATE.get(state_key) != trim_state:
        _DEEPSEEK_LAST_TRIM_STATE[state_key] = trim_state
        logger.info(
            "deepseek anthropic thinking trim history source=%s model=%s unresolved=%s cut_from=%s kept=%s",
            source_cfg.get("name"),
            model,
            trim_state[0],
            trim_state[1],
            trim_state[2],
        )


class DeepSeekAnthropicRuleHandler(AnthropicRuleHandler):
    rules = (RULE_DEEPSEEK_ANTHROPIC,)

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return True

    def _inject_cached_thinking(self, source_cfg: SourceConfig, payload: Dict[str, Any]) -> None:
        model = str(payload.get("model") or "")
        if not model:
            return

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return

        entries = _DEEPSEEK_THINKING_CACHE.get(_cache_key(source_cfg, model, messages))
        if not entries:
            unresolved = _assistant_without_thinking_indices(messages)
            if unresolved:
                _trim_unreplayable_history(payload, source_cfg, model, unresolved)
            return

        normalized_entries: list[Dict[str, Any]] = []
        for entry in entries:
            assistant_text = str(entry.get("assistant_text") or "")
            cached_content = entry.get("content")
            if not isinstance(cached_content, list):
                continue
            normalized_entries.append(
                {
                    "assistant_text": assistant_text,
                    "content": copy.deepcopy(cached_content),
                    "used": False,
                }
            )

        injected_count = 0
        miss_count = 0

        for message in messages:
            if not isinstance(message, dict) or str(message.get("role") or "") != "assistant":
                continue

            content = message.get("content")
            if isinstance(content, list) and _has_thinking_block(content):
                assistant_text = _extract_text_from_blocks(content)
                for entry in normalized_entries:
                    if entry["used"]:
                        continue
                    if entry["assistant_text"] == assistant_text:
                        entry["used"] = True
                        break
                continue

            assistant_text = self._extract_message_text(content)
            match_index = -1

            for idx, entry in enumerate(normalized_entries):
                if entry["used"]:
                    continue
                if entry["assistant_text"] == assistant_text:
                    match_index = idx
                    break

            if match_index < 0:
                for idx, entry in enumerate(normalized_entries):
                    if not entry["used"]:
                        match_index = idx
                        break

            if match_index < 0:
                miss_count += 1
                continue

            chosen = normalized_entries[match_index]
            message["content"] = copy.deepcopy(chosen["content"])
            chosen["used"] = True
            injected_count += 1

        if miss_count:
            unresolved = _assistant_without_thinking_indices(messages)
            if unresolved:
                _trim_unreplayable_history(payload, source_cfg, model, unresolved)

            logger.info(
                "deepseek anthropic thinking inject partial source=%s model=%s injected=%s misses=%s cache_size=%s",
                source_cfg.get("name"),
                model,
                injected_count,
                miss_count,
                len(normalized_entries),
            )

    def _remember_cached_thinking(self, source_cfg: SourceConfig, model: str, blocks: Any, messages: Any = None) -> None:
        if not isinstance(blocks, list) or not _has_thinking_block(blocks):
            return

        assistant_text = _extract_text_from_blocks(blocks)
        cache_key = _cache_key(source_cfg, model, messages)
        entries = _DEEPSEEK_THINKING_CACHE.setdefault(cache_key, [])
        entries.append(
            {
                "assistant_text": assistant_text,
                "content": copy.deepcopy(blocks),
            }
        )
        if len(entries) > _THINKING_CACHE_LIMIT:
            del entries[:-_THINKING_CACHE_LIMIT]
        _save_persistent_cache()

    def _extract_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)

    async def list_models(self, source_cfg: SourceConfig) -> list[Dict[str, Any]]:
        base_url = str(source_cfg.get("base_url") or "").rstrip("/")
        if base_url.endswith("/anthropic"):
            base_url = base_url[: -len("/anthropic")]
        url = f"{base_url}/models"
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
            model_name = str(item.get("id") or item.get("name") or "").strip().lower()
            if not model_name:
                continue
            meta: Dict[str, Any] = {"upstream": dict(item)}
            model_meta = item.get("model_info")
            if isinstance(model_meta, dict):
                meta["model_info"] = dict(model_meta)
            model = {
                "name": model_name,
                "upstream_model": model_name,
                "meta": meta,
            }
            fallback_ctx = _DEEPSEEK_CONTEXT_FALLBACK.get(model_name)
            if fallback_ctx and not meta.get("context_length"):
                meta["context_length"] = fallback_ctx
            models.append(model)
        return models


_load_persistent_cache()