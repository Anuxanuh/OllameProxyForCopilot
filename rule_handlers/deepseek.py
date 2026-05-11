from __future__ import annotations

import copy
import logging
from typing import Any, Dict

import httpx

from .base import make_http_error, source_headers
from .base import SourceConfig
from .anthropic import AnthropicRuleHandler

RULE_DEEPSEEK = "deepseek"
_THINKING_CACHE_LIMIT = 256
_DEEPSEEK_THINKING_CACHE: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
logger = logging.getLogger("ollama_proxy.deepseek")

_DEEPSEEK_CONTEXT_FALLBACK = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
}


def _cache_key(source_cfg: SourceConfig, model: str) -> tuple[str, str]:
    return (str(source_cfg.get("name") or ""), str(model))


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


class DeepSeekRuleHandler(AnthropicRuleHandler):
    rules = (RULE_DEEPSEEK,)

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return True

    def _inject_cached_thinking(self, source_cfg: SourceConfig, payload: Dict[str, Any]) -> None:
        model = str(payload.get("model") or "")
        if not model:
            return

        entries = _DEEPSEEK_THINKING_CACHE.get(_cache_key(source_cfg, model))
        if not entries:
            return

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return

        # Keep stable order, prefer exact text match, and fallback to next unused cache entry.
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
        miss_indices: list[int] = []

        for index, message in enumerate(messages):
            if not isinstance(message, dict) or str(message.get("role") or "") != "assistant":
                continue

            content = message.get("content")
            if isinstance(content, list) and _has_thinking_block(content):
                # If client already contains thinking, consume one matching cached entry to keep order aligned.
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
                miss_indices.append(index)
                continue

            chosen = normalized_entries[match_index]
            message["content"] = copy.deepcopy(chosen["content"])
            chosen["used"] = True
            injected_count += 1

        if miss_count:
            first_replayable_assistant = -1
            for idx, message in enumerate(messages):
                if _message_has_thinking(message):
                    first_replayable_assistant = idx
                    break

            if first_replayable_assistant >= 0 and miss_indices and min(miss_indices) < first_replayable_assistant:
                preserved_prefix: list[Dict[str, Any]] = []
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    if str(message.get("role") or "") == "system":
                        preserved_prefix.append(message)

                trimmed_suffix = messages[first_replayable_assistant:]
                payload["messages"] = preserved_prefix + trimmed_suffix
                logger.warning(
                    "deepseek thinking trim history source=%s model=%s misses=%s cut_before=%s kept=%s",
                    source_cfg.get("name"),
                    model,
                    miss_count,
                    first_replayable_assistant,
                    len(payload["messages"]),
                )

            logger.info(
                "deepseek thinking inject partial source=%s model=%s injected=%s misses=%s cache_size=%s",
                source_cfg.get("name"),
                model,
                injected_count,
                miss_count,
                len(normalized_entries),
            )

    def _remember_cached_thinking(self, source_cfg: SourceConfig, model: str, blocks: Any) -> None:
        if not isinstance(blocks, list) or not _has_thinking_block(blocks):
            return

        assistant_text = _extract_text_from_blocks(blocks)
        cache_key = _cache_key(source_cfg, model)
        entries = _DEEPSEEK_THINKING_CACHE.setdefault(cache_key, [])
        entries.append(
            {
                "assistant_text": assistant_text,
                "content": copy.deepcopy(blocks),
            }
        )
        if len(entries) > _THINKING_CACHE_LIMIT:
            del entries[:-_THINKING_CACHE_LIMIT]

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
            if meta.get("context_length"):
                continue
            fallback_ctx = _DEEPSEEK_CONTEXT_FALLBACK.get(model_name)
            if fallback_ctx:
                meta["context_length"] = fallback_ctx
            models.append(model)
        return models
