from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import httpx

from .base import SourceConfig, make_http_error, source_headers
from .openai import OpenAIRuleHandler, _extract_upstream_model_meta

RULE_DEEPSEEK_OPENAI = "deepseek_openai"
logger = logging.getLogger("ollama_proxy.deepseek_openai")
_REASONING_CACHE_LIMIT = 256
_DEEPSEEK_OPENAI_REASONING_CACHE: Dict[tuple[str,
                                             str, str], list[Dict[str, str]]] = {}
_DEEPSEEK_OPENAI_LAST_TRIM_STATE: Dict[tuple[str,
                                             str, str], tuple[int, int, int]] = {}
_DEEPSEEK_OPENAI_CACHE_PATH = Path(__file__).resolve(
).parents[1] / "Logs" / "deepseek_openai_reasoning_cache.json"

_DEEPSEEK_OPENAI_CONTEXT_FALLBACK = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
}


def _extract_message_text(content: Any) -> str:
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


def _thread_key(messages: Any) -> str:
    if not isinstance(messages, list):
        return "empty"

    first_system = ""
    first_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content_text = _extract_message_text(message.get("content"))
        if role == "system" and not first_system:
            first_system = content_text[:200]
        elif role == "user" and not first_user:
            first_user = content_text[:200]
        if first_system and first_user:
            break

    seed = first_system + "|" + first_user
    return hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _cache_key(source_cfg: SourceConfig, model: str, messages: Any) -> tuple[str, str, str]:
    return (str(source_cfg.get("name") or ""), str(model), _thread_key(messages))


def _serialize_cache_key(cache_key: tuple[str, str, str]) -> str:
    return f"{cache_key[0]}::{cache_key[1]}::{cache_key[2]}"


def _deserialize_cache_key(text: str) -> tuple[str, str, str] | None:
    parts = text.split("::", 2)
    if len(parts) != 3:
        return None
    return (parts[0], parts[1], parts[2])


def _load_persistent_cache() -> None:
    if not _DEEPSEEK_OPENAI_CACHE_PATH.exists():
        return
    try:
        raw = json.loads(
            _DEEPSEEK_OPENAI_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "deepseek openai reasoning cache load failed path=%s error=%s",
            _DEEPSEEK_OPENAI_CACHE_PATH,
            exc,
        )
        return

    if not isinstance(raw, dict):
        return

    for raw_key, entries in raw.items():
        cache_key = _deserialize_cache_key(str(raw_key))
        if cache_key is None or not isinstance(entries, list):
            continue
        normalized_entries: list[Dict[str, str]] = []
        for entry in entries[-_REASONING_CACHE_LIMIT:]:
            if not isinstance(entry, dict):
                continue
            assistant_text = str(entry.get("assistant_text") or "")
            reasoning_content = str(entry.get("reasoning_content") or "")
            if not reasoning_content:
                continue
            normalized_entries.append(
                {
                    "assistant_text": assistant_text,
                    "reasoning_content": reasoning_content,
                }
            )
        if normalized_entries:
            _DEEPSEEK_OPENAI_REASONING_CACHE[cache_key] = normalized_entries


def _save_persistent_cache() -> None:
    _DEEPSEEK_OPENAI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable: Dict[str, Any] = {}
    for cache_key, entries in _DEEPSEEK_OPENAI_REASONING_CACHE.items():
        serializable[_serialize_cache_key(
            cache_key)] = entries[-_REASONING_CACHE_LIMIT:]

    temp_path = _DEEPSEEK_OPENAI_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(
        serializable, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(_DEEPSEEK_OPENAI_CACHE_PATH)


def _assistant_without_reasoning_indices(messages: list[Any]) -> list[int]:
    indices: list[int] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "assistant":
            continue
        reasoning_content = message.get("reasoning_content")
        if not isinstance(reasoning_content, str) or not reasoning_content:
            indices.append(idx)
    return indices


def _fill_missing_reasoning(
    payload: Dict[str, Any],
    source_cfg: SourceConfig,
    model: str,
    unresolved_indices: list[int],
) -> None:
    """Fill missing reasoning_content with empty string instead of trimming history."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not unresolved_indices:
        return

    for idx in unresolved_indices:
        message = messages[idx]
        if isinstance(message, dict):
            message["reasoning_content"] = ""

    state_key = _cache_key(source_cfg, model, messages)
    fill_state = (len(unresolved_indices),
                  unresolved_indices[0], unresolved_indices[-1])
    if _DEEPSEEK_OPENAI_LAST_TRIM_STATE.get(state_key) != fill_state:
        _DEEPSEEK_OPENAI_LAST_TRIM_STATE[state_key] = fill_state
        logger.info(
            "deepseek openai reasoning fill missing source=%s model=%s thread=%s filled=%s indices=%s",
            source_cfg.get("name"),
            model,
            state_key[2],
            len(unresolved_indices),
            unresolved_indices,
        )


class DeepSeekOpenAIRuleHandler(OpenAIRuleHandler):
    rules = (RULE_DEEPSEEK_OPENAI,)

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return True

    def _prepare_payload(self, source_cfg: SourceConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(payload)
        model = str(normalized.get("model") or "")
        messages = normalized.get("messages")
        if not model or not isinstance(messages, list):
            return normalized

        entries = _DEEPSEEK_OPENAI_REASONING_CACHE.get(
            _cache_key(source_cfg, model, messages))
        if not entries:
            unresolved = _assistant_without_reasoning_indices(messages)
            if unresolved:
                _fill_missing_reasoning(
                    normalized, source_cfg, model, unresolved)
            return normalized

        normalized_entries: list[Dict[str, Any]] = []
        for entry in entries:
            assistant_text = str(entry.get("assistant_text") or "")
            reasoning_content = str(entry.get("reasoning_content") or "")
            if not reasoning_content:
                continue
            normalized_entries.append(
                {
                    "assistant_text": assistant_text,
                    "reasoning_content": reasoning_content,
                    "used": False,
                }
            )

        injected_count = 0
        miss_count = 0
        for message in messages:
            if not isinstance(message, dict) or str(message.get("role") or "") != "assistant":
                continue

            reasoning_content = message.get("reasoning_content")
            assistant_text = _extract_message_text(message.get("content"))
            if isinstance(reasoning_content, str) and reasoning_content:
                for entry in normalized_entries:
                    if entry["used"]:
                        continue
                    if entry["assistant_text"] == assistant_text and entry["reasoning_content"] == reasoning_content:
                        entry["used"] = True
                        break
                continue

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
            message["reasoning_content"] = chosen["reasoning_content"]
            chosen["used"] = True
            injected_count += 1

        if miss_count:
            unresolved = _assistant_without_reasoning_indices(messages)
            if unresolved:
                _fill_missing_reasoning(
                    normalized, source_cfg, model, unresolved)
            logger.info(
                "deepseek openai reasoning inject partial source=%s model=%s thread=%s injected=%s misses=%s cache_size=%s",
                source_cfg.get("name"),
                model,
                _cache_key(source_cfg, model, messages)[2],
                injected_count,
                miss_count,
                len(normalized_entries),
            )

        return normalized

    def _remember_cached_reasoning(
        self,
        source_cfg: SourceConfig,
        model: str,
        assistant_content: Any,
        reasoning_content: Any,
        messages: Any,
    ) -> None:
        assistant_text = _extract_message_text(assistant_content)
        reasoning_text = str(reasoning_content or "")
        if not reasoning_text:
            return

        cache_key = _cache_key(source_cfg, model, messages)
        entries = _DEEPSEEK_OPENAI_REASONING_CACHE.setdefault(cache_key, [])
        entries.append(
            {
                "assistant_text": assistant_text,
                "reasoning_content": reasoning_text,
            }
        )
        if len(entries) > _REASONING_CACHE_LIMIT:
            del entries[:-_REASONING_CACHE_LIMIT]
        _save_persistent_cache()

    async def stream_chat_to_ollama(self, source_cfg: SourceConfig, model: str, payload: Dict[str, Any]):
        prepared_payload = self._prepare_payload(source_cfg, payload)
        full_content = ""
        full_reasoning = ""
        async for line_json in super().stream_chat_to_ollama(source_cfg, model, prepared_payload):
            try:
                obj = json.loads(line_json)
            except Exception:
                yield line_json
                continue

            message = obj.get("message") or {}
            if isinstance(message, dict):
                full_content += _extract_message_text(message.get("content"))
                reasoning_piece = message.get("reasoning_content")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    full_reasoning = reasoning_piece
            if obj.get("done"):
                self._remember_cached_reasoning(
                    source_cfg, model, full_content, full_reasoning, prepared_payload.get("messages"))
            yield line_json

    async def stream_generate_to_ollama(self, source_cfg: SourceConfig, model: str, payload: Dict[str, Any]):
        prepared_payload = self._prepare_payload(source_cfg, payload)
        async for line_json in super().stream_generate_to_ollama(source_cfg, model, prepared_payload):
            yield line_json

    async def stream_chat_openai(self, source_cfg: SourceConfig, display_model: str, payload: Dict[str, Any]):
        prepared_payload = self._prepare_payload(source_cfg, payload)
        full_content = ""
        full_reasoning = ""
        async for chunk_bytes in super().stream_chat_openai(source_cfg, display_model, prepared_payload):
            try:
                line = chunk_bytes.decode("utf-8").strip()
            except Exception:
                yield chunk_bytes
                continue

            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                except Exception:
                    yield chunk_bytes
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                full_content += _extract_message_text(delta.get("content"))
                reasoning_piece = delta.get("reasoning_content")
                if not reasoning_piece:
                    message = choice.get("message") or {}
                    if isinstance(message, dict):
                        reasoning_piece = message.get("reasoning_content")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    full_reasoning += reasoning_piece

                # Keep reasoning only in proxy cache. Do not forward it to Copilot client,
                # otherwise large cumulative reasoning chunks can trigger "Response too long".
                if isinstance(delta, dict) and "reasoning_content" in delta:
                    delta.pop("reasoning_content", None)
                message_obj = choice.get("message")
                if isinstance(message_obj, dict) and "reasoning_content" in message_obj:
                    message_obj.pop("reasoning_content", None)
                if isinstance(choice, dict) and "reasoning_content" in choice:
                    choice.pop("reasoning_content", None)

                yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
                continue

            yield chunk_bytes

        self._remember_cached_reasoning(
            source_cfg,
            str(prepared_payload.get("model") or display_model),
            full_content,
            full_reasoning,
            prepared_payload.get("messages"),
        )

    async def chat_json_openai(self, source_cfg: SourceConfig, display_model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prepared_payload = self._prepare_payload(source_cfg, payload)
        data = await super().chat_json_openai(source_cfg, display_model, prepared_payload)
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict):
                self._remember_cached_reasoning(
                    source_cfg,
                    str(prepared_payload.get("model") or display_model),
                    message.get("content"),
                    message.get("reasoning_content"),
                    prepared_payload.get("messages"),
                )
                # For client compatibility (especially Copilot), do not expose reasoning_content.
                message.pop("reasoning_content", None)
        return data

    async def list_models(self, source_cfg: SourceConfig) -> list[Dict[str, Any]]:
        base_url = str(source_cfg.get("base_url") or "").rstrip("/")
        if base_url.endswith("/v1"):
            models_url = f"{base_url[:-3]}/models"
        else:
            models_url = f"{base_url}/models"

        async with httpx.AsyncClient(timeout=source_cfg["timeout"]) as client:
            response = await client.get(models_url, headers=source_headers(source_cfg))
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
            model_name = str(item.get("id") or item.get(
                "name") or "").strip().lower()
            if not model_name:
                continue
            meta = _extract_upstream_model_meta(item)
            if not meta.get("context_length"):
                fallback_ctx = _DEEPSEEK_OPENAI_CONTEXT_FALLBACK.get(
                    model_name)
                if fallback_ctx:
                    meta["context_length"] = fallback_ctx
            models.append(
                {
                    "name": model_name,
                    "upstream_model": str(item.get("id") or item.get("name") or model_name),
                    "meta": meta,
                }
            )
        return models


_load_persistent_cache()
