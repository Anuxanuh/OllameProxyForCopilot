from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncIterator, Dict

import httpx

from .base import RuleHandler, SourceConfig, make_http_error, source_headers, source_url

RULE_OPENAI = "openai"
RULE_OPENAI_COMPATIBLE_PARTIAL = "openai_compatible_partial"


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="microseconds") + "Z"


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
                    content = delta.get("content") or ""
                    finish_reason = choice.get("finish_reason")
                    done = finish_reason is not None
                    obj: Dict[str, Any] = {
                        "model": model,
                        "created_at": now_iso(),
                        "message": {"role": "assistant", "content": content},
                        "done": done,
                    }
                    if done:
                        usage = chunk.get("usage") or {}
                        obj["done_reason"] = "stop"
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
                        obj["done_reason"] = "stop"
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
                "meta": {},
            })
        return models
