from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import HTTPException, status


SourceConfig = Dict[str, Any]


class RuleHandler(ABC):
    rules: tuple[str, ...] = ()
    default_paths: Dict[str, str] = {}
    default_rule_config: Dict[str, Any] = {
        "supports_embeddings": True,
        "supports_tools": False,
        "use_bearer_auth": True,
        "request_headers": {},
    }

    def normalize_rule_config(self, raw_rule_config: Dict[str, Any] | None) -> Dict[str, Any]:
        if raw_rule_config is None:
            raw_rule_config = {}
        if not isinstance(raw_rule_config, dict):
            raise RuntimeError("rule_config must be an object")

        merged = dict(self.default_rule_config)
        merged.update(raw_rule_config)

        merged["supports_embeddings"] = bool(
            merged.get("supports_embeddings", True))
        merged["supports_tools"] = bool(merged.get("supports_tools", False))
        merged["use_bearer_auth"] = bool(merged.get("use_bearer_auth", True))

        request_headers = merged.get("request_headers") or {}
        if not isinstance(request_headers, dict):
            raise RuntimeError("rule_config.request_headers must be an object")
        merged["request_headers"] = {str(key): str(
            value) for key, value in request_headers.items()}
        return merged

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return False

    async def list_models(self, source_cfg: SourceConfig) -> list[Dict[str, Any]]:
        raise make_not_implemented(source_cfg["rule"], "list models")

    @abstractmethod
    async def proxy_json(self, source_cfg: SourceConfig, path_key: str, body: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def stream_chat_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def stream_generate_to_ollama(
        self,
        source_cfg: SourceConfig,
        model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def stream_chat_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError

    @abstractmethod
    async def chat_json_openai(
        self,
        source_cfg: SourceConfig,
        display_model: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


def source_url(source_cfg: SourceConfig, path_key: str) -> str:
    path = str(source_cfg["paths"][path_key])
    if not path.startswith("/"):
        path = "/" + path
    return f"{source_cfg['base_url']}{path}"


def source_headers(source_cfg: SourceConfig) -> Dict[str, str]:
    headers = dict(source_cfg["headers"])
    for key, value in source_cfg.get("rule_config", {}).get("request_headers", {}).items():
        resolved_value = str(value).replace(
            "{{api_key}}", str(source_cfg.get("api_key") or ""))
        if resolved_value:
            headers[str(key)] = resolved_value
    return headers


def require_rule_capability(source_cfg: SourceConfig, capability_key: str, operation: str) -> None:
    if source_cfg.get("rule_config", {}).get(capability_key, False):
        return
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            f"source '{source_cfg['name']}' rule '{source_cfg['rule']}' does not support {operation}; "
            f"set rule_config.{capability_key}=true only if the upstream actually supports it"
        ),
    )


def make_http_error(response: httpx.Response) -> HTTPException:
    return HTTPException(status_code=response.status_code, detail=response.text)


def make_not_implemented(rule: str, operation: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"source rule '{rule}' is not implemented for {operation}",
    )
