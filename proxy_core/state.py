from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from rule_handlers import get_rule_handler, normalize_source_rule


def guess_model_family(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen" in lower:
        return "qwen2"
    if "glm" in lower:
        return "glm"
    if "mistral" in lower:
        return "mistral"
    if "llava" in lower or "llama" in lower:
        return "llama"
    return "llama"


def default_model_meta(model_name: str) -> Dict[str, Any]:
    family = guess_model_family(model_name)
    return {
        "family": family,
        "parameter_size": "7B",
        "quantization_level": "Q4_K_M",
        "context_length": 32768,
        "parameter_count": 7000000000,
        "capabilities": ["completion"],
        "size": 0,
        "digest": "",
    }


def register_model_alias(
    model_aliases: Dict[str, str],
    alias: str,
    target: str,
    *,
    strict: bool = True,
) -> None:
    alias_text = str(alias).strip()
    if not alias_text:
        return
    existing = model_aliases.get(alias_text)
    if existing and existing != target:
        if strict:
            raise RuntimeError(
                f"duplicate model alias '{alias_text}' for '{existing}' and '{target}'")
        return
    model_aliases[alias_text] = target


def register_model_entry(
    model_registry: Dict[str, Dict[str, Any]],
    model_aliases: Dict[str, str],
    source_name: str,
    model_name: str,
    model_cfg: Dict[str, Any],
    *,
    discovered: bool = False,
) -> str:
    desired_name = str(model_name).strip()
    if not desired_name:
        raise RuntimeError(
            f"source '{source_name}' contains an empty model name")
    if not isinstance(model_cfg, dict):
        raise RuntimeError(
            f"model '{desired_name}' in source '{source_name}' must be an object")

    canonical_name = desired_name
    existing = model_registry.get(canonical_name)
    if existing and existing["source"] != source_name:
        if not discovered:
            raise RuntimeError(
                f"duplicate model name '{desired_name}' across sources")
        canonical_name = f"{source_name}/{desired_name}"
        existing = model_registry.get(canonical_name)
        if existing and existing["source"] != source_name:
            raise RuntimeError(
                f"duplicate discovered model name '{canonical_name}' across sources")
    if existing and existing["source"] == source_name:
        return canonical_name

    meta = default_model_meta(desired_name)
    raw_meta = model_cfg.get("meta") or {}
    if raw_meta:
        if not isinstance(raw_meta, dict):
            raise RuntimeError(
                f"model '{desired_name}' meta must be an object")
        meta.update(raw_meta)

    aliases = model_cfg.get("aliases") or []
    if aliases and not isinstance(aliases, list):
        raise RuntimeError(f"model '{desired_name}' aliases must be an array")

    model_registry[canonical_name] = {
        "name": canonical_name,
        "source": source_name,
        "upstream_model": str(model_cfg.get("upstream_model") or desired_name),
        "meta": meta,
        "discovered": discovered,
    }

    register_model_alias(model_aliases, canonical_name, canonical_name)
    register_model_alias(model_aliases, canonical_name.lower(), canonical_name)
    register_model_alias(
        model_aliases, f"{canonical_name}:latest", canonical_name)
    register_model_alias(
        model_aliases, f"{canonical_name.lower()}:latest", canonical_name)

    friendly_alias_strict = not discovered
    register_model_alias(model_aliases, desired_name,
                         canonical_name, strict=friendly_alias_strict)
    register_model_alias(model_aliases, desired_name.lower(),
                         canonical_name, strict=friendly_alias_strict)
    register_model_alias(
        model_aliases, f"{desired_name}:latest", canonical_name, strict=friendly_alias_strict)
    register_model_alias(
        model_aliases,
        f"{desired_name.lower()}:latest",
        canonical_name,
        strict=friendly_alias_strict,
    )

    if discovered:
        register_model_alias(
            model_aliases, f"{source_name}/{desired_name}", canonical_name, strict=False)
        register_model_alias(
            model_aliases, f"{source_name.lower()}/{desired_name.lower()}", canonical_name, strict=False)
        register_model_alias(
            model_aliases, f"{source_name}:{desired_name}", canonical_name, strict=False)
        register_model_alias(
            model_aliases, f"{source_name.lower()}:{desired_name.lower()}", canonical_name, strict=False)

    for alias in aliases:
        alias_text = str(alias).strip()
        if alias_text:
            register_model_alias(model_aliases, alias_text,
                                 canonical_name, strict=friendly_alias_strict)
            register_model_alias(model_aliases, alias_text.lower(
            ), canonical_name, strict=friendly_alias_strict)

    return canonical_name


def merge_source_headers(api_key: str, extra_headers: Dict[str, Any], use_bearer_auth: bool = True) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and use_bearer_auth:
        headers["Authorization"] = f"Bearer {api_key}"
    for key, value in extra_headers.items():
        headers[str(key)] = str(value)
    return headers


class ProxyState:
    def __init__(self, config_path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("ollama_proxy.state")
        config = self._load_proxy_config(config_path)
        self.source_registry: Dict[str, Dict[str, Any]] = config["sources"]
        self.source_order: List[str] = config["source_order"]
        self.model_registry: Dict[str, Dict[str, Any]] = config["models"]
        self.model_aliases: Dict[str, str] = config["aliases"]
        self.default_model: str = config["default_model"]

    def _load_proxy_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise RuntimeError(f"config file not found: {config_path}")

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid config JSON: {exc}") from exc

        sources = raw.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise RuntimeError("config 'sources' must be a non-empty object")

        source_registry: Dict[str, Dict[str, Any]] = {}
        model_registry: Dict[str, Dict[str, Any]] = {}
        model_aliases: Dict[str, str] = {}
        source_order: List[str] = []
        has_auto_discovery_source = False

        for source_name, source_cfg in sources.items():
            if not isinstance(source_cfg, dict):
                raise RuntimeError(f"source '{source_name}' must be an object")

            source_enabled = source_cfg.get("enable", True)
            if not isinstance(source_enabled, bool):
                raise RuntimeError(
                    f"source '{source_name}' enable must be a boolean")
            if not source_enabled:
                continue

            rule = normalize_source_rule(source_cfg.get("rule"))
            handler = get_rule_handler(rule)
            base_url = str(source_cfg.get("base_url")
                           or "").strip().rstrip("/")
            if not base_url:
                raise RuntimeError(f"source '{source_name}' missing base_url")

            extra_headers = source_cfg.get("headers") or {}
            if not isinstance(extra_headers, dict):
                raise RuntimeError(
                    f"source '{source_name}' headers must be an object")

            paths = source_cfg.get("paths") or {}
            if not isinstance(paths, dict):
                raise RuntimeError(
                    f"source '{source_name}' paths must be an object")

            models = source_cfg.get("models") or {}
            if not isinstance(models, dict):
                raise RuntimeError(
                    f"source '{source_name}' models must be an object")

            merged_paths = {
                key: str(paths.get(key) or default_path)
                for key, default_path in handler.default_paths.items()
            }
            for path_key, path_value in paths.items():
                if path_key not in merged_paths:
                    merged_paths[str(path_key)] = str(path_value)

            rule_config = handler.normalize_rule_config(
                source_cfg.get("rule_config"))

            source_entry = {
                "name": source_name,
                "rule": rule,
                "handler": handler,
                "api_key": str(source_cfg.get("api_key") or ""),
                "base_url": base_url,
                "headers": merge_source_headers(
                    str(source_cfg.get("api_key") or ""),
                    extra_headers,
                    use_bearer_auth=bool(
                        rule_config.get("use_bearer_auth", True)),
                ),
                "timeout": float(source_cfg.get("timeout", 300.0)),
                "rule_config": rule_config,
                "paths": merged_paths,
                "auto_discover_models": False,
                "models_discovered": False,
            }
            source_entry["auto_discover_models"] = handler.supports_model_discovery(
                source_entry)
            source_registry[source_name] = source_entry
            source_order.append(source_name)
            has_auto_discovery_source = has_auto_discovery_source or source_entry[
                "auto_discover_models"]

            if not models and not source_entry["auto_discover_models"]:
                raise RuntimeError(
                    f"source '{source_name}' models must be a non-empty object unless rule '{rule}' supports automatic model discovery"
                )

            for model_name, model_cfg in models.items():
                if isinstance(model_cfg, str):
                    model_cfg = {"upstream_model": model_cfg}
                register_model_entry(
                    model_registry, model_aliases, source_name, model_name, model_cfg)

        default_model = str(raw.get("default_model") or "").strip()
        if default_model:
            default_model = model_aliases.get(
                default_model, model_aliases.get(default_model.lower(), default_model))
            if default_model not in model_registry and not has_auto_discovery_source:
                raise RuntimeError(
                    f"default_model '{default_model}' not found in config")
        elif model_registry:
            default_model = next(iter(model_registry))
        elif not has_auto_discovery_source:
            raise RuntimeError(
                "config must provide at least one model or one source with automatic model discovery")

        return {
            "sources": source_registry,
            "source_order": source_order,
            "models": model_registry,
            "aliases": model_aliases,
            "default_model": default_model,
        }

    def normalize_model_name(self, model_name: Optional[str]) -> str:
        if not model_name:
            return self.default_model or ""
        candidate = str(model_name).strip()
        if not candidate:
            return self.default_model or ""
        if candidate in self.model_aliases:
            return self.model_aliases[candidate]
        lowered = candidate.lower()
        if lowered in self.model_aliases:
            return self.model_aliases[lowered]
        if ":" in candidate:
            base = candidate.split(":", 1)[0]
            if base in self.model_aliases:
                return self.model_aliases[base]
            base_lower = base.lower()
            if base_lower in self.model_aliases:
                return self.model_aliases[base_lower]
        return candidate

    def resolve_source_config(self, source_name: str) -> Dict[str, Any]:
        return self.source_registry[source_name]

    async def discover_source_models(self, source_cfg: Dict[str, Any]) -> None:
        if not source_cfg.get("auto_discover_models") or source_cfg.get("models_discovered"):
            return

        discovered_models = await source_cfg["handler"].list_models(source_cfg)
        before_count = len(self.model_registry)
        for model_cfg in discovered_models:
            if not isinstance(model_cfg, dict):
                continue
            model_name = str(model_cfg.get("name") or "").strip()
            if not model_name:
                continue
            register_model_entry(
                self.model_registry,
                self.model_aliases,
                source_cfg["name"],
                model_name,
                model_cfg,
                discovered=True,
            )

        source_cfg["models_discovered"] = True
        self.logger.info(
            "model discovery source=%s discovered=%s total=%s",
            source_cfg["name"],
            len(self.model_registry) - before_count,
            len(self.model_registry),
        )

    async def ensure_discoverable_models_loaded(self) -> List[Exception]:
        discovery_errors: List[Exception] = []
        for source_name in self.source_order:
            source_cfg = self.source_registry[source_name]
            if not source_cfg.get("auto_discover_models") or source_cfg.get("models_discovered"):
                continue
            try:
                await self.discover_source_models(source_cfg)
            except Exception as exc:
                self.logger.warning(
                    "model discovery failed source=%s error=%s", source_name, exc)
                discovery_errors.append(exc)
        return discovery_errors

    async def resolve_default_model_config(
        self,
        discovery_errors: Optional[List[Exception]] = None,
    ) -> Dict[str, Any]:
        if self.default_model:
            local_name = self.normalize_model_name(self.default_model)
            model_cfg = self.model_registry.get(local_name)
            if model_cfg:
                return model_cfg
            if discovery_errors and not self.model_registry:
                self.raise_discovery_error(discovery_errors[0])
            raise HTTPException(
                status_code=404, detail=f"default_model '{self.default_model}' not found")

        if self.model_registry:
            return next(iter(self.model_registry.values()))

        if discovery_errors:
            self.raise_discovery_error(discovery_errors[0])
        raise HTTPException(status_code=404, detail="no models available")

    async def resolve_model_config(self, model_name: Optional[str], allow_default: bool = True) -> Dict[str, Any]:
        local_name = self.normalize_model_name(model_name)
        model_cfg = self.model_registry.get(local_name)
        if model_cfg:
            return model_cfg

        discovery_errors: List[Exception] = []
        for source_name in self.source_order:
            source_cfg = self.source_registry[source_name]
            if not source_cfg.get("auto_discover_models") or source_cfg.get("models_discovered"):
                continue
            try:
                await self.discover_source_models(source_cfg)
            except Exception as exc:
                self.logger.warning(
                    "model discovery failed source=%s error=%s", source_name, exc)
                discovery_errors.append(exc)
                continue

            local_name = self.normalize_model_name(model_name)
            model_cfg = self.model_registry.get(local_name)
            if model_cfg:
                return model_cfg

        local_name = self.normalize_model_name(model_name)
        model_cfg = self.model_registry.get(local_name)
        if model_cfg:
            return model_cfg

        if allow_default:
            return await self.resolve_default_model_config(discovery_errors)

        if discovery_errors and not self.model_registry:
            self.raise_discovery_error(discovery_errors[0])
        raise HTTPException(
            status_code=404, detail=f"model '{local_name}' not found")

    @staticmethod
    def raise_discovery_error(exc: Exception) -> None:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
