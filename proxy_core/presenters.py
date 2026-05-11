from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from proxy_core.state import guess_model_family


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds") + "Z"


def effective_capabilities(meta: Dict[str, Any], source_cfg: Dict[str, Any]) -> List[str]:
    capabilities = list(meta.get("capabilities", ["completion"]))
    if source_cfg.get("rule_config", {}).get("supports_tools", False):
        if "tools" not in capabilities:
            capabilities.append("tools")
    else:
        capabilities = [item for item in capabilities if item != "tools"]
    return capabilities


def merged_model_info(meta: Dict[str, Any], family: str) -> Dict[str, Any]:
    context_length = meta.get("context_length", 8192)
    info: Dict[str, Any] = {
        "general.architecture": family,
        "general.file_type": 2,
        "general.parameter_count": meta.get("parameter_count", 7000000000),
        "general.quantization_version": 2,
        f"{family}.context_length": context_length,
        "llama.context_length": context_length,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }

    extra_model_info = meta.get("model_info")
    if isinstance(extra_model_info, dict):
        info.update(extra_model_info)
    return info


def make_ollama_response(model: str, response_text: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "model": model,
        "created_at": now_iso(),
        "response": response_text,
        "done": True,
    }
    if extra:
        result.update(extra)
    return result


def make_chat_response(model: str, message: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "model": model,
        "created_at": now_iso(),
        "message": message,
        "done": True,
    }
    if extra:
        result.update(extra)
    return result


def build_tags_models(model_registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": now_iso(),
            "size": model_cfg["meta"].get("size", 0),
            "digest": model_cfg["meta"].get("digest", ""),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": model_cfg["meta"].get("family", guess_model_family(name)),
                "families": [model_cfg["meta"].get("family", guess_model_family(name))],
                "parameter_size": model_cfg["meta"].get("parameter_size", "7B"),
                "quantization_level": model_cfg["meta"].get("quantization_level", "Q4_K_M"),
            },
        }
        for name, model_cfg in model_registry.items()
    ]


def build_ps_models(model_registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "size": model_cfg["meta"].get("size", 0),
            "digest": model_cfg["meta"].get("digest", ""),
            "expires_at": now_iso(),
            "size_vram": model_cfg["meta"].get("size", 0),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": model_cfg["meta"].get("family", guess_model_family(name)),
                "families": [model_cfg["meta"].get("family", guess_model_family(name))],
                "parameter_size": model_cfg["meta"].get("parameter_size", "7B"),
                "quantization_level": model_cfg["meta"].get("quantization_level", "Q4_K_M"),
            },
        }
        for name, model_cfg in model_registry.items()
    ]
