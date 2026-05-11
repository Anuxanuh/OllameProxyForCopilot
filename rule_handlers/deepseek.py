from __future__ import annotations

from typing import Any, Dict

from .base import SourceConfig
from .openai import OpenAIRuleHandler

RULE_DEEPSEEK = "deepseek"

_DEEPSEEK_CONTEXT_FALLBACK = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
}


class DeepSeekRuleHandler(OpenAIRuleHandler):
    rules = (RULE_DEEPSEEK,)

    def supports_model_discovery(self, source_cfg: SourceConfig) -> bool:
        return source_cfg.get("rule") == RULE_DEEPSEEK

    async def list_models(self, source_cfg: SourceConfig) -> list[Dict[str, Any]]:
        models = await super().list_models(source_cfg)
        for model in models:
            model_name = str(model.get("name") or "").strip().lower()
            meta = model.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                model["meta"] = meta
            if meta.get("context_length"):
                continue
            fallback_ctx = _DEEPSEEK_CONTEXT_FALLBACK.get(model_name)
            if fallback_ctx:
                meta["context_length"] = fallback_ctx
        return models
