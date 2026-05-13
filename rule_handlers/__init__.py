from __future__ import annotations

from typing import Dict

from .anthropic import AnthropicRuleHandler, RULE_ANTHROPIC
from .base import RuleHandler
from .deepseek_anthropic import DeepSeekAnthropicRuleHandler, RULE_DEEPSEEK_ANTHROPIC
from .deepseek_openai import DeepSeekOpenAIRuleHandler, RULE_DEEPSEEK_OPENAI
from .openai import OpenAIRuleHandler, RULE_OPENAI, RULE_OPENAI_COMPATIBLE_PARTIAL


_HANDLERS = [
    OpenAIRuleHandler(),
    DeepSeekAnthropicRuleHandler(),
    DeepSeekOpenAIRuleHandler(),
    AnthropicRuleHandler(),
]
RULE_HANDLER_REGISTRY: Dict[str, RuleHandler] = {}
for handler in _HANDLERS:
    for rule_name in handler.rules:
        RULE_HANDLER_REGISTRY[rule_name] = handler

SUPPORTED_SOURCE_RULES = set(RULE_HANDLER_REGISTRY)


def normalize_source_rule(rule_name: str | None) -> str:
    candidate = str(
        rule_name or RULE_OPENAI_COMPATIBLE_PARTIAL).strip().lower()
    aliases = {
        "openai-compatible-partial": RULE_OPENAI_COMPATIBLE_PARTIAL,
        "openai_compatible_partial": RULE_OPENAI_COMPATIBLE_PARTIAL,
        "openai-compatible": RULE_OPENAI_COMPATIBLE_PARTIAL,
        "openai_compatible": RULE_OPENAI_COMPATIBLE_PARTIAL,
        "openai": RULE_OPENAI,
        "openai_aliyun": RULE_OPENAI_COMPATIBLE_PARTIAL,
        "deepseek": RULE_DEEPSEEK_ANTHROPIC,
        "deepseek_anthropic": RULE_DEEPSEEK_ANTHROPIC,
        "deepseek-anthropic": RULE_DEEPSEEK_ANTHROPIC,
        "deepseek_openai": RULE_DEEPSEEK_OPENAI,
        "deepseek-openai": RULE_DEEPSEEK_OPENAI,
        "anthropic": RULE_ANTHROPIC,
    }
    normalized = aliases.get(candidate, candidate)
    if normalized not in SUPPORTED_SOURCE_RULES:
        raise RuntimeError(f"unsupported source rule '{rule_name}'")
    return normalized


def get_rule_handler(rule_name: str) -> RuleHandler:
    try:
        return RULE_HANDLER_REGISTRY[rule_name]
    except KeyError as exc:
        raise RuntimeError(
            f"handler not registered for source rule '{rule_name}'") from exc
