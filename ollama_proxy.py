import json
import time
import ast
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from rule_handlers import get_rule_handler, normalize_source_rule
from rule_handlers.base import require_rule_capability

app = FastAPI(title="Ollama-Compatible Proxy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ollama_proxy")

CONFIG_PATH = Path(__file__).with_name("proxy_config.json")
PORT = 11434
OLLAMA_VERSION = "0.6.4"


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


def effective_capabilities(meta: Dict[str, Any], source_cfg: Dict[str, Any]) -> List[str]:
    capabilities = list(meta.get("capabilities", ["completion"]))
    if source_cfg.get("rule_config", {}).get("supports_tools", False):
        if "tools" not in capabilities:
            capabilities.append("tools")
    else:
        capabilities = [item for item in capabilities if item != "tools"]
    return capabilities


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
            raise RuntimeError(f"duplicate model alias '{alias_text}' for '{existing}' and '{target}'")
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
        raise RuntimeError(f"source '{source_name}' contains an empty model name")
    if not isinstance(model_cfg, dict):
        raise RuntimeError(f"model '{desired_name}' in source '{source_name}' must be an object")

    canonical_name = desired_name
    existing = model_registry.get(canonical_name)
    if existing and existing["source"] != source_name:
        if not discovered:
            raise RuntimeError(f"duplicate model name '{desired_name}' across sources")
        canonical_name = f"{source_name}/{desired_name}"
        existing = model_registry.get(canonical_name)
        if existing and existing["source"] != source_name:
            raise RuntimeError(f"duplicate discovered model name '{canonical_name}' across sources")
    if existing and existing["source"] == source_name:
        return canonical_name

    meta = default_model_meta(desired_name)
    raw_meta = model_cfg.get("meta") or {}
    if raw_meta:
        if not isinstance(raw_meta, dict):
            raise RuntimeError(f"model '{desired_name}' meta must be an object")
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
    register_model_alias(model_aliases, f"{canonical_name}:latest", canonical_name)
    register_model_alias(model_aliases, f"{canonical_name.lower()}:latest", canonical_name)

    friendly_alias_strict = not discovered
    register_model_alias(model_aliases, desired_name, canonical_name, strict=friendly_alias_strict)
    register_model_alias(model_aliases, desired_name.lower(), canonical_name, strict=friendly_alias_strict)
    register_model_alias(model_aliases, f"{desired_name}:latest", canonical_name, strict=friendly_alias_strict)
    register_model_alias(
        model_aliases,
        f"{desired_name.lower()}:latest",
        canonical_name,
        strict=friendly_alias_strict,
    )

    if discovered:
        register_model_alias(model_aliases, f"{source_name}/{desired_name}", canonical_name, strict=False)
        register_model_alias(model_aliases, f"{source_name.lower()}/{desired_name.lower()}", canonical_name, strict=False)
        register_model_alias(model_aliases, f"{source_name}:{desired_name}", canonical_name, strict=False)
        register_model_alias(model_aliases, f"{source_name.lower()}:{desired_name.lower()}", canonical_name, strict=False)

    for alias in aliases:
        alias_text = str(alias).strip()
        if alias_text:
            register_model_alias(model_aliases, alias_text, canonical_name, strict=friendly_alias_strict)
            register_model_alias(model_aliases, alias_text.lower(), canonical_name, strict=friendly_alias_strict)

    return canonical_name


def merge_source_headers(api_key: str, extra_headers: Dict[str, Any], use_bearer_auth: bool = True) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and use_bearer_auth:
        headers["Authorization"] = f"Bearer {api_key}"
    for key, value in extra_headers.items():
        headers[str(key)] = str(value)
    return headers


def load_proxy_config(config_path: Path) -> Dict[str, Any]:
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

        rule = normalize_source_rule(source_cfg.get("rule"))
        handler = get_rule_handler(rule)
        base_url = str(source_cfg.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError(f"source '{source_name}' missing base_url")

        extra_headers = source_cfg.get("headers") or {}
        if not isinstance(extra_headers, dict):
            raise RuntimeError(f"source '{source_name}' headers must be an object")

        paths = source_cfg.get("paths") or {}
        if not isinstance(paths, dict):
            raise RuntimeError(f"source '{source_name}' paths must be an object")

        models = source_cfg.get("models") or {}
        if not isinstance(models, dict):
            raise RuntimeError(f"source '{source_name}' models must be an object")

        merged_paths = {
            key: str(paths.get(key) or default_path)
            for key, default_path in handler.default_paths.items()
        }
        for path_key, path_value in paths.items():
            if path_key not in merged_paths:
                merged_paths[str(path_key)] = str(path_value)

        rule_config = handler.normalize_rule_config(source_cfg.get("rule_config"))

        source_entry = {
            "name": source_name,
            "rule": rule,
            "handler": handler,
            "api_key": str(source_cfg.get("api_key") or ""),
            "base_url": base_url,
            "headers": merge_source_headers(
                str(source_cfg.get("api_key") or ""),
                extra_headers,
                use_bearer_auth=bool(rule_config.get("use_bearer_auth", True)),
            ),
            "timeout": float(source_cfg.get("timeout", 300.0)),
            "rule_config": rule_config,
            "paths": merged_paths,
            "auto_discover_models": False,
            "models_discovered": False,
        }
        source_entry["auto_discover_models"] = handler.supports_model_discovery(source_entry)
        source_registry[source_name] = source_entry
        source_order.append(source_name)
        has_auto_discovery_source = has_auto_discovery_source or source_entry["auto_discover_models"]

        if not models and not source_entry["auto_discover_models"]:
            raise RuntimeError(
                f"source '{source_name}' models must be a non-empty object unless rule '{rule}' supports automatic model discovery"
            )

        for model_name, model_cfg in models.items():
            if isinstance(model_cfg, str):
                model_cfg = {"upstream_model": model_cfg}
            register_model_entry(model_registry, model_aliases, source_name, model_name, model_cfg)

    default_model = str(raw.get("default_model") or "").strip()
    if default_model:
        default_model = model_aliases.get(default_model, model_aliases.get(default_model.lower(), default_model))
        if default_model not in model_registry and not has_auto_discovery_source:
            raise RuntimeError(f"default_model '{default_model}' not found in config")
    elif model_registry:
        default_model = next(iter(model_registry))
    elif not has_auto_discovery_source:
        raise RuntimeError("config must provide at least one model or one source with automatic model discovery")

    return {
        "sources": source_registry,
        "source_order": source_order,
        "models": model_registry,
        "aliases": model_aliases,
        "default_model": default_model,
    }


PROXY_CONFIG = load_proxy_config(CONFIG_PATH)
SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = PROXY_CONFIG["sources"]
SOURCE_ORDER: List[str] = PROXY_CONFIG["source_order"]
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = PROXY_CONFIG["models"]
MODEL_ALIASES: Dict[str, str] = PROXY_CONFIG["aliases"]
DEFAULT_MODEL = PROXY_CONFIG["default_model"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds") + "Z"


def safe_text_preview(raw: bytes, limit: int = 240) -> str:
    text = raw.decode("utf-8", errors="replace").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@app.middleware("http")
async def request_trace_middleware(request: Request, call_next):
    trace_paths = {"/api/show", "/api/chat", "/api/generate", "/api/tags", "/api/version", "/v1/chat/completions"}
    path = request.url.path
    method = request.method
    start = time.perf_counter()
    body_preview = ""

    if method == "POST" and path in {"/api/show", "/api/chat", "/api/generate", "/v1/chat/completions"}:
        raw = await request.body()
        body_preview = safe_text_preview(raw)
        logger.info("request %s %s body=%s", method, path, body_preview)
    elif path in trace_paths:
        logger.info("request %s %s", method, path)

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception("response %s %s status=500 duration_ms=%.2f", method, path, duration_ms)
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    if path in trace_paths:
        logger.info(
            "response %s %s status=%s duration_ms=%.2f",
            method,
            path,
            response.status_code,
            duration_ms,
        )
    return response


def normalize_model_name(model_name: Optional[str]) -> str:
    if not model_name:
        return DEFAULT_MODEL or ""
    candidate = str(model_name).strip()
    if not candidate:
        return DEFAULT_MODEL or ""
    if candidate in MODEL_ALIASES:
        return MODEL_ALIASES[candidate]
    lowered = candidate.lower()
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]
    if ":" in candidate:
        base = candidate.split(":", 1)[0]
        if base in MODEL_ALIASES:
            return MODEL_ALIASES[base]
        base_lower = base.lower()
        if base_lower in MODEL_ALIASES:
            return MODEL_ALIASES[base_lower]
    return candidate


def raise_discovery_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


async def discover_source_models(source_cfg: Dict[str, Any]) -> None:
    if not source_cfg.get("auto_discover_models") or source_cfg.get("models_discovered"):
        return

    discovered_models = await source_cfg["handler"].list_models(source_cfg)
    before_count = len(MODEL_REGISTRY)
    for model_cfg in discovered_models:
        if not isinstance(model_cfg, dict):
            continue
        model_name = str(model_cfg.get("name") or "").strip()
        if not model_name:
            continue
        register_model_entry(
            MODEL_REGISTRY,
            MODEL_ALIASES,
            source_cfg["name"],
            model_name,
            model_cfg,
            discovered=True,
        )

    source_cfg["models_discovered"] = True
    logger.info(
        "model discovery source=%s discovered=%s total=%s",
        source_cfg["name"],
        len(MODEL_REGISTRY) - before_count,
        len(MODEL_REGISTRY),
    )


async def ensure_discoverable_models_loaded() -> List[Exception]:
    discovery_errors: List[Exception] = []
    for source_name in SOURCE_ORDER:
        source_cfg = SOURCE_REGISTRY[source_name]
        if not source_cfg.get("auto_discover_models") or source_cfg.get("models_discovered"):
            continue
        try:
            await discover_source_models(source_cfg)
        except Exception as exc:
            logger.warning("model discovery failed source=%s error=%s", source_name, exc)
            discovery_errors.append(exc)
    return discovery_errors


async def resolve_default_model_config(discovery_errors: Optional[List[Exception]] = None) -> Dict[str, Any]:
    if DEFAULT_MODEL:
        local_name = normalize_model_name(DEFAULT_MODEL)
        model_cfg = MODEL_REGISTRY.get(local_name)
        if model_cfg:
            return model_cfg
        if discovery_errors and not MODEL_REGISTRY:
            raise_discovery_error(discovery_errors[0])
        raise HTTPException(status_code=404, detail=f"default_model '{DEFAULT_MODEL}' not found")

    if MODEL_REGISTRY:
        return next(iter(MODEL_REGISTRY.values()))

    if discovery_errors:
        raise_discovery_error(discovery_errors[0])
    raise HTTPException(status_code=404, detail="no models available")


async def resolve_model_config(model_name: Optional[str], allow_default: bool = True) -> Dict[str, Any]:
    local_name = normalize_model_name(model_name)
    model_cfg = MODEL_REGISTRY.get(local_name)
    if model_cfg:
        return model_cfg

    discovery_errors: List[Exception] = []
    for source_name in SOURCE_ORDER:
        source_cfg = SOURCE_REGISTRY[source_name]
        if not source_cfg.get("auto_discover_models") or source_cfg.get("models_discovered"):
            continue
        try:
            await discover_source_models(source_cfg)
        except Exception as exc:
            logger.warning("model discovery failed source=%s error=%s", source_name, exc)
            discovery_errors.append(exc)
            continue

        local_name = normalize_model_name(model_name)
        model_cfg = MODEL_REGISTRY.get(local_name)
        if model_cfg:
            return model_cfg

    local_name = normalize_model_name(model_name)
    model_cfg = MODEL_REGISTRY.get(local_name)
    if model_cfg:
        return model_cfg

    if allow_default:
        return await resolve_default_model_config(discovery_errors)

    if discovery_errors and not MODEL_REGISTRY:
        raise_discovery_error(discovery_errors[0])
    raise HTTPException(status_code=404, detail=f"model '{local_name}' not found")


def resolve_source_config(source_name: str) -> Dict[str, Any]:
    return SOURCE_REGISTRY[source_name]


def flatten_options(body: Dict[str, Any], payload: Dict[str, Any]) -> None:
    options = body.get("options") or {}
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "n",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "seed",
        "stop",
        "best_of",
        "min_p",
        "typical_p",
        "repeat_last_n",
        "num_keep",
        "num_predict",
        "num_ctx",
        "num_thread",
    ]:
        if key in options:
            payload[key] = options[key]
    if "stream" in body:
        payload["stream"] = body["stream"]
    if "keep_alive" in body:
        payload["keep_alive"] = body["keep_alive"]


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


async def parse_request_json(request: Request, allow_empty: bool = False) -> Dict[str, Any]:
    """Parse request body as JSON with a safe fallback for single-quoted dict payloads."""
    raw = await request.body()
    if not raw:
        if allow_empty:
            return {}
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="request body is empty")

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        text = raw.decode("utf-8", errors="replace").strip()
        try:
            alt = ast.literal_eval(text)
        except Exception:
            alt = None
        if isinstance(alt, dict):
            body = alt
            logger.info("json fallback ast.literal_eval accepted path=%s", request.url.path)
        else:
            logger.warning("json parse failed path=%s body=%s", request.url.path, safe_text_preview(raw))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid JSON body",
            )

    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON body must be an object")
    return body


@app.get("/v1/models")
@app.get("/api/models")
async def list_models() -> Dict[str, Any]:
    discovery_errors = await ensure_discoverable_models_loaded()
    if not MODEL_REGISTRY and discovery_errors:
        raise_discovery_error(discovery_errors[0])
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "proxy", "root": name}
            for name in MODEL_REGISTRY.keys()
        ],
    }


@app.get("/v1/models/{model_name}")
@app.get("/api/models/{model_name}")
@app.post("/v1/models/{model_name}")
@app.post("/api/models/{model_name}")
async def get_model(model_name: str) -> Dict[str, Any]:
    local = normalize_model_name(model_name)
    model_cfg = await resolve_model_config(local, allow_default=False)
    return {
        "id": local,
        "object": "model",
        "created": 0,
        "owned_by": "proxy",
        "root": model_cfg["name"],
        "permission": [],
    }


@app.get("/api/tags")
async def list_tags() -> Dict[str, Any]:
    discovery_errors = await ensure_discoverable_models_loaded()
    if not MODEL_REGISTRY and discovery_errors:
        raise_discovery_error(discovery_errors[0])
    return {
        "models": [
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
            for name, model_cfg in MODEL_REGISTRY.items()
        ]
    }


@app.get("/api/ps")
async def ps_models() -> Dict[str, Any]:
    discovery_errors = await ensure_discoverable_models_loaded()
    if not MODEL_REGISTRY and discovery_errors:
        raise_discovery_error(discovery_errors[0])
    return {
        "models": [
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
            for name, model_cfg in MODEL_REGISTRY.items()
        ]
    }


@app.post("/api/version")
@app.get("/api/version")
@app.get("/version")
async def version() -> Dict[str, str]:
    return {"version": OLLAMA_VERSION}


@app.post("/api/show")
async def show_model(request: Request) -> Dict[str, Any]:
    body = await parse_request_json(request, allow_empty=True)
    model_name = normalize_model_name(body.get("model") or body.get("name") or DEFAULT_MODEL)
    model_cfg = await resolve_model_config(model_name, allow_default=False)
    source_cfg = resolve_source_config(model_cfg["source"])
    logger.info("show resolved_model=%s request_keys=%s", model_name, sorted(body.keys()))
    meta = model_cfg["meta"]
    family = meta.get("family", guess_model_family(model_name))
    tagged_name = f"{model_name}:latest"
    template = (
        "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
        "{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n{{ end }}"
        "{{ if .Response }}{{ .Response }}<|im_end|>\n{{ end }}"
    )
    return {
        "model": tagged_name,
        "modified_at": now_iso(),
        "license": "",
        "system": "",
        "messages": [],
        "modelfile": (
            f'# Modelfile generated by "ollama show"\n'
            f"# To build a new Modelfile based on this one, replace the FROM line with:\n"
            f"# FROM {tagged_name}\n"
            f"FROM {tagged_name}\n"
            f'TEMPLATE """{ template }"""\n'
            "PARAMETER stop \"<|im_start|>\"\n"
            "PARAMETER stop \"<|im_end|>\"\n"
        ),
        "parameters": (
            "stop                           \"<|im_start|>\"\n"
            "stop                           \"<|im_end|>\"\n"
        ),
        "template": template,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": meta.get("parameter_size", "7B"),
            "quantization_level": meta.get("quantization_level", "Q4_K_M"),
        },
        "model_info": {
            "general.architecture": family,
            "general.file_type": 2,
            "general.parameter_count": meta.get("parameter_count", 7000000000),
            "general.quantization_version": 2,
            f"{family}.context_length": meta.get("context_length", 8192),
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
        },
        "capabilities": effective_capabilities(meta, source_cfg),
    }


@app.post("/api/generate")
async def generate(request: Request) -> Any:
    body = await parse_request_json(request)
    model_cfg = await resolve_model_config(body.get("model"))
    source_cfg = resolve_source_config(model_cfg["source"])
    handler = source_cfg["handler"]
    model = model_cfg["name"]
    prompt = body.get("prompt", "")

    # /api/generate maps to /chat/completions with a user message
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    if body.get("system"):
        messages.insert(0, {"role": "system", "content": body["system"]})

    payload: Dict[str, Any] = {"model": model_cfg["upstream_model"], "messages": messages}
    flatten_options(body, payload)
    payload["stream"] = True

    want_stream = body.get("stream", True)
    logger.info(
        "generate rule=%s model=%s upstream_model=%s want_stream=%s prompt_len=%s",
        source_cfg["rule"],
        normalize_model_name(body.get("model")),
        model_cfg["upstream_model"],
        want_stream,
        len(str(prompt)),
    )
    if want_stream:
        return StreamingResponse(
            handler.stream_generate_to_ollama(source_cfg, model, payload),
            media_type="application/x-ndjson",
        )

    # Non-streaming: consume via generator
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    done_reason = "stop"
    async for line_json in handler.stream_generate_to_ollama(source_cfg, model, payload):
        obj = json.loads(line_json)
        full_text += obj.get("response") or ""
        if obj.get("done"):
            prompt_tokens = obj.get("prompt_eval_count", 0)
            completion_tokens = obj.get("eval_count", 0)
            done_reason = str(obj.get("done_reason") or done_reason)

    result = make_ollama_response(model, full_text, {
        "done_reason": done_reason,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": 0,
    })
    return JSONResponse(content=result)


@app.post("/api/chat")
async def chat(request: Request) -> Any:
    body = await parse_request_json(request)
    model_cfg = await resolve_model_config(body.get("model"))
    source_cfg = resolve_source_config(model_cfg["source"])
    handler = source_cfg["handler"]
    model = model_cfg["name"]
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages must be a list")

    payload: Dict[str, Any] = {"model": model_cfg["upstream_model"], "messages": messages}
    flatten_options(body, payload)
    payload["stream"] = True
    if "tools" in body:
        require_rule_capability(source_cfg, "supports_tools", "tools")
        payload["tools"] = body["tools"]

    want_stream = body.get("stream", True)
    logger.info(
        "chat rule=%s model=%s upstream_model=%s want_stream=%s messages=%s tools=%s",
        source_cfg["rule"],
        normalize_model_name(body.get("model")),
        model_cfg["upstream_model"],
        want_stream,
        len(messages),
        "tools" in body,
    )
    if want_stream:
        return StreamingResponse(
            handler.stream_chat_to_ollama(source_cfg, model, payload),
            media_type="application/x-ndjson",
        )

    # Non-streaming: consume SSE via the same generator and assemble final response
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    tool_calls: List[Dict[str, Any]] = []
    done_reason = "stop"
    async for line_json in handler.stream_chat_to_ollama(source_cfg, model, payload):
        obj = json.loads(line_json)
        message = obj.get("message") or {}
        full_content += message.get("content") or ""
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                tool_calls.append(tool_call)
        if obj.get("done"):
            prompt_tokens = obj.get("prompt_eval_count", 0)
            completion_tokens = obj.get("eval_count", 0)
            done_reason = str(obj.get("done_reason") or done_reason)

    response_message: Dict[str, Any] = {"role": "assistant", "content": full_content}
    if tool_calls:
        response_message["tool_calls"] = tool_calls

    result = make_chat_response(model, response_message, {
        "done_reason": done_reason,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": 0,
    })
    return JSONResponse(content=result)


@app.post("/api/embed")
async def embed(request: Request) -> Any:
    body = await parse_request_json(request)
    model_cfg = await resolve_model_config(body.get("model"))
    source_cfg = resolve_source_config(model_cfg["source"])
    handler = source_cfg["handler"]
    require_rule_capability(source_cfg, "supports_embeddings", "embeddings")
    input_data = body.get("input") or body.get("prompt")
    if input_data is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="input is required")

    payload: Dict[str, Any] = {"model": model_cfg["upstream_model"], "input": input_data}
    if "truncate" in body:
        payload["truncate"] = body["truncate"]
    flatten_options(body, payload)

    data = await handler.proxy_json(source_cfg, "embeddings", payload)
    return JSONResponse(content=data)


@app.post("/api/embeddings")
async def embeddings(request: Request) -> Any:
    return await embed(request)


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request) -> Any:
    body = await parse_request_json(request)
    model_cfg = await resolve_model_config(body.get("model"))
    source_cfg = resolve_source_config(model_cfg["source"])
    handler = source_cfg["handler"]
    model = model_cfg["name"]
    if "tools" in body:
        require_rule_capability(source_cfg, "supports_tools", "tools")
    body["model"] = model_cfg["upstream_model"]
    want_stream = body.get("stream", False)

    logger.info(
        "v1/chat/completions rule=%s model=%s want_stream=%s messages=%s",
        source_cfg["rule"],
        model,
        want_stream,
        len(body.get("messages") or []),
    )

    if want_stream:
        return StreamingResponse(
            handler.stream_chat_openai(source_cfg, model, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: proxy and return JSON with model name rewritten
    data = await handler.chat_json_openai(source_cfg, model, body)
    return JSONResponse(content=data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ollama_proxy:app", host="0.0.0.0", port=PORT, reload=False)
