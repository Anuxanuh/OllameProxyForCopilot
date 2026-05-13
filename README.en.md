# Ollama Compatibility Proxy

[中文文档](README.md)

A middleware that translates upstream LLM APIs into the Ollama HTTP API, enabling VS Code's GitHub Copilot to access custom sources via the Ollama interface.

## Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Edit `proxy_config.json` (configure upstream sources)
3. Start the proxy
```bash
python ollama_proxy.py
```

Optional startup arguments:

| Argument          | Description         | Default             |
| ----------------- | ------------------- | ------------------- |
| `--config <path>` | Path to config file | `proxy_config.json` |
| `--port <port>`   | Listen port         | `11434`             |

Examples:

```bash
# Use custom config and port
python ollama_proxy.py --config ./my_config.json --port 8080
```
4. Set the Ollama address in VS Code Copilot to:
```
http://127.0.0.1:11434
```

## Configuring Upstream Sources

### Fully OpenAI-Compatible Source

```json
{
	"sources": {
		"my-source": {
			"rule": "openai",
			"base_url": "https://api.openai-compatible.example/v1",
			"api_key": "sk-xxx"
		}
	}
}
```

The proxy automatically uses `/chat/completions`, `/embeddings`, and `/models` — no extra configuration needed.

### Fully Anthropic-Compatible Source

```json
{
	"default_model": "claude-sonnet-4-20250514",
	"sources": {
		"my-source": {
			"rule": "anthropic",
			"base_url": "https://api.anthropic.com/v1",
			"api_key": "sk-ant-xxx"
		}
	}
}
```

The proxy automatically uses `/messages` for chat, discovers models from `/models`, and attaches `x-api-key` and `anthropic-version` headers.

### Partially OpenAI-Compatible Source

When the upstream only partially supports the OpenAI API, use `openai_compatible_partial` and declare models manually:

```json
{
	"default_model": "qwen3.6-plus",
	"sources": {
		"aliyun": {
			"rule": "openai_compatible_partial",
			"base_url": "https://example.com/compatible-mode/v1",
			"api_key": "your-api-key",
			"models": {
				"qwen3.6-plus": "qwen3.6-plus",
				"glm-5": "glm-5"
			}
		}
	}
}
```

### DeepSeek Source

Via Anthropic Messages path (with thinking replay):

```json
{
	"sources": {
		"deepseek": {
			"rule": "deepseek_anthropic",
			"base_url": "https://api.deepseek.com",
			"api_key": "sk-xxx"
		}
	}
}
```

Via OpenAI-compatible path:

```json
{
	"sources": {
		"deepseek": {
			"rule": "deepseek_openai",
			"base_url": "https://api.deepseek.com",
			"api_key": "sk-xxx"
		}
	}
}
```

> DeepSeek thinking/reasoning replay caches are persisted under the local `Logs` directory. Context is automatically recovered after a proxy restart.

### Disabling a Source

`sources.<name>.enable` defaults to `true`. Set to `false` to skip the source at startup:

```json
{
	"sources": {
		"my-source": {
			"enable": false,
			"rule": "openai",
			"base_url": "...",
			"api_key": "..."
		}
	}
}
```

## Rule Types

| Rule                        | Auto Model Discovery       | Description                 |
| --------------------------- | -------------------------- | --------------------------- |
| `openai`                    | ✅ Queries `/models`        | Fully OpenAI API compatible |
| `openai_compatible_partial` | ❌ Must write `models`      | Partially OpenAI compatible |
| `anthropic`                 | ✅ Queries `/models`        | Anthropic Messages API      |
| `deepseek_anthropic`        | ✅ Reads DeepSeek `/models` | DeepSeek via Anthropic path |
| `deepseek_openai`           | ✅ Reads DeepSeek `/models` | DeepSeek via OpenAI path    |

> `deepseek` is a backward-compatible alias for `deepseek_anthropic`; `openai_aliyun` is an alias for `openai_compatible_partial`.

## Supported Endpoints

| Method | Path                   | Description                   |
| ------ | ---------------------- | ----------------------------- |
| GET    | `/api/version`         | Version info                  |
| GET    | `/api/tags`            | Model list                    |
| GET    | `/api/models`          | Model list                    |
| GET    | `/api/models/{model}`  | Model details                 |
| POST   | `/api/models/{model}`  | Model details (legacy compat) |
| POST   | `/api/generate`        | Text generation               |
| POST   | `/api/chat`            | Chat completion               |
| POST   | `/api/embed`           | Embedding                     |
| POST   | `/api/embeddings`      | Embedding                     |
| GET    | `/api/ps`              | Process status                |
| POST   | `/api/show`            | Model info                    |
| GET    | `/v1/models`           | Model list (OpenAI style)     |
| GET    | `/v1/models/{model}`   | Model details (OpenAI style)  |
| POST   | `/v1/chat/completions` | Chat (OpenAI style)           |

## Advanced Configuration

### `rule_config` Fields

Each source can override rule defaults via `rule_config`:

```json
{
	"sources": {
		"my-source": {
			"rule": "openai",
			"base_url": "...",
			"api_key": "...",
			"rule_config": {
				"supports_embeddings": false,
				"supports_tools": false,
				"use_bearer_auth": true,
				"request_headers": {
					"x-custom": "value"
				}
			}
		}
	}
}
```

| Field                 | Type   | Default                        | Description                                                                  |
| --------------------- | ------ | ------------------------------ | ---------------------------------------------------------------------------- |
| `supports_embeddings` | bool   | `true`                         | Whether to allow `/api/embed` and `/api/embeddings` (returns 501 if `false`) |
| `supports_tools`      | bool   | Rule-defined                   | Whether to allow `tools` in requests (returns 501 if `false`)                |
| `use_bearer_auth`     | bool   | `true` (`false` for anthropic) | Whether to convert `api_key` to `Authorization: Bearer`                      |
| `request_headers`     | object | `{}`                           | Rule-level extra headers; supports `{{api_key}}` placeholder                 |

### Header Merge Rules

1. `api_key` → `Authorization: Bearer ...` (controlled by `use_bearer_auth`)
2. Source-level `headers` as common headers
3. `rule_config.request_headers` as rule-level protocol headers
4. Conflicting keys are overridden by the later source

### Capability Exposure

- When `supports_tools = true` and a model has no individual capability declaration, `tools` is exposed by default
- `anthropic` rule enables `supports_tools` by default and translates between OpenAI-style and Anthropic-style tool calls
- When `supports_tools = false`, `/api/show` will not expose `tools` even if `meta.capabilities` includes them
- Unimplemented protocol paths return 501 explicitly

## Adding a New Rule

1. Create a new handler file under [rule_handlers](rule_handlers)
2. Implement the `RuleHandler` interface
3. Register it in [rule_handlers/__init__.py](rule_handlers/__init__.py)
4. Set the new `rule` name on a source in [proxy_config.json](proxy_config.json)

The main file [ollama_proxy.py](ollama_proxy.py) doesn't need additional protocol branches.

## Design Notes

This is a middleware that translates upstream LLM APIs into the Ollama HTTP API, enabling VS Code's GitHub Copilot to access custom sources via the Ollama interface.

The proxy uses a "source-based rule dispatching to handlers" architecture. Different upstreams declare which protocol rule they belong to; the main flow has no hardcoded branches. Adding a new protocol only requires creating a handler file and registering it.

The DeepSeek thinking replay mechanism is protocol-agnostic: both `deepseek_anthropic` and `deepseek_openai` must carry the prior reasoning payload into the next request, because DeepSeek enforces continuity for thinking mode while Copilot reaches this proxy via OpenAI-style requests.
