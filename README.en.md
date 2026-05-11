# Ollama Compatibility Proxy

[中文文档](README.md)

This is a middleware that translates upstream LLM APIs into the Ollama HTTP API, enabling VS Code's GitHub Copilot to access custom sources via the Ollama interface.

The proxy currently supports "source-based rule dispatching to handlers." This means different upstreams can declare themselves under different protocol rules, such as:

- `openai`
- `openai_compatible_partial`
- `anthropic`
- `deepseek`

Recommended understanding:

- `openai`: The upstream fully implements the OpenAI API. The proxy works with default paths and automatically discovers models from `/models`.
- `openai_compatible_partial`: The upstream only partially supports the OpenAI API. The proxy won't assume full model query capabilities, so you need to explicitly provide `models`.
- `anthropic`: The upstream fully implements the Anthropic Messages API. The proxy works with default paths and automatically discovers models from `/models`.
- `deepseek`: A DeepSeek source handled internally through the Anthropic Messages API flow, which better fits DeepSeek's thinking / reasoning workflow. Model discovery remains automatic, but it reads DeepSeek's root `/models` endpoint and supplements the missing context length field (DeepSeek v4 series default 1M context).
- `openai_aliyun`: An alias for legacy configurations, internally equivalent to `openai_compatible_partial`.

If you want to add a third protocol later, you don't need to keep adding branches in the main flow — just create a new handler file and register it.

## Usage

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Edit the configuration file [proxy_config.json](proxy_config.json)

3. Start the proxy

```bash
python ollama_proxy.py
```

4. Configure the Ollama address in VS Code Copilot as:

```text
http://127.0.0.1:11434
```

## Supported Endpoints

- `GET /api/version`
- `GET /api/tags`
- `GET /api/models`
- `GET /api/models/{model}`
- `POST /api/models/{model}`
- `POST /api/generate`
- `POST /api/chat`
- `POST /api/embed`
- `POST /api/embeddings`
- `GET /api/ps`
- `POST /api/show`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/chat/completions`

Standard OpenAI-style model query endpoints:

- `GET /v1/models`
- `GET /v1/models/{model}`

The proxy is compatible with both query endpoints. For backward compatibility with internal calls, `POST /api/models/{model}` and `POST /v1/models/{model}` are also retained.

For `openai` rules, the proxy automatically requests the upstream `/models` endpoint on first use and caches the result locally. This means fully OpenAI-compatible sources usually don't require manually written `models`.

For `anthropic` rules, the proxy also automatically requests the upstream `/models` endpoint. It defaults to including `x-api-key` and `anthropic-version` request headers, so standard Anthropic sources typically require minimal configuration.

For DeepSeek, prefer the `deepseek` rule so the proxy can route requests through the Anthropic Messages API flow internally. Model discovery remains automatic through DeepSeek's root `/models` endpoint, which avoids the `reasoning_content` round-trip problem seen on OpenAI-compatible paths.

## Configuration Structure

The core structure of [proxy_config.json](proxy_config.json) is as follows.

Minimal configuration for a fully OpenAI-compatible source:

```json
{
	"sources": {
		"openai-main": {
			"enable": true,
			"rule": "openai",
			"base_url": "https://api.openai-compatible.example/v1",
			"api_key": "sk-xxx"
		}
	}
}
```

`sources.<name>.enable` is an optional boolean and defaults to `true`. When set to `false`, that source is skipped at startup and will not participate in model discovery, model routing, or request forwarding.

This type of configuration doesn't require manually writing `paths`, `rule_config`, or `models`. The proxy automatically uses:

- `/chat/completions`
- `/embeddings`
- `/models`

If the upstream only partially supports the OpenAI API, switch to `openai_compatible_partial` and add only the minimal model mapping:

```json
{
	"default_model": "qwen3.6-plus",
	"sources": {
		"aliyun": {
			"enable": true,
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

If you need to override default paths, append request headers, or disable tools/embeddings, fill in `paths`, `headers`, and `rule_config` as needed — these are optional.

Minimal configuration for a fully Anthropic-compatible source:

```json
{
	"default_model": "claude-sonnet-4-20250514",
	"sources": {
		"anthropic-main": {
			"enable": true,
			"rule": "anthropic",
			"base_url": "https://api.anthropic.com/v1",
			"api_key": "sk-ant-xxx"
		}
	}
}
```

This configuration defaults to:

- Using `/messages` for chat
- Using `/models` for automatic model discovery
- Automatically attaching `x-api-key` and `anthropic-version`
- Enabling tool calls by default

## `rule_config` Field Reference

Each source can define a set of rule-level parameters tied to its `rule`.

- `supports_embeddings`: Whether to allow `/api/embed` and `/api/embeddings`. If `false`, the proxy returns 501 instead of forwarding the request upstream.
- `supports_tools`: Whether to allow `tools` in the request body. If `false`, `/api/chat` and `/v1/chat/completions` will return 501 when `tools` are present.
- `use_bearer_auth`: Whether to convert `api_key` to `Authorization: Bearer ...`. Defaults to `true`, but the `anthropic` rule disables it by default.
- `request_headers`: Rule-level additional request headers. These are merged with the source's `headers` on each upstream request, suitable for protocol-specific headers like Anthropic's `anthropic-version`.

These fields act as "override default behavior" switches. For `openai` rules, configuration is usually unnecessary. For `openai_compatible_partial`, you only need to add overrides when the upstream differs from default assumptions.

The same applies to `anthropic` rules: defaults already fit the standard Anthropic API. Overrides are only needed when proxying to Anthropic-style third-party compatible sources.

Values in `request_headers` support the `{{api_key}}` placeholder to reference the source's `api_key`. This is useful for Anthropic, which typically requires `x-api-key` rather than `Authorization: Bearer ...`.

Example Anthropic source:

```json
{
	"rule": "anthropic",
	"base_url": "https://api.anthropic.com/v1",
	"api_key": "sk-ant-xxx"
}
```

Merge rules:

- `api_key` is converted to `Authorization: Bearer ...`
- If `use_bearer_auth=false`, `Authorization: Bearer ...` is not automatically attached
- `headers` are source-level common headers
- `rule_config.request_headers` are rule-level protocol headers
- `rule_config.request_headers` can use `{{api_key}}` to reference the source's `api_key`
- When key names conflict, `rule_config.request_headers` overrides previous headers with the same name

## Capability Exposure Rules

The proxy not only validates `rule_config` during request forwarding but also reflects it in externally exposed model capabilities.

- If `supports_tools` is `true` and a model has no individually declared capabilities, the proxy defaults to exposing `tools`, so fully OpenAI-compatible sources don't need to repeat `capabilities` for each model.
- The `anthropic` rule enables `supports_tools` by default, converts OpenAI-style `tools` requests to Anthropic `tools`, and rewrites Anthropic `tool_use` responses back to OpenAI/Ollama-style tool calls.
- If `supports_tools` is `false`, `/api/show` will not expose `tools` even if the model's `meta.capabilities` includes them.
- If a rule hasn't implemented the corresponding protocol yet, the handler explicitly returns 501 instead of silently using the wrong protocol.

## Adding a New Rule

To support a new upstream protocol, follow these steps:

1. Create a new handler file in the [rule_handlers](rule_handlers) directory.
2. Implement the unified `RuleHandler` interface.
3. Register the handler in [rule_handlers/__init__.py](rule_handlers/__init__.py).
4. Set the new `rule` name on a source in [proxy_config.json](proxy_config.json).

This way, the main file [ollama_proxy.py](ollama_proxy.py) doesn't need additional protocol branches.
