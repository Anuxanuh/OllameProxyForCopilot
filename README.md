# Ollama 兼容代理

[English Documentation](README.en.md)

这是一个把上游大模型 API 转成 Ollama HTTP API 的中转层，目标是让 VS Code 里的 GitHub Copilot 通过 Ollama 接口访问自定义来源。

当前代理已经支持“按来源规则分发 handler”。也就是说，不同上游可以声明自己属于不同协议规则，例如：

- `openai`
- `openai_compatible_partial`
- `anthropic`
- `deepseek_anthropic`
- `deepseek_openai`

其中建议这样理解：

- `openai`: 上游基本完整实现 OpenAI API，代理会按默认路径工作，并自动从 `/models` 发现模型。
- `openai_compatible_partial`: 上游只兼容部分 OpenAI API，代理不会假设它支持完整模型查询等能力，因此需要你显式补 `models`。
- `anthropic`: 上游基本完整实现 Anthropic Messages API，代理会按默认路径工作，并自动从 `/models` 发现模型。
- `deepseek_anthropic`: DeepSeek 来源，内部按 Anthropic Messages API 路径转发；模型发现仍会读取 DeepSeek 主域的 `/models`，并为缺失上下文长度字段的模型补足默认值（DeepSeek v4 系列默认 1M 上下文）。
- `deepseek_openai`: DeepSeek 来源，走 OpenAI 兼容聊天路径；模型发现同样读取 DeepSeek 主域的 `/models`，并补足缺失的上下文长度字段。

DeepSeek 的 thinking 与 reasoning 回放缓存会持久化到本地 `Logs` 目录。代理重启后会优先尝试补回历史上下文；如果历史消息无法安全回放，则会裁剪不可重放的 assistant 历史，避免把不完整上下文继续转发到上游。

这套回放机制在业务上是“跨协议统一”的：无论来源最终走 `deepseek_anthropic` 还是 `deepseek_openai`，都需要把上一轮的推理内容原样带回下一轮请求。原因是 DeepSeek 对 thinking 模式有连续上下文约束；而 Copilot 进入本代理时走的是 OpenAI 风格接口，因此 `reasoning_content` 的回传能力需要在两条 DeepSeek 规则上同时成立。
- `deepseek`: 兼容旧配置的别名，内部等价于 `deepseek_anthropic`。
- `openai_aliyun`: 兼容旧配置的别名，内部等价于 `openai_compatible_partial`。

后续如果要接第三种协议，不需要继续在主流程里堆分支，只需要新增一个 handler 文件并在注册表里注册。

## 使用方法

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 编辑配置文件 [proxy_config.json](proxy_config.json)

3. 启动代理

```bash
python ollama_proxy.py
```

4. 在 VS Code Copilot 中把 Ollama 地址配置为

```text
http://127.0.0.1:11434
```

## 支持接口

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

其中标准 OpenAI 风格的模型查询接口是：

- `GET /v1/models`
- `GET /v1/models/{model}`

当前代理已经兼容这两个查询接口；同时为了兼容之前的内部调用，也保留了 `POST /api/models/{model}` 和 `POST /v1/models/{model}`。

对于 `openai` 规则，代理会在第一次需要模型信息时自动请求上游的 `/models`，然后把结果缓存到本地注册表。也就是说，完整 OpenAI 兼容源通常不需要手写 `models`。

对于 `anthropic` 规则，代理也会自动请求上游的 `/models`；同时默认附带 `x-api-key` 和 `anthropic-version` 请求头，因此标准 Anthropic 来源通常也只需要最少配置。

对于 DeepSeek，需要 Anthropic Messages 路径和完整 thinking 流程时，可使用 `deepseek_anthropic` 规则。

需要走 OpenAI 兼容接口时，可使用 `deepseek_openai`。这条规则同样会回放并持久化历史 `reasoning_content`，保证代理重启后仍可延续同一会话链路；建议把 `base_url` 配成 DeepSeek OpenAI 根地址，例如 `https://api.deepseek.com`。

## 配置结构

[proxy_config.json](proxy_config.json) 的核心结构如下。

完整 OpenAI 兼容源的最小配置：

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

`sources.<name>.enable` 是可选布尔值，默认 `true`。当设置为 `false` 时，该来源会在启动时被跳过，不参与模型发现、模型路由和请求转发。

这类配置不需要手写 `paths`、`rule_config`、`models`。代理会自动使用：

- `/chat/completions`
- `/embeddings`
- `/models`

如果上游只兼容部分 OpenAI API，则改用 `openai_compatible_partial`，并只补最少的模型映射：

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

如果需要覆盖默认路径、追加请求头、关闭工具或 embeddings，再按需填写 `paths`、`headers`、`rule_config`；不是必填项。

完整 Anthropic 兼容源的最小配置：

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

这类配置默认会：

- 走 `/messages` 处理聊天
- 走 `/models` 自动发现模型
- 自动附带 `x-api-key` 和 `anthropic-version`
- 默认允许工具调用

## `rule_config` 字段说明

每个来源都可以定义一组和 `rule` 绑定的规则级参数。

- `supports_embeddings`: 是否允许走 `/api/embed` 和 `/api/embeddings`。如果为 `false`，代理会直接返回 501，而不是把请求错误转发到上游。
- `supports_tools`: 是否允许请求体里出现 `tools`。如果为 `false`，`/api/chat` 和 `/v1/chat/completions` 收到 `tools` 时会直接返回 501。
- `use_bearer_auth`: 是否把 `api_key` 自动转成 `Authorization: Bearer ...`。默认是 `true`，但 `anthropic` 规则默认会关闭它。
- `request_headers`: 规则级附加请求头。它会在每次发往上游时和来源的 `headers` 合并，适合放协议要求的特殊头，比如 Anthropic 的 `anthropic-version`。

这些字段都属于“覆盖默认行为”的开关。对于 `openai` 规则，通常不需要配置；对于 `openai_compatible_partial`，只有当上游和默认假设不一致时才需要补。

对于 `anthropic` 规则也是同样的思路：默认值已经适配标准 Anthropic API，只有代理到 Anthropic 风格的第三方兼容源时才需要覆盖。

`request_headers` 里的值支持使用 `{{api_key}}` 占位符引用当前 source 的 `api_key`。这对 Anthropic 很实用，因为它通常需要的是 `x-api-key`，而不是 `Authorization: Bearer ...`。

Anthropic 来源示例：

```json
{
	"rule": "anthropic",
	"base_url": "https://api.anthropic.com/v1",
	"api_key": "sk-ant-xxx"
}
```

合并规则如下：

- `api_key` 会被转成 `Authorization: Bearer ...`
- 如果 `use_bearer_auth=false`，则不会自动附带 `Authorization: Bearer ...`
- `headers` 是来源级公共头
- `rule_config.request_headers` 是规则级协议头
- `rule_config.request_headers` 的值里可以使用 `{{api_key}}` 引用 source 的 `api_key`
- 当键名冲突时，`rule_config.request_headers` 会覆盖前面的同名头

## 能力暴露规则

代理不仅会在请求转发时校验 `rule_config`，也会把它反映到对外暴露的模型能力上。

- 如果 `supports_tools` 为 `true`，而模型没有单独声明能力，代理会默认把 `tools` 暴露出来，这样完整 OpenAI 兼容源不需要为每个模型重复写 `capabilities`。
- `anthropic` 规则默认开启 `supports_tools`，并会把 OpenAI 风格的 `tools` 请求转换为 Anthropic `tools`，同时把 Anthropic 的 `tool_use` 响应回写成 OpenAI/Ollama 风格的工具调用。
- 如果 `supports_tools` 为 `false`，即使模型 `meta.capabilities` 里写了 `tools`，`/api/show` 也不会再把 `tools` 暴露出去。
- 如果某个 rule 当前还没实现对应协议，handler 会显式返回 501，而不是静默走错协议。

## 新增一种规则

如果你要支持新的上游协议，推荐按下面的方式扩展：

1. 在 [rule_handlers](rule_handlers) 目录新增一个 handler 文件。
2. 实现统一的 `RuleHandler` 接口。
3. 在 [rule_handlers/__init__.py](rule_handlers/__init__.py) 注册这个 handler。
4. 在 [proxy_config.json](proxy_config.json) 的某个 source 上设置新的 `rule` 名称。

这样主文件 [ollama_proxy.py](ollama_proxy.py) 不需要再追加新的协议分支。