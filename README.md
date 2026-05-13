# Ollama 兼容代理

[English Documentation](README.en.md)

这是一个把上游大模型 API 转成 Ollama HTTP API 的中转层，让 VS Code 的 GitHub Copilot 通过 Ollama 接口访问自定义来源。

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```
2. 编辑配置文件 `proxy_config.json`（配置上游来源）
3. 启动代理
```bash
python ollama_proxy.py
```

可选启动参数：

| 参数              | 说明             | 默认值              |
| ----------------- | ---------------- | ------------------- |
| `--config <路径>` | 指定配置文件路径 | `proxy_config.json` |
| `--port <端口>`   | 指定监听端口     | `11434`             |

示例：

```bash
# 使用自定义配置和端口
python ollama_proxy.py --config ./my_config.json --port 8080
```
4. VS Code Copilot 中设置 Ollama 地址为：
```
http://127.0.0.1:11434
```

## 配置上游来源

### OpenAI 完整兼容源

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

代理自动使用 `/chat/completions`、`/embeddings`、`/models`，无需额外配置。

### Anthropic 完整兼容源

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

代理自动走 `/messages` 处理聊天，从 `/models` 发现模型，附带 `x-api-key` 和 `anthropic-version` 请求头。

### 部分兼容 OpenAI 源

上游只兼容部分 OpenAI API 时改用 `openai_compatible_partial`，手动声明模型：

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

### DeepSeek 来源

走 Anthropic Messages 路径（含 thinking 回放）：

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

走 OpenAI 兼容路径：

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

> DeepSeek 的 thinking/reasoning 回放缓存持久化到本地 `Logs` 目录，代理重启后自动恢复上下文。

### 禁用来源

`sources.<name>.enable` 默认为 `true`，设为 `false` 则跳过该来源：

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

## 规则类型

| 规则                        | 自动发现模型            | 说明                       |
| --------------------------- | ----------------------- | -------------------------- |
| `openai`                    | ✅ 自动请求 `/models`    | 完整 OpenAI API 兼容       |
| `openai_compatible_partial` | ❌ 需手写 `models`       | 部分兼容 OpenAI            |
| `anthropic`                 | ✅ 自动请求 `/models`    | Anthropic Messages API     |
| `deepseek_anthropic`        | ✅ 读 DeepSeek `/models` | DeepSeek 走 Anthropic 路径 |
| `deepseek_openai`           | ✅ 读 DeepSeek `/models` | DeepSeek 走 OpenAI 路径    |

> `deepseek` 是 `deepseek_anthropic` 的兼容别名；`openai_aliyun` 是 `openai_compatible_partial` 的兼容别名。

## 支持接口

| 方法 | 路径                   | 说明                    |
| ---- | ---------------------- | ----------------------- |
| GET  | `/api/version`         | 版本查询                |
| GET  | `/api/tags`            | 模型列表                |
| GET  | `/api/models`          | 模型列表                |
| GET  | `/api/models/{model}`  | 模型详情                |
| POST | `/api/models/{model}`  | 模型详情（兼容旧调用）  |
| POST | `/api/generate`        | 文本生成                |
| POST | `/api/chat`            | 对话                    |
| POST | `/api/embed`           | 嵌入                    |
| POST | `/api/embeddings`      | 嵌入                    |
| GET  | `/api/ps`              | 进程状态                |
| POST | `/api/show`            | 模型信息                |
| GET  | `/v1/models`           | 模型列表（OpenAI 风格） |
| GET  | `/v1/models/{model}`   | 模型详情（OpenAI 风格） |
| POST | `/v1/chat/completions` | 对话（OpenAI 风格）     |

## 高级配置

### `rule_config` 字段

每个来源可通过 `rule_config` 覆盖规则默认行为：

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

| 字段                  | 类型   | 默认值                           | 说明                                                             |
| --------------------- | ------ | -------------------------------- | ---------------------------------------------------------------- |
| `supports_embeddings` | bool   | `true`                           | 是否允许 `/api/embed` 和 `/api/embeddings`（`false` 时返回 501） |
| `supports_tools`      | bool   | 规则自定                         | 是否允许 tools 调用（`false` 时返回 501）                        |
| `use_bearer_auth`     | bool   | `true`（anthropic 默认 `false`） | 是否将 `api_key` 转为 `Authorization: Bearer`                    |
| `request_headers`     | object | `{}`                             | 规则级附加请求头，支持 `{{api_key}}` 占位符                      |

### 请求头合并规则

1. `api_key` 转为 `Authorization: Bearer ...`（受 `use_bearer_auth` 控制）
2. 来源级 `headers` 作为公共头
3. `rule_config.request_headers` 作为规则级协议头
4. 同名键时后者覆盖前者

### 能力暴露

- `supports_tools = true` 且模型未单独声明能力时，默认暴露 `tools`
- `anthropic` 规则默认开启 `supports_tools`，自动在 OpenAI tools ↔ Anthropic tools 间互转
- `supports_tools = false` 时，即使模型 `meta.capabilities` 包含 `tools`，`/api/show` 也不会暴露
- 未实现的协议路径，handler 显式返回 501

## 新增规则

1. 在 [rule_handlers](rule_handlers) 下新建 handler 文件
2. 实现 `RuleHandler` 接口
3. 在 [rule_handlers/__init__.py](rule_handlers/__init__.py) 注册
4. 在 [proxy_config.json](proxy_config.json) 的 source 上使用新的 `rule` 名称

主文件 [ollama_proxy.py](ollama_proxy.py) 无需追加协议分支。

## 设计说明

这是一个把上游大模型 API 转成 Ollama HTTP API 的中转层，让 VS Code 的 GitHub Copilot 通过 Ollama 接口访问自定义来源。

代理采用"按来源规则分发 handler"架构。不同上游声明自己所属的协议规则，主流程不堆分支，新增协议只需添加 handler 文件并注册。

DeepSeek thinking 回放机制是"跨协议统一"的：无论来源走 `deepseek_anthropic` 还是 `deepseek_openai`，都需要把上一轮的推理内容原样带回下一轮请求，因为 DeepSeek 对 thinking 模式有连续上下文约束，而 Copilot 进入本代理时走的是 OpenAI 风格接口。