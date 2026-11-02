# nonebot-plugin-group-chat-memory

为 NoneBot2 OneBot v11 群聊提供按群隔离的长期向量记忆、@/回复对话和按群概率随机回复。插件使用 SQLite + sqlite-vec 保存原文、分块、向量和群配置，聊天与 Embedding 均请求可配置的 OpenAI 兼容 HTTP API。

## 环境与安装

- Python 3.10+
- NoneBot2 2.2+
- OneBot v11 适配器与对应的 OneBot 协议端

选择一种安装方式：

```bash
# nb-cli
nb plugin install nonebot-plugin-group-chat-memory
nb adapter install nonebot-adapter-onebot

# pip
pip install nonebot-plugin-group-chat-memory nonebot-adapter-onebot

# Poetry
poetry add nonebot-plugin-group-chat-memory nonebot-adapter-onebot
```

nb-cli 通常会写入插件配置。使用 pip 或 Poetry 手动安装后，确认项目的 `pyproject.toml` 已加载插件：

```toml
[tool.nonebot]
plugins = ["nonebot_plugin_group_chat_memory"]
```

并在 NoneBot 启动代码中注册 OneBot v11 Adapter。仓库中的 `.env.example` 使用 `DRIVER=~fastapi`；该示例驱动不是本插件的运行时依赖，需要在你的 Bot 项目中另行安装：

```bash
pip install "nonebot2[fastapi]"
# 或
poetry add "nonebot2[fastapi]"
```

## 快速配置

复制 `.env.example` 为 `.env`，至少填写聊天接口、聊天密钥和 Embedding 密钥，并设置 SUPERUSER：

```dotenv
SUPERUSERS=["10001"]
COMMAND_START=["/"]

GROUP_CHAT_MEMORY_CHAT_BASE_URL=https://your-chat-provider.example/v1
GROUP_CHAT_MEMORY_CHAT_API_KEY=sk-your-chat-key
GROUP_CHAT_MEMORY_CHAT_MODEL=your-provider-model-id
GROUP_CHAT_MEMORY_EMBEDDING_API_KEY=sk-your-embedding-key
```

`base_url` 填到 API 根路径（通常包含 `/v1`），插件分别请求其下的 `/chat/completions` 和 `/embeddings`。默认聊天模型标识是 `deepseek-v4.1-flash`，但**模型标识必须以实际服务商支持的值为准**；如果服务商没有该标识，必须修改 `GROUP_CHAT_MEMORY_CHAT_MODEL`。

## 全部配置字段

以下字段都可写在 NoneBot `.env` 中；集合类值使用 JSON 格式。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GROUP_CHAT_MEMORY_CHAT_BASE_URL` | 空 | OpenAI 兼容聊天 API 根地址；实际聊天前必填。 |
| `GROUP_CHAT_MEMORY_CHAT_API_KEY` | 空 | 聊天 API 密钥；实际聊天前必填，不会写入数据库。 |
| `GROUP_CHAT_MEMORY_CHAT_MODEL` | `deepseek-v4.1-flash` | 聊天模型标识，按服务商实际支持值配置。 |
| `GROUP_CHAT_MEMORY_CHAT_TIMEOUT` | `60` | 单次聊天 HTTP 请求超时秒数，必须大于 0。 |
| `GROUP_CHAT_MEMORY_EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容 Embedding API 根地址。 |
| `GROUP_CHAT_MEMORY_EMBEDDING_API_KEY` | 空 | Embedding API 密钥；实际向量化前必填。 |
| `GROUP_CHAT_MEMORY_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型标识。 |
| `GROUP_CHAT_MEMORY_EMBEDDING_DIMENSIONS` | `1024` | 向量维度，必须大于 0，必须与模型及既有数据库一致。 |
| `GROUP_CHAT_MEMORY_EMBEDDING_TIMEOUT` | `10` | 后台分块 Embedding HTTP 请求超时秒数，必须大于 0。 |
| `GROUP_CHAT_MEMORY_QUERY_TIMEOUT` | `3` | 回复前“查询向量 + 本地检索”的整体短超时秒数；超时后只用最近消息聊天。 |
| `GROUP_CHAT_MEMORY_DB_PATH_OVERRIDE` | 空 | 数据库路径覆盖，支持 `~`；留空时使用 nonebot-plugin-localstore 的本插件数据目录下 `group_chat_memory.db`。 |
| `GROUP_CHAT_MEMORY_RECENT_MESSAGES` | `20` | 每次回复最多读取的当前群最近消息数，至少为 1。 |
| `GROUP_CHAT_MEMORY_CHUNK_SIZE` | `10` | 每个长期记忆片段包含的消息数，至少为 1。 |
| `GROUP_CHAT_MEMORY_LONG_TERM_MEMORIES` | `5` | 每次回复最多检索的长期片段数；设为 0 可停用长期检索。 |
| `GROUP_CHAT_MEMORY_CONTEXT_MAX_CHARS` | `6000` | 发送给聊天模型的上下文总字符预算，至少为 1；优先保留当前消息和最近消息。 |
| `GROUP_CHAT_MEMORY_DEFAULT_PROBABILITY` | `0` | 新群随机回复概率，范围 0～100；已有群使用数据库中的概率。 |
| `GROUP_CHAT_MEMORY_IGNORED_QQ` | `[]` | 完全忽略的 QQ 号 JSON 数组，例如 `[12345, 67890]`；这些账号的消息不记忆也不触发回复。 |
| `GROUP_CHAT_MEMORY_SYSTEM_PROMPT` | `你是一个群聊助手。回答自然、简洁，像群友一样交流。只使用提供的聊天记录和长期记忆，不要假装记得没有出现过的内容。` | 自定义系统提示词；设为空时仅使用内置的发言者区分和当前消息约束。 |

更改 `GROUP_CHAT_MEMORY_EMBEDDING_DIMENSIONS` 不会迁移既有向量。如果与数据库首次创建时记录的维度不同，插件会拒绝打开该数据库；请恢复原配置，或备份后另行重建数据库。

## 使用与权限

在群里 @Bot 或回复 Bot 的消息即可对话；这两种定向消息必定进入回复队列，不受随机概率影响。普通纯文本按当前群概率触发，默认 0% 表示关闭随机回复，100% 表示每条符合条件的普通文本都回复。

| 命令 | 权限 | 行为 |
|---|---|---|
| `/回复概率` | 仅 SUPERUSER | 查询当前群概率，显示一位小数。 |
| `/回复概率 <0～100>` | 仅 SUPERUSER | 设置当前群概率，最多一位小数，例如 `/回复概率 5.5`。 |
| `/清空群记忆` | 仅 SUPERUSER | 删除当前群原始消息、分块和向量；保留当前群概率，不影响其他群。 |

非 SUPERUSER 会收到简短权限提示。命令仅在群聊中生效，不进入记忆，也不触发随机回复。SUPERUSER 通过 NoneBot 的 `SUPERUSERS` 配置，而不是群管理员身份确定。

同一群的模型请求串行执行：定向请求逐条排队；等待中的随机请求只保留最新一条，避免集中补发。不同群可以并行处理。

## 记忆范围与隐私

**这是全群文本记忆，不只是与 Bot 的对话。** 符合条件的群成员纯文本、定向文本以及 Bot 成功发送的模型回复都会保存；其他机器人账号也会保存，除非列入 `GROUP_CHAT_MEMORY_IGNORED_QQ`。命令、私聊、临时会话、匿名消息、Bot 自身消息、纯非文本及含不支持消息段的内容不会进入记忆。

每积累 10 条消息（可由 `GROUP_CHAT_MEMORY_CHUNK_SIZE` 修改）形成的完整群聊片段会发送给配置的**第三方 Embedding 服务商**；每次回复前的当前查询文本也会发送给 Embedding 服务商。回复时，当前群最近消息、检索到的长期片段和当前输入会发送给聊天模型服务商。不同群的记忆和检索在数据库层面隔离，但同一群的成员内容可能进入该群后续模型上下文。

原文与向量保存在本地 SQLite 文件中，数据库本身不加密。API 密钥只从环境变量或 NoneBot 配置读取，不写入数据库。请根据群成员授权、服务商数据政策和适用法律决定是否启用，并保护好数据库与备份。

## 缺失凭据与故障降级

- 零配置导入插件和 `nonebot.load_plugin("nonebot_plugin_group_chat_memory")` 均可成功，不要求先填写 API 密钥。
- 聊天地址、密钥或模型为空时，收到定向/随机触发后只发送通用错误提示；传入消息仍会保存，不会把密钥或内部响应发到群里。
- Embedding 密钥缺失或后台向量化失败时，原始消息仍保留，未完成片段等待后续重试或重启恢复，但暂时不能用于长期检索。
- 查询 Embedding 超时或失败时，回复会降级为最近消息上下文，不阻断聊天。
- 数据库向量维度不匹配属于配置/数据错误，应先按备份说明处理，而不是直接删除生产数据。

## 随机回复风险

随机回复可能接错话题、打扰群友、放大敏感内容传播，并产生额外模型费用和限流压力；100% 会对每条符合条件的普通文本调用模型。“等待中只保留最新随机请求”只能减少积压，不能限制每条新消息继续触发。建议保持默认 0%，先在测试群验证，再从较低概率开始，并用忽略列表排除已知机器人。

## 数据库备份

默认数据库位于 nonebot-plugin-localstore 为本插件分配的数据目录，文件名为 `group_chat_memory.db`；配置 `GROUP_CHAT_MEMORY_DB_PATH_OVERRIDE` 时使用该路径。插件启用了 WAL，运行中可能出现同名 `-wal` 和 `-shm` 文件。

运行中备份推荐使用 SQLite CLI 的在线备份命令，它会正确处理 WAL：

```bash
sqlite3 /path/to/group_chat_memory.db ".backup '/path/to/group_chat_memory.backup.db'"
```

也可以先正常停止 Bot，再复制数据库文件及存在的 `-wal`、`-shm` 文件。**不要在运行中只复制主 `.db` 文件**，否则可能得到不完整备份。恢复后必须使用与原数据库相同的 Embedding 维度配置。

## 开发验证

```bash
poetry install
poetry run pytest -v
poetry run ruff check .
poetry run python -m compileall nonebot_plugin_group_chat_memory
poetry build
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
