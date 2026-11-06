# nonebot-plugin-group-chat-memory

为 NoneBot2 OneBot v11 群聊提供按群隔离的长期向量记忆、@/回复对话和按群概率随机回复。插件使用 SQLite 保存原文、分块、向量和群配置，聊天与 Embedding 均请求可配置的 OpenAI 兼容 HTTP API。

## 环境与安装

- Python 3.10+
- NoneBot2 2.2+
- OneBot v11 适配器与对应的 OneBot 协议端

优先使用 sqlite-vec 原生向量索引。如果当前 Python 的 `sqlite3` 不支持加载扩展（例如部分 python.org/Homebrew Python 3.13/3.14 构建），插件会自动改用 SQLite BLOB 保存向量并在 Python 中计算余弦相似度；启动日志会给出提示，功能保持可用。旧 sqlite-vec 数据库中缺少 BLOB 向量的已就绪片段会自动重新排队向量化。

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

复制 `.env.example` 为 `.env` 并设置 SUPERUSER：

```dotenv
SUPERUSERS=["10001"]
COMMAND_START=["/"]
```

启动 Bot 后，由 SUPERUSER 在**私聊**中依次设置四项运行时配置：

```text
/设置聊天地址 https://your-chat-provider.example/v1
/设置聊天密钥 sk-your-chat-key
/设置聊天模型 your-provider-model-id
/设置向量密钥 sk-your-embedding-key
```

聊天地址填到 API 根路径（通常包含 `/v1`），插件分别请求其下的 `/chat/completions` 和 `/embeddings`。默认聊天模型标识是 `deepseek-v4.1-flash`，但**模型标识必须以实际服务商支持的值为准**。这四项保存到 SQLite，不再读取同名环境变量，更新后立即生效且重启后保留。使用 `/模型配置` 可查询脱敏状态。

使用 OpenCode Go 直连 API 时，地址是 `https://opencode.ai/inference/go/openai/v1`，模型是 `deepseek-v4.1-flash`。插件会发送自己的 User-Agent，并对 OpenCode 请求附带按群稳定的 `x-opencode-session`，避免被网关当作通用 SDK 流量。

## 其余环境配置字段

模型与接口凭证字段不再通过 env 设置。其余以下字段可写在 NoneBot `.env` 中；集合类值使用 JSON 格式。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GROUP_CHAT_MEMORY_CHAT_TIMEOUT` | `60` | 单次聊天 HTTP 请求超时秒数，必须大于 0。 |
| `GROUP_CHAT_MEMORY_EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容 Embedding API 根地址。 |
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
| `GROUP_CHAT_MEMORY_HOURLY_REQUEST_LIMIT` | `10` | 新群默认每小时模型请求上限，范围 1～10000；已有群可用 `/每小时次数` 修改。 |
| `GROUP_CHAT_MEMORY_MEME_ENABLED` | `true` | 是否启用回复附图；缺少 ID/KEY 时静默跳过，不影响文字回复。 |
| `GROUP_CHAT_MEMORY_MEME_API_URL` | `https://cn.apihz.cn/api/img/apihzbqbbaidu.php` | 表情包检索接口，默认百度通道（召回率高于官方库）；可换成 `apihzbqb.php`（官方库）或 `apihzbqbsougou.php`（搜狗）。 |
| `GROUP_CHAT_MEMORY_IGNORED_QQ` | `[]` | 完全忽略的 QQ 号 JSON 数组，例如 `[12345, 67890]`；这些账号的消息不记忆也不触发回复。 |
| `GROUP_CHAT_MEMORY_SYSTEM_PROMPT` | 内置“馒头 + 群友口头禅”人格 | 自定义系统提示词并完全覆盖默认人格；设为空时仅使用内置的发言者区分和当前消息约束。 |

更改 `GROUP_CHAT_MEMORY_EMBEDDING_DIMENSIONS` 不会迁移既有向量。如果与数据库首次创建时记录的维度不同，插件会拒绝打开该数据库；请恢复原配置，或备份后另行重建数据库。

## 使用与权限

在群里 @Bot 或回复 Bot 的消息即可对话；这两种定向消息必定进入回复队列，不受随机概率影响。普通纯文本按当前群概率触发，默认 0% 表示关闭随机回复，100% 表示每条符合条件的普通文本都回复。

默认人格为“馒头”：Bot 以长期群友的身份说话，句子短，爱用“确实”“还真是”“坏了”“何意味”“可爱捏”这类群内口头禅，会吐槽、接梗和复读，但正经问题仍优先给准确答案。插件命令优先级为 `-1001`，群聊消息优先级为 `-1000`；@Bot 或回复 Bot 处理完成后会停止事件继续传播，避免其他 NoneBot 插件同时响应。普通群聊仍不会阻塞其他插件。

每个群默认每小时最多触发 10 次模型请求，采用滚动 60 分钟窗口。点名超额会收到简短提示，普通随机回复超额静默跳过；消息仍写入记忆，命令和 Embedding 不消耗次数。好感榜触发和必回名单触发的随机回复不占用也不受这个上限约束。

安装并加载 `nonebot-plugin-mantou-affection` 后，好感榜（该插件默认取前 10 名）上的群友发普通文本时，触发概率至少为当前群 `/好感概率` 的值（默认 30%）。若当前群的 `/回复概率` 更高，则沿用更高值；两者只做一次随机判定。`/好感开关 关` 可临时停用而不丢失概率设置。好感榜读取失败或插件未加载时自动退回群概率，不影响正常聊天。榜单结果缓存 30 秒，避免每条消息都读文件。

这两项设置保存在 SQLite 的当前群记录中，不读取环境变量。

好感榜前 3 名触发回复时，会在文字后追加一张按回复内容检索的表情包；`/好感开关 关` 或概率为 0 时不附图。SUPERUSER 的普通文本 100% 触发并附图，不受群概率和每小时额度限制；`/超管必回 关` 可以关掉这个必定触发，让 SUPERUSER 的发言退回按群概率和好感榜概率判定（`/超管必回` 查询、`/超管必回 开` 恢复，按群保存，默认开启）。@Bot、回复 Bot、必回名单成员和榜内第 4 名及以后仍只回文字。表情包走接口盒子检索（凭证用 `/设置表情包ID` 与 `/设置表情包密钥` 设置，首次启动默认使用内置 ID/KEY），额度为每分钟 10 次，插件做全局节流和 10 分钟关键词缓存；取图失败只发文字，不影响记忆写入。

必回名单用来保证指定群友说话必定触发回复，不受群概率影响，也不消耗每小时额度。群聊里直接用命令增删；也可以在私聊中发 `/必回名单`，Bot 会按最近活跃顺序列出有记忆的群，回复序号或群号选群，再回复 QQ 号（多个用空格、逗号或换行分隔）完成添加。私聊交互 5 分钟内有效，随时可回复“取消”退出。名单按群隔离，`/清空群记忆` 不会清掉名单。

| 命令 | 权限 | 行为 |
|---|---|---|
| `/设置聊天地址 <URL>` | 仅私聊 SUPERUSER | 设置全局 OpenAI 兼容聊天 API 根地址。 |
| `/设置聊天密钥 <密钥>` | 仅私聊 SUPERUSER | 设置全局聊天 API 密钥。 |
| `/设置聊天模型 <模型标识>` | 仅私聊 SUPERUSER | 设置全局聊天模型。 |
| `/设置向量密钥 <密钥>` | 仅私聊 SUPERUSER | 设置全局 Embedding API 密钥。 |
| `/设置表情包ID <数字ID>` | 仅私聊 SUPERUSER | 设置接口盒子开发者数字 ID，必须是数字。 |
| `/设置表情包密钥 <KEY>` | 仅私聊 SUPERUSER | 设置接口盒子通讯 KEY。 |
| `/模型配置` | SUPERUSER，私聊或群聊 | 查询全局配置状态，密钥只显示掩码。 |
| `/回复概率` | 仅 SUPERUSER | 查询当前群概率，显示一位小数。 |
| `/回复概率 <0～100>` | 仅 SUPERUSER | 设置当前群概率，最多一位小数，例如 `/回复概率 5.5`。 |
| `/每小时次数` | 仅 SUPERUSER | 查询当前群每小时模型请求上限。 |
| `/每小时次数 <1～10000>` | 仅 SUPERUSER | 设置当前群每小时请求上限，例如 `/每小时次数 20`。 |
| `/好感概率` | 仅 SUPERUSER | 查询当前群好感榜成员的触发概率。 |
| `/好感概率 <0～100>` | 仅 SUPERUSER | 设置当前群好感榜成员的触发概率，例如 `/好感概率 30`。 |
| `/好感开关` | 仅 SUPERUSER | 查询当前群好感榜触发是否开启。 |
| `/好感开关 <开/关>` | 仅 SUPERUSER | 开启或关闭当前群好感榜触发，也接受 `on/off`。 |
| `/超管必回` | 仅 SUPERUSER | 查询当前群 SUPERUSER 发言是否必定触发回复。 |
| `/超管必回 <开/关>` | 仅 SUPERUSER | 开启或关闭当前群 SUPERUSER 必定触发，也接受 `on/off`；关闭后 SUPERUSER 的普通文本按群概率和好感榜概率判定。 |
| `/必回名单` | 仅 SUPERUSER | 群聊查询本群名单；私聊进入“列群 → 选群 → 填 QQ”交互。 |
| `/必回名单 添加 <@群友/QQ 号…>` | 仅 SUPERUSER，群聊 | 把群友加入本群必回名单，可一次多个。 |
| `/必回名单 删除 <@群友/QQ 号…>` | 仅 SUPERUSER，群聊 | 把群友移出本群必回名单。 |
| `/必回名单 清空` | 仅 SUPERUSER，群聊 | 清空本群必回名单。 |
| `/清空群记忆` | 仅 SUPERUSER | 删除当前群原始消息、分块、向量和请求计数；保留概率、每小时上限和必回名单，不影响其他群。 |

非 SUPERUSER 会收到简短权限提示。命令不进入记忆，也不触发随机回复。SUPERUSER 通过 NoneBot 的 `SUPERUSERS` 配置，而不是群管理员身份确定。

同一群的模型请求串行执行：定向请求逐条排队；等待中的随机请求只保留最新一条，避免集中补发。不同群可以并行处理。

## 记忆范围与隐私

**这是全群文本记忆，不只是与 Bot 的对话。** 符合条件的群成员纯文本、定向文本以及 Bot 成功发送的模型回复都会保存；其他机器人账号也会保存，除非列入 `GROUP_CHAT_MEMORY_IGNORED_QQ`。命令、私聊、临时会话、匿名消息、Bot 自身消息、纯非文本及含不支持消息段的内容不会进入记忆。

每积累 10 条消息（可由 `GROUP_CHAT_MEMORY_CHUNK_SIZE` 修改）形成的完整群聊片段会发送给配置的**第三方 Embedding 服务商**；每次回复前的当前查询文本也会发送给 Embedding 服务商。回复时，当前群最近消息、检索到的长期片段和当前输入会发送给聊天模型服务商。不同群的记忆和检索在数据库层面隔离，但同一群的成员内容可能进入该群后续模型上下文。

原文、向量与命令设置的 API 密钥保存在本地 SQLite 文件中，数据库本身不加密。请保护好数据库与备份，并限制文件访问权限。虽然设置密钥只允许私聊 SUPERUSER，但命令内容仍可能保留在你和 OneBot 协议端的聊天记录或日志中；不要把密钥发到群里，泄露后请立即更换。请根据群成员授权、服务商数据政策和适用法律决定是否启用。

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
