# Group Chat Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 NoneBot2 OneBot v11 插件，支持按群共享的长期向量记忆、点名对话、SUPERUSER 管理的随机回复概率和记忆清理。

**Architecture:** OneBot 事件层只负责提取群文本和权限判断；SQLite/sqlite-vec 负责原文、配置与向量持久化；后台记忆 worker 负责分块和 Embedding；回复协调器串行处理点名请求并合并等待中的随机请求。聊天与 Embedding 均通过可配置的 OpenAI 兼容 HTTP API，长期记忆查询失败时降级到最近消息。

**Tech Stack:** Python 3.10+、NoneBot2、nonebot-adapter-onebot v11、aiosqlite、sqlite-vec、httpx、Pydantic v2、Poetry、pytest、pytest-asyncio、nonebug、respx、Ruff。

**Spec:** `docs/superpowers/specs/2026-11-02-group-chat-memory-design.md`

## Global Constraints

- 所有消息、记忆、检索、概率与清除操作按 `group_id` 隔离，禁止跨群查询。
- 新群默认随机回复概率为 `0%`；@Bot 或回复 Bot 必定触发，不受概率影响。
- `/回复概率` 和 `/清空群记忆` 仅 SUPERUSER 可用；命令本身不记忆、不随机回复。
- 仅处理群文本；私聊、临时会话、匿名消息、自身消息、忽略列表账号和非文本随机触发均排除。
- 默认聊天模型标识为 `deepseek-v4.1-flash`，默认 Embedding 为 `BAAI/bge-m3`，两者均允许配置。
- 后台写入或长期记忆检索失败不得阻断最近上下文聊天；日志和异常提示不得包含 API 密钥。
- 最近消息默认 20 条、分块默认 10 条、长期记忆默认 5 条；BGE-M3 向量维度默认 1024。
- 遵循现有项目的 Poetry、Ruff 和 pytest 结构，不引入 Qdrant、本地 Embedding 或 Web 管理后台。

## Review Focus

- **API 返回缺字段、空向量或非 2xx：** 不得抛出泄露密钥的响应到群里；Task 3 用 respx 固定异常响应并测试降级。
- **插件重启时存在未向量化的分块：** 原文和待处理任务必须仍在；Task 4 测试关闭、重开后继续处理。
- **高概率下消息突发：** 点名逐条、随机只保留最新；Task 5 用可控的阻塞回复函数确定队列行为。
- **混合消息段、空 @、匿名和其他机器人：** 不得意外请求模型或污染记忆；Task 6 添加 OneBot 事件参数化测试。
- **跨群清理与检索：** 清群不能改概率或影响另一群；Task 2 和 Task 6 分别测试数据库与命令层隔离。

---

### Task 1: 项目骨架与配置

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`
- Create: `nonebot_plugin_group_chat_memory/__init__.py`, `nonebot_plugin_group_chat_memory/config.py`, `nonebot_plugin_group_chat_memory/models.py`
- Test: `tests/test_config.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `Config`（字段统一使用 `group_chat_memory_` 前缀）、`Message`、`Chunk`、`ChatContext` 数据模型。
- Produces: `get_settings() -> Config`，从 NoneBot 插件配置读取；测试可直接实例化 `Config`。

- [ ] **Step 1: 写配置失败测试**
  测试默认概率为 0、默认模型和 20/10/5/1024 参数；验证概率边界、`SecretStr`、忽略 QQ 集合及数据库路径配置。
- [ ] **Step 2: 运行测试确认失败**
  Run: `poetry run pytest tests/test_config.py -v`
  Expected: FAIL，模块尚不存在。
- [ ] **Step 3: 创建 Poetry 包和配置模型**
  声明上述运行时与开发依赖，并加入 `nonebot-plugin-localstore`；默认数据库路径由 localstore 的插件数据目录提供。聊天 `base_url/api_key` 与 Embedding 密钥允许为空以保证零配置导入，运行所需功能时检查并给出简短配置提示。`Message` 包含群、用户、显示名、正文、时间和 `is_bot`；`Chunk` 包含 ID、群、正文和状态；`ChatContext` 包含最近消息与长期片段。注册插件元数据和 OneBot v11 支持。
- [ ] **Step 4: 验证骨架**
  Run: `poetry run pytest tests/test_config.py -v && poetry run ruff check .`
  Expected: PASS。
- [ ] **Step 5: 提交**
  `git add pyproject.toml .gitignore .env.example nonebot_plugin_group_chat_memory tests && git commit -m "feat: add plugin configuration and models"`。

### Task 2: SQLite 与 sqlite-vec 持久化

**Files:**
- Create: `nonebot_plugin_group_chat_memory/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `MemoryStore.open() -> None`, `close() -> None`。
- Produces: `add_message(Message) -> int`, `recent_messages(group_id: int, limit: int) -> list[Message]`。
- Produces: `get_probability(group_id: int) -> float`, `set_probability(group_id: int, value: float) -> None`。
- Produces: `create_ready_chunk(group_id: int, size: int) -> Chunk | None`, `pending_chunks() -> list[Chunk]`, `complete_chunk(chunk_id: int, vector: list[float]) -> None`, `search(group_id: int, vector: list[float], limit: int) -> list[Chunk]`。
- Produces: `clear_group(group_id: int) -> None`，保留概率配置。

- [ ] **Step 1: 写持久化失败测试**
  覆盖消息时间顺序、最近条数、概率重启保留、0/100 边界、真实 sqlite-vec 最近邻检索、不同群相同文本不串扰，以及清群后原文/分块/向量删除但概率不变。
- [ ] **Step 2: 运行存储测试确认失败**
  Run: `poetry run pytest tests/test_store.py -v`
  Expected: FAIL，`MemoryStore` 尚不存在。
- [ ] **Step 3: 实现存储层**
  使用 WAL、外键和事务；消息包含可空 `chunk_id`，分块状态至少区分 pending/ready。`create_ready_chunk` 只选取当前群最早的未分块消息并原子关联；向量表使用配置维度，所有 KNN 查询同时限制群号。
- [ ] **Step 4: 验证存储层**
  Run: `poetry run pytest tests/test_store.py -v`
  Expected: PASS，包括关闭并重开数据库后的恢复测试。
- [ ] **Step 5: 提交**
  `git add nonebot_plugin_group_chat_memory/store.py tests/test_store.py && git commit -m "feat: add isolated group memory storage"`。

### Task 3: OpenAI 兼容客户端与上下文构建

**Files:**
- Create: `nonebot_plugin_group_chat_memory/api.py`, `nonebot_plugin_group_chat_memory/context.py`
- Test: `tests/test_api.py`, `tests/test_context.py`

**Interfaces:**
- Produces: `EmbeddingClient.embed(texts: list[str]) -> list[list[float]]`。
- Produces: `ChatClient.complete(context: ChatContext, current_text: str) -> str`。
- Produces: `ContextBuilder.build(group_id: int, query: str) -> ChatContext`，内部并行读取最近消息和查询向量。
- Consumes: Task 2 的 `MemoryStore` 与 Task 1 的配置/数据模型。

- [ ] **Step 1: 写客户端和上下文失败测试**
  用 respx 验证请求模型、Authorization 和批量 input；覆盖非 2xx、超时、空 data、错误向量维度。上下文测试验证系统提示区分 Bot/群友、最近消息与最多 5 条长期片段、当前输入不被截掉，以及 Embedding 失败仍返回最近消息。
- [ ] **Step 2: 运行测试确认失败**
  Run: `poetry run pytest tests/test_api.py tests/test_context.py -v`
  Expected: FAIL，客户端与构建器尚不存在。
- [ ] **Step 3: 实现 HTTP 客户端和上下文构建**
  使用 `httpx.AsyncClient` 请求 `/embeddings` 与 `/chat/completions`，分别应用配置超时；聊天仅做有限重试，长期查询不重试并快速降级。按字符预算优先保留当前输入和最近消息，提示词不包含密钥或数据库原文之外的额外信息。
- [ ] **Step 4: 验证客户端与降级**
  Run: `poetry run pytest tests/test_api.py tests/test_context.py -v`
  Expected: PASS，异常响应测试不输出密钥。
- [ ] **Step 5: 提交**
  `git add nonebot_plugin_group_chat_memory/api.py nonebot_plugin_group_chat_memory/context.py tests/test_api.py tests/test_context.py && git commit -m "feat: add model clients and resilient context building"`。

### Task 4: 异步分块与长期记忆写入

**Files:**
- Create: `nonebot_plugin_group_chat_memory/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `MemoryWriter.start() -> None`, `record(Message) -> None`, `flush() -> None`, `close() -> None`。
- Consumes: `MemoryStore`、`EmbeddingClient`。
- 保证: `record` 先持久化原文再排队；后台每满 10 条生成一个片段，失败不丢原文，启动时处理 `pending_chunks()`。

- [ ] **Step 1: 写后台记忆失败测试**
  验证 9 条不生成向量、第 10 条只批量调用一次；Embedding 失败时原文保留并指数退避，重试成功后才标记 ready；关闭并重开 writer 会恢复 pending 分块；清空群不会留下可写回向量的旧任务。
- [ ] **Step 2: 运行测试确认失败**
  Run: `poetry run pytest tests/test_memory.py -v`
  Expected: FAIL，`MemoryWriter` 尚不存在。
- [ ] **Step 3: 实现记忆 worker**
  使用有界 `asyncio.Queue` 和单一后台任务；record 的数据库异常只记录日志。按群创建 ready chunk 后调用 Embedding，complete 采用事务；清群后完成旧请求时由存储层忽略已不存在的 chunk。close 取消休眠并等待在途任务结束。
- [ ] **Step 4: 验证后台记忆**
  Run: `poetry run pytest tests/test_memory.py -v`
  Expected: PASS，测试无未关闭任务警告。
- [ ] **Step 5: 提交**
  `git add nonebot_plugin_group_chat_memory/memory.py tests/test_memory.py && git commit -m "feat: add asynchronous long-term memory writer"`。

### Task 5: 回复协调器

**Files:**
- Create: `nonebot_plugin_group_chat_memory/replies.py`
- Test: `tests/test_replies.py`

**Interfaces:**
- Produces: `ReplyCoordinator.submit(group_id: int, message: Message, explicit: bool) -> None`, `close() -> None`。
- Produces: 可注入 `reply(group_id: int, message: Message) -> Awaitable[None]` 与 `random_value() -> float`，便于确定性测试；`close` 等待或取消所有每群 worker，避免关闭后遗留任务。
- Produces: `should_random_reply(group_id: int) -> bool`，0% 永不、100% 必定，中间值按群概率判断。

- [ ] **Step 1: 写并发与概率失败测试**
  用 asyncio Event 阻塞首个回复，验证同群 explicit 请求按顺序逐条执行、随机等待任务被最新一条替换、不同群可并行；参数化 0/100/中间概率和 API 异常后继续处理后续请求。
- [ ] **Step 2: 运行协调器测试确认失败**
  Run: `poetry run pytest tests/test_replies.py -v`
  Expected: FAIL，协调器尚不存在。
- [ ] **Step 3: 实现每群调度**
  每群一个 worker；explicit 使用 FIFO，random 使用容量 1 的最新值槽位，提交时不复制旧最近消息快照。worker 调用注入的 reply，由上层在实际执行时构建上下文；单个请求异常被隔离并记录。
- [ ] **Step 4: 验证调度行为**
  Run: `poetry run pytest tests/test_replies.py -v`
  Expected: PASS，无回复风暴和跨群锁。
- [ ] **Step 5: 提交**
  `git add nonebot_plugin_group_chat_memory/replies.py tests/test_replies.py && git commit -m "feat: coordinate explicit and random group replies"`。

### Task 6: OneBot 消息与管理命令

**Files:**
- Create: `nonebot_plugin_group_chat_memory/handlers.py`
- Modify: `nonebot_plugin_group_chat_memory/__init__.py`
- Test: `tests/test_handlers.py`

**Interfaces:**
- Consumes: 前序任务的 store、writer、context、chat client 和 coordinator。
- Produces: `/回复概率 [0～100]`、`/清空群记忆`，以及群消息记录/回复 matcher。
- Produces: 生命周期启动时打开 store、恢复 writer，关闭时 flush/close 所有资源。

- [ ] **Step 1: 写 OneBot 事件失败测试**
  用 nonebug 模拟：@与回复必触发、默认普通消息只记录、100% 普通文本触发；混合图片不随机触发、空 @ 不请求模型、匿名/私聊/自身/忽略账号不处理。测试非 SUPERUSER 权限提示、概率一位小数和非法输入、命令不记忆不随机回复、清群不影响其他群及原概率。
- [ ] **Step 2: 运行事件测试确认失败**
  Run: `poetry run pytest tests/test_handlers.py -v`
  Expected: FAIL，handlers 尚未注册。
- [ ] **Step 3: 实现消息与命令处理**
  命令 matcher 高于普通消息且 block；普通 matcher 仅接收 GroupMessageEvent。定向消息只允许 Text/At/Reply 段，随机消息只允许纯文本。发送模型回复后以 `is_bot=True` 记录 Bot 消息；模型错误仅发送通用提示。
- [ ] **Step 4: 验证 OneBot 行为**
  Run: `poetry run pytest tests/test_handlers.py -v`
  Expected: PASS，所有 mock API 均校验请求次数。
- [ ] **Step 5: 提交**
  `git add nonebot_plugin_group_chat_memory/__init__.py nonebot_plugin_group_chat_memory/handlers.py tests/test_handlers.py && git commit -m "feat: add group chat handlers and admin commands"`。

### Task 7: 文档、完整测试与端到端验收

**Files:**
- Create: `README.md`
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: 完整插件。
- Produces: 安装、配置、隐私、命令和备份说明，以及从群消息到模型回复再到长期检索的验收测试。

- [ ] **Step 1: 写端到端失败测试**
  模拟两个群产生消息、完成 10 条分块、点名请求模型并发送回复；断言群 B 无法检索群 A 向量、查询 Embedding 超时仍聊天、Bot 回复被记录、随机概率与清群命令行为符合设计。
- [ ] **Step 2: 运行端到端测试并修正集成问题**
  Run: `poetry run pytest tests/test_end_to_end.py -v`
  Expected: 初始 FAIL 暴露生命周期或接口集成问题；仅修正实现使其通过，不降低断言。
- [ ] **Step 3: 编写使用文档**
  README 明确聊天/Embedding 配置、DeepSeek 模型标识需按服务商填写、全群文本会发送给 Embedding 服务商、数据库备份方式、随机回复风险和 SUPERUSER 命令。
- [ ] **Step 4: 运行完整交付验证**
  Run: `poetry run pytest -v && poetry run ruff check . && poetry build`
  Expected: 全部 PASS，Ruff 无错误，wheel 与 sdist 构建成功；检查 `git status --short`，删除测试产生且不属于交付物的文件。
- [ ] **Step 5: 提交**
  `git add README.md tests/test_end_to_end.py && git commit -m "test: verify group chat memory end to end"`。
