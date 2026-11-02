# Command Runtime Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 SUPERUSER 私聊命令和 SQLite 持久化替代四个模型 API 环境变量，并让更新立即生效。

**Architecture:** 在现有 SQLite 中增加全局 runtime_config 表，启动时将四项配置应用到共享 Config；命令先写数据库再更新内存，现有 API 客户端继续读取同一对象。新增私聊设置命令与脱敏状态查询，不改动聊天和记忆流水线。

**Tech Stack:** Python 3.10+、NoneBot2、OneBot v11、aiosqlite、Pydantic v2、pytest、nonebug、Ruff、Poetry。

**Spec:** `docs/superpowers/specs/2026-11-02-command-runtime-config-design.md`

## Global Constraints

- 仅 `/设置聊天地址`、`/设置聊天密钥`、`/设置聊天模型`、`/设置向量密钥` 四项改为命令；其余配置方式不变。
- 设置命令仅私聊 SUPERUSER；`/模型配置` 可供 SUPERUSER 在私聊或群聊查询，但密钥必须脱敏。
- 四项配置全 Bot 共用，不再读取 `.env`，SQLite 重开后保留，更新后无需重启。
- 数据库写入成功后才更新共享 Config；失败不得部分生效。
- 密钥明文只存在于数据库和内存，不得出现在命令响应、日志或异常中。
- 保留现有聊天、向量、群隔离和概率行为。

## Review Focus

- **群聊或非 SUPERUSER 伪造设置命令：** 必须拒绝且数据库不变；Task 2 权限测试固定。
- **空密钥、带空白参数和非法 URL：** 不得写入危险或不可用值；Task 2 参数化测试固定。
- **密钥与真实值相似但已脱敏：** 响应不得包含完整密钥；Task 2 日志/响应测试固定。
- **数据库重开及同名 env 同时存在：** 只采用数据库/默认值；Task 1 持久化测试固定。
- **命令更新与正在执行的 API 请求交错：** 下一次请求必须使用新值，已在执行的请求不被强行取消；Task 1/2 测试固定。

---

### Task 1: 全局运行时配置持久化

**Files:**
- Modify: `nonebot_plugin_group_chat_memory/store.py`
- Modify: `nonebot_plugin_group_chat_memory/config.py`
- Modify: `nonebot_plugin_group_chat_memory/handlers.py`（仅启动加载）
- Test: `tests/test_store.py`, `tests/test_runtime_config.py`

**Interfaces:**
- Produces: `MemoryStore.get_runtime_config() -> dict[str, str]`、`set_runtime_config(key: str, value: str) -> None`。
- Produces: `Config.apply_runtime_config(values: dict[str, str]) -> None`，仅映射四个允许字段。
- Produces: 启动时 `store.open()` 后加载并应用运行时配置。

- [ ] **Step 1: 写失败测试**
  验证默认空配置、四项 round-trip、非法 key 拒绝、数据库重开保留、同名 env 被忽略、应用后 `require_chat_config/require_embedding_config` 使用新值，以及未知键不会写入 Config。
- [ ] **Step 2: 运行测试确认失败**
  Run: `poetry run pytest tests/test_runtime_config.py tests/test_store.py -q`
  Expected: FAIL，runtime_config 接口不存在。
- [ ] **Step 3: 实现表、存储接口与启动应用**
  使用现有事务/锁；只允许四个白名单 key。Config 提供严格映射和 SecretStr 更新，启动加载在 writer.start 前完成。
- [ ] **Step 4: 验证任务**
  Run: `poetry run pytest tests/test_runtime_config.py tests/test_store.py tests/test_config.py -q`
  Expected: PASS。
- [ ] **Step 5: 提交**
  `git add nonebot_plugin_group_chat_memory tests && git commit -m "feat: persist runtime model configuration"`。

### Task 2: 私聊命令、脱敏查询与文档

**Files:**
- Modify: `nonebot_plugin_group_chat_memory/handlers.py`
- Modify: `.env.example`, `README.md`
- Test: `tests/test_handlers.py`, `tests/test_runtime_config.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: Task 1 的存储与 Config 接口。
- Produces: 四条私聊设置命令和 `/模型配置`；统一 `apply_setting(key, value) -> None`，先 store 后 Config。

- [ ] **Step 1: 写命令失败测试**
  用 nonebug 验证私聊 SUPERUSER 成功、群聊/非 SUPERUSER/空参数拒绝、URL 仅允许 http/https 且带 host、模型非空、写入后下一次请求使用新值；查询只显示地址/模型和密钥掩码，不含完整密钥；设置命令不记忆不随机。
- [ ] **Step 2: 运行测试确认失败**
  Run: `poetry run pytest tests/test_handlers.py tests/test_runtime_config.py -q`
  Expected: FAIL，命令未注册。
- [ ] **Step 3: 实现命令与文档**
  复用现有 SUPERUSER 和 CommandArg；设置 matcher 仅接受 PrivateMessageEvent，查询 matcher 支持 MessageEvent。成功后简短确认，异常仅记录类型。删除四个 env 示例，README 改为命令表并说明私聊记录和 SQLite 明文风险。
- [ ] **Step 4: 运行完整验证**
  Run: `poetry run pytest -q && poetry run ruff check . && poetry run python -m compileall -q nonebot_plugin_group_chat_memory && poetry build`
  Expected: 全部 PASS，构建成功，工作区无多余产物。
- [ ] **Step 5: 提交**
  `git add .env.example README.md nonebot_plugin_group_chat_memory tests && git commit -m "feat: configure model APIs through private commands"`。
