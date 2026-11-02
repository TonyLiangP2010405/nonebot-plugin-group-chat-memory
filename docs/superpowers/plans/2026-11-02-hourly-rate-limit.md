# 每群每小时请求限制 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每群增加默认 10 次/滚动小时的模型请求限制，SUPERUSER 可通过命令修改。

**Architecture:** SQLite 新增使用记录表和每群上限字段；一次写事务内清理旧记录、检查额度并占用名额。消息处理层在点名和随机回复触发前占用名额，命令和记忆写入不受限制。

**Tech Stack:** SQLite、aiosqlite、NoneBot2、pytest、Ruff、Poetry。

**Spec:** `docs/superpowers/specs/2026-11-02-rate-limit-design.md`

## Global Constraints

- 每群独立，滚动 60 分钟，默认 10 次，允许范围 1～10000。
- 只统计触发的模型请求；命令、记忆、Embedding、HTTP 重试不重复计数。
- 点名超额提示一次，随机超额静默；消息仍写入记忆。
- 配置持久化；清空群记忆删除使用记录但保留上限与概率。
- 并发不能突破上限。

## Review Focus

- **同一秒并发触发：** 事务必须原子占用名额；Task 1 并发测试固定。
- **滚动窗口边界：** 60 分钟前记录应释放，59 分钟记录仍计数；Task 1 注入时间测试固定。
- **点名与随机超额：** 点名收到提示，随机静默，两者都不调用模型；Task 2 固定。
- **非 SUPERUSER/非法数值：** 不改变配置；Task 2 固定。
- **重启和清群：** 使用记录/上限按设计保留或清理；Task 1 固定。

### Task 1: 限流存储

**Files:** Modify `nonebot_plugin_group_chat_memory/store.py`; Test `tests/test_store.py`, `tests/test_rate_limit.py`.

**Interfaces:**
- Produces: `MemoryStore.get_hourly_limit(group_id) -> int`, `set_hourly_limit(group_id, value) -> None`, `consume_request_slot(group_id, now: datetime | None = None) -> bool`, `get_hour_usage(group_id, now=None) -> int`.
- DB: `groups.hourly_limit` 默认 10；`request_usage(group_id, requested_at)`。

- [ ] **Step 1: 写失败测试**
  默认值、修改/重开保留、不同群隔离、第 11 次拒绝、窗口过期、并发不越界、清群删除使用记录但保留上限和概率。
- [ ] **Step 2: Run:** `poetry run pytest tests/test_rate_limit.py tests/test_store.py -q`；Expected FAIL。
- [ ] **Step 3: 实现迁移、设置和原子占用。**
- [ ] **Step 4: Run same command；Expected PASS.**
- [ ] **Step 5: Commit:** `git add ... && git commit -m "feat: add per-group hourly request limits"`。

### Task 2: 命令与触发接入

**Files:** Modify `handlers.py`, `README.md`, `.env.example`, `pyproject.toml`; Test `tests/test_handlers.py`.

**Interfaces:** Consumes Task 1 store methods. Produces `/每小时次数` query/set command, priority `-1001`; group message priority `-1000`, directed event StopPropagation.

- [ ] **Step 1: 写失败测试**
  查询/修改命令、权限和非法值、点名第 11 次提示且不调用模型、随机超额静默、消息仍记录、优先级断言。
- [ ] **Step 2: Run:** `poetry run pytest tests/test_handlers.py -q`；Expected FAIL。
- [ ] **Step 3: 实现命令与占用逻辑，更新版本 0.1.4 和文档。**
- [ ] **Step 4: Complete:** `poetry run pytest -q && poetry run ruff check . && poetry run python -m compileall -q nonebot_plugin_group_chat_memory && poetry build`。
- [ ] **Step 5: Commit:** `git add ... && git commit -m "feat: rate limit group chat requests"`。
