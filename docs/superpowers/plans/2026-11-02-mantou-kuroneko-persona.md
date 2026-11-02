# 馒头黑猫式默认人格 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Bot 默认系统提示词改为“馒头身份 + 五更琉璃（黑猫）式性格”，并发布 0.1.3。

**Architecture:** 只修改默认系统提示词常量、文档、示例配置和版本号；上下文组装、自定义提示词覆盖与聊天流程不变。

**Tech Stack:** Python、Pydantic、pytest、Ruff、Poetry。

**Spec:** `docs/superpowers/specs/2026-11-02-kuroneko-persona-design.md`

## Global Constraints

- 始终自称“馒头”或“我”，不得自称黑猫、五更琉璃或夜魔女王本人。
- 保留中二、毒舌、傲娇、外冷内热；控制频率，不真正辱骂群友。
- 正经问题优先准确实用，不堆砌中二台词，不编造记忆或原作剧情。
- 自定义 `GROUP_CHAT_MEMORY_SYSTEM_PROMPT` 必须继续完全覆盖默认人格。
- 版本更新为 0.1.3。

## Review Focus

- **宠物/正经问题：** 不因角色扮演影响正常回答；提示词明确优先级。
- **身份混淆：** 回复中不自称黑猫或五更琉璃；测试固定关键身份规则。
- **自定义提示词：** 非空自定义内容仍覆盖默认人格；现有测试继续通过。
- **安全话题：** 医疗、法律等警告不被角色扮演弱化；提示词明确要求。
- **密钥与记忆：** 人格改造不得改变现有配置、隐私和记忆行为；完整套件回归。

### Task 1: 默认人格、文档与发布

**Files:**
- Modify: `nonebot_plugin_group_chat_memory/config.py`
- Modify: `.env.example`, `README.md`, `pyproject.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: 新的 `DEFAULT_SYSTEM_PROMPT`；`Config.group_chat_memory_system_prompt` 默认值随之变化。
- 不改变其他公共接口。

- [ ] **Step 1: 写失败测试**
  断言默认提示词包含“馒头”“凡人”“夜魔女王”“不要自称黑猫”等身份与风格规则，并断言同样包含“准确”“安全”等正常回答优先级。
- [ ] **Step 2: 运行测试确认失败**
  Run: `poetry run pytest tests/test_config.py -q`
  Expected: FAIL，默认仍是通用群聊助手。
- [ ] **Step 3: 实现默认提示词与文档**
  更新默认常量、`.env.example` 示例、README 默认值说明和版本号；不改变上下文构建代码。
- [ ] **Step 4: 完整验证**
  Run: `poetry run pytest -q && poetry run ruff check . && poetry run python -m compileall -q nonebot_plugin_group_chat_memory && poetry build`
  Expected: 全部 PASS。
- [ ] **Step 5: 提交**
  `git add . nonebot_plugin_group_chat_memory tests pyproject.toml && git commit -m "feat: add Mantou Kuroneko-style persona"`。
