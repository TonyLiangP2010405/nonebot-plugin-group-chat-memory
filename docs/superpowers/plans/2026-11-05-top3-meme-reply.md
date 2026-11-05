# 好感榜前 3 回复附图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 好感榜前 3 名触发回复时补一张相关表情包，SUPERUSER 100% 触发并附图，其余行为不变。

**Architecture:** 在 `Message` 上加一个 `attach_meme` 标记，由 handler 判定后随消息入队；`reply()` 发完文字后按标记调用独立的 `MemeProvider` 取图并发送。前 3 名由 `AffectionLeaderboard` 新增的 `top_members` 从已有的 `ranking()` 结果中截取。

**Tech Stack:** Python 3.10+、NoneBot2、httpx、SQLite（不变）、pytest + nonebug + respx

**Spec:** `docs/superpowers/specs/2026-11-05-top3-meme-reply-design.md`

## Global Constraints

- Python `>=3.10,<4.0`；不新增第三方依赖，取图复用已有 `httpx`。
- 接口地址默认 `https://cn.apihz.cn/api/img/apihzbqb.php`，参数 `type=2`、`page=1`、`limit=6`，`words` 最多 10 个字符。
- 节流最小间隔 6.5 秒，缓存 TTL 600 秒、最多 200 条。
- 取图失败只发文字，不重试、不向群里报错。
- 前 3 名与 SUPERUSER 触发不占用每小时请求名额。
- 必回名单、@/回复 Bot 的消息保持现有行为，不附图。
- `ruff` line-length 100；测试 `asyncio_mode = "auto"`，直接写 `async def test_...`。
- 提交信息沿用仓库风格：`feat: ...`。

## Review Focus

- apihz 每分钟 10 次：节流必须是进程级全局的，多群并发时不能各算各的。Task 1 的节流测试与 Task 3 的共享实例共同保证。
- 关键词为空或全是标点/emoji：不应发请求，也不应抛异常。Task 1 的 `keyword` 测试与 `fetch` 空关键词测试。
- 发图失败（URL 失效、协议端报错）：文字回复与记忆写入必须照常完成。Task 3 的发送异常测试。
- 好感插件缺失或 `ranking()` 抛异常：前 3 判定退化为空，普通消息流程不受影响。Task 2 的降级测试。
- 好感开关关闭或概率为 0：前 3 名不得附图。Task 3 的开关测试。

---

### Task 1: MemeProvider 与配置项

**Files:**
- Create: `nonebot_plugin_group_chat_memory/meme.py`
- Modify: `nonebot_plugin_group_chat_memory/config.py:54-81`
- Test: `tests/test_meme.py`

**Interfaces:**
- Consumes: 无。
- Produces:
  - `MemeProvider(enabled: bool, api_url: str, api_id: str, api_key: str, *, min_interval: float = 6.5, cache_ttl: float = 600.0, cache_size: int = 200, timeout: float = 10.0, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, random_value: Callable[[], float] = random.random)`
  - `async def MemeProvider.fetch(self, text: str) -> str | None`
  - `@staticmethod def MemeProvider.keyword(text: str) -> str`
  - 配置字段 `group_chat_memory_meme_enabled`、`group_chat_memory_meme_api_url`、`group_chat_memory_apihz_id`、`group_chat_memory_apihz_key`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_meme.py`：

```python
from __future__ import annotations

import httpx
import pytest
import respx

from nonebot_plugin_group_chat_memory.meme import MemeProvider

API_URL = "https://meme.example/api"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_provider(clock: FakeClock | None = None, **overrides) -> MemeProvider:
    clock = clock or FakeClock()
    values = {
        "enabled": True,
        "api_url": API_URL,
        "api_id": "10023437",
        "api_key": "test-key",
        "clock": clock,
        "sleep": clock.sleep,
        "random_value": lambda: 0.0,
    }
    values.update(overrides)
    return MemeProvider(**values)


def test_keyword_strips_punctuation_and_truncates():
    assert MemeProvider.keyword("笑死，我也这么觉得！！123abc") == "笑死我也这么觉得12"


def test_keyword_returns_empty_for_punctuation_only():
    assert MemeProvider.keyword("！！！？？ ……") == ""


async def test_returns_url_from_results():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL).mock(
            return_value=httpx.Response(
                200, json={"code": 200, "res": ["https://img/1.png", "https://img/2.png"]}
            )
        )

        url = await make_provider().fetch("笑死我了")

    assert url == "https://img/1.png"
    request = route.calls[0].request
    assert request.url.params["type"] == "2"
    assert request.url.params["words"] == "笑死我了"
    assert request.url.params["limit"] == "6"


async def test_missing_credentials_returns_none_without_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL)

        assert await make_provider(api_key="").fetch("笑死") is None

    assert len(route.calls) == 0


async def test_disabled_provider_returns_none_without_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL)

        assert await make_provider(enabled=False).fetch("笑死") is None

    assert len(route.calls) == 0


async def test_empty_keyword_returns_none_without_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL)

        assert await make_provider().fetch("！！！") is None

    assert len(route.calls) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"code": 400, "msg": "调用频次过快"},
        {"code": 200, "res": []},
        {"code": 200},
    ],
)
async def test_unsuccessful_payload_returns_none(payload):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(return_value=httpx.Response(200, json=payload))

        assert await make_provider().fetch("笑死") is None


async def test_http_error_returns_none():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(side_effect=httpx.ConnectError("boom"))

        assert await make_provider().fetch("笑死") is None


async def test_cache_hit_skips_second_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL).mock(
            return_value=httpx.Response(200, json={"code": 200, "res": ["https://img/1.png"]})
        )
        provider = make_provider()

        first = await provider.fetch("笑死")
        second = await provider.fetch("笑死")

    assert first == second == "https://img/1.png"
    assert len(route.calls) == 1


async def test_throttle_waits_between_new_keywords():
    clock = FakeClock()
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(
            return_value=httpx.Response(200, json={"code": 200, "res": ["https://img/1.png"]})
        )
        provider = make_provider(clock)

        await provider.fetch("笑死")
        await provider.fetch("无语")

    assert clock.now == pytest.approx(6.5)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_meme.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'nonebot_plugin_group_chat_memory.meme'`

- [ ] **Step 3: 实现 `meme.py` 与配置字段**

`meme.py` 中实现：`keyword` 用 `re.sub(r"[^\w\u4e00-\u9fa5]", "", text)[:10]`；`fetch` 先查缓存（`dict[str, tuple[float, str]]`，命中直接返回），再在全局 `asyncio.Lock` 内等待到 `_next_at`、随后把 `_next_at` 设为当前时间加 `min_interval`；请求用 `httpx.AsyncClient(timeout=self._timeout)`，`raise_for_status()` 后校验 `code == 200` 且 `res` 非空，用 `random_value` 取索引；异常与不成功一律返回 `None` 并记 warning。缓存超过 `cache_size` 时按插入顺序 `next(iter(...))` 淘汰。

`config.py` 的 `Config` 增加：

```python
group_chat_memory_meme_enabled: bool = True
group_chat_memory_meme_api_url: str = "https://cn.apihz.cn/api/img/apihzbqb.php"
group_chat_memory_apihz_id: str = ""
group_chat_memory_apihz_key: SecretStr = SecretStr("")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_meme.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add nonebot_plugin_group_chat_memory/meme.py nonebot_plugin_group_chat_memory/config.py tests/test_meme.py
git commit -m "feat: add apihz meme provider"
```

---

### Task 2: 好感榜前 3 查询

**Files:**
- Modify: `nonebot_plugin_group_chat_memory/affection.py:23-71`
- Test: `tests/test_affection.py`

**Interfaces:**
- Consumes: `service.ranking(group_id: str)`，条目需带 `rank` 与 `profile.user_id`。
- Produces:
  - `async def AffectionLeaderboard.top_members(self, group_id: int, limit: int = 3) -> frozenset[int]`
  - `async def AffectionLeaderboard.is_top_member(self, group_id: int, user_id: int, limit: int = 3) -> bool`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_affection.py`：

```python
class OutOfOrderService:
    def __init__(self, entries):
        self.entries = entries

    async def ranking(self, group_id: str):
        return [
            SimpleNamespace(rank=rank, profile=SimpleNamespace(user_id=user_id))
            for rank, user_id in self.entries
        ]


class NoRankService:
    def __init__(self, user_ids):
        self.user_ids = user_ids

    async def ranking(self, group_id: str):
        return [SimpleNamespace(profile=SimpleNamespace(user_id=uid)) for uid in self.user_ids]


async def test_top_members_returns_first_three():
    leaderboard, _ = make_leaderboard(FakeService(("1001", "1002", "1003", "1004")))

    assert await leaderboard.top_members(20002) == frozenset({1001, 1002, 1003})


async def test_top_members_orders_by_rank_attribute():
    service = OutOfOrderService([(3, "1003"), (1, "1001"), (2, "1002"), (4, "1004")])
    leaderboard, _ = make_leaderboard(service)

    assert await leaderboard.top_members(20002) == frozenset({1001, 1002, 1003})


async def test_top_members_falls_back_to_list_order_without_rank():
    leaderboard, _ = make_leaderboard(NoRankService(("1004", "1002", "1001", "1003")))

    assert await leaderboard.top_members(20002) == frozenset({1004, 1002, 1001})


async def test_top_members_shares_cache_with_membership():
    clock = Clock()
    service = FakeService(("1001", "1002", "1003", "1004"))
    leaderboard = AffectionLeaderboard(
        ttl=30.0, clock=clock, service_loader=lambda: service
    )

    await leaderboard.top_members(20002)
    await leaderboard.is_member(20002, 1004)

    assert service.calls == ["20002"]


async def test_top_members_returns_empty_when_service_missing():
    leaderboard = AffectionLeaderboard(service_loader=lambda: None)

    assert await leaderboard.top_members(20002) == frozenset()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_affection.py -k top_members -v`
Expected: FAIL，`AttributeError: 'AffectionLeaderboard' object has no attribute 'top_members'`

- [ ] **Step 3: 实现 `top_members` 与 `is_top_member`**

`_cache` 从 `dict[int, tuple[float, frozenset[int]]]` 改为 `dict[int, tuple[float, tuple[int, ...], frozenset[int]]]`。`_fetch` 返回按名次排好的 `tuple[int, ...]`：有整数 `rank` 时按其排序，缺失时用出现顺序作为名次；非法 `user_id` 跳过。`is_member` 用缓存的第三项，`top_members` 用第二项截取前 `limit` 个，`is_top_member` 判断成员是否在 `top_members` 中。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_affection.py -v`
Expected: 全部 PASS（含原有测试）

- [ ] **Step 5: 提交**

```bash
git add nonebot_plugin_group_chat_memory/affection.py tests/test_affection.py
git commit -m "feat: expose affection top members"
```

---

### Task 3: 触发判定与发送图片

**Files:**
- Modify: `nonebot_plugin_group_chat_memory/models.py:12-20`
- Modify: `nonebot_plugin_group_chat_memory/handlers.py:101-124`（`reply`）、`:618-668`（`handle_group_message`）、`:70-77`（实例化）
- Test: `tests/test_handlers.py`

**Interfaces:**
- Consumes: `MemeProvider.fetch`（Task 1）、`AffectionLeaderboard.is_top_member`（Task 2）。
- Produces:
  - `Message.attach_meme: bool = False`
  - 模块级 `handlers.meme_provider: MemeProvider`
  - `reply()` 在 `message.attach_meme` 为真时追加一条图片消息

- [ ] **Step 1: 写失败测试**

在 `tests/test_handlers.py` 中扩展 `FakeLeaderboard`：

```python
class FakeLeaderboard:
    def __init__(self, members: set[int] | None = None, top: set[int] | None = None) -> None:
        self.members = members or set()
        self.top = top or set()
        self.calls: list[tuple[int, int]] = []

    async def is_member(self, group_id: int, user_id: int) -> bool:
        self.calls.append((group_id, user_id))
        return user_id in self.members

    async def is_top_member(self, group_id: int, user_id: int, limit: int = 3) -> bool:
        self.calls.append((group_id, user_id))
        return user_id in self.top
```

新增测试（`runtime` fixture 需增加 `monkeypatch.setattr(handlers, "meme_provider", meme)`，`meme = AsyncMock()` 并加入 `SimpleNamespace` 返回值，字段名 `meme`，默认 `meme.fetch.return_value = "https://img/reply.png"`）：

```python
async def test_superuser_triggers_without_draw_and_attaches_meme(app, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    use_random(monkeypatch, 0.99)
    event = make_group_event("超管说话", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
            {"message_id": 123},
        )
        ctx.should_call_api(
            "send_group_msg",
            {
                "group_id": GROUP_ID,
                "message": MessageSegment.image("https://img/reply.png"),
            },
            {"message_id": 124},
        )
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    runtime.meme.fetch.assert_awaited_once_with("模型回复")


async def test_top_three_member_attaches_meme(app, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    monkeypatch.setattr(
        handlers, "affection_leaderboard", FakeLeaderboard({USER_ID}, {USER_ID})
    )
    coordinator = use_random(monkeypatch, 0.29)
    event = make_group_event("前三说话")

    try:
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.should_call_api(
                "send_group_msg",
                {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
                {"message_id": 123},
            )
            ctx.should_call_api(
                "send_group_msg",
                {
                    "group_id": GROUP_ID,
                    "message": MessageSegment.image("https://img/reply.png"),
                },
                {"message_id": 124},
            )
            ctx.receive_event(bot, event)
            await ctx.run()
            await wait_until(lambda: runtime.record.await_count == 2)
    finally:
        await coordinator.close()


async def test_fourth_affection_member_does_not_attach_meme(app, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    monkeypatch.setattr(
        handlers, "affection_leaderboard", FakeLeaderboard({USER_ID}, set())
    )
    coordinator = use_random(monkeypatch, 0.29)
    event = make_group_event("榜内第四说话")

    try:
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.should_call_api(
                "send_group_msg",
                {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
                {"message_id": 123},
            )
            ctx.receive_event(bot, event)
            await ctx.run()
            await wait_until(lambda: runtime.record.await_count == 2)
    finally:
        await coordinator.close()

    runtime.meme.fetch.assert_not_awaited()


async def test_top_three_with_affection_disabled_does_not_attach_meme(app, runtime, monkeypatch):
    runtime.get_probability.return_value = 100.0
    runtime.get_affection_enabled.return_value = False
    monkeypatch.setattr(
        handlers, "affection_leaderboard", FakeLeaderboard({USER_ID}, {USER_ID})
    )
    event = make_group_event("前三说话")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    runtime.meme.fetch.assert_not_awaited()


async def test_directed_message_does_not_attach_meme(app, runtime, monkeypatch):
    runtime.get_probability.return_value = 100.0
    monkeypatch.setattr(
        handlers, "affection_leaderboard", FakeLeaderboard({USER_ID}, {USER_ID})
    )
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("点名")])
    )

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    runtime.meme.fetch.assert_not_awaited()


async def test_image_failure_still_records_bot_reply(app, runtime):
    runtime.meme.fetch.return_value = None
    event = make_group_event("超管说话", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    runtime.record.assert_awaited()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_handlers.py -k "meme or superuser_triggers" -v`
Expected: FAIL——图片消息未发送、`meme_provider` 不存在

- [ ] **Step 3: 实现标记、判定与发送**

`models.py` 的 `Message` 增加 `attach_meme: bool = False`（放在 `is_bot` 之后，带默认值）。

`handlers.py`：

1. 模块初始化处按配置创建单例：`meme_provider = MemeProvider(enabled=..., api_url=..., api_id=..., api_key=config.group_chat_memory_apihz_key.get_secret_value())`。
2. `reply()` 在文字发送成功后执行 `if message.attach_meme:` → `url = await meme_provider.fetch(response)`；`url` 非空则 `await bot.send_group_msg(group_id=group_id, message=MessageSegment.image(url))`，发送异常仅记 warning，随后照常 `writer.record(...)`。
3. `handle_group_message()` 在必回名单分支之后、`_affection_probability` 之前加入 SUPERUSER 分支：`if await SUPERUSER(bot, event):` → `memory_message.attach_meme = True`、`coordinator.submit(..., explicit=False)`、`return`，不消耗名额。
4. 普通分支在通过概率判定后，若 `affection_probability > 0 and await affection_leaderboard.is_top_member(event.group_id, event.user_id)`，置 `memory_message.attach_meme = True` 再提交。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_handlers.py -v`
Expected: 全部 PASS（含原有测试）

- [ ] **Step 5: 提交**

```bash
git add nonebot_plugin_group_chat_memory/models.py nonebot_plugin_group_chat_memory/handlers.py tests/test_handlers.py
git commit -m "feat: attach meme to top affection replies"
```

---

## 验证

- `pytest -q` 全量通过。
- `ruff check .` 与 `ruff format --check .` 无告警。
- 手工验证（可选）：配置 `GROUP_CHAT_MEMORY_APIHZ_ID/KEY` 后，用 SUPERUSER 号在群里发一句话，应收到文字 + 一张图；用榜内非前 3 的号触发，只收到文字。
