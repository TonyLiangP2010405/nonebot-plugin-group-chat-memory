from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    Message,
    MessageSegment,
    PrivateMessageEvent,
)
from nonebot.adapters.onebot.v11.event import Anonymous, Reply, Sender
from nonebug import App

from nonebot_plugin_group_chat_memory import handlers
from nonebot_plugin_group_chat_memory.api import ChatAPIError
from nonebot_plugin_group_chat_memory.config import Config, ConfigError
from nonebot_plugin_group_chat_memory.models import ChatContext
from nonebot_plugin_group_chat_memory.models import Message as MemoryMessage
from nonebot_plugin_group_chat_memory.replies import ReplyCoordinator
from nonebot_plugin_group_chat_memory.store import MemoryStore

BOT_ID = 10000
GROUP_ID = 20002
OTHER_GROUP_ID = 20003
USER_ID = 1001
SUPERUSER_ID = 10001


def make_group_event(
    content: str | Message | MessageSegment,
    *,
    user_id: int = USER_ID,
    group_id: int = GROUP_ID,
    to_me: bool = False,
    reply: Reply | None = None,
    anonymous: Anonymous | None = None,
) -> GroupMessageEvent:
    message = content if isinstance(content, Message) else Message(content)
    return GroupMessageEvent(
        time=1_700_000_000,
        self_id=BOT_ID,
        post_type="message",
        message_type="group",
        sub_type="anonymous" if anonymous else "normal",
        message_id=1,
        user_id=user_id,
        group_id=group_id,
        message=message,
        original_message=message,
        raw_message=message.extract_plain_text(),
        font=0,
        sender=Sender(user_id=user_id, nickname="昵称", card="群名片", role="member"),
        to_me=to_me,
        reply=reply,
        anonymous=anonymous,
    )


def make_private_event(sub_type: str) -> PrivateMessageEvent:
    message = Message("私聊")
    return PrivateMessageEvent(
        time=1_700_000_000,
        self_id=BOT_ID,
        post_type="message",
        message_type="private",
        sub_type=sub_type,
        message_id=1,
        user_id=USER_ID,
        message=message,
        original_message=message,
        raw_message="私聊",
        font=0,
        sender=Sender(user_id=USER_ID, nickname="昵称"),
    )


def bot_reply() -> Reply:
    return Reply(
        time=1_699_999_999,
        message_type="group",
        message_id=99,
        real_id=99,
        sender=Sender(user_id=BOT_ID, nickname="Bot"),
        message=Message("之前的回复"),
    )


async def wait_until(predicate, timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout)


@pytest_asyncio.fixture
async def runtime(monkeypatch):
    record = AsyncMock()
    build = AsyncMock(return_value=ChatContext())
    complete = AsyncMock(return_value="模型回复")
    get_probability = AsyncMock(return_value=0.0)
    set_probability = AsyncMock()
    clear_group = AsyncMock()
    monkeypatch.setattr(handlers.writer, "record", record)
    monkeypatch.setattr(handlers.context_builder, "build", build)
    monkeypatch.setattr(handlers.chat_client, "complete", complete)
    monkeypatch.setattr(handlers.store, "get_probability", get_probability)
    monkeypatch.setattr(handlers.store, "set_probability", set_probability)
    monkeypatch.setattr(handlers.store, "clear_group", clear_group)
    coordinator = ReplyCoordinator(handlers.store, handlers.reply)
    monkeypatch.setattr(handlers, "coordinator", coordinator)
    handlers._bots.clear()
    try:
        yield SimpleNamespace(
            record=record,
            build=build,
            complete=complete,
            get_probability=get_probability,
            set_probability=set_probability,
            clear_group=clear_group,
            coordinator=coordinator,
        )
    finally:
        await coordinator.close()
        handlers._bots.clear()


@pytest.mark.parametrize("kind", ["at", "reply"])
async def test_directed_message_always_replies_and_records_bot(app: App, runtime, kind):
    if kind == "at":
        message = Message([MessageSegment.at(BOT_ID), MessageSegment.text(" 你好 ")])
        event = make_group_event(message, to_me=True)
    else:
        message = Message([MessageSegment.reply(99), MessageSegment.text(" 你好 ")])
        event = make_group_event(message, reply=bot_reply())

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {"group_id": GROUP_ID, "message": "模型回复"},
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    runtime.build.assert_awaited_once_with(GROUP_ID, "你好")
    runtime.complete.assert_awaited_once()
    assert runtime.complete.await_args.args[1] == "你好"
    incoming, outgoing = (call.args[0] for call in runtime.record.await_args_list)
    assert incoming.group_id == GROUP_ID
    assert incoming.text == "你好"
    assert incoming.display_name == "群名片"
    assert incoming.is_bot is False
    assert incoming.created_at == datetime.fromtimestamp(1_700_000_000, timezone.utc)
    assert outgoing.group_id == GROUP_ID
    assert outgoing.user_id == BOT_ID
    assert outgoing.text == "模型回复"
    assert outgoing.is_bot is True


async def test_plain_message_only_records_at_default_probability(app: App, runtime):
    event = make_group_event("普通聊天")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 1)

    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_awaited_once_with(GROUP_ID)
    assert runtime.record.await_args.args[0].text == "普通聊天"


async def test_plain_message_replies_at_100_percent(app: App, runtime):
    runtime.get_probability.return_value = 100.0
    event = make_group_event("必中随机回复")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {"group_id": GROUP_ID, "message": "模型回复"},
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    runtime.complete.assert_awaited_once()
    assert [call.args[0].is_bot for call in runtime.record.await_args_list] == [False, True]


@pytest.mark.parametrize(
    "message",
    [
        Message([MessageSegment.text("看图"), MessageSegment.image("file://image")]),
        Message(
            [
                MessageSegment.at(BOT_ID),
                MessageSegment.text("混合点名"),
                MessageSegment.image("file://image"),
            ]
        ),
    ],
)
async def test_mixed_segments_are_not_recorded_or_answered(app: App, runtime, message):
    runtime.get_probability.return_value = 100.0
    event = make_group_event(message, to_me=True)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_not_awaited()


async def test_empty_directed_message_does_not_call_model_or_record(app: App, runtime):
    event = make_group_event(MessageSegment.at(BOT_ID), to_me=True)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_not_awaited()
    runtime.build.assert_not_awaited()
    runtime.complete.assert_not_awaited()


async def test_reply_to_another_user_is_not_directed_or_recorded(app: App, runtime):
    runtime.get_probability.return_value = 100.0
    reply = bot_reply()
    reply.sender.user_id = USER_ID
    event = make_group_event(
        Message([MessageSegment.reply(99), MessageSegment.text("回复别人")]), reply=reply
    )

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_not_awaited()


@pytest.mark.parametrize("kind", ["private", "temporary", "anonymous", "self", "ignored"])
async def test_excluded_events_do_nothing(app: App, runtime, monkeypatch, kind):
    if kind in {"private", "temporary"}:
        event = make_private_event("friend" if kind == "private" else "group")
    elif kind == "anonymous":
        event = make_group_event(
            "匿名消息", anonymous=Anonymous(id=1, name="匿名", flag="flag")
        )
    elif kind == "self":
        event = make_group_event("自身消息", user_id=BOT_ID)
    else:
        monkeypatch.setattr(handlers.config, "group_chat_memory_ignored_qq", {USER_ID})
        event = make_group_event("忽略账号")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [ConfigError("缺少 sk-secret"), ChatAPIError("HTTP 500 sk-secret")]
)
async def test_model_and_config_errors_send_generic_message(
    app: App, runtime, caplog, error
):
    runtime.complete.side_effect = error
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("你好")]), to_me=True
    )

    with caplog.at_level("ERROR"):
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.should_call_api(
                "send_group_msg",
                {"group_id": GROUP_ID, "message": "暂时无法生成回复，请稍后再试。"},
                {"message_id": 123},
            )
            ctx.receive_event(bot, event)
            await ctx.run()
            await wait_until(lambda: runtime.complete.await_count == 1)

    assert runtime.record.await_count == 1
    assert "sk-secret" not in caplog.text
    assert "Traceback" not in caplog.text


async def test_probability_query_is_superuser_only_and_uses_one_decimal(app: App, runtime):
    runtime.get_probability.return_value = 12.26
    event = make_group_event("/回复概率", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "当前群随机回复概率：12.3%", result=None, bot=bot)
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.probability_matcher)

    runtime.get_probability.assert_awaited_once_with(GROUP_ID)
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


@pytest.mark.parametrize(
    ("value", "expected"), [("0", 0.0), ("0.0", 0.0), ("5.5", 5.5), ("100", 100.0)]
)
async def test_probability_setting_accepts_one_decimal_and_is_not_recorded(
    app: App, runtime, value, expected
):
    event = make_group_event(f"/回复概率 {value}", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, f"已设置当前群随机回复概率：{expected:.1f}%", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.probability_matcher)

    runtime.set_probability.assert_awaited_once_with(GROUP_ID, expected)
    runtime.get_probability.assert_not_awaited()
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


@pytest.mark.parametrize("value", ["-1", "100.1", "1.23", ".5", "5.", "abc", "nan"])
async def test_probability_setting_rejects_invalid_input(app: App, runtime, value):
    event = make_group_event(f"/回复概率 {value}", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event,
            "概率需为 0～100 的数字，最多一位小数。",
            result=None,
            bot=bot,
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.probability_matcher)

    runtime.set_probability.assert_not_awaited()
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


@pytest.mark.parametrize("command", ["/回复概率", "/回复概率 50", "/清空群记忆"])
async def test_non_superuser_gets_short_denial_without_side_effects(
    app: App, runtime, command
):
    event = make_group_event(command)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "仅 SUPERUSER 可执行此命令。", result=None, bot=bot)
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.get_probability.assert_not_awaited()
    runtime.set_probability.assert_not_awaited()
    runtime.clear_group.assert_not_awaited()
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


async def test_clear_group_preserves_probability_and_other_groups(
    app: App, runtime, monkeypatch, tmp_path
):
    store = MemoryStore(
        Config(
            group_chat_memory_db_path_override=str(tmp_path / "handlers.db"),
            group_chat_memory_embedding_dimensions=4,
        )
    )
    await store.open()
    monkeypatch.setattr(handlers, "store", store)
    try:
        for group_id in (GROUP_ID, OTHER_GROUP_ID):
            await store.set_probability(group_id, 11.1 if group_id == GROUP_ID else 22.2)
            await store.add_message(
                MemoryMessage(
                    group_id=group_id,
                    user_id=USER_ID,
                    display_name="Alice",
                    text=f"群 {group_id} 的记忆",
                )
            )
        event = make_group_event("/清空群记忆", user_id=SUPERUSER_ID)

        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.should_call_send(event, "已清空当前群记忆。", result=None, bot=bot)
            ctx.receive_event(bot, event)
            ctx.should_finished(handlers.clear_matcher)

        assert await store.recent_messages(GROUP_ID, 20) == []
        assert [m.text for m in await store.recent_messages(OTHER_GROUP_ID, 20)] == [
            f"群 {OTHER_GROUP_ID} 的记忆"
        ]
        assert await store.get_probability(GROUP_ID) == 11.1
        assert await store.get_probability(OTHER_GROUP_ID) == 22.2
        runtime.record.assert_not_awaited()
        runtime.complete.assert_not_awaited()
    finally:
        await store.close()


async def test_lifecycle_startup_and_shutdown_order(runtime, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        handlers.store, "open", AsyncMock(side_effect=lambda: calls.append("open"))
    )
    monkeypatch.setattr(
        handlers.writer, "start", MagicMock(side_effect=lambda: calls.append("start"))
    )
    monkeypatch.setattr(
        runtime.coordinator, "close", AsyncMock(side_effect=lambda: calls.append("coordinator"))
    )
    monkeypatch.setattr(
        handlers.writer,
        "flush",
        AsyncMock(side_effect=lambda *, timeout: calls.append(f"flush:{timeout}")),
    )
    monkeypatch.setattr(
        handlers.writer, "close", AsyncMock(side_effect=lambda: calls.append("writer close"))
    )
    monkeypatch.setattr(
        handlers.store, "close", AsyncMock(side_effect=lambda: calls.append("store close"))
    )

    await handlers.startup()
    await handlers.shutdown()

    assert calls == [
        "open",
        "start",
        "coordinator",
        "flush:5",
        "writer close",
        "store close",
    ]


async def test_shutdown_closes_resources_after_flush_timeout(runtime, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(runtime.coordinator, "close", AsyncMock())
    monkeypatch.setattr(handlers.writer, "flush", AsyncMock(side_effect=asyncio.TimeoutError))
    monkeypatch.setattr(
        handlers.writer, "close", AsyncMock(side_effect=lambda: calls.append("writer"))
    )
    monkeypatch.setattr(
        handlers.store, "close", AsyncMock(side_effect=lambda: calls.append("store"))
    )

    await handlers.shutdown()

    handlers.writer.flush.assert_awaited_once_with(timeout=5)
    assert calls == ["writer", "store"]
