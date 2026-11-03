from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import nonebot
import pytest
import pytest_asyncio
from nonebot.adapters.onebot.v11 import (
    Adapter,
    Bot,
    GroupMessageEvent,
    Message,
    MessageSegment,
    PrivateMessageEvent,
)
from nonebot.adapters.onebot.v11.event import Anonymous, Reply, Sender
from nonebot.exception import StopPropagation
from nonebot.message import handle_event
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


def make_private_event(
    sub_type: str = "friend", *, content: str = "私聊", user_id: int = USER_ID
) -> PrivateMessageEvent:
    message = Message(content)
    return PrivateMessageEvent(
        time=1_700_000_000,
        self_id=BOT_ID,
        post_type="message",
        message_type="private",
        sub_type=sub_type,
        message_id=1,
        user_id=user_id,
        message=message,
        original_message=message,
        raw_message=content,
        font=0,
        sender=Sender(user_id=user_id, nickname="昵称"),
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
    get_hourly_limit = AsyncMock(return_value=10)
    set_hourly_limit = AsyncMock()
    get_affection_enabled = AsyncMock(return_value=True)
    set_affection_enabled = AsyncMock()
    get_affection_probability = AsyncMock(return_value=30.0)
    set_affection_probability = AsyncMock()
    consume_request_slot = AsyncMock(return_value=True)
    monkeypatch.setattr(handlers.writer, "record", record)
    monkeypatch.setattr(handlers.context_builder, "build", build)
    monkeypatch.setattr(handlers.chat_client, "complete", complete)
    monkeypatch.setattr(handlers.store, "get_probability", get_probability)
    monkeypatch.setattr(handlers.store, "set_probability", set_probability)
    monkeypatch.setattr(handlers.store, "clear_group", clear_group)
    monkeypatch.setattr(handlers.store, "get_hourly_limit", get_hourly_limit)
    monkeypatch.setattr(handlers.store, "set_hourly_limit", set_hourly_limit)
    monkeypatch.setattr(handlers.store, "get_affection_enabled", get_affection_enabled)
    monkeypatch.setattr(handlers.store, "set_affection_enabled", set_affection_enabled)
    monkeypatch.setattr(
        handlers.store, "get_affection_probability", get_affection_probability
    )
    monkeypatch.setattr(
        handlers.store, "set_affection_probability", set_affection_probability
    )
    monkeypatch.setattr(handlers.store, "consume_request_slot", consume_request_slot)
    coordinator = ReplyCoordinator(handlers.store, handlers.reply)
    monkeypatch.setattr(handlers, "coordinator", coordinator)
    saved_config = {
        field: getattr(handlers.config, field)
        for field in (
            "group_chat_memory_chat_base_url",
            "group_chat_memory_chat_api_key",
            "group_chat_memory_chat_model",
            "group_chat_memory_embedding_api_key",
        )
    }
    handlers._bots.clear()
    try:
        yield SimpleNamespace(
            record=record,
            build=build,
            complete=complete,
            get_probability=get_probability,
            set_probability=set_probability,
            clear_group=clear_group,
            get_hourly_limit=get_hourly_limit,
            set_hourly_limit=set_hourly_limit,
            get_affection_enabled=get_affection_enabled,
            set_affection_enabled=set_affection_enabled,
            get_affection_probability=get_affection_probability,
            set_affection_probability=set_affection_probability,
            consume_request_slot=consume_request_slot,
            coordinator=coordinator,
        )
    finally:
        await coordinator.close()
        for field, value in saved_config.items():
            setattr(handlers.config, field, value)
        handlers._bots.clear()


@pytest.mark.parametrize("kind", ["at", "reply"])
async def test_directed_message_always_replies_and_records_bot(app: App, runtime, kind):
    if kind == "at":
        message = Message([MessageSegment.at(BOT_ID), MessageSegment.text(" 你好 ")])
        event = make_group_event(message)
    else:
        message = Message([MessageSegment.reply(99), MessageSegment.text(" 你好 ")])
        event = make_group_event(message, reply=bot_reply())

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


async def test_model_output_is_sent_and_stored_as_literal_plain_text(app: App, runtime):
    response = "字面量 [CQ:at,qq=all] 不应执行"
    runtime.complete.return_value = response
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("你好")]), to_me=True
    )

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        bot.send_group_msg = AsyncMock(return_value={"message_id": 123})
        ctx.receive_event(bot, event)
        await ctx.run()
        await wait_until(lambda: runtime.record.await_count == 2)

    bot.send_group_msg.assert_awaited_once()
    assert bot.send_group_msg.await_args.kwargs["group_id"] == GROUP_ID
    outgoing = bot.send_group_msg.await_args.kwargs["message"]
    assert isinstance(outgoing, MessageSegment)
    assert outgoing.type == "text"
    assert outgoing.data == {"text": response}
    assert runtime.record.await_args_list[-1].args[0].text == response


async def test_plain_message_only_records_at_default_probability(app: App, runtime):
    event = make_group_event("普通聊天")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_awaited_once_with(GROUP_ID)
    assert runtime.record.await_args.args[0].text == "普通聊天"


async def test_nickname_only_to_me_does_not_force_reply(app: App, runtime):
    event = make_group_event("昵称提到了机器人", to_me=True)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_awaited_once_with(GROUP_ID)
    assert runtime.record.await_args.args[0].text == "昵称提到了机器人"


async def test_plain_message_replies_at_100_percent(app: App, runtime):
    runtime.get_probability.return_value = 100.0
    event = make_group_event("必中随机回复")

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


async def test_reply_to_another_user_is_recorded_without_random_reply(app: App, runtime):
    runtime.get_probability.return_value = 100.0
    reply = bot_reply()
    reply.sender.user_id = USER_ID
    message = Message([MessageSegment.reply(99), MessageSegment.text("回复别人")])
    event = make_group_event(message, reply=reply)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_awaited_once()
    assert runtime.record.await_args.args[0].text == "回复别人"
    runtime.complete.assert_not_awaited()
    runtime.get_probability.assert_not_awaited()


async def test_at_another_user_is_recorded_without_random_reply(app: App, runtime):
    runtime.get_probability.return_value = 100.0
    message = Message([MessageSegment.at(USER_ID + 1), MessageSegment.text("提到别人")])
    event = make_group_event(message)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_awaited_once()
    assert runtime.record.await_args.args[0].text == "提到别人"
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
    "error",
    [ConfigError("缺少 sk-secret"), ChatAPIError("sk-secret", status_code=500)],
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
    if isinstance(error, ChatAPIError) and error.status_code is not None:
        assert f"HTTP {error.status_code}" in caplog.text


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


@pytest.mark.parametrize(
    "command",
    [
        "/回复概率",
        "/回复概率 50",
        "/清空群记忆",
        "/每小时次数 20",
        "/好感概率 30",
        "/好感开关 关",
    ],
)
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
        handlers.store,
        "get_runtime_config",
        AsyncMock(side_effect=lambda: calls.append("load") or {}),
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
        "load",
        "start",
        "flush:5",
        "writer close",
        "coordinator",
        "store close",
    ]


async def test_shutdown_closes_resources_after_flush_timeout(runtime, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        runtime.coordinator, "close", AsyncMock(side_effect=lambda: calls.append("coordinator"))
    )
    monkeypatch.setattr(handlers.writer, "flush", AsyncMock(side_effect=asyncio.TimeoutError))
    monkeypatch.setattr(
        handlers.writer, "close", AsyncMock(side_effect=lambda: calls.append("writer"))
    )
    monkeypatch.setattr(
        handlers.store, "close", AsyncMock(side_effect=lambda: calls.append("store"))
    )

    await handlers.shutdown()

    handlers.writer.flush.assert_awaited_once_with(timeout=5)
    assert calls == ["writer", "coordinator", "store"]


@pytest.mark.parametrize("failing", ["writer", "coordinator"])
async def test_shutdown_continues_cleanup_after_close_failure(runtime, monkeypatch, failing):
    calls: list[str] = []

    async def close_writer() -> None:
        calls.append("writer")
        if failing == "writer":
            raise RuntimeError("writer failed")

    async def close_coordinator() -> None:
        calls.append("coordinator")
        if failing == "coordinator":
            raise RuntimeError("coordinator failed")

    monkeypatch.setattr(
        handlers,
        "coordinator",
        SimpleNamespace(close=AsyncMock(side_effect=close_coordinator)),
    )
    monkeypatch.setattr(
        handlers.writer,
        "flush",
        AsyncMock(side_effect=lambda *, timeout: calls.append("flush")),
    )
    monkeypatch.setattr(handlers.writer, "close", AsyncMock(side_effect=close_writer))
    monkeypatch.setattr(
        handlers.store, "close", AsyncMock(side_effect=lambda: calls.append("store"))
    )

    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        await handlers.shutdown()

    assert calls == ["flush", "writer", "coordinator", "store"]


@pytest.mark.parametrize(
    ("command", "key", "value", "field"),
    [
        (
            "/设置聊天地址 https://chat.example.com/v1",
            "chat_base_url",
            "https://chat.example.com/v1",
            "group_chat_memory_chat_base_url",
        ),
        (
            "/设置聊天密钥 chat-secret",
            "chat_api_key",
            "chat-secret",
            "group_chat_memory_chat_api_key",
        ),
        (
            "/设置聊天模型 deepseek-v4.1-flash",
            "chat_model",
            "deepseek-v4.1-flash",
            "group_chat_memory_chat_model",
        ),
        (
            "/设置向量密钥 embedding-secret",
            "embedding_api_key",
            "embedding-secret",
            "group_chat_memory_embedding_api_key",
        ),
    ],
)
async def test_private_superuser_updates_runtime_config_immediately(
    app: App, runtime, monkeypatch, command, key, value, field
):
    setter = AsyncMock()
    monkeypatch.setattr(handlers.store, "set_runtime_config", setter)
    event = make_private_event(content=command, user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "模型配置已更新。", result=None, bot=bot)
        ctx.receive_event(bot, event)
        await ctx.run()

    setter.assert_awaited_once_with(key, value)
    configured = getattr(handlers.config, field)
    if hasattr(configured, "get_secret_value"):
        configured = configured.get_secret_value()
    assert configured == value
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


async def test_group_setting_command_is_rejected_without_memory(app: App, runtime, monkeypatch):
    setter = AsyncMock()
    monkeypatch.setattr(handlers.store, "set_runtime_config", setter)
    event = make_group_event("/设置聊天密钥 group-secret", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "请在私聊中设置模型配置。", result=None, bot=bot)
        ctx.receive_event(bot, event)
        await ctx.run()

    setter.assert_not_awaited()
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


async def test_private_non_superuser_cannot_update_config(app: App, runtime, monkeypatch):
    setter = AsyncMock()
    monkeypatch.setattr(handlers.store, "set_runtime_config", setter)
    event = make_private_event(content="/设置聊天密钥 secret")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "仅 SUPERUSER 可执行此命令。", result=None, bot=bot)
        ctx.receive_event(bot, event)
        await ctx.run()

    setter.assert_not_awaited()


@pytest.mark.parametrize(
    "command",
    ["/设置聊天地址", "/设置聊天密钥", "/设置聊天模型", "/设置向量密钥"],
)
async def test_setting_command_rejects_empty_argument(app: App, runtime, monkeypatch, command):
    setter = AsyncMock()
    monkeypatch.setattr(handlers.store, "set_runtime_config", setter)
    event = make_private_event(content=command, user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "配置参数不能为空。", result=None, bot=bot)
        ctx.receive_event(bot, event)
        await ctx.run()

    setter.assert_not_awaited()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "https:///missing-host",
        "not-a-url",
        "https://exa mple.com/v1",
        "https://example.com:bad/v1",
        "http://[::1/v1",
    ],
)
async def test_chat_url_must_be_http_or_https_with_host(app: App, runtime, monkeypatch, url):
    setter = AsyncMock()
    monkeypatch.setattr(handlers.store, "set_runtime_config", setter)
    event = make_private_event(content=f"/设置聊天地址 {url}", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "聊天 API 地址需为有效的 http(s) URL。", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        await ctx.run()

    setter.assert_not_awaited()


async def test_model_config_query_masks_both_secrets(app: App, runtime):
    handlers.config.apply_runtime_config(
        {
            "chat_base_url": "https://chat.example.com/v1",
            "chat_api_key": "chat-full-secret",
            "chat_model": "deepseek-v4.1-flash",
            "embedding_api_key": "embedding-full-secret",
        }
    )
    event = make_group_event("/模型配置", user_id=SUPERUSER_ID)
    response = (
        "聊天地址：https://chat.example.com/v1\n"
        "聊天模型：deepseek-v4.1-flash\n"
        "聊天密钥：已配置（****）\n"
        "Embedding 密钥：已配置（****）"
    )

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, response, result=None, bot=bot)
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_not_awaited()
    assert "chat-full-secret" not in response
    assert "embedding-full-secret" not in response


async def test_failed_setting_write_preserves_running_config(runtime, monkeypatch):
    handlers.config.apply_runtime_config({"chat_model": "old-model"})
    monkeypatch.setattr(
        handlers.store,
        "set_runtime_config",
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        await handlers.apply_setting("chat_model", "new-model")

    assert handlers.config.group_chat_memory_chat_model == "old-model"


@pytest.mark.parametrize(
    ("command", "secret"),
    [
        ("/设置聊天密钥 chat-log-secret", "chat-log-secret"),
        ("/设置向量密钥 embedding-log-secret", "embedding-log-secret"),
    ],
)
async def test_secret_commands_are_redacted_from_event_logs(
    runtime, monkeypatch, command, secret
):
    monkeypatch.setattr(handlers.store, "set_runtime_config", AsyncMock())
    event = make_private_event(content=command, user_id=SUPERUSER_ID)
    messages: list[str] = []
    sink_id = handlers.nonebot_logger.add(lambda message: messages.append(str(message)))
    try:
        bot = Bot(nonebot.get_adapter(Adapter), str(BOT_ID))
        bot.send = AsyncMock()
        await handle_event(bot, event)
    finally:
        handlers.nonebot_logger.remove(sink_id)

    logs = "".join(messages)
    assert secret not in logs
    assert "****" in logs


async def test_concurrent_settings_keep_memory_consistent_with_database(
    runtime, monkeypatch
):
    saved: dict[str, str] = {}
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def set_runtime_config(key: str, value: str) -> None:
        if key == "chat_base_url":
            first_started.set()
            await release_first.wait()
        saved[key] = value

    monkeypatch.setattr(handlers.store, "set_runtime_config", set_runtime_config)
    first = asyncio.create_task(
        handlers.apply_setting("chat_base_url", "https://new.example/v1")
    )
    await first_started.wait()
    second = asyncio.create_task(handlers.apply_setting("chat_model", "new-model"))
    release_first.set()
    await asyncio.gather(first, second)

    assert saved == {
        "chat_base_url": "https://new.example/v1",
        "chat_model": "new-model",
    }
    assert handlers.config.group_chat_memory_chat_base_url == saved["chat_base_url"]
    assert handlers.config.group_chat_memory_chat_model == saved["chat_model"]


async def test_concurrent_updates_to_same_field_match_final_database_value(
    runtime, monkeypatch
):
    saved: dict[str, str] = {}

    async def set_runtime_config(key: str, value: str) -> None:
        await asyncio.sleep(0)
        saved[key] = value

    monkeypatch.setattr(handlers.store, "set_runtime_config", set_runtime_config)
    await asyncio.gather(
        handlers.apply_setting("chat_model", "model-a"),
        handlers.apply_setting("chat_model", "model-b"),
    )

    assert handlers.config.group_chat_memory_chat_model == saved["chat_model"]


def test_group_commands_and_chat_have_highest_priority():
    assert handlers.probability_matcher.priority == -1001
    assert handlers.clear_matcher.priority == -1001
    assert handlers.hourly_limit_matcher.priority == -1001
    assert handlers.chat_url_matcher.priority == -1001
    assert handlers.chat_key_matcher.priority == -1001
    assert handlers.chat_model_matcher.priority == -1001
    assert handlers.embedding_key_matcher.priority == -1001
    assert handlers.model_config_matcher.priority == -1001
    assert handlers.group_message_matcher.priority == -1000


async def test_directed_message_stops_lower_priority_plugins(runtime, monkeypatch):
    coordinator = SimpleNamespace(submit=MagicMock())
    monkeypatch.setattr(handlers, "coordinator", coordinator)
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("点名")])
    )
    bot = Bot(nonebot.get_adapter(Adapter), str(BOT_ID))

    with pytest.raises(StopPropagation):
        await handlers.handle_group_message(bot, event)

    runtime.record.assert_awaited_once()
    coordinator.submit.assert_called_once()
    assert coordinator.submit.call_args.kwargs["explicit"] is True


async def test_hourly_limit_query_is_superuser_only(app: App, runtime):
    runtime.get_hourly_limit.return_value = 7
    event = make_group_event("/每小时次数", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "当前群每小时请求上限：7 次", result=None, bot=bot)
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.hourly_limit_matcher)

    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


async def test_hourly_limit_setting_is_not_recorded(app: App, runtime):
    event = make_group_event("/每小时次数 20", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, "已设置当前群每小时请求上限：20 次", result=None, bot=bot)
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.hourly_limit_matcher)

    runtime.set_hourly_limit.assert_awaited_once_with(GROUP_ID, 20)
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


@pytest.mark.parametrize("value", ["0", "10001", "1.5", "abc", "-1"])
async def test_hourly_limit_rejects_invalid_values(app: App, runtime, value):
    event = make_group_event(f"/每小时次数 {value}", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "每小时请求上限必须是 1 到 10000 的整数。", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.hourly_limit_matcher)

    runtime.set_hourly_limit.assert_not_awaited()


async def test_explicit_rate_limit_sends_notice_without_model(app: App, runtime):
    runtime.consume_request_slot.return_value = False
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("点名")])
    )

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {
                "group_id": GROUP_ID,
                "message": "本群本小时请求次数已用完，请稍后再来。",
            },
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_awaited_once()
    runtime.complete.assert_not_awaited()
    runtime.consume_request_slot.assert_awaited_once_with(GROUP_ID)


async def test_random_rate_limit_is_silent_without_model(app: App, runtime):
    runtime.get_probability.return_value = 100.0
    runtime.consume_request_slot.return_value = False
    event = make_group_event("随机消息")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.record.assert_awaited_once()
    runtime.complete.assert_not_awaited()
    runtime.consume_request_slot.assert_awaited_once_with(GROUP_ID)


class FakeLeaderboard:
    def __init__(self, members: set[int] | None = None) -> None:
        self.members = members or set()
        self.calls: list[tuple[int, int]] = []

    async def is_member(self, group_id: int, user_id: int) -> bool:
        self.calls.append((group_id, user_id))
        return user_id in self.members


def use_random(monkeypatch, value: float) -> ReplyCoordinator:
    coordinator = ReplyCoordinator(
        handlers.store, handlers.reply, random_value=lambda: value
    )
    monkeypatch.setattr(handlers, "coordinator", coordinator)
    return coordinator


async def test_leaderboard_member_replies_at_thirty_percent(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    leaderboard = FakeLeaderboard({USER_ID})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    coordinator = use_random(monkeypatch, 0.29)
    event = make_group_event("好感榜成员说话")

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

    assert leaderboard.calls == [(GROUP_ID, USER_ID)]
    runtime.complete.assert_awaited_once()


async def test_leaderboard_member_below_threshold_stays_silent(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    leaderboard = FakeLeaderboard({USER_ID})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    coordinator = use_random(monkeypatch, 0.30)
    event = make_group_event("好感榜成员说话")

    try:
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.receive_event(bot, event)
            await ctx.run()
    finally:
        await coordinator.close()

    assert leaderboard.calls == [(GROUP_ID, USER_ID)]
    runtime.consume_request_slot.assert_not_awaited()
    runtime.complete.assert_not_awaited()


async def test_non_leaderboard_member_keeps_group_probability(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    leaderboard = FakeLeaderboard({9999})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    coordinator = use_random(monkeypatch, 0.0)
    event = make_group_event("普通群友说话")

    try:
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.receive_event(bot, event)
            await ctx.run()
    finally:
        await coordinator.close()

    assert leaderboard.calls == [(GROUP_ID, USER_ID)]
    runtime.consume_request_slot.assert_not_awaited()


async def test_group_probability_above_affection_boost_is_kept(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 100.0
    leaderboard = FakeLeaderboard(set())
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    coordinator = use_random(monkeypatch, 0.99)
    event = make_group_event("普通群友说话")

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

    runtime.complete.assert_awaited_once()


async def test_affection_boost_can_be_disabled(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    runtime.get_affection_enabled.return_value = False
    leaderboard = FakeLeaderboard({USER_ID})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    coordinator = use_random(monkeypatch, 0.0)
    event = make_group_event("好感榜成员说话")

    try:
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.receive_event(bot, event)
            await ctx.run()
    finally:
        await coordinator.close()

    assert leaderboard.calls == []
    runtime.consume_request_slot.assert_not_awaited()


async def test_directed_message_skips_leaderboard_lookup(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 100.0
    leaderboard = FakeLeaderboard({USER_ID})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("你好")])
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

    assert leaderboard.calls == []
    runtime.get_probability.assert_not_awaited()


async def test_affection_probability_query_reports_current_value(app: App, runtime):
    runtime.get_affection_probability.return_value = 30.0
    event = make_group_event("/好感概率", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "当前群好感榜触发概率：30.0%", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.affection_probability_matcher)

    runtime.get_affection_probability.assert_awaited_once_with(GROUP_ID)
    runtime.set_affection_probability.assert_not_awaited()
    runtime.record.assert_not_awaited()


async def test_affection_probability_setting_is_persisted(app: App, runtime):
    event = make_group_event("/好感概率 45.5", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "已设置当前群好感榜触发概率：45.5%", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.affection_probability_matcher)

    runtime.set_affection_probability.assert_awaited_once_with(GROUP_ID, 45.5)
    runtime.record.assert_not_awaited()
    runtime.complete.assert_not_awaited()


@pytest.mark.parametrize("value", ["-1", "100.1", "abc", "30.55"])
async def test_affection_probability_rejects_invalid_values(app: App, runtime, value):
    event = make_group_event(f"/好感概率 {value}", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "好感榜触发概率需为 0～100 的数字，最多一位小数。", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.affection_probability_matcher)

    runtime.set_affection_probability.assert_not_awaited()


async def test_affection_switch_query_reports_disabled_state(app: App, runtime):
    runtime.get_affection_enabled.return_value = False
    event = make_group_event("/好感开关", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "当前群好感榜触发：已关闭", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.affection_switch_matcher)

    runtime.get_affection_enabled.assert_awaited_once_with(GROUP_ID)
    runtime.set_affection_enabled.assert_not_awaited()


@pytest.mark.parametrize(
    ("value", "enabled", "reply"),
    [
        ("开", True, "已开启当前群好感榜触发。"),
        ("ON", True, "已开启当前群好感榜触发。"),
        ("关", False, "已关闭当前群好感榜触发。"),
        ("off", False, "已关闭当前群好感榜触发。"),
    ],
)
async def test_affection_switch_setting(app: App, runtime, value, enabled, reply):
    event = make_group_event(f"/好感开关 {value}", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(event, reply, result=None, bot=bot)
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.affection_switch_matcher)

    runtime.set_affection_enabled.assert_awaited_once_with(GROUP_ID, enabled)


async def test_affection_switch_rejects_unknown_value(app: App, runtime):
    event = make_group_event("/好感开关 可能吧", user_id=SUPERUSER_ID)

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_send(
            event, "开关参数只能是“开”或“关”。", result=None, bot=bot
        )
        ctx.receive_event(bot, event)
        ctx.should_finished(handlers.affection_switch_matcher)

    runtime.set_affection_enabled.assert_not_awaited()


async def test_affection_command_changes_real_store_and_reply(app: App, monkeypatch, tmp_path):
    store = MemoryStore(
        Config(
            group_chat_memory_db_path_override=str(tmp_path / "handlers.db"),
            group_chat_memory_embedding_dimensions=4,
        )
    )
    await store.open()
    monkeypatch.setattr(handlers, "store", store)
    monkeypatch.setattr(handlers.writer, "record", AsyncMock())
    monkeypatch.setattr(handlers.context_builder, "build", AsyncMock(return_value=ChatContext()))
    monkeypatch.setattr(handlers.chat_client, "complete", AsyncMock(return_value="模型回复"))
    monkeypatch.setattr(handlers, "affection_leaderboard", FakeLeaderboard({USER_ID}))
    coordinator = ReplyCoordinator(store, handlers.reply)
    monkeypatch.setattr(handlers, "coordinator", coordinator)
    handlers._bots.clear()

    try:
        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            command = make_group_event("/好感概率 100", user_id=SUPERUSER_ID)
            ctx.should_call_send(
                command, "已设置当前群好感榜触发概率：100.0%", result=None, bot=bot
            )
            ctx.receive_event(bot, command)
            await ctx.run()

        assert await store.get_affection_probability(GROUP_ID) == 100.0
        assert await store.get_probability(GROUP_ID) == 0.0

        async with app.test_matcher() as ctx:
            bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
            ctx.should_call_api(
                "send_group_msg",
                {"group_id": GROUP_ID, "message": MessageSegment.text("模型回复")},
                {"message_id": 123},
            )
            ctx.receive_event(bot, make_group_event("好感榜成员说话"))
            await ctx.run()
            await wait_until(lambda: handlers.writer.record.await_count == 2)

        assert await store.get_affection_enabled(GROUP_ID) is True
        handlers.chat_client.complete.assert_awaited_once()
    finally:
        await coordinator.close()
        await store.close()
        handlers._bots.clear()


async def test_affection_trigger_bypasses_hourly_limit(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 0.0
    runtime.consume_request_slot.return_value = False
    leaderboard = FakeLeaderboard({USER_ID})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    coordinator = use_random(monkeypatch, 0.29)
    event = make_group_event("好感榜成员说话")

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

    runtime.consume_request_slot.assert_not_awaited()
    runtime.complete.assert_awaited_once()


async def test_non_member_still_subject_to_hourly_limit(app: App, runtime, monkeypatch):
    runtime.get_probability.return_value = 100.0
    runtime.consume_request_slot.return_value = False
    leaderboard = FakeLeaderboard({9999})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    event = make_group_event("普通群友说话")

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.consume_request_slot.assert_awaited_once_with(GROUP_ID)
    runtime.complete.assert_not_awaited()


async def test_directed_message_still_subject_to_hourly_limit(app: App, runtime, monkeypatch):
    runtime.consume_request_slot.return_value = False
    leaderboard = FakeLeaderboard({USER_ID})
    monkeypatch.setattr(handlers, "affection_leaderboard", leaderboard)
    event = make_group_event(
        Message([MessageSegment.at(BOT_ID), MessageSegment.text("点名")])
    )

    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        ctx.should_call_api(
            "send_group_msg",
            {
                "group_id": GROUP_ID,
                "message": "本群本小时请求次数已用完，请稍后再来。",
            },
            {"message_id": 123},
        )
        ctx.receive_event(bot, event)
        await ctx.run()

    runtime.consume_request_slot.assert_awaited_once_with(GROUP_ID)
    runtime.complete.assert_not_awaited()
