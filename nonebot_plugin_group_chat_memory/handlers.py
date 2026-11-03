from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Annotated
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

from nonebot import get_driver, on_command, on_message
from nonebot import logger as nonebot_logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    Message,
    MessageEvent,
    MessageSegment,
    PrivateMessageEvent,
)
from nonebot.exception import StopPropagation
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER

from .affection import AffectionLeaderboard
from .api import ChatAPIError, ChatClient, EmbeddingClient
from .config import get_settings
from .context import ContextBuilder
from .memory import MemoryWriter
from .models import Message as MemoryMessage
from .replies import ReplyCoordinator
from .store import MemoryStore

logger = logging.getLogger(__name__)

GENERIC_REPLY_ERROR = "暂时无法生成回复，请稍后再试。"
PERMISSION_DENIED = "仅 SUPERUSER 可执行此命令。"
INVALID_PROBABILITY = "概率需为 0～100 的数字，最多一位小数。"
INVALID_HOURLY_LIMIT = "每小时请求上限必须是 1 到 10000 的整数。"
RATE_LIMIT_NOTICE = "本群本小时请求次数已用完，请稍后再来。"
PROBABILITY_PATTERN = re.compile(r"(?:100(?:\.0)?|\d{1,2}(?:\.\d)?)")
SECRET_COMMAND_LOG_PATTERN = re.compile(r"(设置聊天密钥|设置向量密钥)\s+[^\r\n]*")


def _redact_secret_command_logs(record) -> None:
    record["message"] = SECRET_COMMAND_LOG_PATTERN.sub(r"\1 ****", record["message"])


nonebot_logger.configure(patcher=_redact_secret_command_logs)

config = get_settings()
store = MemoryStore(config)
embedding_client = EmbeddingClient(config)
writer = MemoryWriter(config, store, embedding_client)
context_builder = ContextBuilder(config, store, embedding_client)
chat_client = ChatClient(config)
_bots: dict[int, Bot] = {}


async def startup() -> None:
    await store.open()
    config.apply_runtime_config(await store.get_runtime_config())
    writer.start()


async def shutdown() -> None:
    try:
        try:
            try:
                await writer.flush(timeout=5)
            except asyncio.TimeoutError:
                logger.warning("关闭插件时等待记忆写入超时，将保留未完成数据供下次启动恢复。")
        finally:
            await writer.close()
    finally:
        try:
            await coordinator.close()
        finally:
            await store.close()


async def reply(group_id: int, message: MemoryMessage) -> None:
    bot = _bots[group_id]
    try:
        context = await context_builder.build(group_id, message.text)
        response = await chat_client.complete(
            context, message.text, session_id=f"group:{group_id}"
        )
    except Exception as exc:
        status = f" HTTP {exc.status_code}" if isinstance(exc, ChatAPIError) else ""
        logger.error("群 %s 的聊天回复失败：%s%s", group_id, type(exc).__name__, status)
        await bot.send_group_msg(group_id=group_id, message=GENERIC_REPLY_ERROR)
        return

    await bot.send_group_msg(group_id=group_id, message=MessageSegment.text(response))
    await writer.record(
        MemoryMessage(
            group_id=group_id,
            user_id=int(bot.self_id),
            display_name="Bot",
            text=response,
            is_bot=True,
        )
    )


coordinator = ReplyCoordinator(store, reply)
affection_leaderboard = AffectionLeaderboard()


async def _eligible_group_message(bot: Bot, event: GroupMessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    if event.anonymous is not None:
        return False
    if event.user_id in {event.self_id, int(bot.self_id)}:
        return False
    return event.user_id not in config.group_chat_memory_ignored_qq


settings_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()


def _settings_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    if (lock := settings_locks.get(loop)) is None:
        lock = asyncio.Lock()
        settings_locks[loop] = lock
    return lock


async def apply_setting(key: str, value: str) -> None:
    async with _settings_lock():
        values = {
            "chat_base_url": config.group_chat_memory_chat_base_url,
            "chat_api_key": config.group_chat_memory_chat_api_key.get_secret_value(),
            "chat_model": config.group_chat_memory_chat_model,
            "embedding_api_key": config.group_chat_memory_embedding_api_key.get_secret_value(),
        }
        values[key] = value
        await store.set_runtime_config(key, value)
        config.apply_runtime_config(values)


chat_url_matcher = on_command("设置聊天地址", priority=-1001, block=True)
chat_key_matcher = on_command("设置聊天密钥", priority=-1001, block=True)
chat_model_matcher = on_command("设置聊天模型", priority=-1001, block=True)
embedding_key_matcher = on_command("设置向量密钥", priority=-1001, block=True)
model_config_matcher = on_command("模型配置", priority=-1001, block=True)


def _valid_http_url(value: str) -> bool:
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        return False
    try:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        _ = parsed.port
    except ValueError:
        return False
    return True


async def _handle_setting(matcher, bot: Bot, event: MessageEvent, args: Message, key: str) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await matcher.finish("请在私聊中设置模型配置。")
    if not await SUPERUSER(bot, event):
        await matcher.finish(PERMISSION_DENIED)

    value = args.extract_plain_text().strip()
    if not value:
        await matcher.finish("配置参数不能为空。")
    if key == "chat_base_url" and not _valid_http_url(value):
        await matcher.finish("聊天 API 地址需为有效的 http(s) URL。")
    try:
        await apply_setting(key, value)
    except Exception as exc:
        logger.error("保存模型配置失败：%s", type(exc).__name__)
        await matcher.finish("配置保存失败，请稍后再试。")
    await matcher.finish("模型配置已更新。")


@chat_url_matcher.handle()
async def handle_chat_url(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(chat_url_matcher, bot, event, args, "chat_base_url")


@chat_key_matcher.handle()
async def handle_chat_key(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(chat_key_matcher, bot, event, args, "chat_api_key")


@chat_model_matcher.handle()
async def handle_chat_model(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(chat_model_matcher, bot, event, args, "chat_model")


@embedding_key_matcher.handle()
async def handle_embedding_key(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(embedding_key_matcher, bot, event, args, "embedding_api_key")


@model_config_matcher.handle()
async def handle_model_config(bot: Bot, event: MessageEvent) -> None:
    if not await SUPERUSER(bot, event):
        await model_config_matcher.finish(PERMISSION_DENIED)

    def status(secret: str) -> str:
        return "已配置（****）" if secret else "未配置"

    await model_config_matcher.finish(
        f"聊天地址：{config.group_chat_memory_chat_base_url or '未配置'}\n"
        f"聊天模型：{config.group_chat_memory_chat_model}\n"
        f"聊天密钥：{status(config.group_chat_memory_chat_api_key.get_secret_value())}\n"
        "Embedding 密钥："
        f"{status(config.group_chat_memory_embedding_api_key.get_secret_value())}"
    )


probability_matcher = on_command(
    "回复概率", rule=_eligible_group_message, priority=-1001, block=True
)
clear_matcher = on_command(
    "清空群记忆", rule=_eligible_group_message, priority=-1001, block=True
)
hourly_limit_matcher = on_command(
    "每小时次数", rule=_eligible_group_message, priority=-1001, block=True
)
group_message_matcher = on_message(rule=_eligible_group_message, priority=-1000, block=False)


@probability_matcher.handle()
async def handle_probability(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await probability_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if not text:
        probability = await store.get_probability(event.group_id)
        await probability_matcher.finish(f"当前群随机回复概率：{probability:.1f}%")
    if PROBABILITY_PATTERN.fullmatch(text) is None:
        await probability_matcher.finish(INVALID_PROBABILITY)

    probability = float(text)
    await store.set_probability(event.group_id, probability)
    await probability_matcher.finish(f"已设置当前群随机回复概率：{probability:.1f}%")


@clear_matcher.handle()
async def handle_clear(bot: Bot, event: GroupMessageEvent) -> None:
    if not await SUPERUSER(bot, event):
        await clear_matcher.finish(PERMISSION_DENIED)
    await store.clear_group(event.group_id)
    await clear_matcher.finish("已清空当前群记忆。")


@hourly_limit_matcher.handle()
async def handle_hourly_limit(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await hourly_limit_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if not text:
        limit = await store.get_hourly_limit(event.group_id)
        await hourly_limit_matcher.finish(f"当前群每小时请求上限：{limit} 次")
    if not text.isdigit() or not 1 <= int(text) <= 10000:
        await hourly_limit_matcher.finish(INVALID_HOURLY_LIMIT)

    limit = int(text)
    await store.set_hourly_limit(event.group_id, limit)
    await hourly_limit_matcher.finish(f"已设置当前群每小时请求上限：{limit} 次")


async def _should_random_reply(event: GroupMessageEvent) -> bool:
    """随机触发判断：好感榜成员至少按配置概率触发，其余按群概率。"""
    minimum = 0.0
    if (
        config.group_chat_memory_affection_enabled
        and config.group_chat_memory_affection_probability > 0
        and await affection_leaderboard.is_member(event.group_id, event.user_id)
    ):
        minimum = config.group_chat_memory_affection_probability
    return await coordinator.should_random_reply(
        event.group_id, minimum_probability=minimum
    )


@group_message_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    explicit_at = any(
        segment.type == "at" and segment.data.get("qq") == str(bot.self_id)
        for segment in event.message
    )
    directed = explicit_at or (
        event.reply is not None and event.reply.sender.user_id == int(bot.self_id)
    )
    allowed_types = {"text", "at", "reply"}
    if any(segment.type not in allowed_types for segment in event.message):
        return

    text = event.message.extract_plain_text().strip()
    if not text:
        return

    memory_message = MemoryMessage(
        group_id=event.group_id,
        user_id=event.user_id,
        display_name=event.sender.card or event.sender.nickname or str(event.user_id),
        text=text,
        created_at=datetime.fromtimestamp(event.time, timezone.utc),
    )
    _bots[event.group_id] = bot
    await writer.record(memory_message)
    if directed:
        if await store.consume_request_slot(event.group_id):
            coordinator.submit(event.group_id, memory_message, explicit=True)
        else:
            await bot.send_group_msg(group_id=event.group_id, message=RATE_LIMIT_NOTICE)
        raise StopPropagation
    elif (
        event.reply is None
        and all(segment.type == "text" for segment in event.message)
        and await _should_random_reply(event)
        and await store.consume_request_slot(event.group_id)
    ):
        coordinator.submit(event.group_id, memory_message, explicit=False)


driver = get_driver()
driver.on_startup(startup)
driver.on_shutdown(shutdown)
