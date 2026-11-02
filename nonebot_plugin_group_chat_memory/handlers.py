from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Annotated

from nonebot import get_driver, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER

from .api import ChatClient, EmbeddingClient
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
PROBABILITY_PATTERN = re.compile(r"(?:100(?:\.0)?|\d{1,2}(?:\.\d)?)")

config = get_settings()
store = MemoryStore(config)
embedding_client = EmbeddingClient(config)
writer = MemoryWriter(config, store, embedding_client)
context_builder = ContextBuilder(config, store, embedding_client)
chat_client = ChatClient(config)
_bots: dict[int, Bot] = {}


async def startup() -> None:
    await store.open()
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
        response = await chat_client.complete(context, message.text)
    except Exception as exc:
        logger.error("群 %s 的聊天回复失败：%s", group_id, type(exc).__name__)
        await bot.send_group_msg(group_id=group_id, message=GENERIC_REPLY_ERROR)
        return

    await bot.send_group_msg(group_id=group_id, message=response)
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


async def _eligible_group_message(bot: Bot, event: GroupMessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    if event.anonymous is not None:
        return False
    if event.user_id in {event.self_id, int(bot.self_id)}:
        return False
    return event.user_id not in config.group_chat_memory_ignored_qq


probability_matcher = on_command(
    "回复概率", rule=_eligible_group_message, priority=5, block=True
)
clear_matcher = on_command("清空群记忆", rule=_eligible_group_message, priority=5, block=True)
group_message_matcher = on_message(rule=_eligible_group_message, priority=20, block=False)


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


@group_message_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    directed = event.to_me or (
        event.reply is not None and event.reply.sender.user_id == int(bot.self_id)
    )
    allowed_types = {"text", "at", "reply"} if directed else {"text"}
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
        coordinator.submit(event.group_id, memory_message, explicit=True)
    elif event.reply is None and await coordinator.should_random_reply(event.group_id):
        coordinator.submit(event.group_id, memory_message, explicit=False)


driver = get_driver()
driver.on_startup(startup)
driver.on_shutdown(shutdown)
