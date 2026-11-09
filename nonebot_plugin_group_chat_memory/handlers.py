from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
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
from .meme import MemeProvider
from .memory import MemoryWriter
from .models import Message as MemoryMessage
from .replies import ReplyCoordinator
from .store import MemoryStore

logger = logging.getLogger(__name__)

GENERIC_REPLY_ERROR = "暂时无法生成回复，请稍后再试。"
PERMISSION_DENIED = "仅 SUPERUSER 可执行此命令。"
INVALID_PROBABILITY = "概率需为 0～100 的数字，最多一位小数。"
INVALID_QUOTE_PROBABILITY = "引用概率需为 0～100 的数字，最多一位小数。"
INVALID_HOURLY_LIMIT = "每小时请求上限必须是 1 到 10000 的整数。"
INVALID_AFFECTION_PROBABILITY = "好感榜触发概率需为 0～100 的数字，最多一位小数。"
INVALID_AFFECTION_SWITCH = "开关参数只能是“开”或“关”。"
INVALID_SUPERUSER_REPLY_SWITCH = "开关参数只能是“开”或“关”。"
ALWAYS_REPLY_USAGE = (
    "用法：/必回名单｜/必回名单 添加 @群友 123456｜"
    "/必回名单 删除 @群友｜/必回名单 清空"
)
ALWAYS_REPLY_ADD = {"添加", "加", "add"}
ALWAYS_REPLY_REMOVE = {"删除", "移除", "删", "remove", "del"}
ALWAYS_REPLY_CLEAR = {"清空", "clear"}
CANCEL_WORDS = {"取消", "退出", "cancel", "q"}
QQ_PATTERN = re.compile(r"\d{5,12}")
PRIVATE_DRAFT_TTL = 300.0
PRIVATE_GROUP_CHOICE_LIMIT = 50
CANCEL_NOTICE = "已取消必回名单操作。"
NO_KNOWN_GROUP = "还没有任何群的聊天记录，先在群里用一次机器人再试。"
AFFECTION_SWITCH_ON = {"开", "开启", "on", "1", "true"}
AFFECTION_SWITCH_OFF = {"关", "关闭", "off", "0", "false"}
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
meme_provider: MemeProvider


def _refresh_meme_provider() -> None:
    global meme_provider
    meme_provider = MemeProvider(
        enabled=config.group_chat_memory_meme_enabled,
        api_url=config.group_chat_memory_meme_api_url,
        api_id=config.group_chat_memory_apihz_id,
        api_key=config.group_chat_memory_apihz_key.get_secret_value(),
    )


_refresh_meme_provider()
_bots: dict[int, Bot] = {}


async def startup() -> None:
    await store.open()
    config.apply_runtime_config(await store.get_runtime_config())
    _refresh_meme_provider()
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


async def _outgoing_text(
    group_id: int, text: str, message_id: int | None
) -> Message | MessageSegment:
    """按当前群引用概率构造回复：命中且知道触发消息时附带引用。"""
    if message_id is not None and await coordinator.should_quote(group_id):
        return Message([MessageSegment.reply(message_id), MessageSegment.text(text)])
    return MessageSegment.text(text)


async def reply(group_id: int, message: MemoryMessage) -> None:
    bot = _bots[group_id]
    message_id = message.message_id
    try:
        context = await context_builder.build(group_id, message.text)
        response = await chat_client.complete(
            context, message.text, session_id=f"group:{group_id}"
        )
    except Exception as exc:
        status = f" HTTP {exc.status_code}" if isinstance(exc, ChatAPIError) else ""
        logger.error("群 %s 的聊天回复失败：%s%s", group_id, type(exc).__name__, status)
        await bot.send_group_msg(
            group_id=group_id,
            message=await _outgoing_text(group_id, GENERIC_REPLY_ERROR, message_id),
        )
        return

    outgoing = await _outgoing_text(group_id, response, message_id)
    await bot.send_group_msg(group_id=group_id, message=outgoing)
    if message.attach_meme:
        try:
            image_url = await meme_provider.fetch(response)
            if image_url:
                await bot.send_group_msg(group_id=group_id, message=MessageSegment.image(image_url))
        except Exception as exc:
            logger.warning("群 %s 的表情包发送失败：%s", group_id, type(exc).__name__)
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


@dataclass
class _PrivateDraft:
    """私聊里“选群 → 填 QQ”两步交互的临时状态。"""

    step: str
    expires_at: float
    groups: tuple[int, ...] = ()
    group_id: int | None = None


_private_drafts: dict[int, _PrivateDraft] = {}


def _private_message(event: MessageEvent) -> bool:
    return isinstance(event, PrivateMessageEvent)


def _looks_like_command(text: str) -> bool:
    starts = get_driver().config.command_start or {"/"}
    return any(start and text.startswith(start) for start in starts)


def _pending_private_draft(event: MessageEvent) -> bool:
    if not isinstance(event, PrivateMessageEvent):
        return False
    draft = _private_drafts.get(event.user_id)
    if draft is None:
        return False
    if draft.expires_at <= time.monotonic():
        _private_drafts.pop(event.user_id, None)
        return False
    return not _looks_like_command(event.get_plaintext().strip())


async def _group_names(bot: Bot) -> dict[int, str]:
    """尽力取群名；取不到时返回空表，列表里只显示群号。"""
    try:
        groups = await bot.get_group_list()
    except Exception as exc:
        logger.warning("读取群列表失败：%s", type(exc).__name__)
        return {}
    names: dict[int, str] = {}
    for item in groups:
        try:
            names[int(item["group_id"])] = str(item.get("group_name") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return names


def _group_choice_list(group_ids: tuple[int, ...], names: dict[int, str]) -> str:
    lines = ["请选择要设置必回名单的群："]
    for index, group_id in enumerate(group_ids, start=1):
        name = names.get(group_id)
        lines.append(
            f"{index}. {name}（{group_id}）" if name else f"{index}. {group_id}"
        )
    lines.append("回复序号或群号进行选择，回复“取消”退出。")
    return "\n".join(lines)


def _member_prompt(group_id: int) -> str:
    return (
        f"已选择群 {group_id}。\n"
        "请发送要加入必回名单的 QQ 号，多个用空格、逗号或换行分隔，回复“取消”退出。"
    )


def _select_group(draft: _PrivateDraft, text: str) -> int | None:
    if not text.isdigit():
        return None
    value = int(text)
    if value in draft.groups:
        return value
    if 1 <= value <= len(draft.groups):
        return draft.groups[value - 1]
    return None


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
            "apihz_id": config.group_chat_memory_apihz_id,
            "apihz_key": config.group_chat_memory_apihz_key.get_secret_value(),
        }
        values[key] = value
        await store.set_runtime_config(key, value)
        config.apply_runtime_config(values)
        if key in {"apihz_id", "apihz_key"}:
            _refresh_meme_provider()


chat_url_matcher = on_command("设置聊天地址", priority=-1001, block=True)
chat_key_matcher = on_command("设置聊天密钥", priority=-1001, block=True)
chat_model_matcher = on_command("设置聊天模型", priority=-1001, block=True)
embedding_key_matcher = on_command("设置向量密钥", priority=-1001, block=True)
meme_id_matcher = on_command("设置表情包ID", priority=-1001, block=True)
meme_key_matcher = on_command("设置表情包密钥", priority=-1001, block=True)
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
    if key == "apihz_id" and not value.isdigit():
        await matcher.finish("接口 ID 必须是数字。")
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


@meme_id_matcher.handle()
async def handle_meme_id(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(meme_id_matcher, bot, event, args, "apihz_id")


@meme_key_matcher.handle()
async def handle_meme_key(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(meme_key_matcher, bot, event, args, "apihz_key")


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
        f"{status(config.group_chat_memory_embedding_api_key.get_secret_value())}\n"
        f"接口 ID：{config.group_chat_memory_apihz_id or '未配置'}\n"
        f"接口密钥：{status(config.group_chat_memory_apihz_key.get_secret_value())}"
    )


probability_matcher = on_command(
    "回复概率", rule=_eligible_group_message, priority=-1001, block=True
)
quote_probability_matcher = on_command(
    "引用概率", rule=_eligible_group_message, priority=-1001, block=True
)
clear_matcher = on_command(
    "清空群记忆", rule=_eligible_group_message, priority=-1001, block=True
)
hourly_limit_matcher = on_command(
    "每小时次数", rule=_eligible_group_message, priority=-1001, block=True
)
affection_probability_matcher = on_command(
    "好感概率", rule=_eligible_group_message, priority=-1001, block=True
)
affection_switch_matcher = on_command(
    "好感开关", rule=_eligible_group_message, priority=-1001, block=True
)
superuser_reply_matcher = on_command(
    "超管必回", rule=_eligible_group_message, priority=-1001, block=True
)
always_reply_matcher = on_command(
    "必回名单", rule=_eligible_group_message, priority=-1001, block=True
)
private_always_reply_matcher = on_command(
    "必回名单", rule=_private_message, priority=-1001, block=True
)
private_draft_matcher = on_message(
    rule=_pending_private_draft, priority=-1001, block=True
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


@quote_probability_matcher.handle()
async def handle_quote_probability(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await quote_probability_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if not text:
        probability = await store.get_quote_probability(event.group_id)
        await quote_probability_matcher.finish(f"当前群引用回复概率：{probability:.1f}%")
    if PROBABILITY_PATTERN.fullmatch(text) is None:
        await quote_probability_matcher.finish(INVALID_QUOTE_PROBABILITY)

    probability = float(text)
    await store.set_quote_probability(event.group_id, probability)
    await quote_probability_matcher.finish(f"已设置当前群引用回复概率：{probability:.1f}%")


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


@affection_probability_matcher.handle()
async def handle_affection_probability(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await affection_probability_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if not text:
        probability = await store.get_affection_probability(event.group_id)
        await affection_probability_matcher.finish(
            f"当前群好感榜触发概率：{probability:.1f}%"
        )
    if PROBABILITY_PATTERN.fullmatch(text) is None:
        await affection_probability_matcher.finish(INVALID_AFFECTION_PROBABILITY)

    probability = float(text)
    await store.set_affection_probability(event.group_id, probability)
    await affection_probability_matcher.finish(
        f"已设置当前群好感榜触发概率：{probability:.1f}%"
    )


@affection_switch_matcher.handle()
async def handle_affection_switch(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await affection_switch_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip().lower()
    if not text:
        enabled = await store.get_affection_enabled(event.group_id)
        await affection_switch_matcher.finish(
            f"当前群好感榜触发：{'已开启' if enabled else '已关闭'}"
        )
    if text in AFFECTION_SWITCH_ON:
        await store.set_affection_enabled(event.group_id, True)
        await affection_switch_matcher.finish("已开启当前群好感榜触发。")
    if text in AFFECTION_SWITCH_OFF:
        await store.set_affection_enabled(event.group_id, False)
        await affection_switch_matcher.finish("已关闭当前群好感榜触发。")
    await affection_switch_matcher.finish(INVALID_AFFECTION_SWITCH)


@superuser_reply_matcher.handle()
async def handle_superuser_reply_switch(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await superuser_reply_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip().lower()
    if not text:
        enabled = await store.get_superuser_always_reply(event.group_id)
        await superuser_reply_matcher.finish(
            f"当前群 SUPERUSER 必定触发：{'已开启' if enabled else '已关闭'}"
        )
    if text in AFFECTION_SWITCH_ON:
        await store.set_superuser_always_reply(event.group_id, True)
        await superuser_reply_matcher.finish("已开启当前群 SUPERUSER 必定触发。")
    if text in AFFECTION_SWITCH_OFF:
        await store.set_superuser_always_reply(event.group_id, False)
        await superuser_reply_matcher.finish("已关闭当前群 SUPERUSER 必定触发。")
    await superuser_reply_matcher.finish(INVALID_SUPERUSER_REPLY_SWITCH)


def _always_reply_targets(args: Message) -> list[int]:
    """从命令参数中提取目标 QQ：支持 @ 群友和直接写号码。"""
    targets: list[int] = []
    for segment in args:
        if segment.type == "at":
            qq = str(segment.data.get("qq", ""))
            if qq.isdigit() and int(qq) not in targets:
                targets.append(int(qq))
        elif segment.type == "text":
            for token in str(segment.data.get("text", "")).split():
                if token.isdigit() and int(token) not in targets:
                    targets.append(int(token))
    return targets


@always_reply_matcher.handle()
async def handle_always_reply(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await always_reply_matcher.finish(PERMISSION_DENIED)

    parts = args.extract_plain_text().strip().split()
    action = parts[0] if parts else ""
    if not action:
        members = await store.list_always_reply(event.group_id)
        if not members:
            await always_reply_matcher.finish("本群必回名单为空。")
        lines = [f"本群必回名单（{len(members)} 人）："]
        lines.extend(
            f"{user_id}（{name}）" if name else str(user_id) for user_id, name in members
        )
        await always_reply_matcher.finish("\n".join(lines))

    targets = _always_reply_targets(args)
    if action in ALWAYS_REPLY_CLEAR:
        removed = await store.clear_always_reply(event.group_id)
        if removed == 0:
            await always_reply_matcher.finish("本群必回名单本来就是空的。")
        await always_reply_matcher.finish(f"已清空本群必回名单（{removed} 人）。")

    if action in ALWAYS_REPLY_ADD:
        if not targets:
            await always_reply_matcher.finish(ALWAYS_REPLY_USAGE)
        added = 0
        for user_id in targets:
            if await store.add_always_reply(event.group_id, user_id):
                added += 1
        skipped = len(targets) - added
        if added == 0:
            await always_reply_matcher.finish("这些群友已经在必回名单里了。")
        suffix = f"，{skipped} 人原本已在名单中" if skipped else ""
        await always_reply_matcher.finish(f"已加入必回名单：{added} 人{suffix}。")

    if action in ALWAYS_REPLY_REMOVE:
        if not targets:
            await always_reply_matcher.finish(ALWAYS_REPLY_USAGE)
        removed = 0
        for user_id in targets:
            if await store.remove_always_reply(event.group_id, user_id):
                removed += 1
        if removed == 0:
            await always_reply_matcher.finish("这些群友本来就不在必回名单里。")
        skipped = len(targets) - removed
        suffix = f"，{skipped} 人原本就不在名单中" if skipped else ""
        await always_reply_matcher.finish(f"已移出必回名单：{removed} 人{suffix}。")

    await always_reply_matcher.finish(ALWAYS_REPLY_USAGE)


@private_always_reply_matcher.handle()
async def handle_private_always_reply(
    bot: Bot,
    event: PrivateMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await private_always_reply_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if text.lower() in CANCEL_WORDS:
        _private_drafts.pop(event.user_id, None)
        await private_always_reply_matcher.finish(CANCEL_NOTICE)

    groups = await store.list_groups()
    if not groups:
        await private_always_reply_matcher.finish(NO_KNOWN_GROUP)

    group_ids = tuple(group_id for group_id, _, _ in groups)[
        :PRIVATE_GROUP_CHOICE_LIMIT
    ]
    _private_drafts[event.user_id] = _PrivateDraft(
        step="group",
        expires_at=time.monotonic() + PRIVATE_DRAFT_TTL,
        groups=group_ids,
    )
    names = await _group_names(bot)
    await private_always_reply_matcher.finish(_group_choice_list(group_ids, names))


@private_draft_matcher.handle()
async def handle_private_draft(bot: Bot, event: PrivateMessageEvent) -> None:
    draft = _private_drafts.get(event.user_id)
    if draft is None:
        await private_draft_matcher.finish()
    if not await SUPERUSER(bot, event):
        _private_drafts.pop(event.user_id, None)
        await private_draft_matcher.finish(PERMISSION_DENIED)

    text = event.get_plaintext().strip()
    if text.lower() in CANCEL_WORDS:
        _private_drafts.pop(event.user_id, None)
        await private_draft_matcher.finish(CANCEL_NOTICE)

    if draft.step == "group":
        group_id = _select_group(draft, text)
        if group_id is None:
            await private_draft_matcher.finish(
                "没有匹配的群，请重新回复序号或群号，回复“取消”退出。"
            )
        draft.group_id = group_id
        draft.step = "member"
        draft.expires_at = time.monotonic() + PRIVATE_DRAFT_TTL
        await private_draft_matcher.finish(_member_prompt(group_id))

    targets = list(dict.fromkeys(int(qq) for qq in QQ_PATTERN.findall(text)))
    if not targets:
        await private_draft_matcher.finish(
            "没识别到 QQ 号，请重新发送，回复“取消”退出。"
        )

    group_id = draft.group_id
    if group_id is None:
        _private_drafts.pop(event.user_id, None)
        await private_draft_matcher.finish(CANCEL_NOTICE)

    added = 0
    for user_id in targets:
        if await store.add_always_reply(group_id, user_id):
            added += 1
    _private_drafts.pop(event.user_id, None)
    skipped = len(targets) - added
    suffix = f"，{skipped} 人原本已在名单中" if skipped else ""
    await private_draft_matcher.finish(
        f"已加入群 {group_id} 的必回名单：{added} 人{suffix}。"
    )


async def _affection_probability(event: GroupMessageEvent) -> float:
    """好感榜成员的最低触发概率；不适用时返回 0。"""
    if not await store.get_affection_enabled(event.group_id):
        return 0.0
    probability = await store.get_affection_probability(event.group_id)
    if probability <= 0:
        return 0.0
    if not await affection_leaderboard.is_member(event.group_id, event.user_id):
        return 0.0
    return probability


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
        message_id=event.message_id,
    )
    _bots[event.group_id] = bot
    await writer.record(memory_message)
    if directed:
        if await store.consume_request_slot(event.group_id):
            coordinator.submit(event.group_id, memory_message, explicit=True)
        else:
            await bot.send_group_msg(
                group_id=event.group_id,
                message=await _outgoing_text(
                    event.group_id, RATE_LIMIT_NOTICE, event.message_id
                ),
            )
        raise StopPropagation
    if event.reply is not None:
        return
    if any(segment.type != "text" for segment in event.message):
        return

    if await store.is_always_reply(event.group_id, event.user_id):
        coordinator.submit(event.group_id, memory_message, explicit=False)
        return

    if await SUPERUSER(bot, event) and await store.get_superuser_always_reply(
        event.group_id
    ):
        memory_message.attach_meme = True
        coordinator.submit(event.group_id, memory_message, explicit=False)
        return

    affection_probability = await _affection_probability(event)
    if not await coordinator.should_random_reply(
        event.group_id, minimum_probability=affection_probability
    ):
        return
    if affection_probability <= 0 and not await store.consume_request_slot(
        event.group_id
    ):
        return
    if affection_probability > 0 and await affection_leaderboard.is_top_member(
        event.group_id, event.user_id
    ):
        memory_message.attach_meme = True
    coordinator.submit(event.group_id, memory_message, explicit=False)


driver = get_driver()
driver.on_startup(startup)
driver.on_shutdown(shutdown)
