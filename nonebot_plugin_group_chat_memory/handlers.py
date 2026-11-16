from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
from .config import (
    DEFAULT_TTS_VOICE_ID,
    TTS_PROVIDERS,
    TTS_VOICES,
    get_settings,
)
from .context import ContextBuilder
from .meme import MemeProvider
from .memory import MemoryWriter
from .models import Message as MemoryMessage
from .replies import ReplyCoordinator
from .store import MemoryStore
from .tts import QwenTTSClient, TTSClient

logger = logging.getLogger(__name__)

GENERIC_REPLY_ERROR = "暂时无法生成回复，请稍后再试。"
PERMISSION_DENIED = "仅 SUPERUSER 可执行此命令。"
INVALID_PROBABILITY = "概率需为 0～100 的数字，最多一位小数。"
INVALID_HOURLY_LIMIT = "每小时请求上限必须是 1 到 10000 的整数。"
INVALID_AFFECTION_PROBABILITY = "好感榜触发概率需为 0～100 的数字，最多一位小数。"
INVALID_AFFECTION_SWITCH = "开关参数只能是“开”或“关”。"
INVALID_SUPERUSER_REPLY_SWITCH = "开关参数只能是“开”或“关”。"
INVALID_MEME_SWITCH = "开关参数只能是“开”或“关”。"
MEME_GLOBALLY_DISABLED = (
    "全局表情包已被 GROUP_CHAT_MEMORY_MEME_ENABLED=false 关闭，先在 .env 中打开。"
)
ALWAYS_REPLY_USAGE = (
    "用法：/必回名单｜/必回名单 添加 @群友 123456｜"
    "/必回名单 删除 @群友｜/必回名单 清空"
)
VOICE_LIST_USAGE = (
    "用法：/语音名单｜/语音名单 添加 @群友 123456｜"
    "/语音名单 删除 @群友｜/语音名单 清空"
)
VOICE_USAGE = "请引用一条文字消息，再发送 /语音。"
VOICE_NOT_CONFIGURED = "语音合成未配置，请 SUPERUSER 私聊发送 /设置语音密钥。"
QWEN_VOICE_NOT_CONFIGURED = (
    "Qwen 语音未配置完整，请私聊 SUPERUSER 设置"
    " /设置阿里云密钥、/设置阿里云空间 和 /设置语音参考。"
)
INVALID_VOICE_ID = "语音音色只能是 stella（女声）或 tide（男声）。"
INVALID_VOICE_MODEL = "语音模型只能是 deepseek 或 qwen。"
INVALID_QWEN_WORKSPACE = "阿里云百炼业务空间 ID 只能包含字母、数字、下划线和短横线。"
INVALID_QWEN_REFERENCE = (
    "参考音频需为 http(s) URL，或服务器上的 WAV/MP3/OGG 文件路径，且不超过 10 MB。"
)
QWEN_REFERENCE_MAX_BYTES = 10 * 1024 * 1024
VOICE_PROVIDER_ALIASES = {
    "deepseek": "deepseek",
    "deepseek-tts": "deepseek",
    "deepseek tts": "deepseek",
    "qwen": "qwen",
    "qwen-audio": "qwen",
    "qwen-audio-3.1-tts-next": "qwen",
    "阿里云": "qwen",
    "阿里": "qwen",
    "通义": "qwen",
}
QWEN_REFERENCE_SUFFIXES = {".wav", ".mp3", ".ogg", ".opus"}
MANTOU_VOICE_FILENAME = "mantou_voice.wav"
MANTOU_VOICE_DESCRIPTION = (
    "一位偏萝莉的年轻少女音，音色清亮、软萌、活泼，语速稍快，"
    "像群聊里随口聊天，不要播音腔，不要气声，不要背景音乐和音效。"
)
MANTOU_VOICE_SAMPLE = (
    "确实，今天这局打得有点抽象。刚才那波我明明想撤，结果手一抖又冲上去了。"
    "没事，输了就输了，明天再战。你们也别笑，换你们上也一样。"
    "行了，先吃饭，吃完再说。"
)
VOICE_SAMPLE_NOT_CONFIGURED = (
    "生成馒头音色需要先配置千问 API Key：私聊发送 /设置阿里云密钥。"
)
VOICE_SAMPLE_FAILED = "馒头音色生成失败，请稍后重试或检查阿里云配置。"
VOICE_EXPORT_MISSING = "还没有生成馒头音色，请先私聊发送 /生成馒头音色。"
VOICE_ALIASES = {
    "stella": "stella",
    "女声": "stella",
    "女": "stella",
    "tide": "tide",
    "男声": "tide",
    "男": "tide",
}
ALWAYS_REPLY_ADD = {"添加", "加", "add"}
ALWAYS_REPLY_REMOVE = {"删除", "移除", "删", "remove", "del"}
ALWAYS_REPLY_CLEAR = {"清空", "clear"}
CANCEL_WORDS = {"取消", "退出", "cancel", "q"}
QQ_PATTERN = re.compile(r"\d{5,12}")
PRIVATE_DRAFT_TTL = 300.0
PRIVATE_GROUP_CHOICE_LIMIT = 50
CANCEL_NOTICE = "已取消必回名单操作。"
VOICE_CANCEL_NOTICE = "已取消语音名单操作。"
NO_KNOWN_GROUP = "还没有任何群的聊天记录，先在群里用一次机器人再试。"
AFFECTION_SWITCH_ON = {"开", "开启", "on", "1", "true"}
AFFECTION_SWITCH_OFF = {"关", "关闭", "off", "0", "false"}
RATE_LIMIT_NOTICE = "本群本小时请求次数已用完，请稍后再来。"
PROBABILITY_PATTERN = re.compile(r"(?:100(?:\.0)?|\d{1,2}(?:\.\d)?)")
SECRET_COMMAND_LOG_PATTERN = re.compile(
    r"(设置聊天密钥|设置向量密钥|设置语音密钥|设置阿里云密钥)\s+[^\r\n]*"
)


def _redact_secret_command_logs(record) -> None:
    record["message"] = SECRET_COMMAND_LOG_PATTERN.sub(r"\1 ****", record["message"])


nonebot_logger.configure(patcher=_redact_secret_command_logs)

config = get_settings()
store = MemoryStore(config)
embedding_client = EmbeddingClient(config)
writer = MemoryWriter(config, store, embedding_client)
context_builder = ContextBuilder(config, store, embedding_client)
chat_client = ChatClient(config)
tts_client = TTSClient(config)
qwen_tts_client = QwenTTSClient(config)
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


def _record_segment(audio: bytes) -> MessageSegment:
    """把 WAV 音频编码成 OneBot 语音消息段。"""
    encoded = base64.b64encode(audio).decode()
    return MessageSegment.record(f"base64://{encoded}")


def _voice_label(voice_id: str) -> str:
    return TTS_VOICES.get(voice_id.strip().lower(), TTS_VOICES[DEFAULT_TTS_VOICE_ID])


def _voice_provider() -> str:
    provider = config.group_chat_memory_tts_provider.strip().lower()
    return provider if provider in TTS_PROVIDERS else "deepseek"


def _provider_label(provider: str) -> str:
    return TTS_PROVIDERS.get(provider, TTS_PROVIDERS["deepseek"])


def _voice_configured() -> bool:
    if _voice_provider() == "qwen":
        return bool(
            config.group_chat_memory_qwen_api_key.get_secret_value().strip()
            and config.group_chat_memory_qwen_reference.strip()
        )
    return bool(config.group_chat_memory_tts_api_key.get_secret_value().strip())


def _voice_not_configured_notice() -> str:
    if _voice_provider() == "qwen":
        return QWEN_VOICE_NOT_CONFIGURED
    return VOICE_NOT_CONFIGURED


def _reference_label(reference: str) -> str:
    value = reference.strip()
    if not value:
        return "未配置"
    if value.lower().startswith(("http://", "https://")):
        return "已配置（URL）"
    return f"已配置（{Path(value).name}）"


async def _record_bot_reply(bot: Bot, group_id: int, text: str) -> None:
    await writer.record(
        MemoryMessage(
            group_id=group_id,
            user_id=int(bot.self_id),
            display_name="Bot",
            text=text,
            is_bot=True,
        )
    )


async def _voice_available(group_id: int, user_id: int) -> bool:
    """语音只对榜一和语音名单成员开放；当前模型未配置完整时视为不可用。"""
    if not _voice_configured():
        return False
    if await store.is_voice_member(group_id, user_id):
        return True
    return await affection_leaderboard.is_top_member(group_id, user_id, limit=1)


async def _synthesize(text: str) -> bytes | None:
    provider = _voice_provider()
    try:
        if provider == "qwen":
            return await qwen_tts_client.synthesize(text)
        return await tts_client.synthesize(
            text, voice_id=config.group_chat_memory_tts_voice_id
        )
    except Exception as exc:
        logger.warning("语音合成失败（%s）：%s", provider, type(exc).__name__)
        return None


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
        await bot.send_group_msg(
            group_id=group_id, message=MessageSegment.text(GENERIC_REPLY_ERROR)
        )
        return

    mode = coordinator.choose_reply_mode(
        voice_available=await _voice_available(group_id, message.user_id)
    )
    if mode == "voice":
        audio = await _synthesize(response)
        if audio is not None:
            try:
                await bot.send_group_msg(
                    group_id=group_id, message=_record_segment(audio)
                )
            except Exception as exc:
                logger.warning(
                    "群 %s 的语音发送失败：%s", group_id, type(exc).__name__
                )
            else:
                await _record_bot_reply(bot, group_id, response)
                return
        mode = "text"

    if mode == "quote" and message.message_id is not None:
        outgoing: Message | MessageSegment = Message(
            [MessageSegment.reply(message.message_id), MessageSegment.text(response)]
        )
    else:
        outgoing = MessageSegment.text(response)
    await bot.send_group_msg(group_id=group_id, message=outgoing)
    if message.attach_meme and await store.get_meme_enabled(group_id):
        try:
            image_url = await meme_provider.fetch(response)
            if image_url:
                await bot.send_group_msg(group_id=group_id, message=MessageSegment.image(image_url))
        except Exception as exc:
            logger.warning("群 %s 的表情包发送失败：%s", group_id, type(exc).__name__)
    await _record_bot_reply(bot, group_id, response)


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
    kind: str = "always_reply"


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


def _group_choice_list(
    group_ids: tuple[int, ...], names: dict[int, str], *, label: str = "必回名单"
) -> str:
    lines = [f"请选择要设置{label}的群："]
    for index, group_id in enumerate(group_ids, start=1):
        name = names.get(group_id)
        lines.append(
            f"{index}. {name}（{group_id}）" if name else f"{index}. {group_id}"
        )
    lines.append("回复序号或群号进行选择，回复“取消”退出。")
    return "\n".join(lines)


def _member_prompt(group_id: int, *, label: str = "必回名单") -> str:
    return (
        f"已选择群 {group_id}。\n"
        f"请发送要加入{label}的 QQ 号，多个用空格、逗号或换行分隔，回复“取消”退出。"
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
            "tts_provider": config.group_chat_memory_tts_provider,
            "tts_api_key": config.group_chat_memory_tts_api_key.get_secret_value(),
            "tts_voice_id": config.group_chat_memory_tts_voice_id,
            "qwen_api_key": config.group_chat_memory_qwen_api_key.get_secret_value(),
            "qwen_workspace_id": config.group_chat_memory_qwen_workspace_id,
            "qwen_reference": config.group_chat_memory_qwen_reference,
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
voice_key_matcher = on_command("设置语音密钥", priority=-1001, block=True)
voice_id_matcher = on_command("设置语音音色", priority=-1001, block=True)
voice_model_matcher = on_command("设置语音模型", priority=-1001, block=True)
qwen_key_matcher = on_command("设置阿里云密钥", priority=-1001, block=True)
qwen_workspace_matcher = on_command("设置阿里云空间", priority=-1001, block=True)
qwen_reference_matcher = on_command("设置语音参考", priority=-1001, block=True)
mantou_voice_matcher = on_command("生成馒头音色", priority=-1001, block=True)
export_mantou_voice_matcher = on_command("导出馒头音色", priority=-1001, block=True)
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


@voice_key_matcher.handle()
async def handle_voice_key(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(voice_key_matcher, bot, event, args, "tts_api_key")


@voice_id_matcher.handle()
async def handle_voice_id(
    bot: Bot,
    event: MessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await voice_id_matcher.finish("请在私聊中设置语音音色。")
    if not await SUPERUSER(bot, event):
        await voice_id_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip().lower()
    if not text:
        await voice_id_matcher.finish(
            f"当前语音音色：{_voice_label(config.group_chat_memory_tts_voice_id)}"
        )
    voice_id = VOICE_ALIASES.get(text)
    if voice_id is None:
        await voice_id_matcher.finish(INVALID_VOICE_ID)
    try:
        await apply_setting("tts_voice_id", voice_id)
    except Exception as exc:
        logger.error("保存语音音色失败：%s", type(exc).__name__)
        await voice_id_matcher.finish("配置保存失败，请稍后再试。")
    await voice_id_matcher.finish(f"语音音色已设置为：{_voice_label(voice_id)}。")


@voice_model_matcher.handle()
async def handle_voice_model(
    bot: Bot,
    event: MessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await voice_model_matcher.finish("请在私聊中设置语音模型。")
    if not await SUPERUSER(bot, event):
        await voice_model_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip().lower()
    if not text:
        await voice_model_matcher.finish(
            f"当前语音模型：{_provider_label(_voice_provider())}"
        )
    provider = VOICE_PROVIDER_ALIASES.get(text)
    if provider is None:
        await voice_model_matcher.finish(INVALID_VOICE_MODEL)
    try:
        await apply_setting("tts_provider", provider)
    except Exception as exc:
        logger.error("保存语音模型失败：%s", type(exc).__name__)
        await voice_model_matcher.finish("配置保存失败，请稍后再试。")
    await voice_model_matcher.finish(
        f"语音模型已切换为：{_provider_label(provider)}。"
    )


@qwen_key_matcher.handle()
async def handle_qwen_key(
    bot: Bot, event: MessageEvent, args: Annotated[Message, CommandArg()]
) -> None:
    await _handle_setting(qwen_key_matcher, bot, event, args, "qwen_api_key")


@qwen_workspace_matcher.handle()
async def handle_qwen_workspace(
    bot: Bot,
    event: MessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await qwen_workspace_matcher.finish("请在私聊中设置阿里云空间。")
    if not await SUPERUSER(bot, event):
        await qwen_workspace_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if not text:
        await qwen_workspace_matcher.finish(
            f"当前阿里云空间：{config.group_chat_memory_qwen_workspace_id or '未配置'}"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        await qwen_workspace_matcher.finish(INVALID_QWEN_WORKSPACE)
    try:
        await apply_setting("qwen_workspace_id", text)
    except Exception as exc:
        logger.error("保存阿里云空间失败：%s", type(exc).__name__)
        await qwen_workspace_matcher.finish("配置保存失败，请稍后再试。")
    await qwen_workspace_matcher.finish("阿里云百炼业务空间已更新。")


@qwen_reference_matcher.handle()
async def handle_qwen_reference(
    bot: Bot,
    event: MessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await qwen_reference_matcher.finish("请在私聊中设置语音参考。")
    if not await SUPERUSER(bot, event):
        await qwen_reference_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if not text:
        await qwen_reference_matcher.finish(
            f"当前语音参考：{_reference_label(config.group_chat_memory_qwen_reference)}"
        )
    if text.lower() in {"清除", "清空", "none", "off", "reset"}:
        try:
            await apply_setting("qwen_reference", "")
        except Exception as exc:
            logger.error("清空语音参考失败：%s", type(exc).__name__)
            await qwen_reference_matcher.finish("配置保存失败，请稍后再试。")
        await qwen_reference_matcher.finish("语音参考已清除。")

    value = text
    if value.lower().startswith(("http://", "https://")):
        if not _valid_http_url(value):
            await qwen_reference_matcher.finish(INVALID_QWEN_REFERENCE)
    else:
        path = Path(value).expanduser()
        if (
            not path.is_file()
            or path.suffix.lower() not in QWEN_REFERENCE_SUFFIXES
            or path.stat().st_size > QWEN_REFERENCE_MAX_BYTES
        ):
            await qwen_reference_matcher.finish(INVALID_QWEN_REFERENCE)
        value = str(path.resolve())

    try:
        await apply_setting("qwen_reference", value)
    except Exception as exc:
        logger.error("保存语音参考失败：%s", type(exc).__name__)
        await qwen_reference_matcher.finish("配置保存失败，请稍后再试。")
    await qwen_reference_matcher.finish("语音参考已更新。")


@mantou_voice_matcher.handle()
async def handle_mantou_voice(bot: Bot, event: MessageEvent) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await mantou_voice_matcher.finish("请在私聊中生成馒头音色。")
    if not await SUPERUSER(bot, event):
        await mantou_voice_matcher.finish(PERMISSION_DENIED)
    if not config.group_chat_memory_qwen_api_key.get_secret_value().strip():
        await mantou_voice_matcher.finish(VOICE_SAMPLE_NOT_CONFIGURED)

    try:
        audio = await qwen_tts_client.generate_sample(
            MANTOU_VOICE_SAMPLE, description=MANTOU_VOICE_DESCRIPTION
        )
    except Exception as exc:
        logger.warning("生成馒头音色失败：%s", type(exc).__name__)
        await mantou_voice_matcher.finish(VOICE_SAMPLE_FAILED)

    if len(audio) > QWEN_REFERENCE_MAX_BYTES:
        await mantou_voice_matcher.finish("生成的馒头音色文件过大，请重试。")

    path = config.database_path.parent / MANTOU_VOICE_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        await apply_setting("qwen_reference", str(path))
        await apply_setting("tts_provider", "qwen")
    except Exception as exc:
        logger.error("保存馒头音色失败：%s", type(exc).__name__)
        await mantou_voice_matcher.finish("馒头音色保存失败，请稍后再试。")

    await mantou_voice_matcher.finish(
        "馒头固定音色已生成并保存，已切换为 Qwen 语音。"
        "之后 /语音 和自动语音都会使用这个偏萝莉的少女音。"
    )


@export_mantou_voice_matcher.handle()
async def handle_export_mantou_voice(bot: Bot, event: MessageEvent) -> None:
    if not isinstance(event, PrivateMessageEvent):
        await export_mantou_voice_matcher.finish("请在私聊中导出馒头音色。")
    if not await SUPERUSER(bot, event):
        await export_mantou_voice_matcher.finish(PERMISSION_DENIED)

    path = config.database_path.parent / MANTOU_VOICE_FILENAME
    if not path.is_file():
        await export_mantou_voice_matcher.finish(VOICE_EXPORT_MISSING)

    audio = path.read_bytes()
    encoded = base64.b64encode(audio).decode()
    try:
        await bot.call_api(
            "upload_private_file",
            user_id=event.user_id,
            file=f"base64://{encoded}",
            name=MANTOU_VOICE_FILENAME,
        )
    except Exception as exc:
        logger.warning("导出馒头音色文件失败：%s", type(exc).__name__)
        try:
            await bot.send_private_msg(
                user_id=event.user_id,
                message=MessageSegment.record(f"base64://{encoded}"),
            )
        except Exception as voice_exc:
            logger.warning("导出馒头音色语音失败：%s", type(voice_exc).__name__)
            await export_mantou_voice_matcher.finish("馒头音色导出失败，请稍后再试。")
    await export_mantou_voice_matcher.finish(
        f"馒头音色已导出：{MANTOU_VOICE_FILENAME}（{len(audio)} 字节）"
    )


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
        f"语音模型：{_provider_label(_voice_provider())}\n"
        f"语音密钥：{status(config.group_chat_memory_tts_api_key.get_secret_value())}\n"
        f"语音音色：{_voice_label(config.group_chat_memory_tts_voice_id)}\n"
        "阿里云密钥："
        f"{status(config.group_chat_memory_qwen_api_key.get_secret_value())}\n"
        f"阿里云空间：{config.group_chat_memory_qwen_workspace_id or '未配置（走千问AI平台）'}\n"
        f"语音参考：{_reference_label(config.group_chat_memory_qwen_reference)}\n"
        f"接口 ID：{config.group_chat_memory_apihz_id or '未配置'}\n"
        f"接口密钥：{status(config.group_chat_memory_apihz_key.get_secret_value())}"
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
affection_probability_matcher = on_command(
    "好感概率", rule=_eligible_group_message, priority=-1001, block=True
)
affection_switch_matcher = on_command(
    "好感开关", rule=_eligible_group_message, priority=-1001, block=True
)
meme_switch_matcher = on_command(
    "表情包开关", rule=_eligible_group_message, priority=-1001, block=True
)
superuser_reply_matcher = on_command(
    "超管必回", rule=_eligible_group_message, priority=-1001, block=True
)
always_reply_matcher = on_command(
    "必回名单", rule=_eligible_group_message, priority=-1001, block=True
)
voice_list_matcher = on_command(
    "语音名单", rule=_eligible_group_message, priority=-1001, block=True
)
voice_matcher = on_command(
    "语音", rule=_eligible_group_message, priority=-1001, block=True
)
private_always_reply_matcher = on_command(
    "必回名单", rule=_private_message, priority=-1001, block=True
)
private_voice_list_matcher = on_command(
    "语音名单", rule=_private_message, priority=-1001, block=True
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


@meme_switch_matcher.handle()
async def handle_meme_switch(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await meme_switch_matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip().lower()
    if not text:
        enabled = await store.get_meme_enabled(event.group_id)
        await meme_switch_matcher.finish(
            f"当前群表情包：{'已开启' if enabled else '已关闭'}"
        )
    if text in AFFECTION_SWITCH_ON:
        if not config.group_chat_memory_meme_enabled:
            await meme_switch_matcher.finish(MEME_GLOBALLY_DISABLED)
        await store.set_meme_enabled(event.group_id, True)
        await meme_switch_matcher.finish("已开启当前群表情包。")
    if text in AFFECTION_SWITCH_OFF:
        await store.set_meme_enabled(event.group_id, False)
        await meme_switch_matcher.finish("已关闭当前群表情包。")
    await meme_switch_matcher.finish(INVALID_MEME_SWITCH)


def _target_user_ids(args: Message) -> list[int]:
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


async def _handle_member_list(
    matcher, event: GroupMessageEvent, args: Message, *, kind: str
) -> None:
    if kind == "voice":
        label = "语音名单"
        usage = VOICE_LIST_USAGE
        list_members = store.list_voice_member
        add_member = store.add_voice_member
        remove_member = store.remove_voice_member
        clear_members = store.clear_voice_member
    else:
        label = "必回名单"
        usage = ALWAYS_REPLY_USAGE
        list_members = store.list_always_reply
        add_member = store.add_always_reply
        remove_member = store.remove_always_reply
        clear_members = store.clear_always_reply

    parts = args.extract_plain_text().strip().split()
    action = parts[0] if parts else ""
    if not action:
        members = await list_members(event.group_id)
        if not members:
            await matcher.finish(f"本群{label}为空。")
        lines = [f"本群{label}（{len(members)} 人）："]
        lines.extend(
            f"{user_id}（{name}）" if name else str(user_id) for user_id, name in members
        )
        await matcher.finish("\n".join(lines))

    targets = _target_user_ids(args)
    if action in ALWAYS_REPLY_CLEAR:
        removed = await clear_members(event.group_id)
        if removed == 0:
            await matcher.finish(f"本群{label}本来就是空的。")
        await matcher.finish(f"已清空本群{label}（{removed} 人）。")

    if action in ALWAYS_REPLY_ADD:
        if not targets:
            await matcher.finish(usage)
        added = 0
        for user_id in targets:
            if await add_member(event.group_id, user_id):
                added += 1
        skipped = len(targets) - added
        if added == 0:
            await matcher.finish(f"这些群友已经在{label}里了。")
        suffix = f"，{skipped} 人原本已在名单中" if skipped else ""
        await matcher.finish(f"已加入{label}：{added} 人{suffix}。")

    if action in ALWAYS_REPLY_REMOVE:
        if not targets:
            await matcher.finish(usage)
        removed = 0
        for user_id in targets:
            if await remove_member(event.group_id, user_id):
                removed += 1
        if removed == 0:
            await matcher.finish(f"这些群友本来就不在{label}里。")
        skipped = len(targets) - removed
        suffix = f"，{skipped} 人原本就不在名单中" if skipped else ""
        await matcher.finish(f"已移出{label}：{removed} 人{suffix}。")

    await matcher.finish(usage)


@always_reply_matcher.handle()
async def handle_always_reply(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await always_reply_matcher.finish(PERMISSION_DENIED)
    await _handle_member_list(always_reply_matcher, event, args, kind="always_reply")


@voice_list_matcher.handle()
async def handle_voice_list(
    bot: Bot,
    event: GroupMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    if not await SUPERUSER(bot, event):
        await voice_list_matcher.finish(PERMISSION_DENIED)
    await _handle_member_list(voice_list_matcher, event, args, kind="voice")


@voice_matcher.handle()
async def handle_voice(bot: Bot, event: GroupMessageEvent) -> None:
    if not await SUPERUSER(bot, event):
        await voice_matcher.finish(PERMISSION_DENIED)
    if not _voice_configured():
        await voice_matcher.finish(_voice_not_configured_notice())
    if event.reply is None:
        await voice_matcher.finish(VOICE_USAGE)

    source = event.reply.message.extract_plain_text().strip()
    if not source:
        await voice_matcher.finish("被引用的消息里没有可朗读的文字。")

    _bots[event.group_id] = bot
    try:
        context = await context_builder.build(event.group_id, source)
        response = await chat_client.complete(
            context, source, session_id=f"group:{event.group_id}"
        )
    except Exception as exc:
        status = f" HTTP {exc.status_code}" if isinstance(exc, ChatAPIError) else ""
        logger.error("群 %s 的语音回复生成失败：%s%s", event.group_id, type(exc).__name__, status)
        await voice_matcher.finish(GENERIC_REPLY_ERROR)

    audio = await _synthesize(response)
    if audio is not None:
        try:
            await bot.send_group_msg(
                group_id=event.group_id, message=_record_segment(audio)
            )
        except Exception as exc:
            logger.warning(
                "群 %s 的语音发送失败：%s", event.group_id, type(exc).__name__
            )
            audio = None
    if audio is None:
        await bot.send_group_msg(
            group_id=event.group_id, message=MessageSegment.text(response)
        )
    await _record_bot_reply(bot, event.group_id, response)


def _draft_label(kind: str) -> str:
    return "语音名单" if kind == "voice" else "必回名单"


def _draft_cancel_notice(kind: str) -> str:
    return VOICE_CANCEL_NOTICE if kind == "voice" else CANCEL_NOTICE


async def _start_private_member_draft(
    matcher,
    bot: Bot,
    event: PrivateMessageEvent,
    args: Message,
    *,
    kind: str,
) -> None:
    if not await SUPERUSER(bot, event):
        await matcher.finish(PERMISSION_DENIED)

    text = args.extract_plain_text().strip()
    if text.lower() in CANCEL_WORDS:
        _private_drafts.pop(event.user_id, None)
        await matcher.finish(_draft_cancel_notice(kind))

    groups = await store.list_groups()
    if not groups:
        await matcher.finish(NO_KNOWN_GROUP)

    group_ids = tuple(group_id for group_id, _, _ in groups)[
        :PRIVATE_GROUP_CHOICE_LIMIT
    ]
    _private_drafts[event.user_id] = _PrivateDraft(
        step="group",
        expires_at=time.monotonic() + PRIVATE_DRAFT_TTL,
        groups=group_ids,
        kind=kind,
    )
    names = await _group_names(bot)
    await matcher.finish(
        _group_choice_list(group_ids, names, label=_draft_label(kind))
    )


@private_always_reply_matcher.handle()
async def handle_private_always_reply(
    bot: Bot,
    event: PrivateMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    await _start_private_member_draft(
        private_always_reply_matcher, bot, event, args, kind="always_reply"
    )


@private_voice_list_matcher.handle()
async def handle_private_voice_list(
    bot: Bot,
    event: PrivateMessageEvent,
    args: Annotated[Message, CommandArg()],
) -> None:
    await _start_private_member_draft(
        private_voice_list_matcher, bot, event, args, kind="voice"
    )


@private_draft_matcher.handle()
async def handle_private_draft(bot: Bot, event: PrivateMessageEvent) -> None:
    draft = _private_drafts.get(event.user_id)
    if draft is None:
        await private_draft_matcher.finish()
    if not await SUPERUSER(bot, event):
        _private_drafts.pop(event.user_id, None)
        await private_draft_matcher.finish(PERMISSION_DENIED)

    label = _draft_label(draft.kind)
    text = event.get_plaintext().strip()
    if text.lower() in CANCEL_WORDS:
        _private_drafts.pop(event.user_id, None)
        await private_draft_matcher.finish(_draft_cancel_notice(draft.kind))

    if draft.step == "group":
        group_id = _select_group(draft, text)
        if group_id is None:
            await private_draft_matcher.finish(
                "没有匹配的群，请重新回复序号或群号，回复“取消”退出。"
            )
        draft.group_id = group_id
        draft.step = "member"
        draft.expires_at = time.monotonic() + PRIVATE_DRAFT_TTL
        await private_draft_matcher.finish(_member_prompt(group_id, label=label))

    targets = list(dict.fromkeys(int(qq) for qq in QQ_PATTERN.findall(text)))
    if not targets:
        await private_draft_matcher.finish(
            "没识别到 QQ 号，请重新发送，回复“取消”退出。"
        )

    group_id = draft.group_id
    if group_id is None:
        _private_drafts.pop(event.user_id, None)
        await private_draft_matcher.finish(_draft_cancel_notice(draft.kind))

    add_member = (
        store.add_voice_member if draft.kind == "voice" else store.add_always_reply
    )
    added = 0
    for user_id in targets:
        if await add_member(group_id, user_id):
            added += 1
    _private_drafts.pop(event.user_id, None)
    skipped = len(targets) - added
    suffix = f"，{skipped} 人原本已在名单中" if skipped else ""
    await private_draft_matcher.finish(
        f"已加入群 {group_id} 的{label}：{added} 人{suffix}。"
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
                group_id=event.group_id, message=MessageSegment.text(RATE_LIMIT_NOTICE)
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
