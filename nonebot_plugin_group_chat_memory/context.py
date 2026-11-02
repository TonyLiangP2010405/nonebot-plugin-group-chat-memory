from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .config import Config
from .models import ChatContext, Chunk, Message
from .store import MemoryStore

if TYPE_CHECKING:
    from .api import EmbeddingClient

logger = logging.getLogger(__name__)

MEMORY_SECTION_TITLE = "## 长期记忆（较早的相关片段）"
RECENT_SECTION_TITLE = "## 最近群聊（从旧到新）"
CURRENT_SECTION_TITLE = "## 当前消息"
MEMBER_LABEL = "群友"
BOT_LABEL = "Bot"
SPEAKER_GUIDE = (
    "聊天记录中的「群友」是群里的其他人，「Bot」或「(bot)」标记的都是你自己之前说过的话。"
    "只回复「当前消息」，不要替其他群友发言，也不要编造记录中没有出现过的内容。"
)
TRUNCATION_MARK = "…"


class ContextBuilder:
    """并行读取当前群的最近消息与长期记忆；长期记忆失败时降级为最近消息。"""

    def __init__(self, config: Config, store: MemoryStore, embedding: EmbeddingClient) -> None:
        self._config = config
        self._store = store
        self._embedding = embedding

    async def build(self, group_id: int, query: str) -> ChatContext:
        """构建一次回复所需的上下文；Embedding 或检索失败不影响最近消息。"""
        recent, chunks = await asyncio.gather(
            self._store.recent_messages(group_id, self._config.group_chat_memory_recent_messages),
            self._long_term_chunks(group_id, query),
        )
        return ChatContext(recent_messages=recent, long_term_chunks=chunks)

    async def _long_term_chunks(self, group_id: int, query: str) -> list[Chunk]:
        """在独立短超时内完成查询向量与检索，超时或失败都降级为空列表。"""
        limit = self._config.group_chat_memory_long_term_memories
        if limit < 1 or not query.strip():
            return []
        try:
            return await asyncio.wait_for(
                self._lookup_long_term(group_id, query, limit),
                timeout=self._config.group_chat_memory_query_timeout,
            )
        except Exception as exc:
            logger.warning("长期记忆检索失败，仅使用最近消息：%s: %s", type(exc).__name__, exc)
            return []

    async def _lookup_long_term(self, group_id: int, query: str, limit: int) -> list[Chunk]:
        vectors = await self._embedding.embed([query])
        if not vectors:
            return []
        return await self._store.search(group_id, vectors[0], limit)


def build_chat_messages(
    config: Config, context: ChatContext, current_text: str
) -> list[dict[str, str]]:
    """组装 OpenAI 兼容 messages：系统提示 + 长期记忆 + 最近消息 + 当前输入。

    字符预算优先保证当前输入完整，其次保留最近消息，最后才放长期记忆。
    """
    budget = config.group_chat_memory_context_max_chars
    current_block = f"{CURRENT_SECTION_TITLE}\n{MEMBER_LABEL}: {current_text.strip()}"
    remaining = max(budget - len(current_block), 0)

    recent_lines = _take_recent_lines(context.recent_messages, remaining)
    remaining -= sum(len(line) for line in recent_lines) + max(len(recent_lines) - 1, 0)

    memory_blocks = _take_chunk_blocks(
        context.long_term_chunks,
        config.group_chat_memory_long_term_memories,
        max(remaining, 0),
    )

    sections: list[str] = []
    if memory_blocks:
        sections.append(f"{MEMORY_SECTION_TITLE}\n" + "\n\n".join(memory_blocks))
    if recent_lines:
        sections.append(f"{RECENT_SECTION_TITLE}\n" + "\n".join(recent_lines))
    sections.append(current_block)

    system_prompt = "\n\n".join(
        part for part in (config.group_chat_memory_system_prompt.strip(), SPEAKER_GUIDE) if part
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _take_recent_lines(messages: list[Message], budget: int) -> list[str]:
    """从最新消息往前取，放不下时截断最后一条，返回从旧到新的行。"""
    lines: list[str] = []
    used = 0
    for message in reversed(messages):
        line = _format_message(message)
        separator = 1 if lines else 0
        available = budget - used - separator
        if available <= 0:
            break
        if len(line) > available:
            line = line[: available - 1] + TRUNCATION_MARK if available >= 2 else line[:available]
        lines.append(line)
        used += len(line) + separator
        if used >= budget:
            break
    lines.reverse()
    return lines


def _take_chunk_blocks(chunks: list[Chunk], limit: int, budget: int) -> list[str]:
    """按相关度顺序取最多 limit 个片段，放不下时停止，保留更高优先级的最近消息。"""
    blocks: list[str] = []
    used = 0
    for chunk in chunks[: max(limit, 0)]:
        text = chunk.text.strip()
        if not text:
            continue
        separator = 2 if blocks else 0
        if used + len(text) + separator > budget:
            break
        blocks.append(text)
        used += len(text) + separator
    return blocks


def _format_message(message: Message) -> str:
    created_at = _as_utc(message.created_at)
    label = BOT_LABEL if message.is_bot else MEMBER_LABEL
    name = message.display_name.strip() or str(message.user_id)
    return f"[{created_at:%Y-%m-%d %H:%M}] {label} {name}: {message.text.strip()}"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
