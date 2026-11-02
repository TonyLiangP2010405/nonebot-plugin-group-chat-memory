from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    """群聊中的一条文本消息。"""

    group_id: int
    user_id: int
    display_name: str
    text: str
    created_at: datetime = Field(default_factory=_utc_now)
    is_bot: bool = False


class ChunkStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"


class Chunk(BaseModel):
    """一段待向量化或已向量化的群聊记忆。"""

    id: int
    group_id: int
    text: str
    status: ChunkStatus = ChunkStatus.PENDING


class ChatContext(BaseModel):
    """一次模型请求所需的最近消息与长期记忆。"""

    recent_messages: list[Message] = Field(default_factory=list)
    long_term_chunks: list[Chunk] = Field(default_factory=list)
