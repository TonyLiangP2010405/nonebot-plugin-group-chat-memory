from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite
import sqlite_vec

from .config import Config
from .models import Chunk, ChunkStatus, Message

DIMENSIONS_META_KEY = "embedding_dimensions"

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS groups (
        group_id INTEGER PRIMARY KEY,
        probability REAL NOT NULL DEFAULT 0
            CHECK (probability >= 0 AND probability <= 100),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
        text TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'ready')),
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_group ON chunks(group_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status, id)",
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL,
        display_name TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_bot INTEGER NOT NULL DEFAULT 0 CHECK (is_bot IN (0, 1)),
        chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_group_time
        ON messages(group_id, created_at, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_unchunked
        ON messages(group_id, id) WHERE chunk_id IS NULL
    """,
)

_VECTOR_TABLE_STATEMENT = """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
        chunk_id INTEGER PRIMARY KEY,
        group_id INTEGER PARTITION KEY,
        embedding FLOAT[{dimensions}] distance_metric=cosine
    )
"""


class MemoryStore:
    """按群隔离的 SQLite 与 sqlite-vec 持久化存储。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._dimensions = config.group_chat_memory_embedding_dimensions
        self._default_probability = config.group_chat_memory_default_probability
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """打开数据库、加载 sqlite-vec 并创建缺失的表结构。"""
        async with self._lock:
            if self._db is not None:
                return

            path = self._config.database_path
            path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(path, isolation_level=None)
            try:
                db.row_factory = aiosqlite.Row
                await db.enable_load_extension(True)
                await db.load_extension(sqlite_vec.loadable_path())
                await db.enable_load_extension(False)
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.execute("PRAGMA synchronous = NORMAL")
                self._db = db
                async with self._transaction_on(db) as tx:
                    for statement in _SCHEMA_STATEMENTS:
                        await tx.execute(statement)
                    await self._ensure_dimensions(tx)
                    await tx.execute(_VECTOR_TABLE_STATEMENT.format(dimensions=self._dimensions))
            except BaseException:
                self._db = None
                await db.close()
                raise

    async def close(self) -> None:
        """关闭数据库连接，可重复调用。"""
        async with self._lock:
            db, self._db = self._db, None
            if db is not None:
                await db.close()

    async def add_message(self, message: Message) -> int:
        """持久化一条群消息，返回数据库内的消息 ID。"""
        async with self._transaction() as db:
            await self._ensure_group(db, message.group_id)
            cursor = await db.execute(
                "INSERT INTO messages"
                "(group_id, user_id, display_name, text, created_at, is_bot) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message.group_id,
                    message.user_id,
                    message.display_name,
                    message.text,
                    _format_timestamp(message.created_at),
                    int(message.is_bot),
                ),
            )
            message_id = cursor.lastrowid
            await cursor.close()
        if message_id is None:
            raise RuntimeError("消息写入失败：数据库未返回消息 ID。")
        return int(message_id)

    async def recent_messages(self, group_id: int, limit: int) -> list[Message]:
        """返回当前群最近 limit 条消息，按时间从旧到新排列。"""
        if limit < 1:
            return []
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT group_id, user_id, display_name, text, created_at, is_bot "
                "FROM messages WHERE group_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (group_id, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [_row_to_message(row) for row in reversed(rows)]

    async def get_probability(self, group_id: int) -> float:
        """返回当前群的随机回复概率；未配置的群使用插件默认值。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT probability FROM groups WHERE group_id = ?", (group_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return self._default_probability
        return float(row["probability"])

    async def set_probability(self, group_id: int, value: float) -> None:
        """设置当前群的随机回复概率，取值必须在 0 到 100 之间。"""
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 100.0:
            raise ValueError("随机回复概率必须是 0 到 100 之间的数值。")
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO groups(group_id, probability, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(group_id) DO UPDATE SET "
                "probability = excluded.probability, updated_at = excluded.updated_at",
                (group_id, probability, _format_timestamp(_utc_now())),
            )

    async def create_ready_chunk(self, group_id: int, size: int) -> Chunk | None:
        """用当前群最早的 size 条未分块消息组成片段，并原子关联这些消息。"""
        if size < 1:
            raise ValueError("分块大小必须至少为 1。")
        async with self._transaction() as db:
            await self._ensure_group(db, group_id)
            cursor = await db.execute(
                "SELECT id, user_id, display_name, text, created_at, is_bot "
                "FROM messages WHERE group_id = ? AND chunk_id IS NULL "
                "ORDER BY created_at, id LIMIT ?",
                (group_id, size),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            if len(rows) < size:
                return None

            text = _format_chunk_text(group_id, rows)
            cursor = await db.execute(
                "INSERT INTO chunks(group_id, text, status, created_at) VALUES (?, ?, ?, ?)",
                (
                    group_id,
                    text,
                    ChunkStatus.PENDING.value,
                    _format_timestamp(_utc_now()),
                ),
            )
            chunk_id = cursor.lastrowid
            await cursor.close()
            if chunk_id is None:
                raise RuntimeError("片段写入失败：数据库未返回片段 ID。")
            placeholders = ", ".join("?" for _ in rows)
            await db.execute(
                f"UPDATE messages SET chunk_id = ? WHERE id IN ({placeholders})",
                (chunk_id, *(row["id"] for row in rows)),
            )
        return Chunk(id=chunk_id, group_id=group_id, text=text)

    async def pending_chunks(self) -> list[Chunk]:
        """返回所有群中尚未写入向量的片段。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT id, group_id, text, status FROM chunks WHERE status = ? ORDER BY id",
                (ChunkStatus.PENDING.value,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [_row_to_chunk(row) for row in rows]

    async def pending_groups(self, size: int) -> list[int]:
        """返回未分块消息数量至少为 size 的群，按群号升序。"""
        if size < 1:
            raise ValueError("分块大小必须至少为 1。")
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT group_id FROM messages WHERE chunk_id IS NULL "
                "GROUP BY group_id HAVING count(*) >= ? ORDER BY group_id",
                (size,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [int(row["group_id"]) for row in rows]

    async def complete_chunk(self, chunk_id: int, vector: list[float]) -> None:
        """写入片段向量并标记为 ready；片段已被清空时静默忽略。"""
        embedding = _validated_vector(vector, self._dimensions)
        async with self._transaction() as db:
            cursor = await db.execute("SELECT group_id FROM chunks WHERE id = ?", (chunk_id,))
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return
            await db.execute("DELETE FROM memory_vectors WHERE chunk_id = ?", (chunk_id,))
            await db.execute(
                "INSERT INTO memory_vectors(chunk_id, group_id, embedding) VALUES (?, ?, ?)",
                (chunk_id, row["group_id"], sqlite_vec.serialize_float32(embedding)),
            )
            await db.execute(
                "UPDATE chunks SET status = ? WHERE id = ?",
                (ChunkStatus.READY.value, chunk_id),
            )

    async def search(self, group_id: int, vector: list[float], limit: int) -> list[Chunk]:
        """在当前群已完成的片段中检索最接近的 limit 个片段。"""
        embedding = _validated_vector(vector, self._dimensions)
        if limit < 1:
            return []
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT c.id, c.group_id, c.text, c.status FROM ("
                "SELECT chunk_id, distance FROM memory_vectors "
                "WHERE embedding MATCH ? AND k = ? AND group_id = ?"
                ") AS v JOIN chunks AS c ON c.id = v.chunk_id ORDER BY v.distance",
                (sqlite_vec.serialize_float32(embedding), limit, group_id),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [_row_to_chunk(row) for row in rows]

    async def clear_group(self, group_id: int) -> None:
        """删除当前群的消息、片段与向量，保留概率配置。"""
        async with self._transaction() as db:
            await db.execute("DELETE FROM memory_vectors WHERE group_id = ?", (group_id,))
            await db.execute("DELETE FROM messages WHERE group_id = ?", (group_id,))
            await db.execute("DELETE FROM chunks WHERE group_id = ?", (group_id,))

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("数据库尚未打开，请先调用 MemoryStore.open()。")
        return self._db

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        db = self._require_db()
        async with self._lock, self._transaction_on(db) as tx:
            yield tx

    @asynccontextmanager
    async def _transaction_on(
        self, db: aiosqlite.Connection
    ) -> AsyncIterator[aiosqlite.Connection]:
        await db.execute("BEGIN IMMEDIATE")
        try:
            yield db
        except BaseException:
            await db.rollback()
            raise
        else:
            await db.commit()

    async def _ensure_group(self, db: aiosqlite.Connection, group_id: int) -> None:
        await db.execute(
            "INSERT INTO groups(group_id, probability, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(group_id) DO NOTHING",
            (group_id, self._default_probability, _format_timestamp(_utc_now())),
        )

    async def _ensure_dimensions(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("SELECT value FROM meta WHERE key = ?", (DIMENSIONS_META_KEY,))
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            await db.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                (DIMENSIONS_META_KEY, str(self._dimensions)),
            )
            return
        stored = int(row["value"])
        if stored != self._dimensions:
            raise RuntimeError(
                f"数据库向量维度为 {stored}，与配置的 {self._dimensions} 不一致；"
                "请恢复原配置或删除数据库文件后重建。"
            )


def _row_to_message(row: aiosqlite.Row) -> Message:
    return Message(
        group_id=row["group_id"],
        user_id=row["user_id"],
        display_name=row["display_name"],
        text=row["text"],
        created_at=_parse_timestamp(row["created_at"]),
        is_bot=bool(row["is_bot"]),
    )


def _row_to_chunk(row: aiosqlite.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        group_id=row["group_id"],
        text=row["text"],
        status=row["status"],
    )


def _format_chunk_text(group_id: int, rows: Sequence[aiosqlite.Row]) -> str:
    first = _parse_timestamp(rows[0]["created_at"])
    last = _parse_timestamp(rows[-1]["created_at"])
    lines = [f"群 {group_id} 聊天记录（{first:%Y-%m-%d %H:%M} ~ {last:%Y-%m-%d %H:%M} UTC）"]
    for row in rows:
        created_at = _parse_timestamp(row["created_at"])
        name = row["display_name"].strip() or str(row["user_id"])
        suffix = "(bot)" if row["is_bot"] else ""
        lines.append(f"[{created_at:%Y-%m-%d %H:%M}] {name}{suffix}: {row['text']}")
    return "\n".join(lines)


def _validated_vector(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) != dimensions:
        raise ValueError(f"向量维度为 {len(vector)}，配置要求 {dimensions}。")
    values = [float(item) for item in vector]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("向量包含非有限数值。")
    return values


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
