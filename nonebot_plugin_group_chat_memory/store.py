from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import struct
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone

import aiosqlite
import sqlite_vec

from .config import (
    DEFAULT_AFFECTION_ENABLED,
    DEFAULT_AFFECTION_PROBABILITY,
    DEFAULT_SUPERUSER_ALWAYS_REPLY,
    RUNTIME_CONFIG_DEFAULTS,
    Config,
)
from .models import Chunk, ChunkStatus, Message

logger = logging.getLogger(__name__)

DIMENSIONS_META_KEY = "embedding_dimensions"


def _supports_load_extension() -> bool:
    return hasattr(sqlite3.Connection, "enable_load_extension")

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS groups (
        group_id INTEGER PRIMARY KEY,
        probability REAL NOT NULL DEFAULT 0
            CHECK (probability >= 0 AND probability <= 100),
        hourly_limit INTEGER NOT NULL DEFAULT 10
            CHECK (hourly_limit >= 1 AND hourly_limit <= 10000),
        affection_enabled INTEGER NOT NULL DEFAULT 1
            CHECK (affection_enabled IN (0, 1)),
        affection_probability REAL NOT NULL DEFAULT 30
            CHECK (affection_probability >= 0 AND affection_probability <= 100),
        superuser_always_reply INTEGER NOT NULL DEFAULT 1
            CHECK (superuser_always_reply IN (0, 1)),
        quote_probability REAL
            CHECK (quote_probability >= 0 AND quote_probability <= 100),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS always_reply_members (
        group_id INTEGER NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL,
        display_name TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        PRIMARY KEY (group_id, user_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS request_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
        requested_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_request_usage_group_time
        ON request_usage(group_id, requested_at)
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
    """
    CREATE TABLE IF NOT EXISTS memory_vector_blobs (
        chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        group_id INTEGER NOT NULL,
        embedding BLOB NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_vector_blobs_group
        ON memory_vector_blobs(group_id)
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
        self._default_quote_probability = config.group_chat_memory_default_quote_probability
        self._default_hourly_limit = config.group_chat_memory_hourly_request_limit
        self._default_affection_enabled = DEFAULT_AFFECTION_ENABLED
        self._default_affection_probability = DEFAULT_AFFECTION_PROBABILITY
        self._default_superuser_always_reply = DEFAULT_SUPERUSER_ALWAYS_REPLY
        self._db: aiosqlite.Connection | None = None
        self._use_sqlite_vec = False
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """打开数据库；优先加载 sqlite-vec，不支持时使用 BLOB 向量回退。"""
        async with self._lock:
            if self._db is not None:
                return

            path = self._config.database_path
            path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(path, isolation_level=None)
            use_sqlite_vec = _supports_load_extension()
            try:
                db.row_factory = aiosqlite.Row
                if use_sqlite_vec:
                    try:
                        await db.enable_load_extension(True)
                        await db.load_extension(sqlite_vec.loadable_path())
                    except Exception as exc:
                        logger.warning(
                            "sqlite-vec 加载失败，改用 SQLite BLOB 向量检索：%s",
                            type(exc).__name__,
                        )
                        use_sqlite_vec = False
                    finally:
                        if hasattr(db, "enable_load_extension"):
                            with suppress(Exception):
                                await db.enable_load_extension(False)
                else:
                    logger.warning(
                        "当前 Python 的 sqlite3 不支持加载扩展，改用 SQLite BLOB 向量检索。"
                    )
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.execute("PRAGMA synchronous = NORMAL")
                self._db = db
                self._use_sqlite_vec = use_sqlite_vec
                async with self._transaction_on(db) as tx:
                    for statement in _SCHEMA_STATEMENTS:
                        await tx.execute(statement)
                    await self._ensure_groups_hourly_limit(tx)
                    await self._ensure_groups_affection_settings(tx)
                    await self._ensure_groups_superuser_reply(tx)
                    await self._ensure_groups_quote_probability(tx)
                    await self._ensure_dimensions(tx)
                    if use_sqlite_vec:
                        await tx.execute(
                            _VECTOR_TABLE_STATEMENT.format(dimensions=self._dimensions)
                        )
                    else:
                        cursor = await tx.execute(
                            "UPDATE chunks SET status = ? WHERE status = ? AND NOT EXISTS ("
                            "SELECT 1 FROM memory_vector_blobs AS b WHERE b.chunk_id = chunks.id"
                            ")",
                            (ChunkStatus.PENDING.value, ChunkStatus.READY.value),
                        )
                        if cursor.rowcount > 0:
                            logger.warning(
                                "%s 个片段缺少 BLOB 向量，已重新排队等待向量化。",
                                cursor.rowcount,
                            )
                        await cursor.close()
            except BaseException:
                self._db = None
                self._use_sqlite_vec = False
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

    async def get_quote_probability(self, group_id: int) -> float:
        """返回当前群引用回复的概率；未单独设置时使用插件默认值。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT quote_probability FROM groups WHERE group_id = ?", (group_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None or row["quote_probability"] is None:
            return self._default_quote_probability
        return float(row["quote_probability"])

    async def set_quote_probability(self, group_id: int, value: float) -> None:
        """设置当前群引用回复的概率，取值必须在 0 到 100 之间。"""
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 100.0:
            raise ValueError("引用回复概率必须是 0 到 100 之间的数值。")
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO groups"
                "(group_id, probability, quote_probability, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(group_id) DO UPDATE SET "
                "quote_probability = excluded.quote_probability, "
                "updated_at = excluded.updated_at",
                (
                    group_id,
                    self._default_probability,
                    probability,
                    _format_timestamp(_utc_now()),
                ),
            )

    async def get_hourly_limit(self, group_id: int) -> int:
        """返回当前群每小时模型请求上限。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT hourly_limit FROM groups WHERE group_id = ?", (group_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return self._default_hourly_limit
        return int(row["hourly_limit"])

    async def set_hourly_limit(self, group_id: int, value: int) -> None:
        """设置当前群每小时模型请求上限，范围 1 到 10000。"""
        limit = int(value)
        if not 1 <= limit <= 10000:
            raise ValueError("每小时请求上限必须是 1 到 10000 的整数。")
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO groups(group_id, probability, hourly_limit, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(group_id) DO UPDATE SET "
                "hourly_limit = excluded.hourly_limit, updated_at = excluded.updated_at",
                (
                    group_id,
                    self._default_probability,
                    limit,
                    _format_timestamp(_utc_now()),
                ),
            )

    async def get_affection_enabled(self, group_id: int) -> bool:
        """返回当前群是否启用好感榜成员额外触发。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT affection_enabled FROM groups WHERE group_id = ?", (group_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return self._default_affection_enabled
        return bool(row["affection_enabled"])

    async def set_affection_enabled(self, group_id: int, enabled: bool) -> None:
        """启用或关闭当前群的好感榜成员额外触发。"""
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO groups"
                "(group_id, probability, hourly_limit, affection_enabled, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(group_id) DO UPDATE SET "
                "affection_enabled = excluded.affection_enabled, "
                "updated_at = excluded.updated_at",
                (
                    group_id,
                    self._default_probability,
                    self._default_hourly_limit,
                    int(enabled),
                    _format_timestamp(_utc_now()),
                ),
            )

    async def get_affection_probability(self, group_id: int) -> float:
        """返回当前群好感榜成员的额外触发概率。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT affection_probability FROM groups WHERE group_id = ?", (group_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return self._default_affection_probability
        return float(row["affection_probability"])

    async def set_affection_probability(self, group_id: int, value: float) -> None:
        """设置当前群好感榜成员的额外触发概率，取值必须在 0 到 100 之间。"""
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 100.0:
            raise ValueError("好感榜触发概率必须是 0 到 100 之间的数值。")
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO groups"
                "(group_id, probability, hourly_limit, affection_probability, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(group_id) DO UPDATE SET "
                "affection_probability = excluded.affection_probability, "
                "updated_at = excluded.updated_at",
                (
                    group_id,
                    self._default_probability,
                    self._default_hourly_limit,
                    probability,
                    _format_timestamp(_utc_now()),
                ),
            )

    async def get_superuser_always_reply(self, group_id: int) -> bool:
        """返回当前群是否让 SUPERUSER 的普通文本必定触发回复。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT superuser_always_reply FROM groups WHERE group_id = ?",
                (group_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return self._default_superuser_always_reply
        return bool(row["superuser_always_reply"])

    async def set_superuser_always_reply(self, group_id: int, enabled: bool) -> None:
        """启用或关闭当前群 SUPERUSER 的必定触发。"""
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO groups"
                "(group_id, probability, hourly_limit, superuser_always_reply, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(group_id) DO UPDATE SET "
                "superuser_always_reply = excluded.superuser_always_reply, "
                "updated_at = excluded.updated_at",
                (
                    group_id,
                    self._default_probability,
                    self._default_hourly_limit,
                    int(enabled),
                    _format_timestamp(_utc_now()),
                ),
            )

    async def is_always_reply(self, group_id: int, user_id: int) -> bool:
        """返回成员是否在当前群的必回名单中。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT 1 FROM always_reply_members WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def add_always_reply(
        self, group_id: int, user_id: int, display_name: str = ""
    ) -> bool:
        """把成员加入必回名单；返回是否为新增（已存在时只更新缓存的群名片）。"""
        async with self._transaction() as db:
            await self._ensure_group(db, group_id)
            cursor = await db.execute(
                "SELECT 1 FROM always_reply_members WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            existed = await cursor.fetchone() is not None
            await cursor.close()
            await db.execute(
                "INSERT INTO always_reply_members"
                "(group_id, user_id, display_name, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(group_id, user_id) DO UPDATE SET "
                "display_name = excluded.display_name",
                (group_id, user_id, display_name, _format_timestamp(_utc_now())),
            )
        return not existed

    async def remove_always_reply(self, group_id: int, user_id: int) -> bool:
        """把成员移出必回名单，返回是否真的删除了记录。"""
        async with self._transaction() as db:
            cursor = await db.execute(
                "DELETE FROM always_reply_members WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            removed = cursor.rowcount > 0
            await cursor.close()
        return removed

    async def clear_always_reply(self, group_id: int) -> int:
        """清空当前群的必回名单，返回删除的人数。"""
        async with self._transaction() as db:
            cursor = await db.execute(
                "DELETE FROM always_reply_members WHERE group_id = ?", (group_id,)
            )
            removed = cursor.rowcount
            await cursor.close()
        return max(int(removed or 0), 0)

    async def list_always_reply(self, group_id: int) -> list[tuple[int, str]]:
        """返回当前群必回名单的 (QQ, 群名片)，群名片优先取最近一条消息。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT a.user_id, COALESCE(("
                "SELECT m.display_name FROM messages AS m "
                "WHERE m.group_id = a.group_id AND m.user_id = a.user_id "
                "ORDER BY m.id DESC LIMIT 1"
                "), a.display_name, '') AS display_name "
                "FROM always_reply_members AS a "
                "WHERE a.group_id = ? ORDER BY a.created_at, a.user_id",
                (group_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [(int(row["user_id"]), str(row["display_name"])) for row in rows]

    async def list_groups(self) -> list[tuple[int, int, str | None]]:
        """返回插件记录过的群：(群号, 消息数, 最后一条消息时间)，最近活跃在前。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT g.group_id AS group_id, "
                "(SELECT count(*) FROM messages AS m "
                "WHERE m.group_id = g.group_id) AS messages, "
                "(SELECT max(m.created_at) FROM messages AS m "
                "WHERE m.group_id = g.group_id) AS last_at "
                "FROM groups AS g ORDER BY COALESCE(last_at, '') DESC, g.group_id"
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            (
                int(row["group_id"]),
                int(row["messages"]),
                str(row["last_at"]) if row["last_at"] is not None else None,
            )
            for row in rows
        ]

    async def consume_request_slot(
        self, group_id: int, now: datetime | None = None
    ) -> bool:
        """在滚动 60 分钟窗口内尝试占用一次模型请求名额。"""
        current = _as_utc(now or _utc_now())
        timestamp = current.timestamp()
        cutoff = (current - timedelta(hours=1)).timestamp()
        async with self._transaction() as db:
            await self._ensure_group(db, group_id)
            await db.execute(
                "DELETE FROM request_usage WHERE group_id = ? AND requested_at <= ?",
                (group_id, cutoff),
            )
            cursor = await db.execute(
                "SELECT count(*) AS usage FROM request_usage WHERE group_id = ?",
                (group_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            cursor = await db.execute(
                "SELECT hourly_limit FROM groups WHERE group_id = ?", (group_id,)
            )
            group_row = await cursor.fetchone()
            await cursor.close()
            limit = (
                int(group_row["hourly_limit"])
                if group_row is not None
                else self._default_hourly_limit
            )
            if row is not None and int(row["usage"]) >= limit:
                return False
            await db.execute(
                "INSERT INTO request_usage(group_id, requested_at) VALUES (?, ?)",
                (group_id, timestamp),
            )
            return True

    async def get_hour_usage(self, group_id: int, now: datetime | None = None) -> int:
        """返回当前群滚动 60 分钟窗口内已占用的请求次数。"""
        cutoff = (_as_utc(now or _utc_now()) - timedelta(hours=1)).timestamp()
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute(
                "SELECT count(*) AS usage FROM request_usage "
                "WHERE group_id = ? AND requested_at > ?",
                (group_id, cutoff),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return int(row["usage"]) if row is not None else 0

    async def get_runtime_config(self) -> dict[str, str]:
        """读取全 Bot 共用的模型运行时配置。"""
        async with self._lock:
            db = self._require_db()
            cursor = await db.execute("SELECT key, value FROM runtime_config")
            rows = await cursor.fetchall()
            await cursor.close()
        return {row["key"]: row["value"] for row in rows}

    async def set_runtime_config(self, key: str, value: str) -> None:
        """持久化一项白名单内的全局模型配置。"""
        if key not in RUNTIME_CONFIG_DEFAULTS:
            raise ValueError(f"不支持的运行时配置项：{key}")
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO runtime_config(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
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
            if self._use_sqlite_vec:
                await db.execute("DELETE FROM memory_vectors WHERE chunk_id = ?", (chunk_id,))
                await db.execute(
                    "INSERT INTO memory_vectors(chunk_id, group_id, embedding) VALUES (?, ?, ?)",
                    (chunk_id, row["group_id"], sqlite_vec.serialize_float32(embedding)),
                )
            else:
                await db.execute(
                    "DELETE FROM memory_vector_blobs WHERE chunk_id = ?", (chunk_id,)
                )
                await db.execute(
                    "INSERT INTO memory_vector_blobs(chunk_id, group_id, embedding) "
                    "VALUES (?, ?, ?)",
                    (chunk_id, row["group_id"], _pack_vector(embedding)),
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
            if self._use_sqlite_vec:
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

            cursor = await db.execute(
                "SELECT c.id, c.group_id, c.text, c.status, b.embedding "
                "FROM memory_vector_blobs AS b JOIN chunks AS c ON c.id = b.chunk_id "
                "WHERE b.group_id = ? AND c.status = ?",
                (group_id, ChunkStatus.READY.value),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        scored = [
            (
                _cosine_similarity(embedding, _unpack_vector(row["embedding"], self._dimensions)),
                int(row["id"]),
                row,
            )
            for row in rows
            if len(row["embedding"]) == self._dimensions * 4
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [_row_to_chunk(row) for _, _, row in scored[:limit]]

    async def clear_group(self, group_id: int) -> None:
        """删除当前群的消息、片段、向量和请求计数，保留概率、引用概率与每小时上限。"""
        async with self._transaction() as db:
            await db.execute(
                "DELETE FROM memory_vector_blobs WHERE group_id = ?", (group_id,)
            )
            if self._use_sqlite_vec:
                await db.execute("DELETE FROM memory_vectors WHERE group_id = ?", (group_id,))
            await db.execute("DELETE FROM request_usage WHERE group_id = ?", (group_id,))
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
            "INSERT INTO groups(group_id, probability, hourly_limit, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(group_id) DO NOTHING",
            (
                group_id,
                self._default_probability,
                self._default_hourly_limit,
                _format_timestamp(_utc_now()),
            ),
        )

    async def _ensure_groups_hourly_limit(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(groups)")
        rows = await cursor.fetchall()
        await cursor.close()
        if not any(row["name"] == "hourly_limit" for row in rows):
            await db.execute(
                f"ALTER TABLE groups ADD COLUMN hourly_limit INTEGER NOT NULL DEFAULT "
                f"{self._default_hourly_limit}"
            )

    async def _ensure_groups_affection_settings(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(groups)")
        rows = await cursor.fetchall()
        await cursor.close()
        columns = {row["name"] for row in rows}
        if "affection_enabled" not in columns:
            await db.execute(
                "ALTER TABLE groups ADD COLUMN affection_enabled INTEGER NOT NULL "
                f"DEFAULT {int(self._default_affection_enabled)}"
            )
        if "affection_probability" not in columns:
            await db.execute(
                "ALTER TABLE groups ADD COLUMN affection_probability REAL NOT NULL "
                f"DEFAULT {self._default_affection_probability}"
            )

    async def _ensure_groups_superuser_reply(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(groups)")
        rows = await cursor.fetchall()
        await cursor.close()
        if not any(row["name"] == "superuser_always_reply" for row in rows):
            await db.execute(
                "ALTER TABLE groups ADD COLUMN superuser_always_reply INTEGER NOT NULL "
                f"DEFAULT {int(self._default_superuser_always_reply)}"
            )

    async def _ensure_groups_quote_probability(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(groups)")
        rows = await cursor.fetchall()
        await cursor.close()
        if not any(row["name"] == "quote_probability" for row in rows):
            await db.execute(
                "ALTER TABLE groups ADD COLUMN quote_probability REAL "
                "CHECK (quote_probability >= 0 AND quote_probability <= 100)"
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


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(value: bytes, dimensions: int) -> list[float]:
    return list(struct.unpack(f"<{dimensions}f", value))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


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
