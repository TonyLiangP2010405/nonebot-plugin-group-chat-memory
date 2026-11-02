from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import NamedTuple

from .api import EmbeddingClient
from .config import Config
from .models import Chunk, Message
from .store import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)
DEFAULT_QUEUE_SIZE = 1024


class _Wake(NamedTuple):
    """提示后台任务为某个群尝试分块。"""

    group_id: int


class _Recover:
    """启动时处理上次运行遗留的待向量化片段。"""


class _Stop:
    """关闭后台任务的哨兵。"""


_RECOVER = _Recover()
_STOP = _Stop()

_Item = _Wake | Chunk | _Recover | _Stop


class MemoryWriter:
    """在后台把已持久化的群消息分块并写入长期记忆向量。

    `record` 只负责持久化原文并唤醒后台任务；分块、Embedding 与写入向量都在
    单一后台任务中完成，任何失败都不会抛回消息处理流程。关闭前应先 `flush()`
    等待已排队的工作结束，`close()` 会取消退避休眠并等待在途请求收尾。
    """

    def __init__(
        self,
        config: Config,
        store: MemoryStore,
        embedding: EmbeddingClient,
        *,
        retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._embedding = embedding
        self._chunk_size = config.group_chat_memory_chunk_size
        self._retry_delays = tuple(retry_delays)
        self._sleep_impl = sleep
        self._queue: asyncio.Queue[_Item] = asyncio.Queue(maxsize=max(1, queue_size))
        self._stop_event = asyncio.Event()
        self._closing = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """启动后台任务，并安排处理上次运行遗留的待向量化片段。"""
        if self._task is not None:
            return
        self._closing = False
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="group-chat-memory-writer")
        try:
            self._queue.put_nowait(_RECOVER)
        except asyncio.QueueFull:
            logger.warning("记忆队列已满，启动时未能安排待处理片段的恢复。")

    async def record(self, message: Message) -> None:
        """先持久化消息原文，再交给后台任务分块；数据库失败只记录日志。"""
        try:
            await self._store.add_message(message)
        except Exception:
            logger.exception("群 %s 的消息写入数据库失败，已跳过长期记忆。", message.group_id)
            return
        if self._closing:
            return
        try:
            self._queue.put_nowait(_Wake(message.group_id))
        except asyncio.QueueFull:
            logger.warning(
                "记忆队列已满，群 %s 的本次消息将在下次消息到达时再分块。", message.group_id
            )

    async def flush(self) -> None:
        """等待当前已排队的分块与向量写入全部结束。"""
        if self._task is None:
            return
        await self._queue.join()

    async def close(self) -> None:
        """停止后台任务：取消退避休眠，等待在途请求结束，可重复调用。"""
        task, self._task = self._task, None
        if task is None:
            return
        self._closing = True
        self._stop_event.set()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(_STOP)
        try:
            await task
        except Exception:
            logger.exception("记忆后台任务异常退出。")

    async def _run(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                try:
                    if isinstance(item, _Stop) or self._closing:
                        return
                    if isinstance(item, _Recover):
                        await self._recover_pending()
                    elif isinstance(item, _Wake):
                        await self._chunk_group(item.group_id)
                    else:
                        await self._embed_chunk(item)
                finally:
                    self._queue.task_done()
        finally:
            self._drain_queue()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.task_done()

    async def _recover_pending(self) -> None:
        try:
            chunks = await self._store.pending_chunks()
        except Exception:
            logger.exception("读取待处理记忆片段失败，已跳过启动恢复。")
            return
        for chunk in chunks:
            if self._closing:
                return
            await self._embed_chunk(chunk)

    async def _chunk_group(self, group_id: int) -> None:
        while not self._closing:
            try:
                chunk = await self._store.create_ready_chunk(group_id, self._chunk_size)
            except Exception:
                logger.exception("群 %s 的消息分块失败，原文已保留。", group_id)
                return
            if chunk is None:
                return
            await self._embed_chunk(chunk)

    async def _embed_chunk(self, chunk: Chunk) -> None:
        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                vectors = await self._embedding.embed([chunk.text])
            except Exception as exc:
                if attempt + 1 >= attempts:
                    logger.error("片段 %s 向量化失败，已保留待处理状态：%s", chunk.id, exc)
                    return
                logger.warning("片段 %s 向量化失败，将重试：%s", chunk.id, exc)
                if not await self._sleep(self._retry_delays[attempt]):
                    return
                continue
            if len(vectors) != 1:
                logger.error(
                    "片段 %s 的 Embedding 返回 %s 条向量，期望 1 条。", chunk.id, len(vectors)
                )
                return
            try:
                await self._store.complete_chunk(chunk.id, vectors[0])
            except Exception:
                logger.exception("片段 %s 写入向量失败，已保留待处理状态。", chunk.id)
            return

    async def _sleep(self, delay: float) -> bool:
        """等待退避时间；返回 False 表示等待期间收到关闭请求。"""
        if self._sleep_impl is not None:
            await self._sleep_impl(delay)
            return not self._closing
        if delay <= 0:
            await asyncio.sleep(0)
            return not self._closing
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return True
        return False
