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
    """提示后台任务为某个群尝试分块；同群唤醒会合并。"""

    group_id: int


class _Stop:
    """关闭后台任务的哨兵。"""


_STOP = _Stop()

_Item = _Wake | Chunk | _Stop


class MemoryWriter:
    """在后台把已持久化的群消息分块并写入长期记忆向量。

    `record` 只负责持久化原文并唤醒后台任务；分块、Embedding 与写入向量都在
    单一后台任务中完成，任何失败都不会抛回消息处理流程。后台任务被意外取消或
    抛出 `BaseException` 时由监督回调记录并允许重新 `start()`，`flush()` 不会
    无限等待。关闭前应先 `flush()`，`close()` 会取消退避休眠并等待在途请求
    收尾。
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
        self._recovered = asyncio.Event()
        self._queued_wakes: set[int] = set()
        self._dirty_groups: set[int] = set()
        self._overflow_warned = False
        self._closing = False
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """后台任务是否仍在运行。"""
        return self._is_running()

    def start(self) -> None:
        """启动后台任务并恢复待处理记忆；已在运行时不做任何事。"""
        if self._is_running():
            return
        self._task = None
        self._closing = False
        self._stop_event.clear()
        self._recovered.clear()
        task = asyncio.create_task(self._run(), name="group-chat-memory-writer")
        self._task = task
        task.add_done_callback(self._on_task_done)

    async def record(self, message: Message) -> None:
        """先持久化消息原文，再交给后台任务分块；数据库失败只记录日志。"""
        try:
            await self._store.add_message(message)
        except Exception:
            logger.exception("群 %s 的消息写入数据库失败，已跳过长期记忆。", message.group_id)
            return
        self._dirty_groups.add(message.group_id)
        if self._closing:
            return
        self._wake(message.group_id)

    async def flush(self, timeout: float | None = None) -> None:
        """等待已排队的分块与向量写入全部结束。

        传入 `timeout`（秒）时最多等待该时长，超时抛出 `asyncio.TimeoutError`；
        该异常只表示本次等待超时，后台任务与后续 `close()` 都不受影响。后台任务
        未运行时立即返回，未完成的群聊消息由下次启动恢复。
        """
        if timeout is None:
            await self._wait_until_idle()
            return
        await asyncio.wait_for(self._wait_until_idle(), timeout)

    async def close(self) -> None:
        """停止后台任务：取消退避休眠，等待在途请求结束，可重复调用。

        等待时使用 `asyncio.shield`，调用方取消 `close()` 不会级联取消后台任务，
        取消异常照常向上抛出；只有后台任务本身被取消时才按关闭处理并只记录日志。
        """
        task = self._task
        if task is None or task.done():
            return
        first = not self._closing
        self._closing = True
        self._stop_event.set()
        if first:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(_STOP)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            if first:
                logger.warning("记忆后台任务在关闭时被取消，未完成的消息由下次启动补做。")
        except Exception:
            if first:
                logger.exception("记忆后台任务异常退出。")

    def _is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if self._task is task:
            self._task = None
            self._recovered.set()
        if task.cancelled():
            if not self._closing:
                logger.warning("记忆后台任务被意外取消，未处理的消息将在下次启动时补做。")
            return
        error = task.exception()
        if error is not None and not self._closing:
            logger.error("记忆后台任务意外退出：%r", error, exc_info=error)

    def _wake(self, group_id: int) -> None:
        if not self._is_running() or group_id in self._queued_wakes:
            return
        try:
            self._queue.put_nowait(_Wake(group_id))
        except asyncio.QueueFull:
            if not self._overflow_warned:
                self._overflow_warned = True
                logger.warning("记忆队列已满，未完成的分块将在下次启动时补做。")
            return
        self._queued_wakes.add(group_id)
        self._overflow_warned = False

    async def _wait_until_idle(self) -> None:
        if not self._is_running():
            self._warn_stranded()
            return
        await self._recovered.wait()
        if not self._is_running():
            self._warn_stranded()
            return
        await self._queue.join()

    def _warn_stranded(self) -> None:
        if not self._queue.empty():
            logger.warning(
                "记忆后台任务未运行，%s 项排队工作将在下次启动时补做。", self._queue.qsize()
            )

    async def _run(self) -> None:
        try:
            try:
                await self._recover_pending()
            finally:
                self._recovered.set()
            while True:
                item = await self._queue.get()
                try:
                    if isinstance(item, _Stop) or self._closing:
                        return
                    if isinstance(item, _Wake):
                        self._queued_wakes.discard(item.group_id)
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
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, _Wake):
                self._queued_wakes.discard(item.group_id)
                self._dirty_groups.add(item.group_id)
            self._queue.task_done()

    async def _recover_pending(self) -> None:
        try:
            chunks = await self._store.pending_chunks()
        except Exception:
            logger.exception("读取待处理记忆片段失败，已跳过片段恢复。")
            chunks = []
        try:
            groups = await self._store.pending_groups(self._chunk_size)
        except Exception:
            logger.exception("读取待处理群聊消息失败，已跳过补分块恢复。")
            groups = []
        for chunk in chunks:
            if self._closing:
                return
            await self._embed_chunk(chunk)
        for group_id in dict.fromkeys([*groups, *sorted(self._dirty_groups)]):
            if self._closing:
                return
            self._dirty_groups.add(group_id)
            await self._chunk_group(group_id)

    async def _chunk_group(self, group_id: int) -> None:
        while not self._closing:
            try:
                chunk = await self._store.create_ready_chunk(group_id, self._chunk_size)
            except Exception:
                logger.exception("群 %s 的消息分块失败，原文已保留。", group_id)
                return
            if chunk is None:
                self._dirty_groups.discard(group_id)
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
