from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from collections.abc import Awaitable, Callable

from .models import Message
from .store import MemoryStore

logger = logging.getLogger(__name__)

ReplyCallable = Callable[[int, Message], Awaitable[None]]
RandomValue = Callable[[], float]

WORKER_NAME_PREFIX = "group-chat-memory-reply"


class _GroupState:
    """单群的待回复槽位：点名请求 FIFO，随机请求只保留最新一条。"""

    __slots__ = ("explicit", "latest_random", "task", "wake")

    def __init__(self) -> None:
        self.explicit: deque[Message] = deque()
        self.latest_random: Message | None = None
        self.wake = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    def take_next(self) -> Message | None:
        """取出下一条要回复的消息：先点名队列，再随机槽位。"""
        if self.explicit:
            return self.explicit.popleft()
        message, self.latest_random = self.latest_random, None
        return message

    def has_pending(self) -> bool:
        return bool(self.explicit) or self.latest_random is not None

    def discard_pending(self) -> int:
        """丢弃全部等待中的请求，返回丢弃条数（用于关闭日志）。"""
        pending = len(self.explicit) + (1 if self.latest_random is not None else 0)
        self.explicit.clear()
        self.latest_random = None
        return pending


class ReplyCoordinator:
    """按群串行、跨群并行的回复调度器。

    每个群一个 worker：`explicit` 请求进入 FIFO，保证逐条且全部被回复；
    非点名（随机）请求只保留最新一条，等待中的旧请求被直接替换，避免突发消息
    造成回复风暴。不同群各自独立执行，一个群的慢回复不会阻塞其他群。

    提交时只保存消息本身，不复制上下文；`reply` 回调在实际执行时才构建上下文，
    因此回复使用的永远是执行时刻的群聊状态。单个请求的异常被隔离并记录日志，
    不会中断该群后续请求。`submit` 必须在事件循环中调用；关闭后不再接受请求。
    """

    def __init__(
        self,
        store: MemoryStore,
        reply: ReplyCallable,
        *,
        random_value: RandomValue = random.random,
    ) -> None:
        self._store = store
        self._reply = reply
        self._random_value = random_value
        self._groups: dict[int, _GroupState] = {}
        self._closing = False

    async def should_random_reply(self, group_id: int) -> bool:
        """按当前群概率判断是否随机回复：0% 永不、100% 必定。

        概率每次从存储读取，管理命令修改后立即生效。0 与 100 的边界不消耗随机值。
        """
        probability = await self._store.get_probability(group_id)
        if probability <= 0.0:
            return False
        if probability >= 100.0:
            return True
        return self._random_value() < probability / 100.0

    def submit(self, group_id: int, message: Message, explicit: bool) -> None:
        """提交一条待回复消息；`explicit` 表示点名或回复触发的必须回复请求。"""
        if self._closing:
            logger.debug("回复协调器已关闭，忽略群 %s 的回复请求。", group_id)
            return
        state = self._groups.get(group_id)
        if state is None:
            state = _GroupState()
            self._groups[group_id] = state
        if explicit:
            state.explicit.append(message)
        else:
            state.latest_random = message
        state.wake.set()
        self._ensure_worker(group_id, state)

    async def close(self) -> None:
        """停止所有每群 worker：不再接受新请求，等待在途回复结束。

        等待中的请求在关闭时被丢弃（只记录群号与条数），在途回复会执行完毕，
        因此关闭后不会遗留任何 worker 任务。可重复调用；调用方取消 `close()`
        不会级联取消 worker，取消异常照常向上抛出，可再次调用完成收尾。
        """
        first = not self._closing
        self._closing = True
        tasks: list[asyncio.Task[None]] = []
        for group_id, state in list(self._groups.items()):
            if first:
                pending = state.discard_pending()
                if pending:
                    logger.warning(
                        "回复协调器关闭，群 %s 的 %s 条等待中的回复请求被丢弃。",
                        group_id,
                        pending,
                    )
            state.wake.set()
            task = state.task
            if task is not None and not task.done():
                tasks.append(task)
        if tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    def _ensure_worker(self, group_id: int, state: _GroupState) -> None:
        task = state.task
        if task is not None and not task.done():
            return
        task = asyncio.create_task(
            self._run_group(group_id, state), name=f"{WORKER_NAME_PREFIX}-{group_id}"
        )
        state.task = task
        task.add_done_callback(
            lambda done, gid=group_id, st=state: self._on_worker_done(gid, st, done)
        )

    async def _run_group(self, group_id: int, state: _GroupState) -> None:
        while not self._closing:
            message = state.take_next()
            if message is None:
                state.wake.clear()
                if state.has_pending():
                    continue
                await state.wake.wait()
                continue
            try:
                await self._reply(group_id, message)
            except Exception:
                logger.exception("群 %s 的回复失败，已跳过该请求。", group_id)

    def _on_worker_done(self, group_id: int, state: _GroupState, task: asyncio.Task[None]) -> None:
        if state.task is task:
            state.task = None
        if self._closing:
            self._groups.pop(group_id, None)
        if task.cancelled():
            if not self._closing:
                logger.warning("群 %s 的回复 worker 被意外取消。", group_id)
            return
        error = task.exception()
        if error is not None and not self._closing:
            logger.error("群 %s 的回复 worker 意外退出：%r", group_id, error, exc_info=error)
