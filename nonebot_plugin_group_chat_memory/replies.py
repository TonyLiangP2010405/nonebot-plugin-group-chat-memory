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

    def discard_random(self) -> bool:
        """丢弃等待中的随机请求；点名请求保留给 worker 排空。"""
        had_random = self.latest_random is not None
        self.latest_random = None
        return had_random


class ReplyCoordinator:
    """按群串行、跨群并行的回复调度器。

    每个群一个 worker：`explicit` 请求进入 FIFO，保证逐条且全部被回复；
    非点名（随机）请求只保留最新一条，等待中的旧请求被直接替换，避免突发消息
    造成回复风暴。不同群各自独立执行，一个群的慢回复不会阻塞其他群。

    提交时只保存消息本身，不复制上下文；`reply` 回调在实际执行时才构建上下文，
    因此回复使用的永远是执行时刻的群聊状态。单个请求的异常被隔离并记录日志，
    不会中断该群后续请求；`reply` 抛出的 `CancelledError` 会结束 worker，并由
    监督回调为剩余请求重新拉起 worker。`submit` 必须在事件循环中调用；关闭后
    不再接受请求，已接受的点名请求仍会全部执行完毕。
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

    async def should_random_reply(
        self, group_id: int, *, minimum_probability: float = 0.0
    ) -> bool:
        """按当前群概率判断是否随机回复：0% 永不、100% 必定。

        概率每次从存储读取，管理命令修改后立即生效。0 与 100 的边界不消耗随机值。
        `minimum_probability` 用于好感榜等额外加成：取群概率与该值的较大者，
        仍然只消耗一次随机值。
        """
        probability = max(
            await self._store.get_probability(group_id), minimum_probability
        )
        if probability <= 0.0:
            return False
        if probability >= 100.0:
            return True
        return self._random_value() < probability / 100.0

    async def should_quote(self, group_id: int) -> bool:
        """按当前群引用概率决定这条回复是否引用触发消息：0% 从不、100% 必定。

        引用概率与随机回复概率相互独立，只决定回复的发送形式。概率每次从存储
        读取，管理命令修改后立即生效；0 与 100 的边界不消耗随机值。
        """
        probability = await self._store.get_quote_probability(group_id)
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
        """停止接受新请求，并在所有已接受的点名请求回复完毕后关闭 worker。

        已接受的 explicit（点名/回复）请求全部执行完毕才返回；等待中的随机请求
        被丢弃（只记录群号，不含正文），在途回复也会执行完毕，因此关闭后不会
        遗留任何 worker 或群状态。已结束 worker 的 done callback 可能仍在队列里
        并因排队的点名请求拉起新 worker，所以这里循环等待到完全静默。
        可重复调用；调用方取消 `close()` 不会级联取消 worker，取消异常照常向上
        抛出，可再次调用完成收尾。
        """
        first = not self._closing
        self._closing = True
        for group_id, state in list(self._groups.items()):
            if first and state.discard_random():
                logger.warning("回复协调器关闭，群 %s 的等待中的随机回复请求被丢弃。", group_id)
            state.wake.set()
        while True:
            # 先让已结束 worker 的 done callback 执行: 它可能为排队的点名请求拉起新 worker.
            await asyncio.sleep(0)
            tasks = [
                state.task
                for state in self._groups.values()
                if state.task is not None and not state.task.done()
            ]
            if tasks:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in tasks), return_exceptions=True
                )
                continue
            pending = [
                (group_id, state) for group_id, state in self._groups.items() if state.explicit
            ]
            if not pending:
                break
            # 兜底: worker 的 done callback 尚未执行时, 由 close 自己续跑点名请求.
            for group_id, state in pending:
                state.wake.set()
                self._ensure_worker(group_id, state)
        self._groups.clear()

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
        while True:
            message = state.take_next()
            if message is None:
                if self._closing:
                    return
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
        if task.cancelled():
            if not self._closing:
                logger.warning("群 %s 的回复 worker 被取消。", group_id)
        else:
            error = task.exception()
            if error is not None and not self._closing:
                logger.error("群 %s 的回复 worker 意外退出：%r", group_id, error, exc_info=error)
        if state.task is task:
            state.task = None
        if state.task is not None and not state.task.done():
            return
        if self._needs_recovery(state):
            state.wake.set()
            self._ensure_worker(group_id, state)
        elif self._closing:
            self._groups.pop(group_id, None)

    def _needs_recovery(self, state: _GroupState) -> bool:
        """是否还有必须继续处理的请求: 关闭后仍要排空已接受的点名请求。"""
        if state.explicit:
            return True
        return not self._closing and state.latest_random is not None
