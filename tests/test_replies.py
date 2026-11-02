from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.models import Message
from nonebot_plugin_group_chat_memory.replies import WORKER_NAME_PREFIX, ReplyCoordinator
from nonebot_plugin_group_chat_memory.store import MemoryStore

DIMENSIONS = 4
BASE_TIME = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def make_config(tmp_path: Path, *, index: int = 0, probability: float = 0.0) -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / f"memory-{index}.db"),
        group_chat_memory_embedding_dimensions=DIMENSIONS,
        group_chat_memory_default_probability=probability,
    )


def make_message(group_id: int, text: str, *, index: int = 0) -> Message:
    return Message(
        group_id=group_id,
        user_id=1000,
        display_name="Alice",
        text=text,
        created_at=BASE_TIME + timedelta(minutes=index),
    )


async def wait_until(predicate, timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout)


def reply_workers() -> set[asyncio.Task]:
    """当前仍存活的每群回复 worker。"""
    return {task for task in asyncio.all_tasks() if task.get_name().startswith(WORKER_NAME_PREFIX)}


class FakeReply:
    """记录回复调用；默认在每条回复上等待独立闸门，便于精确控制交错。"""

    def __init__(
        self,
        *,
        auto: bool = False,
        fail_on: set[str] | None = None,
        cancel_on: set[str] | None = None,
    ) -> None:
        self.calls: list[tuple[int, Message]] = []
        self.completed: list[tuple[int, str]] = []
        self.entered: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self.gates: list[asyncio.Event] = []
        self.auto = auto
        self.fail_on = fail_on or set()
        self.cancel_on = cancel_on or set()

    async def __call__(self, group_id: int, message: Message) -> None:
        self.calls.append((group_id, message))
        self.entered.put_nowait((group_id, message.text))
        if not self.auto:
            gate = asyncio.Event()
            self.gates.append(gate)
            await gate.wait()
        if message.text in self.cancel_on:
            raise asyncio.CancelledError
        self.completed.append((group_id, message.text))
        if message.text in self.fail_on:
            raise RuntimeError("聊天请求失败：HTTP 502")

    async def next_call(self, timeout: float = 1.0) -> tuple[int, str]:
        return await asyncio.wait_for(self.entered.get(), timeout)

    def release(self, index: int) -> None:
        self.gates[index].set()

    @property
    def texts(self) -> list[str]:
        return [message.text for _, message in self.calls]


@pytest_asyncio.fixture(loop_scope="function")
async def make_store(tmp_path):
    stores: list[MemoryStore] = []

    async def create(**overrides) -> MemoryStore:
        memory = MemoryStore(make_config(tmp_path, index=len(stores), **overrides))
        await memory.open()
        stores.append(memory)
        return memory

    try:
        yield create
    finally:
        for memory in stores:
            await memory.close()


@pytest_asyncio.fixture(loop_scope="function")
async def store(make_store):
    return await make_store()


class FreshContextReply:
    """执行时才读取存储的回复替身，用于验证上下文不在提交时复制。"""

    def __init__(self, store: MemoryStore, *, gate_text: str | None = None) -> None:
        self._store = store
        self._gate_text = gate_text
        self.recent_counts: list[int] = []
        self.gates: list[asyncio.Event] = []

    async def __call__(self, group_id: int, message: Message) -> None:
        self.recent_counts.append(len(await self._store.recent_messages(group_id, 20)))
        if message.text == self._gate_text:
            gate = asyncio.Event()
            self.gates.append(gate)
            await gate.wait()


@pytest_asyncio.fixture(loop_scope="function")
async def make_coordinator(store):
    coordinators: list[ReplyCoordinator] = []
    replies: list[FakeReply | FreshContextReply] = []

    def create(reply, *, random_value=None) -> ReplyCoordinator:
        if random_value is None:
            coordinator = ReplyCoordinator(store, reply)
        else:
            coordinator = ReplyCoordinator(store, reply, random_value=random_value)
        replies.append(reply)
        coordinators.append(coordinator)
        return coordinator

    try:
        yield create
    finally:
        for coordinator in coordinators:
            close_task = asyncio.create_task(coordinator.close())
            # 断言失败时可能遗留未放行的闸门, 反复放行直到 close 返回.
            for _ in range(100):
                for reply in replies:
                    for gate in getattr(reply, "gates", []):
                        gate.set()
                if close_task.done():
                    break
                await asyncio.sleep(0.01)
            await asyncio.wait_for(close_task, 1.0)


class TestExplicitReplies:
    async def test_same_group_explicit_requests_run_one_at_a_time_in_order(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        for index, text in enumerate(("第一条", "第二条", "第三条")):
            coordinator.submit(100, make_message(100, text, index=index), True)

        assert await reply.next_call() == (100, "第一条")
        with pytest.raises(asyncio.TimeoutError):
            await reply.next_call(timeout=0.1)
        reply.release(0)

        assert await reply.next_call() == (100, "第二条")
        with pytest.raises(asyncio.TimeoutError):
            await reply.next_call(timeout=0.1)
        reply.release(1)

        assert await reply.next_call() == (100, "第三条")
        reply.release(2)
        await wait_until(
            lambda: len(reply.gates) == 3 and all(gate.is_set() for gate in reply.gates)
        )
        assert reply.texts == ["第一条", "第二条", "第三条"]

    async def test_every_explicit_request_is_answered(self, make_coordinator):
        reply = FakeReply(auto=True)
        coordinator = make_coordinator(reply)
        for index in range(5):
            coordinator.submit(100, make_message(100, f"第{index}条", index=index), True)

        await wait_until(lambda: len(reply.calls) == 5)
        assert reply.texts == [f"第{index}条" for index in range(5)]

    async def test_reply_runs_immediately_when_group_is_idle(self, make_coordinator):
        reply = FakeReply(auto=True)
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "点名"), True)

        assert await reply.next_call() == (100, "点名")


class TestRandomReplies:
    async def test_random_requests_coalesce_to_the_latest_only(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "随机一"), False)
        assert await reply.next_call() == (100, "随机一")

        for index, text in enumerate(("随机二", "随机三", "随机四"), start=1):
            coordinator.submit(100, make_message(100, text, index=index), False)
        reply.release(0)

        assert await reply.next_call() == (100, "随机四")
        reply.release(1)
        await wait_until(lambda: len(reply.gates) == 2 and reply.gates[1].is_set())
        with pytest.raises(asyncio.TimeoutError):
            await reply.next_call(timeout=0.1)
        assert reply.texts == ["随机一", "随机四"]

    async def test_explicit_request_preempts_waiting_random(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "随机一"), False)
        assert await reply.next_call() == (100, "随机一")

        coordinator.submit(100, make_message(100, "点名", index=1), True)
        coordinator.submit(100, make_message(100, "随机二", index=2), False)
        reply.release(0)

        assert await reply.next_call() == (100, "点名")
        reply.release(1)
        assert await reply.next_call() == (100, "随机二")
        reply.release(2)
        await wait_until(lambda: len(reply.gates) == 3 and reply.gates[2].is_set())

    async def test_random_request_waits_while_an_explicit_reply_runs(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "点名"), True)
        assert await reply.next_call() == (100, "点名")

        coordinator.submit(100, make_message(100, "随机", index=1), False)
        with pytest.raises(asyncio.TimeoutError):
            await reply.next_call(timeout=0.1)
        reply.release(0)

        assert await reply.next_call() == (100, "随机")
        reply.release(1)
        await wait_until(lambda: len(reply.gates) == 2 and reply.gates[1].is_set())


class TestGroupIsolation:
    async def test_different_groups_reply_concurrently(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "群一百"), True)
        coordinator.submit(200, make_message(200, "群二百"), True)

        entered = {await reply.next_call(), await reply.next_call()}
        assert entered == {(100, "群一百"), (200, "群二百")}
        reply.release(0)
        reply.release(1)
        await wait_until(lambda: len(reply.gates) == 2 and all(g.is_set() for g in reply.gates))

    async def test_stalled_group_does_not_block_another_group(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "卡住"), True)
        assert await reply.next_call() == (100, "卡住")

        for index, text in enumerate(("甲", "乙"), start=1):
            coordinator.submit(200, make_message(200, text, index=index), True)
        for gate, text in ((1, "甲"), (2, "乙")):
            assert await reply.next_call() == (200, text)
            reply.release(gate)

        reply.release(0)
        await wait_until(lambda: len(reply.gates) == 3 and all(g.is_set() for g in reply.gates))
        assert reply.texts == ["卡住", "甲", "乙"]


class TestFreshContext:
    async def test_reply_sees_context_built_at_execution_time(self, store, make_coordinator):
        reply = FreshContextReply(store, gate_text="在途")
        coordinator = make_coordinator(reply)
        await store.add_message(make_message(100, "旧消息"))
        coordinator.submit(100, make_message(100, "在途"), True)
        await wait_until(lambda: len(reply.recent_counts) == 1)

        await store.add_message(make_message(100, "等待期间新增", index=1))
        await store.add_message(make_message(100, "等待期间又新增", index=2))
        coordinator.submit(100, make_message(100, "随机", index=3), False)
        reply.gates[0].set()

        await wait_until(lambda: len(reply.recent_counts) == 2)
        assert reply.recent_counts == [1, 3]

    async def test_submit_does_not_snapshot_context(self, store, make_coordinator, monkeypatch):
        reads: list[int] = []

        async def forbidden(group_id: int, limit: int) -> list[Message]:
            reads.append(group_id)
            raise AssertionError("submit 时不应构建上下文")

        monkeypatch.setattr(store, "recent_messages", forbidden)
        reply = FakeReply(auto=True)
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "消息"), True)

        await wait_until(lambda: len(reply.calls) == 1)
        assert reads == []


class TestProbability:
    @pytest.mark.parametrize(
        ("probability", "value", "expected"),
        [
            (0.0, 0.0, False),
            (100.0, 0.99, True),
            (50.0, 0.0, True),
            (50.0, 0.49, True),
            (50.0, 0.5, False),
            (50.0, 0.99, False),
            (1.0, 0.009, True),
            (1.0, 0.01, False),
        ],
    )
    async def test_probability_threshold(
        self, store, make_coordinator, probability, value, expected
    ):
        await store.set_probability(100, probability)
        coordinator = make_coordinator(FakeReply(auto=True), random_value=lambda: value)

        assert await coordinator.should_random_reply(100) is expected

    @pytest.mark.parametrize(("probability", "expected"), [(0.0, False), (100.0, True)])
    async def test_bounds_never_consult_random_value(
        self, store, make_coordinator, probability, expected
    ):
        calls: list[int] = []

        def random_value() -> float:
            calls.append(1)
            return 0.5

        await store.set_probability(100, probability)
        coordinator = make_coordinator(FakeReply(auto=True), random_value=random_value)

        assert await coordinator.should_random_reply(100) is expected
        assert calls == []

    async def test_unconfigured_group_uses_configured_default(self, make_store):
        store = await make_store(probability=100.0)
        coordinator = ReplyCoordinator(store, FakeReply(auto=True))
        try:
            assert await coordinator.should_random_reply(999) is True
        finally:
            await coordinator.close()

    async def test_probability_changes_apply_without_restart(self, store, make_coordinator):
        coordinator = make_coordinator(FakeReply(auto=True), random_value=lambda: 0.5)

        await store.set_probability(100, 0.0)
        assert await coordinator.should_random_reply(100) is False
        await store.set_probability(100, 100.0)
        assert await coordinator.should_random_reply(100) is True
        await store.set_probability(100, 40.0)
        assert await coordinator.should_random_reply(100) is False


class TestErrorIsolation:
    async def test_failed_reply_is_logged_and_later_requests_still_run(
        self, make_coordinator, caplog
    ):
        reply = FakeReply(auto=True, fail_on={"第一条"})
        coordinator = make_coordinator(reply)

        with caplog.at_level(logging.ERROR):
            coordinator.submit(100, make_message(100, "第一条"), True)
            coordinator.submit(100, make_message(100, "第二条", index=1), True)
            await wait_until(lambda: len(reply.calls) == 2)

        assert reply.texts == ["第一条", "第二条"]
        assert "100" in caplog.text
        assert "RuntimeError" in caplog.text
        assert "回复失败" in caplog.text
        assert "意外退出" not in caplog.text
        assert "第一条" not in caplog.text

    async def test_failed_random_reply_does_not_stop_the_group(self, make_coordinator):
        reply = FakeReply(auto=True, fail_on={"随机一"})
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "随机一"), False)
        await wait_until(lambda: len(reply.calls) == 1)

        coordinator.submit(100, make_message(100, "随机二", index=1), False)
        await wait_until(lambda: len(reply.calls) == 2)
        assert reply.texts == ["随机一", "随机二"]


class TestWorkerRecovery:
    async def test_cancelled_reply_does_not_strand_queued_explicit_requests(
        self, make_coordinator, caplog
    ):
        reply = FakeReply(cancel_on={"第一条"})
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "第一条"), True)
        assert await reply.next_call() == (100, "第一条")
        coordinator.submit(100, make_message(100, "第二条", index=1), True)

        with caplog.at_level(logging.WARNING):
            reply.release(0)
            # 不再提交任何新请求, 监督回调必须自行拉起 worker 处理排队的第二条.
            assert await reply.next_call() == (100, "第二条")
            reply.release(1)
            await wait_until(lambda: reply.completed == [(100, "第二条")])

        assert reply.texts == ["第一条", "第二条"]
        assert "被取消" in caplog.text
        assert "100" in caplog.text
        assert "第一条" not in caplog.text


class TestClose:
    async def test_close_waits_for_in_flight_reply(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "在途"), True)
        assert await reply.next_call() == (100, "在途")
        assert reply.completed == []

        close_task = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0.01)
        assert not close_task.done()

        reply.release(0)
        await asyncio.wait_for(close_task, 1.0)
        assert reply.completed == [(100, "在途")]
        assert reply.texts == ["在途"]

    async def test_close_drains_accepted_explicit_and_discards_random(
        self, make_coordinator, caplog
    ):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "在途"), True)
        assert await reply.next_call() == (100, "在途")
        coordinator.submit(100, make_message(100, "排队", index=1), True)
        coordinator.submit(100, make_message(100, "随机内容", index=2), False)

        with caplog.at_level(logging.WARNING):
            close_task = asyncio.create_task(coordinator.close())
            await asyncio.sleep(0.01)
            assert not close_task.done()
            reply.release(0)

            assert await reply.next_call() == (100, "排队")
            reply.release(1)
            await asyncio.wait_for(close_task, 1.0)

        assert reply.texts == ["在途", "排队"]
        assert reply.completed == [(100, "在途"), (100, "排队")]
        assert reply_workers() == set()
        assert "丢弃" in caplog.text
        assert "100" in caplog.text
        assert "排队" not in caplog.text
        assert "随机内容" not in caplog.text

    async def test_close_drains_every_accepted_explicit_request(self, make_coordinator):
        reply = FakeReply()
        coordinator = make_coordinator(reply)
        for index, text in enumerate(("一", "二", "三")):
            coordinator.submit(100, make_message(100, text, index=index), True)
        assert await reply.next_call() == (100, "一")

        close_task = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0.01)
        assert not close_task.done()

        for index, text in ((0, "一"), (1, "二"), (2, "三")):
            if index:
                assert await reply.next_call() == (100, text)
            reply.release(index)
        await asyncio.wait_for(close_task, 1.0)

        assert reply.texts == ["一", "二", "三"]
        assert reply.completed == [(100, "一"), (100, "二"), (100, "三")]
        assert reply_workers() == set()

    async def test_close_leaves_no_workers(self, make_coordinator):
        baseline = asyncio.all_tasks()
        reply = FakeReply(auto=True)
        coordinator = make_coordinator(reply)
        for group_id in (100, 200, 300):
            coordinator.submit(group_id, make_message(group_id, "消息"), False)
        await wait_until(lambda: len(reply.calls) == 3)

        await asyncio.wait_for(coordinator.close(), 1.0)
        assert reply_workers() == set()
        assert asyncio.all_tasks() == baseline

    async def test_close_is_idempotent(self, make_coordinator):
        reply = FakeReply(auto=True)
        coordinator = make_coordinator(reply)
        coordinator.submit(100, make_message(100, "消息"), True)
        await wait_until(lambda: len(reply.calls) == 1)

        await asyncio.wait_for(coordinator.close(), 1.0)
        await asyncio.wait_for(coordinator.close(), 0.5)

    async def test_submit_after_close_is_ignored(self, make_coordinator):
        reply = FakeReply(auto=True)
        coordinator = make_coordinator(reply)
        await asyncio.wait_for(coordinator.close(), 1.0)

        coordinator.submit(100, make_message(100, "迟到"), True)
        await asyncio.sleep(0.05)
        assert reply.calls == []
