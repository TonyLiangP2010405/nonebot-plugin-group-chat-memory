import asyncio
import itertools
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
import sqlite_vec

from nonebot_plugin_group_chat_memory.api import EmbeddingAPIError
from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.memory import DEFAULT_RETRY_DELAYS, MemoryWriter
from nonebot_plugin_group_chat_memory.models import ChunkStatus, Message
from nonebot_plugin_group_chat_memory.store import MemoryStore

DIMENSIONS = 4
VECTOR = [1.0, 0.0, 0.0, 0.0]
BASE_TIME = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def make_config(tmp_path: Path, *, chunk_size: int = 10) -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / "memory.db"),
        group_chat_memory_embedding_dimensions=DIMENSIONS,
        group_chat_memory_chunk_size=chunk_size,
    )


def make_message(group_id: int, text: str, *, index: int = 0) -> Message:
    return Message(
        group_id=group_id,
        user_id=1000,
        display_name="Alice",
        text=text,
        created_at=BASE_TIME + timedelta(minutes=index),
    )


def count_rows(path: Path, group_id: int) -> dict[str, int]:
    """通过独立连接统计各表行数，避免 JOIN 掩盖残留的孤儿行。"""
    connection = sqlite3.connect(path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    try:
        return {
            table: connection.execute(
                f"SELECT count(*) FROM {table} WHERE group_id = ?", (group_id,)
            ).fetchone()[0]
            for table in ("messages", "chunks", "memory_vectors")
        }
    finally:
        connection.close()


async def wait_until(predicate, timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout)


class FakeEmbedding:
    """可记录调用、可延迟、可失败的 Embedding 替身。"""

    def __init__(self, *, failures: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.failures = failures
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failures > 0:
            self.failures -= 1
            raise EmbeddingAPIError("Embedding 服务返回 HTTP 503。")
        return [list(VECTOR) for _ in texts]


class WorkerFatal(BaseException):
    """模拟逃逸出 Exception 捕获的后台任务致命错误。"""


class FatalEmbedding:
    """Embedding 阶段抛出 BaseException，用于验证后台任务监督。"""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise WorkerFatal("模拟致命错误")


class FailingStore:
    """记录消息写入必定失败的存储替身。"""

    def __init__(self) -> None:
        self.attempts = 0

    async def add_message(self, message: Message) -> int:
        self.attempts += 1
        raise RuntimeError("数据库写入失败：disk I/O error")

    async def pending_chunks(self) -> list:
        return []

    async def pending_groups(self, size: int) -> list[int]:
        return []


class BrokenPendingStore:
    """启动恢复读取待处理记录失败的存储替身。"""

    async def pending_chunks(self) -> list:
        raise RuntimeError("数据库读取失败：disk I/O error")

    async def pending_groups(self, size: int) -> list[int]:
        raise RuntimeError("数据库读取失败：disk I/O error")


class FlakyStore:
    """包装真实存储，按次数让分块失败。"""

    def __init__(self, store: MemoryStore, *, failures: int = 1) -> None:
        self._store = store
        self.failures = failures
        self.chunk_attempts = 0

    async def add_message(self, message: Message) -> int:
        return await self._store.add_message(message)

    async def pending_chunks(self) -> list:
        return await self._store.pending_chunks()

    async def pending_groups(self, size: int) -> list[int]:
        return await self._store.pending_groups(size)

    async def create_ready_chunk(self, group_id: int, size: int):
        self.chunk_attempts += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("数据库写入失败：disk I/O error")
        return await self._store.create_ready_chunk(group_id, size)

    async def complete_chunk(self, chunk_id: int, vector: list[float]) -> None:
        await self._store.complete_chunk(chunk_id, vector)


async def record_many(writer: MemoryWriter, group_id: int, count: int, *, start: int = 0) -> None:
    for offset in range(count):
        index = start + offset
        await writer.record(make_message(group_id, f"消息{index}", index=index))


async def seed_ready_chunk(store: MemoryStore, group_id: int, *, count: int = 10) -> int:
    for index in range(count):
        await store.add_message(make_message(group_id, f"旧消息{index}", index=index))
    chunk = await store.create_ready_chunk(group_id, count)
    assert chunk is not None
    await store.complete_chunk(chunk.id, VECTOR)
    return chunk.id


class Harness:
    """共享同一配置与存储，便于测试关闭后重开。"""

    def __init__(self, config: Config, store: MemoryStore) -> None:
        self.config = config
        self.store = store
        self.writers: list[MemoryWriter] = []

    @property
    def path(self) -> Path:
        return self.config.database_path

    def writer(self, embedding: FakeEmbedding | None = None, **kwargs) -> MemoryWriter:
        writer = MemoryWriter(self.config, self.store, embedding or FakeEmbedding(), **kwargs)
        self.writers.append(writer)
        return writer

    async def aclose(self) -> None:
        for writer in self.writers:
            await writer.close()
        await self.store.close()


@pytest_asyncio.fixture(loop_scope="function")
async def harness_factory(tmp_path):
    harnesses: list[Harness] = []

    async def make(**kwargs) -> Harness:
        config = make_config(tmp_path, **kwargs)
        store = MemoryStore(config)
        await store.open()
        harness = Harness(config, store)
        harnesses.append(harness)
        return harness

    try:
        yield make
    finally:
        for harness in harnesses:
            await harness.aclose()


@pytest_asyncio.fixture(loop_scope="function")
async def harness(harness_factory):
    return await harness_factory()


class TestChunking:
    async def test_nine_messages_do_not_embed(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        await record_many(writer, 100, 9)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert embedding.calls == []
        assert await harness.store.pending_chunks() == []
        assert count_rows(harness.path, 100) == {
            "messages": 9,
            "chunks": 0,
            "memory_vectors": 0,
        }

    async def test_tenth_message_embeds_the_chunk_once(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        await record_many(writer, 100, 10)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 1
        (texts,) = embedding.calls
        assert len(texts) == 1
        assert all(f"消息{index}" in texts[0] for index in range(10))
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY
        assert chunk.text == texts[0]
        assert count_rows(harness.path, 100) == {
            "messages": 10,
            "chunks": 1,
            "memory_vectors": 1,
        }

    async def test_each_chunk_uses_one_batch_call(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        await record_many(writer, 100, 20)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert [len(texts) for texts in embedding.calls] == [1, 1]
        assert "消息0" in embedding.calls[0][0]
        assert "消息10" in embedding.calls[1][0]
        assert count_rows(harness.path, 100) == {
            "messages": 20,
            "chunks": 2,
            "memory_vectors": 2,
        }

    async def test_chunk_size_comes_from_config(self, harness_factory):
        harness = await harness_factory(chunk_size=3)
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        await record_many(writer, 100, 3)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 1
        assert count_rows(harness.path, 100)["chunks"] == 1

    async def test_groups_are_chunked_independently(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        await record_many(writer, 100, 10)
        await record_many(writer, 200, 10)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 2
        assert count_rows(harness.path, 100)["memory_vectors"] == 1
        assert count_rows(harness.path, 200)["memory_vectors"] == 1
        (chunk,) = await harness.store.search(200, VECTOR, 5)
        assert chunk.group_id == 200

    async def test_bot_messages_are_chunked_with_user_messages(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        for index in range(9):
            await writer.record(make_message(100, f"消息{index}", index=index))
        await writer.record(
            Message(
                group_id=100,
                user_id=1,
                display_name="小助手",
                text="Bot 的回复",
                created_at=BASE_TIME + timedelta(minutes=9),
                is_bot=True,
            )
        )
        await asyncio.wait_for(writer.flush(), 1.0)

        (texts,) = embedding.calls
        assert "Bot 的回复" in texts[0]
        assert "(bot)" in texts[0]


class TestRecord:
    async def test_record_awaits_persistence_before_returning(self, harness):
        gate = asyncio.Event()
        entered = asyncio.Event()
        add_message = harness.store.add_message

        async def gated_add_message(message: Message) -> int:
            entered.set()
            await gate.wait()
            return await add_message(message)

        harness.store.add_message = gated_add_message
        writer = harness.writer()
        writer.start()

        task = asyncio.create_task(writer.record(make_message(100, "消息0")))
        await asyncio.wait_for(entered.wait(), 1.0)
        assert await harness.store.recent_messages(100, 10) == []
        gate.set()
        await asyncio.wait_for(task, 1.0)
        assert [message.text for message in await harness.store.recent_messages(100, 10)] == [
            "消息0"
        ]

    async def test_record_does_not_block_when_embedding_is_stuck(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        await asyncio.wait_for(writer.record(make_message(100, "消息10", index=10)), 0.5)

        assert len(await harness.store.recent_messages(100, 20)) == 11
        embedding.gate.set()
        await asyncio.wait_for(writer.flush(), 1.0)

    async def test_database_error_is_logged_without_message_text(self, harness, caplog):
        store = FailingStore()
        writer = MemoryWriter(harness.config, store, FakeEmbedding())
        writer.start()

        with caplog.at_level(logging.ERROR):
            await writer.record(make_message(100, "私密内容"))
            await asyncio.wait_for(writer.flush(), 1.0)

        assert store.attempts == 1
        assert "disk I/O error" in caplog.text
        assert "私密内容" not in caplog.text

    async def test_failed_record_does_not_queue_work(self, harness):
        store = FailingStore()
        embedding = FakeEmbedding()
        writer = MemoryWriter(harness.config, store, embedding)
        writer.start()

        await writer.record(make_message(100, "消息0"))
        await asyncio.wait_for(writer.flush(), 1.0)

        assert embedding.calls == []


class TestRetry:
    async def test_transient_failure_retries_with_backoff_then_ready(self, harness):
        delays: list[float] = []

        async def sleep(delay: float) -> None:
            delays.append(delay)

        embedding = FakeEmbedding(failures=2)
        writer = harness.writer(embedding, retry_delays=(0.5, 1.0), sleep=sleep)
        writer.start()

        await record_many(writer, 100, 10)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 3
        assert delays == [0.5, 1.0]
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY
        assert count_rows(harness.path, 100)["memory_vectors"] == 1

    def test_default_retry_delays_are_bounded_and_exponential(self):
        assert 1 <= len(DEFAULT_RETRY_DELAYS) <= 5
        assert all(delay > 0 for delay in DEFAULT_RETRY_DELAYS)
        for previous, current in itertools.pairwise(DEFAULT_RETRY_DELAYS):
            assert current == pytest.approx(previous * 2)

    async def test_exhausted_retries_keep_messages_and_pending_chunk(self, harness):
        embedding = FakeEmbedding(failures=99)
        writer = harness.writer(embedding, retry_delays=(0.0, 0.0))
        writer.start()

        await record_many(writer, 100, 10)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 3
        (pending,) = await harness.store.pending_chunks()
        assert pending.status is ChunkStatus.PENDING
        assert count_rows(harness.path, 100) == {
            "messages": 10,
            "chunks": 1,
            "memory_vectors": 0,
        }

    async def test_failure_is_logged_without_chunk_text(self, harness, caplog):
        embedding = FakeEmbedding(failures=99)
        writer = harness.writer(embedding, retry_delays=(0.0,))
        writer.start()

        with caplog.at_level(logging.WARNING):
            await record_many(writer, 100, 10)
            await asyncio.wait_for(writer.flush(), 1.0)

        assert "Embedding 服务返回 HTTP 503" in caplog.text
        assert "消息0" not in caplog.text


class TestRecovery:
    async def test_restart_resumes_pending_chunks(self, harness_factory):
        first = await harness_factory()
        failing = FakeEmbedding(failures=99)
        writer = first.writer(failing, retry_delays=(0.0,))
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(writer.flush(), 1.0)
        (pending,) = await first.store.pending_chunks()
        assert pending.status is ChunkStatus.PENDING
        await first.aclose()

        second = await harness_factory()
        working = FakeEmbedding()
        restarted = second.writer(working)
        restarted.start()
        await asyncio.wait_for(restarted.flush(), 1.0)

        assert working.calls == [[pending.text]]
        assert await second.store.pending_chunks() == []
        (chunk,) = await second.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY
        assert chunk.text == pending.text

    async def test_startup_only_embeds_pending_chunks(self, harness):
        await seed_ready_chunk(harness.store, 200)
        for index in range(10):
            await harness.store.add_message(make_message(100, f"消息{index}", index=index))
        pending = await harness.store.create_ready_chunk(100, 10)
        assert pending is not None

        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()
        await asyncio.wait_for(writer.flush(), 1.0)

        assert embedding.calls == [[pending.text]]
        assert await harness.store.pending_chunks() == []
        assert count_rows(harness.path, 100)["memory_vectors"] == 1
        assert count_rows(harness.path, 200)["memory_vectors"] == 1

    async def test_restart_chunks_raw_unchunked_messages_once(self, harness_factory):
        first = await harness_factory()
        for index in range(10):
            await first.store.add_message(make_message(100, f"消息{index}", index=index))
        await first.aclose()

        second = await harness_factory()
        working = FakeEmbedding()
        writer = second.writer(working)
        writer.start()
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(working.calls) == 1
        assert count_rows(second.path, 100) == {
            "messages": 10,
            "chunks": 1,
            "memory_vectors": 1,
        }
        assert await second.store.pending_chunks() == []
        assert await second.store.pending_groups(10) == []
        await second.aclose()

        third = await harness_factory()
        again = FakeEmbedding()
        restarted = third.writer(again)
        restarted.start()
        await asyncio.wait_for(restarted.flush(), 1.0)

        assert again.calls == []
        assert count_rows(third.path, 100) == {
            "messages": 10,
            "chunks": 1,
            "memory_vectors": 1,
        }

    async def test_startup_recovery_failure_is_logged(self, harness, caplog):
        writer = MemoryWriter(harness.config, BrokenPendingStore(), FakeEmbedding())
        writer.start()

        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(writer.flush(), 1.0)

        assert "待处理" in caplog.text
        assert "disk I/O error" in caplog.text

    async def test_chunking_error_keeps_messages_for_retry(self, harness, caplog):
        flaky = FlakyStore(harness.store, failures=99)
        embedding = FakeEmbedding()
        writer = MemoryWriter(harness.config, flaky, embedding)
        writer.start()

        with caplog.at_level(logging.ERROR):
            await record_many(writer, 100, 10)
            await asyncio.wait_for(writer.flush(), 1.0)

        assert flaky.chunk_attempts >= 1
        assert "分块失败" in caplog.text
        assert count_rows(harness.path, 100) == {
            "messages": 10,
            "chunks": 0,
            "memory_vectors": 0,
        }

        flaky.failures = 0
        await asyncio.wait_for(writer.close(), 1.0)
        writer.start()
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 1
        assert count_rows(harness.path, 100) == {
            "messages": 10,
            "chunks": 1,
            "memory_vectors": 1,
        }


class TestLostWakes:
    async def test_queue_full_wake_is_recovered_after_restart(self, harness_factory, caplog):
        first = await harness_factory()
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = first.writer(embedding, queue_size=1)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        with caplog.at_level(logging.WARNING):
            await record_many(writer, 200, 10)
            await record_many(writer, 300, 10)

        assert "记忆队列已满" in caplog.text

        embedding.gate.set()
        await asyncio.wait_for(writer.flush(), 1.0)

        assert count_rows(first.path, 100)["memory_vectors"] == 1
        stranded = [
            group_id
            for group_id in (200, 300)
            if count_rows(first.path, group_id)["memory_vectors"] == 0
        ]
        assert stranded, "队列容量为 1 时至少应有一个群的唤醒被丢弃"
        assert set(await first.store.pending_groups(10)) == set(stranded)
        await first.aclose()

        second = await harness_factory()
        working = FakeEmbedding()
        restarted = second.writer(working)
        restarted.start()
        await asyncio.wait_for(restarted.flush(), 1.0)

        for group_id in (100, 200, 300):
            assert count_rows(second.path, group_id) == {
                "messages": 10,
                "chunks": 1,
                "memory_vectors": 1,
            }
        assert len(working.calls) == len(stranded)
        assert await second.store.pending_chunks() == []
        assert await second.store.pending_groups(10) == []

    async def test_wake_queued_during_close_is_recovered_after_restart(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        await writer.record(make_message(100, "消息10", index=10))
        assert writer._queued_wakes == {100}

        close_task = asyncio.create_task(writer.close())
        await wait_until(lambda: writer._closing)
        embedding.gate.set()
        await asyncio.wait_for(close_task, 1.0)

        assert count_rows(harness.path, 100) == {
            "messages": 11,
            "chunks": 1,
            "memory_vectors": 1,
        }

        writer.start()
        await asyncio.wait_for(writer.flush(), 1.0)
        await record_many(writer, 100, 10, start=11)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 2
        assert count_rows(harness.path, 100) == {
            "messages": 21,
            "chunks": 2,
            "memory_vectors": 2,
        }
        assert writer._queued_wakes == set()
        assert await harness.store.pending_chunks() == []
        assert await harness.store.pending_groups(10) == []


class TestClearGroup:
    async def test_clear_during_inflight_embedding_leaves_no_vectors(self, harness):
        await seed_ready_chunk(harness.store, 200)
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        await harness.store.clear_group(100)
        embedding.gate.set()
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 1
        assert count_rows(harness.path, 100) == {
            "messages": 0,
            "chunks": 0,
            "memory_vectors": 0,
        }
        assert count_rows(harness.path, 200) == {
            "messages": 10,
            "chunks": 1,
            "memory_vectors": 1,
        }

    async def test_clear_before_writer_processes_wake_creates_nothing(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        await record_many(writer, 100, 10)
        await harness.store.clear_group(100)

        writer.start()
        await asyncio.wait_for(writer.flush(), 1.0)

        assert embedding.calls == []
        assert count_rows(harness.path, 100) == {
            "messages": 0,
            "chunks": 0,
            "memory_vectors": 0,
        }


class TestSupervision:
    async def test_unexpected_worker_exit_is_supervised(self, harness, caplog):
        embedding = FatalEmbedding()
        writer = harness.writer(embedding)
        writer.start()

        with caplog.at_level(logging.ERROR):
            await record_many(writer, 100, 10)
            await wait_until(lambda: not writer.running)

            assert "意外退出" in caplog.text
            assert "模拟致命错误" in caplog.text

            await asyncio.wait_for(writer.flush(timeout=1.0), 1.0)
            await asyncio.wait_for(writer.close(), 1.0)

            await writer.record(make_message(100, "消息10", index=10))

        assert embedding.calls == 1
        assert writer._queue.empty()
        assert len(await harness.store.recent_messages(100, 20)) == 11

        restarted = harness.writer(FakeEmbedding())
        restarted.start()
        await asyncio.wait_for(restarted.flush(), 1.0)

        assert count_rows(harness.path, 100) == {
            "messages": 11,
            "chunks": 1,
            "memory_vectors": 1,
        }

    async def test_cancelled_worker_is_supervised(self, harness, caplog):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        task = writer._task
        assert task is not None
        with caplog.at_level(logging.WARNING):
            task.cancel()
            await wait_until(lambda: not writer.running)
            assert "意外取消" in caplog.text

        await asyncio.wait_for(writer.flush(timeout=1.0), 1.0)
        await asyncio.wait_for(writer.close(), 1.0)

    async def test_close_suppresses_worker_cancellation(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        worker_task = writer._task
        assert worker_task is not None
        close_task = asyncio.create_task(writer.close())
        await asyncio.sleep(0)
        worker_task.cancel()

        await asyncio.wait_for(close_task, 1.0)

    async def test_close_propagates_caller_cancellation(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        close_task = asyncio.create_task(writer.close())
        await asyncio.sleep(0)
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        embedding.gate.set()
        await asyncio.wait_for(writer.close(), 1.0)
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY


class TestFlushTimeout:
    async def test_flush_timeout_raises_without_breaking_close(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        with pytest.raises(asyncio.TimeoutError):
            await writer.flush(timeout=0.1)

        embedding.gate.set()
        await asyncio.wait_for(writer.flush(timeout=1.0), 1.0)
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY
        await asyncio.wait_for(writer.close(), 1.0)

    async def test_flush_timeout_bounds_startup_recovery(self, harness):
        for index in range(10):
            await harness.store.add_message(make_message(100, f"消息{index}", index=index))
        pending = await harness.store.create_ready_chunk(100, 10)
        assert pending is not None
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        with pytest.raises(asyncio.TimeoutError):
            await writer.flush(timeout=0.1)

        embedding.gate.set()
        await asyncio.wait_for(writer.flush(timeout=1.0), 1.0)
        assert await harness.store.pending_chunks() == []

    async def test_concurrent_flushes_wait_for_the_same_work(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        first = asyncio.create_task(writer.flush())
        second = asyncio.create_task(writer.flush(timeout=1.0))
        for task in (first, second):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 0.1)

        embedding.gate.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1.0)
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY

    async def test_flush_returns_without_live_worker(self, harness):
        writer = harness.writer()
        writer.start()
        await asyncio.wait_for(writer.close(), 1.0)

        assert writer.running is False
        await asyncio.wait_for(writer.flush(timeout=0.5), 0.5)


class TestLifecycle:
    async def test_flush_waits_for_background_embedding(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        flush_task = asyncio.create_task(writer.flush())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(flush_task), 0.1)

        embedding.gate.set()
        await asyncio.wait_for(flush_task, 1.0)
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY

    async def test_close_stops_background_task_without_leftovers(self, harness):
        baseline = asyncio.all_tasks()
        writer = harness.writer()
        writer.start()
        await record_many(writer, 100, 9)
        await asyncio.wait_for(writer.flush(), 1.0)

        await asyncio.wait_for(writer.close(), 1.0)

        assert asyncio.all_tasks() - baseline == set()
        await asyncio.wait_for(writer.close(), 1.0)
        await asyncio.wait_for(writer.flush(), 1.0)

    async def test_start_is_idempotent(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()
        writer.start()

        await record_many(writer, 100, 10)
        await asyncio.wait_for(writer.flush(), 1.0)

        assert len(embedding.calls) == 1

    async def test_close_waits_for_inflight_embedding(self, harness):
        embedding = FakeEmbedding()
        embedding.gate = asyncio.Event()
        writer = harness.writer(embedding)
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        close_task = asyncio.create_task(writer.close())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(close_task), 0.1)

        embedding.gate.set()
        await asyncio.wait_for(close_task, 1.0)
        (chunk,) = await harness.store.search(100, VECTOR, 5)
        assert chunk.status is ChunkStatus.READY

    async def test_close_interrupts_backoff_sleep(self, harness):
        embedding = FakeEmbedding(failures=99)
        writer = harness.writer(embedding, retry_delays=(30.0,))
        writer.start()
        await record_many(writer, 100, 10)
        await asyncio.wait_for(embedding.entered.wait(), 1.0)

        await asyncio.wait_for(writer.close(), 1.0)

        (pending,) = await harness.store.pending_chunks()
        assert pending.status is ChunkStatus.PENDING

    async def test_record_after_close_is_ignored(self, harness):
        embedding = FakeEmbedding()
        writer = harness.writer(embedding)
        writer.start()
        await asyncio.wait_for(writer.close(), 1.0)

        await asyncio.wait_for(writer.record(make_message(100, "消息0")), 1.0)

        assert len(await harness.store.recent_messages(100, 10)) == 1
        assert embedding.calls == []
        assert writer._queue.empty()
