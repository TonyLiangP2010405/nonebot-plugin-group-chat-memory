import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
import sqlite_vec

from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.models import Chunk, ChunkStatus, Message
from nonebot_plugin_group_chat_memory.store import MemoryStore

DIMENSIONS = 4
BASE_TIME = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def make_config(
    tmp_path,
    *,
    dimensions: int = DIMENSIONS,
    default_probability: float = 0.0,
    filename: str = "memory.db",
) -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / filename),
        group_chat_memory_embedding_dimensions=dimensions,
        group_chat_memory_default_probability=default_probability,
    )


async def open_store(config: Config) -> MemoryStore:
    store = MemoryStore(config)
    await store.open()
    return store


def make_message(
    group_id: int,
    text: str,
    *,
    index: int = 0,
    user_id: int = 1000,
    display_name: str = "Alice",
    is_bot: bool = False,
) -> Message:
    return Message(
        group_id=group_id,
        user_id=user_id,
        display_name=display_name,
        text=text,
        created_at=BASE_TIME + timedelta(minutes=index),
        is_bot=is_bot,
    )


async def seed_chunk(
    store: MemoryStore,
    group_id: int,
    texts: list[str],
    vector: list[float],
    *,
    start: int = 0,
    display_name: str = "Alice",
) -> Chunk:
    for offset, text in enumerate(texts):
        await store.add_message(
            make_message(group_id, text, index=start + offset, display_name=display_name)
        )
    chunk = await store.create_ready_chunk(group_id, len(texts))
    assert chunk is not None
    await store.complete_chunk(chunk.id, vector)
    return chunk


def table_counts(path, group_id: int) -> dict[str, int]:
    """通过独立连接直接统计各表的行数，避免经过会过滤孤儿行的 JOIN 查询。"""
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


@pytest.fixture
async def store(tmp_path):
    store = await open_store(make_config(tmp_path))
    try:
        yield store
    finally:
        await store.close()


class TestMessages:
    async def test_add_message_returns_increasing_ids(self, store):
        first = await store.add_message(make_message(100, "one", index=0))
        second = await store.add_message(make_message(100, "two", index=1))
        third = await store.add_message(make_message(100, "three", index=2))

        assert isinstance(first, int)
        assert first < second < third

    async def test_recent_messages_returns_chronological_order(self, store):
        for index, text in ((2, "third"), (0, "first"), (1, "second")):
            await store.add_message(make_message(100, text, index=index))

        messages = await store.recent_messages(100, 10)

        assert [message.text for message in messages] == ["first", "second", "third"]

    async def test_recent_messages_respects_limit(self, store):
        for index in range(5):
            await store.add_message(make_message(100, f"m{index}", index=index))

        messages = await store.recent_messages(100, 3)

        assert [message.text for message in messages] == ["m2", "m3", "m4"]

    async def test_recent_messages_are_group_isolated(self, store):
        await store.add_message(make_message(100, "group-a", index=0))
        await store.add_message(make_message(200, "group-b", index=1))

        assert [message.text for message in await store.recent_messages(100, 10)] == ["group-a"]
        assert [message.text for message in await store.recent_messages(200, 10)] == ["group-b"]

    async def test_recent_messages_round_trips_message_fields(self, store):
        created_at = BASE_TIME + timedelta(minutes=7, microseconds=123456)
        await store.add_message(
            Message(
                group_id=100,
                user_id=200,
                display_name="小助手",
                text="你好",
                created_at=created_at,
                is_bot=True,
            )
        )

        (message,) = await store.recent_messages(100, 10)

        assert message.group_id == 100
        assert message.user_id == 200
        assert message.display_name == "小助手"
        assert message.text == "你好"
        assert message.is_bot is True
        assert message.created_at == created_at
        assert message.created_at.tzinfo is not None

    async def test_recent_messages_zero_limit_and_unknown_group(self, store):
        await store.add_message(make_message(100, "hello"))

        assert await store.recent_messages(100, 0) == []
        assert await store.recent_messages(999, 10) == []

    async def test_concurrent_adds_keep_unique_ids(self, store):
        messages = [make_message(100, f"m{index}", index=index) for index in range(20)]

        ids = await asyncio.gather(*(store.add_message(message) for message in messages))

        assert len(set(ids)) == 20
        assert len(await store.recent_messages(100, 50)) == 20


class TestProbability:
    async def test_new_group_defaults_to_zero(self, store):
        assert await store.get_probability(100) == 0.0

    async def test_new_group_uses_configured_default(self, tmp_path):
        store = await open_store(make_config(tmp_path, default_probability=3.5))
        try:
            assert await store.get_probability(100) == 3.5
        finally:
            await store.close()

    async def test_set_and_get_probability(self, store):
        await store.set_probability(100, 5.5)

        assert await store.get_probability(100) == pytest.approx(5.5)
        assert await store.get_probability(200) == 0.0

    @pytest.mark.parametrize("value", [0, 0.0, 100, 100.0, 0.1])
    async def test_probability_boundaries_are_accepted(self, store, value):
        await store.set_probability(100, value)

        assert await store.get_probability(100) == pytest.approx(value)

    @pytest.mark.parametrize("value", [-0.1, 100.1, -1, 101])
    async def test_probability_out_of_range_is_rejected(self, store, value):
        await store.set_probability(100, 10.0)

        with pytest.raises(ValueError):
            await store.set_probability(100, value)

        assert await store.get_probability(100) == pytest.approx(10.0)

    async def test_probability_survives_restart(self, tmp_path):
        config = make_config(tmp_path)
        first = await open_store(config)
        try:
            await first.set_probability(100, 42.5)
        finally:
            await first.close()

        second = await open_store(config)
        try:
            assert await second.get_probability(100) == pytest.approx(42.5)
        finally:
            await second.close()


class TestChunks:
    async def test_insufficient_messages_returns_none(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"m{index}", index=index))

        assert await store.create_ready_chunk(100, 3) is None
        assert await store.pending_chunks() == []

    async def test_chunk_uses_earliest_unassigned_messages(self, store):
        for index in range(3):
            await store.add_message(make_message(100, f"m{index}", index=index))

        chunk = await store.create_ready_chunk(100, 2)

        assert chunk is not None
        assert chunk.group_id == 100
        assert chunk.status is ChunkStatus.PENDING
        assert "m0" in chunk.text
        assert "m1" in chunk.text
        assert "m2" not in chunk.text
        assert await store.create_ready_chunk(100, 2) is None

        await store.add_message(make_message(100, "m3", index=3))
        next_chunk = await store.create_ready_chunk(100, 2)
        assert next_chunk is not None
        assert "m2" in next_chunk.text
        assert "m3" in next_chunk.text
        assert next_chunk.id != chunk.id

    async def test_chunk_text_keeps_group_speakers_times_and_order(self, store):
        await store.add_message(
            make_message(100, "大家好", index=0, user_id=11, display_name="Alice")
        )
        await store.add_message(
            make_message(
                100,
                "你好呀",
                index=1,
                user_id=22,
                display_name="小助手",
                is_bot=True,
            )
        )

        chunk = await store.create_ready_chunk(100, 2)

        assert chunk is not None
        lines = chunk.text.splitlines()
        assert "群 100" in lines[0]
        assert "2026-11-02 08:00" in lines[0]
        assert "2026-11-02 08:01" in lines[0]
        assert lines[1] == "[2026-11-02 08:00] Alice: 大家好"
        assert lines[2] == "[2026-11-02 08:01] 小助手(bot): 你好呀"

    async def test_chunks_do_not_mix_groups(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"a{index}", index=index))
            await store.add_message(make_message(200, f"b{index}", index=index))

        assert await store.create_ready_chunk(100, 3) is None
        assert await store.create_ready_chunk(200, 3) is None

        chunk_a = await store.create_ready_chunk(100, 2)
        chunk_b = await store.create_ready_chunk(200, 2)

        assert chunk_a is not None
        assert chunk_b is not None
        assert chunk_a.group_id == 100
        assert chunk_b.group_id == 200
        assert "a0" in chunk_a.text
        assert "a1" in chunk_a.text
        assert "b0" not in chunk_a.text
        assert "b0" in chunk_b.text
        assert "a0" not in chunk_b.text

    async def test_pending_chunks_lists_created_chunks_for_all_groups(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"a{index}", index=index))
            await store.add_message(make_message(200, f"b{index}", index=index))

        chunk_a = await store.create_ready_chunk(100, 2)
        chunk_b = await store.create_ready_chunk(200, 2)

        pending = await store.pending_chunks()

        assert [chunk.id for chunk in pending] == [chunk_a.id, chunk_b.id]
        assert {chunk.group_id for chunk in pending} == {100, 200}
        assert all(chunk.status is ChunkStatus.PENDING for chunk in pending)
        assert pending[0].text == chunk_a.text

    async def test_pending_chunks_survive_restart(self, tmp_path):
        config = make_config(tmp_path)
        first = await open_store(config)
        try:
            for index in range(2):
                await first.add_message(make_message(100, f"m{index}", index=index))
            chunk = await first.create_ready_chunk(100, 2)
        finally:
            await first.close()

        second = await open_store(config)
        try:
            (pending,) = await second.pending_chunks()
            assert pending.id == chunk.id
            assert pending.group_id == 100
            assert pending.status is ChunkStatus.PENDING

            await second.complete_chunk(pending.id, [1.0, 0.0, 0.0, 0.0])
            assert await second.pending_chunks() == []
            assert (await second.search(100, [1.0, 0.0, 0.0, 0.0], 5))[0].id == pending.id
        finally:
            await second.close()

    async def test_complete_chunk_marks_ready_and_removes_from_pending(self, store):
        chunk = await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])

        assert await store.pending_chunks() == []
        (found,) = await store.search(100, [1.0, 0.0, 0.0, 0.0], 5)
        assert found.id == chunk.id
        assert found.status is ChunkStatus.READY
        assert found.text == chunk.text

    async def test_complete_chunk_replaces_existing_vector(self, store):
        chunk = await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])

        await store.complete_chunk(chunk.id, [0.0, 1.0, 0.0, 0.0])

        assert (await store.search(100, [0.0, 1.0, 0.0, 0.0], 5))[0].id == chunk.id
        assert (await store.search(100, [0.0, 1.0, 0.0, 0.0], 5))[0].status is ChunkStatus.READY

    async def test_complete_unknown_chunk_is_noop(self, store):
        await store.complete_chunk(999, [1.0, 0.0, 0.0, 0.0])

        assert await store.pending_chunks() == []
        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []

    async def test_complete_chunk_rejects_wrong_dimensions(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"m{index}", index=index))
        chunk = await store.create_ready_chunk(100, 2)

        with pytest.raises(ValueError):
            await store.complete_chunk(chunk.id, [1.0, 0.0])

        assert [pending.id for pending in await store.pending_chunks()] == [chunk.id]
        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []

    async def test_rejects_non_finite_vectors(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"m{index}", index=index))
        chunk = await store.create_ready_chunk(100, 2)

        with pytest.raises(ValueError):
            await store.complete_chunk(chunk.id, [float("nan"), 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            await store.search(100, [float("inf"), 0.0, 0.0, 0.0], 5)

        assert [pending.id for pending in await store.pending_chunks()] == [chunk.id]

    async def test_create_chunk_rejects_non_positive_size(self, store):
        with pytest.raises(ValueError):
            await store.create_ready_chunk(100, 0)


class TestSearch:
    async def test_returns_nearest_chunks_in_distance_order(self, store):
        farthest = await seed_chunk(store, 100, ["m0", "m1"], [0.0, 1.0, 0.0, 0.0], start=0)
        middle = await seed_chunk(store, 100, ["m2", "m3"], [0.6, 0.8, 0.0, 0.0], start=2)
        nearest = await seed_chunk(store, 100, ["m4", "m5"], [1.0, 0.0, 0.0, 0.0], start=4)

        found = await store.search(100, [1.0, 0.0, 0.0, 0.0], 5)

        assert [chunk.id for chunk in found] == [nearest.id, middle.id, farthest.id]

    async def test_uses_cosine_distance(self, store):
        orthogonal = await seed_chunk(store, 100, ["m0", "m1"], [0.0, 1.0, 0.0, 0.0], start=0)
        scaled = await seed_chunk(store, 100, ["m2", "m3"], [10.0, 0.0, 0.0, 0.0], start=2)

        found = await store.search(100, [1.0, 0.0, 0.0, 0.0], 5)

        assert [chunk.id for chunk in found] == [scaled.id, orthogonal.id]

    async def test_respects_limit(self, store):
        await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0], start=0)
        await seed_chunk(store, 100, ["m2", "m3"], [0.6, 0.8, 0.0, 0.0], start=2)

        assert len(await store.search(100, [1.0, 0.0, 0.0, 0.0], 1)) == 1
        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 0) == []

    async def test_identical_text_does_not_cross_groups(self, store):
        chunk_a = await seed_chunk(store, 100, ["同一个话题", "继续"], [1.0, 0.0, 0.0, 0.0])
        chunk_b = await seed_chunk(store, 200, ["同一个话题", "继续"], [0.0, 1.0, 0.0, 0.0])

        found_a = await store.search(100, [1.0, 0.0, 0.0, 0.0], 5)
        found_b = await store.search(200, [1.0, 0.0, 0.0, 0.0], 5)

        assert [chunk.id for chunk in found_a] == [chunk_a.id]
        assert [chunk.id for chunk in found_b] == [chunk_b.id]
        assert found_a[0].group_id == 100
        assert found_b[0].group_id == 200

    async def test_pending_chunks_are_not_searchable(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"m{index}", index=index))
        assert await store.create_ready_chunk(100, 2) is not None

        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []

    async def test_rejects_wrong_query_dimensions(self, store):
        await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            await store.search(100, [1.0, 0.0], 5)

    async def test_search_survives_restart(self, tmp_path):
        config = make_config(tmp_path)
        first = await open_store(config)
        try:
            chunk = await seed_chunk(first, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])
            await first.set_probability(100, 12.5)
        finally:
            await first.close()

        second = await open_store(config)
        try:
            (found,) = await second.search(100, [1.0, 0.0, 0.0, 0.0], 5)
            assert found.id == chunk.id
            assert [message.text for message in await second.recent_messages(100, 10)] == [
                "m0",
                "m1",
            ]
            assert await second.get_probability(100) == pytest.approx(12.5)
        finally:
            await second.close()

    async def test_uses_configured_dimensions(self, tmp_path):
        store = await open_store(make_config(tmp_path, dimensions=8))
        try:
            await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            with pytest.raises(ValueError):
                await store.search(100, [1.0, 0.0, 0.0, 0.0], 5)
        finally:
            await store.close()


class TestClearGroup:
    async def test_clear_removes_messages_chunks_and_vectors_but_keeps_probability(self, store):
        await store.set_probability(100, 42.5)
        await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])
        await store.add_message(make_message(100, "m2", index=2))

        await store.clear_group(100)

        assert await store.recent_messages(100, 10) == []
        assert await store.pending_chunks() == []
        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []
        assert await store.get_probability(100) == pytest.approx(42.5)

        await store.add_message(make_message(100, "fresh-0", index=0))
        assert await store.create_ready_chunk(100, 2) is None
        await store.add_message(make_message(100, "fresh-1", index=1))
        chunk = await store.create_ready_chunk(100, 2)
        assert chunk is not None
        assert "m0" not in chunk.text
        assert "fresh-0" in chunk.text

    async def test_clear_deletes_vector_and_chunk_rows_from_database(self, store, tmp_path):
        await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])
        await store.add_message(make_message(100, "m2", index=2))
        path = tmp_path / "memory.db"

        assert table_counts(path, 100) == {
            "messages": 3,
            "chunks": 1,
            "memory_vectors": 1,
        }

        await store.clear_group(100)

        assert table_counts(path, 100) == {
            "messages": 0,
            "chunks": 0,
            "memory_vectors": 0,
        }

    async def test_clear_only_affects_target_group(self, store, tmp_path):
        await store.set_probability(100, 10.0)
        await store.set_probability(200, 20.0)
        chunk_a = await seed_chunk(store, 100, ["a0", "a1"], [1.0, 0.0, 0.0, 0.0])
        chunk_b = await seed_chunk(store, 200, ["b0", "b1"], [0.0, 1.0, 0.0, 0.0])
        for index in range(2):
            await store.add_message(make_message(200, f"pending{index}", index=index + 2))
        pending_b = await store.create_ready_chunk(200, 2)

        await store.clear_group(100)

        path = tmp_path / "memory.db"
        assert table_counts(path, 100) == {
            "messages": 0,
            "chunks": 0,
            "memory_vectors": 0,
        }
        assert table_counts(path, 200) == {
            "messages": 4,
            "chunks": 2,
            "memory_vectors": 1,
        }
        assert await store.recent_messages(100, 10) == []
        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []
        assert [message.text for message in await store.recent_messages(200, 10)] == [
            "b0",
            "b1",
            "pending0",
            "pending1",
        ]
        assert [chunk.id for chunk in await store.pending_chunks()] == [pending_b.id]
        assert [chunk.id for chunk in await store.search(200, [0.0, 1.0, 0.0, 0.0], 5)] == [
            chunk_b.id
        ]
        assert await store.get_probability(200) == pytest.approx(20.0)
        assert chunk_a.id != chunk_b.id

    async def test_clear_ignores_unknown_group(self, store):
        await store.clear_group(999)

        assert await store.get_probability(999) == 0.0

    async def test_complete_after_clear_is_noop(self, store):
        for index in range(2):
            await store.add_message(make_message(100, f"m{index}", index=index))
        chunk = await store.create_ready_chunk(100, 2)

        await store.clear_group(100)
        await store.complete_chunk(chunk.id, [1.0, 0.0, 0.0, 0.0])

        assert await store.pending_chunks() == []
        assert await store.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []


class TestSchemaAndLifecycle:
    async def test_uses_wal_and_foreign_keys(self, store):
        db = store._db
        assert db is not None
        journal_mode = await (await db.execute("PRAGMA journal_mode")).fetchone()
        foreign_keys = await (await db.execute("PRAGMA foreign_keys")).fetchone()

        assert journal_mode[0].lower() == "wal"
        assert foreign_keys[0] == 1

    async def test_schema_declares_foreign_keys(self, tmp_path):
        path = tmp_path / "memory.db"
        store = await open_store(make_config(tmp_path))
        await store.close()

        connection = sqlite3.connect(path)
        try:
            referenced = {row[2] for row in connection.execute("PRAGMA foreign_key_list(messages)")}
            assert referenced == {"groups", "chunks"}
            assert {row[2] for row in connection.execute("PRAGMA foreign_key_list(chunks)")} == {
                "groups"
            }
        finally:
            connection.close()

    async def test_memory_vectors_is_a_real_sqlite_vec_table(self, tmp_path):
        path = tmp_path / "memory.db"
        store = await open_store(make_config(tmp_path))
        try:
            await seed_chunk(store, 100, ["m0", "m1"], [1.0, 0.0, 0.0, 0.0])
        finally:
            await store.close()

        connection = sqlite3.connect(path)
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        try:
            (sql,) = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
            ).fetchone()
            assert "vec0" in sql.lower()
            assert "float[4]" in sql.lower()
            assert "partition key" in sql.lower()
            assert connection.execute("SELECT vec_version()").fetchone()[0].startswith("v0.1")
            assert connection.execute("SELECT count(*) FROM memory_vectors").fetchone()[0] == 1
        finally:
            connection.close()

    async def test_open_creates_parent_directories(self, tmp_path):
        config = Config(
            group_chat_memory_db_path_override=str(tmp_path / "nested" / "data" / "memory.db"),
            group_chat_memory_embedding_dimensions=DIMENSIONS,
        )

        store = await open_store(config)
        try:
            await store.add_message(make_message(100, "hello"))
            assert await store.recent_messages(100, 5)
        finally:
            await store.close()

        assert (tmp_path / "nested" / "data" / "memory.db").exists()

    async def test_dimension_mismatch_is_reported(self, tmp_path):
        first = await open_store(make_config(tmp_path, dimensions=4))
        await first.close()

        with pytest.raises(RuntimeError):
            await open_store(make_config(tmp_path, dimensions=8))

    async def test_operations_require_open(self, tmp_path):
        store = MemoryStore(make_config(tmp_path))

        with pytest.raises(RuntimeError):
            await store.recent_messages(100, 5)

        await store.open()
        await store.close()
        with pytest.raises(RuntimeError):
            await store.get_probability(100)

    async def test_close_is_idempotent(self, tmp_path):
        store = await open_store(make_config(tmp_path))

        await store.close()
        await store.close()

    async def test_open_is_idempotent(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.open()
            await store.add_message(make_message(100, "hello"))
            assert len(await store.recent_messages(100, 5)) == 1
        finally:
            await store.close()
