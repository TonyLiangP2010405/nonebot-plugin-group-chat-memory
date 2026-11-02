from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

import nonebot_plugin_group_chat_memory.store as store_module
from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.models import Message
from nonebot_plugin_group_chat_memory.store import MemoryStore


def make_config(tmp_path) -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / "fallback.db"),
        group_chat_memory_embedding_dimensions=4,
    )


def make_message(group_id: int, text: str, index: int = 0) -> Message:
    return Message(
        group_id=group_id,
        user_id=1000,
        display_name="Alice",
        text=text,
        created_at=datetime(2026, 11, 2, 8, index, tzinfo=timezone.utc),
    )


async def seed_chunk(store: MemoryStore, group_id: int, vector: list[float], prefix: str):
    await store.add_message(make_message(group_id, f"{prefix}-0", 0))
    await store.add_message(make_message(group_id, f"{prefix}-1", 1))
    chunk = await store.create_ready_chunk(group_id, 2)
    assert chunk is not None
    await store.complete_chunk(chunk.id, vector)
    return chunk


@pytest.fixture
async def fallback_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "_supports_load_extension", lambda: False)
    store = MemoryStore(make_config(tmp_path))
    await store.open()
    try:
        yield store
    finally:
        await store.close()


async def test_without_extension_loading_store_uses_blob_vectors(fallback_store, tmp_path):
    chunk = await seed_chunk(fallback_store, 100, [1.0, 0.0, 0.0, 0.0], "alpha")

    (found,) = await fallback_store.search(100, [1.0, 0.0, 0.0, 0.0], 5)

    assert found.id == chunk.id
    connection = sqlite3.connect(tmp_path / "fallback.db")
    try:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        blob_count = connection.execute("SELECT count(*) FROM memory_vector_blobs").fetchone()[0]
    finally:
        connection.close()
    assert "memory_vector_blobs" in names
    assert "memory_vectors" not in names
    assert blob_count == 1


async def test_blob_search_ranks_similarity_and_isolates_groups(fallback_store):
    close_chunk = await seed_chunk(fallback_store, 100, [1.0, 0.0, 0.0, 0.0], "close")
    far_chunk = await seed_chunk(fallback_store, 100, [0.0, 1.0, 0.0, 0.0], "far")
    await seed_chunk(fallback_store, 200, [1.0, 0.0, 0.0, 0.0], "other-group")

    chunks = await fallback_store.search(100, [1.0, 0.0, 0.0, 0.0], 5)

    assert [chunk.id for chunk in chunks] == [close_chunk.id, far_chunk.id]


async def test_blob_vectors_survive_restart_and_clear(fallback_store, tmp_path, monkeypatch):
    chunk = await seed_chunk(fallback_store, 100, [1.0, 0.0, 0.0, 0.0], "persisted")

    reopened = MemoryStore(make_config(tmp_path))
    await reopened.open()
    try:
        (found,) = await reopened.search(100, [1.0, 0.0, 0.0, 0.0], 5)
        assert found.id == chunk.id

        await reopened.clear_group(100)
        assert await reopened.search(100, [1.0, 0.0, 0.0, 0.0], 5) == []
        connection = sqlite3.connect(tmp_path / "fallback.db")
        try:
            count = connection.execute(
                "SELECT count(*) FROM memory_vector_blobs WHERE group_id = 100"
            ).fetchone()[0]
        finally:
            connection.close()
        assert count == 0
    finally:
        await reopened.close()


async def test_ready_chunks_without_blob_are_requeued_for_reembedding(
    fallback_store, tmp_path, monkeypatch
):
    chunk = await seed_chunk(fallback_store, 100, [1.0, 0.0, 0.0, 0.0], "legacy")
    await fallback_store.close()
    connection = sqlite3.connect(tmp_path / "fallback.db")
    try:
        connection.execute("DELETE FROM memory_vector_blobs WHERE chunk_id = ?", (chunk.id,))
        connection.commit()
    finally:
        connection.close()

    reopened = MemoryStore(make_config(tmp_path))
    await reopened.open()
    try:
        pending = await reopened.pending_chunks()
        assert [item.id for item in pending] == [chunk.id]
    finally:
        await reopened.close()
