from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.store import MemoryStore

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def make_config(tmp_path, **overrides) -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / "rate-limit.db"),
        group_chat_memory_embedding_dimensions=4,
        **overrides,
    )


async def open_store(config: Config) -> MemoryStore:
    store = MemoryStore(config)
    await store.open()
    return store


@pytest.fixture
async def store(tmp_path):
    store = await open_store(make_config(tmp_path))
    try:
        yield store
    finally:
        await store.close()


async def test_default_limit_is_ten_and_eleventh_request_is_denied(store):
    assert await store.get_hourly_limit(100) == 10

    allowed = [await store.consume_request_slot(100, NOW) for _ in range(11)]

    assert allowed == [True] * 10 + [False]
    assert await store.get_hour_usage(100, NOW) == 10


async def test_rolling_window_releases_expired_requests(store):
    for _ in range(10):
        assert await store.consume_request_slot(100, NOW) is True

    assert await store.consume_request_slot(100, NOW + timedelta(seconds=3599)) is False
    assert await store.consume_request_slot(100, NOW + timedelta(seconds=3601)) is True
    assert await store.get_hour_usage(100, NOW + timedelta(seconds=3601)) == 1


async def test_concurrent_requests_cannot_exceed_limit(store):
    results = await asyncio.gather(
        *(store.consume_request_slot(100, NOW) for _ in range(20))
    )

    assert sum(results) == 10
    assert await store.get_hour_usage(100, NOW) == 10


async def test_limits_are_group_isolated(store):
    for _ in range(10):
        await store.consume_request_slot(100, NOW)

    assert await store.consume_request_slot(100, NOW) is False
    assert await store.consume_request_slot(200, NOW) is True


async def test_limit_survives_restart(tmp_path):
    config = make_config(tmp_path)
    first = await open_store(config)
    await first.set_hourly_limit(100, 3)
    await first.consume_request_slot(100, NOW)
    await first.close()

    reopened = await open_store(config)
    try:
        assert await reopened.get_hourly_limit(100) == 3
        assert await reopened.get_hour_usage(100, NOW) == 1
    finally:
        await reopened.close()


async def test_clear_group_deletes_usage_but_keeps_limit_and_probability(store):
    await store.set_probability(100, 12.5)
    await store.set_hourly_limit(100, 5)
    for _ in range(3):
        await store.consume_request_slot(100, NOW)

    await store.clear_group(100)

    assert await store.get_hour_usage(100, NOW) == 0
    assert await store.get_hourly_limit(100) == 5
    assert await store.get_probability(100) == pytest.approx(12.5)


@pytest.mark.parametrize("value", [0, -1, 10001])
async def test_rejects_limit_outside_allowed_range(store, value):
    with pytest.raises(ValueError, match="1 到 10000"):
        await store.set_hourly_limit(100, value)


async def test_opening_existing_database_adds_default_limit_column(tmp_path):
    import sqlite3

    path = tmp_path / "rate-limit.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE groups ("
            "group_id INTEGER PRIMARY KEY, "
            "probability REAL NOT NULL DEFAULT 0, "
            "updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO groups(group_id, probability, updated_at) VALUES (100, 12.5, 'now')"
        )
        connection.commit()
    finally:
        connection.close()

    store = await open_store(make_config(tmp_path))
    try:
        assert await store.get_hourly_limit(100) == 10
        assert await store.get_probability(100) == pytest.approx(12.5)
        assert await store.consume_request_slot(100, NOW) is True
    finally:
        await store.close()
