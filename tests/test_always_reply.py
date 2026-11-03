from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.models import Message
from nonebot_plugin_group_chat_memory.store import MemoryStore

DIMENSIONS = 4
BASE_TIME = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def make_config(tmp_path, filename: str = "always.db") -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / filename),
        group_chat_memory_embedding_dimensions=DIMENSIONS,
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
) -> Message:
    return Message(
        group_id=group_id,
        user_id=user_id,
        display_name=display_name,
        text=text,
        created_at=BASE_TIME + timedelta(minutes=index),
    )


class TestMembership:
    async def test_added_member_is_detected(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            assert await store.is_always_reply(20002, 1001) is False
            assert await store.add_always_reply(20002, 1001) is True
            assert await store.is_always_reply(20002, 1001) is True
        finally:
            await store.close()

    async def test_duplicate_add_reports_not_new(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_always_reply(20002, 1001)
            assert await store.add_always_reply(20002, 1001) is False
        finally:
            await store.close()

    async def test_membership_is_isolated_per_group(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_always_reply(20002, 1001)

            assert await store.is_always_reply(20003, 1001) is False
        finally:
            await store.close()

    async def test_remove_returns_whether_row_existed(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_always_reply(20002, 1001)

            assert await store.remove_always_reply(20002, 1001) is True
            assert await store.remove_always_reply(20002, 1001) is False
            assert await store.is_always_reply(20002, 1001) is False
        finally:
            await store.close()


class TestListing:
    async def test_list_returns_members_with_latest_display_name(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_message(
                make_message(20002, "早", user_id=1001, display_name="旧名字")
            )
            await store.add_message(
                make_message(20002, "早", index=1, user_id=1001, display_name="新名字")
            )
            await store.add_message(
                make_message(20002, "你好", index=2, user_id=1002, display_name="阿伟")
            )
            await store.add_always_reply(20002, 1001)
            await store.add_always_reply(20002, 1002)

            assert await store.list_always_reply(20002) == [
                (1001, "新名字"),
                (1002, "阿伟"),
            ]
        finally:
            await store.close()

    async def test_list_falls_back_to_stored_name(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_always_reply(20002, 1001, "手写名字")

            assert await store.list_always_reply(20002) == [(1001, "手写名字")]
        finally:
            await store.close()

    async def test_list_is_empty_by_default(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            assert await store.list_always_reply(20002) == []
        finally:
            await store.close()

    async def test_clear_removes_all_members(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_always_reply(20002, 1001)
            await store.add_always_reply(20002, 1002)

            assert await store.clear_always_reply(20002) == 2
            assert await store.list_always_reply(20002) == []
            assert await store.clear_always_reply(20002) == 0
        finally:
            await store.close()

    async def test_clear_group_keeps_always_reply_members(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_always_reply(20002, 1001)
            await store.add_message(make_message(20002, "会被清掉"))

            await store.clear_group(20002)

            assert await store.is_always_reply(20002, 1001) is True
        finally:
            await store.close()

    async def test_members_persist_across_reopen(self, tmp_path):
        config = make_config(tmp_path)
        store = await open_store(config)
        try:
            await store.add_always_reply(20002, 1001)
        finally:
            await store.close()

        reopened = await open_store(config)
        try:
            assert await reopened.is_always_reply(20002, 1001) is True
        finally:
            await reopened.close()


class TestGroupListing:
    async def test_groups_are_ordered_by_recent_activity(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.add_message(make_message(20002, "一", index=0))
            await store.add_message(make_message(20003, "二", index=5))
            await store.add_message(make_message(20003, "三", index=6))

            groups = await store.list_groups()

            assert [group_id for group_id, _, _ in groups] == [20003, 20002]
            assert [count for _, count, _ in groups] == [2, 1]
        finally:
            await store.close()

    async def test_group_without_messages_still_listed(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            await store.set_probability(20002, 10.0)

            assert await store.list_groups() == [(20002, 0, None)]
        finally:
            await store.close()

    async def test_unknown_database_has_no_groups(self, tmp_path):
        store = await open_store(make_config(tmp_path))
        try:
            assert await store.list_groups() == []
        finally:
            await store.close()


async def test_existing_database_gains_always_reply_table(tmp_path):
    path = tmp_path / "always.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE groups ("
            "group_id INTEGER PRIMARY KEY, "
            "probability REAL NOT NULL DEFAULT 0, "
            "updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO groups(group_id, probability, updated_at) "
            "VALUES (20002, 12.5, 'now')"
        )
        connection.commit()
    finally:
        connection.close()

    store = await open_store(make_config(tmp_path))
    try:
        assert await store.add_always_reply(20002, 1001) is True
        assert await store.is_always_reply(20002, 1001) is True
        assert await store.list_groups() == [(20002, 0, None)]
    finally:
        await store.close()


@pytest.mark.parametrize("user_id", [1001, 123456789])
async def test_common_qq_formats_are_accepted(tmp_path, user_id):
    store = await open_store(make_config(tmp_path))
    try:
        assert await store.add_always_reply(20002, user_id) is True

        assert await store.is_always_reply(20002, user_id) is True
    finally:
        await store.close()
