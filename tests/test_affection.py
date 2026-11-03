from __future__ import annotations

from types import SimpleNamespace

import pytest

from nonebot_plugin_group_chat_memory.affection import AffectionLeaderboard


class FakeService:
    def __init__(self, user_ids=("1001", "1002"), error: Exception | None = None):
        self.user_ids = user_ids
        self.error = error
        self.calls: list[str] = []

    async def ranking(self, group_id: str):
        self.calls.append(group_id)
        if self.error is not None:
            raise self.error
        return [
            SimpleNamespace(rank=index, profile=SimpleNamespace(user_id=user_id))
            for index, user_id in enumerate(self.user_ids, start=1)
        ]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_leaderboard(
    service, *, ttl: float = 30.0
) -> tuple[AffectionLeaderboard, Clock]:
    clock = Clock()
    leaderboard = AffectionLeaderboard(
        ttl=ttl, clock=clock, service_loader=lambda: service
    )
    return leaderboard, clock


class TestMembership:
    async def test_member_is_detected(self):
        leaderboard, _ = make_leaderboard(FakeService(("1001", "1002")))

        assert await leaderboard.is_member(20002, 1001) is True

    async def test_absent_user_is_not_member(self):
        leaderboard, _ = make_leaderboard(FakeService(("1001",)))

        assert await leaderboard.is_member(20002, 9999) is False

    async def test_ranking_receives_group_id_as_string(self):
        service = FakeService(("1001",))
        leaderboard, _ = make_leaderboard(service)

        await leaderboard.is_member(20002, 1001)

        assert service.calls == ["20002"]

    async def test_invalid_user_ids_are_skipped(self):
        service = FakeService()
        service.user_ids = ("1001", "not-a-number", None)
        leaderboard, _ = make_leaderboard(service)

        assert await leaderboard.is_member(20002, 1001) is True
        assert await leaderboard.is_member(20002, 0) is False


class TestFallbacks:
    async def test_missing_plugin_is_treated_as_not_member(self):
        leaderboard, _ = make_leaderboard(None)

        assert await leaderboard.is_member(20002, 1001) is False

    async def test_ranking_error_is_treated_as_not_member(self):
        leaderboard, _ = make_leaderboard(FakeService(error=RuntimeError("boom")))

        assert await leaderboard.is_member(20002, 1001) is False

    async def test_ranking_error_is_logged_once_per_cache_window(self, monkeypatch):
        from nonebot_plugin_group_chat_memory import affection

        warnings: list[str] = []
        monkeypatch.setattr(
            affection.logger, "warning", lambda *args: warnings.append(args[0])
        )
        service = FakeService(error=RuntimeError("boom"))
        leaderboard, _ = make_leaderboard(service)

        await leaderboard.is_member(20002, 1001)
        await leaderboard.is_member(20002, 1002)

        assert len(service.calls) == 1
        assert warnings == ["读取馒头好感榜失败：%s"]


class TestCache:
    async def test_repeated_queries_reuse_cached_ranking(self):
        service = FakeService(("1001",))
        leaderboard, _ = make_leaderboard(service)

        await leaderboard.is_member(20002, 1001)
        await leaderboard.is_member(20002, 1002)

        assert service.calls == ["20002"]

    async def test_cache_expires_after_ttl(self):
        service = FakeService(("1001",))
        leaderboard, clock = make_leaderboard(service, ttl=30.0)

        await leaderboard.is_member(20002, 1001)
        clock.now = 29.9
        await leaderboard.is_member(20002, 1001)
        clock.now = 30.0
        await leaderboard.is_member(20002, 1001)

        assert service.calls == ["20002", "20002"]

    async def test_groups_are_cached_separately(self):
        service = FakeService(("1001",))
        leaderboard, _ = make_leaderboard(service)

        await leaderboard.is_member(20002, 1001)
        await leaderboard.is_member(20003, 1001)

        assert service.calls == ["20002", "20003"]

    async def test_clear_forces_refresh(self):
        service = FakeService(("1001",))
        leaderboard, _ = make_leaderboard(service)

        await leaderboard.is_member(20002, 1001)
        leaderboard.clear()
        await leaderboard.is_member(20002, 1001)

        assert service.calls == ["20002", "20002"]


class TestServiceLoader:
    async def test_plugin_lookup_requires_loaded_plugin(self, monkeypatch):
        import sys

        from nonebot_plugin_group_chat_memory import affection

        monkeypatch.setattr(affection, "get_plugin", lambda _name: None)
        monkeypatch.delitem(sys.modules, affection.AFFECTION_PLUGIN_ID, raising=False)

        assert affection._load_service() is None

    async def test_plugin_lookup_returns_service_attribute(self, monkeypatch):
        from nonebot_plugin_group_chat_memory import affection

        service = object()
        plugin = SimpleNamespace(module=SimpleNamespace(service=service))
        monkeypatch.setattr(affection, "get_plugin", lambda _name: plugin)

        assert affection._load_service() is service


@pytest.mark.parametrize("user_id", [1001, 1002])
async def test_leaderboard_accepts_all_ranked_members(user_id):
    leaderboard, _ = make_leaderboard(FakeService(("1001", "1002")))

    assert await leaderboard.is_member(20002, user_id) is True


class TestAffectionSettings:
    def make_config(self, tmp_path, filename: str = "affection.db"):
        from nonebot_plugin_group_chat_memory.config import Config

        return Config(
            group_chat_memory_db_path_override=str(tmp_path / filename),
            group_chat_memory_embedding_dimensions=4,
        )

    async def open_store(self, config):
        from nonebot_plugin_group_chat_memory.store import MemoryStore

        store = MemoryStore(config)
        await store.open()
        return store

    async def test_defaults_are_enabled_and_thirty_percent(self, tmp_path):
        store = await self.open_store(self.make_config(tmp_path))
        try:
            assert await store.get_affection_enabled(20002) is True
            assert await store.get_affection_probability(20002) == pytest.approx(30.0)
        finally:
            await store.close()

    async def test_settings_persist_across_reopen(self, tmp_path):
        config = self.make_config(tmp_path)
        store = await self.open_store(config)
        try:
            await store.set_affection_enabled(20002, False)
            await store.set_affection_probability(20002, 12.5)
        finally:
            await store.close()

        reopened = await self.open_store(config)
        try:
            assert await reopened.get_affection_enabled(20002) is False
            assert await reopened.get_affection_probability(20002) == pytest.approx(12.5)
        finally:
            await reopened.close()

    async def test_settings_are_isolated_per_group(self, tmp_path):
        store = await self.open_store(self.make_config(tmp_path))
        try:
            await store.set_affection_enabled(20002, False)
            await store.set_affection_probability(20002, 5.0)

            assert await store.get_affection_enabled(20003) is True
            assert await store.get_affection_probability(20003) == pytest.approx(30.0)
        finally:
            await store.close()

    @pytest.mark.parametrize("value", [-1, 100.1, 101])
    async def test_rejects_probability_outside_range(self, tmp_path, value):
        store = await self.open_store(self.make_config(tmp_path))
        try:
            with pytest.raises(ValueError, match="0 到 100"):
                await store.set_affection_probability(20002, value)
        finally:
            await store.close()

    async def test_clear_group_keeps_affection_settings(self, tmp_path):
        store = await self.open_store(self.make_config(tmp_path))
        try:
            await store.set_affection_enabled(20002, False)
            await store.set_affection_probability(20002, 5.0)

            await store.clear_group(20002)

            assert await store.get_affection_enabled(20002) is False
            assert await store.get_affection_probability(20002) == pytest.approx(5.0)
        finally:
            await store.close()

    async def test_existing_database_gains_default_columns(self, tmp_path):
        import sqlite3

        path = tmp_path / "affection.db"
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

        store = await self.open_store(self.make_config(tmp_path))
        try:
            assert await store.get_affection_enabled(20002) is True
            assert await store.get_affection_probability(20002) == pytest.approx(30.0)
            assert await store.get_probability(20002) == pytest.approx(12.5)
        finally:
            await store.close()
