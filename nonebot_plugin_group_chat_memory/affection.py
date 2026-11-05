from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import Any

from nonebot import get_plugin

logger = logging.getLogger(__name__)

AFFECTION_PLUGIN_ID = "nonebot_plugin_mantou_affection"


def _load_service() -> Any | None:
    """读取已加载的馒头好感插件服务；未安装或未加载时返回 None。"""
    plugin = get_plugin(AFFECTION_PLUGIN_ID)
    module = getattr(plugin, "module", None) or sys.modules.get(AFFECTION_PLUGIN_ID)
    return getattr(module, "service", None)


class AffectionLeaderboard:
    """查询本群好感榜成员，带短缓存；不可用时视为不在榜上。"""

    def __init__(
        self,
        *,
        ttl: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        service_loader: Callable[[], Any | None] = _load_service,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._service_loader = service_loader
        self._cache: dict[int, tuple[float, tuple[int, ...], frozenset[int]]] = {}

    async def is_member(self, group_id: int, user_id: int) -> bool:
        """该用户是否在本群好感榜上；插件不可用时返回 False。"""
        _, members = await self._ranking(group_id)
        return user_id in members

    async def top_members(self, group_id: int, limit: int = 3) -> frozenset[int]:
        """本群好感榜前 `limit` 名；插件不可用时返回空集合。"""
        ordered, _ = await self._ranking(group_id)
        return frozenset(ordered[:limit])

    async def is_top_member(self, group_id: int, user_id: int, limit: int = 3) -> bool:
        """该用户是否在本群好感榜前 `limit` 名内。"""
        return user_id in await self.top_members(group_id, limit)

    async def _ranking(self, group_id: int) -> tuple[tuple[int, ...], frozenset[int]]:
        now = self._clock()
        cached = self._cache.get(group_id)
        if cached is not None and now < cached[0]:
            return cached[1], cached[2]
        ordered = await self._fetch(group_id)
        members = frozenset(ordered)
        self._cache[group_id] = (now + self._ttl, ordered, members)
        return ordered, members

    async def _fetch(self, group_id: int) -> tuple[int, ...]:
        service = self._service_loader()
        if service is None:
            return ()
        try:
            entries = await service.ranking(str(group_id))
        except Exception as exc:
            logger.warning("读取馒头好感榜失败：%s", type(exc).__name__)
            return ()
        rows: list[tuple[int, int]] = []
        for index, entry in enumerate(entries, start=1):
            try:
                user_id = int(entry.profile.user_id)
            except (AttributeError, TypeError, ValueError):
                continue
            rank = getattr(entry, "rank", None)
            rows.append((rank if isinstance(rank, int) else index, user_id))
        rows.sort(key=lambda row: row[0])
        return tuple(user_id for _, user_id in rows)

    def clear(self) -> None:
        """清空缓存，好感榜数据变化后可用。"""
        self._cache.clear()
