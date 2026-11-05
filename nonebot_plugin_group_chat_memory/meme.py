from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

KEYWORD_PATTERN = re.compile(r"[^\w\u4e00-\u9fa5]")
KEYWORD_LENGTH = 10


class MemeProvider:
    """按关键词从接口盒子检索表情包，带全局节流与结果缓存。"""

    def __init__(
        self,
        enabled: bool,
        api_url: str,
        api_id: str,
        api_key: str,
        *,
        min_interval: float = 6.5,
        cache_ttl: float = 600.0,
        cache_size: int = 200,
        timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self._enabled = enabled
        self._api_url = api_url
        self._api_id = api_id
        self._api_key = api_key
        self._min_interval = min_interval
        self._cache_ttl = cache_ttl
        self._cache_size = cache_size
        self._timeout = timeout
        self._clock = clock
        self._sleep = sleep
        self._random_value = random_value
        self._lock = asyncio.Lock()
        self._next_at = 0.0
        self._cache: dict[str, tuple[float, str]] = {}

    @staticmethod
    def keyword(text: str) -> str:
        """去掉标点与空白，只留中英文和数字，截取前 10 个字符。"""
        return KEYWORD_PATTERN.sub("", text)[:KEYWORD_LENGTH]

    async def fetch(self, text: str) -> str | None:
        if not self._enabled or not self._api_id or not self._api_key:
            return None
        keyword = self.keyword(text)
        if not keyword:
            return None

        cached = self._cache.get(keyword)
        if cached is not None and self._clock() < cached[0]:
            return cached[1]

        await self._wait_turn()
        params = {
            "id": self._api_id,
            "key": self._api_key,
            "type": "2",
            "words": keyword,
            "page": "1",
            "limit": "6",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._api_url, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("表情包请求失败：%s", type(exc).__name__)
            return None

        if not isinstance(data, dict) or data.get("code") != 200:
            logger.warning("表情包接口返回异常：%s", data)
            return None
        results = data.get("res") or []
        if not results:
            return None

        index = min(int(self._random_value() * len(results)), len(results) - 1)
        url = str(results[index])
        self._store_cache(keyword, url)
        return url

    async def _wait_turn(self) -> None:
        """全局串行；两次请求之间至少间隔 min_interval 秒。"""
        async with self._lock:
            now = self._clock()
            if now < self._next_at:
                await self._sleep(self._next_at - now)
            self._next_at = self._clock() + self._min_interval

    def _store_cache(self, keyword: str, url: str) -> None:
        self._cache[keyword] = (self._clock() + self._cache_ttl, url)
        while len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))
