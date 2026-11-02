from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx

from .config import Config
from .context import build_chat_messages
from .models import ChatContext

CHAT_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0)


class ModelAPIError(RuntimeError):
    """模型 API 调用失败；异常消息只包含状态或原因，不含密钥与响应正文。"""


class EmbeddingAPIError(ModelAPIError):
    """Embedding 请求失败或响应不可用。"""


class ChatAPIError(ModelAPIError):
    """聊天请求失败或响应不可用。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _make_client(timeout: float) -> httpx.AsyncClient:
    """为单次请求创建客户端，使聊天与 Embedding 各自使用独立超时。"""
    return httpx.AsyncClient(timeout=timeout)


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path}"


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key.strip()}"}


class EmbeddingClient:
    """OpenAI 兼容 `/embeddings` 客户端，用于批量向量化。"""

    def __init__(self, config: Config) -> None:
        self._config = config

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化 texts，返回与输入顺序一致的向量；失败时抛出 EmbeddingAPIError。"""
        if not texts:
            return []
        self._config.require_embedding_config()
        payload = {"model": self._config.group_chat_memory_embedding_model, "input": list(texts)}
        response = await self._post(payload)
        return self._parse(response, len(texts))

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        url = _endpoint(self._config.group_chat_memory_embedding_base_url, "embeddings")
        headers = _auth_headers(self._config.group_chat_memory_embedding_api_key.get_secret_value())
        timeout = self._config.group_chat_memory_embedding_timeout
        try:
            async with _make_client(timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise EmbeddingAPIError("Embedding 请求超时。") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingAPIError("Embedding 请求失败，无法连接服务。") from exc
        if not 200 <= response.status_code < 300:
            raise EmbeddingAPIError(f"Embedding 服务返回 HTTP {response.status_code}。")
        return response

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingAPIError("Embedding 响应不是合法 JSON。") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            raise EmbeddingAPIError("Embedding 响应缺少向量数据。")
        if len(data) != expected:
            raise EmbeddingAPIError(
                f"Embedding 响应返回 {len(data)} 条向量，与请求的 {expected} 条不一致。"
            )

        ordered = sorted(
            enumerate(data),
            key=lambda item: _index_of(item[1], item[0]),
        )
        dimensions = self._config.group_chat_memory_embedding_dimensions
        return [_parse_vector(item, dimensions) for _, item in ordered]


class ChatClient:
    """OpenAI 兼容 `/chat/completions` 客户端，带有限重试。"""

    def __init__(
        self,
        config: Config,
        *,
        retry_delays: tuple[float, ...] = CHAT_RETRY_DELAYS,
    ) -> None:
        self._config = config
        self._retry_delays = tuple(retry_delays)

    async def complete(self, context: ChatContext, current_text: str) -> str:
        """按上下文请求一条回复；瞬时失败按配置次数重试，最终失败抛出 ChatAPIError。"""
        self._config.require_chat_config()
        if not current_text.strip():
            raise ChatAPIError("当前消息为空，无法请求聊天模型。")
        payload = {
            "model": self._config.group_chat_memory_chat_model,
            "messages": build_chat_messages(self._config, context, current_text),
        }

        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                response = await self._post(payload)
            except ChatAPIError as exc:
                if not exc.retryable or attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(self._retry_delays[attempt])
                continue
            return self._parse(response)
        raise ChatAPIError("聊天请求重试次数已用尽。")

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        url = _endpoint(self._config.group_chat_memory_chat_base_url, "chat/completions")
        headers = _auth_headers(self._config.group_chat_memory_chat_api_key.get_secret_value())
        timeout = self._config.group_chat_memory_chat_timeout
        try:
            async with _make_client(timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ChatAPIError("聊天请求超时。", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ChatAPIError("聊天请求失败，无法连接服务。", retryable=True) from exc
        if not 200 <= response.status_code < 300:
            raise ChatAPIError(
                f"聊天服务返回 HTTP {response.status_code}。",
                retryable=response.status_code == 429 or response.status_code >= 500,
            )
        return response

    @staticmethod
    def _parse(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ChatAPIError("聊天响应不是合法 JSON。") from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ChatAPIError("聊天响应缺少 choices。")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ChatAPIError("聊天响应缺少回复内容。")
        return content.strip()


def _index_of(item: Any, fallback: int) -> int:
    index = item.get("index") if isinstance(item, dict) else None
    if isinstance(index, int) and not isinstance(index, bool):
        return index
    return fallback


def _parse_vector(item: Any, dimensions: int) -> list[float]:
    vector = item.get("embedding") if isinstance(item, dict) else None
    if not isinstance(vector, list):
        raise EmbeddingAPIError("Embedding 响应缺少向量。")
    if len(vector) != dimensions:
        raise EmbeddingAPIError(f"Embedding 向量维度为 {len(vector)}，配置要求 {dimensions}。")
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EmbeddingAPIError("Embedding 向量包含非数值。")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingAPIError("Embedding 向量包含非有限数值。")
        values.append(number)
    return values
