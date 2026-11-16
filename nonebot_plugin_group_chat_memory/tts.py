from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .config import DEFAULT_TTS_VOICE_ID, Config

TTS_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)
TTS_MAX_CHARS = 5_000
TTS_VOICE_IDS = frozenset({"stella", "tide"})
CLIENT_USER_AGENT = "nonebot-plugin-group-chat-memory"


class TTSError(RuntimeError):
    """语音合成失败；异常消息只包含原因或状态，不含密钥与响应正文。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class TTSClient:
    """DeepSeek 语音合成客户端，返回 WAV 音频字节。"""

    def __init__(
        self,
        config: Config,
        *,
        retry_delays: tuple[float, ...] = TTS_RETRY_DELAYS,
    ) -> None:
        self._config = config
        self._retry_delays = tuple(retry_delays)

    async def synthesize(self, text: str, *, voice_id: str | None = None) -> bytes:
        """把文本合成为 WAV；失败抛出 TTSError，瞬时错误按配置重试。"""
        content = text.strip()
        if not content:
            raise TTSError("没有可朗读的文本。")
        api_key = self._config.group_chat_memory_tts_api_key.get_secret_value().strip()
        if not api_key:
            raise TTSError("语音合成未配置。")
        payload = {
            "text": content[:TTS_MAX_CHARS],
            "voice_id": _normalize_voice_id(
                voice_id or self._config.group_chat_memory_tts_voice_id
            ),
            "format": "wav",
            "filter_markdown": True,
        }

        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                response = await self._post(payload, api_key)
            except TTSError as exc:
                if not exc.retryable or attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(self._retry_delays[attempt])
                continue
            audio = response.content
            if not audio:
                raise TTSError("语音合成返回了空音频。")
            return audio
        raise TTSError("语音合成重试次数已用尽。")

    async def _post(self, payload: dict[str, Any], api_key: str) -> httpx.Response:
        url = self._config.group_chat_memory_tts_base_url
        timeout = self._config.group_chat_memory_tts_timeout
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": CLIENT_USER_AGENT,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise TTSError("语音合成请求超时。", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise TTSError("语音合成请求失败，无法连接服务。", retryable=True) from exc
        if not 200 <= response.status_code < 300:
            retryable = response.status_code in {429, 503, 504} or response.status_code >= 500
            raise TTSError(
                f"语音合成服务返回 HTTP {response.status_code}。",
                status_code=response.status_code,
                retryable=retryable,
            )
        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("application/json"):
            raise TTSError(
                "语音合成服务返回了错误信息。",
                status_code=response.status_code,
            )
        return response


def _normalize_voice_id(value: str | None) -> str:
    voice_id = (value or "").strip().lower()
    return voice_id if voice_id in TTS_VOICE_IDS else DEFAULT_TTS_VOICE_ID
