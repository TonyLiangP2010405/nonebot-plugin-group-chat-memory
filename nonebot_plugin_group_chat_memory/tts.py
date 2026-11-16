from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path
from typing import Any

import httpx

from .config import DEFAULT_TTS_VOICE_ID, Config

TTS_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)
TTS_MAX_CHARS = 5_000
TTS_VOICE_IDS = frozenset({"stella", "tide"})
CLIENT_USER_AGENT = "nonebot-plugin-group-chat-memory"
QWEN_MAX_PROMPT = 3_000
QWEN_MAX_REFERENCE_BYTES = 10 * 1024 * 1024
QWEN_REFERENCE_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}


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


class QwenTTSClient:
    """阿里云百炼 qwen-audio-3.1-tts-next 客户端，支持参考音频音色复刻。"""

    def __init__(
        self,
        config: Config,
        *,
        retry_delays: tuple[float, ...] = TTS_RETRY_DELAYS,
    ) -> None:
        self._config = config
        self._retry_delays = tuple(retry_delays)

    async def synthesize(self, text: str, *, reference: str | None = None) -> bytes:
        """用参考音频复刻音色并合成 WAV；失败抛出 TTSError。"""
        content = text.strip()
        if not content:
            raise TTSError("没有可朗读的文本。")
        ref = (reference or self._config.group_chat_memory_qwen_reference).strip()
        if not ref:
            raise TTSError("阿里云语音参考音频未配置。")
        return await self._request(
            {
                "model": self._config.group_chat_memory_qwen_model,
                "input": {
                    "text_prompt": _qwen_prompt(content),
                    "references": [_reference_payload(ref)],
                    "format": "wav",
                    "sample_rate": 24_000,
                    "channels": 1,
                    "volume": 50,
                },
            }
        )

    async def generate_sample(self, text: str, *, description: str) -> bytes:
        """不使用参考音频，按文字描述生成一段固定音色样本。"""
        content = text.strip()
        if not content:
            raise TTSError("没有可朗读的文本。")
        return await self._request(
            {
                "model": self._config.group_chat_memory_qwen_model,
                "input": {
                    "text_prompt": _qwen_sample_prompt(description, content),
                    "format": "wav",
                    "sample_rate": 24_000,
                    "channels": 1,
                    "volume": 50,
                },
            }
        )

    async def _request(self, payload: dict[str, Any]) -> bytes:
        api_key = self._config.group_chat_memory_qwen_api_key.get_secret_value().strip()
        if not api_key:
            raise TTSError("阿里云语音未配置 API Key。")
        workspace = self._config.group_chat_memory_qwen_workspace_id.strip()
        if not workspace:
            raise TTSError("阿里云百炼业务空间未配置。")
        endpoint = _qwen_endpoint(workspace, self._config.group_chat_memory_qwen_region)

        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                response = await self._post(endpoint, payload, api_key)
            except TTSError as exc:
                if not exc.retryable or attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(self._retry_delays[attempt])
                continue
            return await self._audio_from_response(response)
        raise TTSError("阿里云语音合成重试次数已用尽。")

    async def _post(
        self, endpoint: str, payload: dict[str, Any], api_key: str
    ) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": CLIENT_USER_AGENT,
        }
        timeout = self._config.group_chat_memory_qwen_timeout
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise TTSError("阿里云语音合成请求超时。", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise TTSError("阿里云语音合成请求失败，无法连接服务。", retryable=True) from exc
        if not 200 <= response.status_code < 300:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise TTSError(
                f"阿里云语音合成服务返回 HTTP {response.status_code}。",
                status_code=response.status_code,
                retryable=retryable,
            )
        return response

    async def _audio_from_response(self, response: httpx.Response) -> bytes:
        try:
            payload = response.json()
        except ValueError as exc:
            raise TTSError("阿里云语音合成响应不是合法 JSON。") from exc
        output = payload.get("output") if isinstance(payload, dict) else None
        audio = output.get("audio") if isinstance(output, dict) else None
        if not isinstance(audio, dict):
            raise TTSError("阿里云语音合成响应缺少音频信息。")

        data = audio.get("data")
        if isinstance(data, str) and data.strip():
            encoded = data.split(",", 1)[-1] if data.startswith("data:") else data
            try:
                return base64.b64decode(encoded)
            except ValueError as exc:
                raise TTSError("阿里云语音合成返回了无效的音频数据。") from exc

        url = audio.get("url")
        if not isinstance(url, str) or not url.strip():
            raise TTSError("阿里云语音合成响应缺少音频地址。")
        try:
            async with httpx.AsyncClient(
                timeout=self._config.group_chat_memory_qwen_timeout
            ) as client:
                download = await client.get(url)
        except httpx.TimeoutException as exc:
            raise TTSError("下载阿里云语音音频超时。", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise TTSError("下载阿里云语音音频失败。", retryable=True) from exc
        if not 200 <= download.status_code < 300:
            raise TTSError(
                f"下载阿里云语音音频返回 HTTP {download.status_code}。",
                status_code=download.status_code,
                retryable=download.status_code == 429 or download.status_code >= 500,
            )
        if not download.content:
            raise TTSError("阿里云语音合成返回了空音频。")
        return download.content


def _qwen_endpoint(workspace: str, region: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", workspace):
        raise TTSError("阿里云百炼业务空间 ID 格式不正确。")
    if not re.fullmatch(r"[a-z0-9-]+", region):
        raise TTSError("阿里云地域格式不正确。")
    return (
        f"https://{workspace}.{region}.maas.aliyuncs.com"
        "/api/v1/services/audio/tts/SpeechSynthesizer"
    )


def _qwen_prompt(text: str) -> str:
    prefix = "@voice1 用自然、口语化的语气说："
    suffix = "。只输出 @voice1 的人声，不要添加音效、背景音乐或其他旁白。"
    budget = QWEN_MAX_PROMPT - len(prefix) - len(suffix) - 2
    return f"{prefix}“{text[:budget]}”{suffix}"


def _qwen_sample_prompt(description: str, text: str) -> str:
    suffix = "只输出这个声音的人声，不要添加音效、背景音乐或其他旁白。"
    budget = QWEN_MAX_PROMPT - len(description) - len(suffix) - 10
    return f"{description} 请说：“{text[:budget]}”。{suffix}"


def _reference_payload(reference: str) -> dict[str, str]:
    if reference.lower().startswith(("http://", "https://")):
        return {"audio_url": reference}
    path = Path(reference).expanduser()
    if not path.is_file():
        raise TTSError("参考音频文件不存在。")
    mime = QWEN_REFERENCE_MIME.get(path.suffix.lower())
    if mime is None:
        raise TTSError("参考音频只支持 WAV、MP3、OGG Opus。")
    if path.stat().st_size > QWEN_MAX_REFERENCE_BYTES:
        raise TTSError("参考音频不能超过 10 MB。")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"audio_data": f"data:{mime};base64,{encoded}"}
