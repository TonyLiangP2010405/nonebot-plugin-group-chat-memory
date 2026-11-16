import json

import httpx
import pytest
import respx

from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.tts import TTSClient, TTSError

TTS_URL = "https://api.deepseek.com/v1/tools/tts"


def make_client(*, key: str = "sk-tts", retry_delays: tuple[float, ...] = ()) -> TTSClient:
    config = Config(
        group_chat_memory_tts_api_key=key,
        group_chat_memory_tts_voice_id="stella",
        group_chat_memory_tts_base_url=TTS_URL,
    )
    return TTSClient(config, retry_delays=retry_delays)


@respx.mock
async def test_synthesize_posts_text_and_returns_wav_bytes():
    route = respx.post(TTS_URL).mock(
        return_value=httpx.Response(
            200, content=b"wav-bytes", headers={"content-type": "audio/wav"}
        )
    )

    audio = await make_client().synthesize("你好", voice_id="tide")

    assert audio == b"wav-bytes"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-tts"
    assert json.loads(request.content) == {
        "text": "你好",
        "voice_id": "tide",
        "format": "wav",
        "filter_markdown": True,
    }


@respx.mock
async def test_invalid_voice_id_falls_back_to_stella():
    route = respx.post(TTS_URL).mock(
        return_value=httpx.Response(200, content=b"wav", headers={"content-type": "audio/wav"})
    )

    await make_client().synthesize("你好", voice_id="unknown")

    assert json.loads(route.calls.last.request.content)["voice_id"] == "stella"


async def test_missing_key_raises_without_request():
    with pytest.raises(TTSError, match="未配置"):
        await make_client(key="").synthesize("你好")


async def test_empty_text_raises():
    with pytest.raises(TTSError, match="没有可朗读"):
        await make_client().synthesize("   ")


@respx.mock
async def test_400_is_not_retried():
    route = respx.post(TTS_URL).mock(return_value=httpx.Response(400, json={"error": "risk"}))

    with pytest.raises(TTSError) as excinfo:
        await make_client(retry_delays=(0, 0)).synthesize("你好")

    assert excinfo.value.status_code == 400
    assert route.call_count == 1


@respx.mock
async def test_503_is_retried_then_succeeds():
    route = respx.post(TTS_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, content=b"wav", headers={"content-type": "audio/wav"}),
        ]
    )

    audio = await make_client(retry_delays=(0,)).synthesize("你好")

    assert audio == b"wav"
    assert route.call_count == 2


@respx.mock
async def test_empty_audio_is_rejected():
    respx.post(TTS_URL).mock(
        return_value=httpx.Response(200, content=b"", headers={"content-type": "audio/wav"})
    )

    with pytest.raises(TTSError, match="空音频"):
        await make_client().synthesize("你好")


@respx.mock
async def test_json_error_body_is_rejected():
    respx.post(TTS_URL).mock(
        return_value=httpx.Response(
            200, json={"error": "boom"}, headers={"content-type": "application/json"}
        )
    )

    with pytest.raises(TTSError, match="错误信息"):
        await make_client().synthesize("你好")


@respx.mock
async def test_text_is_truncated_to_api_limit():
    route = respx.post(TTS_URL).mock(
        return_value=httpx.Response(200, content=b"wav", headers={"content-type": "audio/wav"})
    )

    await make_client().synthesize("字" * 5_001)

    assert len(json.loads(route.calls.last.request.content)["text"]) == 5_000
