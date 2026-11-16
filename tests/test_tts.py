import base64
import json

import httpx
import pytest
import respx

from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.tts import QwenTTSClient, TTSClient, TTSError

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


QWEN_URL = (
    "https://ws-test.cn-beijing.maas.aliyuncs.com"
    "/api/v1/services/audio/tts/SpeechSynthesizer"
)
QWEN_AUDIO_URL = "https://cdn.example/qwen.wav"


def make_qwen_client(
    *,
    reference: str = "https://example.com/reference.wav",
    key: str = "sk-qwen",
    workspace: str = "ws-test",
    retry_delays: tuple[float, ...] = (),
) -> QwenTTSClient:
    config = Config(
        group_chat_memory_qwen_api_key=key,
        group_chat_memory_qwen_workspace_id=workspace,
        group_chat_memory_qwen_region="cn-beijing",
        group_chat_memory_qwen_reference=reference,
        group_chat_memory_tts_timeout=5,
    )
    return QwenTTSClient(config, retry_delays=retry_delays)


@respx.mock
async def test_qwen_clones_reference_and_downloads_audio():
    post = respx.post(QWEN_URL).mock(
        return_value=httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}})
    )
    respx.get(QWEN_AUDIO_URL).mock(
        return_value=httpx.Response(200, content=b"qwen-wav")
    )

    audio = await make_qwen_client().synthesize("你好，我是馒头")

    assert audio == b"qwen-wav"
    request = post.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-qwen"
    payload = json.loads(request.content)
    assert payload["model"] == "qwen-audio-3.1-tts-next"
    assert payload["input"]["format"] == "wav"
    assert payload["input"]["channels"] == 1
    assert payload["input"]["references"] == [
        {"audio_url": "https://example.com/reference.wav"}
    ]
    assert "@voice1" in payload["input"]["text_prompt"]
    assert "你好，我是馒头" in payload["input"]["text_prompt"]


@respx.mock
async def test_qwen_local_reference_is_sent_as_data_uri(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-bytes")
    post = respx.post(QWEN_URL).mock(
        return_value=httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}})
    )
    respx.get(QWEN_AUDIO_URL).mock(
        return_value=httpx.Response(200, content=b"qwen-wav")
    )

    await make_qwen_client(reference=str(reference)).synthesize("你好")

    payload = json.loads(post.calls.last.request.content)
    audio_data = payload["input"]["references"][0]["audio_data"]
    assert audio_data.startswith("data:audio/wav;base64,")
    assert base64.b64decode(audio_data.split(",", 1)[1]) == b"reference-bytes"


@respx.mock
async def test_qwen_accepts_inline_base64_audio():
    encoded = base64.b64encode(b"inline-audio").decode()
    respx.post(QWEN_URL).mock(
        return_value=httpx.Response(
            200, json={"output": {"audio": {"data": f"data:audio/wav;base64,{encoded}"}}}
        )
    )

    audio = await make_qwen_client().synthesize("你好")

    assert audio == b"inline-audio"


async def test_qwen_requires_key_and_reference():
    with pytest.raises(TTSError, match="API Key"):
        await make_qwen_client(key="").synthesize("你好")
    with pytest.raises(TTSError, match="参考音频"):
        await make_qwen_client(reference="").synthesize("你好")


@respx.mock
async def test_qwen_without_workspace_uses_qianwen_platform_endpoint():
    platform_url = (
        "https://maas.qianwenaiapi.com"
        "/api/v1/services/audio/tts/SpeechSynthesizer"
    )
    post = respx.post(platform_url).mock(
        return_value=httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}})
    )
    respx.get(QWEN_AUDIO_URL).mock(
        return_value=httpx.Response(200, content=b"platform-wav")
    )

    audio = await make_qwen_client(workspace="").synthesize("你好")

    assert audio == b"platform-wav"
    assert post.call_count == 1


@respx.mock
async def test_qwen_retries_429_then_succeeds():
    route = respx.post(QWEN_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}}),
        ]
    )
    respx.get(QWEN_AUDIO_URL).mock(
        return_value=httpx.Response(200, content=b"qwen-wav")
    )

    audio = await make_qwen_client(retry_delays=(0,)).synthesize("你好")

    assert audio == b"qwen-wav"
    assert route.call_count == 2


@respx.mock
async def test_qwen_download_failure_is_reported():
    respx.post(QWEN_URL).mock(
        return_value=httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}})
    )
    respx.get(QWEN_AUDIO_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(TTSError, match="HTTP 404"):
        await make_qwen_client().synthesize("你好")


@respx.mock
async def test_qwen_prompt_stays_within_limit():
    post = respx.post(QWEN_URL).mock(
        return_value=httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}})
    )
    respx.get(QWEN_AUDIO_URL).mock(
        return_value=httpx.Response(200, content=b"qwen-wav")
    )

    await make_qwen_client().synthesize("字" * 5_000)

    prompt = json.loads(post.calls.last.request.content)["input"]["text_prompt"]
    assert len(prompt) <= 3_000


@respx.mock
async def test_qwen_generates_sample_without_reference():
    post = respx.post(QWEN_URL).mock(
        return_value=httpx.Response(200, json={"output": {"audio": {"url": QWEN_AUDIO_URL}}})
    )
    respx.get(QWEN_AUDIO_URL).mock(
        return_value=httpx.Response(200, content=b"mantou-wav")
    )

    audio = await make_qwen_client(reference="").generate_sample(
        "确实，今天这局有点抽象。", description="一位偏萝莉的年轻少女音。"
    )

    assert audio == b"mantou-wav"
    payload = json.loads(post.calls.last.request.content)
    assert "references" not in payload["input"]
    assert "偏萝莉" in payload["input"]["text_prompt"]
    assert "确实，今天这局有点抽象。" in payload["input"]["text_prompt"]
