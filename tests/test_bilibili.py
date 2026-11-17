from __future__ import annotations

import httpx
import pytest
import respx

from nonebot_plugin_group_chat_memory.bilibili import (
    HOME_URL,
    NAV_URL,
    SEARCH_URL,
    WBI_SEARCH_URL,
    BilibiliClient,
    BilibiliError,
    BilibiliVideo,
    build_decision_messages,
    decide_video,
    parse_decision,
    sign_params,
    video_card_text,
)

SEARCH_PAYLOAD = {
    "code": 0,
    "data": {
        "result": [
            {
                "type": "video",
                "bvid": "BV1g9sW6DEC9",
                "author": "一半月X",
                "title": '<em class="keyword">派蒙</em>本体能带到大世界？【原神】',
                "pic": "//i1.hdslb.com/bfs/archive/cover.jpg",
                "play": 37033,
                "like": "2167",
                "review": 39,
            },
            {"type": "live", "title": "直播"},
            {"type": "video", "bvid": "", "title": "没有 bvid"},
            {"type": "video", "bvid": "BV2", "title": "&lt;第二&gt;个"},
        ]
    },
}

NAV_PAYLOAD = {
    "code": -101,
    "data": {
        "wbi_img": {
            "img_url": "https://i0.hdslb.com/bfs/wbi/7cd084941338484aae1ad9425b84077c.png",
            "sub_url": "https://i0.hdslb.com/bfs/wbi/4932caff0ff746eab6f01bf08b70ac45.png",
        }
    },
}

VIDEO = BilibiliVideo(
    title="派蒙本体能带到大世界？【原神】",
    author="一半月X",
    url="https://www.bilibili.com/video/BV1g9sW6DEC9",
    cover="https://i1.hdslb.com/bfs/archive/cover.jpg",
    play=37033,
    like=2167,
    review=39,
)


def test_card_text_keeps_group_share_style() -> None:
    assert video_card_text(VIDEO) == (
        "🎬 派蒙本体能带到大世界？【原神】\n"
        "👤 UP: 一半月X\n"
        "▶ 播放: 37,033 👍 点赞: 2,167 💬 评论: 39\n"
        "🔗 https://www.bilibili.com/video/BV1g9sW6DEC9"
    )


def test_parse_result_strips_tags_and_normalizes_fields() -> None:
    from nonebot_plugin_group_chat_memory.bilibili import _parse_payload

    videos = _parse_payload(httpx.Response(200, json=SEARCH_PAYLOAD))

    assert [video.title for video in videos] == [
        "派蒙本体能带到大世界？【原神】",
        "<第二>个",
    ]
    first = videos[0]
    assert first.url == "https://www.bilibili.com/video/BV1g9sW6DEC9"
    assert first.cover == "https://i1.hdslb.com/bfs/archive/cover.jpg"
    assert (first.play, first.like, first.review) == (37033, 2167, 39)


def test_parse_payload_reports_risk_control_code() -> None:
    from nonebot_plugin_group_chat_memory.bilibili import _parse_payload

    with pytest.raises(BilibiliError):
        _parse_payload(httpx.Response(200, json={"code": -412, "message": "请求被拦截"}))


def test_sign_params_filters_unsafe_chars_and_signs() -> None:
    signed = sign_params(
        {"keyword": "派蒙(原神)!", "search_type": "video"}, "mixin-key", timestamp=1700000000
    )

    assert signed["wts"] == "1700000000"
    assert signed["keyword"] == "派蒙原神"
    assert len(signed["w_rid"]) == 32
    assert signed["w_rid"] == sign_params(
        {"keyword": "派蒙(原神)!", "search_type": "video"}, "mixin-key", timestamp=1700000000
    )["w_rid"]


def test_wbi_key_uses_shuffle_table() -> None:
    from nonebot_plugin_group_chat_memory.bilibili import _mixin_key

    key = _mixin_key("a" * 64, "b" * 64)

    assert len(key) == 32
    assert set(key) <= {"a", "b"}
    with pytest.raises(BilibiliError):
        _mixin_key("short", "keys")


def test_parse_decision_reads_json_from_prose() -> None:
    cue = parse_decision('好的\n{"search": true, "keyword": " 原神 派蒙 "}\n就这样')

    assert cue is not None
    assert cue.keyword == "原神 派蒙"
    assert parse_decision('{"search": false}') is None
    assert parse_decision('{"search": true, "keyword": ""}') is None
    assert parse_decision("不想搜") is None


def test_build_decision_messages_contains_reply_context() -> None:
    messages = build_decision_messages("来个视频", "给")

    assert messages[0]["role"] == "system"
    assert "B站" in messages[0]["content"]
    assert messages[1]["content"] == "群友说：来个视频\n馒头准备回复：给"


class FakeChatClient:
    def __init__(self, content: str = '{"search": true, "keyword": "派蒙"}', error=None):
        self.content = content
        self.error = error
        self.messages: list[dict[str, str]] = []

    async def complete_prompt(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.content


async def test_decide_video_returns_keyword() -> None:
    client = FakeChatClient()

    cue = await decide_video(client, "来个视频", "给")  # type: ignore[arg-type]

    assert cue is not None
    assert cue.keyword == "派蒙"
    assert client.messages[1]["content"].startswith("群友说：来个视频")


async def test_decide_video_swallows_api_errors() -> None:
    client = FakeChatClient(error=RuntimeError("boom"))

    assert await decide_video(client, "来个视频", "给") is None  # type: ignore[arg-type]


@respx.mock
async def test_search_returns_first_video() -> None:
    route = respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SEARCH_PAYLOAD)
    )

    video = await BilibiliClient().search("派蒙")

    assert video is not None
    assert video.title == "派蒙本体能带到大世界？【原神】"
    assert route.calls[0].request.url.params["keyword"] == "派蒙"


@respx.mock
async def test_search_skips_blank_keyword_without_request() -> None:
    route = respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SEARCH_PAYLOAD)
    )

    assert await BilibiliClient().search("   ") is None
    assert route.call_count == 0


@respx.mock
async def test_search_returns_none_when_nothing_matches() -> None:
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {}})
    )
    respx.get(HOME_URL).mock(return_value=httpx.Response(200, text="home"))
    respx.get(NAV_URL).mock(return_value=httpx.Response(200, json=NAV_PAYLOAD))
    respx.get(WBI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"result": []}})
    )

    assert await BilibiliClient().search("没有这个视频") is None


@respx.mock
async def test_search_falls_back_to_wbi_when_plain_is_blocked() -> None:
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"code": -412, "message": "请求被拦截"})
    )
    respx.get(HOME_URL).mock(
        return_value=httpx.Response(
            200, text="home", headers={"set-cookie": "buvid3=test-buvid; Path=/"}
        )
    )
    respx.get(NAV_URL).mock(return_value=httpx.Response(200, json=NAV_PAYLOAD))
    wbi_route = respx.get(WBI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SEARCH_PAYLOAD)
    )

    video = await BilibiliClient().search("派蒙")

    assert video is not None
    assert video.url.endswith("BV1g9sW6DEC9")
    request = wbi_route.calls[0].request
    assert request.url.params["w_rid"]
    assert request.url.params["keyword"] == "派蒙"
    assert "buvid3=test-buvid" in request.headers["cookie"]


@respx.mock
async def test_search_raises_when_both_endpoints_fail() -> None:
    respx.get(SEARCH_URL).mock(return_value=httpx.Response(500))
    respx.get(HOME_URL).mock(return_value=httpx.Response(500))
    respx.get(NAV_URL).mock(return_value=httpx.Response(200, json={"code": 0}))

    with pytest.raises(BilibiliError):
        await BilibiliClient().search("派蒙")
