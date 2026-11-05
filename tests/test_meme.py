from __future__ import annotations

import httpx
import pytest
import respx

from nonebot_plugin_group_chat_memory.meme import MemeProvider

API_URL = "https://meme.example/api"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_provider(clock: FakeClock | None = None, **overrides) -> MemeProvider:
    clock = clock or FakeClock()
    values = {
        "enabled": True,
        "api_url": API_URL,
        "api_id": "10023437",
        "api_key": "test-key",
        "clock": clock,
        "sleep": clock.sleep,
        "random_value": lambda: 0.0,
    }
    values.update(overrides)
    return MemeProvider(**values)


def test_keyword_strips_punctuation_and_truncates():
    assert MemeProvider.keyword("笑死，我也这么觉得！！123abc") == "笑死我也这么觉得12"


def test_keyword_returns_empty_for_punctuation_only():
    assert MemeProvider.keyword("！！！？？ ……") == ""


async def test_returns_url_from_results():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL).mock(
            return_value=httpx.Response(
                200, json={"code": 200, "res": ["https://img/1.png", "https://img/2.png"]}
            )
        )

        url = await make_provider().fetch("笑死我了")

    assert url == "https://img/1.png"
    request = route.calls[0].request
    assert request.url.params["type"] == "2"
    assert request.url.params["words"] == "笑死我了"
    assert request.url.params["limit"] == "6"


async def test_missing_credentials_returns_none_without_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL)

        assert await make_provider(api_key="").fetch("笑死") is None

    assert len(route.calls) == 0


async def test_disabled_provider_returns_none_without_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL)

        assert await make_provider(enabled=False).fetch("笑死") is None

    assert len(route.calls) == 0


async def test_empty_keyword_returns_none_without_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL)

        assert await make_provider().fetch("！！！") is None

    assert len(route.calls) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"code": 400, "msg": "调用频次过快"},
        {"code": 200, "res": []},
        {"code": 200},
    ],
)
async def test_unsuccessful_payload_returns_none(payload):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(return_value=httpx.Response(200, json=payload))

        assert await make_provider().fetch("笑死") is None


async def test_http_error_returns_none():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(side_effect=httpx.ConnectError("boom"))

        assert await make_provider().fetch("笑死") is None


async def test_cache_hit_skips_second_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(API_URL).mock(
            return_value=httpx.Response(200, json={"code": 200, "res": ["https://img/1.png"]})
        )
        provider = make_provider()

        first = await provider.fetch("笑死")
        second = await provider.fetch("笑死")

    assert first == second == "https://img/1.png"
    assert len(route.calls) == 1


async def test_throttle_waits_between_new_keywords():
    clock = FakeClock()
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(
            return_value=httpx.Response(200, json={"code": 200, "res": ["https://img/1.png"]})
        )
        provider = make_provider(clock)

        await provider.fetch("笑死")
        await provider.fetch("无语")

    assert clock.now == pytest.approx(6.5)


async def test_malformed_results_returns_none():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(
            return_value=httpx.Response(200, json={"code": 200, "res": {"a": "b"}})
        )

        assert await make_provider().fetch("笑死") is None


async def test_repeated_failures_log_once(caplog):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(API_URL).mock(
            return_value=httpx.Response(200, json={"code": 400, "msg": "调用频次过快"})
        )
        provider = make_provider()

        await provider.fetch("笑死")
        await provider.fetch("笑死")

    messages = [record.message for record in caplog.records if "表情包" in record.message]
    assert len(messages) == 1
