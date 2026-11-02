import json

import httpx
import pytest
import respx

from nonebot_plugin_group_chat_memory import api
from nonebot_plugin_group_chat_memory.api import (
    ChatAPIError,
    ChatClient,
    EmbeddingAPIError,
    EmbeddingClient,
)
from nonebot_plugin_group_chat_memory.config import Config, ConfigError
from nonebot_plugin_group_chat_memory.models import ChatContext, Chunk, ChunkStatus, Message

CHAT_BASE_URL = "https://chat.example/v1"
EMBED_BASE_URL = "https://embed.example/v1"
EMBEDDINGS_URL = f"{EMBED_BASE_URL}/embeddings"
CHAT_URL = f"{CHAT_BASE_URL}/chat/completions"
CHAT_KEY = "sk-chat-secret-123456"
EMBED_KEY = "sk-embed-secret-654321"
DIMENSIONS = 4


def make_config(**overrides) -> Config:
    values = {
        "group_chat_memory_chat_base_url": CHAT_BASE_URL,
        "group_chat_memory_chat_api_key": CHAT_KEY,
        "group_chat_memory_chat_model": "test-chat-model",
        "group_chat_memory_chat_timeout": 12.5,
        "group_chat_memory_embedding_base_url": EMBED_BASE_URL,
        "group_chat_memory_embedding_api_key": EMBED_KEY,
        "group_chat_memory_embedding_model": "test-embed-model",
        "group_chat_memory_embedding_dimensions": DIMENSIONS,
        "group_chat_memory_embedding_timeout": 3.5,
    }
    values.update(overrides)
    return Config(**values)


def embedding_response(vectors: list[list[float]], *, start_index: int = 0) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": start_index + offset, "embedding": vector}
                for offset, vector in enumerate(vectors)
            ]
        },
    )


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]},
    )


def make_context() -> ChatContext:
    return ChatContext(
        recent_messages=[
            Message(group_id=100, user_id=200, display_name="Alice", text="早上好"),
        ],
        long_term_chunks=[
            Chunk(
                id=7,
                group_id=100,
                text="群 100 聊天记录\n[2026-11-01 09:00] Alice: 昨天聊了电影",
                status=ChunkStatus.READY,
            ),
        ],
    )


@pytest.fixture
def config() -> Config:
    return make_config()


class TestEmbeddingClient:
    async def test_sends_batch_input_with_auth_and_model(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(EMBEDDINGS_URL).mock(
                return_value=embedding_response([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
            )

            vectors = await EmbeddingClient(config).embed(["你好", "再见"])

        assert vectors == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        assert len(route.calls) == 1
        request = route.calls[0].request
        assert request.headers["Authorization"] == f"Bearer {EMBED_KEY}"
        assert json.loads(request.content) == {
            "model": "test-embed-model",
            "input": ["你好", "再见"],
        }

    async def test_empty_input_returns_empty_without_request(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([]))

            assert await EmbeddingClient(config).embed([]) == []

        assert len(route.calls) == 0

    async def test_orders_vectors_by_response_index(self, config):
        response = httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 0.0, 0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                ]
            },
        )
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=response)

            vectors = await EmbeddingClient(config).embed(["第一", "第二"])

        assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    async def test_applies_configured_embedding_timeout(self, config, monkeypatch):
        timeouts: list[float] = []
        real_make_client = api._make_client

        def spy(timeout):
            timeouts.append(timeout)
            return real_make_client(timeout)

        monkeypatch.setattr(api, "_make_client", spy)
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([[0.1, 0.2, 0.3, 0.4]]))

            await EmbeddingClient(config).embed(["x"])

        assert timeouts == [3.5]

    async def test_embedding_timeout_is_independent_of_query_timeout(self, monkeypatch):
        config = make_config(group_chat_memory_query_timeout=0.25)
        timeouts: list[float] = []
        real_make_client = api._make_client

        def spy(timeout):
            timeouts.append(timeout)
            return real_make_client(timeout)

        monkeypatch.setattr(api, "_make_client", spy)
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([[0.1, 0.2, 0.3, 0.4]]))

            await EmbeddingClient(config).embed(["x"])

        assert timeouts == [3.5]

    @pytest.mark.parametrize("status", [400, 401, 500, 503])
    async def test_non_2xx_raises_sanitized_error(self, config, status):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(EMBEDDINGS_URL).mock(
                return_value=httpx.Response(status, json={"error": f"bad key {EMBED_KEY}"})
            )

            with pytest.raises(EmbeddingAPIError) as excinfo:
                await EmbeddingClient(config).embed(["x"])

        message = str(excinfo.value)
        assert str(status) in message
        assert EMBED_KEY not in message
        assert "sk-embed" not in message
        assert len(route.calls) == 1

    async def test_timeout_raises_without_retry(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(EMBEDDINGS_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

        assert len(route.calls) == 1

    async def test_connection_error_raises_embedding_error(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(side_effect=httpx.ConnectError("refused"))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_empty_data_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([]))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_missing_data_field_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json={}))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_malformed_json_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, content=b"not json"))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_vector_count_mismatch_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([[0.1, 0.2, 0.3, 0.4]]))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["a", "b"])

    async def test_wrong_dimensions_raise(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([[0.1, 0.2]]))

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_non_numeric_dimension_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(
                return_value=embedding_response([[0.1, "0.2", 0.3, 0.4]])  # type: ignore[list-item]
            )

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_non_finite_vector_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(EMBEDDINGS_URL).mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"data": [{"embedding": [NaN, 0.0, 0.0, 0.0]}]}',
                )
            )

            with pytest.raises(EmbeddingAPIError):
                await EmbeddingClient(config).embed(["x"])

    async def test_missing_config_raises_config_error_without_request(self):
        config = make_config(group_chat_memory_embedding_api_key="")
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(EMBEDDINGS_URL).mock(return_value=embedding_response([]))

            with pytest.raises(ConfigError):
                await EmbeddingClient(config).embed(["x"])

        assert len(route.calls) == 0


class TestChatClient:
    async def test_opencode_requests_identify_client_and_session(self):
        config = make_config(
            group_chat_memory_chat_base_url="https://opencode.ai/inference/go/openai/v1"
        )
        url = "https://opencode.ai/inference/go/openai/v1/chat/completions"
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(url).mock(return_value=chat_response("ok"))

            answer = await ChatClient(config).complete(
                make_context(), "在吗", session_id="group:100"
            )

        assert answer == "ok"
        request = route.calls[0].request
        assert request.headers["User-Agent"] == "nonebot-plugin-group-chat-memory/0.1.2"
        assert request.headers["x-opencode-session"] == "group:100"

    async def test_non_opencode_provider_does_not_receive_session_header(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(return_value=chat_response("ok"))

            await ChatClient(config).complete(make_context(), "在吗", session_id="group:100")

        assert "x-opencode-session" not in route.calls[0].request.headers
        assert route.calls[0].request.headers["User-Agent"].startswith(
            "nonebot-plugin-group-chat-memory/"
        )

    async def test_sends_context_and_returns_stripped_content(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(return_value=chat_response(" 你好呀 "))

            answer = await ChatClient(config).complete(make_context(), "在吗")

        assert answer == "你好呀"
        assert len(route.calls) == 1
        request = route.calls[0].request
        assert request.headers["Authorization"] == f"Bearer {CHAT_KEY}"
        body = json.loads(request.content)
        assert body["model"] == "test-chat-model"
        assert body["messages"][0]["role"] == "system"
        assert "群友" in body["messages"][0]["content"]
        assert "Bot" in body["messages"][0]["content"]
        assert body["messages"][-1]["role"] == "user"
        content = body["messages"][-1]["content"]
        assert "Alice" in content
        assert "昨天聊了电影" in content
        assert "在吗" in content

    async def test_applies_configured_chat_timeout(self, config, monkeypatch):
        timeouts: list[float] = []
        real_make_client = api._make_client

        def spy(timeout):
            timeouts.append(timeout)
            return real_make_client(timeout)

        monkeypatch.setattr(api, "_make_client", spy)
        with respx.mock(assert_all_called=False) as mock:
            mock.post(CHAT_URL).mock(return_value=chat_response("ok"))

            await ChatClient(config).complete(make_context(), "在吗")

        assert timeouts == [12.5]

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    async def test_retries_transient_status_then_succeeds(self, config, status):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(
                side_effect=[
                    httpx.Response(status, json={"error": "busy"}),
                    chat_response("ok"),
                ]
            )

            answer = await ChatClient(config, retry_delays=(0.0, 0.0)).complete(
                make_context(), "在吗"
            )

        assert answer == "ok"
        assert len(route.calls) == 2

    async def test_retries_timeout_then_succeeds(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(
                side_effect=[httpx.ReadTimeout("slow"), chat_response("ok")]
            )

            answer = await ChatClient(config, retry_delays=(0.0, 0.0)).complete(
                make_context(), "在吗"
            )

        assert answer == "ok"
        assert len(route.calls) == 2

    async def test_gives_up_after_limited_retries(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(
                return_value=httpx.Response(500, json={"error": "down"})
            )

            with pytest.raises(ChatAPIError):
                await ChatClient(config, retry_delays=(0.0, 0.0)).complete(make_context(), "在吗")

        assert len(route.calls) == 3

    async def test_retry_exhaustion_message_has_no_secret(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(CHAT_URL).mock(
                return_value=httpx.Response(500, json={"error": f"key {CHAT_KEY}"})
            )

            with pytest.raises(ChatAPIError) as excinfo:
                await ChatClient(config, retry_delays=(0.0,)).complete(make_context(), "在吗")

        message = str(excinfo.value)
        assert "500" in message
        assert CHAT_KEY not in message
        assert "sk-chat" not in message

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_client_errors_are_not_retried(self, config, status):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(
                return_value=httpx.Response(status, json={"error": f"key {CHAT_KEY}"})
            )

            with pytest.raises(ChatAPIError) as excinfo:
                await ChatClient(config, retry_delays=(0.0, 0.0)).complete(make_context(), "在吗")

        assert len(route.calls) == 1
        assert CHAT_KEY not in str(excinfo.value)

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": ["数组"]}}]},
        ],
    )
    async def test_malformed_response_raises_without_retry(self, config, payload):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(return_value=httpx.Response(200, json=payload))

            with pytest.raises(ChatAPIError):
                await ChatClient(config, retry_delays=(0.0, 0.0)).complete(make_context(), "在吗")

        assert len(route.calls) == 1

    async def test_malformed_json_raises(self, config):
        with respx.mock(assert_all_called=False) as mock:
            mock.post(CHAT_URL).mock(return_value=httpx.Response(200, content=b"<html>"))

            with pytest.raises(ChatAPIError):
                await ChatClient(config).complete(make_context(), "在吗")

    async def test_blank_current_text_is_rejected_without_request(self, config):
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(return_value=chat_response("ok"))

            with pytest.raises(ChatAPIError):
                await ChatClient(config).complete(make_context(), "   ")

        assert len(route.calls) == 0

    async def test_missing_config_raises_config_error_without_request(self):
        config = make_config(group_chat_memory_chat_api_key="")
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(CHAT_URL).mock(return_value=chat_response("ok"))

            with pytest.raises(ConfigError):
                await ChatClient(config).complete(make_context(), "在吗")

        assert len(route.calls) == 0
