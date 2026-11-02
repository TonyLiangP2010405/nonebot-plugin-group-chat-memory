import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from nonebot_plugin_group_chat_memory.api import EmbeddingAPIError
from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.context import ContextBuilder, build_chat_messages
from nonebot_plugin_group_chat_memory.models import ChatContext, Chunk, ChunkStatus, Message
from nonebot_plugin_group_chat_memory.store import MemoryStore

DIMENSIONS = 4
BASE_TIME = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)
QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]


def make_config(tmp_path, **overrides) -> Config:
    values = {
        "group_chat_memory_db_path_override": str(tmp_path / "memory.db"),
        "group_chat_memory_embedding_dimensions": DIMENSIONS,
        "group_chat_memory_recent_messages": 3,
        "group_chat_memory_long_term_memories": 5,
        "group_chat_memory_context_max_chars": 6000,
    }
    values.update(overrides)
    return Config(**values)


def make_message(
    text: str,
    *,
    index: int = 0,
    group_id: int = 100,
    user_id: int = 1000,
    display_name: str = "Alice",
    is_bot: bool = False,
) -> Message:
    return Message(
        group_id=group_id,
        user_id=user_id,
        display_name=display_name,
        text=text,
        created_at=BASE_TIME + timedelta(minutes=index),
        is_bot=is_bot,
    )


def make_chunk(chunk_id: int, text: str, *, group_id: int = 100) -> Chunk:
    return Chunk(id=chunk_id, group_id=group_id, text=text, status=ChunkStatus.READY)


class FakeEmbedding:
    def __init__(self, vector: list[float] | None = None, error: Exception | None = None) -> None:
        self._vector = vector or QUERY_VECTOR
        self._error = error
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._error is not None:
            raise self._error
        return [list(self._vector) for _ in texts]


async def seed_chunk(
    store: MemoryStore,
    group_id: int,
    texts: list[str],
    vector: list[float],
    *,
    start: int = 0,
) -> Chunk:
    for offset, text in enumerate(texts):
        await store.add_message(make_message(text, index=start + offset, group_id=group_id))
    chunk = await store.create_ready_chunk(group_id, len(texts))
    assert chunk is not None
    await store.complete_chunk(chunk.id, vector)
    return chunk


@pytest.fixture
async def store(tmp_path):
    store = MemoryStore(make_config(tmp_path))
    await store.open()
    try:
        yield store
    finally:
        await store.close()


class TestContextBuilder:
    async def test_recent_messages_respect_configured_limit(self, store, tmp_path):
        for index in range(5):
            await store.add_message(make_message(f"m{index}", index=index))

        context = await ContextBuilder(make_config(tmp_path), store, FakeEmbedding()).build(
            100, "现在聊什么"
        )

        assert [message.text for message in context.recent_messages] == ["m2", "m3", "m4"]
        assert context.long_term_chunks == []

    async def test_returns_at_most_configured_long_term_chunks(self, store, tmp_path):
        for index in range(6):
            await seed_chunk(
                store, 100, [f"c{index}a", f"c{index}b"], QUERY_VECTOR, start=index * 2
            )

        context = await ContextBuilder(make_config(tmp_path), store, FakeEmbedding()).build(
            100, "查询"
        )

        assert len(context.long_term_chunks) == 5

    async def test_searches_with_query_text_and_embedding(self, store, tmp_path):
        near = await seed_chunk(store, 100, ["最近的记忆", "继续"], [1.0, 0.0, 0.0, 0.0])
        far = await seed_chunk(store, 100, ["无关的记忆", "继续"], [0.0, 1.0, 0.0, 0.0], start=2)
        embedding = FakeEmbedding([1.0, 0.0, 0.0, 0.0])

        context = await ContextBuilder(make_config(tmp_path), store, embedding).build(
            100, "查询内容"
        )

        assert embedding.calls == [["查询内容"]]
        assert [chunk.id for chunk in context.long_term_chunks] == [near.id, far.id]

    async def test_long_term_search_is_group_isolated(self, store, tmp_path):
        await seed_chunk(store, 200, ["群二记忆", "继续"], QUERY_VECTOR)
        await store.add_message(make_message("群一消息", group_id=100))

        context = await ContextBuilder(make_config(tmp_path), store, FakeEmbedding()).build(
            100, "查询"
        )

        assert context.long_term_chunks == []
        assert [message.text for message in context.recent_messages] == ["群一消息"]

    @pytest.mark.parametrize(
        "error",
        [
            EmbeddingAPIError("Embedding 服务返回 HTTP 500。"),
            EmbeddingAPIError("Embedding 请求超时。"),
            ValueError("向量维度为 2，配置要求 4。"),
        ],
    )
    async def test_embedding_failure_degrades_to_recent_messages(self, store, tmp_path, error):
        await store.add_message(make_message("hello"))
        await seed_chunk(store, 100, ["旧记忆", "继续"], QUERY_VECTOR, start=1)

        context = await ContextBuilder(
            make_config(tmp_path), store, FakeEmbedding(error=error)
        ).build(100, "查询")

        assert [message.text for message in context.recent_messages] == ["hello", "旧记忆", "继续"]
        assert context.long_term_chunks == []

    async def test_search_failure_degrades_to_recent_messages(self, store, tmp_path):
        await store.add_message(make_message("hello"))

        class FailingSearchStore:
            def __init__(self, inner: MemoryStore) -> None:
                self._inner = inner

            async def recent_messages(self, group_id: int, limit: int):
                return await self._inner.recent_messages(group_id, limit)

            async def search(self, group_id: int, vector: list[float], limit: int):
                raise RuntimeError("database error")

        context = await ContextBuilder(
            make_config(tmp_path),
            FailingSearchStore(store),
            FakeEmbedding(),  # type: ignore[arg-type]
        ).build(100, "查询")

        assert [message.text for message in context.recent_messages] == ["hello"]
        assert context.long_term_chunks == []

    async def test_long_term_lookup_is_skipped_when_disabled(self, store, tmp_path):
        embedding = FakeEmbedding()

        context = await ContextBuilder(
            make_config(tmp_path, group_chat_memory_long_term_memories=0), store, embedding
        ).build(100, "查询")

        assert embedding.calls == []
        assert context.long_term_chunks == []

    async def test_blank_query_skips_embedding(self, store, tmp_path):
        embedding = FakeEmbedding()

        context = await ContextBuilder(make_config(tmp_path), store, embedding).build(100, "   ")

        assert embedding.calls == []
        assert context.long_term_chunks == []

    async def test_reads_recent_messages_and_embeds_concurrently(self, tmp_path):
        recent_started = asyncio.Event()
        embed_started = asyncio.Event()
        searched: list[list[float]] = []

        class GatedStore:
            async def recent_messages(self, group_id: int, limit: int):
                recent_started.set()
                await asyncio.wait_for(embed_started.wait(), timeout=1.0)
                return [make_message("hello")]

            async def search(self, group_id: int, vector: list[float], limit: int):
                searched.append(list(vector))
                return []

        class GatedEmbedding:
            async def embed(self, texts: list[str]):
                embed_started.set()
                await asyncio.wait_for(recent_started.wait(), timeout=1.0)
                return [list(QUERY_VECTOR)]

        context = await ContextBuilder(
            make_config(tmp_path),
            GatedStore(),
            GatedEmbedding(),  # type: ignore[arg-type]
        ).build(100, "查询")

        assert searched == [QUERY_VECTOR]
        assert [message.text for message in context.recent_messages] == ["hello"]


class TestBuildChatMessages:
    def test_system_prompt_is_configurable_and_labels_speakers(self):
        config = Config(group_chat_memory_system_prompt="你是测试助手。")

        messages = build_chat_messages(config, ChatContext(), "在吗")

        assert messages[0]["role"] == "system"
        assert "你是测试助手。" in messages[0]["content"]
        assert "群友" in messages[0]["content"]
        assert "Bot" in messages[0]["content"]

    def test_orders_long_term_recent_and_current_input(self):
        context = ChatContext(
            recent_messages=[make_message("最近消息")],
            long_term_chunks=[make_chunk(1, "长期片段内容")],
        )

        messages = build_chat_messages(Config(), context, "当前输入")

        assert [message["role"] for message in messages] == ["system", "user"]
        content = messages[1]["content"]
        assert content.index("长期片段内容") < content.index("最近消息")
        assert content.index("最近消息") < content.index("当前输入")

    def test_recent_messages_distinguish_bot_and_members(self):
        context = ChatContext(
            recent_messages=[
                make_message("大家好", index=0, display_name="Alice"),
                make_message("你好", index=1, display_name="小助手", is_bot=True),
            ]
        )

        content = build_chat_messages(Config(), context, "在吗")[1]["content"]

        assert "[2026-11-02 08:00] 群友 Alice: 大家好" in content
        assert "[2026-11-02 08:01] Bot 小助手: 你好" in content

    def test_renders_at_most_configured_chunks(self):
        context = ChatContext(
            long_term_chunks=[make_chunk(index, f"片段{index}") for index in range(7)]
        )

        content = build_chat_messages(Config(), context, "在吗")[1]["content"]

        for index in range(5):
            assert f"片段{index}" in content
        assert "片段5" not in content
        assert "片段6" not in content

    def test_current_input_is_never_truncated(self):
        config = Config(group_chat_memory_context_max_chars=40)
        current_text = "很长" * 200
        context = ChatContext(
            recent_messages=[make_message("旧消息")],
            long_term_chunks=[make_chunk(1, "旧记忆")],
        )

        content = build_chat_messages(config, context, current_text)[1]["content"]

        assert current_text in content
        assert "旧消息" not in content
        assert "旧记忆" not in content

    def test_recent_messages_take_priority_over_long_term_memory(self):
        config = Config(group_chat_memory_context_max_chars=120)
        context = ChatContext(
            recent_messages=[make_message("最近的话")],
            long_term_chunks=[make_chunk(1, "很长的记忆" * 40)],
        )

        content = build_chat_messages(config, context, "现在")[1]["content"]

        assert "最近的话" in content
        assert "长期记忆" not in content

    def test_trimming_recent_messages_keeps_the_newest(self):
        config = Config(group_chat_memory_context_max_chars=60)
        context = ChatContext(
            recent_messages=[make_message(f"第{index}条消息", index=index) for index in range(5)]
        )

        content = build_chat_messages(config, context, "现在")[1]["content"]

        assert "第4条消息" in content
        assert "第3条消息" not in content
        assert "第0条消息" not in content

    def test_prompt_does_not_contain_api_keys(self):
        config = Config(
            group_chat_memory_chat_api_key="sk-chat-secret",
            group_chat_memory_embedding_api_key="sk-embed-secret",
        )
        context = ChatContext(recent_messages=[make_message("hi")])

        messages = build_chat_messages(config, context, "在吗")

        text = json.dumps(messages, ensure_ascii=False)
        assert "sk-chat-secret" not in text
        assert "sk-embed-secret" not in text

    def test_empty_context_still_contains_current_input(self):
        messages = build_chat_messages(Config(), ChatContext(), "在吗")

        assert len(messages) == 2
        content = messages[1]["content"]
        assert "在吗" in content
        assert "长期记忆" not in content
        assert "最近群聊" not in content

    def test_empty_display_name_falls_back_to_user_id(self):
        context = ChatContext(
            recent_messages=[make_message("hi", user_id=424242, display_name="   ")]
        )

        content = build_chat_messages(Config(), context, "在吗")[1]["content"]

        assert "群友 424242: hi" in content
