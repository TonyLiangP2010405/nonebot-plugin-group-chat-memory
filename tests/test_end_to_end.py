from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest_asyncio
import respx
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.adapters.onebot.v11.event import Sender
from nonebug import App

from nonebot_plugin_group_chat_memory import handlers
from nonebot_plugin_group_chat_memory.api import ChatClient, EmbeddingClient
from nonebot_plugin_group_chat_memory.config import Config
from nonebot_plugin_group_chat_memory.context import MEMORY_SECTION_TITLE, ContextBuilder
from nonebot_plugin_group_chat_memory.memory import MemoryWriter
from nonebot_plugin_group_chat_memory.replies import ReplyCoordinator
from nonebot_plugin_group_chat_memory.store import MemoryStore

BOT_ID = 10000
GROUP_A = 20002
GROUP_B = 20003
USER_ID = 1001
SUPERUSER_ID = 10001


def group_event(
    content: str | Message,
    *,
    group_id: int = GROUP_A,
    user_id: int = USER_ID,
    to_me: bool = False,
    message_id: int = 1,
) -> GroupMessageEvent:
    message = content if isinstance(content, Message) else Message(content)
    return GroupMessageEvent(
        time=1_700_000_000 + message_id,
        self_id=BOT_ID,
        post_type="message",
        message_type="group",
        sub_type="normal",
        message_id=message_id,
        user_id=user_id,
        group_id=group_id,
        message=message,
        original_message=message,
        raw_message=str(message),
        font=0,
        sender=Sender(user_id=user_id, nickname="昵称", card="群名片", role="member"),
        to_me=to_me,
    )


def chat_payload(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def embedding_payload(vectors: list[list[float]]) -> dict[str, object]:
    return {
        "data": [
            {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
        ]
    }


@pytest_asyncio.fixture(loop_scope="function")
async def integrated_runtime(monkeypatch, tmp_path):
    config = Config(
        group_chat_memory_chat_base_url="https://chat.example/v1",
        group_chat_memory_chat_api_key="sk-chat",
        group_chat_memory_chat_timeout=1,
        group_chat_memory_embedding_base_url="https://embedding.example/v1",
        group_chat_memory_embedding_api_key="sk-embedding",
        group_chat_memory_embedding_dimensions=2,
        group_chat_memory_embedding_timeout=1,
        group_chat_memory_query_timeout=0.2,
        group_chat_memory_db_path_override=str(tmp_path / "acceptance.db"),
    )
    store = MemoryStore(config)
    embedding = EmbeddingClient(config)
    writer = MemoryWriter(config, store, embedding, retry_delays=())
    coordinator = ReplyCoordinator(store, handlers.reply)
    monkeypatch.setattr(handlers, "config", config)
    monkeypatch.setattr(handlers, "store", store)
    monkeypatch.setattr(handlers, "embedding_client", embedding)
    monkeypatch.setattr(handlers, "writer", writer)
    monkeypatch.setattr(handlers, "context_builder", ContextBuilder(config, store, embedding))
    monkeypatch.setattr(handlers, "chat_client", ChatClient(config, retry_delays=()))
    monkeypatch.setattr(handlers, "coordinator", coordinator)
    handlers._bots.clear()
    await handlers.startup()
    try:
        yield SimpleNamespace(config=config, store=store, writer=writer)
    finally:
        await handlers.shutdown()
        handlers._bots.clear()


async def wait_for_message_count(store: MemoryStore, group_id: int, count: int) -> None:
    async def poll() -> None:
        while len(await store.recent_messages(group_id, 100)) != count:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout=2)


async def receive(
    app: App,
    event: GroupMessageEvent,
    *,
    reply: str | None = None,
    runtime: SimpleNamespace | None = None,
    stored_count: int | None = None,
) -> None:
    async with app.test_matcher() as ctx:
        bot = ctx.create_bot(base=Bot, self_id=str(BOT_ID))
        send_group_msg = AsyncMock(return_value={"message_id": 123})
        if reply is not None and runtime is None:
            ctx.should_call_send(event, reply, result=None, bot=bot)
        elif reply is not None:
            bot.send_group_msg = send_group_msg
        ctx.receive_event(bot, event)
        await ctx.run()
        if runtime is not None and stored_count is not None:
            await wait_for_message_count(runtime.store, event.group_id, stored_count)
        if reply is not None and runtime is not None:
            send_group_msg.assert_awaited_once_with(group_id=event.group_id, message=reply)


@respx.mock
async def test_two_groups_chunk_reply_and_retrieval_stay_isolated(
    app: App, integrated_runtime
):
    chat_requests: list[dict[str, object]] = []

    def embed(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        vector = [1.0, 0.0] if any("阿尔法" in text for text in texts) else [0.0, 1.0]
        return httpx.Response(200, json=embedding_payload([vector for _ in texts]))

    def chat(request: httpx.Request) -> httpx.Response:
        chat_requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_payload("定向模型回复"))

    respx.post("https://embedding.example/v1/embeddings").mock(side_effect=embed)
    chat_route = respx.post("https://chat.example/v1/chat/completions").mock(side_effect=chat)

    for index in range(10):
        await receive(app, group_event(f"群 A 阿尔法机密 {index}", message_id=index))
        await receive(
            app,
            group_event(f"群 B 贝塔隐私 {index}", group_id=GROUP_B, message_id=index + 10),
        )
    await integrated_runtime.writer.flush(timeout=2)

    group_a_chunks = await integrated_runtime.store.search(GROUP_A, [1.0, 0.0], 5)
    group_b_chunks = await integrated_runtime.store.search(GROUP_B, [1.0, 0.0], 5)
    assert len(group_a_chunks) == 1
    assert len(group_b_chunks) == 1
    assert "阿尔法机密" in group_a_chunks[0].text
    assert all(chunk.group_id == GROUP_B for chunk in group_b_chunks)
    assert all("阿尔法机密" not in chunk.text for chunk in group_b_chunks)

    directed = Message([MessageSegment.at(BOT_ID), MessageSegment.text(" 阿尔法后续 ")])
    await receive(
        app,
        group_event(directed, to_me=True, message_id=20),
        reply="定向模型回复",
        runtime=integrated_runtime,
        stored_count=12,
    )

    messages = await integrated_runtime.store.recent_messages(GROUP_A, 20)
    assert messages[-1].text == "定向模型回复"
    assert messages[-1].user_id == BOT_ID
    assert messages[-1].is_bot is True
    assert chat_route.call_count == 1
    prompt = chat_requests[0]["messages"][1]["content"]
    assert "阿尔法机密" in prompt
    assert "贝塔隐私" not in prompt
    assert "阿尔法后续" in prompt


@respx.mock
async def test_query_embedding_timeout_still_chats_and_records_reply(
    app: App, integrated_runtime
):
    embedding_started = asyncio.Event()

    async def timeout_embedding(request: httpx.Request) -> httpx.Response:
        embedding_started.set()
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    respx.post("https://embedding.example/v1/embeddings").mock(side_effect=timeout_embedding)
    chat_route = respx.post("https://chat.example/v1/chat/completions").respond(
        json=chat_payload("超时后仍能聊天")
    )
    directed = Message([MessageSegment.at(BOT_ID), MessageSegment.text(" 最近上下文问题 ")])

    await receive(
        app,
        group_event(directed, to_me=True),
        reply="超时后仍能聊天",
        runtime=integrated_runtime,
        stored_count=2,
    )

    assert embedding_started.is_set()
    assert chat_route.call_count == 1
    request = chat_route.calls.last.request
    prompt = json.loads(request.content)["messages"][1]["content"]
    assert MEMORY_SECTION_TITLE not in prompt
    assert "最近上下文问题" in prompt
    messages = await integrated_runtime.store.recent_messages(GROUP_A, 20)
    assert [(message.text, message.is_bot) for message in messages] == [
        ("最近上下文问题", False),
        ("超时后仍能聊天", True),
    ]


@respx.mock
async def test_probability_and_clear_commands_use_real_group_state(
    app: App, integrated_runtime
):
    respx.post("https://embedding.example/v1/embeddings").respond(
        json=embedding_payload([[0.5, 0.5]])
    )
    chat_route = respx.post("https://chat.example/v1/chat/completions").respond(
        json=chat_payload("随机模型回复")
    )

    for index in range(10):
        await receive(app, group_event(f"群 A 默认关闭随机 {index}", message_id=index))
    await receive(app, group_event("群 B 必须保留", group_id=GROUP_B, message_id=20))
    await integrated_runtime.writer.flush(timeout=2)
    assert chat_route.call_count == 0
    assert len(await integrated_runtime.store.search(GROUP_A, [0.5, 0.5], 5)) == 1

    await receive(
        app,
        group_event("/回复概率 100", user_id=SUPERUSER_ID, message_id=21),
        reply="已设置当前群随机回复概率：100.0%",
    )
    await receive(
        app,
        group_event("命中随机回复", message_id=22),
        reply="随机模型回复",
        runtime=integrated_runtime,
        stored_count=12,
    )
    await receive(
        app,
        group_event("/回复概率", user_id=SUPERUSER_ID, message_id=23),
        reply="当前群随机回复概率：100.0%",
    )
    await receive(
        app,
        group_event(
            "/回复概率 22.2", group_id=GROUP_B, user_id=SUPERUSER_ID, message_id=24
        ),
        reply="已设置当前群随机回复概率：22.2%",
    )
    await receive(
        app,
        group_event("/清空群记忆", user_id=SUPERUSER_ID, message_id=25),
        reply="已清空当前群记忆。",
    )

    assert await integrated_runtime.store.recent_messages(GROUP_A, 100) == []
    assert await integrated_runtime.store.search(GROUP_A, [0.5, 0.5], 5) == []
    group_b_messages = await integrated_runtime.store.recent_messages(GROUP_B, 100)
    assert [message.text for message in group_b_messages] == ["群 B 必须保留"]
    assert await integrated_runtime.store.get_probability(GROUP_A) == 100.0
    assert await integrated_runtime.store.get_probability(GROUP_B) == 22.2
    assert chat_route.call_count == 1
