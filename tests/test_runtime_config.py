from __future__ import annotations

import pytest

from nonebot_plugin_group_chat_memory.config import Config, ConfigError
from nonebot_plugin_group_chat_memory.store import MemoryStore

RUNTIME_VALUES = {
    "chat_base_url": "https://chat.example.com/v1",
    "chat_api_key": "chat-secret",
    "chat_model": "deepseek-v4.1-flash",
    "embedding_api_key": "embedding-secret",
}


def make_config(tmp_path, **overrides) -> Config:
    return Config(
        group_chat_memory_db_path_override=str(tmp_path / "runtime.db"),
        group_chat_memory_embedding_dimensions=4,
        **overrides,
    )


async def open_store(config: Config) -> MemoryStore:
    store = MemoryStore(config)
    await store.open()
    return store


@pytest.fixture
async def store(tmp_path):
    store = await open_store(make_config(tmp_path))
    try:
        yield store
    finally:
        await store.close()


async def test_runtime_config_defaults_to_empty(store):
    assert await store.get_runtime_config() == {}


async def test_runtime_config_round_trip(store):
    for key, value in RUNTIME_VALUES.items():
        await store.set_runtime_config(key, value)

    assert await store.get_runtime_config() == RUNTIME_VALUES


async def test_runtime_config_rejects_unknown_key(store):
    with pytest.raises(ValueError, match="不支持"):
        await store.set_runtime_config("chat_timeout", "1")

    assert await store.get_runtime_config() == {}


async def test_runtime_config_survives_reopen(tmp_path):
    config = make_config(tmp_path)
    store = await open_store(config)
    await store.set_runtime_config("chat_api_key", "persisted-secret")
    await store.close()

    reopened = await open_store(config)
    try:
        assert await reopened.get_runtime_config() == {"chat_api_key": "persisted-secret"}
    finally:
        await reopened.close()


async def test_apply_runtime_config_ignores_environment_values(tmp_path):
    config = make_config(
        tmp_path,
        group_chat_memory_chat_base_url="https://env.example.com/v1",
        group_chat_memory_chat_api_key="env-secret",
        group_chat_memory_chat_model="env-model",
        group_chat_memory_embedding_api_key="env-embedding-secret",
    )

    config.apply_runtime_config({})

    assert config.group_chat_memory_chat_base_url == ""
    assert config.group_chat_memory_chat_api_key.get_secret_value() == ""
    assert config.group_chat_memory_chat_model == "deepseek-v4.1-flash"
    assert config.group_chat_memory_embedding_api_key.get_secret_value() == ""
    with pytest.raises(ConfigError):
        config.require_chat_config()
    with pytest.raises(ConfigError):
        config.require_embedding_config()


async def test_apply_runtime_config_updates_clients_shared_config(tmp_path):
    config = make_config(tmp_path)

    config.apply_runtime_config(RUNTIME_VALUES)

    assert config.group_chat_memory_chat_base_url == RUNTIME_VALUES["chat_base_url"]
    assert config.group_chat_memory_chat_api_key.get_secret_value() == "chat-secret"
    assert config.group_chat_memory_chat_model == "deepseek-v4.1-flash"
    assert (
        config.group_chat_memory_embedding_api_key.get_secret_value()
        == "embedding-secret"
    )
    config.require_chat_config()
    config.require_embedding_config()


async def test_apply_runtime_config_ignores_unknown_values(tmp_path):
    config = make_config(tmp_path)

    config.apply_runtime_config({"chat_timeout": "1", "unexpected": "value"})

    assert config.group_chat_memory_chat_timeout == 60.0
    assert not hasattr(config, "unexpected")


async def test_apply_runtime_config_sets_apihz_defaults(tmp_path):
    config = make_config(tmp_path)

    config.apply_runtime_config({})

    assert config.group_chat_memory_apihz_id == "10023437"
    assert config.group_chat_memory_apihz_key.get_secret_value() == "sk1287663323"


async def test_apply_runtime_config_overrides_apihz(tmp_path):
    config = make_config(tmp_path)

    config.apply_runtime_config({"apihz_id": "42", "apihz_key": "new-key"})

    assert config.group_chat_memory_apihz_id == "42"
    assert config.group_chat_memory_apihz_key.get_secret_value() == "new-key"
