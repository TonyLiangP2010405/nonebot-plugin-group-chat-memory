from datetime import datetime, timezone

import pytest
from pydantic import SecretStr, ValidationError

from nonebot_plugin_group_chat_memory import config as config_module
from nonebot_plugin_group_chat_memory.config import Config, ConfigError, get_settings
from nonebot_plugin_group_chat_memory.models import (
    ChatContext,
    Chunk,
    ChunkStatus,
    Message,
)


class TestConfigDefaults:
    def test_zero_config_defaults(self):
        config = Config()

        assert config.group_chat_memory_chat_base_url == ""
        assert config.group_chat_memory_chat_api_key.get_secret_value() == ""
        assert config.group_chat_memory_chat_model == "deepseek-v4.1-flash"
        assert config.group_chat_memory_embedding_base_url == "https://api.siliconflow.cn/v1"
        assert config.group_chat_memory_embedding_api_key.get_secret_value() == ""
        assert config.group_chat_memory_embedding_model == "BAAI/bge-m3"
        assert config.group_chat_memory_embedding_dimensions == 1024
        assert config.group_chat_memory_query_timeout == 3.0
        assert config.group_chat_memory_recent_messages == 20
        assert config.group_chat_memory_chunk_size == 10
        assert config.group_chat_memory_long_term_memories == 5
        assert config.group_chat_memory_default_probability == 0.0
        assert config.group_chat_memory_ignored_qq == set()
        assert config.group_chat_memory_db_path_override is None

    def test_default_system_prompt_is_mantou_with_kuroneko_personality(self):
        prompt = Config().group_chat_memory_system_prompt

        assert "馒头" in prompt
        assert "凡人" in prompt
        assert "夜魔女王" in prompt
        assert "绝不自称“黑猫”" in prompt
        assert "准确" in prompt
        assert "安全" in prompt

    def test_ignores_unrelated_nonebot_fields(self):
        config = Config(superusers={"10001"}, driver="~fastapi")

        assert config.group_chat_memory_recent_messages == 20


class TestQueryTimeout:
    def test_default_is_short_and_independent_of_embedding_timeout(self):
        config = Config()

        assert config.group_chat_memory_query_timeout == 3.0
        assert config.group_chat_memory_query_timeout < config.group_chat_memory_embedding_timeout

    def test_accepts_positive_timeout(self):
        config = Config(group_chat_memory_query_timeout=0.25)

        assert config.group_chat_memory_query_timeout == 0.25

    @pytest.mark.parametrize("value", [0, -0.5, -3])
    def test_rejects_non_positive_timeout(self, value):
        with pytest.raises(ValidationError):
            Config(group_chat_memory_query_timeout=value)


class TestDatabasePath:
    def test_default_database_path_is_non_empty_and_raw_field_is_not_usable(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(config_module, "_localstore_data_dir", lambda: tmp_path)

        config = Config()

        assert config.database_path == tmp_path / "group_chat_memory.db"
        assert str(config.database_path) != ""
        assert config.group_chat_memory_db_path_override is None
        assert not hasattr(config, "group_chat_memory_db_path")
        with pytest.raises(AttributeError):
            _ = config.group_chat_memory_db_path

    def test_blank_override_falls_back_to_localstore(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_module, "_localstore_data_dir", lambda: tmp_path)

        assert Config(group_chat_memory_db_path_override="   ").database_path == (
            tmp_path / "group_chat_memory.db"
        )

    def test_configured_override_takes_precedence(self, monkeypatch, tmp_path):
        def unexpected_call():
            raise AssertionError("localstore should not be consulted for a configured path")

        monkeypatch.setattr(config_module, "_localstore_data_dir", unexpected_call)
        path = tmp_path / "custom" / "memory.db"

        assert Config(group_chat_memory_db_path_override=str(path)).database_path == path


class TestProbability:
    @pytest.mark.parametrize("value", [0, 0.0, 5.5, 100, 100.0])
    def test_accepts_values_in_range(self, value):
        config = Config(group_chat_memory_default_probability=value)

        assert config.group_chat_memory_default_probability == value

    @pytest.mark.parametrize("value", [-0.1, 100.1, 101])
    def test_rejects_values_outside_range(self, value):
        with pytest.raises(ValidationError):
            Config(group_chat_memory_default_probability=value)


class TestSecrets:
    def test_api_keys_are_secret_values(self):
        config = Config(
            group_chat_memory_chat_api_key="sk-chat-secret",
            group_chat_memory_embedding_api_key="sk-embed-secret",
        )

        assert isinstance(config.group_chat_memory_chat_api_key, SecretStr)
        assert isinstance(config.group_chat_memory_embedding_api_key, SecretStr)
        assert config.group_chat_memory_chat_api_key.get_secret_value() == "sk-chat-secret"
        assert config.group_chat_memory_embedding_api_key.get_secret_value() == "sk-embed-secret"

    def test_api_keys_are_redacted_from_repr_and_dump(self):
        config = Config(
            group_chat_memory_chat_api_key="sk-chat-secret",
            group_chat_memory_embedding_api_key="sk-embed-secret",
        )

        rendered = repr(config) + config.model_dump_json()
        assert "sk-chat-secret" not in rendered
        assert "sk-embed-secret" not in rendered


class TestIgnoredQQ:
    def test_accepts_iterable_and_deduplicates(self):
        config = Config(group_chat_memory_ignored_qq=[10001, 10002, 10001])

        assert config.group_chat_memory_ignored_qq == {10001, 10002}

    def test_rejects_non_integer_entries(self):
        with pytest.raises(ValidationError):
            Config(group_chat_memory_ignored_qq=["not-a-qq"])


class TestRequiredSettings:
    def test_chat_configuration_is_checked_only_when_required(self):
        config = Config()

        with pytest.raises(ConfigError) as excinfo:
            config.require_chat_config()

        message = str(excinfo.value)
        assert "GROUP_CHAT_MEMORY_CHAT_BASE_URL" in message
        assert "GROUP_CHAT_MEMORY_CHAT_API_KEY" in message

    def test_chat_configuration_accepts_base_url_and_key(self):
        config = Config(
            group_chat_memory_chat_base_url="https://api.example.com/v1",
            group_chat_memory_chat_api_key="sk-chat",
        )

        config.require_chat_config()

    def test_chat_configuration_error_does_not_leak_secret(self):
        config = Config(group_chat_memory_chat_api_key="sk-chat-secret")

        with pytest.raises(ConfigError) as excinfo:
            config.require_chat_config()

        assert "sk-chat-secret" not in str(excinfo.value)

    def test_embedding_configuration_requires_api_key(self):
        config = Config()

        with pytest.raises(ConfigError) as excinfo:
            config.require_embedding_config()

        assert "GROUP_CHAT_MEMORY_EMBEDDING_API_KEY" in str(excinfo.value)

    def test_embedding_configuration_rejects_empty_base_url(self):
        config = Config(
            group_chat_memory_embedding_base_url="",
            group_chat_memory_embedding_api_key="sk-embed",
        )

        with pytest.raises(ConfigError) as excinfo:
            config.require_embedding_config()

        assert "GROUP_CHAT_MEMORY_EMBEDDING_BASE_URL" in str(excinfo.value)

    def test_embedding_configuration_accepts_defaults_with_key(self):
        config = Config(group_chat_memory_embedding_api_key="sk-embed")

        config.require_embedding_config()


class TestSettings:
    def test_get_settings_reads_nonebot_plugin_config(self):
        settings = get_settings()

        assert isinstance(settings, Config)
        assert settings.group_chat_memory_chat_timeout == 12.5

    def test_get_settings_reads_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("GROUP_CHAT_MEMORY_CHAT_MODEL", "env-model")
        monkeypatch.setenv("GROUP_CHAT_MEMORY_IGNORED_QQ", "[123, 456]")

        settings = get_settings()

        assert settings.group_chat_memory_chat_model == "env-model"
        assert settings.group_chat_memory_ignored_qq == {123, 456}


class TestModels:
    def test_message_defaults_to_aware_utc_time_and_human(self):
        before = datetime.now(timezone.utc)

        message = Message(
            group_id=100,
            user_id=200,
            display_name="Alice",
            text="hello",
        )

        after = datetime.now(timezone.utc)
        assert message.is_bot is False
        assert message.created_at.tzinfo is not None
        assert before <= message.created_at <= after

    def test_message_preserves_explicit_values(self):
        created_at = datetime(2026, 11, 2, 8, 30, tzinfo=timezone.utc)

        message = Message(
            group_id=100,
            user_id=200,
            display_name="Bot",
            text="hi there",
            created_at=created_at,
            is_bot=True,
        )

        assert message.group_id == 100
        assert message.user_id == 200
        assert message.display_name == "Bot"
        assert message.text == "hi there"
        assert message.created_at == created_at
        assert message.is_bot is True

    def test_chunk_defaults_to_pending(self):
        chunk = Chunk(id=1, group_id=100, text="a fragment")

        assert chunk.status is ChunkStatus.PENDING

    def test_chunk_accepts_ready_status(self):
        chunk = Chunk(id=1, group_id=100, text="a fragment", status="ready")

        assert chunk.status is ChunkStatus.READY

    def test_chat_context_defaults_to_empty_lists(self):
        context = ChatContext()

        assert context.recent_messages == []
        assert context.long_term_chunks == []

    def test_chat_context_holds_recent_and_long_term_items(self):
        message = Message(group_id=100, user_id=200, display_name="Alice", text="hello")
        chunk = Chunk(id=1, group_id=100, text="older fragment", status=ChunkStatus.READY)

        context = ChatContext(recent_messages=[message], long_term_chunks=[chunk])

        assert context.recent_messages == [message]
        assert context.long_term_chunks == [chunk]


class TestPluginMetadata:
    def test_registers_onebot_v11_support(self):
        import nonebot_plugin_group_chat_memory as plugin

        assert plugin.__plugin_meta__.supported_adapters == {"~onebot.v11"}
        assert plugin.__plugin_meta__.config is Config
