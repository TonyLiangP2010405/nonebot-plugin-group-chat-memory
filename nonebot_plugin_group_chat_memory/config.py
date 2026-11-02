from __future__ import annotations

from pathlib import Path

from nonebot import get_plugin_config
from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_DB_FILENAME = "group_chat_memory.db"

DEFAULT_SYSTEM_PROMPT = (
    "你是一个群聊助手。回答自然、简洁，像群友一样交流。"
    "只使用提供的聊天记录和长期记忆，不要假装记得没有出现过的内容。"
)


class ConfigError(RuntimeError):
    """功能所需的配置缺失时抛出。"""


def _localstore_data_dir() -> Path:
    from nonebot_plugin_localstore import get_plugin_data_dir

    return get_plugin_data_dir()


class Config(BaseModel):
    """群聊记忆插件配置，所有字段均可通过环境变量或 NoneBot 配置提供。"""

    model_config = ConfigDict(extra="ignore")

    group_chat_memory_chat_base_url: str = ""
    group_chat_memory_chat_api_key: SecretStr = SecretStr("")
    group_chat_memory_chat_model: str = "deepseek-v4.1-flash"
    group_chat_memory_chat_timeout: float = Field(default=60.0, gt=0)

    group_chat_memory_embedding_base_url: str = "https://api.siliconflow.cn/v1"
    group_chat_memory_embedding_api_key: SecretStr = SecretStr("")
    group_chat_memory_embedding_model: str = "BAAI/bge-m3"
    group_chat_memory_embedding_dimensions: int = Field(default=1024, gt=0)
    group_chat_memory_embedding_timeout: float = Field(default=10.0, gt=0)

    group_chat_memory_db_path_override: str | None = None

    group_chat_memory_recent_messages: int = Field(default=20, ge=1)
    group_chat_memory_chunk_size: int = Field(default=10, ge=1)
    group_chat_memory_long_term_memories: int = Field(default=5, ge=0)
    group_chat_memory_context_max_chars: int = Field(default=6000, ge=1)
    group_chat_memory_default_probability: float = Field(default=0.0, ge=0.0, le=100.0)
    group_chat_memory_ignored_qq: set[int] = Field(default_factory=set)
    group_chat_memory_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @property
    def database_path(self) -> Path:
        """数据库文件路径；未配置时使用 localstore 的插件数据目录。"""
        configured = (self.group_chat_memory_db_path_override or "").strip()
        if configured:
            return Path(configured).expanduser()
        return _localstore_data_dir() / DEFAULT_DB_FILENAME

    def require_chat_config(self) -> None:
        """检查聊天模型配置，仅在真正调用聊天功能时使用。"""
        missing = self._missing(
            ("GROUP_CHAT_MEMORY_CHAT_BASE_URL", self.group_chat_memory_chat_base_url),
            (
                "GROUP_CHAT_MEMORY_CHAT_API_KEY",
                self.group_chat_memory_chat_api_key.get_secret_value(),
            ),
            ("GROUP_CHAT_MEMORY_CHAT_MODEL", self.group_chat_memory_chat_model),
        )
        if missing:
            raise ConfigError(f"聊天模型未配置，请设置：{'、'.join(missing)}")

    def require_embedding_config(self) -> None:
        """检查 Embedding 配置，仅在真正调用向量化功能时使用。"""
        missing = self._missing(
            (
                "GROUP_CHAT_MEMORY_EMBEDDING_BASE_URL",
                self.group_chat_memory_embedding_base_url,
            ),
            (
                "GROUP_CHAT_MEMORY_EMBEDDING_API_KEY",
                self.group_chat_memory_embedding_api_key.get_secret_value(),
            ),
            ("GROUP_CHAT_MEMORY_EMBEDDING_MODEL", self.group_chat_memory_embedding_model),
        )
        if missing:
            raise ConfigError(f"Embedding 未配置，请设置：{'、'.join(missing)}")

    @staticmethod
    def _missing(*items: tuple[str, str]) -> list[str]:
        return [name for name, value in items if not value.strip()]


def get_settings() -> Config:
    """从 NoneBot 全局配置读取本插件配置。"""
    return get_plugin_config(Config)
