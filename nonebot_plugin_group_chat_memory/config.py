from __future__ import annotations

from pathlib import Path

from nonebot import get_plugin_config
from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_DB_FILENAME = "group_chat_memory.db"

DEFAULT_SYSTEM_PROMPT = (
    "你的名字是馒头。你借用了《我的妹妹哪有这么可爱！》中五更琉璃（网名“黑猫”）的"
    "性格和说话方式，但你不是黑猫或五更琉璃本人。\n\n"
    "性格与表达：\n"
    "- 中二、毒舌、傲娇、外冷内热；对陌生群友称“凡人”，熟悉后可直呼名字。\n"
    "- 偶尔使用“夜魔女王”“黑暗眷属”“诅咒”等中二措辞，保留“……”停顿和"
    "“哼”“别误会”等傲娇表达，但不要连续堆砌，也不要每句话加“喵”。\n"
    "- 被夸奖或关心时容易害羞，用嘴硬掩饰；不真正辱骂、贬低或威胁群友。\n"
    "- 对朋友和群友实际很关心，会认真倾听并给出可靠建议。\n"
    "- 擅长游戏、动画、同人创作、漫画和料理；不了解的内容坦率承认。\n\n"
    "规则：\n"
    "- 始终自称“我”或“馒头”，绝不自称“黑猫”“五更琉璃”或“夜魔女王”本人。\n"
    "- 用自然简洁的中文回答。正经问题优先给准确、实用的答案，角色感不能牺牲可读性。\n"
    "- 不编造聊天记录、记忆或原作剧情；不连续复述原作台词，不输出角色介绍。\n"
    "- 医疗、法律、安全等高风险问题必须准确、谨慎，不因角色扮演弱化警告。\n"
    "- 你只是借用这种性格的群聊助手，不要声称自己是现实人物。"
)

RUNTIME_CONFIG_DEFAULTS = {
    "chat_base_url": "",
    "chat_api_key": "",
    "chat_model": "deepseek-v4.1-flash",
    "embedding_api_key": "",
}


class ConfigError(RuntimeError):
    """功能所需的配置缺失时抛出。"""


def _localstore_data_dir() -> Path:
    from nonebot_plugin_localstore import get_plugin_data_dir

    return get_plugin_data_dir()


class Config(BaseModel):
    """群聊记忆插件配置；四个模型运行时字段由私聊命令和数据库覆盖。"""

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
    group_chat_memory_query_timeout: float = Field(default=3.0, gt=0)

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

    def apply_runtime_config(self, values: dict[str, str]) -> None:
        """应用数据库中的四项全局配置，未保存的字段恢复内置默认值。"""
        for key, default in RUNTIME_CONFIG_DEFAULTS.items():
            value = values.get(key, default)
            if key == "chat_base_url":
                self.group_chat_memory_chat_base_url = value
            elif key == "chat_api_key":
                self.group_chat_memory_chat_api_key = SecretStr(value)
            elif key == "chat_model":
                self.group_chat_memory_chat_model = value
            elif key == "embedding_api_key":
                self.group_chat_memory_embedding_api_key = SecretStr(value)

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
