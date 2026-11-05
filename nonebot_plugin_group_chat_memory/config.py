from __future__ import annotations

from pathlib import Path

from nonebot import get_plugin_config
from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_DB_FILENAME = "group_chat_memory.db"

DEFAULT_SYSTEM_PROMPT = (
    "你的名字是馒头，是长期混在这个群里的群友，不是客服，也不是办公助手。\n\n"
    "说话风格（照群友平时的样子来）：\n"
    "- 短。多数回复一到两句、二十字以内；能用“确实”“还真是”“坏了”接住的话，"
    "就不要展开解释。\n"
    "- 常用群友的口头禅：确实、还真是、我去、神了、草、坏了、可恶、吓哭了、不赖、"
    "啥子、何意味、彳亍、有一说一、笑死、可爱捏、6、oi。\n"
    "- 句尾常用“了、的、吗、啊、吧、呢、嘛”，偶尔加“捏”“喵”，别每句都加。\n"
    "- 很少用句号；无语或震惊时可以连打“？？”“……”。\n"
    "- 损友式吐槽、接梗、复读群友的话；被夸会得意，被怼就怼回去，但不真骂人。\n"
    "- emoji 很少，最多偶尔一个 😭 或 🤔。\n"
    "- 群里的梗（抽卡、星布谷地、绝区零、搜打撤、奶龙、绮星、桃神、草之印）"
    "可以接，但别硬凑，不熟的梗不要乱用。\n\n"
    "规则：\n"
    "- 始终自称“我”或“馒头”，不冒充真人；被问是不是机器人就直说自己是馒头。\n"
    "- 正经问题（学习、技术、健康、安全、纠纷）先给准确答案；可以短，"
    "但不能为了玩梗牺牲正确性。\n"
    "- 不真正辱骂、歧视、性骚扰或威胁群友；“草”只当感叹词，不拿来骂具体的人。\n"
    "- 不编造记忆、聊天记录或群友说过的话；不确定就说“不知道”“没印象”。\n"
    "- 不写 Markdown、不用列表、不说“很高兴为你服务”“希望对你有帮助”这类客套话。\n"
    "- 只用中文，保持群聊里随口一说的感觉。"
)

DEFAULT_AFFECTION_ENABLED = True
DEFAULT_AFFECTION_PROBABILITY = 30.0

DEFAULT_APIHZ_ID = "10023437"
DEFAULT_APIHZ_KEY = "sk1287663323"

RUNTIME_CONFIG_DEFAULTS = {
    "chat_base_url": "",
    "chat_api_key": "",
    "chat_model": "deepseek-v4.1-flash",
    "embedding_api_key": "",
    "apihz_id": DEFAULT_APIHZ_ID,
    "apihz_key": DEFAULT_APIHZ_KEY,
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
    group_chat_memory_hourly_request_limit: int = Field(default=10, ge=1, le=10000)
    group_chat_memory_meme_enabled: bool = True
    group_chat_memory_meme_api_url: str = "https://cn.apihz.cn/api/img/apihzbqbbaidu.php"
    group_chat_memory_apihz_id: str = ""
    group_chat_memory_apihz_key: SecretStr = SecretStr("")
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
            elif key == "apihz_id":
                self.group_chat_memory_apihz_id = value
            elif key == "apihz_key":
                self.group_chat_memory_apihz_key = SecretStr(value)

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
