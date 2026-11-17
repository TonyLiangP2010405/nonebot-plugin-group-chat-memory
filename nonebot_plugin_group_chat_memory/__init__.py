from nonebot import get_driver, require
from nonebot.plugin import PluginMetadata

from .config import Config

__plugin_meta__ = PluginMetadata(
    name="群聊记忆",
    description="为群聊提供共享长期记忆、点名对话、按群概率回复、语音回复和 B站视频转发",
    usage=(
        "在群聊中 @机器人 或回复机器人的消息即可对话。\n"
        "SUPERUSER：/回复概率 [0~100] 查询或设置当前群随机回复概率；"
        "/设置语音模型 [deepseek/qwen]、/设置语音密钥 或"
        "/设置阿里云密钥（千问AI平台）或再加 /设置阿里云空间（百炼），"
        "配 /设置语音参考 使用语音；"
        "/生成馒头音色 生成固定的原创馒头声音，/导出馒头音色 导出音频；"
        "/语音名单 管理可用语音的固定群友（私聊选群和 QQ）；"
        "/语音 引用一条消息并生成语音回复；"
        "/超管必回 [开/关] 查询或设置 SUPERUSER 发言是否必定触发回复；"
        "/清空群记忆 清除当前群聊天记录与长期记忆。"
    ),
    type="application",
    homepage="https://github.com/TonyLiangP2010405/nonebot-plugin-group-chat-memory",
    config=Config,
    supported_adapters={"~onebot.v11"},
)

try:
    get_driver()
except ValueError:
    pass  # 允许在 NoneBot 初始化前单独导入模块
else:
    require("nonebot_plugin_localstore")
    from . import handlers as handlers
