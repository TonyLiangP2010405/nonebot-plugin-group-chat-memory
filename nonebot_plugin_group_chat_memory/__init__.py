from nonebot import get_driver, require
from nonebot.plugin import PluginMetadata

from .config import Config

__plugin_meta__ = PluginMetadata(
    name="群聊记忆",
    description="为群聊提供共享长期记忆、点名对话和按群概率回复",
    usage=(
        "在群聊中 @机器人 或回复机器人的消息即可对话。\n"
        "SUPERUSER：/回复概率 [0~100] 查询或设置当前群随机回复概率；"
        "/引用概率 [0~100] 查询或设置当前群回复引用触发消息的概率；"
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
