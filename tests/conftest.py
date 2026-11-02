import tempfile
from pathlib import Path

import nonebot
from nonebot.adapters.onebot.v11 import Adapter


def pytest_configure() -> None:
    localstore_root = Path(tempfile.mkdtemp(prefix="group-chat-memory-test-"))
    nonebot.init(
        driver="~none",
        superusers={"10001"},
        group_chat_memory_chat_timeout="12.5",
        localstore_data_dir=localstore_root / "data",
    )
    nonebot.get_driver().register_adapter(Adapter)
    plugin = nonebot.load_plugin("nonebot_plugin_group_chat_memory")
    if plugin is None:
        raise RuntimeError("nonebot_plugin_group_chat_memory 加载失败")
