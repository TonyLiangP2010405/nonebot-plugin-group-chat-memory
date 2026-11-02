from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_without_plugin_config(code: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GROUP_CHAT_MEMORY_")
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_zero_config_import_succeeds_before_nonebot_init():
    result = run_without_plugin_config(
        "import nonebot_plugin_group_chat_memory as plugin; "
        "assert plugin.__plugin_meta__.name == '群聊记忆'"
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_nonebot_load_plugin_succeeds_without_plugin_config():
    result = run_without_plugin_config(
        "import nonebot; "
        "nonebot.init(driver='~none'); "
        "plugin = nonebot.load_plugin('nonebot_plugin_group_chat_memory'); "
        "assert plugin is not None"
    )

    assert result.returncode == 0, result.stdout + result.stderr
