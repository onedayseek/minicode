"""执行 shell 命令。"""

import re
import subprocess
from pathlib import Path

from ..errors import ToolError
from .base import Tool

DEFAULT_TIMEOUT = 60
MAX_OUTPUT = 30_000

# 命令跑完了但退出码非零。对模型来说这是有效信息（测试失败的输出正是它要读的），
# 所以不当成工具错误抛出，只在 UI 上和成功区分开。
EXIT_PREFIX = "退出码 "

# 明显破坏性、且几乎不会是正常开发意图的命令。审批之外的最后一道拦截。
BLOCKED = [
    (re.compile(r"\brm\s+(-\w*\s+)*-\w*[rf]\w*\s+/(\s|$)"), "递归删除根目录"),
    (re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;?\s*:"), "fork 炸弹"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh"), "下载脚本直接执行"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "格式化磁盘"),
    (re.compile(r">\s*/dev/[sh]d[a-z]"), "直接写裸设备"),
]


def _truncate(text: str) -> str:
    """保留首尾。编译/测试输出的关键信息通常在开头（命令、配置）和结尾（错误摘要）。"""
    if len(text) <= MAX_OUTPUT:
        return text
    head = text[: MAX_OUTPUT // 2]
    tail = text[-MAX_OUTPUT // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n... [中间 {omitted} 字符已省略] ...\n\n{tail}"


def make_tools(root: Path) -> list[Tool]:
    def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        for pattern, why in BLOCKED:
            if pattern.search(command):
                raise ToolError(f"命令被拦截（{why}）。如果确有必要，请让用户手动执行。")

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=min(timeout, 300),
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"命令超过 {timeout} 秒未结束，已终止。"
                f"如果是长任务，请加上超时参数或拆成更小的步骤。"
            )

        parts = []
        if proc.stdout.strip():
            parts.append(_truncate(proc.stdout.rstrip()))
        if proc.stderr.strip():
            parts.append("[stderr]\n" + _truncate(proc.stderr.rstrip()))
        body = "\n".join(parts) or "（无输出）"

        # grep / diff / find 的 exit 1 表示『无匹配』，不是失败，不该误导模型
        soft_fail = proc.returncode == 1 and re.match(r"\s*(grep|rg|diff|find)\b", command)
        if proc.returncode != 0 and not soft_fail:
            return f"{EXIT_PREFIX}{proc.returncode}\n{body}"
        return body

    return [
        Tool(
            name="bash",
            description=(
                "在工作目录下执行 shell 命令，返回合并后的输出与退出码。"
                "用于运行测试、安装依赖、查看 git 状态等。切换目录请用 `cd X && cmd` 形式。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": f"秒，默认 {DEFAULT_TIMEOUT}"},
                },
                "required": ["command"],
            },
            run=bash,
            writes=True,
        )
    ]
