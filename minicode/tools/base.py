"""工具基类与注册表。

工具 = 名字 + 描述 + JSON Schema + 一个本地 Python 函数，
全部在本进程内执行。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..errors import ToolError


# 工具结果是每一轮上下文里最容易失控的部分。各工具仍按自己的语义做更细的
# 限制，但这里保留最后一道统一上限，避免新工具忘了加限制就把整段会话撑爆。
MAX_TOOL_OUTPUT_CHARS = 30_000


def cap_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """把工具结果限制在统一大小，保留首尾并说明中间省略了多少。"""
    if len(text) <= limit:
        return text
    # 说明文字本身也占长度，先按预算留出空间，再用实际数字校正一次。
    head = tail = max(0, (limit - 40) // 2)
    for _ in range(3):
        omitted = len(text) - head - tail
        marker = f"\n\n... [中间 {omitted} 字符已省略] ...\n\n"
        available = max(0, limit - len(marker))
        head = available // 2
        tail = available - head
    omitted = len(text) - head - tail
    marker = f"\n\n... [中间 {omitted} 字符已省略] ...\n\n"
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"[:limit]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]
    writes: bool = False  # 是否修改工作区 / 执行命令，用于权限审批

    def __post_init__(self) -> None:
        # 统一包住所有工具，直接调用 Tool.run 和主循环都经过同一条上限。
        raw_run = self.run
        self.last_raw_result: str | None = None

        def bounded_run(*args, **kwargs):
            self.last_raw_result = raw_run(*args, **kwargs)
            return cap_output(self.last_raw_result)

        self.run = bounded_run

    @property
    def primary(self) -> str | None:
        """展示时突出的那个参数，取第一个必填项。

        `edit_file(app.py)` 比 `edit_file(path='app.py')` 好扫 —— 一眼看到
        动作作用在什么上，参数名本身没有信息量。
        """
        return next(iter(self.parameters.get("required", [])), None)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"{self.description}\n返回结果最多 {MAX_TOOL_OUTPUT_CHARS} 个字符，"
                    "过长时保留开头和结尾。"
                ),
                "parameters": self.parameters,
            },
        }


class Registry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            available = "、".join(sorted(self._tools))
            raise ToolError(f"没有名为 `{name}` 的工具。可用工具：{available}。")
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


def resolve(root: Path, path: str) -> Path:
    """把模型给的路径解析成绝对路径，并拒绝逃出工作区。

    信任边界就在这里：agent 在本机直接执行，唯一的空间约束是工作目录。
    """
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise ToolError(f"路径 `{path}` 在工作目录之外，已拒绝。只能操作 {root} 内的文件。")
    return target
