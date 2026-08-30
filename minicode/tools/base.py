"""工具基类与注册表。

工具 = 名字 + 描述 + JSON Schema + 一个本地 Python 函数，
全部在本进程内执行。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..errors import ToolError


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]
    writes: bool = False  # 是否修改工作区 / 执行命令，用于权限审批

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
                "description": self.description,
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
