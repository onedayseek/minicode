"""列目录与文本搜索。"""

import re
import time
from pathlib import Path

from ..errors import ToolError
from .base import Tool, resolve
from .textfile import decode_bytes, normalize

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".minicode", "dist", "build"}
MAX_HITS = 100
MAX_SCAN_FILES = 5000
MAX_SCAN_SECONDS = 20


def _walk(base: Path):
    """深度优先遍历 base 下的文件，遇到 SKIP_DIRS 就不进去。

    跳过判断只看 base 以内的目录名。拿绝对路径的 parts 去判断会有个隐蔽的后果：
    工作目录的某一级祖先恰好叫 build / venv / dist 时，整棵树都会被跳掉，
    grep 一声不响地返回「没有匹配」。

    不用 rglob 是因为它没法剪枝 —— .git 和 node_modules 下的文件仍然会被
    逐个产出再逐个丢掉，大仓库上光遍历就要等很久。
    """
    stack = [base]
    while stack:
        try:
            entries = sorted(stack.pop().iterdir())
        except OSError:
            continue  # 权限不足之类，跳过这一层，不中断整次搜索
        for entry in entries:
            if entry.is_dir():
                # 不跟随目录符号链接：指回上层就会绕不出来
                if entry.name not in SKIP_DIRS and not entry.is_symlink():
                    stack.append(entry)
            elif entry.is_file():
                yield entry


def _scan(base: Path, deadline: float) -> tuple[list[Path], str]:
    """带预算地收集 base 下的文件，返回 (文件列表, 截断说明)。

    预算按「看过的文件数」算而不是「命中数」—— 否则一个很窄的 include
    仍然会把整棵树走完，而模型那边只是干等，什么反馈也拿不到。
    """
    files: list[Path] = []
    for path in _walk(base):
        files.append(path)
        if len(files) >= MAX_SCAN_FILES:
            return files, f"\n... 已扫描 {MAX_SCAN_FILES} 个文件后停止，请缩小 path 或 include"
        if len(files) % 256 == 0 and time.monotonic() > deadline:
            return files, f"\n... 遍历超过 {MAX_SCAN_SECONDS} 秒，已停止，请缩小 path 或 include"
    return files, ""


def make_tools(root: Path) -> list[Tool]:
    def list_files(path: str = ".", pattern: str = "*") -> str:
        base = resolve(root, path)
        if not base.exists():
            raise ToolError(f"目录不存在：{path}")
        files, truncated = _scan(base, time.monotonic() + MAX_SCAN_SECONDS)
        hits = [p for p in files if p.match(pattern)]
        if not hits:
            return f"{path} 下没有匹配 `{pattern}` 的文件。{truncated}"
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = hits[:MAX_HITS]
        out = "\n".join(str(p.relative_to(root)) for p in shown)
        if len(hits) > len(shown):
            out += f"\n... 还有 {len(hits) - len(shown)} 个文件"
        return out + truncated

    def grep(pattern: str, path: str = ".", include: str = "*") -> str:
        base = resolve(root, path)
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"正则表达式非法：{e}")

        # 遍历和读取共用一份预算：读文件加逐行匹配才是耗时大头，
        # 两段各给 20 秒等于最坏要等 40 秒。
        deadline = time.monotonic() + MAX_SCAN_SECONDS
        files, truncated = _scan(base, deadline)

        results = []
        for scanned, file in enumerate(files):
            if scanned % 64 == 0 and time.monotonic() > deadline:
                truncated = (
                    f"\n... 搜索超过 {MAX_SCAN_SECONDS} 秒，"
                    f"已停在第 {scanned}/{len(files)} 个文件，请缩小 path 或 include"
                )
                break
            if not file.match(include):
                continue
            try:
                raw = file.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8000]:
                continue  # 二进制文件跳过
            # 和 read_file 用同一套解码，避免非 UTF-8 文件搜不到
            text = normalize(decode_bytes(raw)[0])
            for lineno, line in enumerate(text.split("\n"), 1):
                if regex.search(line):
                    rel = file.relative_to(root)
                    results.append(f"{rel}:{lineno}:{line.strip()[:200]}")
                    if len(results) >= MAX_HITS:
                        return "\n".join(results) + f"\n... 命中过多，已截断至 {MAX_HITS} 条"
        if not results:
            return f"没有匹配 `{pattern}` 的内容。{truncated}"
        return "\n".join(results) + truncated

    return [
        Tool(
            name="list_files",
            description="递归列出目录下的文件，按修改时间倒序。自动跳过 .git、node_modules 等。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "起始目录，默认当前目录"},
                    "pattern": {"type": "string", "description": "glob 模式，如 *.py"},
                },
                "required": [],
            },
            run=list_files,
        ),
        Tool(
            name="grep",
            description="用正则在文件内容中搜索，返回 文件:行号:内容。用来定位代码比逐个读文件快得多。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python 正则表达式"},
                    "path": {"type": "string"},
                    "include": {"type": "string", "description": "只搜索匹配此 glob 的文件"},
                },
                "required": ["pattern"],
            },
            run=grep,
        ),
    ]
