"""列目录与文本搜索。"""

import re
from pathlib import Path

from ..errors import ToolError
from .base import Tool, resolve

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".minicode", "dist", "build"}
MAX_HITS = 100


def _walk(base: Path):
    for path in base.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def make_tools(root: Path) -> list[Tool]:
    def list_files(path: str = ".", pattern: str = "*") -> str:
        base = resolve(root, path)
        if not base.exists():
            raise ToolError(f"目录不存在：{path}")
        hits = [p for p in _walk(base) if p.match(pattern)]
        if not hits:
            return f"{path} 下没有匹配 `{pattern}` 的文件。"
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = hits[:MAX_HITS]
        out = "\n".join(str(p.relative_to(root)) for p in shown)
        if len(hits) > len(shown):
            out += f"\n... 还有 {len(hits) - len(shown)} 个文件"
        return out

    def grep(pattern: str, path: str = ".", include: str = "*") -> str:
        base = resolve(root, path)
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"正则表达式非法：{e}")

        results = []
        for file in _walk(base):
            if not file.match(include):
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = file.relative_to(root)
                    results.append(f"{rel}:{lineno}:{line.strip()[:200]}")
                    if len(results) >= MAX_HITS:
                        return "\n".join(results) + f"\n... 命中过多，已截断至 {MAX_HITS} 条"
        return "\n".join(results) if results else f"没有匹配 `{pattern}` 的内容。"

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
