"""文件读写与编辑。"""

import os
from pathlib import Path

from ..errors import ToolError
from .base import Tool, resolve

MAX_READ_LINES = 2000
MAX_READ_BYTES = 400_000


def _read_text(target: Path) -> str:
    raw = target.read_bytes()
    if b"\0" in raw[:8000]:
        raise ToolError(f"{target.name} 看起来是二进制文件，无法以文本读取。")
    if len(raw) > MAX_READ_BYTES:
        raise ToolError(
            f"{target.name} 有 {len(raw)} 字节，超出单次读取上限。"
            f"请用 offset / limit 分段读，或先用 grep 定位。"
        )
    return raw.decode("utf-8", errors="replace")


def _atomic_write(target: Path, content: str) -> None:
    """先写临时文件再 replace，避免中途失败留下半截文件。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".minicode.tmp")
    tmp.write_text(content, encoding="utf-8", newline="")
    os.replace(tmp, target)


def make_tools(root: Path, seen: set) -> list[Tool]:
    """seen 由 loop 共享，用于 read-before-edit 约束。"""

    def read_file(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        target = resolve(root, path)
        if not target.exists():
            siblings = sorted(p.name for p in target.parent.iterdir())[:15] if target.parent.exists() else []
            hint = f" 同目录下有：{'、'.join(siblings)}" if siblings else ""
            raise ToolError(f"文件不存在：{path}。{hint}")
        if target.is_dir():
            raise ToolError(f"{path} 是目录，请用 list_files。")

        lines = _read_text(target).splitlines()
        seen.add(str(target))
        start = max(1, offset)
        chunk = lines[start - 1 : start - 1 + min(limit, MAX_READ_LINES)]
        if not chunk:
            return f"（{path} 共 {len(lines)} 行，第 {start} 行起为空）"
        # 带行号返回，让模型引用位置时更准
        body = "\n".join(f"{start + i:>5}\t{line}" for i, line in enumerate(chunk))
        tail = ""
        if start - 1 + len(chunk) < len(lines):
            tail = f"\n... 还有 {len(lines) - (start - 1 + len(chunk))} 行未显示"
        return body + tail

    def write_file(path: str, content: str) -> str:
        target = resolve(root, path)
        existed = target.exists()
        _atomic_write(target, content)
        seen.add(str(target))
        n = content.count("\n") + 1
        return f"已{'覆盖' if existed else '创建'} {path}（{n} 行）"

    def edit_file(path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
        target = resolve(root, path)
        if not target.exists():
            raise ToolError(f"文件不存在：{path}")
        if str(target) not in seen:
            raise ToolError(f"请先用 read_file 读取 {path}，确认要改的内容后再编辑。")

        text = _read_text(target)
        count = text.count(old_str)

        if count == 0:
            raise ToolError(
                f"在 {path} 中没有找到要替换的内容。\n"
                f"{_near_miss(text, old_str)}"
                f"请重新 read_file 确认原文（注意缩进和空格必须逐字符一致）。"
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"在 {path} 中匹配到 {count} 处相同内容。"
                f"请扩大 old_str 使其唯一，或传 replace_all=true 全部替换。"
            )

        _atomic_write(target, text.replace(old_str, new_str, -1 if replace_all else 1))
        return f"已编辑 {path}（替换 {count if replace_all else 1} 处）"

    return [
        Tool(
            name="read_file",
            description="读取文件内容，返回带行号的文本。编辑任何文件前必须先读。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作目录的路径"},
                    "offset": {"type": "integer", "description": "起始行号，默认 1"},
                    "limit": {"type": "integer", "description": "最多读取行数"},
                },
                "required": ["path"],
            },
            run=read_file,
        ),
        Tool(
            name="write_file",
            description="创建新文件或整体覆盖已有文件。修改已有文件优先用 edit_file。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "完整文件内容"},
                },
                "required": ["path", "content"],
            },
            run=write_file,
            writes=True,
        ),
        Tool(
            name="edit_file",
            description=(
                "把文件中的 old_str 替换成 new_str。old_str 必须与原文逐字符一致，"
                "且在文件中唯一（否则用 replace_all）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string", "description": "要被替换的原文片段"},
                    "new_str": {"type": "string", "description": "替换后的内容"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_str", "new_str"],
            },
            run=edit_file,
            writes=True,
        ),
    ]


def _near_miss(text: str, old_str: str) -> str:
    """匹配失败时，指出文件里最像的一行，让模型能自我纠正而不是反复瞎试。

    TODO(v2): 升级成分级匹配（空白归一化 / 缩进无关 / 相似度），
    目前只提示，不自动放宽匹配。
    """
    import difflib

    probe = old_str.strip().splitlines()[0] if old_str.strip() else ""
    if not probe:
        return ""
    lines = text.splitlines()
    best = difflib.get_close_matches(probe, lines, n=2, cutoff=0.6)
    if not best:
        return ""
    hits = "\n".join(f"  第 {lines.index(b) + 1} 行：{b}" for b in best)
    return f"文件中最接近的是：\n{hits}\n"
