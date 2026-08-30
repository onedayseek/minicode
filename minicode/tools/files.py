"""文件读写与编辑。

对模型呈现的一律是 LF 归一化文本，匹配也在归一化文本上做；
写回时由 textfile 还原原文件的编码与行尾符。
"""

import difflib
from pathlib import Path

from ..errors import ToolError
from .base import Tool, resolve
from .textfile import TextMeta, normalize, read_source, write_source

MAX_READ_LINES = 2000
# 整读上限。这只是内存保护 —— 返回给模型的内容由 limit 行数控制，跟这个值无关。
# 早先设成 400KB，把一整类正常文件（打包产物、CSV、日志）挡在外面，
# 而报错还让模型用 offset / limit 重试，那条路径同样要整读，于是必然再失败一次。
MAX_READ_BYTES = 5_000_000
# 编辑结果里回带的 diff 行数上限。replace_all 改几十处时不能让 diff 淹掉上下文。
MAX_DIFF_LINES = 40


def _diff(before: str, after: str) -> str:
    """编辑结果的紧凑 diff。

    同时给两边看：模型据此确认改到了预期的位置（尤其 replace_all），
    用户据此一眼看清改了什么。省掉 --- / +++ 两行，文件名在上一行已经说过。
    """
    lines = list(difflib.unified_diff(before.split("\n"), after.split("\n"), lineterm="", n=2))[2:]
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... 还有 {len(lines) - MAX_DIFF_LINES} 行改动未显示"]
    return "\n".join(lines)


def make_tools(root: Path, seen: set) -> list[Tool]:
    """seen 由 loop 共享，用于 read-before-edit 约束。"""

    def _load(target: Path) -> tuple[str, TextMeta]:
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            # 不提 offset / limit：它们要先整读再切片，走不通。只给真正做得到的两条路。
            raise ToolError(
                f"{target.name} 有 {size} 字节，超过 {MAX_READ_BYTES} 字节的整读上限。"
                f"请改用 grep 在文件里定位，或用 shell 按它描述里的解释器语法分段取。"
            )
        return read_source(target)

    def read_file(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        if limit < 1:
            raise ToolError("limit 必须是大于 0 的整数。")
        target = resolve(root, path)
        if not target.exists():
            siblings = (
                sorted(p.name for p in target.parent.iterdir())[:15]
                if target.parent.exists()
                else []
            )
            hint = f" 同目录下有：{'、'.join(siblings)}" if siblings else ""
            raise ToolError(f"文件不存在：{path}。{hint}")
        if target.is_dir():
            raise ToolError(f"{path} 是目录，请用 list_files。")

        text, meta = _load(target)
        seen.add(str(target))
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # 末尾换行不算一行

        start = max(1, offset)
        chunk = lines[start - 1 : start - 1 + min(limit, MAX_READ_LINES)]
        header = f"[{meta.encoding} · {meta.eol_name}{' · 行尾符混排' if meta.mixed_eol else ''} · 共 {len(lines)} 行]"
        if not chunk:
            return f"{header}\n（第 {start} 行起为空）"

        body = "\n".join(f"{start + i:>5}\t{line}" for i, line in enumerate(chunk))
        tail = ""
        if start - 1 + len(chunk) < len(lines):
            tail = f"\n... 还有 {len(lines) - (start - 1 + len(chunk))} 行未显示"
        return f"{header}\n{body}{tail}"

    def write_file(path: str, content: str) -> str:
        target = resolve(root, path)
        existed = target.exists()
        # 覆盖已有文件时沿用它原来的编码和行尾符，新文件用 UTF-8 + LF
        meta = read_source(target)[1] if existed else TextMeta()
        write_source(target, content, meta)
        seen.add(str(target))
        return (
            f"已{'覆盖' if existed else '创建'} {path}"
            f"（{normalize(content).count(chr(10)) + 1} 行，{meta.encoding} · {meta.eol_name}）"
        )

    def edit_file(path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
        target = resolve(root, path)
        if not target.exists():
            raise ToolError(f"文件不存在：{path}")
        if str(target) not in seen:
            raise ToolError(f"请先用 read_file 读取 {path}，确认要改的内容后再编辑。")

        text, meta = _load(target)
        needle, replacement = normalize(old_str), normalize(new_str)
        count = text.count(needle)

        if count == 0:
            raise ToolError(_no_match_hint(path, text, needle))
        if count > 1 and not replace_all:
            raise ToolError(
                f"在 {path} 中匹配到 {count} 处相同内容。"
                f"请扩大 old_str 使其唯一，或传 replace_all=true 全部替换。"
            )

        updated = text.replace(needle, replacement, -1 if replace_all else 1)
        write_source(target, updated, meta)
        return f"已编辑 {path}（替换 {count if replace_all else 1} 处）\n{_diff(text, updated)}"

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
            description=(
                "创建新文件或整体覆盖已有文件。修改已有文件优先用 edit_file。"
                "覆盖时自动沿用原文件的编码与行尾符。"
            ),
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
                "把文件中的 old_str 替换成 new_str。old_str 需与 read_file 显示的内容一致"
                "（缩进与空白要逐字符对上；行尾符差异由工具处理，无需关心），"
                "且在文件中唯一，否则传 replace_all。"
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


def _no_match_hint(path: str, text: str, needle: str) -> str:
    """匹配失败时给出模型能验证的下一步，而不是笼统地要它"再检查一遍"。

    TODO(v2): 升级成分级匹配（空白归一化 → 缩进无关 → 相似度），目前只提示不放宽。
    """
    head = f"在 {path} 中没有找到要替换的内容（已忽略行尾符差异后比对）。\n"
    probe = next((line for line in needle.split("\n") if line.strip()), "")
    if not probe:
        return head

    lines = text.split("\n")
    best = difflib.get_close_matches(probe, lines, n=2, cutoff=0.6)
    if not best:
        return head + "文件里没有相近的行，请重新 read_file 确认这段内容确实存在。"

    detail = "\n".join(f"  第 {lines.index(b) + 1} 行：{b}" for b in best)
    reason = ""
    if best[0].strip() == probe.strip():
        got, want = len(best[0]) - len(best[0].lstrip()), len(probe) - len(probe.lstrip())
        if got != want:
            reason = f"差异在缩进：文件里是 {got} 个空格，你给的是 {want} 个。\n"
        elif best[0].rstrip() == probe.rstrip():
            reason = "差异在行尾的空白字符。\n"
    return f"{head}文件中最接近的是：\n{detail}\n{reason}"
