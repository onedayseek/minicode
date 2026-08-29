"""文本文件的读写保真。

模型看到的永远是 LF 归一化后的文本，写回时按原文件的编码和行尾符还原。
行尾符、BOM、末尾换行这类字节级属性由工具自己维持 —— 要求模型跨十几步
记住"这个文件是 CRLF"，等于把正确性押在模型行为上。
"""

import locale
import os
from dataclasses import dataclass
from pathlib import Path

from ..errors import ToolError

# 解码顺序：UTF-8 优先，失败再退回系统编码（中文 Windows 上通常是 cp936）
_FALLBACK_ENCODING = locale.getpreferredencoding(False) or "utf-8"


@dataclass
class TextMeta:
    """一个文本文件里需要原样保留的属性。"""

    encoding: str = "utf-8"
    bom: bool = False
    eol: str = "\n"
    final_newline: bool = True
    mixed_eol: bool = False

    @property
    def eol_name(self) -> str:
        return {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}.get(self.eol, "?")


def normalize(text: str) -> str:
    """统一到 LF。模型给的字符串和文件内容都要经过这里再比较。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_bytes(raw: bytes) -> tuple[str, str, bool]:
    """返回 (文本, 编码名, 是否有 BOM)。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8", True
    try:
        return raw.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        return raw.decode(_FALLBACK_ENCODING, errors="replace"), _FALLBACK_ENCODING, False


def detect_eol(text: str) -> tuple[str, bool]:
    """返回 (主导行尾符, 是否混排)。"""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    counts = {"\r\n": crlf, "\n": lf, "\r": cr}
    kinds = [k for k, v in counts.items() if v]
    if not kinds:
        return "\n", False  # 没有换行的文件按 LF 处理
    dominant = max(counts, key=lambda k: counts[k])
    return dominant, len(kinds) > 1


def read_source(path: Path) -> tuple[str, TextMeta]:
    """读成 LF 归一化的文本，同时记下还原所需的属性。"""
    raw = path.read_bytes()
    if b"\0" in raw[:8000]:
        raise ToolError(f"{path.name} 看起来是二进制文件，无法以文本读取。")

    text, encoding, bom = decode_bytes(raw)
    eol, mixed = detect_eol(text)
    normalized = normalize(text)
    return normalized, TextMeta(
        encoding=encoding,
        bom=bom,
        eol=eol,
        final_newline=normalized.endswith("\n") or not normalized,
        mixed_eol=mixed,
    )


def write_source(path: Path, text: str, meta: TextMeta) -> None:
    """把 LF 文本按 meta 还原后原子写入。

    先写临时文件再 replace，避免中途失败留下半截文件。
    """
    body = normalize(text)
    if meta.final_newline and body and not body.endswith("\n"):
        body += "\n"
    elif not meta.final_newline and body.endswith("\n"):
        body = body[:-1]

    if meta.eol != "\n":
        body = body.replace("\n", meta.eol)

    raw = body.encode(meta.encoding, errors="replace")
    if meta.bom:
        raw = b"\xef\xbb\xbf" + raw

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".minicode.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
