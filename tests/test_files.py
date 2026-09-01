"""文件保真：模型看到 LF，写回还原原文件属性。"""

from pathlib import Path

import pytest

from minicode.errors import ToolError
from minicode.tools import build_registry
from minicode.tools.base import MAX_TOOL_OUTPUT_CHARS
from minicode.tools import textfile
from minicode.tools.textfile import TextMeta, decode_bytes, detect_eol, read_source, write_source


@pytest.fixture
def tools(tmp_path):
    seen: set[str] = set()
    reg = build_registry(tmp_path, seen)
    return {n: reg.get(n).run for n in ("read_file", "write_file", "edit_file")}, seen


def test_探测行尾符():
    assert detect_eol("a\r\nb\r\n") == ("\r\n", False)
    assert detect_eol("a\nb\n") == ("\n", False)
    assert detect_eol("a\r\nb\n") [1] is True  # 混排
    assert detect_eol("单行无换行") == ("\n", False)


def test_读取归一化为LF(tmp_path):
    f = tmp_path / "crlf.py"
    f.write_bytes(b"import os\r\ndef main():\r\n    return 1\r\n")
    text, meta = read_source(f)
    assert "\r" not in text
    assert meta.eol == "\r\n" and meta.final_newline


def test_写回还原CRLF(tmp_path):
    f = tmp_path / "crlf.py"
    f.write_bytes(b"a\r\nb\r\n")
    text, meta = read_source(f)
    write_source(f, text.replace("b", "c"), meta)
    assert f.read_bytes() == b"a\r\nc\r\n"


def test_保留BOM与无末尾换行(tmp_path):
    f = tmp_path / "bom.txt"
    f.write_bytes(b"\xef\xbb\xbfhello")
    text, meta = read_source(f)
    assert text == "hello" and meta.bom and not meta.final_newline
    write_source(f, "hello world", meta)
    assert f.read_bytes() == b"\xef\xbb\xbfhello world"


def test_CRLF文件上的多行编辑(tools, tmp_path):
    """这是 --resume 那次会话里连续失败两次的场景：模型从视图抄多行 old_str。"""
    run, seen = tools
    f = tmp_path / "crlf.py"
    f.write_bytes(b"import os\r\n\r\ndef main():\r\n    return 1\r\n")

    view = run["read_file"]("crlf.py")
    assert "CRLF" in view.splitlines()[0]  # 元信息可见
    assert "\r" not in view

    # 直接用视图里的多行内容做 old_str
    run["edit_file"]("crlf.py", "def main():\n    return 1", "def main():\n    return 2")
    assert f.read_bytes() == b"import os\r\n\r\ndef main():\r\n    return 2\r\n"


def test_编辑不引入混排行尾符(tools, tmp_path):
    run, _ = tools
    f = tmp_path / "crlf.py"
    f.write_bytes(b"a\r\nb\r\n")
    run["read_file"]("crlf.py")
    run["edit_file"]("crlf.py", "b", "b\nc")  # new_str 是 LF
    assert f.read_bytes() == b"a\r\nb\r\nc\r\n"
    assert b"\n" not in f.read_bytes().replace(b"\r\n", b"")


def test_覆盖写沿用原编码与行尾符(tools, tmp_path):
    run, _ = tools
    f = tmp_path / "crlf.py"
    f.write_bytes(b"old\r\n")
    run["write_file"]("crlf.py", "new\nlines\n")
    assert f.read_bytes() == b"new\r\nlines\r\n"


def test_新文件默认LF(tools, tmp_path):
    run, _ = tools
    run["write_file"]("new.py", "a\nb\n")
    assert (tmp_path / "new.py").read_bytes() == b"a\nb\n"


def test_匹配失败时指出缩进差异(tools, tmp_path):
    run, _ = tools
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    run["read_file"]("a.py")
    with pytest.raises(ToolError) as e:
        run["edit_file"]("a.py", "        return 1", "        return 2")
    msg = str(e.value)
    assert "缩进" in msg and "第 2 行" in msg


def test_匹配失败不再提示行尾符(tools, tmp_path):
    """行尾符已由工具处理，错误消息不该再把模型往那个方向引。"""
    run, _ = tools
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    run["read_file"]("a.py")
    with pytest.raises(ToolError) as e:
        run["edit_file"]("a.py", "完全不存在的内容", "y")
    msg = str(e.value)
    assert "已忽略行尾符差异" in msg
    assert "缩进和空格必须逐字符一致" not in msg


def test_read_before_edit仍然生效(tools, tmp_path):
    run, _ = tools
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ToolError, match="请先用 read_file"):
        run["edit_file"]("a.py", "x = 1", "x = 2")


def test_read_file拒绝负数limit(tools, tmp_path):
    run, _ = tools
    (tmp_path / "a.py").write_text("a\nb\nc\n", encoding="utf-8")
    with pytest.raises(ToolError, match="大于 0"):
        run["read_file"]("a.py", limit=-1)


def test_工具结果统一限制大小(tools, tmp_path):
    run, _ = tools
    (tmp_path / "large.txt").write_text("x" * (MAX_TOOL_OUTPUT_CHARS * 2), encoding="utf-8")

    result = run["read_file"]("large.txt")

    assert len(result) <= MAX_TOOL_OUTPUT_CHARS
    assert "中间" in result and "字符已省略" in result


def test_原编码无法表示新字符时不修改文件(tmp_path):
    target = tmp_path / "ascii.txt"
    target.write_bytes(b"original\n")
    meta = TextMeta(encoding="ascii", final_newline=True)

    with pytest.raises(ToolError, match="原文件未修改"):
        write_source(target, "emoji: 😀\n", meta)
    assert target.read_bytes() == b"original\n"


def test_无法无损解码时不替换坏字节(monkeypatch):
    monkeypatch.setattr(textfile, "_FALLBACK_ENCODING", "ascii")
    with pytest.raises(ToolError, match="无损解码"):
        decode_bytes(b"\xff")


def test_原子写入不占用固定同名临时文件(tmp_path):
    target = tmp_path / "a.py"
    companion = tmp_path / "a.py.minicode.tmp"
    companion.write_text("用户文件", encoding="utf-8")

    write_source(target, "x = 1\n", TextMeta())

    assert target.read_text(encoding="utf-8") == "x = 1\n"
    assert companion.read_text(encoding="utf-8") == "用户文件"
