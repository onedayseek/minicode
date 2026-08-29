"""列目录与搜索：跳过规则的作用范围，以及扫描预算。"""

import pytest

from minicode.tools import search
from minicode.tools.search import make_tools


def tools_at(root):
    root.mkdir(parents=True, exist_ok=True)
    return {t.name: t.run for t in make_tools(root)}


def test_祖先目录名不影响遍历(tmp_path):
    """工作目录本身位于 build/ 之下时，整棵树不能被跳掉。

    早先用绝对路径的 parts 判断 SKIP_DIRS，于是 ~/dev/build/proj 这种路径下
    grep 会一声不响地返回「没有匹配」—— 静默的空结果比报错更难发现。
    """
    root = tmp_path / "build" / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("target_symbol = 1\n", encoding="utf-8")

    run = tools_at(root)
    assert "a.py" in run["list_files"]()
    assert "target_symbol" in run["grep"]("target_symbol")


@pytest.mark.parametrize("skipped", ["node_modules", ".git", "__pycache__", "dist"])
def test_跳过规则在工作目录内仍然生效(tmp_path, skipped):
    run = tools_at(tmp_path)
    (tmp_path / skipped).mkdir()
    (tmp_path / skipped / "x.py").write_text("target_symbol", encoding="utf-8")
    (tmp_path / "ok.py").write_text("target_symbol", encoding="utf-8")

    out = run["grep"]("target_symbol")
    assert "ok.py" in out
    assert skipped not in out


def test_扫描文件数有上限(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "MAX_SCAN_FILES", 5)
    run = tools_at(tmp_path)
    for i in range(20):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    assert "已扫描 5 个文件后停止" in run["list_files"]()


def test_窄include也受扫描上限约束(tmp_path, monkeypatch):
    """上限按看过的文件数算，不是按命中数 —— 否则一个窄 include 仍会走完整棵树，
    模型那边只是干等着，什么反馈也拿不到。"""
    monkeypatch.setattr(search, "MAX_SCAN_FILES", 5)
    run = tools_at(tmp_path)
    for i in range(30):
        (tmp_path / f"f{i}.txt").write_text("target_symbol", encoding="utf-8")

    out = run["grep"]("target_symbol", include="*.py")
    assert "没有匹配" in out and "已扫描 5 个文件后停止" in out


def test_非UTF8文件也能搜到(tmp_path):
    run = tools_at(tmp_path)
    (tmp_path / "gbk.txt").write_bytes("中文内容".encode("gbk"))
    assert "gbk.txt" in run["grep"]("中文内容")


def test_二进制文件被跳过(tmp_path):
    run = tools_at(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"target_symbol\x00\x00")
    assert "没有匹配" in run["grep"]("target_symbol")


def test_目录不存在时报错(tmp_path):
    run = tools_at(tmp_path)
    with pytest.raises(Exception, match="目录不存在"):
        run["list_files"]("没有这个目录")
