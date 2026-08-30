"""CLI 启动路径。

用子进程跑真实入口：这些接线错误（参数个数对不上、模块导入顺序）
只有在进程真正起来时才暴露，单测里 import 一下是发现不了的。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def run_cli(*args, stdin: str = "", cwd: Path | None = None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MINICODE_API_KEY", "sk-test-not-used")
    return subprocess.run(
        [sys.executable, "-m", "minicode", *args],
        input=stdin,
        cwd=cwd or REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


def test_交互模式能启动并退出(tmp_path):
    """banner 在 -p 模式下不会被调用，只有交互模式才走这条路。"""
    r = run_cli("-C", str(tmp_path), stdin="/exit\n")
    assert r.returncode == 0, r.stderr
    assert "✻ minicode" in r.stdout
    assert "shell" in r.stdout  # 解析到的解释器要显示出来


def test_斜杠命令都不崩(tmp_path):
    r = run_cli("-C", str(tmp_path), stdin="/help\n/status\n/log\n/clear\n/nonsense\n/exit\n")
    assert r.returncode == 0, r.stderr
    for expected in ("显示本帮助", "已清空对话历史", "未知命令"):
        assert expected in r.stdout


def test_会话记录被创建(tmp_path):
    run_cli("-C", str(tmp_path), stdin="/exit\n")
    files = list((tmp_path / ".minicode" / "sessions").glob("*.jsonl"))
    assert len(files) == 1
    first = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert first["kind"] == "session_start"
    assert first["shell"]  # 解释器要记进会话，便于复现


def test_resume与prompt互斥(tmp_path):
    r = run_cli("-C", str(tmp_path), "--resume", "-p", "做点事")
    assert r.returncode == 2
    assert "不能同时使用" in r.stderr


def test_resume文件不存在(tmp_path):
    r = run_cli("-C", str(tmp_path), "--resume", "没有这个文件.jsonl")
    assert r.returncode == 2


def test_没有会话可恢复(tmp_path):
    r = run_cli("-C", str(tmp_path), "--resume")
    assert r.returncode == 2


def test_工作目录不存在(tmp_path):
    r = run_cli("-C", str(tmp_path / "不存在"))
    assert r.returncode == 2


def test_从会话恢复(tmp_path):
    sessions = tmp_path / ".minicode" / "sessions"
    sessions.mkdir(parents=True)
    events = [
        {"kind": "session_start", "root": str(tmp_path), "model": "x", "shell": "y"},
        {"kind": "user", "text": "第一轮问题"},
        {"kind": "reply", "step": 1, "text": "第一轮回答", "prompt_tokens": 42, "tool_calls": []},
    ]
    (sessions / "old.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8"
    )

    r = run_cli("-C", str(tmp_path), "--resume", "old.jsonl", stdin="/exit\n")
    assert r.returncode == 0, r.stderr
    assert "恢复会话" in r.stdout
    assert "第一轮问题" in r.stdout
    assert "第一轮回答" in r.stdout
    assert "继续会话" in r.stdout

    # 新记录自成完整历史，原记录不动
    files = sorted(p.name for p in sessions.glob("*.jsonl"))
    assert len(files) == 2
    new = [p for p in sessions.glob("*.jsonl") if p.name != "old.jsonl"][0]
    kinds = [json.loads(l)["kind"] for l in new.read_text(encoding="utf-8").splitlines()]
    assert "resumed_from" in kinds and "user" in kinds
