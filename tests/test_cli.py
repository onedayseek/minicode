"""CLI 启动路径。

用子进程跑真实入口：这些接线错误（参数个数对不上、模块导入顺序）
只有在进程真正起来时才暴露，单测里 import 一下是发现不了的。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import minicode.cli as cli_module

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


def test_verify模式在banner中明确标成实验性(tmp_path):
    r = run_cli("-C", str(tmp_path), "--verify", stdin="/exit\n")
    assert r.returncode == 0, r.stderr
    assert "验证增强（实验）" in r.stdout


def test_系统提示要求工具调用前说明意图(tmp_path):
    prompt = cli_module.load_system_prompt(tmp_path)
    assert "每个工具调用" in prompt
    assert "`intent`" in prompt


def test_斜杠命令都不崩(tmp_path):
    r = run_cli("-C", str(tmp_path), stdin="/help\n/status\n/usage\n/log\n/clear\n/nonsense\n/exit\n")
    assert r.returncode == 0, r.stderr
    for expected in ("显示本帮助", "已清空对话历史", "未知命令"):
        assert expected in r.stdout
    log = next((tmp_path / ".minicode" / "sessions").glob("*.jsonl"))
    commands = [
        json.loads(line)["command"]
        for line in log.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("kind") == "command"
    ]
    assert commands == ["/help", "/status", "/usage", "/log", "/clear", "/nonsense", "/exit"]


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


def test_resume接受sessions目录的完整相对路径(tmp_path):
    sessions = tmp_path / ".minicode" / "sessions"
    sessions.mkdir(parents=True)
    path = sessions / "old.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    assert cli_module._resolve_resume(".minicode/sessions/old.jsonl", tmp_path) == path


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


def fake_cli(monkeypatch, inputs, run_result=True):
    captured = []
    ui = SimpleNamespace(
        prompt=lambda: next(inputs),
        console=SimpleNamespace(print=lambda *_args, **_kwargs: None),
        banner=lambda *_args, **_kwargs: None,
        notice=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        end_stream=lambda: None,
        status_line="",
    )
    log = SimpleNamespace(
        path=Path("fake.jsonl"),
        event=lambda *_args, **_kwargs: None,
        stop=lambda *_args, **_kwargs: None,
        command=lambda *_args, **_kwargs: None,
    )
    shell = SimpleNamespace(executable="shell", kind="pwsh")
    llm = SimpleNamespace(model="test")

    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            self.context = SimpleNamespace(reset=lambda: None)
            self.seen_files = set()
            self.shell = shell

        def run(self, text):
            captured.append(text)
            return run_result

    monkeypatch.setattr(cli_module, "UI", lambda **_kwargs: ui)
    monkeypatch.setattr(cli_module, "SessionLog", lambda *_args, **_kwargs: log)
    monkeypatch.setattr(cli_module, "Agent", FakeAgent)
    monkeypatch.setattr(cli_module, "resolve_shell", lambda: shell)
    monkeypatch.setattr(cli_module.LLMClient, "from_env", lambda **_kwargs: llm)
    monkeypatch.setattr(cli_module, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(cli_module, "load_system_prompt", lambda _root: "system")
    return captured


def test_交互输入传给agent时保留首尾空白(tmp_path, monkeypatch):
    captured = fake_cli(monkeypatch, iter(("  第一行\n    第二行  ", "/exit")))

    assert cli_module.main(["-C", str(tmp_path)]) == 0
    assert captured == ["  第一行\n    第二行  "]


def test_单次模式异常终止返回非零(tmp_path, monkeypatch):
    captured = fake_cli(monkeypatch, iter(()), run_result=False)

    assert cli_module.main(["-C", str(tmp_path), "-p", "任务"]) == 1
    assert captured == ["任务"]


def test_单次模式中断时不打印traceback(tmp_path, monkeypatch):
    captured = []
    fake_cli(monkeypatch, iter(()))

    class InterruptingAgent:
        def __init__(self, *_args, **_kwargs):
            self.context = SimpleNamespace(reset=lambda: None)
            self.seen_files = set()
            self.shell = SimpleNamespace(executable="shell", kind="pwsh")

        def run(self, text):
            captured.append(text)
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "Agent", InterruptingAgent)

    assert cli_module.main(["-C", str(tmp_path), "-p", "任务"]) == 130
    assert captured == ["任务"]


def test_verify开关把任务交给外层工作流(tmp_path, monkeypatch):
    fake_cli(monkeypatch, iter(()))
    calls = []

    class FakeWorkflow:
        def __init__(self, developer, *_args, **_kwargs):
            calls.append(("init", developer))

        def run(self, text):
            calls.append(("run", text))
            return True

    monkeypatch.setattr(cli_module, "VerificationWorkflow", FakeWorkflow)

    assert cli_module.main(["-C", str(tmp_path), "--verify", "-p", "实现功能"]) == 0
    assert calls[0][0] == "init"
    assert calls[1] == ("run", "实现功能")


def test_上下文窗口低于项目下限时拒绝启动(tmp_path, monkeypatch):
    fake_cli(monkeypatch, iter(()))
    monkeypatch.setenv("MINICODE_CONTEXT_WINDOW", "8000")

    assert cli_module.main(["-C", str(tmp_path)]) == 2


def test_输出预留不小于窗口时拒绝启动(tmp_path, monkeypatch):
    fake_cli(monkeypatch, iter(()))
    monkeypatch.setenv("MINICODE_CONTEXT_WINDOW", "128000")
    monkeypatch.setenv("MINICODE_MAX_OUTPUT_TOKENS", "128000")

    assert cli_module.main(["-C", str(tmp_path)]) == 2
