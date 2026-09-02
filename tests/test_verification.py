"""验证增强的角色边界、计划协议和外层收敛循环。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minicode.errors import ToolError
from minicode.parsing import Reply, ToolCall
from minicode.session import SessionLog, load_session
from minicode.tools import resolve_shell
from minicode.verification import (
    PlanSink,
    VerificationLog,
    VerificationPlan,
    VerificationWorkflow,
    build_verifier_registry,
    is_test_path,
    validate_verification_command,
    verifier_policy,
)


class FakeUI:
    def __init__(self, approve=True):
        self.approve = approve
        self.notices = []
        self.errors = []
        self.started = []
        self.ended = []

    def notice(self, text):
        self.notices.append(text)

    def error(self, text):
        self.errors.append(text)

    def tool_start(self, name, args, primary=None, intent=""):
        self.started.append((name, args, primary, intent))

    def tool_end(self, name, status, detail):
        self.ended.append((name, status, detail))

    def start_thinking(self):
        pass

    def end_stream(self):
        pass

    def stream(self, _text):
        pass

    def retry_notice(self, *_args):
        pass

    def set_status(self, *_args):
        pass

    def add_usage(self, *_args):
        pass

    def confirm(self, name, args):
        return self.approve


class FakeLog:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def workflow_task(self, task, state):
        self.event("workflow_task", task=task, state=state)


class FakeDeveloper:
    def __init__(self, changed=True, results=None):
        self.last_run_changed_files = {"app.py"} if changed else set()
        self.results = iter(results or [True, True])
        self.calls = []
        self.last_reply_text = "已有实现总结"
        self.finalize_calls = []
        self.last_run_outcome = "completed"

    def run(self, text, *, source="user"):
        self.calls.append((text, source))
        result = next(self.results)
        self.last_run_outcome = "completed" if result else "paused"
        return result

    def finalize(self, handoff):
        self.finalize_calls.append(handoff)
        self.last_reply_text = "最终用户响应"
        return self.last_reply_text

class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, messages, tools, on_text=None, on_retry=None):
        reply = next(self.replies)
        if on_text and reply.text:
            on_text(reply.text)
        return reply


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_api.py",
        "src/__tests__/api.test.ts",
        "foo_test.go",
        "ParserTests.cs",
        "spec/parser_spec.rb",
    ],
)
def test_识别常见测试路径(path):
    assert is_test_path(path)


@pytest.mark.parametrize("path", ["app.py", "src/parser.py", "package.json", "pytest.ini"])
def test_产品与配置路径不算测试文件(path):
    assert not is_test_path(path)


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q",
        "uv run pytest tests -q",
        "npm run test:unit",
        "cargo test",
        "go test ./...",
        "ctest --output-on-failure",
    ],
)
def test_接受单条常见验证命令(command):
    assert validate_verification_command(command) == command


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q && Remove-Item app.py",
        "npm test | more",
        "python deploy.py",
        "pytest --passWithNoTests",
    ],
)
def test_拒绝串联或非验证命令(command):
    with pytest.raises(ToolError):
        validate_verification_command(command)


def test_TestDesigner不能写产品代码():
    with pytest.raises(ToolError, match="只能写入"):
        verifier_policy("write_file", {"path": "src/app.py"})
    verifier_policy("write_file", {"path": "tests/test_app.py"})


def test_submit_tests要求文件真实存在(tmp_path):
    sink = PlanSink()
    registry = build_verifier_registry(tmp_path, set(), sink)
    submit = registry.get("submit_tests")

    with pytest.raises(ToolError, match="不存在"):
        submit.run(
            "python -m pytest -q", ["tests/test_app.py"], "覆盖主要行为", ["正常输入"]
        )

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    submit.run(
        "python -m pytest -q", ["tests/test_app.py"], "覆盖主要行为",
        ["正常输入", "边界输入"],
    )

    assert sink.plan == VerificationPlan(
        "python -m pytest -q", (str(Path("tests/test_app.py")),), "覆盖主要行为",
        ("正常输入", "边界输入"),
    )


def test_编译类验证可以不提供测试文件(tmp_path):
    sink = PlanSink()
    registry = build_verifier_registry(tmp_path, set(), sink)
    registry.get("submit_tests").run(
        "python -m compileall src", [], "确认所有源码可以编译", ["源码可编译"]
    )
    assert sink.plan == VerificationPlan(
        "python -m compileall src", (), "确认所有源码可以编译", ("源码可编译",)
    )


def test_TestDesigner真实工具循环能写测试并提交计划(tmp_path):
    test_content = "def test_ok():\n    assert True\n"
    calls = [
        ToolCall(
            "write",
            "write_file",
            '{"intent":"添加独立验收测试","path":"tests/test_app.py",'
            f'"content":{json.dumps(test_content)}}}',
        ),
        ToolCall(
            "submit",
            "submit_tests",
            '{"intent":"登记权威验证命令","command":"python -m pytest -q",'
            '"test_files":["tests/test_app.py"],"summary":"覆盖主要行为",'
            '"coverage":["正常输入","边界输入"]}',
        ),
    ]
    llm = ScriptedLLM([Reply(tool_calls=calls), Reply(text="测试计划已提交")])
    workflow = VerificationWorkflow(
        FakeDeveloper(), llm, tmp_path, FakeUI(), FakeLog(), resolve_shell(), 128_000, 32_000
    )

    plan = workflow._design("实现 app", {"app.py"})

    assert (tmp_path / "tests" / "test_app.py").read_text(encoding="utf-8") == test_content
    assert plan == VerificationPlan(
        "python -m pytest -q", (str(Path("tests/test_app.py")),), "覆盖主要行为",
        ("正常输入", "边界输入"),
    )


def make_workflow(tmp_path, developer=None, approve=True):
    return VerificationWorkflow(
        developer or FakeDeveloper(),
        llm=SimpleNamespace(),
        root=tmp_path,
        ui=FakeUI(approve),
        log=FakeLog(),
        shell=resolve_shell(),
        window=128_000,
        max_output=32_000,
    )


def test_关闭归因后失败原样回流并重跑(tmp_path, monkeypatch):
    developer = FakeDeveloper(results=[True, True])
    workflow = make_workflow(tmp_path, developer)
    plan = VerificationPlan(
        "python -m pytest -q", ("tests/test_app.py",), "边界测试", ("边界输入",)
    )
    runs = iter([(False, "退出码 1\n断言失败"), (True, "1 passed")])
    monkeypatch.setattr(workflow, "_design", lambda *_args: plan)
    monkeypatch.setattr(workflow, "_run_plan", lambda *_args: next(runs))

    assert workflow.run("修复解析器") is True
    assert len(developer.calls) == 2
    feedback, source = developer.calls[1]
    assert source == "verification"
    assert "断言失败" in feedback
    assert "代码还是测试" not in feedback
    assert len(developer.finalize_calls) == 1
    assert "Verifier summary:" in developer.finalize_calls[0]
    assert "边界测试" in developer.finalize_calls[0]
    assert "python -m pytest -q" in developer.finalize_calls[0]
    assert "1 passed" in developer.finalize_calls[0]
    assert any(kind == "verification_complete" and fields["status"] == "passed"
               for kind, fields in workflow.log.events)


def test_没有文件修改时保持原有行为(tmp_path, monkeypatch):
    developer = FakeDeveloper(changed=False, results=[True])
    workflow = make_workflow(tmp_path, developer)
    designed = []
    monkeypatch.setattr(workflow, "_design", lambda *_args: designed.append(1))

    assert workflow.run("解释一下代码") is True
    assert designed == []


def test_继续时Verifier仍收到最初任务(tmp_path, monkeypatch):
    developer = FakeDeveloper(results=[False, True])
    workflow = make_workflow(tmp_path, developer)
    plan = VerificationPlan(
        "python -m pytest -q", ("tests/test_app.py",), "测试", ("主要行为",)
    )
    seen_tasks = []
    monkeypatch.setattr(
        workflow, "_design", lambda task, _files: seen_tasks.append(task) or plan
    )
    monkeypatch.setattr(workflow, "_run_plan", lambda *_args: (True, "1 passed"))

    assert workflow.run("实现解析器") is False
    assert workflow.state == "waiting_continue"
    assert workflow.run("继续") is True
    assert seen_tasks == ["实现解析器"]
    assert developer.calls == [("实现解析器", "user"), ("继续", "user")]


def test_用户拒绝权威命令后不会把取消当成代码失败(tmp_path):
    developer = FakeDeveloper(results=[True])
    workflow = make_workflow(tmp_path, developer, approve=False)
    plan = VerificationPlan(
        "python -m pytest -q", ("tests/test_app.py",), "测试", ("主要行为",)
    )

    assert workflow._run_plan(plan, 1)[0] is None


def test_Verifier事件不污染恢复但计入用量(tmp_path):
    base = SessionLog(tmp_path, "test")
    scoped = VerificationLog(base)
    usage = SimpleNamespace(
        prompt_tokens=17,
        completion_tokens=5,
        cache_hit_tokens=3,
        cache_miss_tokens=14,
    )
    scoped.user("测试设计任务")
    scoped.reply(1, "测试设计", [], usage)

    restored = load_session(base.path)
    assert restored.messages == []
    assert restored.prompt_tokens == 17
    assert restored.completion_tokens == 5
    assert restored.cache_hit_tokens == 3


def test_框架反馈恢复后仍属于Developer上下文(tmp_path):
    log = SessionLog(tmp_path, "test")
    log.workflow_feedback("验证输出：断言失败")

    assert load_session(log.path).messages == [
        {"role": "user", "content": "验证输出：断言失败"}
    ]
