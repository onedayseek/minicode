"""实验性验证增强：独立设计测试，再由 Developer 收敛到同一条验证命令通过。"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .context import Context
from .errors import ToolError
from .loop import Agent
from .tools import Registry, Shell
from .tools import files, search
from .tools.base import Tool, resolve
from .tools.shell import EXIT_PREFIX, run_command

VERIFY_MAX_SECONDS = 15 * 60
VERIFY_TIMEOUT = 180
MAX_COMMAND_CHARS = 500

_TEST_DIRS = {"test", "tests", "testing", "spec", "specs", "__tests__"}
_TEST_NAMES = (
    re.compile(r"^test_.+"),
    re.compile(r"^.+_test\.[^.]+$"),
    re.compile(r"^.+\.(test|spec)\.[^.]+$"),
    re.compile(r"^test.+\.java$"),
    re.compile(r"^.+tests?\.cs$"),
)
_CONTROL_TOKENS = ("\n", "\r", "&", "||", ";", "|", ">", "<", "`", "$(")
_VERIFY_COMMANDS = (
    re.compile(r"^(python|python\d+(\.\d+)?|py)\s+-m\s+(pytest|unittest|compileall)\b"),
    re.compile(r"^(python|python\d+(\.\d+)?|py)\s+[^\s]*(test|spec)[^\s]*\.py\b"),
    re.compile(r"^(pytest|tox|nox|ruff|mypy|pyright)\b"),
    re.compile(r"^(uv|poetry|pipenv)\s+run\s+((python|py)\s+-m\s+)?(pytest|unittest|tox|nox|ruff|mypy|pyright)\b"),
    re.compile(r"^(npm|pnpm|yarn)\s+(test|run\s+(test|lint|build|check))\b"),
    re.compile(r"^bun\s+(test|run\s+(test|lint|build|check))\b"),
    re.compile(r"^cargo\s+(test|check|clippy)\b"),
    re.compile(r"^go\s+test\b"),
    re.compile(r"^dotnet\s+(test|build)\b"),
    re.compile(r"^(mvn|mvnw|gradle|gradlew|\.\\gradlew(\.bat)?|\./gradlew)\s+.*\b(test|check|build)\b"),
    re.compile(r"^make\s+(test|check|build)\b"),
    re.compile(r"^cmake\s+--build\b"),
    re.compile(r"^ctest\b"),
)


def _needs_test_files(command: str) -> bool:
    lowered = command.lower()
    return any(token in lowered for token in ("pytest", "unittest", "tox", "nox", "test", "spec"))


@dataclass(frozen=True)
class VerificationPlan:
    command: str
    test_files: tuple[str, ...]
    summary: str
    coverage: tuple[str, ...]


class PlanSink:
    def __init__(self) -> None:
        self.plan: VerificationPlan | None = None


def is_test_path(path: str) -> bool:
    """只允许 Test Designer 写明显属于测试的路径。"""
    candidate = Path(path)
    parts = {part.lower() for part in candidate.parts[:-1]}
    name = candidate.name.lower()
    return bool(parts & _TEST_DIRS) or any(pattern.match(name) for pattern in _TEST_NAMES)


def validate_verification_command(command: str) -> str:
    """验证命令必须是单条、可辨认的测试/构建命令。"""
    normalized = " ".join(command.strip().split())
    if not normalized:
        raise ToolError("验证命令不能为空。")
    if len(normalized) > MAX_COMMAND_CHARS:
        raise ToolError(f"验证命令不能超过 {MAX_COMMAND_CHARS} 个字符。")
    if any(token in command for token in _CONTROL_TOKENS):
        raise ToolError("验证命令必须是单条命令，不能包含串联、管道、重定向或命令替换。")
    lowered = normalized.lower()
    if "passwithnotests" in lowered or "--force" in lowered:
        raise ToolError("验证命令不能使用忽略无测试或强制通过的选项。")
    if not any(pattern.search(lowered) for pattern in _VERIFY_COMMANDS):
        raise ToolError("只接受常见的测试、lint、编译或构建命令；请提交一条更直接的验证命令。")
    return normalized


def verifier_policy(name: str, args: dict) -> None:
    if name in ("write_file", "edit_file"):
        path = args.get("path")
        if not isinstance(path, str) or not is_test_path(path):
            raise ToolError("Test Designer 只能写入 tests/spec 等测试目录或明显的测试文件。")


def build_verifier_registry(root: Path, seen: set[str], sink: PlanSink) -> Registry:
    def submit_tests(
        command: str,
        test_files: list[str],
        summary: str,
        coverage: list[str],
    ) -> str:
        verified_command = validate_verification_command(command)
        if not test_files and _needs_test_files(verified_command):
            raise ToolError("至少要提交一个实际参与验证的测试文件。")
        normalized_files = []
        for raw in test_files:
            if not isinstance(raw, str) or not is_test_path(raw):
                raise ToolError(f"`{raw}` 不是可识别的测试文件路径。")
            path = resolve(root, raw)
            if not path.is_file() or path.stat().st_size == 0:
                raise ToolError(f"测试文件不存在或为空：{raw}")
            normalized_files.append(str(path.relative_to(root)))
        clean_summary = " ".join(summary.split())
        if not clean_summary:
            raise ToolError("测试摘要不能为空。")
        clean_coverage = tuple(" ".join(item.split()) for item in coverage if isinstance(item, str))
        clean_coverage = tuple(item for item in clean_coverage if item)
        if not clean_coverage:
            raise ToolError("coverage 不能为空，请列出原始需求中实际检查的行为。")
        sink.plan = VerificationPlan(
            command=verified_command,
            test_files=tuple(dict.fromkeys(normalized_files)),
            summary=clean_summary,
            coverage=clean_coverage,
        )
        return (
            f"已登记验证计划：{verified_command}（{len(sink.plan.test_files)} 个测试文件，"
            f"覆盖 {len(sink.plan.coverage)} 项需求）"
        )

    submit = Tool(
        name="submit_tests",
        description=(
            "提交唯一的权威验证命令并结束测试设计。命令由框架在 Developer 每次修改后重复执行；"
            "必须是单条常见测试/构建命令，不能串联或忽略无测试。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "单条测试、lint、编译或构建命令"},
                "test_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "实际参与验证且已经存在的测试文件；仅编译/lint/build 检查时可为空",
                },
                "summary": {"type": "string", "description": "这些测试覆盖了哪些需求与边界"},
                "coverage": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "原始需求中被测试或检查的具体行为，每项一句",
                },
            },
            "required": ["command", "test_files", "summary", "coverage"],
        },
        run=submit_tests,
    )
    return Registry(files.make_tools(root, seen) + search.make_tools(root) + [submit])


class VerificationLog:
    """把 Verifier 事件留在同一 JSONL，但不伪装成 Developer 对话。"""

    def __init__(self, base) -> None:
        self.base = base
        self._contract_recorded = False

    def _event(self, kind: str, **fields) -> None:
        self.base.event(f"verification_{kind}", **fields)

    def user(self, text: str) -> None:
        self._event("user", text=text)

    def workflow_feedback(self, text: str) -> None:
        self._event("feedback", text=text)

    def request(self, step: int, messages: list[dict], tools: list[dict], budget=None) -> None:
        shape = [{"role": m.get("role"), "chars": len(m.get("content") or "")} for m in messages]
        fields = {
            "step": step,
            "messages": shape,
            "tools": [t.get("function", {}).get("name") for t in tools],
            "estimated_tokens": getattr(budget, "tokens", None),
        }
        if not self._contract_recorded:
            fields["system_prompt"] = messages[0].get("content", "") if messages else ""
            fields["tool_schema"] = tools
            self._contract_recorded = True
        self._event("request", **fields)

    def reply(self, step: int, text: str, tool_calls, usage, finish_reason=None) -> None:
        self._event(
            "reply",
            step=step,
            text=text,
            tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_tokens=usage.cache_hit_tokens,
            cache_miss_tokens=usage.cache_miss_tokens,
            finish_reason=finish_reason,
        )

    def tool_result(self, call, status: str, content: str, raw_content=None) -> None:
        self._event(
            "tool_result", id=call.id, name=call.name, status=status, content=content,
            **({"raw_content": raw_content} if raw_content is not None else {}),
        )

    def stop(self, reason: str) -> None:
        self._event("stop", reason=reason)

    def system_note(self, content: str) -> None:
        self._event("system_note", content=content)

    def context_elision(self, notice: str, changes: list[dict]) -> None:
        self._event("context_elision", notice=notice, changes=changes)

    def checkpoint(self, summary: str, covers: int) -> None:
        self._event("checkpoint", summary=summary, covers=covers)

    def compaction_usage(self, usage) -> None:
        self._event(
            "compaction_usage",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_tokens=usage.cache_hit_tokens,
            cache_miss_tokens=usage.cache_miss_tokens,
        )

    def internal_error(self, call, traceback_text: str) -> None:
        self._event("internal_error", id=call.id, name=call.name, traceback=traceback_text)


def load_verifier_prompt(root: Path, shell: Shell) -> str:
    template = (Path(__file__).parent / "prompts" / "verifier.md").read_text(encoding="utf-8")
    return template.format(root=root, shell=shell.executable, shell_kind=shell.kind)


class VerificationWorkflow:
    """默认关闭的外层状态机；Developer 内部循环保持原样。"""

    def __init__(
        self,
        developer: Agent,
        llm,
        root: Path,
        ui,
        log,
        shell: Shell,
        window: int,
        max_output: int,
    ) -> None:
        self.developer = developer
        self.llm = llm
        self.root = root
        self.ui = ui
        self.log = log
        self.shell = shell
        self.window = window
        self.max_output = max_output
        self.active_task: str | None = None
        self.state = "idle"

    def restore_state(self, task: str | None, state: str = "idle") -> None:
        """恢复会话级任务身份，避免 Verifier 把“继续”当成原始需求。"""
        self.active_task = task
        self.state = state if state in {"idle", "running", "waiting_continue", "completed"} else "idle"

    def clear(self) -> None:
        self.active_task = None
        self.state = "idle"
        self.log.workflow_task(None, "idle")

    def run(self, user_input: str) -> bool:
        # 暂停中的普通输入默认是续跑；任务已完成后才开启新的 active_task。
        if self.active_task is None or self.state in ("idle", "completed"):
            self.active_task = user_input
        self.state = "running"
        self.log.workflow_task(self.active_task, self.state)

        try:
            developer_ok = self.developer.run(user_input)
        except KeyboardInterrupt:
            self.state = "waiting_continue"
            self.log.workflow_task(self.active_task, self.state)
            raise

        if not developer_ok:
            if getattr(self.developer, "last_run_outcome", "failed") == "paused":
                self.state = "waiting_continue"
            else:
                self.state = "idle"
            self.log.workflow_task(self.active_task if self.state == "waiting_continue" else None, self.state)
            return False
        if not self.developer.last_run_changed_files:
            self.state = "completed"
            self.log.workflow_task(self.active_task, self.state)
            return True

        started = time.monotonic()
        self.ui.notice("验证增强：Developer 已完成，Test Designer 正在独立设计测试…")
        self.log.event(
            "verification_start",
            task=self.active_task,
            changed_files=sorted(self.developer.last_run_changed_files),
        )
        plan = self._design(self.active_task or user_input, self.developer.last_run_changed_files)
        if plan is None:
            self.ui.error("Test Designer 未能提交有效验证计划，验证增强已停止。")
            self.log.event("verification_complete", status="failed", reason="missing_plan")
            self.state = "idle"
            self.log.workflow_task(None, self.state)
            return False

        self.log.event(
            "verification_plan",
            command=plan.command,
            test_files=list(plan.test_files),
            summary=plan.summary,
            coverage=list(plan.coverage),
        )
        round_no = 0
        while True:
            round_no += 1
            passed, output = self._run_plan(plan, round_no)
            if passed is None:
                self.log.event(
                    "verification_complete",
                    status="cancelled",
                    command=plan.command,
                    rounds=round_no,
                )
                self.state = "idle"
                return False
            if passed:
                self.ui.notice(
                    f"验证通过：`{plan.command}`（修复轮次 {round_no - 1}）\n"
                    f"需求覆盖：{'；'.join(plan.coverage)}"
                )
                handoff = self._finalize_handoff(plan, output)
                summary = self.developer.finalize(handoff)
                if summary:
                    self.ui.notice("Developer 已生成最终用户响应。")
                else:
                    self.ui.error("Developer 没有生成最终用户响应；验证结果已记录。")
                self.log.event(
                    "verification_complete",
                    status="passed",
                    command=plan.command,
                    rounds=round_no,
                    coverage=list(plan.coverage),
                    developer_summary=summary,
                )
                self.state = "completed"
                self.log.workflow_task(self.active_task, self.state)
                return True

            if time.monotonic() - started >= VERIFY_MAX_SECONDS:
                self.ui.error("验证增强达到总时限，权威验证命令仍未通过。")
                self.log.event(
                    "verification_complete",
                    status="failed",
                    reason="time_limit",
                    command=plan.command,
                    rounds=round_no,
                )
                self.state = "idle"
                return False

            feedback = self._repair_feedback(self.active_task or user_input, plan, output, round_no)
            self.ui.notice(f"验证未通过，正在回流给 Developer 修复（第 {round_no} 轮）…")
            if not self.developer.run(feedback, source="verification"):
                self.state = "idle"
                self.log.event(
                    "verification_complete",
                    status="failed",
                    reason="developer_stopped",
                    command=plan.command,
                    rounds=round_no,
                )
                return False

    def _design(self, task: str, changed_files: set[str]) -> VerificationPlan | None:
        sink = PlanSink()
        seen: set[str] = set()
        tools = build_verifier_registry(self.root, seen, sink)
        prompt = load_verifier_prompt(self.root, self.shell)
        verifier = Agent(
            self.llm,
            self.root,
            prompt,
            self.ui,
            VerificationLog(self.log),
            self.shell,
            context=Context(prompt, window=self.window, max_output=self.max_output),
            tools=tools,
            tool_policy=verifier_policy,
        )
        request = (
            "请为下面这次已经完成的修改设计独立验证。\n\n"
            f"原始任务：\n{task}\n\n"
            "Developer 报告修改过的文件：\n"
            + "\n".join(f"- {path}" for path in sorted(changed_files))
        )
        if not verifier.run(request):
            return None
        if sink.plan is None:
            reminder = (
                "你尚未调用 submit_tests。继续检查并补齐测试，然后必须用 submit_tests "
                "提交一条可重复执行的验证命令。"
            )
            if not verifier.run(reminder):
                return None
        return sink.plan

    def _run_plan(self, plan: VerificationPlan, round_no: int) -> tuple[bool | None, str]:
        args = {"command": plan.command, "timeout": VERIFY_TIMEOUT}
        intent = "运行 Test Designer 提交的权威验证命令"
        self.ui.tool_start("shell", args, "command", intent=intent)
        if not self.ui.confirm("shell", args):
            output = "[已取消] 用户拒绝执行权威验证命令。"
            self.ui.tool_end("shell", "fail", output)
            self.log.event(
                "verification_run",
                round=round_no,
                command=plan.command,
                status="cancelled",
                content=output,
            )
            return None, output
        try:
            output = run_command(self.root, self.shell, plan.command, VERIFY_TIMEOUT)
            passed = not output.startswith(EXIT_PREFIX)
            status = "ok" if passed else "warn"
        except ToolError as e:
            output = f"错误：{e}"
            passed, status = False, "fail"
        except Exception as e:
            output = f"错误：{type(e).__name__}: {e}"
            passed, status = False, "fail"
        self.ui.tool_end("shell", status, output)
        self.log.event(
            "verification_run",
            round=round_no,
            command=plan.command,
            status="passed" if passed else "failed",
            content=output,
        )
        return passed, output

    @staticmethod
    def _finalize_handoff(plan: VerificationPlan, output: str) -> str:
        return (
            "Verifier summary:\n"
            f"{plan.summary}\n\n"
            "Coverage:\n"
            + "\n".join(f"- {item}" for item in plan.coverage)
            + "\n\nRunner result:\n"
            f"Command: {plan.command}\n"
            f"{output}\n\n"
            "验证已经通过，现在请向用户给出最终答复。"
        )

    @staticmethod
    def _repair_feedback(
        task: str,
        plan: VerificationPlan,
        output: str,
        round_no: int,
    ) -> str:
        files = "\n".join(f"- {path}" for path in plan.test_files)
        return (
            "下面是验证增强工作流产生的环境反馈。继续自主处理，不要把问题交还给用户；"
            "完成后框架会重新运行同一条权威验证命令。\n\n"
            f"原始任务：\n{task}\n\n"
            f"测试设计摘要：{plan.summary}\n"
            f"需求覆盖：{'；'.join(plan.coverage)}\n"
            f"测试文件：\n{files}\n"
            f"权威验证命令：{plan.command}\n"
            f"当前修复轮次：{round_no}\n\n"
            "验证输出：\n"
            f"{output}\n\n"
            "优先修复产品代码。只有测试与原始需求冲突、无法运行或自身实现错误时才修改测试，"
            "并在工具 intent 中说明理由；不得删除断言、跳过测试或制造空测试来变绿。"
        )
