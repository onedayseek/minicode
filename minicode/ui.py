"""终端渲染与用户交互。

显示原则：输入完整、输出折中。模型写的命令和参数要能看全 —— 那是判断它意图
是否正确、要不要批准这次操作的依据；工具输出只留少量首尾，因为关键信息通常在
开头（命令、配置）和结尾（错误摘要、结论），中间是过程噪声，完整内容在会话记录里。

输出走 rich，输入走 prompt_toolkit。分工的原因是终端输入远比看上去复杂：
粘贴的多行文本必须整体作为一次输入（否则会被逐行当成独立命令执行），
这要处理 bracketed paste 转义序列，而各平台的原始输入接口并不一致。
"""

import difflib
import html as html_lib
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# 工具结果：几行足够判断「做了什么、成没成」
RESULT_MAX_LINES = 5
RESULT_HEAD = 3
RESULT_TAIL = 1
# 编辑的 diff 给得宽些：改动行本身就是要看的东西，砍掉就没意义了
DIFF_MAX_LINES = 12
DIFF_CONTEXT = 3
# 调用参数：给得宽，模型写的命令和文件内容是审批的依据
ARG_MAX_LINES = 24
ARG_HEAD = 16
ARG_TAIL = 6
# 单行过长时的截断宽度
LINE_WIDTH = 200
# 参数值超过这个长度就换行独立显示，而不是挤在一行里
ARG_INLINE_LIMIT = 72
# 调用摘要行里主参数的显示宽度
HEADLINE_WIDTH = 64
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")

ACCENT = "#d97757"
PRIMARY_MARK = f"[bold {ACCENT}]●[/]"
SLASH_COMMANDS = ("/help", "/clear", "/status", "/log", "/exit")
_COMMAND_COMPLETER = WordCompleter(SLASH_COMMANDS, sentence=True)
_AUTO_SUGGEST = AutoSuggestFromHistory()
TOOL_LABELS = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "list_files": "List",
    "grep": "Search",
    "shell": "Run",
}

_PROMPT_STYLE = Style.from_dict(
    {
        "prompt": f"{ACCENT} bold",
        "supplement": f"{ACCENT} bold",
        "continuation": "ansibrightblack",
        "bottom-toolbar": "bg:#242424 #a8a8a8",
        "status": "bg:#242424 #d7d7d7",
        "hint": "bg:#242424 #808080",
        "completion-menu.completion": "bg:#303030 #d7d7d7",
        "completion-menu.completion.current": "bg:#6f3d2f #ffffff",
    }
)

_CHOICE_STYLE = Style.from_dict(
    {
        "question": "bold",
        "selected-option": f"{ACCENT} bold",
        "number": "#707070",
        "option": "#b0b0b0",
        "choice-hint": "#707070",
    }
)


def _key_bindings(completer=None, auto_suggest=None) -> KeyBindings:
    """Enter 提交，Alt+Enter / Ctrl-J 手动换行。

    没绑 Shift+Enter：绝大多数终端不把它和 Enter 区分开，发出来的是同一个键码，
    程序这边收不到任何区别。Ctrl-J 可靠是因为它在终端里是 LF，而 Enter 是 CR。
    """
    kb = KeyBindings()

    @kb.add("escape", "enter")  # Alt+Enter
    @kb.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("tab")
    def _(event) -> None:
        """Tab 接受当前建议；没有建议时不把控制字符写进输入框。"""
        buffer = event.current_buffer
        candidates = (
            list(completer.get_completions(
                buffer.document, CompleteEvent(completion_requested=True)
            ))
            if completer is not None
            else []
        )
        if len(candidates) == 1:
            buffer.apply_completion(candidates[0])
        elif candidates:
            buffer.start_completion(select_first=True)
        else:
            suggestion = buffer.suggestion
            if suggestion is None and auto_suggest is not None:
                suggestion = auto_suggest.get_suggestion(buffer, buffer.document)
            if suggestion:
                buffer.insert_text(suggestion.text)

    return kb


def _choice_bindings() -> KeyBindings:
    """方向键由 RadioList 处理；Esc 是拒绝的快捷出口。"""
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event) -> None:
        event.app.exit(result="cancel")

    return kb


def _make_session(
    history,
    ptk_input,
    ptk_output,
    bottom_toolbar,
    *,
    completer=None,
    auto_suggest=None,
    history_search: bool = False,
) -> PromptSession | None:
    """建输入会话，终端撑不住就返回 None，由调用方降级。

    Windows 下 prompt_toolkit 要求一个真正的 console screen buffer。
    Git Bash / MSYS 的伪终端把 TERM 报成 xterm 却没有那个 buffer，
    创建时直接抛异常 —— 不兜住的话，在 Git Bash 里连启动都做不到。
    """
    try:
        return PromptSession(
            history=history,
            style=_PROMPT_STYLE,
            key_bindings=_key_bindings(completer, auto_suggest),
            completer=completer,
            complete_while_typing=completer is not None,
            auto_suggest=auto_suggest,
            enable_history_search=history_search,
            bottom_toolbar=bottom_toolbar,
            prompt_continuation=lambda *_: HTML("<continuation>· </continuation>"),
            input=ptk_input,
            output=ptk_output,
        )
    except Exception:
        return None


def clip(text: str, max_lines: int, head: int, tail: int) -> str:
    """保留首尾、折叠中段。"""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        body = lines
    else:
        omitted = len(lines) - head - tail
        body = lines[:head] + [f"… 省略 {omitted} 行 …"] + lines[-tail:]
    return "\n".join(
        line if len(line) <= LINE_WIDTH else line[:LINE_WIDTH] + f" …(本行还有 {len(line) - LINE_WIDTH} 字符)"
        for line in body
    )


def _diff_style(line: str) -> str:
    """给 unified diff 的行上色。"""
    if line.startswith("+"):
        return "green"
    if line.startswith("-"):
        return "red"
    return "dim"


def _diff_lines(before: str, after: str) -> list[str]:
    """两段文本的 diff，去掉文件头和位置标记 —— 这里比的是片段，没有行号可言。"""
    raw = difflib.unified_diff(
        before.split("\n"), after.split("\n"), lineterm="", n=DIFF_CONTEXT
    )
    return [l for l in list(raw)[2:] if not l.startswith("@@")]


def _brief(value) -> str:
    """摘要行里的主参数：折成一行，超宽截断。完整内容仍在下面展开。"""
    text = " ".join(str(value).split())
    return text if len(text) <= HEADLINE_WIDTH else text[:HEADLINE_WIDTH] + "…"


def _is_inline(value) -> bool:
    text = str(value)
    return "\n" not in text and len(text) <= ARG_INLINE_LIMIT


class UI:
    def __init__(
        self,
        auto_approve: bool = False,
        history_path: Path | None = None,
        ptk_input=None,
        ptk_output=None,
    ) -> None:
        self.console = Console()
        self.auto_approve = auto_approve
        self._always: set[str] = set()
        self._streaming = False
        self._stream_text = ""
        self._stream_marker_pending = True
        self._status = None
        self._ptk_input = ptk_input
        self._ptk_output = ptk_output
        self.supplemental_message = ""
        self.status_line = ""
        self.total_in = 0
        self.total_out = 0
        self.total_cached = 0

        history = InMemoryHistory()
        if history_path is not None:
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(history_path))
            except OSError:
                pass  # 历史记录存不下不该拦住会话
        # 非交互（管道、重定向）时不需要增强输入，省掉一次注定失败的尝试
        self._session = (
            _make_session(
                history,
                ptk_input,
                ptk_output,
                self._bottom_toolbar,
                completer=_COMMAND_COMPLETER,
                auto_suggest=_AUTO_SUGGEST,
                history_search=True,
            )
            if ptk_input is not None or sys.stdin.isatty()
            else None
        )

    @property
    def rich_input(self) -> bool:
        """增强输入是否可用。不可用时没有多行粘贴和输入历史。"""
        return self._session is not None

    # ---- 等待提示 ----

    def start_thinking(self) -> None:
        """首个 token 到达前终端是静的，没有提示的话看起来像卡住了。"""
        if self._status is None:
            self._status = self.console.status(
                f"[bold {ACCENT}]✻[/] [dim]正在思考…[/]", spinner="dots"
            )
            self._status.start()

    def stop_thinking(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # ---- 模型输出 ----

    def stream(self, chunk: str) -> None:
        self.stop_thinking()
        if not chunk:
            return
        if not self._streaming:
            self.console.print()
            self._streaming = True
            self._stream_text = ""
            self._stream_marker_pending = True
        self._stream_text += chunk

        blocks, self._stream_text = _complete_markdown_blocks(self._stream_text)
        for block in blocks:
            self._print_stream_block(block)

    def end_stream(self) -> None:
        self.stop_thinking()
        if self._streaming:
            if self._stream_text:
                self._print_stream_block(self._stream_text)
            self._streaming = False
            self._stream_text = ""
            self._stream_marker_pending = True

    def _print_stream_block(self, text: str) -> None:
        """稳定块只追加一次；已打印内容永不参与后续刷新。"""
        marker = self._stream_marker_pending
        try:
            self.console.print(_safe_markdown_response(text.rstrip(), marker=marker))
        except Exception:
            try:
                self.console.print(_plain_response(text.rstrip(), marker=marker))
            except Exception:
                pass  # UI 失败不能终止 agent，会话日志仍保留完整回复
        self._stream_marker_pending = False

    def retry_notice(self, reason: str, partial: bool) -> None:
        """请求失败、正在退避重试。

        已经流到终端上的文本擦不掉，只能说明它会被重新生成一遍 ——
        否则用户看到的是屏幕上凭空多出半截重复的回答，且不知道为什么要等。
        """
        self.end_stream()
        detail = reason.splitlines()[0][:120] if reason.strip() else "未知原因"
        tail = "，上面这段会重新生成" if partial else ""
        self.console.print(
            f"[yellow]↻[/] [dim]请求中断（{escape(detail)}），正在重试{tail}[/]"
        )

    # ---- 工具调用 ----

    def tool_start(self, name: str, args: dict, primary: str | None = None) -> None:
        """摘要一行 + 完整参数。参数不截断 —— 那是批不批准这次操作的依据。"""
        self.stop_thinking()
        self.console.print()
        label = TOOL_LABELS.get(name, name)
        self.console.print(
            f"{PRIMARY_MARK} [bold]{label}[/]({escape(_headline(args, primary))})"
        )

        if name == "edit_file" and self._preview_edit(args):
            return
        for key, value in args.items():
            if _is_inline(value):
                continue  # 短参数已经完整出现在摘要行里了
            text = value if isinstance(value, str) else str(value)
            self.console.print(f"[dim]  │ {key}:[/]")
            for line in clip(text, ARG_MAX_LINES, ARG_HEAD, ARG_TAIL).splitlines():
                self.console.print(f"[dim]  │[/] {escape(line)}")

    def _preview_edit(self, args: dict) -> bool:
        """把 old_str / new_str 直接渲染成 diff。

        审批时要判断的是「这一改对不对」，两段原文并排看得费劲，
        而这正是执行后会看到的同一种形式 —— 批准前后的视图应当一致。
        """
        old, new = args.get("old_str"), args.get("new_str")
        if not isinstance(old, str) or not isinstance(new, str):
            return False
        lines = _diff_lines(old, new)
        if not lines:
            return False
        for line in clip("\n".join(lines), ARG_MAX_LINES, ARG_HEAD, ARG_TAIL).splitlines():
            self.console.print(f"[dim]  │[/] [{_diff_style(line)}]{escape(line)}[/]")
        return True

    def tool_end(self, name: str, status: str, detail: str) -> None:
        """status: ok / warn / fail。warn 表示工具跑完了但结果不理想（如退出码非零）。"""
        self.stop_thinking()
        color = {"ok": "dim", "warn": "yellow", "fail": "red"}[status]

        if status == "ok" and name == "edit_file":
            self._render_edit(detail)
            return

        total = len(detail.splitlines())
        lines = clip(detail.rstrip(), RESULT_MAX_LINES, RESULT_HEAD, RESULT_TAIL).splitlines() or [""]
        self.console.print(f"  [{color}]⎿ {escape(lines[0])}[/]")
        for line in lines[1:]:
            self.console.print(f"    [{color}]{escape(line)}[/]")
        if total > RESULT_MAX_LINES:
            self.console.print(f"    [dim]（共 {total} 行）[/]")

    def _render_edit(self, detail: str) -> None:
        """编辑结果：首行是摘要，其余是 diff。

        diff 的折叠规则和普通输出不同 —— 改动行是全部价值所在，上下文行只用于定位。
        套用「留头留尾、砍中段」会把 `-` 行留下而砍掉 `+` 行，
        正好丢掉「改成了什么」这个唯一要看的信息。所以超长时先丢上下文行。
        """
        head, _, body = detail.partition("\n")
        self.console.print(f"  [dim]⎿ {escape(head)}[/]")

        lines = [l for l in body.splitlines() if not l.startswith("@@")]
        dropped = 0
        if len(lines) > DIFF_MAX_LINES:
            changed = [l for l in lines if l[:1] in "+-"]
            dropped, lines = len(lines) - len(changed), changed
        for line in lines[:DIFF_MAX_LINES]:
            self.console.print(f"    [{_diff_style(line)}]{escape(line)}[/]")

        rest = len(lines) - DIFF_MAX_LINES
        if rest > 0:
            self.console.print(f"    [dim]… 还有 {rest} 行改动[/]")
        elif dropped:
            self.console.print(f"    [dim]… 省略 {dropped} 行上下文[/]")

    def confirm(self, name: str, args: dict) -> bool:
        self.stop_thinking()
        self.supplemental_message = ""
        if self.auto_approve or name in self._always:
            return True
        if self.rich_input:
            session = (
                create_app_session(input=self._ptk_input, output=self._ptk_output)
                if self._ptk_input is not None and self._ptk_output is not None
                else nullcontext()
            )
            with session:
                answer = choice(
                    HTML(
                        f"<question>? 允许执行 {html_lib.escape(name)}？</question>\n"
                        "<choice-hint>  ↑/↓ 选择 · Enter 确认 · Esc 拒绝</choice-hint>"
                    ),
                    options=[
                        ("once", "允许一次"),
                        ("always", f"本会话始终允许 {name}"),
                        ("reject", "拒绝，并补充消息"),
                    ],
                    default="once",
                    symbol="❯",
                    style=_CHOICE_STYLE,
                    key_bindings=_choice_bindings(),
                )
        else:
            answer = self._confirm_fallback(name)

        if answer == "always":
            self._always.add(name)
            return True
        if answer == "reject" and self.rich_input:
            self.supplemental_message = self._ask_supplemental_message()
        return answer == "once"

    def _ask_supplemental_message(self) -> str:
        """收集给模型看的补充消息，复用所有用户文本输入的按键与粘贴语义。"""
        session = (
            create_app_session(input=self._ptk_input, output=self._ptk_output)
            if self._ptk_input is not None and self._ptk_output is not None
            else nullcontext()
        )
        try:
            with session:
                prompt = _make_session(
                    InMemoryHistory(),
                    self._ptk_input,
                    self._ptk_output,
                    HTML("<hint> Alt+Enter 换行 · Enter 提交 </hint>"),
                )
                if prompt is None:
                    raise RuntimeError("当前终端无法创建增强输入")
                return prompt.prompt(
                    HTML("<supplement>  ❯ </supplement>补充消息（可选）：")
                ).strip()
        except (KeyboardInterrupt, EOFError):
            return ""
        except Exception:
            return self.console.input(
                f"  [bold {ACCENT}]❯[/] 补充消息（可选）："
            ).strip()

    def _confirm_fallback(self, name: str) -> str:
        """管道与简陋终端没有方向键 UI，保留数字输入作为降级路径。"""
        self.console.print(f"  [bold yellow]?[/] [bold]允许执行 {escape(name)}？[/]")
        self.console.print(f"    [bold {ACCENT}]1[/] 允许一次")
        self.console.print(f"    [bold {ACCENT}]2[/] 本会话始终允许 {escape(name)}")
        self.console.print(f"    [bold {ACCENT}]3[/] 拒绝，并补充消息")
        answer = self.console.input(
            f"  [dim]请选择[/] [bold {ACCENT}][1][/][dim]：[/] "
        ).strip().lower()
        if answer in ("2", "a", "always"):
            return "always"
        if answer in ("3", "n", "no"):
            self.supplemental_message = self.console.input(
                f"  [bold {ACCENT}]❯[/] 补充消息（可选）："
            ).strip()
            return "reject"
        return "once" if answer in ("", "1", "y", "yes") else "reject"

    # ---- 状态与提示 ----

    def set_status(self, step: int, ratio: float, usage) -> None:
        """每步刷新。累计量单独记 —— 单步用量看不出一次任务总共花了多少。"""
        self.total_in += usage.prompt_tokens
        self.total_out += usage.completion_tokens
        self.total_cached += usage.cache_hit_tokens
        parts = [
            f"第 {step} 步",
            f"上下文 {ratio:.0%}",
            f"本轮 {usage.prompt_tokens:,} tokens",
        ]
        if usage.cache_hit_tokens and usage.prompt_tokens:
            parts.append(f"缓存命中 {usage.cache_hit_tokens / usage.prompt_tokens:.0%}")
        self.status_line = " · ".join(parts)

    def show_status(self) -> None:
        if not self.status_line:
            self.console.print("[dim]尚无统计[/]")
            return
        self.console.print(f"[dim]{self.status_line}[/]")
        cached = f"（其中缓存命中 {self.total_cached}）" if self.total_cached else ""
        self.console.print(
            f"[dim]本会话累计：输入 {self.total_in} tokens{cached}，输出 {self.total_out} tokens[/]"
        )

    def notice(self, text: str) -> None:
        self.stop_thinking()
        self.console.print(f"[dim]↳ {escape(text)}[/]")

    def error(self, text: str) -> None:
        self.stop_thinking()
        self.console.print(f"[bold red]× 错误[/] {escape(text)}")

    def banner(self, model: str, root, mode: str, log_path, shell=None) -> None:
        lines = [
            Text.assemble(("✻ ", f"bold {ACCENT}"), ("minicode", "bold")),
            Text(""),
            _meta_line("model", model),
            _meta_line("directory", str(root)),
            _meta_line("approval", mode),
        ]
        if shell is not None:
            lines.append(_meta_line("shell", shell.executable))
            if os.environ.get("MSYSTEM") and shell.kind in ("pwsh", "powershell", "cmd"):
                # 从 Git Bash 启动却拿到 PowerShell 会让人以为是 bug，
                # 实际是 Windows 上的 Python 一律走 Windows 解释器
                lines.append(
                    Text(
                        "Git Bash 中仍使用 Windows shell；可设 MINICODE_SHELL=bash",
                        style="yellow",
                    )
                )
        self.console.print()
        self.console.print(
            Panel(Group(*lines), border_style="#6f554d", padding=(0, 1))
        )
        if not self.rich_input:
            # 说清楚少了什么，否则用户只会发现「粘贴多行怎么变成好几条命令了」
            self.console.print(
                "[yellow]![/] [dim]当前终端不支持增强输入，多行粘贴与输入历史不可用[/]"
            )
        self.console.print(
            "[dim]/help 命令 · Tab 接受建议 · Alt+Enter 换行 · Ctrl-C 中断 · Ctrl-D 退出[/]"
        )

    def show_history(self, messages: list[dict], tool_statuses: dict[str, str]) -> None:
        """重放恢复出的对话，仅渲染，不执行工具也不修改上下文。"""
        self.console.print()
        self.console.print(Rule("已恢复的对话", style="#6f554d"))
        pending: dict[str, dict] = {}

        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user":
                self.console.print()
                self.console.print(f"[bold {ACCENT}]❯[/] {escape(content)}", highlight=False)
            elif role == "assistant":
                if content:
                    self.console.print()
                    try:
                        self.console.print(_safe_markdown_response(content))
                    except Exception:
                        self.console.print(_plain_response(content))
                for call in message.get("tool_calls") or []:
                    pending[call.get("id", "")] = call
            elif role == "tool":
                call_id = message.get("tool_call_id", "")
                call = pending.pop(call_id, {})
                function = call.get("function") or {}
                name = message.get("name") or function.get("name") or "tool"
                args = _display_arguments(function.get("arguments", ""))
                self.tool_start(name, args, next(iter(args), None))
                status = tool_statuses.get(call_id, "ok")
                self.tool_end(name, status if status in ("ok", "warn", "fail") else "ok", content)

        self.console.print()
        self.console.print(Rule("继续会话", style="bright_black"))

    def _bottom_toolbar(self):
        """增强输入的常驻状态栏；放在输入区里，不向对话记录反复刷状态行。"""
        status = html_lib.escape(self.status_line or "就绪")
        return HTML(
            f"<status> {status} </status>"
            "<hint>  Tab 接受建议 · / 命令 · Alt+Enter 换行 </hint>"
        )

    def prompt(self) -> str:
        """带状态的输入提示。状态跟在提示符上方，作为常驻显示。

        增强输入可用时，多行粘贴会整体成为一次输入而不是被拆成多条命令，
        上下方向键翻历史，Alt+Enter / Ctrl-J 手动换行。
        不可用时退回逐行读取，功能少但不影响使用。
        """
        self.stop_thinking()
        self.console.print()
        if self.status_line and self._session is None:
            self.console.print(f"[dim]{self.status_line}[/]")
        if self._session is not None:
            try:
                return self._session.prompt(HTML("<prompt>❯ </prompt>"))
            except (KeyboardInterrupt, EOFError):
                raise  # 中断和 EOF 是正常信号，不是终端不兼容
            except Exception:
                self._session = None  # 这个终端用不了，之后别再试
        return self.console.input(f"[bold {ACCENT}]❯[/] ")


def _meta_line(label: str, value) -> Text:
    """启动面板中的等宽元信息行。"""
    return Text.assemble((f"{label:<10}", "dim"), (str(value), ""))


def _headline(args: dict, primary: str | None) -> str:
    """调用摘要行里括号内的部分。主参数不带名字 —— 参数名本身没有信息量。"""
    parts = []
    if primary and primary in args:
        parts.append(_brief(args[primary]))
    parts += [
        f"{k}={v!r}" for k, v in args.items() if k != primary and _is_inline(v)
    ]
    return ", ".join(parts)


def _display_arguments(raw) -> dict:
    """恢复展示里的参数解析失败时保留原文，而不是隐藏那次调用。"""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"arguments": str(raw)}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


def _markdown_response(text: str, marker: bool = True) -> Table:
    """模型回复的统一视图：固定事件标记 + Rich Markdown 正文。"""
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(width=1, no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(
        Text("●", style=f"bold {ACCENT}") if marker else Text(""),
        Markdown(text, code_theme="monokai", hyperlinks=True),
    )
    return table


def _plain_response(text: str, marker: bool = True) -> Table:
    """Markdown 解析或渲染失败时保留完整内容。"""
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(width=1, no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(
        Text("●", style=f"bold {ACCENT}") if marker else Text(""),
        Text(text),
    )
    return table


def _safe_markdown_response(text: str, marker: bool = True) -> Table:
    try:
        return _markdown_response(text, marker=marker)
    except Exception:
        return _plain_response(text, marker=marker)


def _complete_markdown_blocks(text: str) -> tuple[list[str], str]:
    """取出空行结尾且不在 fenced code 内的稳定 Markdown 块。"""
    blocks: list[str] = []
    start = 0
    position = 0
    fence_char = ""
    fence_length = 0

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        match = _FENCE_RE.match(body)
        if match:
            fence = match.group(1)
            if not fence_char:
                fence_char, fence_length = fence[0], len(fence)
            elif fence[0] == fence_char and len(fence) >= fence_length:
                fence_char, fence_length = "", 0

        position += len(line)
        complete_line = line.endswith(("\n", "\r"))
        if not fence_char and complete_line and not body.strip():
            block = text[start:position]
            if block.strip():
                blocks.append(block)
            start = position

    return blocks, text[start:]
