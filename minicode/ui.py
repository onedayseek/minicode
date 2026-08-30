"""终端渲染与用户交互。

显示原则：输入完整、输出折中。模型写的命令和参数要能看全 —— 那是判断它意图
是否正确、要不要批准这次操作的依据；工具输出只留少量首尾，因为关键信息通常在
开头（命令、配置）和结尾（错误摘要、结论），中间是过程噪声，完整内容在会话记录里。

输出走 rich，输入走 prompt_toolkit。分工的原因是终端输入远比看上去复杂：
粘贴的多行文本必须整体作为一次输入（否则会被逐行当成独立命令执行），
这要处理 bracketed paste 转义序列，而各平台的原始输入接口并不一致。
"""

import difflib
import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape

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

_PROMPT_STYLE = Style.from_dict({"prompt": "ansigreen bold"})


def _key_bindings() -> KeyBindings:
    """Enter 提交，Alt+Enter / Ctrl-J 手动换行。

    没绑 Shift+Enter：绝大多数终端不把它和 Enter 区分开，发出来的是同一个键码，
    程序这边收不到任何区别。Ctrl-J 可靠是因为它在终端里是 LF，而 Enter 是 CR。
    """
    kb = KeyBindings()

    @kb.add("escape", "enter")  # Alt+Enter
    @kb.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    return kb


def _make_session(history, ptk_input, ptk_output) -> PromptSession | None:
    """建输入会话，终端撑不住就返回 None，由调用方降级。

    Windows 下 prompt_toolkit 要求一个真正的 console screen buffer。
    Git Bash / MSYS 的伪终端把 TERM 报成 xterm 却没有那个 buffer，
    创建时直接抛异常 —— 不兜住的话，在 Git Bash 里连启动都做不到。
    """
    try:
        return PromptSession(
            history=history,
            style=_PROMPT_STYLE,
            key_bindings=_key_bindings(),
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
        self._status = None
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
            _make_session(history, ptk_input, ptk_output)
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
            self._status = self.console.status("[dim]思考中…[/]", spinner="dots")
            self._status.start()

    def stop_thinking(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # ---- 模型输出 ----

    def stream(self, chunk: str) -> None:
        self.stop_thinking()
        if not self._streaming:
            self.console.print("[bold cyan]●[/] ", end="")
            self._streaming = True
        self.console.print(escape(chunk), end="", highlight=False)

    def end_stream(self) -> None:
        self.stop_thinking()
        if self._streaming:
            self.console.print()
            self._streaming = False

    def retry_notice(self, reason: str, partial: bool) -> None:
        """请求失败、正在退避重试。

        已经流到终端上的文本擦不掉，只能说明它会被重新生成一遍 ——
        否则用户看到的是屏幕上凭空多出半截重复的回答，且不知道为什么要等。
        """
        self.end_stream()
        detail = reason.splitlines()[0][:120] if reason.strip() else "未知原因"
        tail = "，上面这段会重新生成" if partial else ""
        self.console.print(f"[dim]— 请求中断（{escape(detail)}），正在重试{tail}[/]")

    # ---- 工具调用 ----

    def tool_start(self, name: str, args: dict, primary: str | None = None) -> None:
        """摘要一行 + 完整参数。参数不截断 —— 那是批不批准这次操作的依据。"""
        self.stop_thinking()
        self.console.print(f"[bold yellow]⏺[/] [bold]{name}[/]({escape(_headline(args, primary))})")

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
        if self.auto_approve or name in self._always:
            return True
        answer = self.console.input(
            f"  [bold]允许执行 {name}?[/] [dim](y=允许 / n=拒绝 / a=本会话总是允许)[/] "
        ).strip().lower()
        if answer == "a":
            self._always.add(name)
            return True
        return answer in ("", "y", "yes")

    # ---- 状态与提示 ----

    def set_status(self, step: int, ratio: float, usage) -> None:
        """每步刷新。累计量单独记 —— 单步用量看不出一次任务总共花了多少。"""
        self.total_in += usage.prompt_tokens
        self.total_out += usage.completion_tokens
        self.total_cached += usage.cache_hit_tokens
        parts = [f"第 {step} 步", f"上下文 {ratio:.0%}", f"{usage.prompt_tokens} tokens"]
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
        self.console.print(f"[dim]— {escape(text)}[/]")

    def error(self, text: str) -> None:
        self.stop_thinking()
        self.console.print(f"[bold red]错误[/] {escape(text)}")

    def banner(self, model: str, root, mode: str, log_path, shell=None) -> None:
        self.console.print(f"\n[bold]minicode[/] [dim]·[/] {model} [dim]·[/] {mode}")
        self.console.print(f"[dim]  工作目录 {root}[/]")
        if shell is not None:
            self.console.print(f"[dim]  命令解释器 {shell.executable}[/]")
            if os.environ.get("MSYSTEM") and shell.kind in ("pwsh", "powershell", "cmd"):
                # 从 Git Bash 启动却拿到 PowerShell 会让人以为是 bug，
                # 实际是 Windows 上的 Python 一律走 Windows 解释器
                self.console.print(
                    "[dim]  （当前在 MSYS / Git Bash，但命令走 Windows 解释器；"
                    "要用 bash 语法请设 MINICODE_SHELL=bash）[/]"
                )
        self.console.print(f"[dim]  会话记录 {log_path}[/]")
        if not self.rich_input:
            # 说清楚少了什么，否则用户只会发现「粘贴多行怎么变成好几条命令了」
            self.console.print("[dim]  当前终端不支持增强输入，多行粘贴与输入历史不可用[/]")
        self.console.print(
            "[dim]  /help 查看命令 · Alt+Enter 换行 · Ctrl-C 中断 · Ctrl-D 退出[/]\n"
        )

    def prompt(self) -> str:
        """带状态的输入提示。状态跟在提示符上方，作为常驻显示。

        增强输入可用时，多行粘贴会整体成为一次输入而不是被拆成多条命令，
        上下方向键翻历史，Alt+Enter / Ctrl-J 手动换行。
        不可用时退回逐行读取，功能少但不影响使用。
        """
        self.stop_thinking()
        if self.status_line:
            self.console.print(f"[dim]{self.status_line}[/]")
        if self._session is not None:
            try:
                return self._session.prompt(HTML("<prompt>› </prompt>"))
            except (KeyboardInterrupt, EOFError):
                raise  # 中断和 EOF 是正常信号，不是终端不兼容
            except Exception:
                self._session = None  # 这个终端用不了，之后别再试
        return self.console.input("[bold green]›[/] ")


def _headline(args: dict, primary: str | None) -> str:
    """调用摘要行里括号内的部分。主参数不带名字 —— 参数名本身没有信息量。"""
    parts = []
    if primary and primary in args:
        parts.append(_brief(args[primary]))
    parts += [
        f"{k}={v!r}" for k, v in args.items() if k != primary and _is_inline(v)
    ]
    return ", ".join(parts)
