"""终端渲染与用户交互。

显示原则：输入完整、输出折中。模型写的命令和参数要能看全 —— 那是判断它意图
是否正确的依据；工具输出则保留首尾、折叠中段，因为关键信息通常在开头
（命令、配置）和结尾（错误摘要、结论）。
"""

from rich.console import Console
from rich.markup import escape

# 工具输出显示的行数上限，超出则折叠中段
RESULT_MAX_LINES = 24
RESULT_HEAD = 16
RESULT_TAIL = 6
# 单行过长时的截断宽度
LINE_WIDTH = 200
# 参数值超过这个长度就换行独立显示，而不是挤在一行里
ARG_INLINE_LIMIT = 72


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


class UI:
    def __init__(self, auto_approve: bool = False) -> None:
        self.console = Console()
        self.auto_approve = auto_approve
        self._always: set[str] = set()
        self._streaming = False
        self.status_line = ""

    # ---- 模型输出 ----

    def stream(self, chunk: str) -> None:
        if not self._streaming:
            self.console.print("[bold cyan]●[/] ", end="")
            self._streaming = True
        self.console.print(escape(chunk), end="", highlight=False)

    def end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    # ---- 工具调用 ----

    def tool_start(self, name: str, args: dict) -> None:
        """参数完整显示。命令、文件内容这类长参数换行展开，不截断。"""
        short = {k: v for k, v in args.items() if _is_inline(v)}
        long = {k: v for k, v in args.items() if not _is_inline(v)}

        inline = ", ".join(f"{k}={v!r}" for k, v in short.items())
        self.console.print(f"[bold yellow]⏺[/] [bold]{name}[/]({escape(inline)})")
        for key, value in long.items():
            text = value if isinstance(value, str) else str(value)
            self.console.print(f"[dim]  │ {key}:[/]")
            for line in clip(text, RESULT_MAX_LINES, RESULT_HEAD, RESULT_TAIL).splitlines():
                self.console.print(f"[dim]  │[/] {escape(line)}")

    def tool_end(self, status: str, detail: str) -> None:
        """status: ok / warn / fail。warn 表示工具跑完了但结果不理想（如退出码非零）。"""
        mark = {"ok": "[green]✓[/]", "warn": "[yellow]▲[/]", "fail": "[red]✗[/]"}[status]
        color = {"ok": "dim", "warn": "yellow", "fail": "red"}[status]
        total = len(detail.splitlines())
        shown = clip(detail.rstrip(), RESULT_MAX_LINES, RESULT_HEAD, RESULT_TAIL)

        lines = shown.splitlines() or [""]
        self.console.print(f"  {mark} [{color}]{escape(lines[0])}[/]")
        for line in lines[1:]:
            self.console.print(f"    [{color}]{escape(line)}[/]")
        if total > RESULT_MAX_LINES:
            self.console.print(f"    [dim]（共 {total} 行）[/]")

    def confirm(self, name: str, args: dict) -> bool:
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

    def set_status(self, step: int, ratio: float, tokens: int) -> None:
        self.status_line = f"第 {step} 步 · 上下文 {ratio:.0%} · {tokens} tokens"

    def show_status(self) -> None:
        self.console.print(f"[dim]{self.status_line or '尚无统计'}[/]")

    def notice(self, text: str) -> None:
        self.console.print(f"[dim]— {escape(text)}[/]")

    def error(self, text: str) -> None:
        self.console.print(f"[bold red]错误[/] {escape(text)}")

    def banner(self, model: str, root, mode: str, log_path) -> None:
        self.console.print(f"\n[bold]minicode[/] · {model} · {root} · {mode}")
        self.console.print(f"[dim]会话记录 {log_path}[/]")
        self.console.print("[dim]/help 查看命令，Ctrl-C 中断当前任务，Ctrl-D 退出[/]\n")

    def prompt(self) -> str:
        """带状态的输入提示。状态跟在提示符上方，作为常驻显示。"""
        if self.status_line:
            self.console.print(f"[dim]{self.status_line}[/]")
        return self.console.input("[bold green]›[/] ")


def _is_inline(value) -> bool:
    text = str(value)
    return "\n" not in text and len(text) <= ARG_INLINE_LIMIT
