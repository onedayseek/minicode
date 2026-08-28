"""终端渲染与用户交互。"""

from rich.console import Console
from rich.markup import escape

SUMMARY_LEN = 300


class UI:
    def __init__(self, auto_approve: bool = False) -> None:
        self.console = Console()
        self.auto_approve = auto_approve
        self._always: set[str] = set()
        self._streaming = False

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
        preview = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        self.console.print(f"[bold yellow]⏺[/] [bold]{name}[/]({escape(preview)})")

    def tool_end(self, status: str, detail: str) -> None:
        """status: ok / warn / fail。warn 表示工具跑完了但结果不理想（如命令退出码非零）。"""
        mark = {"ok": "[green]  ✓[/]", "warn": "[yellow]  ▲[/]", "fail": "[red]  ✗[/]"}[status]
        first = detail.strip().splitlines()[0] if detail.strip() else ""
        lines = len(detail.splitlines())
        tail = f"  [dim](共 {lines} 行)[/]" if lines > 1 else ""
        self.console.print(f"{mark} [dim]{escape(first[:SUMMARY_LEN])}[/]{tail}")

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

    # ---- 提示 ----

    def notice(self, text: str) -> None:
        self.console.print(f"[dim]— {escape(text)}[/]")

    def error(self, text: str) -> None:
        self.console.print(f"[bold red]错误[/] {escape(text)}")

    def banner(self, model: str, root, mode: str) -> None:
        self.console.print(f"\n[bold]minicode[/] · {model} · {root} · {mode}")
        self.console.print("[dim]/help 查看命令，Ctrl-C 中断当前任务，Ctrl-D 退出[/]\n")

    def status(self, ratio: float, tokens: int) -> None:
        self.console.print(f"[dim]上下文 {ratio:.0%} · {tokens} tokens[/]")


def _short(value, limit: int = 60) -> str:
    text = str(value).replace("\n", "⏎")
    return text if len(text) <= limit else text[:limit] + "…"
