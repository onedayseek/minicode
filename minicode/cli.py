"""命令行入口：参数解析、REPL、斜杠命令。"""

import argparse
import sys
from pathlib import Path

from .errors import FatalError
from .llm import LLMClient, load_dotenv
from .loop import Agent
from .session import SessionLog
from .ui import UI

HELP = """\
/help     显示本帮助
/clear    清空对话历史，开始新会话
/status   查看上下文占用
/log      显示本次会话记录文件的位置
/exit     退出
"""


def load_system_prompt(root: Path) -> str:
    template = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")
    return template.format(root=root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minicode", description="一个小型编程 agent")
    parser.add_argument("-p", "--prompt", help="非交互模式：执行一个任务后退出")
    parser.add_argument("-C", "--cwd", default=".", help="工作目录，默认当前目录")
    parser.add_argument(
        "--yes", action="store_true", help="自动批准所有写操作（演示用，谨慎）"
    )
    args = parser.parse_args(argv)

    root = Path(args.cwd).resolve()
    if not root.is_dir():
        print(f"工作目录不存在：{root}", file=sys.stderr)
        return 2

    load_dotenv(root / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env")

    ui = UI(auto_approve=args.yes)
    try:
        llm = LLMClient.from_env()
    except FatalError as e:
        ui.error(str(e))
        return 2

    log = SessionLog(root, llm.model)
    agent = Agent(llm, root, load_system_prompt(root), ui, log)

    if args.prompt:
        agent.run(args.prompt)
        return 0

    ui.banner(llm.model, root, "自动批准" if args.yes else "写操作需确认", log.path)
    while True:
        try:
            line = ui.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0

        if not line:
            continue
        if line.startswith("/"):
            if line in ("/exit", "/quit"):
                return 0
            if line == "/help":
                ui.console.print(HELP)
            elif line == "/clear":
                agent.context.reset()
                agent.seen_files.clear()
                ui.status_line = ""
                log.event("clear")
                ui.notice("已清空对话历史")
            elif line == "/status":
                ui.show_status()
            elif line == "/log":
                ui.notice(str(log.path))
            else:
                ui.notice(f"未知命令 {line}，试试 /help")
            continue

        try:
            agent.run(line)
        except KeyboardInterrupt:
            ui.end_stream()
            log.stop("用户中断")
            ui.notice("已中断。对话历史保留，可以直接继续。")


if __name__ == "__main__":
    raise SystemExit(main())
