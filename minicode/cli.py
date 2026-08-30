"""命令行入口：参数解析、REPL、斜杠命令。"""

import argparse
import sys
from pathlib import Path

from .errors import FatalError
from .llm import LLMClient, load_dotenv
from .loop import Agent
from .session import SessionLog, load_session
from .tools import resolve_shell
from .ui import UI

HELP = """\
/help     显示本帮助
/clear    清空对话历史，开始新会话
/status   查看上下文占用
/log      显示本次会话记录文件的位置
/exit     退出

输入时 Tab 接受灰色建议，Alt+Enter 或 Ctrl-J 手动换行，Enter 提交。
"""


def load_system_prompt(root: Path) -> str:
    template = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")
    return template.format(root=root)


def _resolve_resume(resume, root: Path) -> Path | None:
    """把 --resume 的值解析成会话文件路径。返回 None 时已打印错误。"""
    if resume is None:
        return None
    sess_dir = root / ".minicode" / "sessions"
    if resume == "__latest__":
        if not sess_dir.is_dir():
            print(f"没有会话记录目录：{sess_dir}", file=sys.stderr)
            return None
        files = sorted(
            sess_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not files:
            print(f"{sess_dir} 下没有可恢复的会话记录", file=sys.stderr)
            return None
        return files[0]
    p = Path(resume)
    if not p.is_absolute():
        p = sess_dir / p
    if not p.exists():
        print(f"会话文件不存在：{p}", file=sys.stderr)
        return None
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minicode", description="一个小型编程 agent")
    parser.add_argument("-p", "--prompt", help="非交互模式：执行一个任务后退出")
    parser.add_argument("-C", "--cwd", default=".", help="工作目录，默认当前目录")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        metavar="SESSION",
        help="恢复历史会话：不带值取最近一次，或传 .minicode/sessions 下的文件名/路径",
    )
    parser.add_argument(
        "--yes", action="store_true", help="自动批准所有写操作（演示用，谨慎）"
    )
    args = parser.parse_args(argv)

    if args.resume and args.prompt:
        parser.error("--resume 与 --prompt 不能同时使用（恢复后进入交互模式）")

    root = Path(args.cwd).resolve()
    if not root.is_dir():
        print(f"工作目录不存在：{root}", file=sys.stderr)
        return 2

    resume_path = _resolve_resume(args.resume, root)
    if args.resume and resume_path is None:
        return 2

    restored = None
    if resume_path:
        restored = load_session(resume_path)
        resumed_root = Path(restored.root).resolve()
        if resumed_root != root:
            print(
                f"会话记录的工作目录是 {resumed_root}，已切换（忽略 -C 参数）",
                file=sys.stderr,
            )
        root = resumed_root

    load_dotenv(root / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env")

    ui = UI(auto_approve=args.yes, history_path=root / ".minicode" / "history")
    try:
        llm = LLMClient.from_env()
    except FatalError as e:
        ui.error(str(e))
        return 2

    try:
        shell = resolve_shell()
    except Exception as e:
        ui.error(str(e))
        return 2

    log = SessionLog(root, llm.model, shell.executable)
    agent = Agent(llm, root, load_system_prompt(root), ui, log, shell)

    if restored is not None:
        log.inherit(resume_path)
        agent.context.messages.extend(restored.messages)
        agent.context.prompt_tokens = restored.prompt_tokens
        agent.seen_files.update(restored.seen_files)

    if args.prompt:
        return 0 if agent.run(args.prompt) else 1

    ui.banner(llm.model, root, "自动批准" if args.yes else "写操作需确认", log.path, agent.shell)
    if restored is not None:
        ui.notice(
            f"已从 {resume_path.name} 恢复会话"
            f"（历史 {len(restored.messages)} 条消息，{len(restored.seen_files)} 个已读文件）"
        )
        ui.show_history(restored.messages, restored.tool_statuses)
    while True:
        try:
            line = ui.prompt()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0

        command = line.strip()
        if not command:
            continue
        if command.startswith("/"):
            if command in ("/exit", "/quit"):
                return 0
            if command == "/help":
                ui.console.print(HELP)
            elif command == "/clear":
                agent.context.reset()
                agent.seen_files.clear()
                ui.status_line = ""
                log.event("clear")
                ui.notice("已清空对话历史")
            elif command == "/status":
                ui.show_status()
            elif command == "/log":
                ui.notice(str(log.path))
            else:
                ui.notice(f"未知命令 {command}，试试 /help")
            continue

        try:
            agent.run(line)
        except KeyboardInterrupt:
            ui.end_stream()
            log.stop("用户中断")
            ui.notice("已中断。对话历史保留，可以直接继续。")


if __name__ == "__main__":
    raise SystemExit(main())
