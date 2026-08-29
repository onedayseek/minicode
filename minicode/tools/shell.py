"""执行命令。

解释器在启动时解析一次，解析结果既用于实际调用，也写进工具描述 ——
模型必须知道自己调用的到底是 PowerShell 还是 bash，那决定了它能用什么语法。
不用 shell=True：它的语义跨平台不同，而且对模型完全不可见。
"""

import locale
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..errors import ToolError
from .base import Tool

DEFAULT_TIMEOUT = 60
MAX_OUTPUT = 30_000

# 命令跑完了但退出码非零。对模型来说这是有效信息（测试失败的输出正是它要读的），
# 所以不当成工具错误抛出，只在 UI 上和成功区分开。
EXIT_PREFIX = "退出码 "

WINDOWS = os.name == "nt"

# 各解释器的调用参数。Windows 上按 pwsh → powershell → cmd 的顺序解析：
# PowerShell 与 Windows 的路径、环境变量、.exe/.cmd/.ps1 语义天然一致；
# Git Bash 会引入 /c/... 与 C:\... 两套路径，只在显式指定时才用。
_ARGS = {
    "pwsh": ("-NoProfile", "-NonInteractive", "-Command"),
    "powershell": ("-NoProfile", "-NonInteractive", "-Command"),
    "cmd": ("/d", "/s", "/c"),
    "bash": ("-c",),
    "sh": ("-c",),
}
_ORDER = ("pwsh", "powershell", "cmd") if WINDOWS else ("bash", "sh")

# 「命令不存在」在各解释器下的报错形态。知道了解释器，这个匹配才可靠。
_NOT_FOUND = {
    "pwsh": [r"The term '([^']+)' is not recognized", r"无法将“([^”]+)”项识别"],
    "cmd": [r"'([^']+)' is not recognized as an internal or external command",
            r"'([^']+)' 不是内部或外部命令"],
    "bash": [r"[^\n:]*: ([^:\n]+): command not found", r"([^:\n]+): 未找到命令"],
}
_NOT_FOUND["powershell"] = _NOT_FOUND["pwsh"]
_NOT_FOUND["sh"] = _NOT_FOUND["bash"]


@dataclass(frozen=True)
class Shell:
    kind: str
    executable: str

    @property
    def args(self) -> tuple[str, ...]:
        return _ARGS[self.kind]

    def argv(self, command: str) -> list[str]:
        return [self.executable, *self.args, command]

    @property
    def invocation(self) -> str:
        return f"{Path(self.executable).name} {' '.join(self.args)} <command>"

    @property
    def path_style(self) -> str:
        return "反斜杠（C:\\path\\to\\file）" if self.kind in ("pwsh", "powershell", "cmd") else "正斜杠（/path/to/file）"


def resolve_shell() -> Shell:
    """解析出本次会话实际使用的解释器。整个会话内不再变化。"""
    override = os.environ.get("MINICODE_SHELL", "").strip()
    if override:
        candidate = Path(override)
        found = str(candidate) if candidate.is_file() else shutil.which(override)
        if not found:
            raise ToolError(f"MINICODE_SHELL 指定的 {override} 找不到。")
        kind = Path(found).stem.lower()
        if kind not in _ARGS:
            raise ToolError(f"不支持的解释器 {kind}，可选：{'、'.join(_ARGS)}。")
        return Shell(kind, found)

    for kind in _ORDER:
        found = shutil.which(kind)
        if found:
            return Shell(kind, found)

    if WINDOWS:  # cmd 一定在，只是可能不在 PATH 里
        return Shell("cmd", os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"))
    return Shell("sh", "/bin/sh")


def describe(shell: Shell, root: Path) -> str:
    return (
        "在工作目录下执行命令，返回输出与退出码。用于运行测试、查看 git 状态、安装依赖等。\n"
        "执行语义：\n"
        f"- 平台：{'Windows' if WINDOWS else os.uname().sysname}\n"
        f"- 解释器：{shell.executable}\n"
        f"- 调用形式：{shell.invocation}\n"
        f"- 工作目录：{root}（命令已在此目录下执行，不要再 cd 过去）\n"
        f"- 路径风格：{shell.path_style}\n"
        "命令语法必须符合上面这个解释器，不要假定是别的 shell。"
    )


# 明显破坏性、且几乎不会是正常开发意图的命令。按解释器分组 ——
# 黑名单是减速带，不是安全边界；真正的边界是用户在场审批。
_POSIX_BLOCKED = [
    (re.compile(r"\brm\s+(-\w*\s+)*-\w*[rf]\w*\s+/(\s|$)"), "递归删除根目录"),
    (re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;?\s*:"), "fork 炸弹"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "格式化磁盘"),
    (re.compile(r">\s*/dev/[sh]d[a-z]"), "直接写裸设备"),
]
_WINDOWS_BLOCKED = [
    (re.compile(r"\b(del|erase)\b[^|]*\/s\b[^|]*\b[a-zA-Z]:\\?(\s|$)"), "递归删除整个盘符"),
    (re.compile(r"\b(rd|rmdir)\b[^|]*\/s\b[^|]*\b[a-zA-Z]:\\?(\s|$)"), "递归删除整个盘符"),
    (re.compile(r"\bRemove-Item\b[^|]*-Recurse\b[^|]*\b[a-zA-Z]:\\?(\s|$)", re.I), "递归删除整个盘符"),
    (re.compile(r"\bformat\s+[a-zA-Z]:"), "格式化磁盘"),
    (re.compile(r"\bdiskpart\b", re.I), "磁盘分区工具"),
]
_SHARED_BLOCKED = [
    (re.compile(r"\b(curl|wget|iwr|Invoke-WebRequest)\b[^|]*\|\s*(sudo\s+)?((ba)?sh|iex|Invoke-Expression)", re.I),
     "下载脚本直接执行"),
]


def blocked_patterns(shell: Shell):
    platform = _WINDOWS_BLOCKED if shell.kind in ("pwsh", "powershell", "cmd") else _POSIX_BLOCKED
    return platform + _SHARED_BLOCKED


def _decode(raw: bytes) -> str:
    """PowerShell 5.1 按控制台代码页输出，pwsh 7 是 UTF-8，所以两种都试。"""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False) or "utf-8", errors="replace")


def _truncate(text: str) -> str:
    """保留首尾。编译/测试输出的关键信息通常在开头（命令、配置）和结尾（错误摘要）。"""
    if len(text) <= MAX_OUTPUT:
        return text
    head = text[: MAX_OUTPUT // 2]
    tail = text[-MAX_OUTPUT // 2 :]
    return f"{head}\n\n... [中间 {len(text) - len(head) - len(tail)} 字符已省略] ...\n\n{tail}"


def _kill_tree(proc: subprocess.Popen) -> None:
    """连子进程一起杀。只 kill 自己会留下 shell 起的孤儿继续跑。"""
    if WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _missing_executable(shell: Shell, stderr: str) -> str | None:
    for pattern in _NOT_FOUND.get(shell.kind, []):
        match = re.search(pattern, stderr)
        if match:
            return match.group(1) if match.groups() else "该命令"
    return None


def make_tools(root: Path, shell: Shell) -> list[Tool]:
    patterns = blocked_patterns(shell)

    def run_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        for pattern, why in patterns:
            if pattern.search(command):
                raise ToolError(f"命令被拦截（{why}）。如果确有必要，请让用户手动执行。")

        kwargs: dict = {
            "cwd": root,
            "stdin": subprocess.DEVNULL,  # 交互式提示会让命令永远等下去
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        proc = subprocess.Popen(shell.argv(command), **kwargs)
        try:
            out, err = proc.communicate(timeout=min(timeout, 300))
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            raise ToolError(
                f"命令超过 {timeout} 秒未结束，已连同子进程一并终止。"
                f"如果是长任务，请加大 timeout 或拆成更小的步骤。"
            )

        stdout, stderr = _decode(out), _decode(err)
        parts = []
        if stdout.strip():
            parts.append(_truncate(stdout.rstrip()))
        if stderr.strip():
            parts.append("[stderr]\n" + _truncate(stderr.rstrip()))
        body = "\n".join(parts) or "（无输出）"

        if proc.returncode != 0:
            missing = _missing_executable(shell, stderr)
            hint = (
                f"\n（{missing} 在当前 PATH 下找不到。它可能装在虚拟环境里，"
                f"或者需要换一个等价命令。）"
                if missing
                else ""
            )
            return f"{EXIT_PREFIX}{proc.returncode}\n{body}{hint}"
        return body

    return [
        Tool(
            name="shell",
            description=describe(shell, root),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": f"秒，默认 {DEFAULT_TIMEOUT}"},
                },
                "required": ["command"],
            },
            run=run_command,
            writes=True,
        )
    ]
