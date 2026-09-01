"""测试共用的断言与替身。"""


def assert_groups_valid(messages: list[dict]) -> None:
    """校验消息数组满足 API 的成组约束。两个方向都要成立：

    - role=tool 的消息必须紧跟在带 tool_calls 的 assistant 之后，且 id 对得上。
      不满足时 API 返回 400: "Messages with role 'tool' must be a response to
      a preceding message with 'tool_calls'"。
    - 反过来，assistant 声明的每个 tool_call 都必须有对应的 tool 结果，
      少一个下次请求同样被拒。

    这个不变量有两处要维持，所以断言放在 conftest 里共享：
    从磁盘恢复（session.load_session）和内存里被中断（loop.Agent._close_pending）。
    """
    pending: set[str] = set()
    opened_at = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "tool":
            assert msg["tool_call_id"] in pending, (
                f"第 {i} 条是孤儿 tool 消息（tool_call_id={msg['tool_call_id']}）"
            )
            pending.discard(msg["tool_call_id"])
            continue
        assert not pending, f"第 {opened_at} 条 assistant 声明的 {pending} 没有对应的 tool 结果"
        pending = {c["id"] for c in msg.get("tool_calls", [])} if msg["role"] == "assistant" else set()
        opened_at = i
    assert not pending, f"第 {opened_at} 条 assistant 声明的 {pending} 没有对应的 tool 结果"


class SilentUI:
    """吃掉所有渲染。终端长什么样不是这几个测试要验证的东西。"""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.notices: list[str] = []

    def stream(self, chunk: str) -> None: ...
    def end_stream(self) -> None: ...
    def start_thinking(self) -> None: ...
    def stop_thinking(self) -> None: ...
    def retry_notice(self, reason: str, partial: bool) -> None: ...
    def tool_start(self, name: str, args: dict, primary=None) -> None: ...
    def tool_end(self, name: str, status: str, detail: str) -> None: ...
    def set_status(self, *args) -> None: ...
    def add_usage(self, *args) -> None: ...
    def error(self, text: str) -> None: ...

    def confirm(self, name: str, args: dict) -> bool:
        return self.approve

    def notice(self, text: str) -> None:
        self.notices.append(text)


class NullLog:
    """不落盘的会话记录。"""

    def __init__(self) -> None:
        self.stops: list[str] = []
        self.system_notes: list[str] = []
        self.elisions: list[dict] = []
        self.checkpoints: list[dict] = []
        self.internal_errors: list[str] = []

    def user(self, text: str) -> None: ...
    def request(self, *args) -> None: ...
    def reply(self, *args) -> None: ...
    def tool_result(self, call, status: str, content: str, raw_content=None) -> None: ...

    def system_note(self, content: str) -> None:
        self.system_notes.append(content)

    def context_elision(self, notice: str, changes: list[dict]) -> None:
        self.elisions.append({"notice": notice, "changes": changes})

    def checkpoint(self, summary: str, covers: int) -> None:
        self.checkpoints.append({"summary": summary, "covers": covers})

    def internal_error(self, call, traceback_text: str) -> None:
        self.internal_errors.append(traceback_text)

    def stop(self, reason: str) -> None:
        self.stops.append(reason)
