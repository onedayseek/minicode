"""Agent 主循环：调模型 → 解析 → 执行工具 → 回灌，直到满足某个终止条件。"""

import time
from dataclasses import dataclass
from pathlib import Path

from .context import Context
from .errors import FatalError, RetryableError, ToolError, UserAbort
from .llm import LLMClient
from .parsing import ToolCall, parse_arguments, validate
from .tools import build_registry
from .tools.shell import EXIT_PREFIX

MAX_STEPS = 40
MAX_SECONDS = 15 * 60
STUCK_THRESHOLD = 3  # 同一工具连续失败多少次算卡住


@dataclass
class Stop:
    reason: str
    fatal: bool = False


class Agent:
    def __init__(self, llm: LLMClient, root: Path, system_prompt: str, ui) -> None:
        self.llm = llm
        self.root = root
        self.ui = ui
        self.context = Context(system_prompt)
        self.seen_files: set[str] = set()
        self.tools = build_registry(root, self.seen_files)
        self._nudged = False

    def run(self, user_input: str) -> None:
        self.context.add_user(user_input)
        started = time.monotonic()
        step = 0
        failures: dict[str, int] = {}

        while True:
            step += 1
            stop = self._check_stop(step, started)
            if stop:
                self.ui.notice(stop.reason)
                return

            note = self.context.ensure_budget()
            if note:
                self.ui.notice(note)

            try:
                reply = self.llm.chat(
                    self.context.render(), self.tools.schemas(), on_text=self.ui.stream
                )
            except RetryableError as e:
                self.ui.notice(f"网络重试已用尽：{e}")
                return
            except FatalError as e:
                self.ui.error(str(e))
                return

            self.ui.end_stream()
            self.context.prompt_tokens = self.llm.last_usage.prompt_tokens
            self.context.add_assistant(reply.text, reply.tool_calls)

            # 模型不再请求动作，说明它认为这一轮做完了，把控制权还给用户
            if not reply.tool_calls:
                return

            for call in reply.tool_calls:
                ok = self._dispatch(call)
                key = call.name
                failures[key] = 0 if ok else failures.get(key, 0) + 1
                if failures[key] >= STUCK_THRESHOLD:
                    if self._nudged:
                        self.ui.notice(f"{key} 连续失败，已停止。请调整任务描述后重试。")
                        return
                    self._nudged = True
                    self.context.add_system_note(
                        f"工具 {key} 已连续失败 {failures[key]} 次。"
                        f"不要重复同样的调用，请换一种思路，或直接告诉用户你卡在哪里。"
                    )
                    failures[key] = 0

    def _check_stop(self, step: int, started: float) -> Stop | None:
        if step > MAX_STEPS:
            return Stop(f"已达步数上限（{MAX_STEPS} 步）。可以再发一条消息让它继续。")
        if time.monotonic() - started > MAX_SECONDS:
            return Stop("单次任务已超时，已中断。")
        if self.context.usage_ratio() > 0.95:
            return Stop("上下文接近上限，请用 /clear 开新会话。", fatal=True)
        return None

    def _dispatch(self, call: ToolCall) -> bool:
        """执行一次工具调用，把结果或错误写回历史。返回是否成功。

        任何失败都以 tool result 的形式回灌，而不是抛出 —— 一个工具出错
        不应该终止整个会话，模型往往能自己纠正。
        """
        try:
            tool = self.tools.get(call.name)
            args = parse_arguments(call)
            validate(args, tool.parameters, call.name)
            self.ui.tool_start(call.name, args)

            if tool.writes and not self.ui.confirm(call.name, args):
                raise UserAbort("用户拒绝了这次操作")

            result = tool.run(**args)

        except UserAbort as e:
            self.context.add_tool_result(call, f"[已取消] {e}")
            self.ui.tool_end("fail", str(e))
            return False
        except ToolError as e:
            self.context.add_tool_result(call, f"错误：{e}")
            self.ui.tool_end("fail", str(e))
            return False
        except TypeError as e:
            # 参数名对不上工具签名
            self.context.add_tool_result(call, f"错误：参数不匹配（{e}）")
            self.ui.tool_end("fail", str(e))
            return False
        except Exception as e:  # 兜底：任何工具内部异常都不应让 agent 整个退出
            self.context.add_tool_result(
                call, f"错误：{type(e).__name__}: {e}"
            )
            self.ui.tool_end("fail", f"{type(e).__name__}: {e}")
            return False

        self.context.add_tool_result(call, result)
        # 命令退出码非零不算工具失败：模型需要读那段输出才能修问题，
        # 也不该让连续的测试失败触发卡死检测。
        self.ui.tool_end("warn" if result.startswith(EXIT_PREFIX) else "ok", result)
        return True
