"""Agent 主循环：调模型 → 解析 → 执行工具 → 回灌，直到满足某个终止条件。"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .context import Context
from .errors import FatalError, RetryableError, ToolError, UserAbort
from .llm import LLMClient
from .parsing import ToolCall, parse_arguments, validate
from .session import SessionLog
from .tools import Shell, build_registry, resolve_shell
from .tools.shell import EXIT_PREFIX

MAX_STEPS = 40
MAX_SECONDS = 15 * 60

# 卡死的两种形态，阈值分开：一模一样的调用重复发是明确的原地打转，早点打断；
# 同一个工具换着参数失败，可能还在正常试探（连读三个猜错的路径不算卡住），给它更多余地。
STUCK_REPEAT = 2
STUCK_TOOL = 4


@dataclass
class Stop:
    reason: str
    fatal: bool = False


def fingerprint(call: ToolCall) -> tuple[str, str]:
    """一次调用的身份。参数能解析就规范化后再比 —— 键序或空白的差异
    不该让两次实际相同的调用看起来像是不同的尝试。
    """
    try:
        args = json.dumps(json.loads(call.arguments), sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        args = call.arguments.strip()
    return call.name, args


class StuckDetector:
    """判断模型是不是在原地打转。每次 run() 新建，状态不跨任务。

    两个计数的清零规则不同，是刻意的：
    - 按调用指纹计数，别处成功不清零。试了别的又绕回来发一模一样的调用，
      仍然是原地打转 —— 中间那次成功并没有让这次调用变得更可能成功。
    - 按工具名计数，成功就清零。工具能用一次说明它本身没问题，之后的失败重新算。
    """

    def __init__(self) -> None:
        self._repeats: dict[tuple[str, str], int] = {}
        self._by_tool: dict[str, int] = {}
        self.nudged = False

    def record(self, call: ToolCall, ok: bool) -> str | None:
        """记一次调用结果，返回卡死的描述，或 None 表示还算正常。"""
        fp = fingerprint(call)
        if ok:
            self._repeats[fp] = 0
            self._by_tool[call.name] = 0
            return None

        self._repeats[fp] = self._repeats.get(fp, 0) + 1
        self._by_tool[call.name] = self._by_tool.get(call.name, 0) + 1
        if self._repeats[fp] >= STUCK_REPEAT:
            return f"完全相同的 {call.name} 调用已经失败 {self._repeats[fp]} 次"
        if self._by_tool[call.name] >= STUCK_TOOL:
            return f"{call.name} 换着参数连续失败了 {self._by_tool[call.name]} 次"
        return None

    def forgive(self, call: ToolCall) -> None:
        """注入提示后清零计数，给模型一次真正换思路的机会。"""
        self._repeats[fingerprint(call)] = 0
        self._by_tool[call.name] = 0


class Agent:
    def __init__(
        self, llm: LLMClient, root: Path, system_prompt: str, ui, log: SessionLog,
        shell: Shell | None = None,
    ) -> None:
        self.llm = llm
        self.root = root
        self.ui = ui
        self.log = log
        self.context = Context(system_prompt)
        self.seen_files: set[str] = set()
        self.shell = shell or resolve_shell()
        self.tools = build_registry(root, self.seen_files, self.shell)

    def run(self, user_input: str) -> None:
        self.log.user(user_input)
        self.context.add_user(user_input)
        started = time.monotonic()
        step = 0
        stuck = StuckDetector()

        while True:
            step += 1
            stop = self._check_stop(step, started)
            if stop:
                self.log.stop(stop.reason)
                self.ui.notice(stop.reason)
                return

            note = self.context.ensure_budget()
            if note:
                self.ui.notice(note)

            self.log.request(step, self.context.render(), [t["function"]["name"] for t in self.tools.schemas()])
            self.ui.start_thinking()
            try:
                reply = self.llm.chat(
                    self.context.render(),
                    self.tools.schemas(),
                    on_text=self.ui.stream,
                    on_retry=self.ui.retry_notice,
                )
            except RetryableError as e:
                self.log.stop(f"网络重试已用尽：{e}")
                self.ui.notice(f"网络重试已用尽：{e}")
                return
            except FatalError as e:
                self.log.stop(f"致命错误：{e}")
                self.ui.error(str(e))
                return

            self.ui.end_stream()
            self.context.prompt_tokens = self.llm.last_usage.prompt_tokens
            self.log.reply(step, reply.text, reply.tool_calls, self.llm.last_usage)
            self.context.add_assistant(reply.text, reply.tool_calls)
            self.ui.set_status(step, self.context.usage_ratio(), self.llm.last_usage)

            # 模型不再请求动作，说明它认为这一轮做完了，把控制权还给用户
            if not reply.tool_calls:
                self.log.stop("自然终止")
                return

            if self._execute(reply.tool_calls, stuck):
                return

    def _check_stop(self, step: int, started: float) -> Stop | None:
        if step > MAX_STEPS:
            return Stop(f"已达步数上限（{MAX_STEPS} 步）。可以再发一条消息让它继续。")
        if time.monotonic() - started > MAX_SECONDS:
            return Stop("单次任务已超时，已中断。")
        if self.context.usage_ratio() > 0.95:
            return Stop("上下文接近上限，请用 /clear 开新会话。", fatal=True)
        return None

    def _execute(self, calls: list[ToolCall], stuck: StuckDetector) -> bool:
        """执行本轮的全部调用。返回 True 表示该终止整个任务。"""
        for index, call in enumerate(calls):
            try:
                ok = self._dispatch(call)
            except KeyboardInterrupt:
                # assistant 消息已经声明了这一组 tool_calls，其中每一个都必须有对应的
                # tool 结果，否则下一次请求会被 API 以孤儿 tool 消息拒绝。中断路径
                # 也不例外 —— 这个不变量得在循环的每个出口维持住。
                self._close_pending(calls[index:])
                raise

            reason = stuck.record(call, ok)
            if not reason:
                continue
            if stuck.nudged:
                self.log.stop(reason)
                self.ui.notice(f"{reason}，已停止。请调整任务描述后重试。")
                return True
            stuck.nudged = True
            self.context.add_system_note(
                f"{reason}。不要再重复同样的调用，换一种思路，"
                "或者直接告诉用户你卡在哪里。"
            )
            stuck.forgive(call)
        return False

    def _dispatch(self, call: ToolCall) -> bool:
        """执行一次工具调用，把结果或错误写回历史。返回是否成功。

        任何失败都以 tool result 的形式回灌，而不是抛出 —— 一个工具出错
        不应该终止整个会话，模型往往能自己纠正。
        """
        try:
            tool = self.tools.get(call.name)
            args = parse_arguments(call)
            validate(args, tool.parameters, call.name)
            self.ui.tool_start(call.name, args, tool.primary)

            if tool.writes and not self.ui.confirm(call.name, args):
                raise UserAbort("用户拒绝了这次操作")

            result = tool.run(**args)

        except UserAbort as e:
            return self._record(call, "fail", f"[已取消] {e}")
        except ToolError as e:
            return self._record(call, "fail", f"错误：{e}")
        except TypeError as e:
            # 参数名对不上工具签名
            return self._record(call, "fail", f"错误：参数不匹配（{e}）")
        except Exception as e:  # 兜底：任何工具内部异常都不应让 agent 整个退出
            return self._record(call, "fail", f"错误：{type(e).__name__}: {e}")

        # 命令退出码非零不算工具失败：模型需要读那段输出才能修问题，
        # 也不该让连续的测试失败触发卡死检测。
        return self._record(call, "warn" if result.startswith(EXIT_PREFIX) else "ok", result)

    def _close_pending(self, calls: list[ToolCall]) -> None:
        """给还没执行的调用补一条结果，让消息组保持闭合。

        会话恢复（session.load_session）处理的是同一个不变量的磁盘版本 ——
        那边只能靠丢弃残缺的组来收场，这边还在内存里，可以直接把缺的补齐。
        """
        for call in calls:
            self._record(call, "fail", "[已中断] 用户中断了本次任务，这个调用没有执行。")

    def _record(self, call: ToolCall, status: str, content: str) -> bool:
        self.context.add_tool_result(call, content)
        self.log.tool_result(call, status, content)
        self.ui.tool_end(call.name, status, content)
        return status != "fail"
