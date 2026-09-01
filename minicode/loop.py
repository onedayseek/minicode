"""Agent 主循环：调模型 → 解析 → 执行工具 → 回灌，直到满足某个终止条件。"""

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .compact import CompactionFailed, Compactor, shrank_enough
from .context import Budget, Context
from .errors import (
    ContextLimitError,
    FatalError,
    ProtocolError,
    RetryableError,
    ToolError,
    UserAbort,
)
from .llm import LLMClient
from .parsing import ToolCall, parse_arguments, validate
from .session import SessionLog
from .tools import Shell, build_registry, resolve_shell
from .tools.base import cap_output, extract_intent
from .tools.shell import EXIT_PREFIX

MAX_STEPS = 40
MAX_SECONDS = 15 * 60
NEAR_LIMIT_RATIO = 0.95

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
        parsed = json.loads(call.arguments)
        if isinstance(parsed, dict):
            parsed.pop("intent", None)  # 展示文案变化不代表实际调用发生了变化
        args = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
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
        shell: Shell | None = None, context: Context | None = None,
        compactor: "Compactor | None" = None,
    ) -> None:
        self.llm = llm
        self.root = root
        self.ui = ui
        self.log = log
        self.context = context or Context(system_prompt)
        self.compactor = compactor or Compactor(llm)
        self.seen_files: set[str] = set()
        self.current_step = 0
        self._last_dispatch_rejected = False
        self.shell = shell or resolve_shell()
        self.tools = build_registry(root, self.seen_files, self.shell)

    def run(self, user_input: str) -> bool:
        self.log.user(user_input)
        self.context.add_user(user_input)
        started = time.monotonic()
        step = 0
        stuck = StuckDetector()
        schemas = self.tools.schemas()  # 整个会话内不变

        while True:
            step += 1
            self.current_step = step

            # 两级收敛，低损失的先做。工具输出全都能重新拿到，砍掉不损失
            # 什么；而交接状态是把对话理由压成一段文字，那是有损的。
            messages = self.context.render()
            budget = self.context.measure(messages, schemas)
            elision = self.context.ensure_budget(
                budget,
                force=budget.ratio >= NEAR_LIMIT_RATIO,
                keep_groups=3,
            )
            if elision:
                self.log.context_elision(elision.notice, elision.changes)
                self.ui.notice(elision.notice)
                messages = self.context.render()
                budget = self.context.measure(messages, schemas)

            if budget.needs_checkpoint:
                try:
                    messages, budget = self._compact(messages, budget, schemas)
                except CompactionFailed as e:
                    reason = f"上下文压缩失败，任务停止：{e}"
                    self.log.stop(reason)
                    self.ui.error(reason)
                    return False

            # 进入窗口尾部后，允许清理最近一组工具输出，尽量为本次回复留空间。
            if budget.ratio >= NEAR_LIMIT_RATIO:
                elision = self.context.ensure_budget(budget, force=True, keep_groups=0)
                if elision:
                    self.log.context_elision(elision.notice, elision.changes)
                    self.ui.notice(elision.notice)
                    messages = self.context.render()
                    budget = self.context.measure(messages, schemas)

            stop = self._check_stop(step, started, budget)
            if stop:
                self.log.stop(stop.reason)
                self.ui.notice(stop.reason)
                return False

            # 记录的就是实际发出去的那一份 —— render 只调这一次
            self.log.request(step, messages, schemas, budget)
            self.ui.start_thinking()
            try:
                reply = self._chat_with_budget(
                    messages,
                    schemas,
                    budget,
                    on_text=self.ui.stream,
                    on_retry=self.ui.retry_notice,
                )
            except RetryableError as e:
                self.log.stop(f"网络重试已用尽：{e}")
                self.ui.notice(f"网络重试已用尽：{e}")
                return False
            except ContextLimitError as e:
                self.log.stop(f"上下文错误：{e}")
                self.ui.error(str(e))
                return False
            except ProtocolError as e:
                self.log.stop(f"工具协议错误：{e}")
                self.ui.error(str(e))
                return False
            except FatalError as e:
                self.log.stop(f"致命错误：{e}")
                self.ui.error(str(e))
                return False

            self.ui.end_stream()
            # 实测值回来了，用它校准估算系数，而不是拿它当下一轮的预算
            self.context.calibrate(reply.usage.prompt_tokens, budget.chars)
            self.log.reply(
                step, reply.text, reply.tool_calls, reply.usage,
                reply.finish_reason,
            )
            if reply.finish_reason in ("length", "content_filter"):
                self.ui.add_usage(reply.usage)
                reason = (
                    "模型输出达到长度上限，截断的内容和工具调用均未执行。"
                    if reply.finish_reason == "length"
                    else "模型输出被 provider 的内容过滤器截断，工具调用未执行。"
                )
                # 已经流到屏幕上的那段文本照常进历史，只是不带 tool_calls。
                # 丢掉它会让同一段回复在两条路径上不一致：本进程的下一轮追问
                # 看不到，而从日志 --resume 重建时它又回来了。
                if reply.text:
                    self.context.add_assistant(reply.text, [])
                self.log.stop(reason)
                self.ui.error(reason)
                return False
            self.context.add_assistant(reply.text, reply.tool_calls)
            # 状态栏展示的是“下一次请求会看到的 active context”，不是刚发出去
            # 那一份。回复已经进入历史，因此要在这里重新渲染和估算。
            next_budget = self.context.measure(self.context.render(), schemas)
            self.ui.set_status(step, next_budget, reply.usage)

            # 模型不再请求动作，说明它认为这一轮做完了，把控制权还给用户
            if not reply.tool_calls:
                self.log.stop("自然终止")
                return True

            if self._execute(reply.tool_calls, stuck):
                return False

    def compact_now(self, schemas: list[dict] | None = None) -> str:
        """手动触发收敛，返回给用户看的说明。

        走的是和自动路径一样的两级流水线：先收敛工具输出（无损，重调即可拿回），
        再做交接压缩（有损）。只做第二级的话，在工具输出主导的会话里几乎没有
        效果 —— 大头都在被保护的最近几组里，压缩根本碰不到。

        /compact 是刚需而非锦上添花：窗口大到 1M 时自然触发极难，没有手动入口
        的话，这个能力实际上没法验证也没法演示。
        """
        schemas = self.tools.schemas() if schemas is None else schemas
        before = self.context.measure(self.context.render(), schemas)

        notes = []
        elision = self.context.ensure_budget(before, force=True, keep_groups=1)
        if elision:
            self.log.context_elision(elision.notice, elision.changes)
            notes.append(f"省略了 {len(elision.changes)} 条较早的工具输出")

        try:
            checkpoint = self._run_compactor()
        except CompactionFailed as e:
            after = self.context.measure(self.context.render(), schemas)
            if not notes:
                return str(e)
            return (
                f"{'；'.join(notes)}。{e}"
                f"估算 {before.tokens:,} → {after.tokens:,} tokens"
            )

        self.context.checkpoint = checkpoint
        self.log.checkpoint(checkpoint.summary, checkpoint.covers)
        after = self.context.measure(self.context.render(), schemas)
        notes.append(f"{checkpoint.covers - 1} 条历史收进了交接状态")
        return (
            f"{'；'.join(notes)}。估算 {before.tokens:,} → {after.tokens:,} tokens"
            f"（历史仍完整保留在会话记录里）"
        )

    def _compact(self, messages: list[dict], budget: Budget, schemas: list[dict]):
        """自动压缩。失败或降幅不够都交给 run() 停止当前任务。"""
        self.ui.notice("上下文仍然偏大，正在收敛成交接状态…")
        try:
            checkpoint = self._run_compactor()
        except CompactionFailed as e:
            self.log.system_note(f"压缩未执行：{e}")
            raise

        previous = self.context.checkpoint
        self.context.checkpoint = checkpoint
        new_messages = self.context.render()
        new_budget = self.context.measure(new_messages, schemas)

        if not shrank_enough(budget.tokens, new_budget.tokens):
            # 降幅不够说明这次压缩没有解决预算问题，当前任务不能安全继续。
            self.context.checkpoint = previous
            self.ui.notice("收敛后体量没有明显下降，已放弃压缩。")
            raise CompactionFailed("收敛后体量没有明显下降。")

        self.log.checkpoint(checkpoint.summary, checkpoint.covers)
        self.ui.notice(
            f"已收敛为交接状态：估算 {budget.tokens:,} → {new_budget.tokens:,} tokens"
        )
        return new_messages, new_budget

    def _run_compactor(self):
        """交接状态应当简洁，不使用 provider 的超大输出能力上限。"""
        previous = getattr(self.llm, "max_output", None)
        if previous is not None:
            self.llm.max_output = min(previous, 32_000)
        try:
            return self.compactor.compact(self.context)
        finally:
            usage = getattr(self.compactor, "last_usage", None)
            if usage is not None:
                self.ui.add_usage(usage)
                if hasattr(self.log, "compaction_usage"):
                    self.log.compaction_usage(usage)
            if previous is not None:
                self.llm.max_output = previous

    def _check_stop(self, step: int, started: float, budget: Budget) -> Stop | None:
        if step > MAX_STEPS:
            return Stop(f"已达步数上限（{MAX_STEPS} 步）。可以再发一条消息让它继续。")
        if time.monotonic() - started > MAX_SECONDS:
            return Stop("单次任务已超时，已中断。")
        if not budget.has_room:
            return Stop(
                "上下文已接近模型上限，清理和交接压缩都无法继续。请用 /clear 开新会话。",
                fatal=True,
            )
        return None

    def _chat_with_budget(self, messages, schemas, budget: Budget, **kwargs):
        """给真实客户端设置本次动态输出上限；测试替身无需支持新参数。"""
        if not hasattr(self.llm, "max_output"):
            return self.llm.chat(messages, schemas, **kwargs)
        previous = self.llm.max_output
        self.llm.max_output = budget.output_budget
        try:
            return self.llm.chat(messages, schemas, **kwargs)
        finally:
            self.llm.max_output = previous

    def _execute(self, calls: list[ToolCall], stuck: StuckDetector) -> bool:
        """执行本轮的全部调用。返回 True 表示该终止整个任务。"""
        nudge_reason = None
        for index, call in enumerate(calls):
            try:
                ok = self._dispatch(call)
            except KeyboardInterrupt:
                # assistant 消息已经声明了这一组 tool_calls，其中每一个都必须有对应的
                # tool 结果，否则下一次请求会被 API 以孤儿 tool 消息拒绝。中断路径
                # 也不例外 —— 这个不变量得在循环的每个出口维持住。
                self._close_pending(calls[index:])
                raise

            if self._last_dispatch_rejected:
                # API 要求 assistant 声明的每个 tool_call 都有结果，但不要求真的执行。
                # 用户拒绝已经改变了计划，取消剩余调用并闭合整组，下一轮立刻回给模型。
                self._close_pending(
                    calls[index + 1 :],
                    "[未执行] 用户拒绝了本轮操作，剩余调用已取消。",
                )
                return False

            reason = stuck.record(call, ok)
            if not reason:
                continue
            if stuck.nudged:
                self._close_pending(
                    calls[index + 1 :],
                    "[未执行] 检测到重复失败，本轮剩余调用已取消。",
                )
                self.log.stop(reason)
                self.ui.notice(f"{reason}，已停止。请调整任务描述后重试。")
                return True
            stuck.nudged = True
            nudge_reason = reason
            stuck.forgive(call)
        if nudge_reason:
            note = (
                f"{nudge_reason}。不要再重复同样的调用，换一种思路，"
                "或者直接告诉用户你卡在哪里。"
            )
            self.context.add_system_note(note)
            self.log.system_note(note)
        return False

    def _dispatch(self, call: ToolCall) -> bool:
        """执行一次工具调用，把结果或错误写回历史。返回是否成功。

        任何失败都以 tool result 的形式回灌，而不是抛出 —— 一个工具出错
        不应该终止整个会话，模型往往能自己纠正。
        """
        self._last_dispatch_rejected = False
        try:
            tool = self.tools.get(call.name)
            args = parse_arguments(call)
            intent = extract_intent(args, call.name)
            args = validate(args, tool.parameters, call.name)
            self.ui.tool_start(call.name, args, tool.primary, intent=intent)

            if tool.writes and not self.ui.confirm(call.name, args):
                message = getattr(self.ui, "supplemental_message", "").strip()
                detail = f"。用户补充消息：\n{message}" if message else ""
                raise UserAbort(f"用户拒绝了这次操作{detail}")

            result = tool.run(**args)
            raw_result = getattr(tool, "last_raw_result", None) or result
            result = cap_output(result)

        except UserAbort as e:
            self._last_dispatch_rejected = True
            return self._record(call, "fail", f"[已取消] {e}")
        except ToolError as e:
            return self._record(call, "fail", f"错误：{e}")
        except TypeError as e:
            # 参数名对不上工具签名
            return self._record(call, "fail", f"错误：参数不匹配（{e}）")
        except Exception as e:  # 兜底：任何工具内部异常都不应让 agent 整个退出
            self.log.internal_error(call, traceback.format_exc())
            return self._record(call, "fail", f"错误：{type(e).__name__}: {e}")

        # 命令退出码非零不算工具失败：模型需要读那段输出才能修问题，
        # 也不该让连续的测试失败触发卡死检测。
        return self._record(
            call,
            "warn" if result.startswith(EXIT_PREFIX) else "ok",
            result,
            raw_content=raw_result if raw_result != result else None,
        )

    def _close_pending(
        self,
        calls: list[ToolCall],
        content: str = "[已中断] 用户中断了本次任务，这个调用没有执行。",
    ) -> None:
        """给还没执行的调用补一条结果，让消息组保持闭合。

        会话恢复（session.load_session）处理的是同一个不变量的磁盘版本 ——
        那边只能靠丢弃残缺的组来收场，这边还在内存里，可以直接把缺的补齐。
        """
        for call in calls:
            self._record(call, "fail", content)

    def _record(self, call: ToolCall, status: str, content: str, raw_content: str | None = None) -> bool:
        self.context.add_tool_result(call, content)
        if raw_content is None:
            self.log.tool_result(call, status, content)
        else:
            self.log.tool_result(call, status, content, raw_content=raw_content)
        self.ui.tool_end(call.name, status, content)
        return status != "fail"
