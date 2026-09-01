"""交接式压缩：把一段历史收敛成可接手的状态。

单独一个模块，不放进 Context —— Context 只管「如何看历史」，它不该知道
LLM 的存在。什么时候压缩由 Agent 决定，怎么压缩在这里。
"""

import json
from pathlib import Path

from .context import CHECKPOINT_CLOSE, CHECKPOINT_OPEN, Checkpoint, Context, group_messages
from .errors import MinicodeError

# 摘要本身也要占地方，太短说明模型没照着栏目写，多半不可用
MIN_SUMMARY_CHARS = 80
# 压缩后至少要降下这么多，否则不值得再来一次
MIN_SHRINK_RATIO = 0.2
# 尾部保留的消息组数。比日常收敛留得多一些：压缩是有损的，
# 而最近这几步正是模型接着要用的。
KEEP_AFTER_COMPACT = 4


class CompactionFailed(MinicodeError):
    """压缩没成功。调用方应当降级，而不是让任务中断。"""


def load_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "compact.md").read_text(encoding="utf-8")


def find_cut(messages: list[dict], keep_groups: int = KEEP_AFTER_COMPACT) -> int:
    """选一个切点：保留最后若干个消息组，其余交给压缩。

    切点必须落在组边界上。落在组中间的话，投影里会出现没有 assistant 声明的
    孤儿 tool 消息，下一次请求直接被 API 拒掉。返回 0 表示不值得压。
    """
    groups = group_messages(messages)
    if len(groups) <= keep_groups + 1:  # 除去 system 那一组，没剩下什么可压的
        return 0
    cut = groups[-keep_groups][0]
    return cut if cut > 1 else 0


def transcript(messages: list[dict]) -> str:
    """把要压缩的那段历史摊平成文本。

    工具结果只给开头 —— 压缩要提炼的是决策和约束，不是把命令输出再抄一遍，
    而那些输出本来就能重新拿到。
    """
    lines = []
    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "user":
            lines.append(f"[用户] {content}")
        elif role == "assistant":
            if content:
                lines.append(f"[助手] {content}")
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {})
                lines.append(f"[调用] {fn.get('name')} {fn.get('arguments', '')[:200]}")
        elif role == "tool":
            lines.append(f"[结果] {msg.get('name')}: {content[:300]}")
        elif role == "system":
            lines.append(f"[框架提示] {content}")
    return "\n".join(lines)


class Compactor:
    """持有 LLM，负责生成交接状态。"""

    def __init__(self, llm, prompt: str | None = None) -> None:
        self.llm = llm
        self.prompt = prompt if prompt is not None else load_prompt()
        self.last_usage = None

    def compact(self, context: Context, keep_groups: int = KEEP_AFTER_COMPACT) -> Checkpoint:
        """把 context 里较早的一段收敛成新的交接状态。

        滚动进行：已有的交接状态连同它之后的新历史一起交给模型，要求整合成
        一份新的。既不是每次从头重新总结（旧的细节早就不在上下文里了），
        也不是只总结增量（那样旧状态里仍然有效的约束会被丢掉）。
        """
        self.last_usage = None
        cut = find_cut(context.messages, keep_groups)
        if cut <= context.projection_start:
            raise CompactionFailed("可压缩的历史不足，再多做几步吧。")

        body = transcript(context.messages[context.projection_start : cut])
        previous = context.checkpoint.summary if context.checkpoint else ""
        summary = self._ask(body, previous)
        return Checkpoint(summary=summary, covers=cut)

    def _ask(self, body: str, previous: str) -> str:
        parts = [self.prompt]
        if previous:
            parts.append(
                "下面是上一次生成的交接状态。其中仍然有效的信息必须保留到新状态里，"
                "已经过时或已被后续工作推翻的才可以更新；不要原样照抄，也不要凭空丢弃。\n\n"
                f"{CHECKPOINT_OPEN}\n{previous}\n{CHECKPOINT_CLOSE}"
            )
        parts.append(f"这是需要收敛的会话历史：\n\n{body}")

        try:
            # 不带工具定义：带上的话模型可能返回 tool_calls，而这次请求不在
            # 主循环里，那组调用永远不会有结果，历史就此破掉。
            reply = self.llm.chat(
                [{"role": "user", "content": "\n\n".join(parts)}],
                [],
            )
        except Exception as e:  # 网络、限流、协议，一律降级处理
            raise CompactionFailed(f"生成交接状态失败：{e}") from e

        self.last_usage = reply.usage
        if getattr(reply, "finish_reason", None) in ("length", "content_filter"):
            raise CompactionFailed(
                f"模型输出未完整结束（finish_reason={reply.finish_reason}）。"
            )
        summary = (reply.text or "").strip()
        if len(summary) < MIN_SUMMARY_CHARS:
            raise CompactionFailed("模型没有按栏目给出交接状态，本次压缩作废。")
        return summary


def shrank_enough(before: int, after: int) -> bool:
    """压缩后降幅够不够。

    不看这一条的话会来回抖：刚过阈值一点点就压一次，压完只降了几个百分点，
    过两步又超过阈值，于是每两步压一次，每次都花一次请求还没什么效果。
    """
    return before > 0 and (before - after) / before >= MIN_SHRINK_RATIO
