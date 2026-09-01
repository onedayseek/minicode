"""对话历史与上下文管理。

两件事分开：

- 历史：发生过什么。只增不改。
- 上下文：这一轮让模型看到什么。每轮从历史投影出来。

预算作用于「下一次请求的估算」，而不是「上一次请求的实测」。实测值天生滞后
一轮 —— 检查预算时，中间执行的工具输出已经进了历史，却还没进过任何一次请求，
于是一个 80K 的旧数字后面可能跟着一次 180K 的实际请求。

交接式压缩由 compact.py 负责；Context 只维护历史、投影和预算。
"""

import json
from dataclasses import dataclass

from .parsing import ToolCall

# 缺省窗口。不按模型名建映射表 —— 写死的表必然在某个模型上过期，而且是自信地错。
# 默认面向 DeepSeek V4 Flash；用户可通过 MINICODE_CONTEXT_WINDOW 覆盖，并在 banner
# 与 /status 里看到当前值。
DEFAULT_WINDOW = 1_000_000
# 单次回复的能力上限。它不是从上下文窗口中永久扣除的预留；请求前会根据
# 当前 active context 动态计算本次实际允许的输出量。
DEFAULT_MAX_OUTPUT = 384_000

# 工具输出清理使用绝对体量阈值；有损的 checkpoint 单独使用窗口占比阈值。
ELIDE_TOKENS = 120_000
CHECKPOINT_RATIO = 0.7
ELIDE_KEEP = 200  # 省略后为旧工具结果保留的字符数
SAFETY_MARGIN = 4_096
MIN_OUTPUT_BUDGET = 1_024

# 最近几个消息组原样保留，模型正在依赖它们。按组而不是按条：一个组是一条
# assistant 连同它全部 tool 结果，按条数算的话，保护范围会随每轮调用几个工具
# 而漂移 —— 一轮调五个工具时只覆盖一轮，调一个时覆盖三轮。
KEEP_RECENT_GROUPS = 3

# 估算系数的初值，单位是「字符 / token」。首轮没有实测值可用，之后每轮校准。
DEFAULT_CHARS_PER_TOKEN = 2.5


@dataclass
class Elision:
    notice: str
    changes: list[dict]


@dataclass
class Checkpoint:
    """把一段历史收敛成的交接状态。

    covers 是它替代掉的消息条数：投影时 messages[1:covers] 不再发送，
    但它们仍然完整地留在历史里。模型可以忘记，系统不用真的忘记。
    """

    summary: str
    covers: int

    def as_message(self) -> dict:
        """以 assistant 身份进上下文。

        不用 system —— 那等于把模型自己生成的旧文本提升到系统指令的权限
        层级，后面的用户指令就压不住它了。
        """
        return {
            "role": "assistant",
            "content": f"{CHECKPOINT_OPEN}\n{self.summary}\n{CHECKPOINT_CLOSE}",
        }


CHECKPOINT_OPEN = "<交接状态>"
CHECKPOINT_CLOSE = "</交接状态>"


@dataclass
class Budget:
    """对下一次请求体量的估算。"""

    tokens: int
    chars: int  # 事后校准要用
    limit: int  # 模型真实上下文窗口
    output_cap: int  # 本次回复允许的最大值
    calibrated: bool  # 系数是否已被实测校准过

    @property
    def ratio(self) -> float:
        return self.tokens / self.limit if self.limit > 0 else 0.0

    @property
    def needs_elision(self) -> bool:
        return self.tokens >= ELIDE_TOKENS

    @property
    def needs_checkpoint(self) -> bool:
        return self.ratio >= CHECKPOINT_RATIO

    @property
    def output_budget(self) -> int:
        available = self.limit - self.tokens - SAFETY_MARGIN
        return max(0, min(self.output_cap, available))

    @property
    def has_room(self) -> bool:
        return self.output_budget >= MIN_OUTPUT_BUDGET


def group_messages(messages: list[dict]) -> list[list[int]]:
    """把消息按「组」切开，返回每组包含的下标。

    一个组 = 一条 assistant 连同它声明的全部 tool 结果，或者一条独立消息
    （user、system note）。协议要求 tool 结果紧跟在声明它的 assistant 之后，
    所以保护范围、投影切点、压缩切点都只能落在组边界上。

    这个定义在三处要用到，因此集中在这里：中断时补齐未执行的调用、从会话记录
    恢复、以及这里的预算裁剪。tests/conftest.py 的断言也从同一个定义出发。
    """
    groups: list[list[int]] = []
    pending: set[str] = set()
    for index, msg in enumerate(messages):
        if msg.get("role") == "tool" and pending:
            groups[-1].append(index)
            pending.discard(msg.get("tool_call_id"))
            continue
        groups.append([index])
        pending = {c["id"] for c in msg.get("tool_calls") or []}
    return groups


def payload_chars(messages: list[dict], tools: list[dict]) -> int:
    """一次请求的字符量。

    按序列化后的长度算，而不是只数 content：role、键名、tool_calls 的结构
    本身也要过网络、也要计费。工具 schema 尤其不能漏 —— 它不在消息数组里，
    却是每一次请求都要发的固定开销。
    """
    return len(json.dumps(messages, ensure_ascii=False)) + len(
        json.dumps(tools or [], ensure_ascii=False)
    )


class Context:
    def __init__(
        self,
        system_prompt: str,
        window: int = DEFAULT_WINDOW,
        max_output: int = DEFAULT_MAX_OUTPUT,
    ) -> None:
        self.window = window
        self.max_output = max_output
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # 投影状态。历史一律不改，这里记的是「这一轮怎么看它」：
        # 哪些工具结果只发短版本，以及从哪里起用交接状态替代。
        self.elided: dict[str, str] = {}
        self.checkpoint: Checkpoint | None = None
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        self.calibrated = False
        self.last_actual_tokens = 0  # 仅用于展示实测值

    # ---- 写入 ----

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str, tool_calls: list[ToolCall]) -> None:
        msg: dict = {"role": "assistant", "content": text or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in tool_calls
            ]
        self.messages.append(msg)

    def add_tool_result(self, call: ToolCall, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
        )

    def add_system_note(self, text: str) -> None:
        """框架向模型注入的提示，例如『你似乎在原地打转』。"""
        self.messages.append({"role": "system", "content": text})

    # ---- 读取 ----

    @property
    def projection_start(self) -> int:
        """投影从历史的第几条开始。checkpoint 之前的不再发送。"""
        return self.checkpoint.covers if self.checkpoint else 1

    def render(self) -> list[dict]:
        """投影出这一轮要发给模型的消息。

        历史本身一个字都不改：被省略的工具输出仍然完整地躺在 messages 里，
        被 checkpoint 覆盖的消息也一条没删。这里只是决定「这一次让它看到什么」。
        """
        out: list[dict] = [self.messages[0]]
        if self.checkpoint:
            out.append(self.checkpoint.as_message())
        for msg in self.messages[self.projection_start :]:
            short = (
                self.elided.get(msg.get("tool_call_id", ""))
                if msg["role"] == "tool"
                else None
            )
            out.append({**msg, "content": short} if short else msg)
        return out

    def reset(self) -> None:
        del self.messages[1:]
        self.elided.clear()
        self.checkpoint = None
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        self.calibrated = False
        self.last_actual_tokens = 0

    # ---- 预算 ----

    @property
    def input_limit(self) -> int:
        """兼容旧调用方：返回模型真实窗口，不扣除输出上限。"""
        return self.window

    def measure(self, messages: list[dict], tools: list[dict]) -> Budget:
        """估算这一份请求的体量。发出去之前就能知道，不必等它回来。"""
        chars = payload_chars(messages, tools)
        return Budget(
            tokens=int(chars / self.chars_per_token),
            chars=chars,
            limit=self.window,
            output_cap=self.max_output,
            calibrated=self.calibrated,
        )

    def calibrate(self, actual_tokens: int, chars: int) -> None:
        """用刚拿到的实测值更新估算系数。

        比硬编码一个除数可靠得多：中英文比例、模型自己的分词器都能让这个系数
        差出一倍，而它每轮都能从上一次请求免费拿到。拿不到实测值时保持原系数，
        估算继续工作，只是在界面上标明当前是未校准的估算。
        """
        if actual_tokens > 0 and chars > 0:
            self.chars_per_token = chars / actual_tokens
            self.calibrated = True
        self.last_actual_tokens = actual_tokens

    def ensure_budget(
        self,
        budget: Budget,
        force: bool = False,
        keep_groups: int = KEEP_RECENT_GROUPS,
    ) -> Elision | None:
        """超过收敛阈值时省略较早的工具输出，返回可记录的裁剪事件或 None。

        工具输出是最先该丢的一类：它全都能重新拿到 —— 文件可以重读、搜索可以
        重搜、命令可以重跑 —— 丢掉不损失任何拿不回来的东西。用户提过的约束、
        我们试过什么失败了、为什么否决了某个方案，才是只存在于对话里的部分。

        force / keep_groups 供手动收敛用：用户明确要求腾地方时不必再看阈值，
        保护范围也该收紧。默认的保护范围和交接压缩的切点几乎重叠，两级会互相
        挡住 —— 该收敛的工具输出恰好落在保护区里，压缩又够不着它们。

        只改投影，不动历史 —— 原文仍然完整地留在 messages 里。
        """
        if not force and not budget.needs_elision:
            return None

        # 只看投影范围内的消息：checkpoint 之前的已经不发送了，给它们记省略
        # 状态既没有收益，也会让 changes 里出现根本不在上下文里的条目。
        start = self.projection_start
        groups = group_messages(self.messages)
        protected = (
            {i for group in groups[-keep_groups:] for i in group} if keep_groups > 0 else set()
        )

        changes = []
        for index in range(start, len(self.messages)):
            msg = self.messages[index]
            if index in protected or msg["role"] != "tool":
                continue
            call_id = msg.get("tool_call_id", "")
            if call_id in self.elided:
                continue
            content = msg["content"]
            if len(content) <= ELIDE_KEEP:
                continue
            self.elided[call_id] = (
                content[:ELIDE_KEEP]
                + f"\n... 此处省略 {len(content) - ELIDE_KEEP} 字符，"
                f"需要时请重新调用 {msg['name']} 获取 [已省略]"
            )
            changes.append({"tool_call_id": call_id, "content": self.elided[call_id]})

        if changes:
            return Elision(
                f"上下文估算 {budget.tokens:,} tokens，省略了 {len(changes)} 条较早的工具输出",
                changes,
            )
        return None
