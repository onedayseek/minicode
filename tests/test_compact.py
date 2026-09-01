"""交接式压缩：切点、滚动合并、失败降级。"""

from types import SimpleNamespace

import pytest

from conftest import NullLog, SilentUI, assert_groups_valid
from minicode.compact import (
    KEEP_AFTER_COMPACT,
    CompactionFailed,
    Compactor,
    find_cut,
    shrank_enough,
    transcript,
)
from minicode.context import Checkpoint, Context, group_messages
from minicode.loop import Agent
from minicode.parsing import Reply, ToolCall
from minicode.tools import resolve_shell

LONG = "交接状态正文" * 30  # 够长，不会被最短长度检查拦掉


class FakeLLM:
    """记录收到了什么，按剧本回答。"""

    def __init__(self, *replies, error: Exception | None = None) -> None:
        self.replies = list(replies) or [Reply(text=LONG)]
        self.error = error
        self.calls = []

    def chat(self, messages, tools, on_text=None, on_retry=None):
        self.calls.append({"messages": messages, "tools": tools})
        if self.error:
            raise self.error
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def build_context(groups: int = 8, calls_per_group: int = 2) -> Context:
    context = Context("系统提示", window=1_000_000)
    context.add_user("把这个项目理一遍")
    index = 0
    for _ in range(groups):
        ids = [f"c{index + i}" for i in range(calls_per_group)]
        context.messages.append(
            {
                "role": "assistant",
                "content": "我看看",
                "tool_calls": [
                    {"id": cid, "type": "function",
                     "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}
                    for cid in ids
                ],
            }
        )
        for cid in ids:
            context.messages.append(
                # 要比 ELIDE_KEEP 长，否则收敛会认为它本来就够短、不值得动
                {"role": "tool", "tool_call_id": cid, "name": "read_file", "content": "文件内容" * 100}
            )
        index += calls_per_group
    return context


# ---- 切点 ----


def test_切点落在消息组边界上():
    """落在组中间的话，投影里会出现没有 assistant 声明的孤儿 tool 消息，
    下一次请求直接被 API 拒掉。这个不变量此前在中断和恢复两处出现过。
    """
    context = build_context()
    cut = find_cut(context.messages)

    assert cut in {g[0] for g in group_messages(context.messages)}

    context.checkpoint = Checkpoint(summary=LONG, covers=cut)
    assert_groups_valid([m for m in context.render() if m["role"] != "system"])


def test_历史太短就不压():
    context = Context("系统提示")
    context.add_user("你好")

    assert find_cut(context.messages) == 0

    with pytest.raises(CompactionFailed, match="不足"):
        Compactor(FakeLLM()).compact(context)


def test_切点之后保留足够的原始消息():
    """压缩是有损的，最近这几步正是模型接着要用的，得留够。"""
    context = build_context(groups=10)

    cut = find_cut(context.messages, keep_groups=KEEP_AFTER_COMPACT)

    kept = [g for g in group_messages(context.messages) if g[0] >= cut]
    assert len(kept) == KEEP_AFTER_COMPACT


# ---- 生成 ----


def test_压缩请求不带工具定义():
    """带上的话模型可能返回 tool_calls，而这次请求不在主循环里，
    那组调用永远不会有结果，历史就此破掉。"""
    context = build_context()
    llm = FakeLLM()

    Compactor(llm).compact(context)

    assert llm.calls[0]["tools"] == []


def test_滚动合并把旧状态一起交给模型():
    """既不是每次从头重新总结（旧细节早就不在上下文里了），
    也不是只总结增量（那样旧状态里仍然有效的约束会被丢掉）。"""
    context = build_context()
    context.checkpoint = Checkpoint(summary="用户说过不要动 public API", covers=3)
    llm = FakeLLM()

    Compactor(llm).compact(context)

    sent = llm.calls[0]["messages"][0]["content"]
    assert "不要动 public API" in sent
    assert "仍然有效的信息必须保留" in sent


def test_压缩只读取尚未被覆盖的那一段():
    """已经收进上一份交接状态的部分不该再被送进去重压一遍。"""
    context = Context("系统提示", window=1_000_000)
    context.add_user("这句在覆盖范围之内")
    context.messages.append({"role": "assistant", "content": "早期回答"})
    covered = len(context.messages)
    for i in range(8):
        context.add_user(f"这句在覆盖范围之外 {i}")
        context.messages.append({"role": "assistant", "content": f"后续回答 {i}"})
    context.checkpoint = Checkpoint(summary="旧状态", covers=covered)

    llm = FakeLLM()
    Compactor(llm).compact(context)

    sent = llm.calls[0]["messages"][0]["content"]
    assert "这句在覆盖范围之内" not in sent
    assert "这句在覆盖范围之外" in sent


def test_摘要太短视为失败():
    context = build_context()

    with pytest.raises(CompactionFailed, match="栏目"):
        Compactor(FakeLLM(Reply(text="做完了。"))).compact(context)


def test_被截断的交接状态视为失败():
    context = build_context()

    with pytest.raises(CompactionFailed, match="finish_reason=length"):
        Compactor(FakeLLM(Reply(text=LONG, finish_reason="length"))).compact(context)


def test_网络错误转成可降级的失败():
    context = build_context()
    compactor = Compactor(FakeLLM(error=RuntimeError("连接断了")))

    with pytest.raises(CompactionFailed, match="连接断了"):
        compactor.compact(context)


def test_摊平历史时工具输出只取开头():
    """压缩要提炼的是决策和约束，不是把命令输出再抄一遍 ——
    那些输出本来就能重新拿到。"""
    messages = [
        {"role": "user", "content": "跑测试"},
        {"role": "tool", "tool_call_id": "A", "name": "shell", "content": "x" * 5000},
    ]

    text = transcript(messages)

    assert "跑测试" in text
    assert len(text) < 1000


# ---- 防抖 ----


def test_降幅不够不算成功():
    assert not shrank_enough(100_000, 95_000)
    assert shrank_enough(100_000, 50_000)


# ---- 接进主循环 ----


def agent_with(context: Context, llm, tmp_path, ui=None):
    return Agent(
        llm=llm, root=tmp_path, system_prompt="系统提示",
        ui=ui or SilentUI(), log=NullLog(), shell=resolve_shell(),
        context=context, compactor=Compactor(llm),
    )


class OnlyCompactFails(FakeLLM):
    """压缩请求挂掉，主循环的请求照常。两者靠有没有工具定义区分。"""

    def chat(self, messages, tools, on_text=None, on_retry=None):
        self.calls.append({"messages": messages, "tools": tools})
        if not tools:
            raise RuntimeError("压缩挂了")
        return Reply(text="照常回答")


def crowded_context() -> Context:
    """收敛之后仍然超标的上下文，一定会走进压缩那条路。

    刻意用对话文本撑起来而不是工具输出：工具输出会被第一级收敛掉，压缩就
    轮不上了 —— 那正是两级流水线该有的样子，但这里要测的是第二级。
    """
    context = Context("系统提示", window=20_000, max_output=4_000)
    for i in range(40):
        context.add_user(f"第 {i} 个问题：" + "描述" * 200)
        context.messages.append({"role": "assistant", "content": "回答" * 200})
    return context


def test_压缩失败时任务停止(tmp_path):
    """压缩失败后不能把未经压缩的大请求继续发给模型。"""
    context = crowded_context()
    ui = SilentUI()
    agent = agent_with(context, OnlyCompactFails(), tmp_path, ui)

    assert context.measure(context.render(), agent.tools.schemas()).needs_checkpoint
    assert agent.run("继续") is False
    assert any("上下文仍然偏大" in n for n in ui.notices)


def test_压缩失败后不发送主循环请求(tmp_path):
    """压缩请求失败后当前任务立即停止。"""
    context = crowded_context()

    class MultiStep(FakeLLM):
        """主循环连跑几步，每一步之前压缩都会挂。"""

        def __init__(self) -> None:
            super().__init__()
            self.steps = 0

        def chat(self, messages, tools, on_text=None, on_retry=None):
            self.calls.append({"messages": messages, "tools": tools})
            if not tools:
                raise RuntimeError("压缩挂了")
            self.steps += 1
            if self.steps < 3:
                return Reply(
                    tool_calls=[ToolCall(
                        id=f"t{self.steps}", name="list_files",
                        arguments='{"intent":"查看项目文件"}',
                    )]
                )
            return Reply(text="做完了")

    llm = MultiStep()
    agent = agent_with(context, llm, tmp_path)

    assert agent.run("继续") is False

    compact_calls = sum(1 for c in llm.calls if not c["tools"])
    assert llm.steps == 0
    assert compact_calls == 1


def test_降幅不够就回退到压缩前(tmp_path):
    """压完几乎没变小的话，这次压缩只是白白损失了一段历史的细节。"""
    context = crowded_context()

    class Tiny(FakeLLM):
        def chat(self, messages, tools, on_text=None, on_retry=None):
            self.calls.append({"messages": messages, "tools": tools})
            if not tools:
                return Reply(text="摘要" * 15000)  # 比它替代掉的那段还长
            return Reply(text="照常回答")

    ui = SilentUI()
    agent = agent_with(context, Tiny(), tmp_path, ui)

    agent.run("继续")

    assert context.checkpoint is None, "没效果的压缩不该留下交接状态"
    assert any("没有明显下降" in n for n in ui.notices)


def test_手动收敛走完整的两级流水线(tmp_path):
    """只做交接压缩的话，在工具输出主导的会话里几乎没效果 —— 大头都在被保护的
    最近几组里，压缩够不着。而默认的保护范围和压缩切点几乎重叠，两级会互相挡住，
    所以手动路径要收紧保护范围。

    组数少的时候这个重叠最明显：真实会话常常只有四五组，默认保护三组的话，
    能动的工具输出就只剩最外面那一两条了。
    """
    context = build_context(groups=4, calls_per_group=2)
    total_tools = sum(1 for m in context.messages if m["role"] == "tool")
    agent = agent_with(context, FakeLLM(), tmp_path)
    before = context.measure(context.render(), [])

    message = agent.compact_now([])
    after = context.measure(context.render(), [])

    assert context.checkpoint is not None, "更早的历史应当收进交接状态"
    assert len(context.elided) >= total_tools - 2, (
        f"只收敛了 {len(context.elided)}/{total_tools} 条工具输出，"
        "保护范围没有跟着收紧"
    )
    assert "省略" in message and "交接状态" in message
    assert after.tokens < before.tokens
    # 降幅本身有单独的用例覆盖；小会话里交接状态自己就占掉一块，压不出多少


def test_压缩后存盘再恢复投影一致(tmp_path):
    """整条链路的往返：压缩 → 落盘 → 恢复 → 重新投影。

    覆盖范围这一段最容易错，而且错了不会报错，只会安静地多切或少切几条 ——
    第一版就把「读到 checkpoint 事件时已重建多少条」当成了游标，恢复出来的
    上下文里除了交接状态什么都不剩。
    """
    from minicode.context import Checkpoint
    from minicode.session import SessionLog, load_session

    context = build_context(groups=8, calls_per_group=2)
    log = SessionLog(tmp_path, "test")
    # 把历史按真实形状写进日志
    log.user("把这个项目理一遍")
    for msg in context.messages[2:]:
        if msg["role"] == "assistant":
            calls = [
                SimpleNamespace(id=c["id"], name=c["function"]["name"],
                                arguments=c["function"]["arguments"])
                for c in msg.get("tool_calls") or []
            ]
            log.reply(1, msg["content"], calls, SimpleNamespace(
                prompt_tokens=0, completion_tokens=0, cache_hit_tokens=0, cache_miss_tokens=0))
        else:
            log.tool_result(
                SimpleNamespace(id=msg["tool_call_id"], name=msg["name"]), "ok", msg["content"]
            )

    agent = agent_with(context, FakeLLM(), tmp_path)
    agent.log = log
    agent.compact_now([])

    restored = load_session(log.path)
    revived = Context("系统提示", window=1_000_000)
    revived.messages.extend(restored.messages)
    revived.elided.update(restored.elided)
    summary, covers = restored.checkpoint
    revived.checkpoint = Checkpoint(summary=summary, covers=covers + 1)

    assert [m["role"] for m in revived.render()] == [m["role"] for m in context.render()]
    assert revived.projection_start == context.projection_start
    assert len(revived.messages) == len(context.messages), "历史一条不该少"
    assert_groups_valid([m for m in revived.render() if m["role"] != "system"])


def test_手动压缩装上交接状态并落盘(tmp_path):
    context = build_context()
    llm = FakeLLM()
    log = NullLog()
    agent = Agent(
        llm=llm, root=tmp_path, system_prompt="系统提示", ui=SilentUI(),
        log=log, shell=resolve_shell(), context=context, compactor=Compactor(llm),
    )
    before = len(context.messages)

    message = agent.compact_now([])

    assert context.checkpoint is not None
    assert context.checkpoint.summary == LONG
    assert len(context.messages) == before, "历史一条都不该少"
    assert log.checkpoints == [
        {"summary": LONG, "covers": context.checkpoint.covers}
    ]
    assert "交接状态" in message
    assert_groups_valid([m for m in context.render() if m["role"] != "system"])
