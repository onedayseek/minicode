"""主循环：中断时的历史闭合，以及卡死检测的粒度。"""

import pytest

from conftest import NullLog, SilentUI, assert_groups_valid
from minicode.llm import Usage
from minicode.loop import Agent, StuckDetector, fingerprint
from minicode.parsing import Reply, ToolCall
from minicode.tools import resolve_shell


class ScriptedLLM:
    """按剧本逐轮返回，不发真实请求。"""

    def __init__(self, replies: list[Reply]) -> None:
        self._replies = list(replies)
        self.last_usage = Usage()

    @property
    def remaining(self) -> int:
        return len(self._replies)

    def chat(self, messages, tools, on_text=None, on_retry=None) -> Reply:
        return self._replies.pop(0)


def call(cid: str, name: str = "read_file", arguments: str = '{"path":"没有这个文件.py"}'):
    return ToolCall(id=cid, name=name, arguments=arguments)


def replying(*calls: ToolCall) -> Reply:
    return Reply(tool_calls=list(calls))


@pytest.fixture
def agent(tmp_path):
    return Agent(
        llm=None, root=tmp_path, system_prompt="系统提示",
        ui=SilentUI(), log=NullLog(), shell=resolve_shell(),
    )


# ---- 中断时的消息组闭合 ----


def test_中断时补齐未执行的调用(agent):
    """Ctrl-C 落在一组调用的中间，历史里不能留下没有结果的 tool_call。

    KeyboardInterrupt 继承自 BaseException，_dispatch 的兜底 except Exception
    接不住它，所以要在 _execute 这一层显式收口。
    """
    calls = [call("A"), call("B", "shell", '{"command":"echo hi"}'), call("C", "grep", '{"pattern":"x"}')]
    agent.context.add_assistant("我来看几个地方", calls)

    real = agent._dispatch
    agent._dispatch = lambda c: (_ for _ in ()).throw(KeyboardInterrupt) if c.id == "B" else real(c)

    with pytest.raises(KeyboardInterrupt):
        agent._execute(calls, StuckDetector())

    assert_groups_valid(agent.context.messages)
    results = [m for m in agent.context.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["A", "B", "C"]
    assert all("[已中断]" in m["content"] for m in results if m["tool_call_id"] in ("B", "C"))


def test_中断发生在第一个调用上(agent):
    """一个结果都没来得及产生时，整组都要补齐。"""
    calls = [call("A"), call("B")]
    agent.context.add_assistant("", calls)
    agent._dispatch = lambda c: (_ for _ in ()).throw(KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        agent._execute(calls, StuckDetector())
    assert_groups_valid(agent.context.messages)


def test_正常执行完历史也是闭合的(agent):
    calls = [call("A"), call("B")]
    agent.context.add_assistant("", calls)
    agent._execute(calls, StuckDetector())
    assert_groups_valid(agent.context.messages)


def test_用户补充消息会作为工具结果回灌给模型(agent):
    rejected = call("R", "shell", '{"command":"Remove-Item generated -Recurse"}')
    agent.context.add_assistant("准备清理", [rejected])
    agent.ui.approve = False
    agent.ui.supplemental_message = "generated 目录需要保留"

    assert not agent._dispatch(rejected)

    result = agent.context.messages[-1]
    assert result["role"] == "tool"
    assert "用户补充消息" in result["content"]
    assert "generated 目录需要保留" in result["content"]


# ---- 卡死检测 ----


def test_调用指纹忽略键序与空白():
    a = call("1", "edit_file", '{"path": "a.py", "old_str": "x"}')
    b = call("2", "edit_file", '{"old_str":"x","path":"a.py"}')
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(call("3", "edit_file", '{"path":"b.py","old_str":"x"}'))


def test_参数不是合法JSON时指纹退回原文():
    assert fingerprint(call("1", "shell", "不是 json")) == ("shell", "不是 json")


def test_相同调用重复失败先提示再终止(agent):
    stuck = StuckDetector()
    assert agent._execute([call("A")], stuck) is False
    assert agent._execute([call("B")], stuck) is False  # 第 2 次：注入提示，先不终止
    assert stuck.nudged
    assert any(
        m["role"] == "system" and "换一种思路" in m["content"] for m in agent.context.messages[1:]
    )

    assert agent._execute([call("C")], stuck) is False  # 提示后计数清零，再给一次机会
    assert agent._execute([call("D")], stuck) is True  # 还是原封不动的调用，终止


def test_多调用中触发提示时先闭合整组(agent):
    calls = [call("A"), call("B", "grep", '{"pattern":"x"}')]
    agent.context.add_assistant("", calls)
    stuck = StuckDetector()
    stuck._repeats[fingerprint(calls[0])] = 1

    assert agent._execute(calls, stuck) is False
    assert_groups_valid(agent.context.messages)
    assert [m["role"] for m in agent.context.messages[-3:]] == ["tool", "tool", "system"]
    assert agent.log.system_notes


def test_多调用中决定终止时补齐剩余结果(agent):
    calls = [call("A"), call("B", "grep", '{"pattern":"x"}')]
    agent.context.add_assistant("", calls)
    stuck = StuckDetector()
    stuck.nudged = True
    stuck._repeats[fingerprint(calls[0])] = 1

    assert agent._execute(calls, stuck) is True
    assert_groups_valid(agent.context.messages)
    results = [m for m in agent.context.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["A", "B"]
    assert "未执行" in results[-1]["content"]


def test_长度截断的工具调用不会执行(agent):
    agent.llm = ScriptedLLM([
        Reply(
            tool_calls=[call("A", "write_file", '{"path":"x.py","content":"half')],
            finish_reason="length",
        )
    ])

    assert agent.run("写文件") is False
    assert not any(m["role"] == "assistant" for m in agent.context.messages)
    assert not any(m["role"] == "tool" for m in agent.context.messages)


def test_工具内部异常的traceback写入日志(agent):
    tool = agent.tools.get("read_file")
    tool.run = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("内部坏了"))

    assert agent._dispatch(call("A", arguments='{"path":"a.py"}')) is False
    assert "RuntimeError: 内部坏了" in agent.log.internal_errors[-1]


def test_换参数的失败不算原地打转(agent):
    """连读三个猜错的路径是正常试探。按工具名计数会把这种情况误判成卡死。"""
    stuck = StuckDetector()
    for name in ("a.py", "b.py", "c.py"):
        assert agent._execute([call("X", arguments=f'{{"path":"{name}"}}')], stuck) is False
    assert not stuck.nudged


def test_成功会清零工具级计数(agent, tmp_path):
    """同一工具换着参数失败，中间成功一次就重新计数 —— 工具本身是能用的。"""
    (tmp_path / "真的存在.py").write_text("x = 1\n", encoding="utf-8")
    stuck = StuckDetector()
    for name in ("a.py", "b.py", "c.py"):  # 3 次不同的失败，还差一次到 STUCK_TOOL
        agent._execute([call("X", arguments=f'{{"path":"{name}"}}')], stuck)
    agent._execute([call("Y", arguments='{"path":"真的存在.py"}')], stuck)

    assert agent._execute([call("Z", arguments='{"path":"d.py"}')], stuck) is False
    assert not stuck.nudged  # 不清零的话，这里已经是第 4 次了


def test_中间做成过别的事不赦免重复的调用(agent, tmp_path):
    """试了别的又绕回来发同一个失败调用，仍然算原地打转 ——
    中间那次成功没有让这次调用变得更可能成功。"""
    (tmp_path / "真的存在.py").write_text("x = 1\n", encoding="utf-8")
    stuck = StuckDetector()
    agent._execute([call("A")], stuck)
    agent._execute([call("B", arguments='{"path":"真的存在.py"}')], stuck)
    agent._execute([call("C")], stuck)  # 和 A 一模一样
    assert stuck.nudged


def test_卡死状态不跨任务(agent):
    """上一个任务用掉的『提示机会』不该让下一个任务一失败就被掐断。"""
    agent.llm = ScriptedLLM(
        [replying(call("A"))] * 4  # 第一个任务：失败到终止
        + [replying(call("B"))] * 2  # 第二个任务：同样失败两次
        + [Reply(text="我卡住了")]  # 若状态跨任务，这条永远不会被取用
    )
    agent.run("第一个任务")
    agent.run("第二个任务")
    assert agent.llm.remaining == 0
