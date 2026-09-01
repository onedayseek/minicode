"""主循环：中断时的历史闭合，以及卡死检测的粒度。"""

import json

import pytest

from conftest import NullLog, SilentUI, assert_groups_valid
from minicode.context import Context
from minicode.loop import Agent, StuckDetector, fingerprint
from minicode.parsing import Reply, ToolCall, Usage
from minicode.tools import resolve_shell


class ScriptedLLM:
    """按剧本逐轮返回，不发真实请求。"""

    def __init__(self, replies: list[Reply]) -> None:
        self._replies = list(replies)

    @property
    def remaining(self) -> int:
        return len(self._replies)

    def chat(self, messages, tools, on_text=None, on_retry=None) -> Reply:
        return self._replies.pop(0)


def call(cid: str, name: str = "read_file", arguments: str = '{"path":"没有这个文件.py"}'):
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, dict):
            parsed.setdefault("intent", "测试这次工具调用")
            arguments = json.dumps(parsed, ensure_ascii=False)
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


def test_intent只用于展示不会传给本地工具(agent):
    seen = {}
    tool = agent.tools.get("read_file")
    tool.run = lambda **kwargs: (seen.update(kwargs), "读取完成")[1]

    assert agent._dispatch(call(
        "A", "read_file",
        '{"intent":"确认配置来源","path":"config.py"}',
    )) is True

    assert seen == {"path": "config.py"}


def test_缺少intent时不执行工具(agent):
    executed = []
    agent.tools.get("read_file").run = lambda **kwargs: executed.append(kwargs) or "不该执行"
    missing = ToolCall("A", "read_file", '{"path":"config.py"}')

    assert agent._dispatch(missing) is False

    assert executed == []
    assert "intent" in agent.context.messages[-1]["content"]


def test_拒绝后取消本组剩余调用并立刻回给模型(agent):
    calls = [
        call("A", "shell", '{"command":"Remove-Item generated -Recurse"}'),
        call("B", "read_file", '{"path":"随后不该读取.py"}'),
    ]
    agent.context.add_assistant("准备执行", calls)
    agent.ui.approve = False

    assert agent._execute(calls, StuckDetector()) is False

    results = [m for m in agent.context.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["A", "B"]
    assert "已取消" in results[0]["content"]
    assert "剩余调用已取消" in results[1]["content"]
    assert_groups_valid(agent.context.messages)


# ---- 预算与请求 ----


def test_收敛排在终止判断前面(tmp_path):
    """反过来的话，单步涨得猛会直接撞上硬停，而本来只要省掉几条旧工具输出就能继续。"""
    context = Context("系统提示", window=20_000, max_output=4_000)
    for i in range(10):
        context.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"c{i}", "type": "function",
                     "function": {"name": "shell", "arguments": "{}"}}
                ],
            }
        )
        context.messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "name": "shell", "content": "x" * 6000}
        )

    ui = SilentUI()
    agent = Agent(
        llm=ScriptedLLM([Reply(text="接着做完了")]), root=tmp_path,
        system_prompt="系统提示", ui=ui, log=NullLog(),
        shell=resolve_shell(), context=context,
    )

    assert agent.context.measure(context.render(), []).ratio > 0.95  # 已经过了硬停线
    assert agent.run("继续") is True  # 收敛之后仍然跑得下去
    assert any("省略" in n for n in ui.notices)


def test_日志记录的就是实际发出去的那一份(agent):
    """投影生效后，两次 render 之间上下文可能已经变了。
    记一份、发另一份的话，事后照着日志根本复现不出模型当时看到的东西。
    """
    logged, sent, rendered = [], [], []

    class Spy(ScriptedLLM):
        def chat(self, messages, tools, on_text=None, on_retry=None):
            sent.append(messages)
            return super().chat(messages, tools, on_text, on_retry)

    real_render = agent.context.render
    agent.context.render = lambda: (rendered.append(1), real_render())[1]
    agent.llm = Spy([Reply(text="做完了")])
    agent.log.request = lambda step, messages, tools, budget=None: logged.append(messages)

    agent.run("任务")

    assert len(rendered) == 2, "一次用于实际请求，一次用于刷新 next active context"
    assert logged[0] is sent[0]


def test_用量校准用的是本次请求的字符数(agent):
    """校准配错请求的话，系数会朝错误方向漂，而且不会有任何报错。"""
    agent.llm = ScriptedLLM([Reply(text="做完了", usage=Usage(prompt_tokens=400))])

    agent.run("任务")

    assert agent.context.calibrated
    assert agent.context.last_actual_tokens == 400


def test_每次请求按剩余窗口动态收紧输出上限(tmp_path):
    context = Context("系统提示", window=100_000, max_output=80_000)
    context.add_user("x" * 100_000)

    class BoundedLLM(ScriptedLLM):
        def __init__(self):
            super().__init__([Reply(text="完成")])
            self.max_output = 80_000
            self.seen_limits = []

        def chat(self, messages, tools, on_text=None, on_retry=None):
            self.seen_limits.append(self.max_output)
            return super().chat(messages, tools, on_text, on_retry)

    llm = BoundedLLM()
    agent = Agent(
        llm=llm, root=tmp_path, system_prompt="系统提示", ui=SilentUI(),
        log=NullLog(), shell=resolve_shell(), context=context,
    )

    assert agent.run("继续") is True
    assert 0 < llm.seen_limits[0] < 80_000
    assert llm.max_output == 80_000


# ---- 卡死检测 ----


def test_调用指纹忽略键序与空白():
    a = call("1", "edit_file", '{"path": "a.py", "old_str": "x"}')
    b = call("2", "edit_file", '{"old_str":"x","path":"a.py"}')
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(call("3", "edit_file", '{"path":"b.py","old_str":"x"}'))


def test_调用指纹忽略intent措辞变化():
    a = call("1", "read_file", '{"intent":"读取配置","path":"a.py"}')
    b = call("2", "read_file", '{"intent":"确认配置来源","path":"a.py"}')
    assert fingerprint(a) == fingerprint(b)


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


def test_长度截断仍然保留已经说出口的那段话(agent):
    """被截断的是工具调用，不是那段文本 —— 它已经流到用户屏幕上了。

    丢掉它会让同一段回复在两条路径上不一致：本进程的下一轮追问看不到，
    而从会话记录 --resume 重建时它又回来了。
    """
    agent.llm = ScriptedLLM([
        Reply(
            text="我准备分三步来做，第一步是",
            tool_calls=[call("A", "write_file", '{"path":"x.py","content":"half')],
            finish_reason="length",
        )
    ])

    assert agent.run("写文件") is False
    assistant = [m for m in agent.context.messages if m["role"] == "assistant"]
    assert [m["content"] for m in assistant] == ["我准备分三步来做，第一步是"]
    assert "tool_calls" not in assistant[0]
    assert_groups_valid(agent.context.messages)


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
