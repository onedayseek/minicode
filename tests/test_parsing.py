"""模型输出的解析：流式碎片累积、JSON 修复、schema 校验。

这一层的错误最容易是静默的 —— 参数被解析成了别的东西，工具照样跑完，
返回一个看上去正常的结果。所以每条修复规则都要有用例钉住。
"""

from types import SimpleNamespace

import pytest

from minicode.errors import ToolError
from minicode.parsing import ToolCall, ToolCallAccumulator, parse_arguments, validate

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "limit": {"type": "integer"},
        "ratio": {"type": "number"},
        "replace_all": {"type": "boolean"},
        "tags": {"type": "array"},
    },
    "required": ["path"],
}


def delta(index, id=None, name=None, arguments=None):
    """模拟一个流式 chunk 里的 tool_call 增量。"""
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def parsed(raw: str) -> dict:
    return parse_arguments(ToolCall("1", "read_file", raw))


# ---- 流式累积 ----


def test_按块拼出完整调用():
    acc = ToolCallAccumulator()
    acc.feed([delta(0, id="call_1", name="read_file", arguments='{"pa')])
    acc.feed([delta(0, arguments='th": "a.py"}')])
    calls = acc.finish()
    assert len(calls) == 1
    assert (calls[0].id, calls[0].name) == ("call_1", "read_file")
    assert parse_arguments(calls[0]) == {"path": "a.py"}


def test_index不从零开始():
    """部分兼容网关会让文本 content block 先占掉 index 0。用数组下标存会留空洞。"""
    acc = ToolCallAccumulator()
    acc.feed([delta(1, id="c1", name="grep", arguments="{}")])
    acc.feed([delta(2, id="c2", name="shell", arguments="{}")])
    assert [c.name for c in acc.finish()] == ["grep", "shell"]


def test_name晚于第一个块到达():
    """不能一收到 index 就急着校验工具名。"""
    acc = ToolCallAccumulator()
    acc.feed([delta(0, id="c1")])
    acc.feed([delta(0, name="read_file")])
    acc.feed([delta(0, arguments='{"path":"a.py"}')])
    assert acc.finish()[0].name == "read_file"


def test_没有name的槽位被丢弃():
    """坏块不能污染历史 —— 带着空 name 进到 dispatch 只会换个地方报错。"""
    acc = ToolCallAccumulator()
    acc.feed([delta(0, arguments="{}")])
    acc.feed([delta(1, id="c1", name="grep", arguments="{}")])
    assert [c.name for c in acc.finish()] == ["grep"]


def test_多个调用按index排序():
    acc = ToolCallAccumulator()
    acc.feed([delta(2, id="c2", name="b", arguments="{}"), delta(0, id="c0", name="a", arguments="{}")])
    assert [c.name for c in acc.finish()] == ["a", "b"]


def test_缺id时补一个():
    acc = ToolCallAccumulator()
    acc.feed([delta(0, name="grep", arguments="{}")])
    assert acc.finish()[0].id


def test_空的tool_calls不报错():
    acc = ToolCallAccumulator()
    acc.feed(None)
    acc.feed([])
    assert acc.finish() == []


# ---- JSON 解析与修复 ----


def test_正常JSON():
    assert parsed('{"path": "a.py", "limit": 10}') == {"path": "a.py", "limit": 10}


def test_空参数当成空字典():
    assert parsed("") == {} and parsed("   ") == {}


@pytest.mark.parametrize(
    "raw",
    [
        '```json {"path": "a.py"} ```',  # 压成一行，按 \n 切剥不掉
        '```json\n{"path": "a.py"}\n```',
        '```\n{"path": "a.py"}\n```',
        '```json\n{"path": "a.py"}',  # 结尾围栏被截断
        '```JSON {"path": "a.py"}```',
    ],
)
def test_剥掉代码围栏(raw):
    assert parsed(raw) == {"path": "a.py"}


def test_去掉尾逗号():
    assert parsed('{"path": "a.py",}') == {"path": "a.py"}


def test_补上未闭合的括号():
    assert parsed('{"path": "a.py", "tags": ["x"') == {"path": "a.py", "tags": ["x"]}


def test_按嵌套顺序补括号():
    """只数 { 和 [ 的差额、再各补各的，会得到 ["x"}] 这种照样不合法的结果。"""
    assert parsed('{"a": {"b": ["c"') == {"a": {"b": ["c"]}}


def test_完整字符串里的括号不算数():
    """写代码是最常见的用法，content 里带 { 再正常不过。"""
    got = parsed('{"path": "a.py", "content": "def f() {"}')
    assert got == {"path": "a.py", "content": "def f() {"}


def test_截断在字符串内部时拒绝修复():
    """无法知道字符串后半段是什么，补引号会把半份写入内容当成完整调用执行。"""
    with pytest.raises(ToolError, match="不是合法 JSON"):
        parsed('{"path": "a.py", "content": "def f() {')


def test_转义引号不会被当成字符串结束():
    assert parsed(r'{"old_str": "say \"hi\"", "tags": ["x"')["old_str"] == 'say "hi"'


def test_内容里的反引号不被当成围栏():
    assert parsed('{"path": "a.py", "old_str": "```"}')["old_str"] == "```"


def test_彻底非法时报错且给出下一步():
    with pytest.raises(ToolError) as e:
        parsed("这根本不是 JSON")
    assert "read_file" in str(e.value)  # 告诉它重调哪个工具
    assert "\\n" in str(e.value)  # 以及换行该怎么写


def test_不是对象时报错():
    with pytest.raises(ToolError, match="必须是 JSON 对象"):
        parsed('["a", "b"]')


# ---- schema 校验 ----


def test_缺必填参数():
    with pytest.raises(ToolError, match="缺少必填参数"):
        validate({"limit": 10}, SCHEMA, "read_file")


def test_类型不符():
    with pytest.raises(ToolError, match="应为 integer"):
        validate({"path": "a.py", "limit": "十"}, SCHEMA, "read_file")


def test_bool不能当数字用():
    """Python 里 bool 是 int 的子类，isinstance(True, int) 为真。

    不单独挡掉的话 limit=true 会静默通过，再被当成 1 用 —— 读回一行内容，
    看上去还成功了。静默地做错事比报错难查得多。
    """
    for field in ("limit", "ratio"):
        with pytest.raises(ToolError, match="收到 bool"):
            validate({"path": "a.py", field: True}, SCHEMA, "read_file")


def test_bool本身仍然是合法的boolean():
    validate({"path": "a.py", "replace_all": True}, SCHEMA, "edit_file")


def test_整数可以当number():
    validate({"path": "a.py", "ratio": 1}, SCHEMA, "read_file")


def test_多给的参数被忽略():
    """不值得为模型多塞一个字段就打断它。"""
    got = validate({"path": "a.py", "没这个参数": 1}, SCHEMA, "read_file")
    assert got == {"path": "a.py"}


def test_没写类型的字段放行():
    validate({"x": object()}, {"properties": {"x": {}}}, "t")
