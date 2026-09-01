"""预算与裁剪：估算作用于下一次请求，裁剪只碰能重新拿回来的东西。"""

from conftest import assert_groups_valid
from minicode.context import (
    DEFAULT_CHARS_PER_TOKEN,
    ELIDE_TOKENS,
    KEEP_RECENT_GROUPS,
    Checkpoint,
    Context,
    group_messages,
    payload_chars,
)

SCHEMAS = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]


def tool_msg(index: int, size: int = 300, name: str = "shell") -> dict:
    return {
        "role": "tool",
        "tool_call_id": f"call-{index}",
        "name": name,
        "content": f"result-{index}:" + "x" * size,
    }


def assistant_msg(*call_ids: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "shell", "arguments": "{}"}}
            for cid in call_ids
        ],
    }


def fill(context: Context, groups: int, calls_per_group: int = 1, size: int = 300) -> None:
    index = 0
    for _ in range(groups):
        ids = [f"call-{index + i}" for i in range(calls_per_group)]
        context.messages.append(assistant_msg(*ids))
        for i in range(calls_per_group):
            context.messages.append(tool_msg(index + i, size))
        index += calls_per_group


# ---- 估算 ----


def test_估算把工具schema算进去():
    """schema 不在消息数组里，却是每一次请求都要发的固定开销。"""
    context = Context("system")
    messages = context.render()

    without = context.measure(messages, [])
    with_tools = context.measure(messages, SCHEMAS)

    assert with_tools.chars - without.chars == payload_chars([], SCHEMAS) - payload_chars([], [])
    assert with_tools.tokens > without.tokens


def test_估算系数用实测值校准():
    """比硬编码一个除数可靠：中英文比例和分词器能让系数差出一倍，
    而实测值每轮都能从上一次请求免费拿到。"""
    context = Context("system")
    assert context.chars_per_token == DEFAULT_CHARS_PER_TOKEN

    context.calibrate(actual_tokens=500, chars=2000)

    assert context.chars_per_token == 4.0
    assert context.calibrated


def test_拿不到用量时保持原系数且不声称已校准():
    """provider 不返回 usage 时估算继续工作，但界面上要能看出它没被校准过。"""
    context = Context("system")
    context.calibrate(actual_tokens=0, chars=2000)

    assert context.chars_per_token == DEFAULT_CHARS_PER_TOKEN
    assert not context.calibrated
    assert not context.measure(context.render(), SCHEMAS).calibrated


def test_真实窗口不提前扣掉最大输出能力():
    context = Context("system", window=100_000, max_output=16_000)
    assert context.input_limit == 100_000
    assert context.measure(context.render(), []).limit == 100_000


def test_本次输出预算按剩余窗口动态收紧():
    context = Context("system", window=100_000, max_output=16_000)
    small = context.measure(context.render(), [])
    assert small.output_budget == 16_000

    context.messages.append({"role": "user", "content": "x" * 230_000})
    crowded = context.measure(context.render(), [])
    assert crowded.tokens > 90_000
    assert 0 < crowded.output_budget < 16_000


def test_预算看的是当下的消息而不是上一次的实测值():
    """这是最核心的一条：实测值天生滞后一轮。

    上一次请求 80K 之后，工具吐出的大量输出已经进了历史却还没进过任何一次
    请求；若拿实测值当预算，下一次就会在毫无察觉的情况下发出去一个大得多的
    请求。估算必须针对「即将发出的这一份」。
    """
    context = Context("system", window=1_000_000)
    context.calibrate(actual_tokens=80_000, chars=200_000)  # 上一轮的实测
    before = context.measure(context.render(), SCHEMAS)

    context.messages.append(tool_msg(0, size=400_000))
    after = context.measure(context.render(), SCHEMAS)

    assert before.tokens < 1_000  # 上一轮那 80K 不该出现在这里
    assert after.tokens > 100_000  # 新来的工具输出立刻反映在预算里


# ---- 触发条件 ----

# 让估算恰好落在想要的量级上：默认系数是 2.5 字符/token
CHARS_FOR = lambda tokens: int(tokens * DEFAULT_CHARS_PER_TOKEN)


def test_绝对体量单独就能触发():
    """1M 窗口下按占比永远到不了阈值，机制会变成死代码。"""
    context = Context("system", window=1_000_000)
    fill(context, groups=KEEP_RECENT_GROUPS + 2, size=CHARS_FOR(ELIDE_TOKENS) // 4)
    budget = context.measure(context.render(), SCHEMAS)

    assert budget.tokens >= ELIDE_TOKENS
    assert budget.ratio < 0.7, "这条要验证的是占比没触发时绝对体量仍然管用"
    assert budget.needs_elision
    assert context.ensure_budget(budget) is not None


def test_窗口占比用于checkpoint而不是工具清理():
    """小窗口下占比可以触发 checkpoint，但不会把两种治理动作混为一谈。"""
    context = Context("system", window=20_000, max_output=4_000)
    fill(context, groups=KEEP_RECENT_GROUPS + 2, size=CHARS_FOR(12_000) // 4)
    budget = context.measure(context.render(), SCHEMAS)

    assert budget.tokens < ELIDE_TOKENS, "这条要验证的是绝对体量没触发时占比仍然管用"
    assert budget.ratio >= 0.7
    assert budget.needs_checkpoint
    assert context.ensure_budget(budget) is None


def test_两条都不满足就不动():
    context = Context("system", window=1_000_000)
    fill(context, groups=5)
    budget = context.measure(context.render(), SCHEMAS)

    assert not budget.needs_elision
    assert context.ensure_budget(budget) is None


# ---- 裁剪范围 ----


def test_保护范围按消息组算而不是按条数():
    """一轮调五个工具时，按条数算的保护范围只覆盖一轮；调一个时覆盖三轮。
    同一个设置，效果随模型每轮调几个工具而漂移。
    """
    context = Context("system", window=1_000_000)
    fill(
        context,
        groups=KEEP_RECENT_GROUPS + 1,
        calls_per_group=5,
        size=CHARS_FOR(ELIDE_TOKENS) // 20,
    )
    budget = context.measure(context.render(), SCHEMAS)
    assert budget.needs_elision

    context.ensure_budget(budget)

    groups = group_messages(context.messages)
    protected = {i for g in groups[-KEEP_RECENT_GROUPS:] for i in g}
    for index, msg in enumerate(context.messages):
        if msg["role"] != "tool":
            continue
        is_elided = msg["tool_call_id"] in context.elided
        assert is_elided is (index not in protected), f"第 {index} 条的裁剪与组边界不一致"


def test_收敛只改投影不动历史():
    """这是「历史只增不改」的落点：原文留在 messages 里，短版本只出现在投影里。

    做不到的话，压缩就成了不可逆的销毁 —— 想回看当时读到了什么就只能翻日志。
    """
    context = Context("system", window=1_000_000)
    fill(context, groups=KEEP_RECENT_GROUPS + 2, size=CHARS_FOR(ELIDE_TOKENS) // 4)
    original = [m["content"] for m in context.messages if m["role"] == "tool"]

    context.ensure_budget(context.measure(context.render(), SCHEMAS))

    assert [m["content"] for m in context.messages if m["role"] == "tool"] == original
    projected = [m["content"] for m in context.render() if m["role"] == "tool"]
    assert any("[已省略]" in c for c in projected)
    assert projected != original


def test_投影里的工具结果用短版本():
    context = Context("system", window=1_000_000)
    fill(context, groups=KEEP_RECENT_GROUPS + 2, size=CHARS_FOR(ELIDE_TOKENS) // 4)

    elision = context.ensure_budget(context.measure(context.render(), SCHEMAS))

    assert elision is not None and elision.changes
    projected = {
        m.get("tool_call_id"): m["content"] for m in context.render() if m["role"] == "tool"
    }
    for change in elision.changes:
        assert projected[change["tool_call_id"]] == change["content"]


def test_裁剪过的不会被反复裁剪():
    context = Context("system", window=1_000_000)
    fill(context, groups=KEEP_RECENT_GROUPS + 2, size=CHARS_FOR(ELIDE_TOKENS) // 4)
    budget = context.measure(context.render(), SCHEMAS)

    first = context.ensure_budget(budget)
    second = context.ensure_budget(context.measure(context.render(), SCHEMAS))

    assert first is not None
    assert second is None


# ---- 消息分组 ----


def test_按组切分把工具结果归给声明它的assistant():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        assistant_msg("a", "b"),
        tool_msg(0),
        tool_msg(1),
        {"role": "assistant", "content": "done"},
    ]
    messages[3]["tool_call_id"] = "a"
    messages[4]["tool_call_id"] = "b"

    groups = group_messages(messages)

    assert groups == [[0], [1], [2, 3, 4], [5]]


def test_孤儿工具消息自成一组():
    """恢复出的历史可能有残缺组，切分不能因此把下标弄丢。"""
    messages = [{"role": "system", "content": "s"}, tool_msg(0)]

    groups = group_messages(messages)

    assert groups == [[0], [1]]
    assert sorted(i for g in groups for i in g) == [0, 1]


# ---- 交接状态 ----


def test_交接状态替代掉被覆盖的消息但历史仍在():
    context = Context("system", window=1_000_000)
    fill(context, groups=4)
    total = len(context.messages)

    context.checkpoint = Checkpoint(summary="做完了前两件事", covers=5)
    projected = context.render()

    assert len(context.messages) == total  # 历史一条没少
    assert projected[0] == context.messages[0]  # system 仍在最前
    assert "做完了前两件事" in projected[1]["content"]
    assert len(projected) == 2 + (total - 5)


def test_交接状态以assistant身份进上下文():
    """不用 system —— 那等于把模型自己生成的旧文本提升到系统指令的权限层级，
    后面的用户指令就压不住它了。
    """
    message = Checkpoint(summary="状态", covers=3).as_message()

    assert message["role"] == "assistant"
    assert "状态" in message["content"]


def test_被覆盖范围内的工具输出不再参与收敛():
    """它们已经不发送了，给它们记收敛状态既没收益，也会让变更列表里
    出现根本不在上下文里的条目。"""
    context = Context("system", window=1_000_000)
    fill(context, groups=KEEP_RECENT_GROUPS + 3, size=CHARS_FOR(ELIDE_TOKENS) // 4)
    context.checkpoint = Checkpoint(summary="早期工作", covers=5)

    elision = context.ensure_budget(context.measure(context.render(), SCHEMAS))

    covered = {
        m["tool_call_id"]
        for m in context.messages[1:5]
        if m["role"] == "tool"
    }
    assert covered  # 覆盖范围里确实有工具结果
    assert not (covered & set(context.elided)), "被覆盖的消息不该出现在收敛记录里"
    if elision:
        assert not (covered & {c["tool_call_id"] for c in elision.changes})


def test_投影出来的消息仍然满足成组约束():
    """投影是发给 API 的东西，配对约束在这里同样成立。

    切点落在组中间的话，投影里会出现没有 assistant 声明的孤儿 tool 消息，
    下一次请求直接被拒。这个不变量此前在中断和恢复两处出现过，投影是第三处。
    """
    context = Context("system", window=1_000_000)
    fill(context, groups=4, calls_per_group=3, size=CHARS_FOR(ELIDE_TOKENS) // 12)
    context.ensure_budget(context.measure(context.render(), SCHEMAS))

    groups = group_messages(context.messages)
    for group in groups[1:]:  # 逐个组边界都试一遍
        context.checkpoint = Checkpoint(summary="交接", covers=group[0])
        projected = context.render()
        assert_groups_valid([m for m in projected if m["role"] != "system"])


# ---- 清空 ----


def test_clear同时清掉投影状态和估算状态():
    """否则新会话会带着上一段对话的收敛记录、交接状态和校准系数。"""
    context = Context("system")
    context.messages.append(tool_msg(0))
    context.elided["call-0"] = "短版本"
    context.checkpoint = Checkpoint(summary="旧状态", covers=2)
    context.calibrate(actual_tokens=500, chars=2000)

    context.reset()

    assert context.messages == [{"role": "system", "content": "system"}]
    assert context.elided == {}
    assert context.checkpoint is None
    assert context.chars_per_token == DEFAULT_CHARS_PER_TOKEN
    assert not context.calibrated
    assert context.last_actual_tokens == 0
    assert context.render() == [{"role": "system", "content": "system"}]
