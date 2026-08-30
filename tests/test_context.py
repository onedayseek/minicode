"""上下文裁剪产生可落盘、可重放的结构化变更。"""

from minicode.context import Context


def test_裁剪返回每条工具消息的实际替换内容():
    context = Context("system")
    context.prompt_tokens = 50_000
    for index in range(7):
        context.messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "name": "shell",
                "content": f"result-{index}:" + "x" * 300,
            }
        )

    elision = context.ensure_budget()

    assert elision is not None
    assert [c["tool_call_id"] for c in elision.changes] == ["call-0"]
    assert elision.changes[0]["content"] == context.messages[1]["content"]
    assert context.messages[2]["content"].endswith("x" * 300)
