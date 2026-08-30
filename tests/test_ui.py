"""终端呈现中影响判断和审批的关键信息。"""

import io
import re
from types import SimpleNamespace

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
import minicode.ui as ui_module
from minicode.ui import UI

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def visible(output: str) -> list[str]:
    """按用户实际看到的样子拆行。

    断言只针对可见字符：颜色码怎么切分、间距落在哪个 style 段里，是 rich 的
    实现细节，各版本不一样，绑上去的测试红了也说明不了 UI 坏没坏。
    """
    return _ANSI.sub("", output).split("\n")


def body_column(line: str) -> int:
    """`● 正文` 这类行里，正文从第几个可见列开始。"""
    index = line.index("●")
    rest = line[index + 1 :]
    return index + 1 + (len(rest) - len(rest.lstrip(" ")))


def test_工具摘要使用动作名并突出主参数(capsys):
    ui = UI()
    ui.tool_start("read_file", {"path": "src/app.py", "start_line": 10}, "path")

    output = capsys.readouterr().out
    assert "Read(src/app.py" in output
    assert "start_line=10" in output


def test_审批参数完整显示不隐藏中段(capsys):
    ui = UI()
    command = "\n".join(f"line-{i}" for i in range(40))

    ui.tool_start("shell", {"command": command}, "command")

    output = capsys.readouterr().out
    assert "line-0" in output and "line-20" in output and "line-39" in output
    assert "省略" not in output


def test_模型回复和所有工具调用共用同一个主事件标记(capsys):
    ui = UI()

    ui.stream("回复")
    ui.end_stream()
    ui.tool_start("read_file", {"path": "app.py"}, "path")
    ui.tool_start("shell", {"command": "pytest"}, "command")

    assert capsys.readouterr().out.count("●") == 3


def test_模型回复与工具调用的正文起始列相同(capsys):
    """标记后的间距由 ui 自己给出，不靠 Table 的 padding 推导出来。

    推导的结果跟 rich 版本有关：同一份代码在 13.x 下多空一格、在 14/15 下不多，
    错位只在部分环境里出现。这个断言把「两种行必须对齐」这件事本身钉住。
    """
    ui = UI()
    ui.stream("回复正文")
    ui.end_stream()
    ui.tool_start("read_file", {"path": "app.py"}, "path")

    marked = [line for line in visible(capsys.readouterr().out) if "●" in line]
    assert len(marked) == 2
    assert body_column(marked[0]) == body_column(marked[1])


def test_模型回复不带行尾空格(capsys):
    """rich 把 Markdown 段落填充到可用宽度，那些空格复制出去都在。"""
    ui = UI()
    ui.stream("回复正文")
    ui.end_stream()

    for line in visible(capsys.readouterr().out):
        assert line == line.rstrip(" "), f"行尾有多余空格：{line!r}"


def test_剥行尾空格不会削掉代码块的背景():
    """只剥无样式填充；带背景色的代码块空格属于内容的一部分。"""
    plain = Segment("正文" + " " * 20)
    code = Segment("  print(1)      ", Style(bgcolor="blue"))

    assert ui_module._trim_line_padding([plain]) == [Segment("正文")]
    assert ui_module._trim_line_padding([code]) == [code]


def test_模型回复仍走Console原生输出管线():
    """直接 file.write 会绕过 record、旧 Windows renderer 和安全写出逻辑。"""
    output = io.StringIO()
    ui = UI()
    ui.console = Console(file=output, record=True, width=60)

    ui.stream("回复正文")
    ui.end_stream()

    assert "回复正文" in output.getvalue()
    assert "回复正文" in ui.console.export_text()


def test_模型回复支持Markdown渲染(capsys):
    ui = UI()
    ui.stream("## 结果\n\n这是 **重点**。\n\n```python\nprint('ok')\n```")
    ui.end_stream()

    output = capsys.readouterr().out
    for expected in ("结果", "这是 重点。", "print('ok')"):
        assert expected in output
    assert "**" not in output
    assert "```" not in output


def test_分块到达的Markdown按完整回复解析(capsys):
    ui = UI()
    ui.stream("这是 **重")
    ui.stream("点**")
    ui.end_stream()

    output = capsys.readouterr().out
    assert "这是 重点" in output
    assert "**" not in output


def test_未完成的Markdown块不会提前交给解析器(monkeypatch):
    ui = UI()
    real_render = ui_module._markdown_response
    parsed = []

    def tracking_render(text, marker=True):
        parsed.append(text)
        return real_render(text, marker=marker)

    monkeypatch.setattr(ui_module, "_markdown_response", tracking_render)
    ui.stream("链接：[文档](https://")
    assert parsed == []
    ui.stream("example.com)\n\n")
    ui.end_stream()

    assert parsed == ["链接：[文档](https://example.com)"]


def test_已输出的Markdown块不会在后续chunk中重复渲染(capsys):
    ui = UI()
    ui.stream("第一段\n\n")
    ui.stream("第二段\n\n")
    ui.end_stream()

    output = capsys.readouterr().out
    assert output.count("第一段") == 1
    assert output.count("第二段") == 1
    assert output.count("●") == 1


def test_最终Markdown解析失败时降级为原始文本(monkeypatch, capsys):
    ui = UI()
    monkeypatch.setattr(
        ui_module,
        "_markdown_response",
        lambda _text: (_ for _ in ()).throw(IndexError("parser bug")),
    )

    ui.stream("[未闭合](https://")
    ui.end_stream()

    assert "[未闭合](https://" in capsys.readouterr().out


def test_审批支持三选一并记住本会话选择(monkeypatch):
    ui = UI()
    monkeypatch.setattr(ui.console, "input", lambda *_args, **_kwargs: "2")

    assert ui.confirm("shell", {})
    monkeypatch.setattr(
        ui.console,
        "input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应再次询问")),
    )
    assert ui.confirm("shell", {})


def test_审批可以用方向键选择并回车确认():
    with create_pipe_input() as pipe:
        ui = UI(ptk_input=pipe, ptk_output=DummyOutput())
        pipe.send_text("\x1b[B\r")  # 下移到“本会话始终允许”并确认

        assert ui.confirm("shell", {})
        assert "shell" in ui._always


def test_审批可以按Esc拒绝():
    with create_pipe_input() as pipe:
        ui = UI(ptk_input=pipe, ptk_output=DummyOutput())
        pipe.send_text("\x1b")

        assert not ui.confirm("shell", {})


def test_选择拒绝后可以输入给模型的补充消息():
    with create_pipe_input() as pipe:
        ui = UI(ptk_input=pipe, ptk_output=DummyOutput())
        pipe.send_text("\x1b[B\x1b[B\r不要修改生成文件\r")

        assert not ui.confirm("shell", {})
        assert ui.supplemental_message == "不要修改生成文件"


def test_补充消息支持手动换行():
    with create_pipe_input() as pipe:
        ui = UI(ptk_input=pipe, ptk_output=DummyOutput())
        pipe.send_text("\x1b[B\x1b[B\r第一行\x0a第二行\r")

        assert not ui.confirm("shell", {})
        assert ui.supplemental_message == "第一行\n第二行"


def test_补充消息支持外部多行粘贴():
    with create_pipe_input() as pipe:
        ui = UI(ptk_input=pipe, ptk_output=DummyOutput())
        pasted = "第一行\n\n第二行"
        pipe.send_text("\x1b[B\x1b[B\r\x1b[200~" + pasted + "\x1b[201~\r")

        assert not ui.confirm("shell", {})
        assert ui.supplemental_message == pasted


def test_补充消息中无候选时Tab不会产生控制字符():
    with create_pipe_input() as pipe:
        ui = UI(ptk_input=pipe, ptk_output=DummyOutput())
        pipe.send_text("\x1b[B\x1b[B\r前半\t后半\r")

        assert not ui.confirm("shell", {})
        assert ui.supplemental_message == "前半后半"


def test_降级审批也会采集补充消息(monkeypatch):
    answers = iter(("3", "这个命令范围太大"))
    ui = UI()
    monkeypatch.setattr(ui.console, "input", lambda *_args, **_kwargs: next(answers))

    assert not ui.confirm("shell", {})
    assert ui.supplemental_message == "这个命令范围太大"


def test_工具块与最终回复之间固定留一行(capsys):
    ui = UI()
    ui.tool_start("read_file", {"path": "app.py"}, "path")
    ui.tool_end("read_file", "ok", "读取完成")
    ui.stream("任务完成")
    ui.end_stream()

    lines = visible(capsys.readouterr().out)
    result = next(i for i, line in enumerate(lines) if "⎿ 读取完成" in line)
    reply = next(i for i, line in enumerate(lines) if "任务完成" in line)
    assert [line.strip() for line in lines[result + 1 : reply]] == [""]


def test_输入提示与上方内容固定留一行(monkeypatch):
    calls = []
    ui = UI()
    monkeypatch.setattr(ui.console, "print", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(ui.console, "input", lambda *_args, **_kwargs: "下一条消息")

    assert ui.prompt() == "下一条消息"
    assert calls[0] == ()


def test_状态栏显示步骤上下文和本轮用量():
    ui = UI()
    usage = SimpleNamespace(prompt_tokens=1234, completion_tokens=56, cache_hit_tokens=617)

    ui.set_status(3, 0.42, usage)

    assert ui.status_line == "第 3 步 · 上下文 42% · 本轮 1,234 tokens · 缓存命中 50%"


def test_banner集中显示运行环境(capsys):
    ui = UI()
    shell = SimpleNamespace(executable="pwsh", kind="pwsh")

    ui.banner("deepseek-chat", "/work/demo", "写操作需确认", "/work/log.jsonl", shell)

    output = capsys.readouterr().out
    for expected in ("✻ minicode", "deepseek-chat", "/work/demo", "写操作需确认", "pwsh"):
        assert expected in output


def test_恢复历史会显示用户模型和工具轨迹(capsys):
    ui = UI()
    messages = [
        {"role": "user", "content": "先检查项目"},
        {
            "role": "assistant",
            "content": "我先读取 **入口**。",
            "tool_calls": [
                {
                    "id": "A",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"app.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "A",
            "name": "read_file",
            "content": "读取完成",
        },
    ]

    ui.show_history(messages, {"A": "ok"})

    output = capsys.readouterr().out
    for expected in ("已恢复的对话", "❯ 先检查项目", "Read(app.py)", "⎿ 读取完成", "继续会话"):
        assert expected in output
    assert "**" not in output
    # 模型回复带主事件标记，但不断言标记后空了几格 —— 那是 rich 的实现细节
    assert any("●" in line and "我先读取 入口。" in line for line in visible(output))
