"""终端输入：粘贴的多行文本必须整体成为一次输入。

用 prompt_toolkit 的管道输入注入真实的按键序列，包括 bracketed paste 的
转义标记 —— 那正是终端在粘贴时实际发出的东西。
"""

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from minicode.ui import UI

PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"
ENTER = "\r"


def type_into(text: str) -> str:
    """把一串按键喂给提示符，返回用户「输入」到的内容。"""
    with create_pipe_input() as pipe:
        pipe.send_text(text)
        return UI(ptk_input=pipe, ptk_output=DummyOutput()).prompt()


def test_单行输入():
    assert type_into("给 utils.py 补测试" + ENTER) == "给 utils.py 补测试"


def test_粘贴多行整体成为一次输入():
    """这是换掉 input() 的原因：它一次只读一行，粘贴的后续行会被
    当成用户接着敲的独立命令，依次执行掉。"""
    pasted = "def f():\n    return 1\n\ndef g():\n    return 2"
    assert type_into(PASTE_START + pasted + PASTE_END + ENTER) == pasted


def test_粘贴的内容能和手敲的拼接():
    got = type_into("把这段加进去：" + PASTE_START + "x = 1\ny = 2" + PASTE_END + ENTER)
    assert got == "把这段加进去：x = 1\ny = 2"


def test_粘贴里的制表符和空行保留():
    pasted = "第一行\n\n\t缩进行\n"
    assert type_into(PASTE_START + pasted + PASTE_END + ENTER) == pasted


def test_空输入():
    assert type_into(ENTER) == ""


def test_Ctrl_D_抛EOF():
    """cli 靠这个退出交互模式。"""
    with pytest.raises(EOFError):
        type_into("\x04")


def test_Ctrl_C_抛中断():
    with pytest.raises(KeyboardInterrupt):
        type_into("\x03")


def test_历史文件存不下不影响使用(tmp_path):
    """历史记录是锦上添花，落不了盘也不该拦住会话。"""
    blocked = tmp_path / "file.txt"
    blocked.write_text("我是文件不是目录", encoding="utf-8")
    with create_pipe_input() as pipe:
        pipe.send_text("你好" + ENTER)
        ui = UI(history_path=blocked / "history", ptk_input=pipe, ptk_output=DummyOutput())
        assert ui.prompt() == "你好"


def test_非交互环境不启用增强输入():
    """pytest 下 stdin 不是 tty，正是管道 / 重定向的场景。

    Windows 上 prompt_toolkit 在这种环境下创建就抛异常，
    兜不住的话连启动都做不到 —— Git Bash 里也是同样的形态。
    """
    assert not UI().rich_input


def test_降级后仍然能读到输入(monkeypatch):
    ui = UI()
    assert not ui.rich_input
    monkeypatch.setattr(ui.console, "input", lambda *a, **k: "手敲的一行")
    assert ui.prompt() == "手敲的一行"


def test_降级时banner说明少了什么(capsys):
    """否则用户只会发现「粘贴多行怎么变成好几条命令了」，却不知道为什么。"""
    ui = UI()
    ui.banner("test-model", "/tmp", "写操作需确认", "/tmp/log.jsonl")
    assert "多行粘贴" in capsys.readouterr().out
