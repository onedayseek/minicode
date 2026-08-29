import json
from pathlib import Path

import pytest

from minicode.session import load_session


def assert_groups_valid(messages: list[dict]) -> None:
    """校验消息数组满足 API 的成组约束。

    role=tool 的消息必须紧跟在带 tool_calls 的 assistant 之后，且 id 对得上。
    不满足时 API 返回 400: "Messages with role 'tool' must be a response to
    a preceding message with 'tool_calls'"。
    """
    open_ids: set[str] = set()
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            open_ids = {c["id"] for c in msg.get("tool_calls", [])}
        elif msg["role"] == "tool":
            assert msg["tool_call_id"] in open_ids, (
                f"第 {i} 条是孤儿 tool 消息（tool_call_id={msg['tool_call_id']}）"
            )
        else:
            open_ids = set()


def write_log(tmp_path: Path, events: list[dict]) -> Path:
    path = tmp_path / "s.jsonl"
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8"
    )
    return path


def reply(*calls, text="", tokens=100):
    return {
        "kind": "reply",
        "step": 1,
        "text": text,
        "prompt_tokens": tokens,
        "tool_calls": [
            {"id": cid, "name": name, "arguments": args} for cid, name, args in calls
        ],
    }


def result(cid, name="read_file", content="内容"):
    return {"kind": "tool_result", "id": cid, "name": name, "status": "ok", "content": content}


START = {"kind": "session_start", "root": ".", "model": "test"}
USER = {"kind": "user", "text": "做事"}


def test_完整一轮原样重建(tmp_path):
    log = write_log(tmp_path, [START, USER, reply(("A", "read_file", '{"path":"a.py"}')), result("A")])
    r = load_session(log)
    assert [m["role"] for m in r.messages] == ["user", "assistant", "tool"]
    assert r.messages[1]["tool_calls"][0]["id"] == "A"
    assert_groups_valid(r.messages)


def test_整组未闭合时剥离(tmp_path):
    log = write_log(tmp_path, [START, USER, reply(("A", "bash", "{}"))])
    r = load_session(log)
    assert all("tool_calls" not in m for m in r.messages)
    assert_groups_valid(r.messages)


def test_半组中断不留孤儿(tmp_path):
    """一组两个调用只回来一个 —— Ctrl-C 打断多工具步骤时的形态。

    此前的实现只剥掉 assistant 的 tool_calls，留下的 tool 消息成为孤儿，
    恢复后第一次请求就被 API 拒绝。
    """
    log = write_log(
        tmp_path,
        [START, USER, reply(("A", "read_file", "{}"), ("B", "bash", "{}")), result("A")],
    )
    r = load_session(log)
    assert_groups_valid(r.messages)
    assert not any(m["role"] == "tool" for m in r.messages)


def test_保留带文本的半组assistant(tmp_path):
    """剥离 tool_calls 后若 assistant 还有文本，应保留文本而非整条丢弃。"""
    log = write_log(
        tmp_path,
        [START, USER, reply(("A", "bash", "{}"), text="我来跑一下测试"), ],
    )
    r = load_session(log)
    assert r.messages[-1]["content"] == "我来跑一下测试"
    assert_groups_valid(r.messages)


def test_归属不明的tool消息被丢弃(tmp_path):
    log = write_log(tmp_path, [START, USER, result("幽灵")])
    r = load_session(log)
    assert not any(m["role"] == "tool" for m in r.messages)
    assert_groups_valid(r.messages)


def test_clear清空已有历史(tmp_path):
    log = write_log(
        tmp_path,
        [START, USER, reply(("A", "read_file", "{}")), result("A"), {"kind": "clear"}, USER],
    )
    r = load_session(log)
    assert [m["role"] for m in r.messages] == ["user"]


def test_坏行被跳过(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps(START) + "\n"
        + "{ 这不是 json\n"
        + '"这是合法 json 但不是对象"\n'
        + "\n"
        + json.dumps(USER) + "\n",
        encoding="utf-8",
    )
    r = load_session(path)
    assert [m["role"] for m in r.messages] == ["user"]


def test_恢复已读文件列表(tmp_path):
    """seen_files 可从 reply 里的调用参数重建，用于恢复 read-before-edit 记录。"""
    log = write_log(
        tmp_path,
        [
            {"kind": "session_start", "root": str(tmp_path), "model": "test"},
            USER,
            reply(("A", "read_file", '{"path": "src/a.py"}'), ("B", "write_file", '{"path": "b.py", "content": "x"}')),
            result("A"),
            result("B", name="write_file"),
        ],
    )
    r = load_session(log)
    assert r.seen_files == {
        str((tmp_path / "src/a.py").resolve()),
        str((tmp_path / "b.py").resolve()),
    }


def test_参数非法不影响重建(tmp_path):
    log = write_log(tmp_path, [START, USER, reply(("A", "read_file", "不是 json")), result("A")])
    r = load_session(log)
    assert r.seen_files == set()
    assert_groups_valid(r.messages)


def test_记录最后的token数(tmp_path):
    log = write_log(
        tmp_path,
        [START, USER, reply(("A", "bash", "{}"), tokens=111), result("A"), reply(tokens=222)],
    )
    assert load_session(log).prompt_tokens == 222
