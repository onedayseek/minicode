import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import assert_groups_valid
from minicode.session import SessionLog, iter_events, load_session


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


def test_恢复累计API用量包含普通回复和压缩请求(tmp_path):
    log = write_log(
        tmp_path,
        [
            START,
            {**reply(text="完成", tokens=100), "completion_tokens": 20, "cache_hit_tokens": 60},
            {
                "kind": "compaction_usage",
                "prompt_tokens": 40,
                "completion_tokens": 10,
                "cache_hit_tokens": 30,
            },
        ],
    )

    restored = load_session(log)

    assert restored.prompt_tokens == 140
    assert restored.completion_tokens == 30
    assert restored.cache_hit_tokens == 90


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


def test_clear同时清空投影状态(tmp_path):
    """否则恢复出的新会话会带着上一段对话的收敛记录和交接状态。"""
    log = write_log(
        tmp_path,
        [
            START,
            USER,
            reply(("A", "read_file", '{"path":"a.py"}')),
            result("A", content="完整内容"),
            {
                "kind": "context_elision",
                "notice": "省略一条",
                "changes": [{"tool_call_id": "A", "content": "开头…"}],
            },
            {"kind": "checkpoint", "summary": "旧的交接状态", "covers": 3},
            {"kind": "clear"},
            USER,
        ],
    )
    restored = load_session(log)

    assert restored.elided == {}
    assert restored.checkpoint is None
    assert [m["role"] for m in restored.messages] == ["user"]


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


def test_失败或未返回的读取不恢复为已读(tmp_path):
    log = write_log(
        tmp_path,
        [
            {"kind": "session_start", "root": str(tmp_path), "model": "test"},
            USER,
            reply(
                ("A", "read_file", '{"path":"failed.py"}'),
                ("B", "read_file", '{"path":"pending.py"}'),
            ),
            {**result("A"), "status": "fail"},
        ],
    )
    assert load_session(log).seen_files == set()


def test_框架提示与上下文裁剪可重建(tmp_path):
    """裁剪恢复成投影状态，历史里的原文不动。

    早先是直接把短版本覆盖进消息内容 —— 那样恢复出来的历史就是残缺的，
    磁盘上明明还留着完整输出，内存里却已经找不回来了。
    """
    shortened = "开头\n... 已省略"
    log = write_log(
        tmp_path,
        [
            START,
            USER,
            reply(("A", "read_file", '{"path":"a.py"}')),
            result("A", content="完整内容"),
            {
                "kind": "context_elision",
                "notice": "省略一条",
                "changes": [{"tool_call_id": "A", "content": shortened}],
            },
            {"kind": "system_note", "content": "换一种思路"},
        ],
    )
    restored = load_session(log)

    assert restored.elided == {"A": shortened}
    assert restored.messages[-2]["content"] == "完整内容"  # 历史保留原文
    assert restored.messages[-1] == {"role": "system", "content": "换一种思路"}


def test_交接范围用事件里记的值而不是事件的位置(tmp_path):
    """压缩总是发生在历史已经写了一大截之后。

    拿「读到 checkpoint 事件时已重建出多少条」当游标，覆盖范围就会变成整段
    历史 —— 恢复出来的上下文里除了交接状态什么都不剩，而它本该还留着最近几组。
    """
    log = write_log(
        tmp_path,
        [
            START,
            USER,
            reply(text="第一轮"),
            USER,
            reply(text="第二轮"),
            USER,
            reply(text="第三轮"),
            # 压缩发生在此刻，但只覆盖前面两条
            {"kind": "checkpoint", "summary": "做完了开头那部分", "covers": 3},
        ],
    )
    restored = load_session(log)

    assert restored.checkpoint == ("做完了开头那部分", 2)
    assert len(restored.messages) == 6  # 历史一条没少


def test_丢弃残缺组后交接范围跟着前移(tmp_path):
    """剥掉半组会让下标整体前移，覆盖范围不跟着调整就会多切或少切几条。"""
    log = write_log(
        tmp_path,
        [
            START,
            USER,
            reply(("A", "read_file", "{}")),  # 半组：没有对应结果，整条会被丢掉
            USER,
            reply(text="第二轮"),
            {"kind": "checkpoint", "summary": "交接", "covers": 5},
        ],
    )
    restored = load_session(log)

    # 原本覆盖前 4 条（下标 0..3），丢掉半组后只剩 3 条落在范围内
    assert restored.checkpoint == ("交接", 3)
    assert_groups_valid(restored.messages)


def test_参数非法不影响重建(tmp_path):
    log = write_log(tmp_path, [START, USER, reply(("A", "read_file", "不是 json")), result("A")])
    r = load_session(log)
    assert r.seen_files == set()
    assert_groups_valid(r.messages)


def test_恢复工具状态供历史展示使用(tmp_path):
    log = write_log(
        tmp_path,
        [START, USER, reply(("A", "shell", "{}")), {**result("A", "shell"), "status": "warn"}],
    )
    assert load_session(log).tool_statuses == {"A": "warn"}


def test_request只记投影的形状不存全文(tmp_path):
    """全文能从 user / reply / tool_result 重建；每步再抄一遍的话，
    会话文件会随步数近似平方增长。"""
    log = SessionLog(tmp_path, "test")
    messages = [
        {"role": "system", "content": "系统提示"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "A", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "A", "name": "read_file", "content": "x" * 5000},
    ]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    log.request(1, messages, tools, SimpleNamespace(tokens=1234))

    request = [e for e in iter_events(log.path) if e.get("kind") == "request"][0]
    assert request["tools"] == ["read_file"]
    assert request["tool_schema"] == tools
    assert request["system_prompt"] == "系统提示"
    assert request["estimated_tokens"] == 1234
    assert [m["role"] for m in request["messages"]] == ["system", "assistant", "tool"]
    assert request["messages"][1]["calls"] == ["read_file"]
    assert request["messages"][2]["chars"] == 5000
    assert "x" * 100 not in log.path.read_text(encoding="utf-8")  # 全文没被抄进去
