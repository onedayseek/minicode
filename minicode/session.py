"""会话事件落盘与恢复。

每次运行写一个 JSONL 文件，一行一个事件。用途有二：事后复查模型当时实际
看到和产生了什么（终端上的显示是截断过的），以及 --resume 重建上下文。

TODO(v2): 上下文压缩前把完整历史归档到同一份文件。
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

SESSIONS_DIR = ".minicode/sessions"


@dataclass
class Restored:
    """从会话记录重建出的可继续状态。"""

    root: str
    messages: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    seen_files: set[str] = field(default_factory=set)
    tool_statuses: dict[str, str] = field(default_factory=dict)


class SessionLog:
    def __init__(self, root: Path, model: str, shell: str = "") -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / SESSIONS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{stamp}-{os.getpid()}.jsonl"
        self._started = time.monotonic()
        self.event("session_start", model=model, root=str(root), shell=shell)

    def event(self, kind: str, **fields) -> None:
        record = {
            "t": round(time.monotonic() - self._started, 3),
            "kind": kind,
            **fields,
        }
        # 单条事件写失败不应影响会话本身
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def inherit(self, source: Path) -> None:
        """把被恢复会话的事件抄进本次记录，使每个文件都自成完整历史。

        否则对一个恢复出来的会话再次 --resume 会丢掉更早的历史。
        """
        self.event("resumed_from", path=str(source))
        for ev in iter_events(source):
            if ev.get("kind") == "session_start":
                continue  # 本次已经写过自己的
            self.event(ev.get("kind", "unknown"), **{k: v for k, v in ev.items() if k not in ("t", "kind")})

    # 下面是几个固定形状的事件，集中在这里方便日后改 schema

    def user(self, text: str) -> None:
        self.event("user", text=text)

    def request(self, step: int, messages: list[dict], tools: list[dict]) -> None:
        # 保存真正发送给 provider 的快照。事件写入会立即 JSON 序列化，后续上下文
        # 原地裁剪不会反过来修改这份记录。
        self.event("request", step=step, messages=messages, tools=tools)

    def reply(self, step: int, text: str, tool_calls, usage, finish_reason=None) -> None:
        self.event(
            "reply",
            step=step,
            text=text,
            tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_tokens=usage.cache_hit_tokens,
            cache_miss_tokens=usage.cache_miss_tokens,
            finish_reason=finish_reason,
        )

    def tool_result(self, call, status: str, content: str) -> None:
        self.event(
            "tool_result",
            id=call.id,
            name=call.name,
            status=status,
            content=content,
        )

    def stop(self, reason: str) -> None:
        self.event("stop", reason=reason)

    def system_note(self, content: str) -> None:
        self.event("system_note", content=content)

    def context_elision(self, notice: str, changes: list[dict]) -> None:
        self.event("context_elision", notice=notice, changes=changes)

    def internal_error(self, call, traceback_text: str) -> None:
        self.event(
            "internal_error",
            id=call.id,
            name=call.name,
            traceback=traceback_text,
        )


def iter_events(path: Path):
    """逐行读取会话 JSONL，容错跳过坏行（写入是尽力而为的，不该反过来阻塞恢复）。"""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_session(path: Path) -> "Restored":
    """从会话记录重建可继续的上下文。

    消息数组不含开头的主 system prompt，但包含运行中注入的 system note，
    可直接 extend 到 Context.messages 后面。

    重建时以「消息组」为单位丢弃未闭合的 tool_calls —— API 要求 role=tool 的消息
    必须紧跟在带 tool_calls 的 assistant 之后。中断可能只留下一组里的部分结果，
    此时既要剥掉 assistant 的 tool_calls，也要丢掉这组里已到达的 tool 消息，
    否则它们会变成孤儿，让恢复后的第一次请求被拒。
    """
    root = "."
    messages: list[dict] = []
    prompt_tokens = 0
    read_paths: list[str] = []
    tool_statuses: dict[str, str] = {}
    touched_by_call: dict[str, str] = {}
    # 每组：[assistant 消息下标, 尚未闭合的 call id, 本组 tool 消息的下标]
    groups: list[list] = []
    drop: set[int] = set()

    def reset() -> None:
        nonlocal prompt_tokens
        messages.clear()
        read_paths.clear()
        tool_statuses.clear()
        touched_by_call.clear()
        groups.clear()
        drop.clear()
        prompt_tokens = 0

    for ev in iter_events(path):
        kind = ev.get("kind")

        if kind == "session_start":
            root = ev.get("root", root)

        elif kind == "user":
            messages.append({"role": "user", "content": ev.get("text", "")})

        elif kind == "reply":
            prompt_tokens = ev.get("prompt_tokens", prompt_tokens)
            msg: dict = {"role": "assistant", "content": ev.get("text", "")}
            calls = ev.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ]
                groups.append([len(messages), {c["id"] for c in calls}, []])
                touched_by_call.update(_touched_paths(calls))
            messages.append(msg)

        elif kind == "tool_result":
            call_id = ev.get("id")
            if isinstance(call_id, str):
                tool_statuses[call_id] = ev.get("status", "ok")
                if ev.get("status", "ok") != "fail" and call_id in touched_by_call:
                    read_paths.append(touched_by_call[call_id])
            index = len(messages)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ev.get("id"),
                    "name": ev.get("name"),
                    "content": ev.get("content", ""),
                }
            )
            for group in groups:
                if ev.get("id") in group[1]:
                    group[1].discard(ev.get("id"))
                    group[2].append(index)
                    break
            else:
                drop.add(index)  # 找不到归属的 tool 消息同样是孤儿

        elif kind == "clear":
            reset()

        elif kind == "system_note":
            messages.append({"role": "system", "content": ev.get("content", "")})

        elif kind == "context_elision":
            for change in ev.get("changes") or []:
                call_id = change.get("tool_call_id")
                for message in reversed(messages):
                    if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
                        message["content"] = change.get("content", message.get("content", ""))
                        break

    for assistant_index, unclosed, tool_indices in groups:
        if not unclosed:
            continue
        messages[assistant_index].pop("tool_calls", None)
        drop.update(tool_indices)
        # 只剩空文本的 assistant 消息没有信息量，一并去掉
        if not messages[assistant_index].get("content"):
            drop.add(assistant_index)

    base = Path(root)
    return Restored(
        root=root,
        messages=[m for i, m in enumerate(messages) if i not in drop],
        prompt_tokens=prompt_tokens,
        seen_files={str((base / p).resolve()) for p in read_paths},
        tool_statuses=tool_statuses,
    )


def _touched_paths(calls: list[dict]) -> dict[str, str]:
    """记录调用 id 与候选路径；只有成功结果到达后才恢复为已读。"""
    paths = {}
    for call in calls:
        if call.get("name") not in ("read_file", "write_file"):
            continue
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict) and isinstance(args.get("path"), str):
            call_id = call.get("id")
            if isinstance(call_id, str):
                paths[call_id] = args["path"]
    return paths
