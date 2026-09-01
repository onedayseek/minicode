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
    """从会话记录重建出的可继续状态。

    历史与投影状态分开返回：messages 是完整原文，elided / checkpoint 描述
    「当时是怎么看它的」。恢复出来的上下文因此和中断前一致，而原始内容一条不少。
    """

    root: str
    messages: list[dict] = field(default_factory=list)
    seen_files: set[str] = field(default_factory=set)
    tool_statuses: dict[str, str] = field(default_factory=dict)
    elided: dict[str, str] = field(default_factory=dict)
    checkpoint: tuple[str, int] | None = None  # (摘要, 覆盖到第几条)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0


class SessionLog:
    def __init__(self, root: Path, model: str, shell: str = "", window: int | None = None,
                 max_output: int | None = None) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / SESSIONS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{stamp}-{os.getpid()}.jsonl"
        self._started = time.monotonic()
        self._contract_recorded = False
        self.event(
            "session_start", model=model, root=str(root), shell=shell,
            **({"context_window": window, "max_output_tokens": max_output}
               if window is not None and max_output is not None else {}),
        )

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

    def command(self, text: str) -> None:
        """记录本地斜杠命令；命令不属于模型对话，也不计入 token。"""
        self.event("command", command=text)

    def request(self, step: int, messages: list[dict], tools: list[dict], budget=None) -> None:
        """记这一轮投影的形状，不存全文。

        存全文的话，每一步都把整个历史再抄一遍，文件随步数近似平方增长。
        而全文本来就能从 user / reply / tool_result / context_elision /
        checkpoint 这几类事件重建 —— 那几类才是事实源，这里只是「当时是
        怎么投影的」的凭证：多少条、什么角色、哪些被收敛成了短版本。
        """
        shape = [
            {
                "role": m.get("role"),
                "chars": len(m.get("content") or ""),
                **({"name": m["name"]} if m.get("name") else {}),
                **({"tool_call_id": m["tool_call_id"]} if m.get("tool_call_id") else {}),
                **(
                    {"calls": [c["function"]["name"] for c in m["tool_calls"]]}
                    if m.get("tool_calls")
                    else {}
                ),
            }
            for m in messages
        ]
        fields = {
            "step": step,
            "messages": shape,
            "tools": [t.get("function", {}).get("name") for t in tools],
            "estimated_tokens": getattr(budget, "tokens", None),
            "context_window": getattr(budget, "limit", None),
            "max_output_tokens": getattr(budget, "output_budget", None),
        }
        if not self._contract_recorded:
            fields["system_prompt"] = messages[0].get("content", "") if messages else ""
            fields["tool_schema"] = tools
            self._contract_recorded = True
        self.event("request", **fields)

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

    def tool_result(self, call, status: str, content: str, raw_content: str | None = None) -> None:
        self.event(
            "tool_result",
            id=call.id,
            name=call.name,
            status=status,
            content=content,
            **({"raw_content": raw_content} if raw_content is not None else {}),
        )

    def restore_snapshot(self, restored: "Restored") -> None:
        """在继承的原始事件之后记录规范化投影，避免恢复下标再次漂移。

        原始事件仍完整保留；快照只是下一次重放时使用的物化视图。
        """
        self.event(
            "restore_snapshot",
            root=restored.root,
            messages=restored.messages,
            seen_files=sorted(restored.seen_files),
            tool_statuses=restored.tool_statuses,
            elided=restored.elided,
            checkpoint=(
                {"summary": restored.checkpoint[0], "covers": restored.checkpoint[1]}
                if restored.checkpoint else None
            ),
        )

    def stop(self, reason: str) -> None:
        self.event("stop", reason=reason)

    def system_note(self, content: str) -> None:
        self.event("system_note", content=content)

    def context_elision(self, notice: str, changes: list[dict]) -> None:
        self.event("context_elision", notice=notice, changes=changes)

    def checkpoint(self, summary: str, covers: int) -> None:
        """一次交接式压缩。旧事件一条不删 —— 它只是宣布「从这里起换个看法」。"""
        self.event("checkpoint", summary=summary, covers=covers)

    def compaction_usage(self, usage) -> None:
        self.event(
            "compaction_usage",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_tokens=usage.cache_hit_tokens,
            cache_miss_tokens=usage.cache_miss_tokens,
        )

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
    read_paths: list[str] = []
    tool_statuses: dict[str, str] = {}
    touched_by_call: dict[str, str] = {}
    elided: dict[str, str] = {}
    checkpoint: tuple[str, int] | None = None
    prompt_tokens = 0
    completion_tokens = 0
    cache_hit_tokens = 0
    # 每组：[assistant 消息下标, 尚未闭合的 call id, 本组 tool 消息的下标]
    groups: list[list] = []
    drop: set[int] = set()

    def reset() -> None:
        nonlocal checkpoint
        messages.clear()
        read_paths.clear()
        tool_statuses.clear()
        touched_by_call.clear()
        elided.clear()
        groups.clear()
        drop.clear()
        checkpoint = None

    for ev in iter_events(path):
        kind = ev.get("kind")

        if kind == "session_start":
            root = ev.get("root", root)

        elif kind == "restore_snapshot":
            root = ev.get("root", root)
            messages[:] = ev.get("messages") or []
            read_paths[:] = []
            tool_statuses.clear()
            tool_statuses.update(ev.get("tool_statuses") or {})
            touched_by_call.clear()
            elided.clear()
            elided.update(ev.get("elided") or {})
            groups.clear()
            drop.clear()
            raw_checkpoint = ev.get("checkpoint")
            checkpoint = (
                (raw_checkpoint.get("summary", ""), raw_checkpoint.get("covers", 0))
                if isinstance(raw_checkpoint, dict) else None
            )

        elif kind == "user":
            messages.append({"role": "user", "content": ev.get("text", "")})

        elif kind == "reply":
            prompt_tokens += ev.get("prompt_tokens", 0) or 0
            completion_tokens += ev.get("completion_tokens", 0) or 0
            cache_hit_tokens += ev.get("cache_hit_tokens", 0) or 0
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

        elif kind == "compaction_usage":
            prompt_tokens += ev.get("prompt_tokens", 0) or 0
            completion_tokens += ev.get("completion_tokens", 0) or 0
            cache_hit_tokens += ev.get("cache_hit_tokens", 0) or 0

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
            # 只记「当时是怎么看它的」，不碰消息本身 —— 原文留在历史里
            for change in ev.get("changes") or []:
                call_id = change.get("tool_call_id")
                if isinstance(call_id, str):
                    elided[call_id] = change.get("content", "")

        elif kind == "checkpoint":
            # 用事件里显式记的覆盖范围，不能用「此刻已重建出多少条」——
            # 压缩总是发生在历史已经写了一大截之后，事件的位置反映的是
            # 「什么时候压的」，而不是「压到哪」，两者差得很远。
            # covers 是内存里 Context.messages 的下标（含开头的 system
            # prompt），这里重建的数组不含它，所以差一位。
            covers = ev.get("covers")
            if isinstance(covers, int):
                checkpoint = (ev.get("summary", ""), max(0, covers - 1))

    for assistant_index, unclosed, tool_indices in groups:
        if not unclosed:
            continue
        messages[assistant_index].pop("tool_calls", None)
        drop.update(tool_indices)
        # 只剩空文本的 assistant 消息没有信息量，一并去掉
        if not messages[assistant_index].get("content"):
            drop.add(assistant_index)

    base = Path(root)
    kept = [i for i in range(len(messages)) if i not in drop]
    if checkpoint is not None:
        # 丢弃残缺组会让下标整体前移，覆盖范围要跟着重映射，
        # 否则恢复出的投影会多切或少切几条。
        summary, covers = checkpoint
        checkpoint = (summary, sum(1 for i in kept if i < covers))

    return Restored(
        root=root,
        messages=[messages[i] for i in kept],
        seen_files={str((base / p).resolve()) for p in read_paths},
        tool_statuses=tool_statuses,
        elided=elided,
        checkpoint=checkpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit_tokens=cache_hit_tokens,
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
