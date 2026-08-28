"""会话事件落盘。

每次运行写一个 JSONL 文件，一行一个事件。用途是事后复查模型当时实际
看到和产生了什么 —— 终端上的显示是截断过的，不足以定位问题。

TODO(v2): 基于同一份文件实现 --resume 与压缩前的历史归档。
"""

import json
import os
import time
from pathlib import Path

SESSIONS_DIR = ".minicode/sessions"


class SessionLog:
    def __init__(self, root: Path, model: str) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / SESSIONS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{stamp}-{os.getpid()}.jsonl"
        self._started = time.monotonic()
        self.event("session_start", model=model, root=str(root))

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

    # 下面是几个固定形状的事件，集中在这里方便日后改 schema

    def user(self, text: str) -> None:
        self.event("user", text=text)

    def request(self, step: int, messages: list[dict], tools: list[str]) -> None:
        self.event("request", step=step, n_messages=len(messages), tools=tools)

    def reply(self, step: int, text: str, tool_calls, usage) -> None:
        self.event(
            "reply",
            step=step,
            text=text,
            tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
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
