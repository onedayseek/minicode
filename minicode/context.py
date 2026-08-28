"""对话历史与上下文管理。

第一版做两件事：
1. 以 OpenAI 消息数组为唯一事实源，保证 assistant.tool_calls 与其 tool 结果成组；
2. 用 API 回传的 usage 做 token 记账，超过软阈值时省略旧的大体积工具输出。

TODO(v2): 硬阈值下的 LLM 摘要式压缩。
"""

from .parsing import ToolCall

# DeepSeek-chat 的上下文窗口。换 provider 时由 CLI 覆盖。
DEFAULT_WINDOW = 64_000
SOFT_RATIO = 0.7  # 超过就开始省略旧工具输出
ELIDE_KEEP = 200  # 省略后为旧工具结果保留的字符数


class Context:
    def __init__(self, system_prompt: str, window: int = DEFAULT_WINDOW) -> None:
        self.window = window
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.prompt_tokens = 0  # 上一轮 API 实测值，比本地估算准

    # ---- 写入 ----

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str, tool_calls: list[ToolCall]) -> None:
        msg: dict = {"role": "assistant", "content": text or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in tool_calls
            ]
        self.messages.append(msg)

    def add_tool_result(self, call: ToolCall, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
        )

    def add_system_note(self, text: str) -> None:
        """框架向模型注入的提示，例如『你似乎在原地打转』。"""
        self.messages.append({"role": "system", "content": text})

    # ---- 读取 ----

    def render(self) -> list[dict]:
        return self.messages

    def usage_ratio(self) -> float:
        return self.prompt_tokens / self.window if self.window else 0.0

    def reset(self) -> None:
        del self.messages[1:]
        self.prompt_tokens = 0

    # ---- 预算 ----

    def ensure_budget(self) -> str | None:
        """超过软阈值时省略较早的工具输出。返回一句给用户看的说明，或 None。

        只改 tool 消息的 content，不删任何消息 —— 这样 assistant.tool_calls
        和它的 tool 结果永远成组存在，不会触发 API 的配对校验错误。
        """
        if self.usage_ratio() < SOFT_RATIO:
            return None

        # 最近 6 条消息原样保留，模型正在依赖它们
        elided = 0
        for msg in self.messages[:-6]:
            if msg["role"] != "tool":
                continue
            content = msg["content"]
            if len(content) <= ELIDE_KEEP or content.endswith("[已省略]"):
                continue
            msg["content"] = (
                content[:ELIDE_KEEP]
                + f"\n... 此处省略 {len(content) - ELIDE_KEEP} 字符，"
                f"需要时请重新调用 {msg['name']} 获取 [已省略]"
            )
            elided += 1
        if elided:
            return f"上下文已用 {self.usage_ratio():.0%}，省略了 {elided} 条较早的工具输出"
        return None
