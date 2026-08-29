"""LLM 客户端层：provider 配置、流式请求、错误分级、退避重试。

openai 包在这里只当 HTTP 客户端用（构造请求、收 SSE）；
tool_calls 的累积、终止判断、历史管理都在 parsing / loop / context 里。
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .errors import FatalError, RetryableError
from .parsing import Reply, ToolCallAccumulator


def load_dotenv(path: Path) -> None:
    """极简 .env 读取，避免为此多一个依赖。已存在的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not api_key or api_key.startswith("sk-your-key"):
            raise FatalError(
                "未配置 API key。请复制 .env.example 为 .env 并填入 MINICODE_API_KEY。"
            )
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self.last_usage = Usage()

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(
            base_url=os.environ.get("MINICODE_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("MINICODE_MODEL", "deepseek-chat"),
            api_key=os.environ.get("MINICODE_API_KEY", ""),
        )

    def chat(self, messages: list[dict], tools: list[dict], on_text=None, on_retry=None) -> Reply:
        """一次带重试的流式对话。

        on_text 边收边打印。on_retry(原因, 本次是否已经吐出过文本) 在每次重试前调用：
        流式已经打到终端上的内容擦不掉，重试会把同一段重新生成一遍，
        不说明的话用户只会看到屏幕上莫名其妙出现了两份半截回答。
        """
        delays = [1.0, 2.0, 4.0]
        for attempt in range(len(delays) + 1):
            streamed = False

            def tap(chunk: str) -> None:
                nonlocal streamed
                streamed = True
                if on_text:
                    on_text(chunk)

            try:
                return self._chat_once(messages, tools, tap)
            except RetryableError as e:
                if attempt == len(delays):
                    raise
                if on_retry:
                    on_retry(str(e), streamed)
                time.sleep(delays[attempt])
        raise RetryableError("重试已用尽")  # 不会走到，兜底

    def _chat_once(self, messages: list[dict], tools: list[dict], on_text) -> Reply:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                stream=True,
                stream_options={"include_usage": True},
            )
            reply = Reply()
            acc = ToolCallAccumulator()
            for chunk in stream:
                if chunk.usage:
                    self.last_usage = Usage(
                        chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                    )
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    reply.text += delta.content
                    if on_text:
                        on_text(delta.content)
                acc.feed(delta.tool_calls)
            reply.tool_calls = acc.finish()
            return reply

        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            raise RetryableError(str(e)) from e
        except APIStatusError as e:
            if e.status_code in (401, 403):
                raise FatalError(f"认证失败（HTTP {e.status_code}），请检查 API key。") from e
            if e.status_code == 402:
                raise FatalError("账户余额不足。") from e
            if e.status_code >= 500 or e.status_code == 429:
                raise RetryableError(f"HTTP {e.status_code}") from e
            raise FatalError(f"请求被拒绝（HTTP {e.status_code}）：{e}") from e
