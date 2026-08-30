"""LLM 客户端层：provider 配置、流式请求、错误分级、退避重试。

openai 包在这里只当 HTTP 客户端用（构造请求、收 SSE）；
tool_calls 的累积、终止判断、历史管理都在 parsing / loop / context 里。
"""

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .errors import ContextLimitError, FatalError, ProtocolError, RetryableError
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
    """一次请求的用量。

    DeepSeek 把输入拆成命中缓存和未命中两部分单独计价，且
    prompt_tokens = cache_hit + cache_miss。只看 prompt_tokens 会以为
    用量在跨轮次下降，实际是命中率在变。不提供这两个字段的 provider 记 0。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not api_key or api_key.startswith("sk-your-key"):
            raise FatalError(
                "未配置 API key。请复制 .env.example 为 .env 并填入 MINICODE_API_KEY。"
            )
        self.model = model
        # 重试策略由这一层统一控制，避免 SDK 默认重试再叠加外层四轮重试。
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=120.0, max_retries=0
        )
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
                delay = (
                    e.retry_after
                    if e.retry_after is not None
                    else delays[attempt] * random.uniform(0.8, 1.2)
                )
                time.sleep(max(0.0, delay))
        raise RetryableError("重试已用尽")  # 不会走到，兜底

    def _chat_once(self, messages: list[dict], tools: list[dict], on_text) -> Reply:
        # provider 没返回 usage 时不能沿用上一轮的旧值。
        self.last_usage = Usage()
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
                    u = chunk.usage
                    self.last_usage = Usage(
                        u.prompt_tokens,
                        u.completion_tokens,
                        getattr(u, "prompt_cache_hit_tokens", 0) or 0,
                        getattr(u, "prompt_cache_miss_tokens", 0) or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    reply.finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    reply.text += delta.content
                    if on_text:
                        on_text(delta.content)
                acc.feed(delta.tool_calls)
            reply.tool_calls = acc.finish()
            return reply

        except RateLimitError as e:
            raise RetryableError(str(e), _retry_after_seconds(e)) from e
        except (APIConnectionError, APITimeoutError) as e:
            raise RetryableError(str(e)) from e
        except APIStatusError as e:
            raise _classify_status_error(e) from e


def _classify_status_error(error) -> Exception:
    status = error.status_code
    if status in (401, 403):
        return FatalError(f"认证失败（HTTP {status}），请检查 API key。")
    if status == 402:
        return FatalError("账户余额不足。")
    if status >= 500 or status in (408, 409, 425, 429):
        return RetryableError(f"HTTP {status}", _retry_after_seconds(error))

    detail = f"{error} {getattr(error, 'body', '')}"
    lowered = detail.lower()
    if status == 400 and any(
        marker in lowered
        for marker in (
            "context length", "context window", "maximum context",
            "max context", "too many tokens",
        )
    ):
        return ContextLimitError(
            "请求超过模型上下文窗口。当前历史需要压缩，或用 /clear 开新会话。"
        )
    if status == 400 and any(
        marker in lowered
        for marker in (
            "tool_call", "tool call", "role 'tool'", 'role "tool"',
            "tool message",
        )
    ):
        return ProtocolError(
            "工具调用消息没有正确配对，会话协议已损坏。请保留日志并用 /clear 重试。"
        )
    return FatalError(f"请求被拒绝（HTTP {status}）：{error}")


def _retry_after_seconds(error) -> float | None:
    """解析 Retry-After，兼容秒数和 HTTP-date 两种标准格式。"""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
