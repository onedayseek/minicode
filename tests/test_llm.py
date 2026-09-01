"""LLM 边界：结束原因、重试归属和 HTTP 错误分类。"""

from types import SimpleNamespace

import minicode.llm as llm_module
from minicode.errors import ContextLimitError, ProtocolError, RetryableError
from minicode.llm import LLMClient, _classify_status_error, _retry_after_seconds
from minicode.parsing import Reply


def test_关闭SDK内部重试():
    client = LLMClient("https://example.com", "test", "sk-test")
    try:
        assert client._client.max_retries == 0
    finally:
        client._client.close()


def test_Retry_After秒数被解析():
    error = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "2.5"}))
    assert _retry_after_seconds(error) == 2.5


def test_外层重试尊重Retry_After(monkeypatch):
    client = LLMClient.__new__(LLMClient)
    attempts = []
    sleeps = []

    def once(_messages, _tools, _tap):
        attempts.append(1)
        if len(attempts) == 1:
            raise RetryableError("限流", retry_after=3.25)
        return Reply(text="ok")

    client._chat_once = once
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    assert client.chat([], []).text == "ok"
    assert sleeps == [3.25]


def _fake_client(chunks, max_output: int = 0):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return chunks

    client = LLMClient.__new__(LLMClient)
    client.model = "test"
    client.max_output = max_output
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, seen


def test_流式结束原因被保留():
    delta = SimpleNamespace(content=None, tool_calls=None)
    chunk = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=delta, finish_reason="length")],
    )
    client, _ = _fake_client([chunk])

    reply = client._chat_once([], [], lambda _text: None)

    assert reply.finish_reason == "length"


def test_用量跟着回复走而不是挂在客户端上():
    """provider 不返回 usage 时拿到的是零值，不会是上一次请求的残留。

    用量做成返回值而不是客户端上的字段，是因为同一个 agent 会发出两种请求
    （主循环的和压缩用的）——挂在客户端上就说不清「上一次」是哪一次了。
    """
    delta = SimpleNamespace(content=None, tool_calls=None)
    quiet = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason="stop")])
    counted = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=999, completion_tokens=7),
        choices=[],
    )

    client, _ = _fake_client([counted, quiet])
    assert client._chat_once([], [], None).usage.prompt_tokens == 999

    client, _ = _fake_client([quiet])
    assert client._chat_once([], [], None).usage.prompt_tokens == 0


def test_生成上限被传给provider():
    """客户端把 Agent 为本次请求算出的动态输出上限传给 provider。"""
    delta = SimpleNamespace(content=None, tool_calls=None)
    chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason="stop")])

    client, seen = _fake_client([chunk], max_output=16_384)
    client._chat_once([], [], None)
    assert seen["max_tokens"] == 16_384

    client, seen = _fake_client([chunk], max_output=0)
    client._chat_once([], [], None)
    assert "max_tokens" not in seen  # 0 表示交给 provider 的默认值


class FakeStatusError:
    def __init__(self, status_code: int, text: str, body=None) -> None:
        self.status_code = status_code
        self.text = text
        self.body = body
        self.response = SimpleNamespace(headers={})

    def __str__(self) -> str:
        return self.text


def test_400上下文与工具协议分别诊断():
    context = _classify_status_error(FakeStatusError(400, "maximum context length exceeded"))
    protocol = _classify_status_error(FakeStatusError(400, "role 'tool' has no tool_call"))

    assert isinstance(context, ContextLimitError)
    assert isinstance(protocol, ProtocolError)


def test_408与5xx交给框架重试():
    assert isinstance(_classify_status_error(FakeStatusError(408, "timeout")), RetryableError)
    assert isinstance(_classify_status_error(FakeStatusError(503, "down")), RetryableError)
