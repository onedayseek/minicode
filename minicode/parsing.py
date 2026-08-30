"""模型输出的解析。

这里只做两件事：
1. 把流式返回里逐块到达的 tool_calls 累积成完整调用；
2. 把模型给的 arguments 字符串解析成参数字典，并做最基本的 schema 校验。
"""

import json
import re
from dataclasses import dataclass, field

from .errors import ToolError

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
# 完整围栏和只剩开头的围栏（输出被截断时）分开处理
_FENCE_FULL = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，延迟到执行前才解析


@dataclass
class Reply:
    """模型一轮回复的解析结果。"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class ToolCallAccumulator:
    """流式 tool_calls 累积器。

    OpenAI 兼容协议下，一次 tool call 的 arguments 是逐 chunk 送来的：
    只有第一个 chunk 带 id 和 name，后续 chunk 只带 index 和 arguments 片段。

    用 dict 而不是 list 存，是因为部分兼容网关的 index 不从 0 开始
    （文本 content block 会先占掉 0），用数组下标存会留空洞。
    """

    def __init__(self) -> None:
        self._slots: dict[int, dict] = {}

    def feed(self, delta_tool_calls) -> None:
        for tc in delta_tool_calls or []:
            slot = self._slots.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            if tc.id:
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                # name 未必在第一个 chunk 就到齐，不能提前校验
                if fn.name:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["args"] += fn.arguments

    def finish(self) -> list[ToolCall]:
        calls = []
        for index in sorted(self._slots):
            slot = self._slots[index]
            if not slot["name"]:
                continue  # 空洞或坏块，直接丢弃，不让它污染历史
            calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=slot["args"] or "{}",
                )
            )
        return calls


def _close_brackets(s: str) -> str:
    """按嵌套顺序补上未闭合的括号，模型输出被截断时常见。

    不能只数 `{` 和 `[` 的差额再各补各的：
    - 顺序会错。`{"tags": ["x"` 先补 `}` 再补 `]` 得到 `["x"}]`，照样不合法。
    - 字符串字面量里的括号不算数。`{"content": "def f() {` 会被多算一个。

    所以扫一遍维护栈。只补结构括号，不补截断在半路的字符串：字符串内容
    无法可靠还原，擅自补引号可能让半份 write_file 内容被当成完整参数执行。
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_string:
        return s
    return s + "".join(reversed(stack))


def _repair_json(raw: str) -> str:
    """对模型常见的 JSON 小毛病做轻量修复。只处理有把握的几种。"""
    s = raw.strip()
    # 剥代码围栏。用正则而不是按行切：模型也会把整个对象压成一行
    # （```json {"path": "a.py"} ```），按 \n 切的话第一行就是全部内容，剥不掉。
    fenced = _FENCE_FULL.match(s)
    s = fenced.group(1).strip() if fenced else _FENCE_OPEN.sub("", s).strip()
    # 去掉 } 或 ] 前的多余逗号
    s = _TRAILING_COMMA.sub(r"\1", s)
    return _close_brackets(s)


def parse_arguments(call: ToolCall) -> dict:
    """把 arguments 字符串解析成 dict。失败时抛 ToolError，由 loop 回灌给模型。"""
    if not call.arguments.strip():
        return {}
    try:
        value = json.loads(call.arguments)
    except json.JSONDecodeError:
        try:
            value = json.loads(_repair_json(call.arguments))
        except json.JSONDecodeError as e:
            raise ToolError(
                f"参数不是合法 JSON：{e}。请重新调用 {call.name}，"
                f"只输出一个 JSON 对象，字符串里的换行要写成 \\n。"
            ) from e
    if not isinstance(value, dict):
        raise ToolError(f"参数必须是 JSON 对象，收到的是 {type(value).__name__}。")
    return value


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _matches(value, expected: str) -> bool:
    """JSON Schema 的类型判断。未知类型一律放行。

    bool 要在数值类型上单独挡掉：Python 里 bool 是 int 的子类，
    `isinstance(True, int)` 为真，于是 `limit=true` 会静默通过校验，
    再被当成 1 用 —— 读回一行内容，看上去还成功了。
    """
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    py_type = _JSON_TYPES.get(expected)
    return py_type is None or isinstance(value, py_type)


def validate(args: dict, schema: dict, tool_name: str) -> dict:
    """极简 schema 校验：必填字段 + 顶层类型。

    不引入 jsonschema/pydantic：实际要覆盖的只有『字段缺失』和『类型写错』
    这两种模型常犯的错，为此拉一个重度校验库不划算。
    """
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            raise ToolError(f"{tool_name} 缺少必填参数 `{name}`。")

    validated = {}
    for name, value in args.items():
        if name not in props:
            continue  # 多给的参数从实际调用中移除，不值得为此打断模型
        expected = props[name].get("type")
        if expected and not _matches(value, expected):
            raise ToolError(
                f"{tool_name} 的参数 `{name}` 应为 {expected}，"
                f"收到 {type(value).__name__}。"
            )
        validated[name] = value
    return validated
