"""错误分级：按『谁能修』划分，决定 loop 如何反应。"""


class MinicodeError(Exception):
    pass


class ToolError(MinicodeError):
    """模型能自己修的错误。不抛到 loop 外，而是作为 tool result 回灌。

    message 应当是 actionable 的：告诉模型出了什么问题、以及下一步可以怎么做。
    """


class RetryableError(MinicodeError):
    """框架能自己修：限流、5xx、连接中断。指数退避后重试。"""


class FatalError(MinicodeError):
    """必须终止：认证失败、余额不足、配置缺失。给用户明确指引后退出。"""


class UserAbort(MinicodeError):
    """用户拒绝了一次操作，或按了 Ctrl-C。"""
