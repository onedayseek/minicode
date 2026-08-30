# minicode

一个跑在终端里的小型编程 agent。给它一个编程任务，它会自己读代码、改文件、跑测试，直到做完。

从零实现，不依赖任何 agent 框架或 SDK —— 对话历史管理、工具定义与执行、模型输出解析、循环控制、错误处理都在 `minicode/` 里。第三方依赖只有三个：`openai`（当 HTTP 客户端用）、`rich`（终端渲染）、`prompt_toolkit`（终端输入，负责多行粘贴与输入历史）。

## 运行

```bash
pip install -e .
cp .env.example .env      # 填入你的 API key
minicode
```

默认接 DeepSeek，改 `.env` 里的 `MINICODE_BASE_URL` / `MINICODE_MODEL` 即可换到任何 OpenAI 兼容端点。

命令解释器在启动时自动解析（Windows 上 `pwsh` → `powershell` → `cmd`，其它平台 `bash` → `sh`），
解析结果会写进 `shell` 工具的描述交给模型。要指定别的解释器就设 `MINICODE_SHELL`，
例如在 Windows 上用 Git Bash：`MINICODE_SHELL=bash`。

```bash
minicode                          # 交互式
minicode -C ./some-project        # 指定工作目录
minicode -p "给 utils.py 补测试并跑通"   # 单次任务
minicode --resume                 # 恢复最近一次会话
minicode --resume 20260829-002954-3420.jsonl   # 恢复指定会话
```

没装的话也能直接跑：`pip install openai rich prompt_toolkit` 之后 `python -m minicode`。

每次运行的会话记录写在 `工作目录/.minicode/sessions/*.jsonl`（一行一个事件，模型实际看到和产生的内容）。`--resume` 从这份记录重建对话上下文继续工作，Ctrl-C 中断或换电脑都能接着上次的进度；恢复后写新会话文件，不覆盖原记录。

交互模式下可用 `/help` `/clear` `/status` `/log` `/exit`；输入 `/` 会显示命令补全，
灰色的历史建议可按 Tab 接受，Alt+Enter 或 Ctrl-J 可在同一条消息中手动换行。
写操作审批使用方向键选择、Enter 确认，Esc 可直接拒绝。

## 测试

```bash
pip install -e ".[dev]"
pytest tests/
```

## 结构

```
minicode/
├── cli.py        REPL、斜杠命令、参数
├── loop.py       主循环、终止条件、工具分发
├── context.py    对话历史、token 记账、预算控制
├── llm.py        provider 配置、流式请求、退避重试
├── parsing.py    流式 tool_calls 累积、JSON 修复、schema 校验
├── errors.py     错误分级
├── ui.py         渲染与审批交互
└── tools/        read_file / write_file / edit_file / list_files / grep / shell
```

设计取舍记录在 `docs/design.md`。

## 安全

- 工具只能操作启动时指定的工作目录，路径越界会被拒绝。
- 写文件和执行命令默认需要用户逐次确认（`a` 可对某个工具本会话免确认）。
- 少数明显破坏性的命令（`rm -rf /`、`del /s /q C:\`、fork 炸弹、`curl | sh` 等）在审批之外额外拦截，拦截规则按解析到的解释器选择。这是减速带，不是安全边界。
- 没有做 OS 级沙箱或容器隔离 —— 信任边界是「用户在场审批」，这是一个明确的取舍。
