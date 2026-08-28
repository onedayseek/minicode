# minicode

一个跑在终端里的小型编程 agent。给它一个编程任务，它会自己读代码、改文件、跑测试，直到做完。

从零实现，不依赖任何 agent 框架或 SDK —— 对话历史管理、工具定义与执行、模型输出解析、循环控制、错误处理都在 `minicode/` 里。第三方依赖只有 `openai`（当 HTTP 客户端用）和 `rich`（终端渲染）。

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入你的 API key
python -m minicode
```

默认接 DeepSeek，改 `.env` 里的 `MINICODE_BASE_URL` / `MINICODE_MODEL` 即可换到任何 OpenAI 兼容端点。

```bash
python -m minicode                          # 交互式
python -m minicode -C ./some-project        # 指定工作目录
python -m minicode -p "给 utils.py 补测试并跑通"   # 单次任务
```

交互模式下可用 `/help` `/clear` `/status` `/exit`。

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
└── tools/        read_file / write_file / edit_file / list_files / grep / bash
```

设计取舍记录在 `docs/design.md`。

## 安全

- 工具只能操作启动时指定的工作目录，路径越界会被拒绝。
- 写文件和执行命令默认需要用户逐次确认（`a` 可对某个工具本会话免确认）。
- 少数明显破坏性的命令（`rm -rf /`、fork 炸弹、`curl | sh` 等）在审批之外额外拦截。
- 没有做 OS 级沙箱或容器隔离 —— 信任边界是「用户在场审批」，这是一个明确的取舍。
