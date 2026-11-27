# Keyline

Keyline 是一个面向长程软件工程任务的本地 Coding Agent。它在命令行中操作本机仓库，保存会话和执行进度，并在收到所需的用户输入后续接任务。模型请求使用 DeepSeek 的 Anthropic Messages 兼容接口。

## 核心设计

- **等待与续接**。将用户补充的信息、敏感输入和操作审批绑定到对应的等待条件。调度器可以在一个 Run 等待时推进其他可运行的 Run，收到输入后续接原 Run。 [调度实现](src/keyline_agent/runtime/scheduler.py)
- **持久化执行状态**。用 SQLite 保存会话、工具结果和执行进度，支持进程重启后恢复。长会话可以压缩历史，大段工具输出可通过本地 Artifact 引用按需读取。 [状态存储](src/keyline_agent/storage/sqlite.py)
- **多 Agent 协作**。主 Agent 可以委派探索、审查和实现任务。实现任务使用独立的 Git worktree，主 Agent 可以检查改动和测试结果，再选择要合入的变更。 [任务委派](src/keyline_agent/runtime/delegation.py)
- **本地执行与输入边界**。文件、搜索、Shell 和测试工具面向当前工作区。Shell 命令默认逐次审批，敏感输入通过专用执行路径交给命令，模型通过句柄引用它。

## 在自己的仓库里运行

需要 Python 3.11 或更新版本、Git 和 uv。克隆本仓库后，在 Keyline 的仓库根目录运行以下命令。

```bash
uv sync --locked
uv run keyline-agent config set-api-key
uv run keyline-agent run /path/to/repo "复现并修复这个问题，并检查相关回归"
```

`config set-api-key` 通过隐藏输入保存 DeepSeek API Key。也可以通过环境变量 `DEEPSEEK_API_KEY` 提供。Keyline 默认将配置和会话存放在 `~/.keyline-agent`。运行开始时会打印 Session ID，可以用它查看或恢复会话。

```bash
uv run keyline-agent list
uv run keyline-agent inspect <session-id>
uv run keyline-agent resume <session-id>
```

可以在启动时用 `--max-steps` 设置模型轮次上限。达到上限后，可通过 `resume <session-id> --max-steps <total>` 调整累计上限并继续。使用 `uv run keyline-agent --help` 查看完整命令。

## 评测

- **端到端修复（Verified80）**。使用基线版本 `403cd47` 和 `deepseek-flash` 在 SWE-bench Verified 的固定 80 题子集上评测。以全部 80 题为分母，只将通过官方评分的题目计为解决。最终解决 **58/80 题（72.5%）**，其余 **22 题**计为失败。
- **工具调用（Tool Calling）**。评估 12 条运行轨迹中的 694 次工具调用，检查工具选择和参数是否符合任务意图。工具选择平均 **4.28/5**（覆盖 551/694 次），参数准确性平均 **4.65/5**（覆盖 550/694 次），均按有效评分计算。
- **上下文压缩（Compaction）**。6 个固定样本各运行 3 次。17 次可计算压缩比的运行中，被替换内容的压缩比中位数为 **1.55×**。各次运行累计有 79 项待评关键信息，其中 72 项获得有效评分，均被完整保留。14 次获得事实一致性评分，均为 **2/2**。

用工具轨迹、候选 Patch 和官方判分复盘失败案例，改进了搜索空结果、编辑冲突和命令超时的反馈，也调整了 Agent 的验证与收尾指令。 [Verified80 任务清单](evals/datasets/verified80/tasks.toml) · [评测运行方式](evals/README.md)

## 许可证

本项目使用 [MIT License](LICENSE)。
