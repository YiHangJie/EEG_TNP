# PROMPTS.md

本文件保存常用科研与维护流程中可以复用的 Codex 提示词。

## 更新 CODEMAP.md

```text
请检查当前仓库结构，并更新 docs/CODEMAP.md。

要求：
- 阅读主要入口、数据读取代码、模型定义、训练流程、评估流程、配置和输出目录。
- 不要修改代码。
- 不要臆测文件职责。不确定的内容标记为 TODO。
- 标出后续研究 idea 可以安全扩展的位置。
- 标出不应修改的文件或目录，例如 raw data、checkpoints、logs 和 previous results。
- 最后总结修改了哪些文档文件，以及做了哪些验证。
```

## 根据 IDEAS.md 实现某个 Idea

```text
请实现 docs/IDEAS.md 中的 IDEA-XXX。

要求：
- 先阅读 docs/CODEMAP.md 和 IDEA-XXX 条目。
- 实现应小范围、可审查、可回滚。
- 默认不要改变 baseline 行为。
- 优先使用新模块、新配置、实验分支或显式配置开关。
- 不要修改 raw data、checkpoints、logs 或 previous results。
- 添加或更新最小相关验证。
- 在 docs/EXPERIMENTS.md 中补充 Pending 或实际实验记录。
- 不要编造实验结果。
```

## 添加冒烟测试

```text
请为最近修改的研究路径添加一个最小 smoke test。

要求：
- 测试应快速、聚焦。
- 如果完整训练成本较高，优先检查语法、导入、参数解析或 dry-run 行为。
- 不要启动长时间训练任务。
- 说明如何运行该测试。
- 只有在计划或实际运行了实验时，才更新 docs/EXPERIMENTS.md。
```

## 调试报错

```text
请调试下面的报错：

<在这里粘贴 traceback/log/command>

要求：
- 找出失败命令、相关文件和可能根因。
- 做最小安全修复。
- 不要修改无关代码。
- 不要删除 logs、checkpoints、data 或 previous results。
- 运行能够复现或检查修复的最小验证命令。
- 如果没有重跑完整任务，请说明剩余风险。
```

## 添加消融配置

```text
请为 IDEA-XXX 添加 ablation configs。

要求：
- 先阅读 docs/CODEMAP.md 和相关 IDEA-XXX 条目。
- 新增配置文件或显式配置开关；不要覆盖 baseline configs。
- 命名应保持一致且便于复现。
- 适用时写清随机种子、输出路径和关键设置。
- 除非明确要求，不要运行长时间实验。
- 在 docs/EXPERIMENTS.md 中添加计划实验，并将 Results 写为 Pending。
```

## 整理实验结果

```text
请把下面的实验结果整理到 docs/EXPERIMENTS.md：

<在这里粘贴 commands、metrics、logs 或 notes>

要求：
- 保留实际报告的指标和命令。
- 缺失值标记为 Pending。
- 不要推断或编造结果。
- 尽可能关联到相关 IDEA 编号。
- 只有在提供的数据支持时，才添加观察、问题、结论和下一步。
```

## 只做代码审查

```text
请只审查当前改动，不要修改文件。

审查重点：
- bug 或行为回归。
- 复现风险。
- baseline 行为变化。
- 缺失测试或验证缺口。
- 对 data、checkpoints、logs、previous results 或 configs 的不安全修改。

请先列出发现的问题，并尽可能提供文件和行号。
```
