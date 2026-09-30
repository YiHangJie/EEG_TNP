# 七类实验结果 Excel（2026-09-28）

交付文件：`EEG_完整实验结果_20260928.xlsx`。

## 内容与覆盖

- 七个实验页分别对应主实验、跨攻击跨范数、自适应攻击、张量网络效果效率、参数敏感性、消融、可视化。
- 主实验采用 EXP-032 正式修订版的统一 `S=512` 子集，覆盖 THUBenchmark、SEED-IV、bciciv2a，六模型，12 个方法/配置，五种子。SEED-IV 已由用户确认。
- 跨攻击页包含 FGSM L∞ 0.03、PGD-200 L∞ 0.03、PGD-200 L2 1.0、AutoAttack L∞ 0.03、无半径约束 CW L2。
- 自适应攻击目前只有四组完整五种子结果：TRP+CAF rank25/30 的 BPDA，以及 MagNet/DCAE 的真实梯度 PGD-10。其余八组明确留空并标记 Pending。EA-forward 是仓库历史 ABAT 路径，展示名保留 `ABAT-style` 和原始标签，未宣称严格原论文复现。
- 结构比较只有 EEGNet 的完整结果，另外五模型未测。除用户指定的 TT、Tucker、SVD，还收录 PTR、普通 TR、时间张量化 TT。预算档 25/30 表示相对于 PTR 的参数预算，不是共同的分解 rank。
- 参数敏感性有 55 组，消融有 9 组，均为五种子；包含原始分类器参考和两个测试净化 rank。
- 可视化页内嵌 20 张完整原图，覆盖五种子的成功、失败、clean 损伤、方法分歧；原图路径和病例索引同时保留。
- 其他页包括覆盖说明、4,550 条逐 seed 指标、独立的完整测试集主实验附录、缺失项、协议和来源索引。

## 统计与比较边界

seed 固定为 42–46、fold0。表中报告五种子的算术均值和样本标准差（ddof=1），不对攻击、模型或范数求混合均值。主实验统一使用 S512；历史完整测试集 F 单独保存在附录，不能与 S512 相减。

SA 是干净样本准确率；对于净化方法，指净化后的干净样本。RA 是相应攻击下的准确率。准确率摘要用 `均值 ± SD (%)`，旁边保留可计算的数值列；逐 seed 明细中的准确率及比例采用 0–100 数值并明确标注 `%`。

非自适应净化结果来自先攻击分类器再净化，不等同于针对完整防御的自适应攻击。历史 TNP BPDA 与可微 AP 的精确梯度攻击使用不同梯度方式，分别标记。MagNet 仅使用 Reformer 的 EEG adaptation，不包含 detector/rejection。

效果指标每 seed 512 样本；计时每 seed 8 个 clean 加 8 个 adversarial 样本，预热不计入，独占 GPU。时间先在 seed 内汇总，再在五种子间计算均值/样本标准差。峰值显存是 PyTorch allocated memory 的峰值，不是整张 GPU 的显存占用。结构 MSE 从逐样本 JSON 先求 seed 内均值，再跨 seed 等权平均；缺少对应诊断的 PTR 项留空。

## 数据来源与验证

正式来源为：

1. EXP-031：`exp031_full_20260729_174215` 的全测试集汇总和历史 BPDA 五种子结果。
2. EXP-032：`exp032_full_20260917_2020/execution_revisions/adaptive_pgd10_v1` 的当前已验收执行范围；原始 450 个 clean-only+TNP 任务仍为 Deferred。
3. EXP-033：`exp033_full_20260923_v6`，395/395 完成，错误列表为空。

源 CSV、JSON、文档、图片的相对/绝对路径、修改时间、SHA256 在 Excel 的来源页和 `validation.json` 中保留。生成过程没有修改实验源产物。

生成器验证了有结果条件均包含且仅包含五个 seed；采用的 EXP-032 均值与标准差均与正式聚合表一致；Excel 可重新打开，13 个 sheet 和 20 张内嵌图片完整。其他机械复核结果见 `review_validation.json`。本次未重新训练、运行攻击或重新推理模型，原实验的科学解释边界保留。

## 重新生成

在仓库根目录执行：

```bash
/home/yihangjie/miniconda3/envs/torch/bin/python docs/experiment_results/20260928/export_results.py
```

该命令只读实验来源，在当前整理目录重新生成 Excel 和 `validation.json`。Excel 内图片可脱离原目录查看；原始来源超链接需要在原工作区访问。
