# 七类实验结果 Excel（2026-09-28，已合入 EXP-034）

交付文件：`EEG_完整实验结果_20260928.xlsx`。本次按用户要求更新原文件，并实现之前约定的参数敏感性图表。

## 内容与覆盖

| 实验页 | 当前内容 | 覆盖情况 |
| --- | --- | --- |
| 1 主实验 | THUBenchmark、SEED-IV、bciciv2a × 六模型 × 14 方法/配置，PGD-200 L∞ 0.03 | 252 组，均为五 seed |
| 2 跨攻击跨范数 | THUBenchmark / EEGNet × 14 方法 × 5 攻击 | 70 组，均为五 seed |
| 3 Adaptive attack | 五个原始分类器、四个 AP 组合、两个 TRP+CAF BPDA 配置 | 11/14 组完成，3 组 Pending |
| 4 张量网络效果效率 | EEGNet 的 PTR、普通 TR、时间 TR、普通 TT、时间 TT、Tucker、SVD，各两预算 | 14 组效果及对应计时；其他五模型原六种分解仍 Pending |
| 5 参数敏感性 | rank 与五项 CE/KL 权重 | 55 组，六张 SA/RA 双面板图 |
| 6 消融实验 | clean-only、Madry、CAF，以及各自搭配 TRP rank25/30 | 9 组，均为五 seed |
| 7 可视化 | TNP/TRP 与 MagNet/DCAE 的同条件对照 | 5 组五 seed 汇总，保留原有 20 张案例图 |

全表共 14 个 sheet、26 张内嵌图、5,220 条逐 seed 指标。其他页为覆盖说明、参数图表源数据、逐 seed 明细、独立完整测试集附录、缺失与边界、协议与指标、来源索引。SEED 按用户确认采用 SEED-IV。

本次新增 MagNet-Reformer + Madry、DCAE + Madry，保留原 MagNet-Reformer + clean-only、DCAE + clean-only 为不同方法。主实验增加 36 组，跨攻击增加 10 组，自适应增加 7 组，时间张量化 TR 增加 2 个预算档的效果和效率结果。EXP-034 的 275 条独立 seed 结果全部纳入；其中 10 条 PGD 结果同时服务于主实验和跨攻击页，因此按工作表用途计数为 285 条新增记录，未当作独立重复实验。

自适应仍缺 CAF 原始分类器、TRP+Madry rank25、TRP+Madry rank30 三组。其余五模型原六种分解的两预算档仍缺 60 组。EXP-032 的 clean-only+TNP 全矩阵 Deferred 范围另列一条边界说明。时间 TR 按用户要求仅测试 THUBenchmark / EEGNet，没有把其他模型增列为缺失任务。

## 参数敏感性图表

六张图嵌在“5_参数敏感性”工作表第 65 行开始的位置，分别为：

- rank：15、20、25、30、35、40；TRP+CAF 为实线，CAF raw 为水平参考均值及 SD 阴影。
- `clean_ce_weight`：默认 1；实际系数 0、0.5、1、2。
- `pur_ce_weight`：默认 0.5；实际系数 0、0.25、0.5、1。
- `adv_pur_ce_weight`：默认 1；实际系数 0、0.5、1、2。
- `lambda_pur`：默认 0.2；实际系数 0、0.1、0.2、0.4。
- `lambda_adv_pur`：默认 0.5；实际系数 0、0.25、0.5、1。

每图分别呈现 SA 和 RA，误差条为五 seed 的样本标准差，散点保留每个 seed 的取值。权重图同时展示 CAF raw、TRP+CAF rank25、rank30，横轴为相对默认倍率 0 / 0.5 / 1 / 2，并附实际系数。1× 点引用各方法/净化 rank 已有的共同默认实验；不是新增实验，也不把这些重复引用计入样本数。纵轴按数据范围缩放，图注已说明。

“参数图表数据”保存 134 个绘图点的均值、SD、五个 seed 值、来源 ID 与原 CSV 行号。独立 PNG、PDF 和源数据 CSV 位于 `sensitivity_figures/`，便于另行用于报告；Excel 内图自包含。

## 统计与比较边界

seed 固定为 42–46、fold0。各条件使用算术均值和样本标准差（ddof=1），不对攻击、模型或范数求混合均值。主表统一采用 S512；历史完整测试集 F 单独保存在附录，不能与 S512 相减。

SA 是干净样本准确率；对于净化方法，指净化后的干净样本。RA 是相应攻击下的准确率。准确率摘要用 `均值 ± SD (%)`；主实验等表旁边保留数值列。结构表的 SA/RA 摘要按百分数显示两位小数，精确逐 seed 数值见明细。逐 seed 明细与图表源数据中的准确率及比例采用 0–100 数值，明确标注 `%`。

非自适应净化结果来自先攻击分类器再净化，不等同于针对完整防御的自适应攻击。PGD-10 的共同更新设置为 L∞ ε=0.03、α=0.006、10 步、clean 起点、无额外重启/EOT、攻击 batch1、返回最后一步；历史 TNP 使用 BPDA，可微 AP 组合使用真实梯度，原始分类器是分类器白盒真实梯度。EXP-034 原始分类器 SA 推理 batch32，AP+Madry SA 推理 batch8；攻击/adv 评估为 batch1，不把攻击批次误写成全部 clean 推理批次。

EA-forward 是仓库历史 ABAT 路径，展示名保留 `ABAT-style`，未宣称严格原论文复现。新增 PGD-10 在 EA 前原始缓存输入坐标施加 ε，并经原 subject-aware forward；其他方法在标准化 EEG 输入施加 ε，比较需区分输入坐标。EA 的批次依赖和历史 clean 漂移说明保留；任务 JSON 的坐标与状态审计可由来源索引追溯。MagNet 仅为 Reformer 的 EEG adaptation，不包含 detector/rejection；AP+Madry 复用既有固定净化器权重，未重新训练净化器。

结构预算 25/30 是相对 PTR 的参数预算，不是所有结构的共同 rank。时间 TR 为标准闭环 TR，使用 validation clean 选秩，实际参数量为 14,733 / 20,325，均在目标 ±5% 内。效果每 seed 512 样本；计时每 seed 8 clean + 8 adversarial 样本，预热不计入，独占 GPU。时间先在 seed 内汇总，再跨五 seed 计算均值/样本 SD。显存为 PyTorch allocated memory 峰值，不是整张 GPU 显存占用。结构 MSE 先求 512 样本均值，再跨 seed 等权平均；缺少对应诊断的 PTR 项留空。

## 数据来源与验证

正式来源：

1. EXP-031：`exp031_full_20260729_174215`，完整测试集汇总及历史 BPDA。
2. EXP-032：`exp032_full_20260917_2020/execution_revisions/adaptive_pgd10_v1`，已验收执行范围。
3. EXP-033：`exp033_full_20260923_v6`，395/395 任务完成。
4. EXP-034：`exp034_full_20260928_v1`，140/140 任务、275/275 指标，正式 report 为 Complete，errors/pending 为空。

所有 430 个已纳入实验条件均有且仅有 seeds42–46。EXP-032 与 EXP-034 的采用统计均从逐 seed 数据重算，并核对正式聚合；EXP-034 涉及 134 对均值/样本 SD（涵盖准确率、时间、显存、参数、压缩率、预算偏差、MSE）。生成后重新打开 Excel，校验工作表、图片、条数与缺失项；独立复核见 `review_validation.json`。来源路径、SHA256、修改时间和验证详情在 Excel 来源页及 `validation.json`。

更新前原 Excel 的 SHA256 与旧独立验证报告一致，未发现交付后的字节变化。原 Excel、生成器、README 和两个验证报告已完整备份到 `backups/before_exp034_20260928_154208/`。本次未修改实验原始结果、日志或 checkpoint，未训练或重新运行攻击。

## 重新生成

在仓库根目录执行：

```bash
/home/yihangjie/miniconda3/envs/torch/bin/python docs/experiment_results/20260928/export_results.py
```

生成器读取正式实验来源，重建敏感性 PNG/PDF/CSV、Excel 和 `validation.json`。工作簿先写临时文件、校验后替换目标；如手工修改了 Excel，应先保存副本，重生成不会合并手工编辑。内嵌图可脱离目录查看；来源超链接需原工作区可访问。
