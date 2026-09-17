# EXP-031 专题结果报告：主实验、跨攻击均值、组件消融与秩敏感性

源快照：2026-09-11T14:24:12.010918；沿用已核验结果，不纳入此后完成任务。

本次任务完成 2832/3073；相比上一版2731项完成新增 101 项。上一版快照保留在相邻的 EXP031_results_20260910_171036 目录。

## 统计口径与阅读限制

- 本次源产物检查错误 0 项，警告 65 项；检查协议、状态、训练记录、索引、标签和形状，未比较输入信号内容，也未重新执行攻击或推理。前一快照的信号抽查不自动覆盖新增产物。
- 准确率为五种子（42–46）、fold0 的均值 ± 样本标准差，单位 %；差值单位为百分点（pp）。未满五种子仅显示 Pending(n/5)。
- [F] 表示完整测试集，[S] 表示确定性最多 512 样本子集。主实验现有普通 AT 为 [F]，RPCF+TNP 为 [S]；该表是现有记录并列展示，**不是严格同样本范围的排名，不能直接将列间差值解释为方法增益**。普通 AT 的统一子集结果尚未整理，正式主表需要补齐同口径评估。
- 普通 AT 指不带 TNP 的 Madry、TRADES、FBF、EA-forward；普通 clean-only 模型未运行。RPCF 未净化及 Madry+TNP 留在组件对照中。
- 常规攻击针对各自分类器生成，非净化链路的自适应攻击；跨模型对照不是固定同一对抗输入的因果实验。
- PGD 为 Linf ε=0.03、200步、步长2/255、无随机起点；AA/FGSM 同为 Linf ε=0.03。CW 为 L2、200步、c=10000、kappa=1、lr=0.1，不受该 Linf ε 约束。
- Mean4：每个种子先计算 (AA+FGSM+PGD+CW)/4，再计算五种子均值与标准差；MeanLinf3 同理，仅包含 AA/FGSM/PGD。标准差不是各攻击标准差的平均。
- Mean4 是异质攻击集合的描述性均值，不是等强度攻击指标，也不是逐样本最坏攻击准确率。任何攻击缺失，该种子不进入该均值；不按已完成攻击数动态改变分母。
- 不跨数据集或 backbone 混合求一个总均值；rank25/30 同时报告，未根据测试表现挑选最优秩。

## 1. 主实验：RPCF+TNP 与普通 AT（PGD-200）

所有 18 个数据集—backbone 条件均列出。请保留表头 [F]/[S] 标签及上述样本范围限制。

| 数据集 | Backbone | Madry [F] | TRADES [F] | FBF [F] | EA-forward [F] | RPCF+TNP r25 [S] | RPCF+TNP r30 [S] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 78.65 ± 0.91 | 67.03 ± 0.57 | 56.42 ± 2.07 | 30.87 ± 1.37 | 81.95 ± 1.14 | 81.41 ± 1.10 |
| thubenchmark | deepconvnet | 76.21 ± 1.61 | 66.52 ± 3.26 | 55.42 ± 3.34 | 7.89 ± 0.43 | 78.71 ± 1.53 | 78.20 ± 1.17 |
| thubenchmark | tsception | 40.39 ± 2.87 | 9.69 ± 1.75 | 5.67 ± 1.95 | 1.75 ± 0.23 | 46.37 ± 1.18 | 45.12 ± 0.84 |
| thubenchmark | atcnet | 86.13 ± 1.46 | 78.81 ± 2.24 | 70.34 ± 1.51 | 22.62 ± 12.67 | 84.34 ± 1.29 | 84.77 ± 1.30 |
| thubenchmark | conformer | 75.60 ± 1.47 | 57.62 ± 10.39 | 57.15 ± 6.39 | 10.40 ± 8.44 | 78.44 ± 1.87 | 78.28 ± 1.22 |
| thubenchmark | tcnet | 50.13 ± 3.21 | 24.75 ± 2.49 | 24.11 ± 2.05 | 0.00 ± 0.00 | 58.79 ± 2.85 | 57.97 ± 3.63 |
| seediv | eegnet | 25.99 ± 3.70 | 1.91 ± 0.14 | 9.96 ± 1.01 | 21.80 ± 4.82 | 30.27 ± 2.39 | 29.53 ± 2.50 |
| seediv | deepconvnet | 23.33 ± 1.31 | 0.83 ± 0.19 | 4.15 ± 0.93 | 12.40 ± 1.82 | 24.77 ± 1.79 | 24.77 ± 1.28 |
| seediv | tsception | 33.94 ± 0.86 | 0.49 ± 0.11 | 3.56 ± 0.45 | 27.05 ± 0.21 | 35.62 ± 3.09 | 35.47 ± 3.54 |
| seediv | atcnet | 28.93 ± 0.97 | 1.88 ± 0.23 | 5.95 ± 0.48 | 16.75 ± 6.35 | 31.05 ± 1.82 | 30.90 ± 1.62 |
| seediv | conformer | 35.17 ± 0.80 | 24.50 ± 7.64 | 5.38 ± 1.14 | 17.15 ± 3.05 | 43.59 ± 2.05 | 44.30 ± 3.59 |
| seediv | tcnet | 32.18 ± 0.32 | 0.41 ± 0.25 | 6.59 ± 0.69 | 19.37 ± 3.89 | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | 30.15 ± 2.89 | 3.95 ± 0.60 | 7.01 ± 0.70 | 0.00 ± 0.00 | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | 17.20 ± 2.14 | 5.10 ± 1.10 | 10.46 ± 2.22 | 0.00 ± 0.00 | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | 25.13 ± 1.50 | 0.19 ± 0.19 | 4.52 ± 2.55 | 23.91 ± 1.73 | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | 25.44 ± 1.29 | 1.34 ± 0.43 | 8.54 ± 1.77 | 2.34 ± 4.22 | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | 20.80 ± 1.86 | 4.56 ± 1.66 | 12.15 ± 1.46 | 6.59 ± 11.19 | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | 16.05 ± 11.26 | 4.41 ± 0.52 | 12.91 ± 0.99 | 0.00 ± 0.00 | Pending(0/5) | Pending(0/5) |

## 2. 跨攻击平均：RPCF+TNP 与普通 AT

### 2.1 四攻击等权平均 Mean4

包括 AA、FGSM、PGD-200、CW-L2；本表仍存在 [F]/[S] 口径差异。

| 数据集 | Backbone | Madry [F] | TRADES [F] | FBF [F] | EA-forward [F] | RPCF+TNP r25 [S] | RPCF+TNP r30 [S] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 58.99 ± 0.61 | 50.77 ± 0.46 | 43.14 ± 1.29 | 28.03 ± 0.69 | 68.31 ± 1.08 | 66.10 ± 1.19 |
| thubenchmark | deepconvnet | 57.62 ± 1.24 | 50.73 ± 2.32 | 43.07 ± 2.22 | 15.94 ± 0.70 | 61.02 ± 0.60 | 60.15 ± 0.54 |
| thubenchmark | tsception | 31.50 ± 2.28 | 9.16 ± 1.21 | 6.19 ± 1.41 | 1.76 ± 0.23 | 40.70 ± 1.20 | 38.03 ± 0.63 |
| thubenchmark | atcnet | 64.62 ± 1.12 | 59.70 ± 1.67 | 53.48 ± 1.04 | 24.49 ± 13.30 | 72.51 ± 2.03 | 70.19 ± 1.45 |
| thubenchmark | conformer | 57.60 ± 1.06 | 46.45 ± 5.28 | 43.64 ± 3.15 | 14.75 ± 12.32 | 65.75 ± 0.67 | 63.83 ± 0.48 |
| thubenchmark | tcnet | 37.92 ± 2.14 | 22.42 ± 2.26 | 22.56 ± 0.90 | 0.73 ± 0.42 | 47.05 ± 2.57 | 45.65 ± 2.87 |
| seediv | eegnet | 19.76 ± 2.08 | 1.47 ± 0.09 | 7.14 ± 0.74 | 18.12 ± 3.91 | 24.55 ± 1.82 | 23.52 ± 1.94 |
| seediv | deepconvnet | 17.21 ± 1.00 | 0.80 ± 0.13 | 3.32 ± 0.58 | 12.75 ± 1.56 | 22.15 ± 1.19 | 21.00 ± 0.94 |
| seediv | tsception | 24.86 ± 0.64 | 0.52 ± 0.12 | 3.30 ± 0.31 | 27.05 ± 0.22 | 28.96 ± 2.77 | 27.87 ± 2.39 |
| seediv | atcnet | 21.27 ± 0.70 | 1.78 ± 0.23 | 4.89 ± 0.36 | 16.72 ± 5.16 | 25.72 ± 2.01 | 24.90 ± 1.69 |
| seediv | conformer | 33.29 ± 3.78 | 19.20 ± 3.48 | 7.86 ± 1.35 | 20.80 ± 3.17 | 42.48 ± 2.02 | 43.19 ± 3.04 |
| seediv | tcnet | 23.96 ± 0.38 | 0.73 ± 0.28 | 6.25 ± 0.52 | 19.24 ± 3.53 | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | 21.96 ± 2.08 | 3.11 ± 0.48 | 5.45 ± 0.63 | 5.34 ± 1.44 | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | 12.61 ± 1.55 | 4.05 ± 1.01 | 7.95 ± 1.61 | 1.77 ± 1.06 | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | 23.09 ± 3.07 | 0.54 ± 0.31 | 3.84 ± 1.41 | 23.85 ± 1.61 | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | 18.72 ± 0.78 | 1.30 ± 0.23 | 6.86 ± 1.51 | 9.83 ± 4.80 | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | 18.28 ± 1.25 | 5.20 ± 0.59 | 10.58 ± 1.83 | 12.19 ± 5.72 | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | 12.21 ± 8.53 | 4.48 ± 0.27 | 11.29 ± 0.91 | 6.44 ± 3.16 | Pending(0/5) | Pending(0/5) |

### 2.2 同范数攻击均值 MeanLinf3

去除 CW-L2 后的敏感性对照，用于避免结论仅由混合范数平均驱动；三种攻击共享 Linf 预算，但算法与计算强度不同。

| 数据集 | Backbone | Madry [F] | TRADES [F] | FBF [F] | EA-forward [F] | RPCF+TNP r25 [S] | RPCF+TNP r30 [S] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 78.65 ± 0.82 | 67.69 ± 0.61 | 57.51 ± 1.72 | 37.38 ± 0.92 | 82.16 ± 1.22 | 81.82 ± 1.39 |
| thubenchmark | deepconvnet | 76.83 ± 1.66 | 67.64 ± 3.10 | 57.42 ± 2.96 | 21.26 ± 0.93 | 79.65 ± 1.23 | 79.45 ± 1.09 |
| thubenchmark | tsception | 41.94 ± 3.00 | 11.95 ± 1.66 | 7.60 ± 1.95 | 1.76 ± 0.23 | 48.85 ± 1.37 | 47.12 ± 0.92 |
| thubenchmark | atcnet | 86.16 ± 1.49 | 79.57 ± 2.22 | 71.31 ± 1.38 | 32.54 ± 18.00 | 85.39 ± 1.17 | 85.64 ± 1.43 |
| thubenchmark | conformer | 76.80 ± 1.41 | 61.92 ± 7.06 | 58.18 ± 4.20 | 19.53 ± 16.60 | 79.34 ± 1.38 | 79.09 ± 1.29 |
| thubenchmark | tcnet | 49.45 ± 2.81 | 27.82 ± 2.59 | 27.59 ± 1.39 | 0.44 ± 0.45 | 59.95 ± 3.44 | 58.95 ± 3.88 |
| seediv | eegnet | 25.28 ± 3.92 | 1.96 ± 0.12 | 9.52 ± 0.99 | 22.70 ± 4.05 | 31.76 ± 2.45 | 31.00 ± 2.55 |
| seediv | deepconvnet | 22.91 ± 1.32 | 1.04 ± 0.20 | 4.42 ± 0.77 | 16.90 ± 2.20 | 26.59 ± 1.50 | 25.96 ± 1.32 |
| seediv | tsception | 33.04 ± 0.85 | 0.69 ± 0.15 | 4.35 ± 0.43 | 27.05 ± 0.22 | 37.20 ± 3.32 | 36.71 ± 3.18 |
| seediv | atcnet | 28.28 ± 0.92 | 2.33 ± 0.31 | 6.52 ± 0.49 | 20.48 ± 6.14 | 32.97 ± 2.19 | 32.55 ± 1.79 |
| seediv | conformer | 35.11 ± 0.97 | 20.91 ± 4.09 | 7.37 ± 0.85 | 22.06 ± 3.46 | 44.62 ± 1.82 | 46.33 ± 3.16 |
| seediv | tcnet | 31.76 ± 0.24 | 0.70 ± 0.31 | 8.09 ± 0.73 | 22.53 ± 2.37 | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | 29.26 ± 2.75 | 4.15 ± 0.63 | 7.27 ± 0.84 | 0.11 ± 0.10 | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | 16.82 ± 2.06 | 5.40 ± 1.34 | 10.60 ± 2.15 | 1.98 ± 0.60 | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | 24.85 ± 1.46 | 0.34 ± 0.23 | 4.33 ± 1.85 | 23.91 ± 1.73 | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | 24.80 ± 1.10 | 1.56 ± 0.43 | 8.93 ± 1.90 | 6.91 ± 4.21 | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | 20.29 ± 1.42 | 5.44 ± 1.49 | 12.44 ± 0.68 | 9.92 ± 10.03 | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | 15.85 ± 11.22 | 5.38 ± 0.38 | 13.82 ± 1.01 | 3.41 ± 2.38 | Pending(0/5) | Pending(0/5) |

## 3. RPCF+TNP 内部消融：现有组件对照

以下四组均使用 [S]，不混入完整测试集数值：A=Madry，B=RPCF_AT（无 TNP），C=Madry+TNP，D=RPCF_AT+TNP。A/B 的 raw 指同一 n512 子集上的未净化分类器结果，不是主表的完整测试集 [F] 数值。

同种子、同攻击类型、同测试索引范围进行比较；攻击分别针对各自模型生成。B/D 含额外 RPCF 适配训练，A/C 未提供等训练预算的额外微调对照，因此这是组件级经验对照，不是排除训练预算混杂的严格机制归因。

EXP-031 实际仅采用全层微调、训练秩15/20/25/30/35/40静态均匀权重1/6、无特征损失的配置；RPCF 额外训练100 epochs。未实际运行的层选择、特征损失、动态/静态多秩等内部机制变体不补造数值；本节不能代替这些机制消融。协议来源见 [实验记录](../EXPERIMENTS.md)。

差值：B−A 为未净化适配差异；D−C 为净化后适配差异；D−B 为 RPCF 模型上的净化前后变化。所有差值均逐种子计算，再求均值和标准差。

### 3.1 PGD / rank25

| 数据集 | Backbone | A:Madry | B:RPCF | C:Madry+TNP | D:RPCF+TNP | B−A (pp) | D−C (pp) | D−B (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 77.81 ± 1.47 | 77.89 ± 1.86 | 79.14 ± 0.94 | 81.95 ± 1.14 | +0.08 ± 0.68 | +2.81 ± 1.26 | +4.06 ± 1.40 |
| thubenchmark | deepconvnet | 75.82 ± 1.79 | 76.80 ± 1.33 | 78.59 ± 1.06 | 78.71 ± 1.53 | +0.98 ± 1.35 | +0.12 ± 0.63 | +1.91 ± 0.35 |
| thubenchmark | tsception | 39.96 ± 2.15 | 36.17 ± 2.37 | 48.71 ± 2.24 | 46.37 ± 1.18 | -3.79 ± 1.01 | -2.34 ± 2.03 | +10.20 ± 2.02 |
| thubenchmark | atcnet | 85.74 ± 1.68 | 85.98 ± 1.83 | 82.97 ± 1.61 | 84.34 ± 1.29 | +0.23 ± 1.18 | +1.37 ± 1.32 | -1.64 ± 0.86 |
| thubenchmark | conformer | 75.31 ± 1.41 | 74.22 ± 2.53 | 79.06 ± 1.93 | 78.44 ± 1.87 | -1.09 ± 3.11 | -0.62 ± 2.95 | +4.22 ± 1.46 |
| thubenchmark | tcnet | 49.84 ± 4.11 | 49.53 ± 2.95 | 56.25 ± 4.11 | 58.79 ± 2.85 | -0.31 ± 1.85 | +2.54 ± 3.08 | +9.26 ± 2.71 |
| seediv | eegnet | 25.47 ± 4.30 | 28.05 ± 2.35 | 26.41 ± 4.47 | 30.27 ± 2.39 | +2.58 ± 3.09 | +3.87 ± 3.98 | +2.23 ± 0.86 |
| seediv | deepconvnet | 23.01 ± 2.22 | 20.08 ± 1.31 | 25.47 ± 2.61 | 24.77 ± 1.79 | -2.93 ± 1.70 | -0.70 ± 1.26 | +4.69 ± 0.95 |
| seediv | tsception | 32.85 ± 3.02 | 31.95 ± 2.59 | 35.12 ± 2.26 | 35.62 ± 3.09 | -0.90 ± 1.25 | +0.51 ± 1.95 | +3.67 ± 1.05 |
| seediv | atcnet | 29.02 ± 0.35 | 28.67 ± 1.38 | 30.55 ± 1.92 | 31.05 ± 1.82 | -0.35 ± 1.11 | +0.51 ± 1.24 | +2.38 ± 1.28 |
| seediv | conformer | 33.55 ± 3.27 | 27.85 ± 2.54 | 37.19 ± 3.21 | 43.59 ± 2.05 | -5.70 ± 1.38 | +6.41 ± 2.19 | +15.74 ± 2.39 |
| seediv | tcnet | 30.47 ± 2.32 | Pending(4/5) | 32.38 ± 1.69 | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |

### 3.2 PGD / rank30

| 数据集 | Backbone | A:Madry | B:RPCF | C:Madry+TNP | D:RPCF+TNP | B−A (pp) | D−C (pp) | D−B (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 77.81 ± 1.47 | 77.89 ± 1.86 | 79.49 ± 1.97 | 81.41 ± 1.10 | +0.08 ± 0.68 | +1.91 ± 1.82 | +3.52 ± 0.95 |
| thubenchmark | deepconvnet | 75.82 ± 1.79 | 76.80 ± 1.33 | 77.81 ± 1.39 | 78.20 ± 1.17 | +0.98 ± 1.35 | +0.39 ± 0.84 | +1.41 ± 0.64 |
| thubenchmark | tsception | 39.96 ± 2.15 | 36.17 ± 2.37 | 47.07 ± 1.76 | 45.12 ± 0.84 | -3.79 ± 1.01 | -1.95 ± 1.94 | +8.95 ± 2.51 |
| thubenchmark | atcnet | 85.74 ± 1.68 | 85.98 ± 1.83 | 83.52 ± 1.44 | 84.77 ± 1.30 | +0.23 ± 1.18 | +1.25 ± 1.21 | -1.21 ± 0.75 |
| thubenchmark | conformer | 75.31 ± 1.41 | 74.22 ± 2.53 | 78.28 ± 1.67 | 78.28 ± 1.22 | -1.09 ± 3.11 | +0.00 ± 2.27 | +4.06 ± 1.47 |
| thubenchmark | tcnet | 49.84 ± 4.11 | 49.53 ± 2.95 | 55.35 ± 3.20 | 57.97 ± 3.63 | -0.31 ± 1.85 | +2.62 ± 2.74 | +8.44 ± 3.09 |
| seediv | eegnet | 25.47 ± 4.30 | 28.05 ± 2.35 | 26.29 ± 4.40 | 29.53 ± 2.50 | +2.58 ± 3.09 | +3.24 ± 4.12 | +1.48 ± 1.45 |
| seediv | deepconvnet | 23.01 ± 2.22 | 20.08 ± 1.31 | 25.35 ± 1.97 | 24.77 ± 1.28 | -2.93 ± 1.70 | -0.59 ± 1.16 | +4.69 ± 0.65 |
| seediv | tsception | 32.85 ± 3.02 | 31.95 ± 2.59 | 35.16 ± 2.63 | 35.47 ± 3.54 | -0.90 ± 1.25 | +0.31 ± 0.98 | +3.52 ± 1.35 |
| seediv | atcnet | 29.02 ± 0.35 | 28.67 ± 1.38 | 30.04 ± 1.67 | 30.90 ± 1.62 | -0.35 ± 1.11 | +0.86 ± 1.12 | +2.23 ± 0.66 |
| seediv | conformer | 33.55 ± 3.27 | 27.85 ± 2.54 | 39.96 ± 3.21 | 44.30 ± 3.59 | -5.70 ± 1.38 | +4.34 ± 2.43 | +16.45 ± 2.34 |
| seediv | tcnet | 30.47 ± 2.32 | Pending(4/5) | 33.28 ± 1.90 | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |

### 3.3 Mean4 / rank25

| 数据集 | Backbone | A:Madry | B:RPCF | C:Madry+TNP | D:RPCF+TNP | B−A (pp) | D−C (pp) | D−B (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 58.41 ± 0.99 | 58.53 ± 1.19 | 66.61 ± 1.23 | 68.31 ± 1.08 | +0.12 ± 0.37 | +1.70 ± 0.91 | +9.79 ± 0.72 |
| thubenchmark | deepconvnet | 57.27 ± 1.36 | 58.20 ± 1.07 | 63.72 ± 1.63 | 61.02 ± 0.60 | +0.94 ± 1.08 | -2.71 ± 1.50 | +2.81 ± 0.55 |
| thubenchmark | tsception | 31.08 ± 1.70 | 29.06 ± 1.50 | 41.70 ± 2.19 | 40.70 ± 1.20 | -2.02 ± 0.79 | -1.00 ± 2.16 | +11.64 ± 1.63 |
| thubenchmark | atcnet | 64.35 ± 1.36 | 64.70 ± 1.32 | 73.12 ± 1.37 | 72.51 ± 2.03 | +0.35 ± 0.90 | -0.62 ± 0.85 | +7.81 ± 1.95 |
| thubenchmark | conformer | 57.39 ± 0.93 | 56.34 ± 1.85 | 64.53 ± 1.42 | 65.75 ± 0.67 | -1.05 ± 2.16 | +1.22 ± 1.33 | +9.41 ± 1.54 |
| thubenchmark | tcnet | 37.59 ± 2.86 | 37.89 ± 2.42 | 45.45 ± 2.73 | 47.05 ± 2.57 | +0.30 ± 0.90 | +1.60 ± 1.05 | +9.16 ± 1.44 |
| seediv | eegnet | 19.34 ± 2.60 | 20.66 ± 1.69 | 21.09 ± 3.29 | 24.55 ± 1.82 | +1.33 ± 1.94 | +3.46 ± 3.28 | +3.89 ± 0.83 |
| seediv | deepconvnet | 16.99 ± 1.54 | 14.79 ± 0.97 | 22.98 ± 1.56 | 22.15 ± 1.19 | -2.20 ± 1.35 | -0.83 ± 0.93 | +7.35 ± 1.21 |
| seediv | tsception | 24.07 ± 2.12 | 23.49 ± 1.90 | 28.37 ± 2.02 | 28.96 ± 2.77 | -0.59 ± 0.48 | +0.60 ± 1.68 | +5.48 ± 1.43 |
| seediv | atcnet | 21.11 ± 0.55 | 20.94 ± 1.05 | 24.65 ± 1.34 | 25.72 ± 2.01 | -0.18 ± 0.60 | +1.07 ± 1.19 | +4.79 ± 1.86 |
| seediv | conformer | 31.85 ± 3.63 | 26.94 ± 2.32 | 37.00 ± 2.72 | 42.48 ± 2.02 | -4.90 ± 1.56 | +5.48 ± 2.46 | +15.54 ± 1.65 |
| seediv | tcnet | 22.79 ± 1.77 | Pending(4/5) | 28.58 ± 1.04 | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |

### 3.4 Mean4 / rank30

| 数据集 | Backbone | A:Madry | B:RPCF | C:Madry+TNP | D:RPCF+TNP | B−A (pp) | D−C (pp) | D−B (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 58.41 ± 0.99 | 58.53 ± 1.19 | 64.72 ± 1.81 | 66.10 ± 1.19 | +0.12 ± 0.37 | +1.39 ± 0.98 | +7.58 ± 0.55 |
| thubenchmark | deepconvnet | 57.27 ± 1.36 | 58.20 ± 1.07 | 61.04 ± 1.43 | 60.15 ± 0.54 | +0.94 ± 1.08 | -0.89 ± 1.36 | +1.94 ± 0.59 |
| thubenchmark | tsception | 31.08 ± 1.70 | 29.06 ± 1.50 | 39.42 ± 1.77 | 38.03 ± 0.63 | -2.02 ± 0.79 | -1.40 ± 1.66 | +8.96 ± 1.50 |
| thubenchmark | atcnet | 64.35 ± 1.36 | 64.70 ± 1.32 | 70.64 ± 1.32 | 70.19 ± 1.45 | +0.35 ± 0.90 | -0.46 ± 0.59 | +5.49 ± 1.35 |
| thubenchmark | conformer | 57.39 ± 0.93 | 56.34 ± 1.85 | 62.76 ± 1.49 | 63.83 ± 0.48 | -1.05 ± 2.16 | +1.06 ± 1.51 | +7.49 ± 1.48 |
| thubenchmark | tcnet | 37.59 ± 2.86 | 37.89 ± 2.42 | 43.86 ± 2.51 | 45.65 ± 2.87 | +0.30 ± 0.90 | +1.80 ± 2.34 | +7.76 ± 2.07 |
| seediv | eegnet | 19.34 ± 2.60 | 20.66 ± 1.69 | 20.59 ± 3.08 | 23.52 ± 1.94 | +1.33 ± 1.94 | +2.93 ± 3.09 | +2.85 ± 1.11 |
| seediv | deepconvnet | 16.99 ± 1.54 | 14.79 ± 0.97 | 22.04 ± 1.10 | 21.00 ± 0.94 | -2.20 ± 1.35 | -1.04 ± 0.85 | +6.20 ± 1.10 |
| seediv | tsception | 24.07 ± 2.12 | 23.49 ± 1.90 | 27.69 ± 1.93 | 27.87 ± 2.39 | -0.59 ± 0.48 | +0.19 ± 0.61 | +4.38 ± 0.97 |
| seediv | atcnet | 21.11 ± 0.55 | 20.94 ± 1.05 | 23.87 ± 1.23 | 24.90 ± 1.69 | -0.18 ± 0.60 | +1.04 ± 1.11 | +3.96 ± 1.50 |
| seediv | conformer | 31.85 ± 3.63 | 26.94 ± 2.32 | 38.50 ± 3.16 | 43.19 ± 3.04 | -4.90 ± 1.56 | +4.70 ± 1.91 | +16.25 ± 1.44 |
| seediv | tcnet | 22.79 ± 1.77 | Pending(4/5) | 27.81 ± 1.16 | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending(4/5) |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending(0/5) |

## 4. RPCF+TNP rank25 与 rank30：测试净化秩敏感性

rank25/30 是测试阶段 TNP 净化秩，不是两种独立训练的 RPCF 配置。正差值表示 rank30 更高；不能只依据鲁棒性忽略 Clean 保真度。Clean 固定使用 AA 产物对应 clean 输入记录。

### 4.1 Clean

| 数据集 | Backbone | rank25 | rank30 | r30−r25 (pp) | r30更高的种子数 |
| --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 90.31 ± 1.05 | 91.17 ± 1.05 | +0.86 ± 0.30 | 5/5 |
| thubenchmark | deepconvnet | 91.52 ± 0.74 | 92.15 ± 0.83 | +0.62 ± 1.01 | 4/5 |
| thubenchmark | tsception | 74.92 ± 0.78 | 75.78 ± 0.84 | +0.86 ± 0.60 | 4/5 |
| thubenchmark | atcnet | 93.28 ± 0.71 | 93.79 ± 0.45 | +0.51 ± 0.41 | 5/5 |
| thubenchmark | conformer | 90.78 ± 0.59 | 91.13 ± 0.78 | +0.35 ± 0.74 | 3/5 |
| thubenchmark | tcnet | 78.52 ± 1.51 | 79.92 ± 1.78 | +1.41 ± 0.65 | 5/5 |
| seediv | eegnet | 36.91 ± 3.37 | 36.76 ± 3.35 | -0.16 ± 0.67 | 3/5 |
| seediv | deepconvnet | 41.09 ± 1.29 | 41.41 ± 0.98 | +0.31 ± 0.51 | 3/5 |
| seediv | tsception | 46.56 ± 3.88 | 46.84 ± 3.19 | +0.27 ± 1.08 | 2/5 |
| seediv | atcnet | 43.32 ± 1.77 | 44.02 ± 1.49 | +0.70 ± 0.78 | 4/5 |
| seediv | conformer | 49.92 ± 2.24 | 54.14 ± 3.04 | +4.22 ± 1.75 | 5/5 |
| seediv | tcnet | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |

### 4.2 PGD

| 数据集 | Backbone | rank25 | rank30 | r30−r25 (pp) | r30更高的种子数 |
| --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 81.95 ± 1.14 | 81.41 ± 1.10 | -0.55 ± 0.72 | 1/5 |
| thubenchmark | deepconvnet | 78.71 ± 1.53 | 78.20 ± 1.17 | -0.51 ± 0.60 | 1/5 |
| thubenchmark | tsception | 46.37 ± 1.18 | 45.12 ± 0.84 | -1.25 ± 0.84 | 0/5 |
| thubenchmark | atcnet | 84.34 ± 1.29 | 84.77 ± 1.30 | +0.43 ± 0.38 | 4/5 |
| thubenchmark | conformer | 78.44 ± 1.87 | 78.28 ± 1.22 | -0.16 ± 0.84 | 2/5 |
| thubenchmark | tcnet | 58.79 ± 2.85 | 57.97 ± 3.63 | -0.82 ± 1.05 | 1/5 |
| seediv | eegnet | 30.27 ± 2.39 | 29.53 ± 2.50 | -0.74 ± 0.70 | 1/5 |
| seediv | deepconvnet | 24.77 ± 1.79 | 24.77 ± 1.28 | +0.00 ± 0.52 | 2/5 |
| seediv | tsception | 35.62 ± 3.09 | 35.47 ± 3.54 | -0.16 ± 1.05 | 3/5 |
| seediv | atcnet | 31.05 ± 1.82 | 30.90 ± 1.62 | -0.16 ± 0.65 | 2/5 |
| seediv | conformer | 43.59 ± 2.05 | 44.30 ± 3.59 | +0.70 ± 2.74 | 2/5 |
| seediv | tcnet | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |

### 4.3 Mean4

| 数据集 | Backbone | rank25 | rank30 | r30−r25 (pp) | r30更高的种子数 |
| --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 68.31 ± 1.08 | 66.10 ± 1.19 | -2.21 ± 0.42 | 0/5 |
| thubenchmark | deepconvnet | 61.02 ± 0.60 | 60.15 ± 0.54 | -0.87 ± 0.25 | 0/5 |
| thubenchmark | tsception | 40.70 ± 1.20 | 38.03 ± 0.63 | -2.68 ± 0.65 | 0/5 |
| thubenchmark | atcnet | 72.51 ± 2.03 | 70.19 ± 1.45 | -2.32 ± 0.65 | 0/5 |
| thubenchmark | conformer | 65.75 ± 0.67 | 63.83 ± 0.48 | -1.92 ± 0.36 | 0/5 |
| thubenchmark | tcnet | 47.05 ± 2.57 | 45.65 ± 2.87 | -1.40 ± 0.76 | 0/5 |
| seediv | eegnet | 24.55 ± 1.82 | 23.52 ± 1.94 | -1.04 ± 0.37 | 0/5 |
| seediv | deepconvnet | 22.15 ± 1.19 | 21.00 ± 0.94 | -1.15 ± 0.37 | 0/5 |
| seediv | tsception | 28.96 ± 2.77 | 27.87 ± 2.39 | -1.09 ± 0.87 | 0/5 |
| seediv | atcnet | 25.72 ± 2.01 | 24.90 ± 1.69 | -0.82 ± 0.43 | 0/5 |
| seediv | conformer | 42.48 ± 2.02 | 43.19 ± 3.04 | +0.71 ± 1.65 | 3/5 |
| seediv | tcnet | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |

### 4.4 MeanLinf3

| 数据集 | Backbone | rank25 | rank30 | r30−r25 (pp) | r30更高的种子数 |
| --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 82.16 ± 1.22 | 81.82 ± 1.39 | -0.34 ± 0.34 | 0/5 |
| thubenchmark | deepconvnet | 79.65 ± 1.23 | 79.45 ± 1.09 | -0.20 ± 0.17 | 0/5 |
| thubenchmark | tsception | 48.85 ± 1.37 | 47.12 ± 0.92 | -1.73 ± 0.60 | 0/5 |
| thubenchmark | atcnet | 85.39 ± 1.17 | 85.64 ± 1.43 | +0.25 ± 0.40 | 3/5 |
| thubenchmark | conformer | 79.34 ± 1.38 | 79.09 ± 1.29 | -0.25 ± 0.27 | 1/5 |
| thubenchmark | tcnet | 59.95 ± 3.44 | 58.95 ± 3.88 | -1.00 ± 0.80 | 0/5 |
| seediv | eegnet | 31.76 ± 2.45 | 31.00 ± 2.55 | -0.76 ± 0.50 | 0/5 |
| seediv | deepconvnet | 26.59 ± 1.50 | 25.96 ± 1.32 | -0.63 ± 0.30 | 0/5 |
| seediv | tsception | 37.20 ± 3.32 | 36.71 ± 3.18 | -0.49 ± 0.64 | 2/5 |
| seediv | atcnet | 32.97 ± 2.19 | 32.55 ± 1.79 | -0.42 ± 0.52 | 1/5 |
| seediv | conformer | 44.62 ± 1.82 | 46.33 ± 3.16 | +1.71 ± 2.05 | 5/5 |
| seediv | tcnet | Pending(4/5) | Pending(4/5) | Pending(4/5) | Pending |
| bciciv2a | eegnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | deepconvnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tsception | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | atcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | conformer | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |
| bciciv2a | tcnet | Pending(0/5) | Pending(0/5) | Pending(0/5) | Pending |

## 5. 描述性汇总与结论边界

- PGD、rank25：完整五种子的同范围 RPCF+TNP−Madry+TNP 对照共 11 组，平均差为正 8 组、为负 3 组、持平 0 组。该计数不是显著性检验，也不代表未完成组合。
- PGD、rank30：完整五种子的同范围 RPCF+TNP−Madry+TNP 对照共 11 组，平均差为正 8 组、为负 2 组、持平 1 组。该计数不是显著性检验，也不代表未完成组合。
- Mean4、rank25：完整五种子的同范围 RPCF+TNP−Madry+TNP 对照共 11 组，平均差为正 7 组、为负 4 组、持平 0 组。该计数不是显著性检验，也不代表未完成组合。
- Mean4、rank30：完整五种子的同范围 RPCF+TNP−Madry+TNP 对照共 11 组，平均差为正 7 组、为负 4 组、持平 0 组。该计数不是显著性检验，也不代表未完成组合。
- Clean 的完整 rank 对照有 11 组：rank30 平均更高 10 组，rank25 平均更高 1 组，持平 0 组。
- PGD 的完整 rank 对照有 11 组：rank30 平均更高 2 组，rank25 平均更高 8 组，持平 1 组。
- Mean4 的完整 rank 对照有 11 组：rank30 平均更高 1 组，rank25 平均更高 10 组，持平 0 组。
- MeanLinf3 的完整 rank 对照有 11 组：rank30 平均更高 2 组，rank25 平均更高 9 组，持平 0 组。

净化秩在鲁棒表现与 Clean 保真之间存在经验取舍，具体方向以上述当前快照的完整条件计数为准；不能将上一版仅9个完整条件的结论直接外推到本次新增条件。所有 Pending 条件仍不参与上述计数。

因此，现有实验支持按数据集、骨干、攻击和净化秩报告条件性结果，不支持仅凭并列主表宣称普遍优于普通 AT。论文主实验仍需统一样本范围；机制消融仍需专门控制变量实验；秩选择应依据独立验证集而不是本报告测试结果。

源报告另有 65 条跨攻击 raw Clean 记录差异及部分 EA-forward 异常表现；本报告保留所有种子，未修正或排除这些值。常规攻击结果不替代自适应攻击评估，已有有限 BPDA 范围见源报告。

## 文件与复现

- [源结果报告](report.md)：原始全指标矩阵、协议及异常说明。
- [专题汇总 CSV](focused_summary.csv)：本文全部单元格的状态、种子数、均值和样本标准差。
- [专题逐种子 CSV](focused_by_seed.csv)：逐种子准确率/差值，统一使用百分数或百分点尺度。
- [生成脚本](build_focused_report.py)：仅从已核验 conditions_long.csv 派生本报告，无第三方依赖。
- [来源校验清单](focused_manifest.json)：输入哈希与派生统计口径。

运行：`python docs/EXP031_results_20260911_142412/build_focused_report.py`。运行会重新生成本专题报告及其派生 CSV/清单，不改变源 CSV、训练、攻击、日志或 checkpoint。
