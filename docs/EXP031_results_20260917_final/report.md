# EXP-031 完整矩阵最终汇总

严格汇总 run-id：`exp031_full_20260729_174215`；任务、训练记录、攻击与净化产物校验通过。

## 口径

- 三数据集 × 六 backbone × 五种子（42–46）、fold0；表格准确率均为五种子均值 ± 样本标准差，单位 %；差值单位百分点（pp）。
- [F] 是完整 test split，[S] 是按预先固定索引选出的最多 n512 子集。[F] 与 [S] 不能直接相减；本报告的配对差值只在同一范围、同一数据集/backbone/seed/attack 内计算。
- Madry 与 RPCF_AT 各用自身分类器生成 white-box 攻击，再分别净化；因此配对比较是方法级结果，不是同一对抗样本上的纯净化效应。
- AA/FGSM/PGD 是 L∞ ε=0.03；CW 是不受该 ε 限制的 L2 攻击。不同范数不求混合平均，也不把任何跨攻击均值解释为最坏情况鲁棒性。
- TNP 净化使用固定测试 rank25/30；RPCF_AT 训练为全层、在线 PGD-10、六 rank 静态 1/6、无 feature loss。
- 以下为已保存产物的严格一致性审核与统计，不是重新训练或重新运行攻击；EA-forward clean 字段的异常单独列出。

## 完整测试集 raw PGD-200 鲁棒准确率 [F]

攻击分别针对各自 checkpoint；以下各列处于相同的完整 test 范围。

| 数据集 | Backbone | Madry | TRADES | FBF | EA-forward | RPCF_AT |
| --- | --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 78.65 ± 0.91 | 67.03 ± 0.57 | 56.42 ± 2.07 | 30.87 ± 1.37 | 78.58 ± 0.94 |
| thubenchmark | deepconvnet | 76.21 ± 1.61 | 66.52 ± 3.26 | 55.42 ± 3.34 | 7.89 ± 0.43 | 77.11 ± 1.48 |
| thubenchmark | tsception | 40.39 ± 2.87 | 9.69 ± 1.75 | 5.67 ± 1.95 | 1.75 ± 0.23 | 36.91 ± 2.42 |
| thubenchmark | atcnet | 86.13 ± 1.46 | 78.81 ± 2.24 | 70.34 ± 1.51 | 22.62 ± 12.67 | 85.96 ± 1.39 |
| thubenchmark | conformer | 75.60 ± 1.47 | 57.62 ± 10.39 | 57.15 ± 6.39 | 10.40 ± 8.44 | 74.18 ± 1.98 |
| thubenchmark | tcnet | 50.13 ± 3.21 | 24.75 ± 2.49 | 24.11 ± 2.05 | 0.00 ± 0.00 | 49.74 ± 2.78 |
| seediv | eegnet | 25.99 ± 3.70 | 1.91 ± 0.14 | 9.96 ± 1.01 | 21.80 ± 4.82 | 28.64 ± 1.15 |
| seediv | deepconvnet | 23.33 ± 1.31 | 0.83 ± 0.19 | 4.15 ± 0.93 | 12.40 ± 1.82 | 20.23 ± 0.72 |
| seediv | tsception | 33.94 ± 0.86 | 0.49 ± 0.11 | 3.56 ± 0.45 | 27.05 ± 0.21 | 32.93 ± 0.45 |
| seediv | atcnet | 28.93 ± 0.97 | 1.88 ± 0.23 | 5.95 ± 0.48 | 16.75 ± 6.35 | 28.29 ± 0.81 |
| seediv | conformer | 35.17 ± 0.80 | 24.50 ± 7.64 | 5.38 ± 1.14 | 17.15 ± 3.05 | 29.39 ± 2.10 |
| seediv | tcnet | 32.18 ± 0.32 | 0.41 ± 0.25 | 6.59 ± 0.69 | 19.37 ± 3.89 | 32.02 ± 0.10 |
| bciciv2a | eegnet | 30.15 ± 2.89 | 3.95 ± 0.60 | 7.01 ± 0.70 | 0.00 ± 0.00 | 28.39 ± 2.90 |
| bciciv2a | deepconvnet | 17.20 ± 2.14 | 5.10 ± 1.10 | 10.46 ± 2.22 | 0.00 ± 0.00 | 14.06 ± 2.30 |
| bciciv2a | tsception | 25.13 ± 1.50 | 0.19 ± 0.19 | 4.52 ± 2.55 | 23.91 ± 1.73 | 23.45 ± 1.51 |
| bciciv2a | atcnet | 25.44 ± 1.29 | 1.34 ± 0.43 | 8.54 ± 1.77 | 2.34 ± 4.22 | 21.53 ± 1.34 |
| bciciv2a | conformer | 20.80 ± 1.86 | 4.56 ± 1.66 | 12.15 ± 1.46 | 6.59 ± 11.19 | 11.65 ± 1.65 |
| bciciv2a | tcnet | 16.05 ± 11.26 | 4.41 ± 0.52 | 12.91 ± 0.99 | 0.00 ± 0.00 | 22.64 ± 6.04 |

## n512 净化后 PGD-200 鲁棒准确率 [S]

Madry 与 RPCF_AT 的 TNP 结果处于相同 source indices/labels；这不是完整 test 准确率。

| 数据集 | Backbone | Madry+TNP r25 | RPCF+TNP r25 | Madry+TNP r30 | RPCF+TNP r30 |
| --- | --- | --- | --- | --- | --- |
| thubenchmark | eegnet | 79.14 ± 0.94 | 81.95 ± 1.14 | 79.49 ± 1.97 | 81.41 ± 1.10 |
| thubenchmark | deepconvnet | 78.59 ± 1.06 | 78.71 ± 1.53 | 77.81 ± 1.39 | 78.20 ± 1.17 |
| thubenchmark | tsception | 48.71 ± 2.24 | 46.37 ± 1.18 | 47.07 ± 1.76 | 45.12 ± 0.84 |
| thubenchmark | atcnet | 82.97 ± 1.61 | 84.34 ± 1.29 | 83.52 ± 1.44 | 84.77 ± 1.30 |
| thubenchmark | conformer | 79.06 ± 1.93 | 78.44 ± 1.87 | 78.28 ± 1.67 | 78.28 ± 1.22 |
| thubenchmark | tcnet | 56.25 ± 4.11 | 58.79 ± 2.85 | 55.35 ± 3.20 | 57.97 ± 3.63 |
| seediv | eegnet | 26.41 ± 4.47 | 30.27 ± 2.39 | 26.29 ± 4.40 | 29.53 ± 2.50 |
| seediv | deepconvnet | 25.47 ± 2.61 | 24.77 ± 1.79 | 25.35 ± 1.97 | 24.77 ± 1.28 |
| seediv | tsception | 35.12 ± 2.26 | 35.62 ± 3.09 | 35.16 ± 2.63 | 35.47 ± 3.54 |
| seediv | atcnet | 30.55 ± 1.92 | 31.05 ± 1.82 | 30.04 ± 1.67 | 30.90 ± 1.62 |
| seediv | conformer | 37.19 ± 3.21 | 43.59 ± 2.05 | 39.96 ± 3.21 | 44.30 ± 3.59 |
| seediv | tcnet | 32.38 ± 1.69 | 33.05 ± 1.96 | 33.28 ± 1.90 | 33.32 ± 2.12 |
| bciciv2a | eegnet | 31.17 ± 2.96 | 32.11 ± 2.70 | 30.39 ± 3.18 | 31.29 ± 2.84 |
| bciciv2a | deepconvnet | 17.58 ± 2.38 | 14.73 ± 2.19 | 17.34 ± 2.05 | 14.34 ± 2.33 |
| bciciv2a | tsception | 25.51 ± 1.14 | 23.98 ± 1.50 | 25.47 ± 1.30 | 24.02 ± 1.39 |
| bciciv2a | atcnet | 25.27 ± 1.05 | 22.11 ± 1.22 | 25.16 ± 0.91 | 21.60 ± 1.03 |
| bciciv2a | conformer | 20.78 ± 1.45 | 11.64 ± 1.76 | 20.62 ± 1.57 | 11.64 ± 1.64 |
| bciciv2a | tcnet | 19.41 ± 9.34 | 23.05 ± 6.16 | 18.79 ± 9.54 | 23.24 ± 6.32 |

## 直接配对：RPCF_AT − Madry

正数表示 RPCF_AT 更高。raw [F] 与 TNP [S] 的差值分别计算，不能跨列相减。

| 数据集 | Backbone | raw [F] | TNP r25 [S] | TNP r30 [S] |
| --- | --- | --- | --- | --- |
| thubenchmark | eegnet | -0.07 ± 0.49 | +2.81 ± 1.26 | +1.91 ± 1.82 |
| thubenchmark | deepconvnet | +0.90 ± 1.28 | +0.12 ± 0.63 | +0.39 ± 0.84 |
| thubenchmark | tsception | -3.48 ± 1.70 | -2.34 ± 2.03 | -1.95 ± 1.94 |
| thubenchmark | atcnet | -0.17 ± 0.78 | +1.37 ± 1.32 | +1.25 ± 1.21 |
| thubenchmark | conformer | -1.42 ± 2.42 | -0.62 ± 2.95 | +0.00 ± 2.27 |
| thubenchmark | tcnet | -0.38 ± 0.95 | +2.54 ± 3.08 | +2.62 ± 2.74 |
| seediv | eegnet | +2.66 ± 2.88 | +3.87 ± 3.98 | +3.24 ± 4.12 |
| seediv | deepconvnet | -3.10 ± 0.88 | -0.70 ± 1.26 | -0.59 ± 1.16 |
| seediv | tsception | -1.00 ± 0.72 | +0.51 ± 1.95 | +0.31 ± 0.98 |
| seediv | atcnet | -0.64 ± 0.32 | +0.51 ± 1.24 | +0.86 ± 1.12 |
| seediv | conformer | -5.78 ± 1.49 | +6.41 ± 2.19 | +4.34 ± 2.43 |
| seediv | tcnet | -0.16 ± 0.26 | +0.66 ± 0.90 | +0.04 ± 0.42 |
| bciciv2a | eegnet | -1.76 ± 1.08 | +0.94 ± 1.38 | +0.90 ± 1.18 |
| bciciv2a | deepconvnet | -3.14 ± 1.14 | -2.85 ± 0.90 | -3.01 ± 1.30 |
| bciciv2a | tsception | -1.69 ± 1.77 | -1.52 ± 1.64 | -1.45 ± 1.67 |
| bciciv2a | atcnet | -3.91 ± 0.66 | -3.16 ± 0.93 | -3.55 ± 0.58 |
| bciciv2a | conformer | -9.16 ± 3.26 | -9.14 ± 3.00 | -8.98 ± 3.00 |
| bciciv2a | tcnet | +6.59 ± 5.68 | +3.63 ± 3.88 | +4.45 ± 4.10 |

各攻击中，18 个数据集–backbone 条件的五种子平均配对差值方向（仅描述性计数，非显著性检验）：

| 攻击 | 比较 | RPCF 更高 | Madry 更高 | 持平 |
| --- | --- | --- | --- | --- |
| autoattack | raw [F] | 6 | 12 | 0 |
| autoattack | TNP r25 [S] | 10 | 7 | 1 |
| autoattack | TNP r30 [S] | 10 | 8 | 0 |
| fgsm | raw [F] | 6 | 12 | 0 |
| fgsm | TNP r25 [S] | 11 | 7 | 0 |
| fgsm | TNP r30 [S] | 11 | 7 | 0 |
| pgd | raw [F] | 3 | 15 | 0 |
| pgd | TNP r25 [S] | 11 | 7 | 0 |
| pgd | TNP r30 [S] | 11 | 6 | 1 |
| cw | raw [F] | 6 | 8 | 4 |
| cw | TNP r25 [S] | 10 | 8 | 0 |
| cw | TNP r30 [S] | 9 | 8 | 1 |

## 同模型净化变化：TNP − 未净化 [S]

下表先在同一 n512 子集、同一 checkpoint、同一攻击和种子内求差，再统计18个条件的方向。正数表示净化后鲁棒准确率更高。

| 攻击 | 模型 | rank | 净化提高 | 净化降低 | 持平 |
| --- | --- | --- | --- | --- | --- |
| autoattack | madry | 25 | 18 | 0 | 0 |
| autoattack | madry | 30 | 18 | 0 | 0 |
| autoattack | rpcf_at | 25 | 18 | 0 | 0 |
| autoattack | rpcf_at | 30 | 18 | 0 | 0 |
| pgd | madry | 25 | 16 | 2 | 0 |
| pgd | madry | 30 | 15 | 3 | 0 |
| pgd | rpcf_at | 25 | 16 | 2 | 0 |
| pgd | rpcf_at | 30 | 16 | 2 | 0 |

## 有限自适应攻击审计

仅 THU/EEGNet/RPCF_AT+TNP 五种子，BPDA+PGD-10（L∞ ε=0.03、步长0.006）；没有对应 Madry BPDA 对照，且步数与上表 PGD-200 不同，不作横向强弱结论。

| rank | 净化 clean | BPDA 鲁棒 |
| --- | --- | --- |
| 25 | 90.20 ± 0.96 | 82.11 ± 1.09 |
| 30 | 91.02 ± 0.97 | 81.37 ± 0.94 |

## 审计警告与解释边界

- EA-forward 同一 checkpoint 的四攻击记录中，65/90 个 dataset–backbone–seed 条件的 raw clean accuracy 不一致；最大跨度 3.11 pp（seediv/conformer/seed46）。攻击脚本在攻击循环后才计算 clean accuracy，但目前未证实这就是数值漂移的充分原因。该异常使 EA-forward clean 数值及其跨攻击解释需要复核；本报告不修正原产物。
- RPCF_AT 含额外 100 epochs 适配训练；此矩阵不能单独归因于 logit 对齐、全层微调或静态 rank 权重，也不能宣称在所有条件下普遍优于 baseline。
- 攻击与净化结果的协议、索引、标签、clean 张量内容与准确率范围经过严格汇总校验；仍未独立重跑模型推理或攻击。

## 可复核产物

- 严格完整性：[completeness.json](../../logs/exp031/exp031_full_20260729_174215/summary/completeness.json)
- 全指标长表：[conditions_long.csv](../../logs/exp031/exp031_full_20260729_174215/summary/conditions_long.csv)
- 五种子聚合：[five_seed_mean_std.csv](../../logs/exp031/exp031_full_20260729_174215/summary/five_seed_mean_std.csv)
- 配对差值：[rpcf_at_minus_madry_paired.csv](../../logs/exp031/exp031_full_20260729_174215/summary/rpcf_at_minus_madry_paired.csv)
- 净化变化：[tnp_minus_raw_paired.csv](../../logs/exp031/exp031_full_20260729_174215/summary/tnp_minus_raw_paired.csv)
- BPDA：[bpda.csv](../../logs/exp031/exp031_full_20260729_174215/summary/bpda.csv)
