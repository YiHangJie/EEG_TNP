# EXP-034：Madry 配对净化、自适应攻击基线与时间张量化 TR

- 日期：2026-09-28；来源：用户本次明确要求新建并补齐实验。
- 状态：**Complete**；`exp034_full_20260928_v1` 于2026-09-28 14:09:53（北京时间）完成140/140任务、275/275指标，验收errors/pending为空。此前8项单元测试和11任务/21指标真实数据smoke通过；本次仅确认执行与汇总完整性，研究结论待结果分析。
- 关联 IDEA-018 和 EXP-033 结构对照；不新增训练策略，不改已有 baseline 或 EXP-031/032/033 产物。
- 先前 EXP-032 记录中的“不要启动 EXP-033–036”为当时执行范围；本次按用户新指令单独建立 EXP-034。

## 固定协议与来源

- seed 42–46、fold0；`stable_subset_indices` 仍使用 NumPy `RandomState(seed + fold*1000)`，每条件固定 S512；smoke 仅取该正式子集前2个样本，不二次抽样。
- 复用 EXP-031 `exp031_full_20260729_174215` 的 Madry/TRADES/FBF/EA-forward 权重及已有 Madry 攻击；clean-only 和净化器权重复用 EXP-032 `exp032_full_20260917_2020`。
- 净化器 MagNet-Reformer（EEG adaptation，无 detector/rejection）与 DCAE 只在现有 train split 训练过；本轮固定其权重，不针对 Madry 或测试攻击再训练。
- 保存源 checkpoint/payload SHA256、索引、标签、split、参数、逐样本预测和范数审计；攻击来源必须对应所选分类器，禁止拿 clean-only 攻击替代 Madry 攻击。
- 所有输出写入独立 `logs/exp034/<run_id>/`；原始数据、历史日志、checkpoint 和已交付 Excel 保持不变。

## 1. 两种 AP + Madry

| 实验 | 数据集 / 模型 | 攻击 | 五 seed 新增结果行 |
| --- | --- | --- | ---: |
| 主实验 | THUBenchmark / SEED-IV / bciciv2a × 六模型 | PGD-200 L∞ ε=.03 | 180 |
| 跨攻击跨范数 | THUBenchmark / EEGNet | FGSM L∞ .03、PGD-200 L∞ .03、PGD-200 L2 1、AutoAttack L∞ .03、CW L2 | 50，其中10行复用本轮主实验同条件 |
| 自适应攻击 | THUBenchmark / EEGNet | PGD-10 L∞ ε=.03、α=.006 | 10 |

六模型为 EEGNet、DeepConvNet、TSCeption、ATCNet、Conformer、TCNet。方法标签为 `madry_magnet`、`madry_dcae`。非自适应攻击沿用 EXP-032 五攻击协议：PGD-200 α=2/255、无随机起点；PGD-L2 为全样本绝对半径1、α=.1、200步、5次重启；CW 为无半径约束L2（200步、lr=.1、c=10000、kappa=1）。

自适应攻击针对 **净化器+Madry 的完整可微组合**，使用真实梯度，不能标为 BPDA。PGD-10 其余参数与既有 EXP-031 BPDA/EXP-032 AP 修订一致：batch1、clean起点、无随机起点、无额外重启/EOT、无额外输入范围裁剪、返回最后一步。

## 2. 五个原始分类器的 PGD-10

THUBenchmark / EEGNet × clean-only、Madry、TRADES、FBF、EA-forward × seeds42–46，共25行。使用与上一节相同 PGD-10 更新/投影规则，记录 `classifier_whitebox`、`exact_through_classifier`；原始分类器不需要 BPDA。

EA-forward 保留原方法的 EA 前缓存输入和逐样本 subject id，经原 subject-aware wrapper 求真实梯度，不对普通输入重复标准化。每次攻击固定 batch1；clean 推理批量沿用 EXP-032 baseline 的32，并记录推理前后的模型状态变化。EA-forward 的展示名可沿用 `ABAT-style`，原始 method 标签保留。

## 3. 时间张量化 TR 效果与效率

- 仅 THUBenchmark / EEGNet / 固定 Madry checkpoint / PGD-200 L∞ .03 / S512 / 五seed。
- 与 EXP-033 相同插值、FFT时间重采样和逆变换：共同 `(10,11,2048)` 表示重排为 `(10,11,2,…,2)`，11个二进制时间模，共13阶。
- 采用标准 TensorLy TR-SVD，模式0首核为H；闭环向量长度14、首尾相等、所有bond至少2，避免退化为开链TT。首切分满足 `r0*r1<=10`；后续rank按 unfolding 行/列维度预先约束，实际分解必须与预登记rank/参数数目一致。
- 沿用 EXP-033 预算档25/30，对应14,731/19,891个分解参数，候选在±5%内；预算档不是各结构共同rank。
- 候选属于固定的“闭环首bond+首内部bond+内部cap”家族，去重并剔除逐bond受支配候选。只用同seed、同EXP-033的32个validation clean样本，以平均相对Frobenius重构误差选秩；并列时按绝对预算偏差和rank字典序。禁止按测试准确率选候选。
- 效果每seed512样本；保存SA/RA、clean MSE、净化adv对clean MSE、removed MSE、实际rank/参数/预算偏差。
- 计时独立进行，普通计算任务全部结束后全局串行；独占GPU、预热剔除，每seed8clean+8adv，记录分解时间、总净化时间、峰值allocated显存和两类压缩率。

## 4. 下次结果整理时增加参数敏感性图表

本轮只登记需求，**不修改现有 Excel、不提前重做结果报告**。用户下次手动要求整理实验结果时：

- rank 图：横轴15/20/25/30/35/40，纵轴SA/RA，分别显示五seed均值与样本标准差，原始CAF作为参考。
- CE/KL 权重图：五个权重分别作图，横轴相对默认倍率（0、0.5、1、2），附实际系数；CAF raw和TRP rank25/30分开，显示SA/RA与误差条。
- 不跨不同参数项/攻击混合平均；保留逐seed散点或明细，未完成点留空，不挑选最优seed。
- Excel内嵌图表及源数据，保持协议、rank、数据范围和来源可追溯。

## 执行、断点与验收

- 正式140任务：90 external（每任务评估两净化器）、25 raw PGD10、5 validation选秩、10结构效果、10独占计时；275个不重复seed级指标行。
- 8张GPU每卡至多一个worker，检测实际空闲后派发，不停止或共享其他任务；CPU线程固定2。计时单独串行。结果/断点不重复保存全量攻击或净化张量，避免占用现有233GiB剩余空间。
- 自适应PGD逐样本断点保留RNG和模型state；结构断点保留预测/诊断和RNG。恢复须通过任务、来源和输入身份指纹，禁止改变随机偏移。
- 任一任务失败后在途任务收尾，等待明确重试入口；已完成结果须通过输出hash及源文件校验，不能用旧run填补缺失。
- 最终汇总校验每任务精确方法/攻击/evaluation/预算键、预测与准确率、样本数、攻击协议和五seed；只有完整正式运行才标Complete。smoke单独标记，不能作论文结果。

```bash
RUN_ID=exp034_full_20260928_v1 bash rpcf/run_exp034.sh plan
RUN_ID=exp034_full_20260928_v1 GPU_IDS=0,1,2,3,4,5,6,7 nohup setsid bash rpcf/run_exp034.sh run > logs/exp034_full_20260928_v1.controller.log 2>&1 < /dev/null &
```

实时状态：`logs/exp034/<run_id>/runtime.json`；逐任务实时日志：`workers/*.log`；最终输出：`summary/metrics_long.csv`、`metrics_grouped.csv`、`report.json`。
