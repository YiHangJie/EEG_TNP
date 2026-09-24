# EXP-032 外部方法复现记录

## 原则与来源

统一使用 EXP-031 的 subject split、预处理、六 backbone 与五 seed，而非复制各论文不同的数据划分。下列实现属于在统一协议下重建方法，不声称复现原论文的表格数值。所有净化器都作用于 clean 和 adversarial 输入，不能用攻击标签决定是否绕过净化。

| 方法 | 一手来源 | 获取状态 |
| --- | --- | --- |
| MagNet-Reformer | [作者实现](https://github.com/Trevillie/MagNet)，[论文](https://arxiv.org/abs/1705.09064) | 已核对作者 `defensive_models.py` 与 `train_defense.py` |
| DCAE | Ding, Li & Li, IEEE Access 2024, [DOI](https://doi.org/10.1109/ACCESS.2024.3467154)，[论文全文](https://www.researchgate.net/publication/384349100_Adversarial_Defense_based_on_Denoising_Convolutional_Autoencoder_in_EEG-based_Brain-computer_Interfaces) | 已核对作者稿公式 (10)–(13)、表 1/2、III-B/IV-C |
| GAN adversarial training | Aissa et al., IEEE Internet of Things Magazine 7(3), 44–49, 2024, [DOI](https://doi.org/10.1109/IOTM.001.2300262)，[论文摘要](https://www.researchgate.net/publication/380303570_Enhancing_EEG_Signal_Classifier_Robustness_Against_Adversarial_Attacks_Using_a_Generative_Adversarial_Network_Approach) | **已移出：用户确认不属于本轮净化方法** |

## MagNet-Reformer (EEG adaptation)

- 使用作者 MNIST-I 的 `[3, average, 3]` 对称自编码器结构：Conv3 → AvgPool2 → Conv3 → Conv3 → Upsample2 → Conv3 → Conv1，各卷积 kernel3/same，sigmoid。
- 作者示例训练设置为 Adam、MSE、Gaussian sigma0.1、100 epochs；batch256。仅比较 reformer，不含检测/拒绝或随机自编码器集成。
- EEG 按每个 dataset/seed 的训练 split 极值做固定标量仿射变换到 [0,1]，输出再反变换；噪声在该编码域添加并裁剪。这个范围桥接是 EEG 适配假设，未使用测试统计。
- 网络 padding 到偶数后裁剪；保留作者 activity L2 权重 1e-9 的训练意图，若实现省略/改变此项必须在产物中显式标识。

## DCAE（统一 EEG 协议复现）

- 表 2 encoder：Conv32/ReLU/MaxPool2 → Conv32/ReLU/MaxPool2 → Conv16/ReLU/MaxPool2 → **Conv16/Sigmoid bottleneck**。表中最后一层不能因 III-B 概括“3 个卷积层”而遗漏。
- Decoder：Upsample2/Conv16/ReLU → Upsample2/Conv32/ReLU → Upsample2/Conv32/ReLU → Conv1/Sigmoid；所有卷积 kernel3。
- 表 1：Adam lr0.001、100 epochs、batch128、Gaussian beta0.5、rho0.02、稀疏 KL 系数7.5e-5。损失是 MSE 加 bottleneck 单元的 Bernoulli KL 总和。
- 论文未充分说明：卷积 padding/bias、上采样插值方式、clip 数值范围、z-score 信号如何对应 sigmoid 输出、KL 中 eta 的具体平均轴、DCAE 验证选模及噪声是否每 epoch 重采样。
- 预先固定的复现假设：conv stride1/padding1/bias=True、nearest upsample、右下 replicate padding 至8的倍数后裁剪；eta 对 batch 平均、对 latent 单元求和；噪声每次访问由全局 torch RNG 生成；Adam 默认 betas/eps、weight_decay0；固定训练100 epochs，验证只作诊断，不根据 test 选模型。
- 为保留论文 sigmoid 结构，采用与 MagNet 相同的 train-only 固定仿射 [0,1] 桥接与噪声输入裁剪，输出反变换。**该桥接未经论文确认，是复现假设**；beta0.5 对应编码域，需在 checkpoint 中保存 low/high 和原 EEG 等效噪声尺度。若取得作者更明确设定，应另开配置变体，不能覆盖旧产物。
- 原论文使用 within-subject 随机划分、不同采样/滤波和 epsilon0.1；EXP-032 使用本仓库协议与 epsilon0.03，以及固定 L2 半径1.0。这是公平比较的协议替换。
- 同时报告“攻击分类器后净化”与“攻击完整 DCAE+classifier”结果；不把前者标为 adaptive white-box。

## GAN adversarial training：已移出本轮

摘要明确描述 FGSM 真实对抗样本、GAN 生成对抗 EEG，以及将正常/真实对抗/生成对抗样本用于分类器训练。因此当前将其登记为训练期防御。网络层数、条件信息、噪声输入、标签来源、损失函数及权重、训练顺序、epoch/batch、FGSM 预算仍需正文确认。

用户检查内容后明确撤回该方法；不实现、不训练，也不将其作为 EXP-032 未完成项。

## 实现验证补充

- DCAE 在 RTX3080 10GB 上直接用 THU shape `[128,1,64,1500]` 反向会 OOM；采用无随机操作的分块 activation checkpoint 重算，保持 batch128、网络与损失定义不变。
- MagNet 作者的极小 activity L2(1e-9)在 EEG 适配版省略，checkpoint 明确写 `magnet_activity_l2_deviation=author_1e-9_omitted`；训练噪声改为沿用仓库 torch 全局种子、每次访问重采样，而非作者一次生成的 NumPy 固定噪声。

## EXP-031 既有 AT baseline 审计口径

- Madry/TRADES/FBF 使用同一仓库 `train_AT.py`；EXP-031 显式配置400epochs上限、patience20、AdamW lr0.001、weight_decay1e-4、PGD10/alpha0.006/eps0.03、gradient clip0.01；实际batch从每任务status提取。clean-only复用该入口和验证选模规则。
- TRADES 的既有实现是 `CE + 0.1 * KL(p_clean.detach() || p_adv)`，inner攻击期间保留train模式。与[TRADES作者实现](https://github.com/yaodongyu/TRADES/blob/master/trades.py)相比，作者在inner攻击时eval、outer损失时train，outer KL的clean分支不detach。因此EXP-031这列应理解为仓库TRADES变体，不能称作者代码的逐行复现。beta较小和这些实现差异可能影响鲁棒性，但本次审计尚未通过消融建立因果；不能断言是异常低结果的唯一原因。
- FBF固定3replays；保留旧实现及训练记录，不因攻击准确率低而删除该方法。
- EA-forward的输入是缓存raw值，普通分支再次按时间维标准化；应核对相同source identity，分别匹配自身输入值。THU两样本最大数值差4.768e-7。旧clean跨攻击漂移仍作为异常；新评估在攻击前统一推理clean并检查重复预测和state_dict变化。
- EXP-032不据测试结果调整AT超参数，也不覆盖旧checkpoint；新主表同时输出 `baseline_audit.csv` 与逐seed配对表，使上述边界可追踪。
