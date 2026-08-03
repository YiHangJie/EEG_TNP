# IDEAS.md

本文件用于记录计划中、实现中、已测试或已放弃的研究 idea。它的作用是在修改代码前，把研究假设和预期实现方式写清楚。

## 编号规则

- 使用连续编号：`IDEA-001`、`IDEA-002`、`IDEA-003`，以此类推。
- idea 被放弃后，编号也不要复用。
- 相关实验使用 `docs/EXPERIMENTS.md` 中的 `EXP-xxx` 编号进行关联。

## Idea 通用模板

### IDEA-XXX：标题

- **状态：** 计划中 / 实现中 / 已测试 / 已放弃
- **动机：**
  - 为什么这个 idea 值得测试。
- **核心假设：**
  - 这个 idea 想验证的、可以被证伪的假设。
- **方法：**
  - 高层方法、算法思路，或训练/评估流程变化。
- **预期实现：**
  - 预计新增或修改的文件、模块、配置、开关或脚本。
  - 优先使用新模块、新配置或显式开关，避免改变 baseline 行为。
- **评估指标：**
  - 指标、数据集、划分方式、随机种子和对比 baseline。
- **风险：**
  - 可能的失败模式、混杂因素、计算成本、复现风险或实现风险。
- **相关实验：**
  - `EXP-XXX`：Pending
- **备注：**
  - 额外上下文、参考资料或待确认问题。

## 示例

### IDEA-001：将 PTR 改造为 rank growth 版本

- **状态：** 已测试
- **动机：**
  - 测试 rank growth 策略能否实现 adaptive rank selection，使得 EEG_TNP 针对不同样本有不同程度的净化。
- **核心假设：**
  - 每个样本有个性化的 rank 选择有助于提升净化效果。
- **方法：**
  - 在原先 PTR 的基础上，一开始初始化高 rank，然后先只优化低 rank 微参数，然后逐步提升 rank。
- **预期实现：**
  - 改造出一个新的 TN 架构。
- **评估指标：**
  - purify 的 standard acc 和 robust acc。
- **风险：**
  - 改造 PTR 的代码有 bug。
- **相关实验：**
  - `EXP-001`
  - `EXP-002`
  - `EXP-003`
  - `EXP-004`
- **备注：**
  - 暂无。

### IDEA-002：评估 rank growth 过程中高频能量的占比

- **状态：** 已测试
- **动机：**
  - EEG_TNP 中 rank 过小会导致干净 EEG 语义结构恢复不足，而 rank 过大可能会进一步重构对抗扰动或非结构化高频噪声。
  - 已有观察表明，对抗扰动相较于原始 EEG 往往具有更明显的高频能量增强，因此可以将新增 rank 所恢复成分的高频能量比例作为判断是否继续增大 rank 的信号。
- **核心假设：**
  - 在 rank 从小到大增加的过程中，早期新增 rank 主要恢复任务相关的低频或结构化 EEG 语义成分。
  - 当 rank 继续增大且重构收益趋于饱和时，新增恢复成分中的高频能量比例会升高，此时继续增大 rank 更可能恢复对抗扰动或噪声，而不是有效 EEG 语义。
  - 因此，可以选择“重构收益已较小且新增成分高频占比较高”之前的 rank 作为当前样本的自适应 rank。
- **方法：**
  - 对每个测试样本，从较小 rank 开始执行 EEG_TNP 净化，得到不同 rank 下的净化结果 $\hat{x}_r$。
  - 计算相邻 rank 之间的新增恢复成分：$a_r = \hat{x}_r - \hat{x}_{r-1}$。
  - 对 $a_r$ 进行频域变换，计算高频能量占比。
- **预期实现：**
  - 改造 tensor_ring_rank_analysis/analyze_tr_rank_predictions.py 里面的分析指标。
- **评估指标：**
  - purify 的 standard acc 和 robust acc，以及 rank 之间高频成分的占比。
- **风险：**
  - 改造 analyze_tr_rank_predictions.py 的代码有 bug。
- **相关实验：**
  - `EXP-005`
  - `EXP-028`
  - `EXP-029`
- **备注：**
  - 暂无。

### IDEA-003：通过 optuna 对 rank growth 配套的早停判断标准做参数寻优

- **状态：** 已测试
- **动机：**
  - optuna 能够更精细地展开对抗扰动规律发现。
- **核心假设：**
  - 我觉得现有的 rank 之间的 JS 指标、 净化前预测标签与净化后预测标签之间的 margin 指标（softmax 后的概率差）、mse指标（原始数据和净化数据之间的差异），应该能有助于 sample wise 的 rank selection。
- **方法：**
  - Rank 之间的 JS divergence：衡量分类结果是否稳定。
  - 净化前预测标签与净化后预测标签之间的 margin：衡量净化后是否保持原始决策语义。
  - MSE 指标：衡量信号保真性。
  - 一方面，以上三者可以设计成通过阈值去约束；另一方面也可以设计成统一在一起的打分函数，施加不同的权重系数（$\mathrm{Score}(r)=\alpha \cdot \mathrm{JS}(p_r, p_{r-1})+\beta \cdot \mathrm{MSE}_r-\gamma \cdot \mathrm{Margin}_r$）。
- **预期实现：**
  - 在原有的 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py` 的框架下，新建一个分析脚本，跑全量的 rank，然后重新统计评估以上三种指标及打分指标，可以参考原有的代码框架。
  - 新建一个基于 optuna 的调参脚本，调整的参数包括 JS、margin、MSE 的阈值，也包括打分函数里面的权重系数。
  - 最终最好能够实现一个严密的 rank selection 标准，有助于 purify 的结果
- **评估指标：**
  - purify 的 standard acc 和 robust acc。
- **风险：**
  - 涉及的代码很多，要注意不要出现隐式 bug。
- **相关实验：**
  - `EXP-007`
  - `EXP-008`
  - `EXP-009`
  - `EXP-010`
  - `EXP-013`
- **备注：**
  - 暂无。

### IDEA-004：在 tensor_ring_rank_analysis 下面新增基于预测熵的指标统计，然后使用 optuna 进行测试。

- **状态：** 已测试
- **动机：**
  - 预测熵是一种uncertainty 的指标，可以衡量分类器输出的可信程度。
- **核心假设：**
  - 若预测熵很大，说明输出的结果不确定（每一类的概率都较大）；若预测熵小，则输出的结果稳定（某一类的概率占大比重）。
- **方法：**
  - $H(p_r) = -\sum_{c=1}^{C} p_r(c)\log p_r(c)$
- **预期实现：**
  - 在原有的 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py` 的框架下，新增预测熵，跑全量的 rank，然后重新统计评估预测熵指标。
  - 在`tensor_ring_rank_analysis/optuna_rank_growth_selection.py`中，新增对预测熵的 optuna 调参。
- **评估指标：**
  - purify 的 standard acc 和 robust acc。
- **风险：**
  - 涉及的代码很多，要注意不要出现隐式 bug。
- **相关实验：**
  - `EXP-011`
- **备注：**
  - 暂无。

### IDEA-005：PTR_3d_rank_soft_mask 可微 soft-rank 净化

- **状态：** 初步验证完成，暂不作为主候选
- **动机：**
  - 现有 `PTR_3d_rank_growth` 使用 hard prefix gate 逐档增加 rank，再通过外部早停规则选择 rank。该路径具备动态 rank 的结构基础，但仍需要离散 rank block 和后验选择，计算成本较高，且优化目标没有直接约束有效 rank。
  - 将离散 `active_rank` 放松为可微 soft-prefix mask，可以在单次 PTR 优化中学习样本级 effective rank，为自适应 rank 分配提供更优雅的建模方式。
- **核心假设：**
  - 通过 `MSE reconstruction + effective-rank penalty` 的目标，soft-prefix gate 可以学习每个样本的最小充分 rank。
  - 相比固定低 rank 或 hard early stopping，soft-rank 机制有机会在保持 robust accuracy 的同时降低 clean accuracy 损伤。
- **方法：**
  - 新增 `PTR_3d_rank_soft_mask`，保留 `max_rank` 的完整 TR 参数容器。
  - 学习一个连续变量 `rho`，并用 `g_i = sigmoid((rho - i) / temperature)` 生成 soft-prefix rank mask。
  - 对时间 bond 使用 `sqrt(g)` 施加 soft mask；空间 core 不加 mask，保持与 `PTR_3d_rank_growth` 相同的 rank 语义。
  - 优化目标只包含重构 MSE 与 `effective_rank / max_rank` 低秩约束，不加入 classifier semantic risk、entropy、JS、margin、残差白化或高频惩罚。
- **预期实现：**
  - 新增 `TN/rank_growth/PTR_3d_rank_soft_mask.py`。
  - 接入 `TN/rank_growth/__init__.py`、`TN/utils.py`、`purify.py`。
  - 在 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py` 新增 `--analysis_mode rank_soft_mask`。
  - 新增 `configs/thubenchmark/PTR3d_rank_soft_mask_8_2048_r40_3d_interpolate.yaml`。
- **评估指标：**
  - clean/adv accuracy、MSE、confidence、entropy。
  - soft-mask 的 `effective_rank`、`rho`、`rank_cost`、近似有效参数量。
  - 与 `EXP-009/010/011` 中固定 rank、`threshold`、`js_mse` 和 entropy selector 对照。
- **风险：**
  - 如果 rank penalty 太小，soft mask 可能退化到接近 `max_rank`。
  - 如果 rank penalty 太大，effective rank 可能过低，导致 clean/robust accuracy 同时下降。
  - 由于目标不包含 classifier-aware 项，不能理论保证 robust accuracy 不下降，需要用实验确认。
- **相关实验：**
  - `EXP-014`
- **备注：**
  - 初版重点验证 soft-rank allocation 是否可运行以及是否形成合理 rank-accuracy trade-off。
  - `EXP-014` 显示 soft mask 可运行，rank penalty 能压低 effective rank，但 `MSE + rank cost` 目标没有学出足够清晰的样本级 rank allocation，也没有超过 `EXP-010/011` 的 hard rank-growth selector。

### IDEA-006：进行对 PTR_3d_rank_growth+JS_MSE 的完整测试，包括不同的 dataset、模型、seed、fold、EPS

- **状态：** 规划中
- **动机：**
  - 多维度的测试才能获得可信的结论。
- **核心假设：**
  - 多组重复性实验能观察性能的稳定性，不同的 dataset、模型观察方法的泛化性。
- **方法：**
  - 构建从训练、攻击、训练集净化、攻击样本净化、净化样本增强的对抗训练（consistancy）、攻击、净化（PTR_3d_rank_growth+JS_MSE）的全流程 sh 脚本。
  - 新建一个存放 log 文件的文件夹。
- **预期实现：**
  - 数据集： thubenchmark、seediv
  - 模型：EEGNet、Conformer
  - seed：42、43、45
  - fold：0、1、2
  - 攻击：autoattack
  - EPS：0.01、0.03、0.05
- **评估指标：**
  - clean/adv accuracy。
- **风险：**
  - 测试量较大，文件较多，要做好文件管理。
- **相关实验：**
  - `EXP-015`
- **备注：**
  - 也要补充用于对比的 baseline 性能的实验，就是普通的 Madry、TRADES、fbf、ABAT（train_AT_ea_forward.py）。
  - baseline 方法都是 AT 方法，只需对抗训练、攻击测试即可。

### IDEA-007：Rank-aware Purification Critical Fine-tuning（RPCF）

- **状态：** 实现完成，跨 seed 复验完成
- **动机：**
  - AT 模型已经具备基础 robust decision boundary，但 EEG_TNP 会引入与 rank 相关的输入分布变化。
  - 全模型继续训练可能破坏已有鲁棒表示；只更新对净化变化最敏感的逻辑层，有机会用较小参数更新适应多 rank purification views。
- **核心假设：**
  - AT 模型中存在 purification-sensitive layers，其 clean-anchor feature shift 显著高于其他层。
  - 对这些层进行多 rank curriculum fine-tuning，可以比原始 AT 更好地适应 purified clean/adversarial inputs，同时保留 AT 能力。
- **方法：**
  - 从 Madry AT checkpoint 出发，对同一训练子集只生成一次 `x_adv`。
  - 对 `x/x_adv` 使用 rank `15,20,25,30,35,40` 分别净化并保存统一 cache。
  - 层评分先计算 clean-anchor 的绝对 shift，再用 `C_l(r)=S_l(r)/(S_previous(r)+eps)` 衡量相邻逻辑层的变化放大率；首层以前一阶段 input shift 为分母。最终分数为 clean-purified 与 adversarial-purified 相对敏感度的跨 rank 等权均值。
  - 选择 Top 40% 逻辑层，冻结其余参数和对应 BatchNorm running statistics。
  - `w.o. sensitivity layer selection` 消融取消冻结并微调 100% 参数，其他训练和评估条件保持不变，用于检验 sensitive-layer selection 是否有独立价值。
  - fine-tuning 使用 clean logits 作为 detached teacher；`x_adv`、多 rank `x_pur` 和 `x_adv_pur` 均通过温度 KL 对齐 clean logits，同时保留 hard-label CE。rank 权重从低 rank 迁移到高 rank，并用于多 rank CE/KL 聚合。
  - `w.o. rank schedule` 消融保持 sensitive-layer selection 不变，但将六个 rank 在所有 epoch 的 CE/KL 聚合权重固定为 `1/6`，用于检验 low-to-high curriculum 的独立作用。
  - 当前 RPCF 不使用 validation early stopping 或 best-checkpoint 回退，固定训练满配置 epochs 并保存最终 epoch；clean/PGD validation 指标只作为诊断。
- **预期实现：**
  - 独立 `rpcf/` 模块、统一 cache、层分析、微调、white-box attack、逐 rank 净化评估、汇总与可续跑 pipeline。
  - 不修改现有 AT、consistency2 或净化 baseline 行为。
- **评估指标：**
  - clean accuracy、AutoAttack accuracy。
  - 各 rank purified clean/adversarial accuracy 与 MSE。
  - selected layer、selected parameter ratio、validation PGD robust accuracy 和动态 rank 权重轨迹。
- **风险：**
  - 六个 rank 对 clean/adv 分别净化，训练集和测试集缓存生成成本很高。
  - clean-anchor `S_advpur` 同时包含攻击与净化导致的 feature shift，这是本方法明确采用的定义，解释时不能等价为纯净化位移。
  - 单条件结果不足以证明跨模型、seed、fold 和 eps 的稳定性。
- **相关实验：**
  - `EXP-017`
  - `EXP-018`
  - `EXP-019`：已完成，seed43 未复现 RPCF 在 seed42 上的净化后鲁棒优势
- **备注：**
  - 首轮只比较 RPCF 与其初始化来源 AT；full-layer 和 static-rank-weight 消融已预留开关，后续按结果决定是否运行。
  - 2026-06-12 起，RPCF 的训练/测试随机子集和 DataLoader seed 逻辑与既有 consistancy pipeline 对齐：子集使用 `seed + fold * 1000`，训练 shuffle 使用 `seed_everything(seed)` 的全局 RNG，不添加 RPCF 私有 offset。

### IDEA-008：RPCF 五方法对比在 eps=0.05 下的跨 seed 复验

- **状态：** 实现中
- **动机：**
  - EXP-019 已在 `eps=0.03`、seed42/43/44 上完成五方法公平对比，需要确认攻击强度提高后结论是否稳定。
- **核心假设：**
  - RPCF 对净化分布的适配收益在 `eps=0.05` 下仍可能存在，但未净化鲁棒性与跨 seed 方差可能进一步恶化。
- **方法：**
  - 完整复用 EXP-019 的 five-method six-rank 协议，只将训练、AutoAttack 和净化评估的 epsilon 从 `0.03` 改为 `0.05`。
  - seed42/43/44、fold0 串行运行，避免 GPU 资源竞争。
- **预期实现：**
  - 新增 `rpcf/run_exp020.sh` 和 `rpcf/run_exp020_all_seeds.sh`。
  - 日志、攻击数据和净化缓存分别隔离到 `logs/exp020`、`ad_data/exp020` 和 `purified_data/exp020`。
- **评估指标：**
  - 五方法的 full clean、full AutoAttack、rank25/30 purified clean/adversarial accuracy。
  - 三 seed 均值、标准差以及与 EXP-019 eps0.03 的差值。
- **风险：**
  - 三个 seed 串行预计耗时约 3 天，任一 seed 失败会使串行脚本停止，需要从对应 seed 断点续跑。
  - 更高 epsilon 可能改变训练收敛和净化分布，不能复用 eps0.03 checkpoint 或缓存。
- **相关实验：**
  - `EXP-020`：运行中
- **备注：**
  - 除 epsilon 和独立产物目录外，其余超参数与 EXP-019 保持一致。

### IDEA-009：RPCF_AT 在线对抗训练与多-rank净化适配

- **状态：** 已测试
- **动机：**
  - 现有 RPCF 只重复使用初始 AT checkpoint 生成的固定 `x_adv`，随着微调参数变化，
    固定对抗视图不再持续追踪当前决策边界，可能导致未净化 white-box 鲁棒性下降。
- **核心假设：**
  - 在每个 RPCF epoch 前对完整原始训练集执行在线 Madry PGD，可以减少基础鲁棒性
    遗忘，同时保留多-rank净化分布上的适配收益。
- **方法：**
  - 每个 epoch 先基于当前模型生成 10-step PGD，对完整训练 split 优化 adversarial CE。
  - 随后在已有 n512 六-rank cache 上执行原 RPCF clean、clean-purified 和
    adversarial-purified CE/KL 损失及动态 rank curriculum。
  - cache 阶段移除固定 `x_adv` 的 CE/KL，避免与在线 Madry AT 重复优化。
  - sensitive-layer selection、冻结 BatchNorm、优化器和最终 epoch 保存策略保持不变。
- **预期实现：**
  - 在 `rpcf/finetune.py` 增加默认关闭的 `--online_madry_at` 模式。
  - 新增 `rpcf/run_exp021_rpcf_at.sh`，复用 EXP-018 seed42 的已有训练缓存。
- **评估指标：**
  - full clean、full AutoAttack、rank25/30 purified clean/adversarial accuracy。
  - 与 Madry AT、six-rank consistancy 和原 RPCF selective 对比。
- **风险：**
  - selective layers 的有限容量可能不足以同时拟合在线攻击与多-rank净化分布。
  - 在线 AT 可能降低 RPCF 当前较高的 clean accuracy，或削弱净化后优势。
  - 与 EXP-020 并发运行会延长耗时，并可能造成 GPU 显存竞争。
- **相关实验：**
  - `EXP-021`：seed42/43/44 已完成
- **备注：**
  - 第一轮只测试 selective + dynamic rank schedule，不同时运行 all-layers 或 uniform 消融。

### IDEA-010：RPCF_AT 在 eps=0.05 下的三 seed 复验

- **状态：** 已测试
- **动机：**
  - EXP-021 已证明 RPCF_AT 在 `eps=0.03` 下能够恢复普通 RPCF 丢失的原始输入
    鲁棒性，并提高 rank25/30 净化后的平均鲁棒准确率。
  - EXP-020 显示普通 RPCF 在 `eps=0.05` 下的 full AutoAttack 明显退化，需要
    验证在线 Madry AT 是否在更强攻击下仍能稳定修复该问题。
- **核心假设：**
  - RPCF_AT 在 `eps=0.05` 下仍能将 full AutoAttack 恢复到 Madry AT 附近，
    同时保留或改善普通 RPCF 的净化后 adversarial accuracy。
- **方法：**
  - 复用 EXP-020 seed42/43/44 的 Madry AT checkpoint、n512 六-rank RPCF
    cache、sensitivity，以及 AT/consistancy 的 rank25/30 净化结果。
  - 从各 seed 的 AT checkpoint 重新执行 100 epochs selective RPCF_AT。
  - 在线 PGD 使用完整训练 split、10 steps、step size `0.01=eps/5`。
  - 对每个 RPCF_AT checkpoint 生成自身 white-box AutoAttack，并执行 n512
    rank25/30 净化评估。
- **预期实现：**
  - 新增 `rpcf/run_exp022_rpcf_at.sh` 和
    `rpcf/run_exp022_all_seeds.sh`。
  - 产物隔离到 `logs/exp022`、`ad_data/exp022` 和
    `purified_data/exp022`。
- **评估指标：**
  - full clean、full AutoAttack、rank25/30 purified clean/adversarial accuracy。
  - 与 EXP-020 的 Madry AT、six-rank consistancy 和普通 RPCF selective 对比。
- **风险：**
  - 更强在线攻击可能造成 clean accuracy 进一步下降。
  - selective layers 容量可能不足以同时适配 `eps=0.05` 原始攻击和净化分布。
- **相关实验：**
  - `EXP-022`：seed42/43/44 已完成
- **备注：**
  - 不重新生成训练 cache 或 sensitivity，避免重复 EXP-020 的高成本阶段。

### IDEA-011：EEG_TNP + RPCF_AT 的 BPDA+PGD adaptive attack

- **状态：** 已实现，smoke 已通过，正式实验待运行
- **动机：**
  - EXP-021/022 的主结果使用“先攻击模型、再离线净化”的评估协议，攻击未直接穿过
    EEG_TNP 净化器。
  - 为了评估 EEG_TNP+RPCF_AT 组合防御在 adaptive white-box 假设下的强度，需要
    将 EEG_TNP 放入攻击循环。
- **核心假设：**
  - EEG_TNP 内部包含非可微或高成本优化过程，可用 BPDA 将其 backward 近似为恒等
    变换；forward 仍保持真实净化，从而给出更严格的 adaptive attack 压力测试。
- **方法：**
  - 复用 EXP-021 seed42/43/44 已完成的 RPCF_AT checkpoint，不重新微调。
  - 对 rank25 和 rank30 分别执行 rank-specific BPDA+PGD-10。
  - 每轮 PGD forward 为 `RPCF_AT(EEG_TNP(x_adv))`，backward 对 EEG_TNP 使用
    identity gradient。
  - epsilon 固定为 `0.03`，PGD step size 为 `0.006=eps/5`，不使用 random
    start，不额外 clamp 数据范围。
  - 测试子集为同 seed/fold 的 n512，抽样规则继续使用
    `stable_subset_indices(len(test), 512, seed, fold)`。
- **预期实现：**
  - 新增 `rpcf/evaluate_bpda_pgd.py`，通过自定义 autograd wrapper 实现 BPDA。
  - 新增 `rpcf/run_exp023_bpda_pgd.sh` 和 `rpcf/run_exp023_all_seeds.sh`。
  - 新增 `rpcf/compare_exp023.py` 汇总三 seed、rank25/30 结果，并可对照 EXP-021
    非 adaptive 净化结果。
- **评估指标：**
  - BPDA purified clean accuracy、BPDA purified adversarial accuracy、attack MSE。
  - 与 EXP-021 同 seed、同 rank 的非 adaptive purified adversarial accuracy 对比。
- **风险：**
  - EEG_TNP 在 PGD 内循环中开销极高，n512×rank25/30×三 seed 可能耗时很长。
  - BPDA identity gradient 是近似攻击，不等价于真实可微净化器梯度。
  - rank-specific 攻击结果不能跨 rank 复用，否则会低估 adaptive attack 强度。
- **相关实验：**
  - `EXP-023`：Pending

### IDEA-012：RPCF_AT 在其他 backbone 上的泛化测试

- **状态：** 实现中
- **动机：**
  - EEGNet 上的 RPCF_AT 已完成三 seed 与 adaptive attack 检查，但仍需要确认方法收益是否依赖特定 backbone。
  - TSCeption、ATCNet、Conformer 的结构差异较大，可用于评估 purification-sensitive layer selection 和在线 AT 的泛化性。
- **核心假设：**
  - 从各 backbone 的 Madry AT checkpoint 出发，RPCF_AT 仍能学习对 rank25/30 EEG_TNP 净化分布的适配。
  - 若收益只在 EEGNet 上出现，则 RPCF_AT 的结论需要收缩到模型特定现象。
- **方法：**
  - 数据集固定为 `thubenchmark`，no-EA subject split，fold0，`eps=0.03`，seed42。
  - backbone 覆盖 `tsception`、`atcnet`、`conformer`。
  - 每个 backbone 先训练 Madry AT，再生成六 rank RPCF cache，分析 sensitivity，执行 selective RPCF_AT 在线微调。
  - 对 Madry AT 和 RPCF_AT 分别执行 white-box AutoAttack 和 rank25/30 净化测试。
  - 同时训练 clean、TRADES、FBF baseline，并在同一攻击协议下评估。
- **预期实现：**
  - 新增 `rpcf/run_exp024_backbone.sh`：单 backbone 可续跑 pipeline。
  - 新增 `rpcf/run_exp024_all_backbones.sh`：按 backbone 串行调度。
  - 新增 `rpcf/compare_exp024.py`：汇总 attack、purification 和 RPCF_AT 诊断产物。
- **评估指标：**
  - full clean accuracy、full AutoAttack accuracy。
  - rank25/30 purified clean/adversarial accuracy 与 MSE。
  - baseline clean/TRADES/FBF 的 clean/AutoAttack accuracy。
- **风险：**
  - 三个 backbone 的从零训练、六 rank cache 和净化评估成本较高。
  - 不同 backbone 的层命名和模块结构可能影响 sensitivity hook 覆盖，需要用 smoke 或早期日志确认。
  - seed42 单 seed 只能作为 backbone 泛化的初步证据，不能替代跨 seed 统计。
- **相关实验：**
  - `EXP-024`：Pending

### IDEA-013：RPCF_AT + Class-conditional MMD

- **状态：** 已实现；EXP-026 已主动停止，正式结论 Pending
- **动机：** RPCF_AT 主要依赖 logit-level CE/KL，未显式约束不同类别的 clean 与多-rank净化表征分布。
- **核心假设：** 在分类层前按类别对齐 clean、clean-purified 和 adversarial-purified 特征分布，可以缓解净化引起的 backbone-dependent distribution shift。
- **方法：** clean feature 作为 detached teacher；对每个 rank 和 batch 内类别计算 multi-kernel RBF biased MMD，并对 clean-pur/adv-pur 等权、按动态 rank 权重聚合。训练使用确定性 4 类×2 trial 均衡 batch。
- **预期实现：** 在 `rpcf/feature_alignment.py` 中新增统一特征 adapter、CMMD 与均衡 sampler；通过 `rpcf.finetune --feature_objective cmmd` 显式启用，不改变 baseline 默认行为。
- **评估指标：** full clean/AutoAttack、rank25/30 purified clean/adversarial accuracy，以及相对 class-balanced RPCF_AT control 的增益。
- **风险：** n512 cache 和 batch8 只能提供小样本类条件估计；biased MMD 与 median bandwidth 的效果可能依赖 backbone。
- **相关实验：** `EXP-026`：Stopped，结果 Pending

### IDEA-014：RPCF_AT + Prototype Learning

- **状态：** 已实现；EXP-026 已主动停止，正式结论 Pending
- **动机：** 仅做全局一致性无法保证净化样本靠近其对应类别的 clean 表征中心。
- **核心假设：** 对齐 clean prototype 与各 rank adversarial-purified prototype，并约束净化 prototype 的类间 margin，可以提升类别条件净化适配而避免中心坍缩。
- **方法：** 使用 L2-normalized penultimate feature；clean prototype detached；按 rank 对齐 adv-pur prototype。预筛比较无 margin 与 `margin=1.0, margin_weight=0.1`。
- **预期实现：** 通过 `rpcf.finetune --feature_objective prototype` 启用，margin 和权重均显式记录到 history。
- **评估指标：** 同 IDEA-013，并记录 prototype alignment/margin loss。
- **风险：** 每 batch 每类只有两个 trial，prototype 方差仍较高；margin 过强可能损伤原始鲁棒边界。
- **相关实验：** `EXP-026`：Stopped，结果 Pending

### IDEA-015：RPCF_AT + Trial-level Contrastive Learning

- **状态：** 已实现；EXP-026 已主动停止，正式结论 Pending
- **动机：** 同一 trial 的攻击与多-rank净化视图具有天然实例一致性，可直接用于约束净化表征。
- **核心假设：** 以 clean trial 为 anchor，将对应 cached adversarial 和各 rank adversarial-purified view 拉近，并推远其他 trial 的全部派生 view，可以增强跨 rank 稳定性。
- **方法：** clean anchor detached；同 trial 的 `x_adv` 与六 rank `x_adv_pur` 为 positives；其他 trial（包括同类别 trial）的所有派生 view 为 negatives；temperature 固定为 0.1。
- **预期实现：** 通过 `rpcf.finetune --feature_objective contrastive` 启用；batch 少于两个 trial 时安全跳过并记录计数。
- **评估指标：** 同 IDEA-013，并记录 contrastive loss 与 skipped batch 数。
- **风险：** 将同类别其他 trial 视为 negative 可能排斥类内共享结构；cached adversarial 来自初始化 AT checkpoint，可能随微调逐渐陈旧。
- **相关实验：** `EXP-026`：Stopped，结果 Pending

### IDEA-016：样本级动态 Rank 的防御上限

- **状态：** 已实现并通过 smoke，EXP-027 正式运行中
- **动机：** 固定 rank 对所有 trial 使用同一净化强度，无法同时适配样本难度、攻击残留和信号保真差异。在投入 selector 训练前，应先确认“每个样本选择一个最终 rank”是否客观存在足够大的鲁棒收益空间。
- **核心假设：** 对同一 clean/adversarial trial 在 `15,20,25,30,35,40` 中联合选择一个 rank，可在基本保持 clean accuracy 的同时显著超过最佳固定 rank；若 Madry AT 与 RPCF_AT 均成立，则该上限不依赖 RPCF_AT，若仅 RPCF_AT 成立，则只能解释为二者的条件化协同。
- **方法：** 在 `thubenchmark / EEGNet / seed42 / fold0 / eps0.03 / n512` 上分别使用 Madry AT 和 RPCF_AT selective 自身的 white-box AutoAttack。最佳固定 rank 依次按 adversarial accuracy、clean accuracy、低 rank 决定；逐样本 robust-first oracle 依次按 adversarial 正确、clean 正确、adversarial true-class margin、clean true-class margin、低 rank 决定。
- **评估指标：** oracle 相对最佳固定 rank 的 adversarial headroom、clean accuracy 变化、10,000 次 paired bootstrap 95% CI、oracle rank 分布和 rescued/lost trial 数。
- **决策标准：** RPCF_AT headroom 至少 `2.0 pp`、95% CI 下界大于 0 且 clean 下降不超过 `1.0 pp` 才是强支持；`0.5–2.0 pp` 或 CI 包含 0 仅记为弱信号；小于 `0.5 pp` 不支持进入 selector 阶段。
- **风险：** oracle 使用真实标签，只表示理论上限，不能作为可部署防御；当前只研究每个样本一个最终 rank，不涉及净化阶段内 rank trajectory，也不替代后续 adaptive attack。
- **相关实验：** `EXP-027`：运行中

### IDEA-017：EEG_TNP 内生自动定秩与阶段级 Bond Adaptation

- **状态：** 四种 TN-only 代理均不支持；暂不继续以纯重构证据逼近 oracle
- **动机：**
  - EXP-027 在 `thubenchmark / EEGNet / seed42 / fold0 / eps0.03 / n512` 上确认了样本级 rank 的显著理论空间：Madry AT 与 RPCF_AT 相对最佳固定 rank25 的 robust oracle headroom 分别为 `+4.10 pp` 和 `+3.91 pp`。
  - 但 EXP-027 oracle 在测试时使用真实标签、各 rank 正确性和 true-class margin，不能部署；额外训练深度学习 rank selector 又会扩大攻击面并削弱“模型无关张量净化”的方法定位。
  - EXP-014 的单标量 soft-prefix gate 只学到较窄的 effective-rank 分布，说明 `MSE + 线性 rank cost` 与统一 `rho` 不足以表示样本级、bond级结构差异。
- **核心假设：**
  - 从 over-complete Tensor Ring 出发，联合估计每条可变 bond 上各 latent component 的相关性，并成对压缩相邻 core 中证据不足的切片，可以在一次 EEG_TNP 优化内得到样本特有的 bond-rank vector，而不需要枚举后再由分类性能挑选 rank。
  - 在 coarse-to-fine 净化过程中于每个分辨率阶段重新执行 bond pruning，并在后续阶段仅对被结构化残差支持的 component 进行 regrowth，可以形成样本特有的阶段级 rank trajectory。
  - 这种内生自动定秩能够减少低 rank 的语义欠拟合和高 rank 的扰动过拟合，从而收回 EXP-027 oracle headroom 的一部分。
- **方法：**
  - 新建 `PTR_3d_rank_ard`，初始化 `max_rank=40`，不改变现有 `PTR_3d`、`PTR_3d_rank_growth` 或 `PTR_3d_rank_soft_mask` baseline。
  - 为每条可变时间 bond 的每个 component 维护 relevance precision；使用经过 core balancing/canonicalization 的相邻 core 配对切片贡献，交替更新 TN cores 与 relevance，避免 TR gauge scaling 造成虚假剪枝。
  - 在阶段边界物理剪除低相关性 component，并重建 contraction path 与 optimizer state；记录每条 bond 的 active rank、pruned/regrown components 和阶段轨迹。
  - 将 masked self-validation 作为显式变体：只用部分时空位置拟合，检查新增 component 是否改善 held-out reconstruction；最终 rank 确定后再使用完整输入重构。
  - rank inference 完全由当前样本和 TN 优化状态决定，不使用标签、分类 logits、预测正确性或 true-class margin；分类器只用于净化结束后的防御评估。
  - 第二策略不再从单个 over-complete core 的 component norm 截断，而是对六个独立 fixed-rank TNP 重构的样本级 log-MSE 曲线取 diminishing-return knee；该规则参数无关且仍不读取分类器，但计算成本暂为六次净化。
  - 第三策略改用 two-site SVD rounding：合并相邻 cores 后在联合矩阵上做局部最优低秩近似，以未知噪声 optimal hard threshold 自动确定 bond rank，从根本上避免 component permutation/gauge 下的逐切片错配。
  - 第四策略不再裁剪 over-complete core，而是让六个候选 rank 各自独立完成短程 masked TN 优化，以 held-out 时间块误差 one-SE 选秩，再从头完整拟合选中结构。
- **预期实现：**
  - `TN/rank_growth/PTR_3d_rank_ard.py`：bond-wise relevance、成对切片剪枝、stage-wise prune/regrow 和 rank trajectory。
  - `configs/thubenchmark/PTR3d_rank_ard_8_2048_r40_3d_interpolate.yaml`：与 EXP-027 固定 rank 配置对齐的自动定秩配置。
  - 扩展 `TN/rank_growth/__init__.py`、`TN/utils.py`、`purify.py`，仅通过新 model/config 显式启用。
  - 新增 EXP-030 runner、汇总入口和合成 Tensor Ring 自动定秩测试。
- **评估指标：**
  - purified clean/adversarial accuracy、重构 MSE、相对最佳 fixed rank 的 robust gain、相对 EXP-027 oracle 的 headroom recovery ratio 与 oracle regret。
  - 最终 bond-rank vector、样本级 mean/max rank、阶段级 rank trajectory、rank 分布宽度、剪枝/回生计数、有效参数量和净化墙钟时间。
  - 与 fixed rank25、现有 hard rank-growth/soft-mask 和 EXP-027 oracle 对照。
- **风险：**
  - 自动结构秩不必然等于分类最优秩；EXP-028/029 已显示 clean 与 adversarial 整体低秩谱非常接近，因此纯张量证据未必能收回全部 oracle headroom。
  - EXP-030 第一版 `auto_rank_ard_regrow` 已实证暴露这一风险：虽然 clean/adv reconstruction MSE
    约为 `0.0266`，但最终 mean rank 偏向 `34.4`，Madry AT 与 RPCF_AT 的 n64 robust accuracy
    均比相同 subset 的 fixed rank25 低 `4.69 pp`。因此更好的重构或更高的 bond relevance
    不能直接作为分类最优净化秩的代理。
  - 第二策略 log-MSE knee 在 n512 上能产生合理的 rank20/25/30 分布，但 Madry/RPCF_AT 相对
    fixed rank25 仍为 `-0.20/-0.39 pp`；与 oracle rank 的一致率只有约 `12%`。这进一步说明即使
    样本级 reconstruction-complexity knee 非退化，也没有足够的分类边界信息恢复 oracle headroom。
  - 第四策略 independent masked-CV 在 n64 上形成覆盖六个候选的非退化分布，Madry/RPCF robust
    accuracy 均与 fixed rank25 持平，但 clean accuracy 分别下降 `3.12/4.69 pp`；同一 trial 的
    clean/adv rank 一致率仅 `20.31%`。独立训练消除了 core truncation 的工程失真，却仍不能让
    held-out reconstruction error 对齐 defense-optimal rank。
  - TR 存在 gauge 与 component permutation 不变性，若不先做 core balancing，直接按单侧切片范数剪枝可能得到错误 rank。
  - coarse 阶段过早剪枝可能删除高分辨率阶段需要的结构；regrowth 必须限制在 held-out residual 明确支持的方向，并作为必要消融。
  - 无额外深度网络不等于免疫 adaptive attack；若初步性能成立，仍需用 unrolled/BPDA/EOT surrogate 攻击完整自动定秩流程。
- **相关实验：**
  - `EXP-027`：Oracle 上限与目标值
  - `EXP-029`：不同 fixed rank 对净化后结构复杂度的影响
  - `EXP-030`：已完成；四种 TN-only 策略均不支持
- **备注：**
  - 方法依据来自 automatic Tensor Ring rank determination、相邻 core coupled shrinkage、TT/TR rounding 与 rank-adaptive approximation；创新点不应表述为“首次自动确定 TR rank”，而应聚焦其在 EEG 对抗净化中的逐样本、bond级和阶段级结合。
