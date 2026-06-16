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
