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
