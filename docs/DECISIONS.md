# DECISIONS.md

本文件用于记录重要研究决策，包括为什么选择、暂停或放弃某些方向。当实验结果或实现发现会影响后续研究方向时，应更新本文件。

## 编号规则

- 使用连续编号：`DEC-001`、`DEC-002`、`DEC-003`，以此类推。
- 尽可能把决策关联到相关的 `IDEA-xxx` 和 `EXP-xxx`。

## 决策通用模板

### DEC-XXX：决策标题

- **日期：** YYYY-MM-DD
- **背景：**
  - 什么情况触发了这个决策。
- **考虑过的选项：**
  - 选项 A：
  - 选项 B：
  - 选项 C：
- **最终选择：**
  - Pending
- **原因：**
  - Pending
- **影响：**
  - 对实现、实验、baseline、计算成本或后续工作的预期影响。
- **相关 idea：**
  - `IDEA-XXX` 或 `None`
- **相关实验：**
  - `EXP-XXX` 或 `None`

### DEC-001：继续评估 rank growth，但先修复 MSE gate

- **日期：** 2026-06-02
- **背景：**
  - `EXP-001` 显示 `PTR_3d_rank_growth` 在 `js=0.005` 下以约 `15.97` 的平均 rank 达到本实验最高 purified adversarial accuracy `0.851562500`。
  - 同时，`rank_growth_max_mse_to_input=0.06` 没有实际传入模型，日志仍显示 `rank_growth_max_mse_to_input: None`。
- **考虑过的选项：**
  - 选项 A：直接采用 `js=0.005` 作为新默认净化配置。
  - 选项 B：暂停 rank growth，回到固定 `rank25/30`。
  - 选项 C：保留 rank growth 作为候选方向，但先修复 MSE gate 并做更系统的 sweep。
- **最终选择：**
  - 选择 C。
- **原因：**
  - `js=0.005` 的 robust acc 优于固定 rank sweep 中的最好结果，但 clean acc 和 MSE 仍不如固定高 rank。
  - MSE gate 的实现链路存在参数未传递问题，不能根据本次 `mse=0.06` 结果做研究判断。
- **影响：**
  - 后续实验应优先修复 `rank_growth_max_mse_to_input` 的传递，再比较 `js=0.005/0.01` 与 MSE gate sweep。
  - 在跨 fold/seed 验证前，不把 rank growth 写入 baseline 默认行为。
- **相关 idea：**
  - `IDEA-001`
- **相关实验：**
  - `EXP-001`

### DEC-002：将 MSE gate 0.08 纳入后续主候选

- **日期：** 2026-06-02
- **背景：**
  - `EXP-002` 修复 MSE gate 传参后，`js=0.02,mse=0.08` 达到 purified adversarial accuracy `0.8515625`，与 `EXP-001` 的 `js=0.005` 持平。
  - 同时，`EXP-002` 的 purified clean accuracy 为 `0.916015625`，高于 `EXP-001/js=0.005` 的 `0.9140625`，且 Adv/Clean MSE 约 `0.064`，明显低于无 MSE gate 动态配置。
- **考虑过的选项：**
  - 选项 A：继续只评估纯 JS 阈值动态 rank。
  - 选项 B：把 `js=0.02,mse=0.08` 作为后续主候选之一。
  - 选项 C：放弃动态 rank，回到固定 `rank25/30`。
- **最终选择：**
  - 选择 B。
- **原因：**
  - `MSE<=0.08` 在保持当前最优 robust acc 的同时改善了 clean acc 和保真度。
  - 该配置的主要代价是平均 rank 和计算成本升高，因此需要继续 sweep，而不是直接改成默认配置。
- **影响：**
  - 后续优先比较固定 `rank25/30`、`js=0.005`、`js=0.02,mse=0.08`。
  - 下一轮 MSE gate sweep 应覆盖 `0.06/0.08/0.10`，并记录计算成本。
- **相关 idea：**
  - `IDEA-001`
- **相关实验：**
  - `EXP-002`

### DEC-003：不将早停图中的 JS 反弹作为 adversarial-specific 证据

- **日期：** 2026-06-02
- **背景：**
  - `EXP-003` 的 `js_by_rank.png` 曾显示 adv 在中高 rank 似乎存在 JS 反弹，但该实验设置了 `js=0.02,mse=0.08` 早停条件，高 rank 的样本集合不完整。
  - `EXP-004` 使用 `--rank_growth_full_sweep` 禁用早停后，clean/adv 每个 rank 都有完整 `64` 个样本。
- **考虑过的选项：**
  - 选项 A：基于 `EXP-003` 继续推进“adv 中高 rank JS 反弹”作为新 idea。
  - 选项 B：以 `EXP-004` full-sweep 结果为准，暂不把该现象作为核心证据。
  - 选项 C：立即扩大到 `512` 样本继续验证同一现象。
- **最终选择：**
  - 选择 B。
- **原因：**
  - `EXP-004` 中 clean/adv 的 aggregate JS 均整体随 rank 增大而下降，adv 没有稳定反弹。
  - 样本级局部回升在 clean/adv 中都很常见，paired `adv-clean` late rebound 均值为负，不能支持 adversarial-specific 解释。
  - `EXP-003` 的高 rank 统计受早停选择偏差影响，不适合作为趋势诊断证据。
- **影响：**
  - 后续寻找 adversarial-specific rank-growth 信号时，应避免只看早停后的 aggregate JS 曲线。
  - 更值得考虑 paired sample-level 指标、分类边界相关指标，或其他能直接刻画 adversarial perturbation 的 proxy。
- **相关 idea：**
  - `IDEA-001`
- **相关实验：**
  - `EXP-003`
  - `EXP-004`

### DEC-004：暂不把高频占比单独作为 rank-growth 停止准则

- **日期：** 2026-06-02
- **背景：**
  - `IDEA-002` 假设 rank 增大后新增恢复成分 `a_r = \hat{x}_r - \hat{x}_{r-1}` 的高频占比会上升，并可能作为停止 rank growth 的信号。
  - `EXP-005` 在 `thubenchmark fold0 seed42 sample_num=64` 上用 full-sweep 记录了 clean/adv 每个相邻 rank pair 的高频能量占比和重构收益。
- **考虑过的选项：**
  - 选项 A：直接把 `incremental_high_freq_ratio` 单独做成 rank-growth 停止准则。
  - 选项 B：把高频占比保留为诊断特征，后续和 absolute gain、分类稳定性或 paired 指标联合分析。
  - 选项 C：放弃高频占比方向。
- **最终选择：**
  - 选择 B。
- **原因：**
  - `EXP-005` 显示 clean/adv 的高频占比都随 rank 增大上升，从约 `0.083` 上升到约 `0.224`，说明该指标能刻画 rank 增长过程中的频谱变化。
  - 但 clean 与 adversarial 曲线几乎重合，paired `adv-clean` 高频占比均值差都接近 `0`，不能证明它是 adversarial-specific 扰动恢复信号。
  - 当前探索性 `hf_stop_candidate` 阈值没有触发，且相对 MSE gain 在末端仍约 `0.27`，说明现有停止定义不成熟。
  - `EXP-006` 将 `10->15` 拆成 `10->11/11->12/12->13/13->14/14->15` 后，没有复现连续正峰值；五个细粒度 pair 的均值平均约为 `-0.002247`。
  - 在 `EXP-005` 追加 bootstrap 和 true-label margin 联合分析后，所有 `delta_hf_ratio_mean` 的 95% CI 都跨 `0`；`10->15` 的区间为 `[-0.001262, 0.004299]`，仍不能视为稳定正效应。
  - 高频占比差与分类边界差的相关性较弱，`hf_up_top1_bad_rate` 基本为 `0`，说明当前指标没有直接指向 rank 增长导致 adversarial top1 恶化。
- **影响：**
  - 后续不把高频占比单独写入默认 rank-growth 停止逻辑。
  - 若继续推进，应改为联合阈值或诊断分析，例如 `incremental_high_freq_ratio` 配合 absolute reconstruction gain、`incremental_l2`、分类稳定性和 robust acc。
  - `EXP-005` 的 `10->15` 局部正峰值暂不作为优先研究方向；bootstrap 置信区间和 margin 联合分析已经补充，但没有支持它是稳定结构。
  - 需要做 `20/25/30/35 Hz` cutoff 敏感性分析，确认该趋势不是单一阈值 artifact。
- **相关 idea：**
  - `IDEA-002`
- **相关实验：**
  - `EXP-005`
  - `EXP-006`

### DEC-005：IDEA-003 后续优先验证 MSE-centric 简化选 rank 规则

- **日期：** 2026-06-03
- **背景：**
  - `EXP-007` 用离线 Optuna 在 `n64` full-sweep logits 上得到 `score` selector 作为当前全局最佳候选。
  - `EXP-008` 对 `JS-only`、`MSE-only`、`Margin-only`、两两组合和 `EXP-007` full score 做 ablation。
- **考虑过的选项：**
  - 选项 A：直接把 `EXP-007` full score 写入在线 rank-growth 早停逻辑。
  - 选项 B：优先验证 `mse_only` 或 `js_mse` 这类 MSE-centric 简化规则。
  - 选项 C：继续增加 JS/margin 相关特征并扩大搜索空间。
- **最终选择：**
  - 选择 B。
- **原因：**
  - `EXP-008` 中所有 selector 的 holdout adversarial selected accuracy 都为 `0.90625`，accuracy 主指标没有区分出复杂 full score 的优势。
  - `EXP-007` full score 的 objective 最高，但相对 `mse_only`、`js_mse`、`mse_margin` 的差距只有约 `1e-4`，主要来自更低平均 rank，而不是更高准确率。
  - `mse_only` 和 `js_mse` 更容易解释、实现和复现；`mse_margin` 的 best margin threshold 为负值，说明 margin 条件在本次实验中基本被放松。
- **影响：**
  - 暂不修改 `purify.py` 和当前 baseline 早停逻辑。
  - 后续若做在线早停实现，应优先从 `mse_only` 或 `js_mse` 开始，并先在更大样本或跨 seed/fold 离线 replay 中确认。
  - `EXP-007` full score 保留为离线上界式对照，不作为默认实现目标。
- **相关 idea：**
  - `IDEA-003`
- **相关实验：**
  - `EXP-007`
  - `EXP-008`

### DEC-006：n512 后优先复验 threshold 与 js_mse rank selector

- **日期：** 2026-06-03
- **背景：**
  - `EXP-009` 生成了 `thubenchmark fold0 seed42 sample_num=512` 的 `r5-40 step5` full-sweep 轨迹。
  - `EXP-010` 在该轨迹上重新运行 Optuna rank-selection，并纳入 `EXP-007` full score 迁移对照。
- **考虑过的选项：**
  - 选项 A：继续把 `EXP-007` full score 作为在线早停实现目标。
  - 选项 B：按 `DEC-005` 的保守判断，优先实现 `mse_only`。
  - 选项 C：把 `threshold` 和 `js_mse` 作为下一轮主候选，`mse_only` 与 `margin_only` 作为对照。
- **最终选择：**
  - 选择 C。
- **原因：**
  - `EXP-010` 中 `threshold` 的全量 adversarial accuracy 最高，为 `0.84765625`；`js_mse` 次高，为 `0.845703125`，且平均 rank 更低 `17.67`。
  - `mse_only` 的 MSE 更低，但全量 adversarial accuracy 为 `0.83984375`，低于 `threshold/js_mse`。
  - `EXP-007` full score 迁移到 `n512, r5-40 step5` 后全量 adversarial accuracy 只有 `0.828125`，不能继续作为稳健默认候选。
  - `margin_only` 的 holdout adversarial accuracy 最高，但大量样本选择 `rank40`，更像高 rank 保真对照，不是低成本 adaptive 早停规则。
- **影响：**
  - 暂不修改 baseline 行为，也不直接把任何 selector 写入 `purify.py` 默认路径。
  - 后续在线 rank-selection 实现应优先提供 `threshold` 和 `js_mse` 的显式配置开关。
  - 在实现在线版本前，应优先补跨 seed/fold 或不同 `eps` 的离线 replay，确认 `threshold/js_mse` 的优势不是单一 split 偶然结果。
- **相关 idea：**
  - `IDEA-003`
- **相关实验：**
  - `EXP-009`
  - `EXP-010`

### DEC-007：预测熵暂不作为主 rank-selection 信号

- **日期：** 2026-06-04
- **背景：**
  - `IDEA-004` 希望把预测熵作为 uncertainty 指标，用于补充 rank-growth 过程中的 sample-wise rank selection。
  - `EXP-011` 在 `EXP-009` 的 `n512, ranks=5..40 step5` full-sweep logits 上新增 entropy 统计，并用 Optuna 测试 `entropy_only`、熵组合阈值和 `score_entropy`。
- **考虑过的选项：**
  - 选项 A：把 `entropy_only` 或 `entropy_margin` 作为新的主 rank selector。
  - 选项 B：把 `js_mse_entropy` 或 `score_entropy` 作为 `EXP-010` 的 `threshold/js_mse` 替代方案。
  - 选项 C：保留 entropy 作为诊断和可选辅助特征，主候选仍优先复验 `threshold/js_mse`。
- **最终选择：**
  - 选择 C。
- **原因：**
  - `EXP-011` 中 `threshold` 的全量 adversarial accuracy 仍最高，为 `0.84765625`。
  - `js_mse_entropy` 的 tune objective 最高，但 holdout adversarial accuracy 为 `0.828125`，低于 `threshold/js_mse/mse_only`；全量 adversarial accuracy `0.845703125` 只与 `js_mse` 持平。
  - `entropy_only` 与 `entropy_margin` 的 holdout accuracy 较高，但平均 rank 分别达到 `38.13` 和 `35.24`；`entropy_only` 在 `1024` 个 clean/adv rows 中有 `948` 个选择 `rank40`，更像高 rank 保真对照。
  - `score_entropy` 降低了 selected entropy，但没有超过 `threshold` 的 robust acc，且平均 rank 高于 `js_mse`。
- **影响：**
  - 保留 `entropy` 输出字段和 Optuna 熵 selector，便于后续诊断与复验。
  - 暂不把预测熵写入默认在线 rank-growth 早停逻辑。
  - 后续若继续推进在线 rank selection，仍优先实现 `threshold` 与 `js_mse` 的显式配置开关。
- **相关 idea：**
  - `IDEA-004`
- **相关实验：**
  - `EXP-011`

### DEC-008：eps=0.1 下不直接沿用 threshold/js_mse 作为默认 rank selector

- **日期：** 2026-06-04
- **背景：**
  - `EXP-012` 将 `EXP-009/010/011` 的 rank-growth full-sweep、paired-delta/bootstrap 和 Optuna rank-selection pipeline 扩展到 `autoattack eps=0.1`。
  - 目标是确认 `threshold/js_mse`、entropy 相关 selector 和固定 rank baseline 在更强攻击下是否保持 `eps=0.03` 的结论。
- **考虑过的选项：**
  - 选项 A：继续把 `threshold/js_mse` 作为在线 rank-selection 的首要实现目标。
  - 选项 B：改以 `score`、`score_entropy` 或 `js_mse_entropy` 作为主候选。
  - 选项 C：暂不确定默认 selector，先保留离线 replay 结论并扩大 seed/fold/eps 复验。
- **最终选择：**
  - 选择 C。
- **原因：**
  - `EXP-012` 中全量 adversarial accuracy 最高的是 `score`（`0.798828`），`js_mse`、`js_mse_entropy` 与固定 `rank15` 均为 `0.796875`，`score_entropy` 为 `0.794922`，差距很小。
  - `threshold` 全量 adversarial accuracy 为 `0.787109`，没有复现 `EXP-010/011` 中相对更稳的表现。
  - 固定高 rank 虽然提升 clean acc 并降低 MSE/entropy，但 adversarial accuracy 从 `rank15` 的 `0.796875` 随 rank 增大降到 `rank40` 的 `0.712891`，说明高保真并不等价于稳健。
  - paired bootstrap 中 `25->30`、`30->35`、`35->40` 的高频占比差 95% CI 均为正，同时 true-label margin 均值为负，提示高 rank 后段可能引入不利于分类 margin 的高频增量。
- **影响：**
  - 暂不修改 `purify.py` 或任何 baseline 默认路径。
  - 在线 rank-selection 实现前，应优先做跨 seed/fold 和不同 `eps` 的离线 replay；实现时必须通过显式配置开关暴露 selector。
  - `entropy` 可继续作为诊断/辅助特征保留，但 `entropy_only` 在 `eps=0.1` 下不适合作为主 selector。
- **相关 idea：**
  - `IDEA-003`
  - `IDEA-004`
- **相关实验：**
  - `EXP-012`

### DEC-009：js_mse 在线早停作为省计算候选而非默认策略

- **日期：** 2026-06-04
- **背景：**
  - `EXP-013` 使用 `EXP-012` 的 `js_mse` 最优阈值做真实在线早停，目标是检查 adaptive rank 是否能降低计算时间，而不是单纯追求最高 robust accuracy。
  - 该实验在 `thubenchmark fold0 seed42 eps=0.1 sample_num=512` 上完成，墙钟时间为 `1:09:11`，相比 `EXP-012` full-sweep 约 `2:02:32` 明显缩短。
- **考虑过的选项：**
  - 选项 A：把当前 `js_mse` 在线早停作为默认 rank-growth 策略。
  - 选项 B：把它作为显式配置开关下的省计算候选。
  - 选项 C：放弃在线 adaptive rank，继续只使用 full-sweep 离线分析。
- **最终选择：**
  - 选择 B。
- **原因：**
  - `EXP-013` 的平均 selected rank 约 `15.81`，平均每个样本评估 rank 数约 `4.16`，说明它确实能减少 rank block 计算量。
  - 但 `EXP-013` 的 adversarial accuracy 为 `0.78125`，低于 `EXP-012` 中离线 `score/js_mse/js_mse_entropy` 和固定 `rank15` 的约 `0.796875~0.798828` 区间。
  - 当前在线早停更适合 speed/compute-oriented 场景，不能作为 accuracy-oriented 默认策略。
- **影响：**
  - 暂不修改 `purify.py` 或 baseline 默认路径。
  - 后续若在线化 rank selector，应通过显式配置开关区分“省计算早停”和“精度优先选 rank”。
  - 继续推进前应优先做跨 seed/fold/eps 的复验，并比较更保守的在线停止语义或 `score` 类规则。
- **相关 idea：**
  - `IDEA-003`
- **相关实验：**
  - `EXP-013`

### DEC-010：PTR_3d_rank_soft_mask 暂不作为主 rank-selection 路线

- **日期：** 2026-06-05
- **背景：**
  - `IDEA-005` 实现了 `PTR_3d_rank_soft_mask`，将 hard prefix active rank 放松为可微 soft-prefix mask。
  - `EXP-014` 在 `thubenchmark fold0 seed42 autoattack eps=0.03 sample_num=512` 上完成 `rank_soft_mask_weight=0.0/0.001/0.003/0.01` sweep。
- **考虑过的选项：**
  - 选项 A：把 soft-mask 作为后续主要动态 rank 架构，继续调 `rank_soft_mask_weight`。
  - 选项 B：保留 soft-mask 作为可微 rank gate 原型，但不把 `MSE + rank cost` 版本作为主候选。
  - 选项 C：完全放弃 soft-mask 代码路径，回到 hard rank-growth。
- **最终选择：**
  - 选择 B。
- **原因：**
  - `EXP-014` 中 rank penalty 可控，mean effective rank 随 lambda 从 `19.536` 降到 `17.715`，说明 soft gate 工程路径成立。
  - 但 best soft-mask adversarial accuracy 为 `lambda=0.01` 的 `0.837891`，低于 `EXP-010/011` 的 `threshold=0.847656` 和 `js_mse=0.845703`。
  - effective rank 的样本级分布较窄，且 clean/adv 均值几乎重合；rank 与 MSE 强相关，但与分类正确性相关很弱，说明当前目标主要学习重构难度而不是鲁棒净化所需 rank。
  - 单纯继续 sweep 线性 rank penalty 不太可能产生论文级创新；需要更有结构的目标或机制，才能支撑“样本级动态 rank 分配”的研究主张。
- **影响：**
  - 保留 `PTR_3d_rank_soft_mask`、`rank_soft_mask` analysis mode 和 EXP-014 结果，作为后续可微 gate 改造的基线。
  - 当前 accuracy-oriented 主线仍优先围绕 hard rank-growth selector，尤其是 `threshold/js_mse` 及其在线化、跨 seed/fold/eps 复验。
  - 若继续 soft-mask，应优先改变优化目标或动态机制，而不是只调 `rank_soft_mask_weight`。
- **相关 idea：**
  - `IDEA-005`
- **相关实验：**
  - `EXP-014`
