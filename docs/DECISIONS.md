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
