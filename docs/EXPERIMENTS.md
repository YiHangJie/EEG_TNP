# EXPERIMENTS.md

本文件用于记录计划中、运行中、已完成或失败的实验。内容应包含命令、配置、指标、观察、问题、结论和下一步计划。

如果实验没有实际运行，**Results 必须写为 `Pending`**。不要编造实验结果、指标、日志或结论。

## 编号规则

- 使用连续编号：`EXP-001`、`EXP-002`、`EXP-003`，以此类推。
- 尽可能把每个实验关联到对应的 `IDEA-xxx`。
- 命令和配置应足够精确，方便后续复现。

## 实验通用模板

### EXP-XXX：标题

- **日期：** YYYY-MM-DD
- **状态：** 待运行（Pending） / 运行中 / 已完成 / 失败
- **相关 idea：** `IDEA-XXX` 或 `None`
- **目的：**
  - 这个实验要测试什么。
- **代码版本 / 分支：**
  - Git 分支、commit hash、patch 名称或 working tree 说明。
- **配置文件：**
  - 配置文件路径，或 `Pending`。
- **命令：**
  ```bash
  # Pending
  ```
- **关键设置：**
  - 数据集/划分：
  - 随机种子：
  - 模型：
  - 训练/评估设置：
  - 输出目录：
- **指标：**
  - 主指标：
  - 次指标：
- **结果：**
  - Pending
- **观察：**
  - Pending
- **问题：**
  - Pending
- **结论：**
  - Pending
- **下一步：**
  - Pending

### EXP-001：PTR rank growth 与固定 rank 净化对比

- **日期：** 2026-06-01 至 2026-06-02
- **状态：** 已完成
- **相关 idea：** `IDEA-001`
- **目的：**
  - 验证 `PTR_3d_rank_growth` 是否能在平均使用更低 rank 的情况下，达到或超过固定 rank PTR3d 的净化效果。
  - 比较默认 JS 阈值、严格 JS 阈值和 MSE 保真门槛对动态 rank 选择、robust acc、standard acc 与重建 MSE 的影响。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`565e290`
  - working tree：包含 `IDEA-001` 相关未提交实现；本实验新增 `rank_growth_max_mse_to_input` 作为非停止稳定性定义的保真筛选门槛。
- **配置文件：**
  - 固定 rank：
    - `configs/thubenchmark/PTR3d_8_2048_rank5_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank10_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank15_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank20_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank25_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank30_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank35_3d_interpolate.yaml`
    - `configs/thubenchmark/PTR3d_8_2048_rank40_3d_interpolate.yaml`
  - 动态 rank：
    - `configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - 动态 rank override：
    - `rank_growth_js_threshold=0.02`
    - `rank_growth_js_threshold=0.01`
    - `rank_growth_js_threshold=0.005`
    - `rank_growth_js_threshold=0.02, rank_growth_max_mse_to_input=0.06`
- **命令：**
  ```bash
  # 实际执行入口
  bash TN/rank_growth/run_exp001.sh

  # 固定 rank sweep 展开命令
  AT_STRATEGY=madry \
  CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
  AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
  MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
  OUTPUT_TAG=exp001_fixed \
  RANKS_CSV=5,10,15,20,25,30,35,40 \
  SAMPLE_NUM=512 \
  MAX_JOBS=2 \
  GPU_IDS=0 \
  bash purify_aug_stage4_rank_sweep.sh

  AT_STRATEGY=madry \
  CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
  AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
  MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
  OUTPUT_TAG=exp001_dynamic_js0p02 \
  SAMPLE_NUM=512 \
  RANK_GROWTH_JS_THRESHOLD=0.02 \
  bash TN/rank_growth/run_purify_rank_growth.sh

  AT_STRATEGY=madry \
  CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
  AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
  MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
  OUTPUT_TAG=exp001_dynamic_js0p01 \
  SAMPLE_NUM=512 \
  RANK_GROWTH_JS_THRESHOLD=0.01 \
  bash TN/rank_growth/run_purify_rank_growth.sh

  AT_STRATEGY=madry \
  CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
  AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
  MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
  OUTPUT_TAG=exp001_dynamic_js0p005 \
  SAMPLE_NUM=512 \
  RANK_GROWTH_JS_THRESHOLD=0.005 \
  bash TN/rank_growth/run_purify_rank_growth.sh

  AT_STRATEGY=madry \
  CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
  AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
  MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
  OUTPUT_TAG=exp001_dynamic_js0p02_mse0p06 \
  SAMPLE_NUM=512 \
  RANK_GROWTH_JS_THRESHOLD=0.02 \
  RANK_GROWTH_MAX_MSE_TO_INPUT=0.06 \
  bash TN/rank_growth/run_purify_rank_growth.sh
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 样本数：`512`
  - 输出目录：`log_purify/`，`purified_data/attacked/`
  - 运行记录：总控 PID `1898031`，总控日志 `logs/exp001_20260601_235419.log`
- **指标：**
  - 主指标：`Purified adversarial data accuracy`，`Purified clean data accuracy`
  - 次指标：`Mean mse of purified adversarial data`，`Mean mse of purified clean data`，动态 rank 的 `Dynamic rank selected` 分布、`Dynamic rank history`
- **结果：**
  - 净化前基线：`Adversarial data accuracy=0.7734375`，`Clean data accuracy=0.943359375`。
  - 固定 rank 结果：

    | 设置 | Purified adv acc | Purified clean acc | Adv MSE | Clean MSE | compression rate |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | `rank5` | 0.666015625 | 0.664062500 | 0.440792204 | 0.441135978 | 0.032197917 |
    | `rank10` | 0.796875000 | 0.812500000 | 0.215098157 | 0.215909958 | 0.048447917 |
    | `rank15` | 0.820312500 | 0.873046875 | 0.132929122 | 0.132802362 | 0.074072917 |
    | `rank20` | 0.835937500 | 0.898437500 | 0.086366628 | 0.086129195 | 0.109072917 |
    | `rank25` | 0.845703125 | 0.906250000 | 0.058501068 | 0.058471672 | 0.153447917 |
    | `rank30` | 0.833984375 | 0.927734375 | 0.040879480 | 0.040777232 | 0.207197917 |
    | `rank35` | 0.835937500 | 0.931640625 | 0.028912855 | 0.028949255 | 0.270322917 |
    | `rank40` | 0.832031250 | 0.933593750 | 0.020668697 | 0.020651383 | 0.342822917 |

  - 动态 rank 结果：

    | 设置 | Purified adv acc | Purified clean acc | Adv MSE | Clean MSE | mean rank all / adv / clean | rank 分布 |
    | --- | ---: | ---: | ---: | ---: | ---: | --- |
    | `js=0.02` | 0.849609375 | 0.910156250 | 0.284306605 | 0.286512990 | 12.104 / 12.100 / 12.109 | `{5:422, 10:238, 15:140, 20:88, 25:65, 30:29, 35:26, 40:16}` |
    | `js=0.01` | 0.849609375 | 0.908203125 | 0.260270964 | 0.261061604 | 13.921 / 14.004 / 13.838 | `{5:371, 10:227, 15:130, 20:99, 25:64, 30:51, 35:42, 40:40}` |
    | `js=0.005` | 0.851562500 | 0.914062500 | 0.239967773 | 0.235349715 | 15.972 / 15.996 / 15.947 | `{5:331, 10:195, 15:118, 20:112, 25:81, 30:51, 35:51, 40:85}` |
    | `js=0.02,mse=0.06` | 0.849609375 | 0.910156250 | 0.284306605 | 0.286512990 | 12.104 / 12.100 / 12.109 | 与 `js=0.02` 相同；本次未实际启用 MSE gate |

  - 动态 rank 平均 compression rate：`js=0.02` 为 `0.071858561`，`js=0.01` 为 `0.086743205`，`js=0.005` 为 `0.104476969`。
  - 结果来源：`logs/exp001_20260601_235419.log` 与 2026-06-02 00:50 至 07:30 生成的 12 个 `log_purify/purify_thubenchmark_eegnet_madry_consistancy_rank25-30_n512_eps0p03_autoattack_eps0.03_42_*` 日志。
- **观察：**
  - 固定 rank 中，robust acc 在 `rank25` 达到最高 `0.845703125`；继续升到 `rank30/35/40` 后 clean acc 和 MSE 继续改善，但 robust acc 下降到 `0.833984375/0.835937500/0.832031250`。
  - 动态 `js=0.005` 的 robust acc 最高，为 `0.851562500`，比固定 rank 最好值 `rank25=0.845703125` 高 `0.005859375`，但 clean acc `0.914062500` 低于固定 `rank30/35/40`。
  - 动态 rank 使用明显低于固定高 rank 的平均参数量：`js=0.005` 平均 rank 约 `15.97`，平均 compression rate `0.104476969`，接近固定 `rank20` 的 `0.109072917`，但 robust acc 高于固定 `rank20` 和 `rank25`。
  - JS 阈值越严格，平均 rank 越高，MSE 越低，robust acc 小幅提升：`js=0.02 -> 0.01 -> 0.005` 的平均 rank 为 `12.10 -> 13.92 -> 15.97`，Adv MSE 为 `0.2843 -> 0.2603 -> 0.2400`，Purified adv acc 为 `0.8496 -> 0.8496 -> 0.8516`。
  - 动态 rank 在 clean 与 adversarial 样本上的平均 rank 非常接近，说明当前选择准则没有明显区分 clean/adversarial 难度。
- **问题：**
  - MSE 门槛 `0.06` 本次没有实际生效：临时 YAML 中写入了 `rank_growth_max_mse_to_input: 0.06`，但日志内 `Dynamic rank history` 全部显示 `rank_growth_max_mse_to_input: None`，且 `js=0.02,mse=0.06` 的指标和 rank 分布与 `js=0.02` 完全一致。
  - 代码检查显示 `TN/utils.py` 的 `get_TN_args()` 在 `PTR_3d_rank_growth` 分支没有把 `rank_growth_max_mse_to_input` 传给 `PTR_3d_rank_growth` 构造函数，因此本次 MSE gate 结果不可解释为有效 sweep。
  - 动态 rank 的 MSE 明显高于固定 `rank20/25`，说明当前 JS/top1 稳定准则偏向分类稳定性而不是重建保真；如果后续目标包含保真，需要修复并重新评估 MSE gate。
  - 本实验只覆盖 `thubenchmark fold0 seed42 sample_num=512 eps=0.03`，还不能证明跨 fold、跨 seed 或更大样本数的稳定性。
- **结论：**
  - `PTR_3d_rank_growth` 的核心假设得到部分支持：在平均 rank 低于固定 `rank25`、接近固定 `rank20` 的情况下，动态 `js=0.005` 达到本实验最高 robust acc。
  - rank growth 暂时不应替代固定高 rank 作为 clean accuracy 最优方案；固定 `rank30/35/40` 的 clean acc 更高、MSE 更低。
  - 当前更合理的候选设置是 `rank_growth_js_threshold=0.005`，作为 robust-oriented 净化配置继续评估；MSE gate 需要先修复参数传递后再运行。
- **下一步：**
  - 修复 `rank_growth_max_mse_to_input` 从 YAML 到 `PTR_3d_rank_growth` 构造函数的传递，并用小样本 dry-run 检查日志中不再显示 `None`。
  - 重新运行 MSE gate sweep，建议至少覆盖 `0.04/0.06/0.08/0.10`，并保留 `js=0.005` 与 `js=0.01` 两个候选阈值。
  - 在修复后追加跨 seed 或更多 fold 的验证，优先比较固定 `rank20/25/30` 与动态 `js=0.005`。

### EXP-002：PTR rank growth JS=0.02 + MSE gate 0.08

- **日期：** 2026-06-02
- **状态：** 已完成
- **相关 idea：** `IDEA-001`
- **目的：**
  - 修复 `rank_growth_max_mse_to_input` 未传入 `PTR_3d_rank_growth` 的问题后，验证 `JS=0.02` 与 `MSE<=0.08` 联合停止条件的效果。
  - 与 `EXP-001` 中未启用 MSE gate 的 `js=0.02` 结果对比，观察 MSE gate 是否提高平均 rank、降低 MSE，并影响 robust/clean acc。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`565e290`
  - working tree：包含 `IDEA-001` 相关未提交实现；本实验新增修复：`TN/utils.py` 的 `get_TN_args()` 会把 `rank_growth_max_mse_to_input` 传入 `PTR_3d_rank_growth`。
- **配置文件：**
  - 基础配置：`configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - override：`rank_growth_js_threshold=0.02`，`rank_growth_max_mse_to_input=0.08`
- **命令：**
  ```bash
  AT_STRATEGY=madry \
  CHECKPOINT_PATH=checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
  AD_DATA_PATH=ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
  MODEL_TAG=consistancy_rank25-30_n512_eps0p03 \
  OUTPUT_TAG=exp002_dynamic_js0p02_mse0p08 \
  SAMPLE_NUM=512 \
  RANK_GROWTH_JS_THRESHOLD=0.02 \
  RANK_GROWTH_MAX_MSE_TO_INPUT=0.08 \
  bash TN/rank_growth/run_purify_rank_growth.sh
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 样本数：`512`
  - 输出目录：`log_purify/`，`purified_data/attacked/`
  - 运行记录：`nohup` 后台 PID `2052287`，总控日志 `logs/exp002_dynamic_js0p02_mse0p08_20260602_103754.log`，净化日志 `log_purify/purify_thubenchmark_eegnet_madry_consistancy_rank25-30_n512_eps0p03_autoattack_eps0.03_42_PTR3d_rank_growth_8_2048_r5-40_3d_interpolate_override_r-sdefault-js0.02-mse0.08.yaml_20260602_103757.log`
  - 备注：`log_purify/..._20260602_102609.log` 是前台试跑被中断后的部分日志，不作为本实验正式结果。
- **指标：**
  - 主指标：`Purified adversarial data accuracy`，`Purified clean data accuracy`
  - 次指标：`Mean mse of purified adversarial data`，`Mean mse of purified clean data`，`Dynamic rank selected` 分布，`rejected_by_mse_gate` 触发次数
- **结果：**
  - 净化前基线：`Adversarial data accuracy=0.7734375`，`Clean data accuracy=0.943359375`。
  - 主要结果：

    | 设置 | Purified adv acc | Purified clean acc | Adv MSE | Clean MSE | mean rank all / adv / clean | mean compression rate |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | `js=0.02,mse=0.08` | 0.8515625 | 0.916015625 | 0.064028169 | 0.064172617 | 27.344 / 27.363 / 27.324 | 0.183266642 |

  - rank 分布：`{15:25, 20:174, 25:344, 30:291, 35:157, 40:33}`。
  - adversarial rank 分布：`{15:12, 20:91, 25:169, 30:142, 35:79, 40:19}`。
  - clean rank 分布：`{15:13, 20:83, 25:175, 30:149, 35:78, 40:14}`。
  - MSE gate 触发：`rejected_by_mse_gate=True` 出现 `5976` 次；总 rank block 数 `6591`。
  - 单样本 MSE 范围：all `0.013950835` 到 `0.079987846`，adversarial `0.017988356` 到 `0.079914838`，clean `0.013950835` 到 `0.079987846`。
  - 输出文件：
    - `purified_data/attacked/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_autoattack_eps0.03_seed42_fold0_PTR3d_rank_growth_8_2048_r5-40_3d_interpolate_override_r-sdefault-js0.02-mse0.08_n512_tagexp002_dynamic_js0p02_mse0p08_ad.pth`
    - `purified_data/attacked/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_autoattack_eps0.03_seed42_fold0_PTR3d_rank_growth_8_2048_r5-40_3d_interpolate_override_r-sdefault-js0.02-mse0.08_n512_tagexp002_dynamic_js0p02_mse0p08_clean.pth`
- **观察：**
  - 相比 `EXP-001` 的 `js=0.02` 无 MSE gate，本实验把平均 rank 从 `12.104` 提高到 `27.344`，Adv MSE 从 `0.284306605` 降到 `0.064028169`，Clean MSE 从 `0.286512990` 降到 `0.064172617`。
  - robust acc 从 `EXP-001/js=0.02` 的 `0.849609375` 提高到 `0.8515625`，clean acc 从 `0.91015625` 提高到 `0.916015625`。
  - 与 `EXP-001/js=0.005` 相比，本实验 robust acc 持平 `0.8515625`，clean acc 更高 `0.916015625` vs `0.9140625`，MSE 明显更低，但平均 rank 明显更高 `27.344` vs `15.972`。
  - 与固定 `rank25` 相比，本实验 robust acc 更高 `0.8515625` vs `0.845703125`，clean acc 更高 `0.916015625` vs `0.90625`，但 MSE 略高于固定 `rank25` 的约 `0.0585`。
  - MSE gate 的实际效果很强：大量低 rank 稳定点被拒绝，最终 rank 主要集中在 `25/30`，说明 `0.08` 门槛把动态策略推向固定 `rank25/30` 附近。
- **问题：**
  - `MSE<=0.08` 显著提高计算成本；本次单条动态实验从 2026-06-02 10:37:54 运行到 12:23:29。
  - 当前只在 `fold0 seed42 sample_num=512 eps=0.03` 上验证，仍需跨 seed/fold 复验。
  - 当前停止逻辑选择的是满足稳定时的上一档 rank；日志中会出现 selected row 自身 `fidelity_gate_pass=False` 的情况，因为 gate 实际检查的是“上一档 rank 是否可接受”，需要后续在日志字段命名上减少歧义。
- **结论：**
  - `rank_growth_max_mse_to_input=0.08` 修复后有效，并显著降低动态 rank 的重建 MSE。
  - `js=0.02,mse=0.08` 是目前更均衡的候选：robust acc 达到 `EXP-001` 最优水平，同时 clean acc 和 MSE 均明显优于无 MSE gate 的动态配置。
  - 代价是平均 rank/计算成本明显上升，适合继续作为保真约束动态净化方向评估，但不能直接视作低成本方案。
- **下一步：**
  - 继续 sweep `MSE=0.06/0.08/0.10`，优先和 `js=0.02`、`js=0.005` 组合比较。
  - 增加跨 seed 或 fold 验证，重点比较固定 `rank25/30`、`js=0.005`、`js=0.02,mse=0.08`。
  - 优化动态 rank 日志字段，区分 `current_rank_gate_pass` 与 `selected_previous_rank_gate_pass`，避免误读。

### EXP-003：用 rank growth 轨迹替代 TensorLy TR rank 预测分析

- **日期：** 2026-06-02
- **状态：** 已完成
- **相关 idea：** `IDEA-001`
- **目的：**
  - 将 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py` 中原本基于 TensorLy 普通 TR 的 rank 预测分析，扩展为可直接运行 `PTR_3d_rank_growth` 的分析模式。
  - 记录每个样本在 rank growth 过程中的逐档 MSE、相邻 rank JS、top1 稳定性、MSE gate 拒绝情况和最终 selected rank，用于后续调 `rank_growth_js_threshold` 与 `rank_growth_max_mse_to_input`。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`565e290`
  - working tree：包含 `IDEA-001` 相关未提交实现；本实验新增 `analyze_tr_rank_predictions.py --analysis_mode rank_growth`。
- **配置文件：**
  - `configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - 计划 override：`rank_growth_js_threshold=0.02`，`rank_growth_max_mse_to_input=0.08`
- **命令：**
  ```bash
  # 建议先用小样本做调参观察；完整 512 样本可把 SAMPLE_NUM 改为 512。
  SAMPLE_NUM=64
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n${SAMPLE_NUM}"
  LOG_FILE="logs/exp003_rank_growth_analysis_js0p02_mse0p08_n${SAMPLE_NUM}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  setsid nohup conda run -n torch python tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --dataset thubenchmark \
    --model eegnet \
    --no_ea \
    --eps 0.03 \
    --fold 0 \
    --attack autoattack \
    --at_strategy madry \
    --consistency_version consistancy \
    --consistency_tag consistancy_rank25-30_n512_eps0p03 \
    --adv_model_tag consistancy_rank25-30_n512_eps0p03 \
    --config PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml \
    --checkpoint_path checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
    --ad_data_path ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
    --sample_num "${SAMPLE_NUM}" \
    --rank_growth_js_threshold 0.02 \
    --rank_growth_max_mse_to_input 0.08 \
    --output_dir "${OUT_DIR}" \
    --no_visualize \
    > "${LOG_FILE}" 2>&1 &
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 样本数：计划先用 `64` 做阈值观察，必要时扩展到 `512`
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64/`
  - 运行记录：`nohup` 后台父进程 PID `2097063`，Python 子进程 PID `2097083`，日志 `logs/exp003_rank_growth_analysis_js0p02_mse0p08_n64_20260602_134226.log`
- **指标：**
  - 主指标：`rank_growth_summary.csv` 中各 rank 的 `selected_count`、`mse_mean`、`js_mean`、`top1_change_rate`、`rejected_by_mse_gate_count`
  - 次指标：`rank_growth_history.csv` 中 clean/adv 单样本动态轨迹和最终 `selected` rank 分布
- **结果：**
  - 进程已结束，结果文件于 2026-06-02 13:54:47 写入。
  - 输出文件：
    - `tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64/rank_growth_history.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64/rank_growth_summary.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64/rank_growth_predictions.pt`
    - `tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64/meta.json`
    - `tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64/plots/`
  - plot-only 生成命令：
    ```bash
    conda run -n torch python tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
      --analysis_mode rank_growth \
      --plot_only \
      --output_dir tensor_ring_rank_analysis/results_rank_growth_exp003_js0p02_mse0p08_n64 \
      --plot_format png \
      --plot_dpi 180
    ```
  - plot 文件：
    - `selected_rank_distribution.png`
    - `mse_by_rank.png`
    - `js_by_rank.png`
    - `mse_gate_rejections_by_rank.png`
    - `selected_mse_boxplot.png`
    - `clean_mse_trajectory_heatmap.png`
    - `adv_mse_trajectory_heatmap.png`
    - `clean_js_trajectory_heatmap.png`
    - `adv_js_trajectory_heatmap.png`
  - history 行数：`806`。
  - selected rank 分布：
    - all：`{15:3, 20:26, 25:44, 30:40, 35:11, 40:4}`，mean rank `26.640625`
    - clean：`{15:1, 20:14, 25:22, 30:21, 35:5, 40:1}`，mean rank `26.40625`
    - adv：`{15:2, 20:12, 25:22, 30:19, 35:6, 40:3}`，mean rank `26.875`
  - selected MSE：
    - clean mean `0.065974544`，min `0.032870781`，max `0.079923972`
    - adv mean `0.064440836`，min `0.029532563`，max `0.078899905`
  - MSE gate 拒绝次数：clean `187`，adv `189`。
- **观察：**
  - `rank_growth_max_mse_to_input=0.08` 在分析脚本中实际生效，最终 selected 样本的 MSE 全部低于 `0.08`。
  - selected rank 主要集中在 `25/30`，与 `EXP-002` 的完整净化实验趋势一致；小样本 `n=64` 的 mean rank `26.64` 接近 `EXP-002` 的 `27.34`。
  - clean 与 adversarial 的 rank 分布仍非常接近，当前 `JS+MSE` 准则没有明显把 adversarial 样本推到更高 rank。
- **问题：**
  - 本实验只跑了 `sample_num=64`，用于调参观察，不直接等价于 `EXP-002` 的 `512` 样本统计。
  - 当前脚本记录的是 rank-growth 轨迹和分类稳定性，不直接输出净化后 accuracy；accuracy 仍需通过 `purify.py` 或读取 selected 重建样本另做评估。
- **结论：**
  - `analyze_tr_rank_predictions.py --analysis_mode rank_growth` 可以用于后续 JS/MSE 阈值调参。
  - `js=0.02,mse=0.08` 在小样本轨迹分析中复现了 EXP-002 的核心现象：MSE gate 明显拒绝低 rank 稳定点，并把最终 rank 推向 `25/30`。
- **下一步：**
  - 继续用同一脚本 sweep `MSE=0.06/0.10` 与 `JS=0.005/0.01/0.02`，优先观察 selected rank 分布、selected MSE 和 MSE gate 拒绝次数。
