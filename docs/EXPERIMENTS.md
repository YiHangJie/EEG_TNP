# EXPERIMENTS.md

本文件用于记录计划中、运行中、已完成或失败的实验。内容应包含命令、配置、指标、观察、问题、结论和下一步计划。

如果实验没有实际运行，**Results 必须写为 `Pending`**。不要编造实验结果、指标、日志或结论。

## AI 阅读规则

AI 处理本文件时，默认不要全文阅读。除非用户明确要求完整审查全部实验记录，否则应按下面顺序读取：

- 先读文件开头的规则、闭环 checklist、编号规则和实验通用模板。
- 再读“实验索引表”，根据 `EXP-XXX`、`IDEA-XXX`、关键词、run id、脚本名、方法名或数据集定位相关条目。
- 只展开阅读与当前任务直接相关的实验条目，以及它们引用的上下游实验。
- 如果任务涉及研究结论、方向取舍或论文叙事，再同步阅读相关 `DEC-XXX` 和 `docs/方法进展梳理.md` 片段。
- 如果无法确定相关实验，先用 `rg` 搜索关键词，不要直接阅读该文档的全文。

## 实验索引表

本表用于帮助 AI 和研究者快速定位实验。新增或完成实验时，应同步更新对应行的状态、相关 idea 和关键词；详细命令、指标和结论仍以实验正文为准。

| EXP | 状态 | 相关 idea | 主题 / 关键词 |
| --- | --- | --- | --- |
| `EXP-001` | 已完成 | `IDEA-001` | PTR rank growth；固定 rank；JS 阈值；MSE gate 初始问题 |
| `EXP-002` | 已完成 | `IDEA-001` | JS=0.02；MSE gate 0.08；保真约束 |
| `EXP-003` | 已完成 | `IDEA-001` | rank growth 轨迹；TensorLy TR 替代分析 |
| `EXP-004` | 已完成 | `IDEA-001` | full-sweep；JS/MSE 诊断；早停偏差 |
| `EXP-005` | 已完成 | `IDEA-002` | 高频能量占比；新增恢复成分；频谱诊断 |
| `EXP-006` | 已完成 | `IDEA-002` | rank 5-20；细粒度高频增量 |
| `EXP-007` | 已完成 | `IDEA-003` | Optuna；离线 rank-selection；JS/MSE/margin |
| `EXP-008` | 已完成 | `IDEA-003` | 单指标 ablation；两两组合 selector |
| `EXP-009` | 已完成 | `IDEA-003` | n512；r5-40 step5；full-sweep 轨迹 |
| `EXP-010` | 已完成 | `IDEA-003` | n512；Optuna；threshold；js_mse |
| `EXP-011` | 已完成 | `IDEA-004` | entropy；rank-selection；Optuna ablation |
| `EXP-012` | 已完成 | `IDEA-002`、`IDEA-003`、`IDEA-004` | eps=0.1；rank-growth pipeline；跨 eps 复验 |
| `EXP-013` | 已完成 | `IDEA-003` | eps=0.1；js_mse 在线早停；计算开销 |
| `EXP-014` | 已完成 | `IDEA-005` | PTR_3d_rank_soft_mask；rank penalty sweep |
| `EXP-015` | Stopped | `IDEA-006` | JS_MSE；跨 dataset/model/seed/fold/eps；完整复验 |
| `EXP-016` | 已完成 | None | 对抗训练；逐 epoch PGD；工程 smoke |
| `EXP-017` | Completed | `IDEA-007` | RPCF 首轮；legacy seed protocol |
| `EXP-018` | 已完成 | `IDEA-007` | RPCF；consistancy；普通 EEG_TNP+AT；公平对比 |
| `EXP-019` | 已完成 | `IDEA-007` | 五方法；seed43；公平复验 |
| `EXP-020` | 已完成 | `IDEA-008` | eps=0.05；五方法；三 seed |
| `EXP-021` | 已完成 | `IDEA-009` | RPCF_AT；eps0.03；三 seed |
| `EXP-022` | 已完成 | `IDEA-010` | RPCF_AT；eps0.05；三 seed |
| `EXP-023` | 结果 Pending | `IDEA-011` | BPDA+PGD-10；adaptive attack；smoke 已通过 |
| `EXP-024` | 已完成 | Pending | 其他 backbone；RPCF_AT；baseline 全流程 |
| `EXP-025` | Pending | Pending | RPCF_AT；敏感层降序累加微调；四 backbone 预算曲线 |

## 实验完成闭环 Checklist

每次实验从 `运行中` 变为 `已完成`、`失败` 或决定长期暂停时，更新本实验条目后，应按下面顺序检查研究闭环。没有触发的项目写 `不需要`，不要空着。

- `IDEAS.md`：如果实验改变了相关 idea 的可信度、状态或下一步，应更新对应 `IDEA-XXX` 的状态、相关实验、风险或备注。
- `EXPERIMENTS.md`：本条实验必须补齐实际命令、关键设置、输出路径、结果、观察、问题、结论和下一步；未实际运行的结果保持 `Pending`。
- `DECISIONS.md`：如果实验结果会影响后续研究方向、默认候选、是否放弃某路线、baseline 公平性或计算资源投入，应新增或更新一个 `DEC-XXX`。
- `方法进展梳理.md`：如果实验改变了论文叙事、阶段性证据链、主方法定位或下一步 presentation 口径，应同步更新。
- `CODEMAP.md`：如果实验引入了新的入口脚本、pipeline、输出目录、重要配置开关或安全扩展点，应同步更新。
- `PROMPTS.md`：如果本次形成了可复用的 AI 操作流程，应补充或修改提示词。

建议在每个实验条目的 `下一步` 中显式写出闭环状态，例如：

```text
- **闭环检查：**
  - `IDEAS.md`：已更新 / 不需要
  - `DECISIONS.md`：已新增 `DEC-XXX` / 不需要
  - `方法进展梳理.md`：已更新 / 不需要
  - `CODEMAP.md`：已更新 / 不需要
```

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
  - 本实验设置了 `rank_growth_js_threshold=0.02` 和 `rank_growth_max_mse_to_input=0.08`，因此属于真实早停策略评估，不适合作为“JS/MSE 随 rank 完整变化趋势”的诊断实验。高 rank 的 `evaluated_count` 会因为样本提前停止而变少，`js_by_rank.png` 后半段存在选择偏差。
- **结论：**
  - `analyze_tr_rank_predictions.py --analysis_mode rank_growth` 可以用于后续 JS/MSE 阈值调参。
  - `js=0.02,mse=0.08` 在小样本轨迹分析中复现了 EXP-002 的核心现象：MSE gate 明显拒绝低 rank 稳定点，并把最终 rank 推向 `25/30`。
- **下一步：**
  - 使用 `--rank_growth_full_sweep` 重新设置完整 rank 诊断实验，关闭早停后再观察 JS/MSE 的 rank 趋势。
  - 继续用同一脚本 sweep `MSE=0.06/0.10` 与 `JS=0.005/0.01/0.02`，但这类 sweep 应解释为停止策略评估，而不是完整趋势诊断。

### EXP-004：rank growth full-sweep JS/MSE 诊断

- **日期：** 2026-06-02
- **状态：** 已完成
- **相关 idea：** `IDEA-001`
- **目的：**
  - 修正 `EXP-003` 的早停选择偏差，强制每个样本完整跑完 `rank_growth_ranks=[5,10,15,20,25,30,35,40]`。
  - 在不设置 JS/MSE 早停阈值的情况下，观察 clean/adv 的相邻 rank JS、MSE、top1 change 是否存在稳定差异，尤其验证 adv 是否在中高 rank 出现 JS 反弹。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：新增 `analyze_tr_rank_predictions.py --rank_growth_full_sweep`，该模式会禁用早停并完整评估每个 rank。
- **配置文件：**
  - `configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - full-sweep override：`rank_growth_js_threshold=-1.0`，`rank_growth_max_mse_to_input=None`
- **命令：**
  ```bash
  SAMPLE_NUM=64
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp004_full_sweep_n${SAMPLE_NUM}"
  LOG_FILE="logs/exp004_rank_growth_full_sweep_n${SAMPLE_NUM}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  setsid nohup conda run -n torch python tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --rank_growth_full_sweep \
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
    --output_dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 样本数：计划先用 `64`，与 `EXP-003` 对齐
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp004_full_sweep_n64/`
  - 运行记录：`nohup` 后台父进程 PID `2123431`，Python 子进程 PID `2123440`，日志 `logs/exp004_rank_growth_full_sweep_n64_20260602_152924.log`
- **指标：**
  - 主指标：所有 rank 上 clean/adv 的 `js_mean`、`mse_mean`、`top1_change_rate`
  - 次指标：sample-level JS rebound rate、paired adv-clean rebound score
- **结果：**
  - 进程已结束，结果文件写入 `tensor_ring_rank_analysis/results_rank_growth_exp004_full_sweep_n64/`。
  - full-sweep 生效：`meta.json` 中 `rank_growth_full_sweep=True`，`rank_growth_js_threshold=-1.0`，`rank_growth_max_mse_to_input=None`。
  - 所有 rank 的 `evaluated_count` 均为 `64`，clean/adv 都没有高 rank 缺失。
  - aggregate JS 均值：

    | source | 5->10 | 10->15 | 15->20 | 20->25 | 25->30 | 30->35 | 35->40 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | clean | 0.137021 | 0.028420 | 0.015852 | 0.014313 | 0.004341 | 0.002103 | 0.002207 |
    | adv | 0.126372 | 0.023216 | 0.012391 | 0.012016 | 0.004616 | 0.004140 | 0.003158 |

  - aggregate MSE 均值随 rank 单调下降：
    - clean：`0.460168 -> 0.231484 -> 0.150901 -> 0.106160 -> 0.075010 -> 0.053228 -> 0.038338 -> 0.027821`
    - adv：`0.458448 -> 0.231631 -> 0.150967 -> 0.105780 -> 0.075135 -> 0.052973 -> 0.038295 -> 0.027943`
  - 样本级 late rebound 统计：
    - clean：late local increase `61/64`，mean late rebound `0.008495`
    - adv：late local increase `60/64`，mean late rebound `0.007536`
    - paired `adv-clean` late rebound 均值 `-0.000958`，adv 更大的样本数 `34/64`
- **观察：**
  - 与 `EXP-003` 的早停图不同，full-sweep 下 clean 和 adv 的 aggregate JS 都整体随 rank 增大而下降。
  - adv 没有表现出稳定的中高 rank JS 反弹；`adv` 的 aggregate JS 从 `5->10` 到 `35->40` 全程下降。
  - clean 在 `35->40` 有极小的 aggregate 回升 `0.000104`，量级很小，不构成主要趋势。
  - 样本级局部回升在 clean/adv 中都很常见，但不是 adversarial-specific signal。
- **问题：**
  - 本实验仍是 `sample_num=64`，结论主要针对当前 fold/seed/eps/checkpoint。
  - full-sweep 会把所有样本都跑到 rank40，因此其 selected rank 分布不再代表真实早停策略，只适合看轨迹形态。
- **结论：**
  - 早先基于 `EXP-003/js_by_rank.png` 得到的“adv 随 rank 增大 JS 先小后大”判断主要来自早停选择偏差，不应作为新 idea 的核心证据。
  - 在 full-sweep 诊断下，clean/adv 的 JS 和 MSE 都呈现由粗到细逐步稳定的趋势，当前数据不支持“adv 特异性 JS 反弹”假设。
- **下一步：**
  - 若继续寻找 adversarial-specific rank-growth 信号，应转向 paired sample-level 指标或分类边界相关指标，而不是 aggregate adjacent-rank JS。
  - 可在更大样本数或不同 seed/fold 上复查，但当前证据不支持优先投入该方向。

### EXP-005：rank growth 新增恢复成分的高频能量占比诊断

- **日期：** 2026-06-02
- **状态：** 已完成
- **相关 idea：** `IDEA-002`
- **目的：**
  - 验证 `IDEA-002` 的核心假设：随着 rank 从小到大增长，相邻 rank 新增恢复成分 `a_r = \hat{x}_r - \hat{x}_{r-1}` 的高频能量占比是否会在重构收益趋于饱和时升高。
  - 判断高频能量占比能否作为 rank-growth 自适应停止信号，补充或替代当前的 JS/top1/MSE gate。
  - 区分该信号是 clean 与 adversarial 共有的重构细化现象，还是对 adversarial 样本更敏感的扰动恢复信号。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：新增 `analyze_tr_rank_predictions.py` 的 EXP-005 频域增量指标实现；该实现只在 `--enable_incremental_frequency_metrics` 开启时记录相邻 rank 的新增恢复成分频域指标，不改变默认 rank-growth 分析输出。
- **配置文件：**
  - `configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - full-sweep override：`rank_growth_js_threshold=-1.0`，`rank_growth_max_mse_to_input=None`
- **命令：**
  ```bash
  SAMPLE_NUM=64
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_n${SAMPLE_NUM}"
  LOG_FILE="logs/exp005_rank_growth_hf_full_sweep_n${SAMPLE_NUM}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  setsid nohup conda run -n torch python tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --rank_growth_full_sweep \
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
    --enable_incremental_frequency_metrics \
    --high_freq_cutoff_hz 30.0 \
    --freq_energy_floor_hz 1.0 \
    --output_dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &
  ```
- **追加 rerun：rank 4-40 step 2 细粒度 full-sweep**
  - 状态：运行中
  - 启动时间：2026-06-02 20:53:00
  - 目的：在 EXP-005 原始设置基础上，把 rank 序列从 `[5,10,15,20,25,30,35,40]` 改为 `range(4,42,2)`，支持更细粒度的 JS、频域增量和 paired-delta 分析。
  - rank 序列：`[4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]`
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_r4-40_step2_n64/`
  - 日志：`logs/exp005_rank_growth_hf_full_sweep_r4-40_step2_n64_20260602_205300.log`
  - 后台父进程 PID：`2197186`
  - 实时日志设置：`conda run --no-capture-output` + `python -u` + `> log 2>&1`
  - 命令：

    ```bash
    setsid nohup conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
      --analysis_mode rank_growth \
      --rank_growth_full_sweep \
      --rank_growth_ranks 4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40 \
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
      --sample_num 64 \
      --enable_incremental_frequency_metrics \
      --high_freq_cutoff_hz 30.0 \
      --freq_energy_floor_hz 1.0 \
      --output_dir tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_r4-40_step2_n64 \
      > logs/exp005_rank_growth_hf_full_sweep_r4-40_step2_n64_20260602_205300.log 2>&1 &
    ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - rank 序列：`[5, 10, 15, 20, 25, 30, 35, 40]`
  - 样本数：先用 `64` 做 full-sweep 指标诊断；若信号清晰，再扩展到 `512`
  - 频域定义：在原始分类器输入空间 `(1, C, T)` 上对时间轴做 `rfft`；总能量默认统计 `>=1 Hz` 的频段，高频能量默认统计 `>=30 Hz` 的频段。
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_n64/`
  - 运行记录：`nohup` 后台 PID `2172313`，Python 子进程 PID `2172322`，日志 `logs/exp005_rank_growth_hf_full_sweep_n64_20260602_191503.log`
  - 启动备注：第一次沙箱内 `nohup` 未保活，第二次沙箱外 `nohup` 实际已成功运行；后续误启动的 tmux 副本 `exp005_hf_20260602_191603` 已终止，避免重复写同一输出目录。
  - 实现验证：`python -m py_compile tensor_ring_rank_analysis/analyze_tr_rank_predictions.py` 通过；`conda run -n torch python ... --help` 能显示新增参数；`sample_num=1, rank_growth_steps_per_rank=1, --no_visualize` smoke run 成功生成频域 CSV；基于 smoke 输出的 `--plot_only` 也成功生成 plots。
- **新增输出指标：**
  - `incremental_energy_total`：`a_r` 在有效频段内的总能量。
  - `incremental_high_freq_energy`：`a_r` 在 `freq >= high_freq_cutoff_hz` 频段内的能量。
  - `incremental_high_freq_ratio`：`incremental_high_freq_energy / incremental_energy_total`。
  - `incremental_l2`：`a_r` 的时域 L2 能量，用于判断新增成分幅度。
  - `reconstruction_gain_abs`：`mse(\hat{x}_{r-1}, x) - mse(\hat{x}_r, x)`。
  - `reconstruction_gain_rel`：`reconstruction_gain_abs / mse(\hat{x}_{r-1}, x)`。
  - `hf_stop_candidate`：探索性停止标记；当 `reconstruction_gain_rel` 较小且 `incremental_high_freq_ratio` 较高时，候选选择上一档 rank。
- **指标：**
  - 主指标：相邻 rank pair 上 clean/adv 的 `incremental_high_freq_ratio_mean/std` 与 `reconstruction_gain_rel_mean/std`
  - 次指标：`incremental_l2_mean/std`，`hf_stop_candidate_count`，candidate rank 分布，clean/adv paired 差值
  - 对照指标：`EXP-004` 已有的 `js_mean`、`mse_mean`、`top1_change_rate`
- **结果：**
  - 进程已结束，结果文件写入 `tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_n64/`。
  - 输出文件：
    - `rank_growth_incremental_frequency.csv`
    - `rank_growth_incremental_frequency_summary.csv`
    - `rank_growth_incremental_frequency_pair_delta.csv`
    - `rank_growth_incremental_frequency_pair_delta_summary.csv`
    - `rank_growth_pair_delta_bootstrap_rows.csv`
    - `rank_growth_pair_delta_bootstrap_summary.csv`
    - `rank_growth_pair_delta_bootstrap_meta.json`
    - `rank_growth_history.csv`
    - `rank_growth_summary.csv`
    - `rank_growth_predictions.pt`
    - `meta.json`
    - `plots/`
  - 追加离线 paired-delta/bootstrap 分析命令：

    ```bash
    conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_rank_growth_pair_delta.py \
      --input_dir tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_n64 \
      --output_prefix rank_growth_pair_delta_bootstrap \
      --bootstrap_iters 5000 \
      --ci 95 \
      --seed 42 \
      --plot_format png \
      --plot_dpi 180 \
      2>&1 | tee logs/exp005_pair_delta_bootstrap_20260602_203145.log
    ```

  - 追加图表：
    - `plots/rank_growth_pair_delta_bootstrap_delta_hf_ratio_ci.png`
    - `plots/rank_growth_pair_delta_bootstrap_delta_margin_ci.png`
    - `plots/rank_growth_pair_delta_bootstrap_hf_up_margin_down_rate.png`
  - `rank_growth_full_sweep=True`，所有 clean/adv 样本都完整评估到 `rank40`；因此 selected rank 在本诊断实验中均为 `40`，不代表真实早停策略。
  - 高频占比与重构收益摘要：

    | source | 5->10 HF ratio | 35->40 HF ratio | HF ratio increase | 5->10 rel gain | 35->40 rel gain | 5->10 L2 | 35->40 L2 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | clean | 0.083405 | 0.224533 | 0.141128 | 0.501132 | 0.274017 | 23620.75 | 1419.04 |
    | adv | 0.083231 | 0.223747 | 0.140516 | 0.499055 | 0.270161 | 23515.21 | 1413.20 |

  - 相邻 rank 的平均高频占比随 rank 增大整体上升：
    - clean：`0.083405 -> 0.156787 -> 0.166721 -> 0.174536 -> 0.191775 -> 0.211350 -> 0.224533`
    - adv：`0.083231 -> 0.158284 -> 0.166493 -> 0.173505 -> 0.191362 -> 0.210415 -> 0.223747`
  - 相邻 rank 的平均相对 MSE gain 随 rank 增大下降，但并未低到当前探索性停止阈值 `0.02`：
    - clean：`0.501132 -> 0.354414 -> 0.304553 -> 0.300820 -> 0.293471 -> 0.280234 -> 0.274017`
    - adv：`0.499055 -> 0.353799 -> 0.307542 -> 0.297601 -> 0.297724 -> 0.278238 -> 0.270161`
  - `hf_stop_candidate_count` 在所有 rank pair、clean/adv 上均为 `0`，因为当前阈值要求 `reconstruction_gain_rel <= 0.02` 且 `incremental_high_freq_ratio >= 0.5`，实际高频占比最高约 `0.2245`，相对重构收益最低约 `0.2702`。
  - paired adv-clean 差值很小：

    | rank pair | delta HF ratio mean | adv HF ratio higher rate | delta rel gain mean | adv gain lower rate |
    | --- | ---: | ---: | ---: | ---: |
    | 5->10 | -0.000174 | 0.562500 | -0.002077 | 0.625000 |
    | 10->15 | 0.001497 | 0.562500 | -0.000614 | 0.484375 |
    | 15->20 | -0.000228 | 0.531250 | 0.002989 | 0.484375 |
    | 20->25 | -0.001032 | 0.453125 | -0.003219 | 0.578125 |
    | 25->30 | -0.000413 | 0.531250 | 0.004253 | 0.390625 |
    | 30->35 | -0.000935 | 0.421875 | -0.001996 | 0.562500 |
    | 35->40 | -0.000785 | 0.468750 | -0.003856 | 0.578125 |
  - paired-delta/bootstrap 与 margin 联合分析：

    | rank pair | delta HF mean | delta HF 95% CI | adv HF higher rate | delta margin mean | delta margin 95% CI | adv margin improve rate | corr HF-margin | HF up + margin down | HF up + top1 bad |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | 5->10 | -0.000174 | [-0.002679, 0.002298] | 0.562500 | -0.361217 | [-0.913571, 0.178385] | 0.734375 | -0.262346 | 0.343750 | 0.031250 |
    | 10->15 | 0.001497 | [-0.001262, 0.004299] | 0.562500 | 0.312065 | [-0.079018, 0.694786] | 0.671875 | -0.124293 | 0.234375 | 0.000000 |
    | 15->20 | -0.000228 | [-0.002982, 0.002327] | 0.531250 | -0.422184 | [-0.748333, -0.100358] | 0.531250 | -0.184134 | 0.375000 | 0.015625 |
    | 20->25 | -0.001032 | [-0.003906, 0.001736] | 0.453125 | 0.169661 | [-0.095488, 0.458584] | 0.750000 | 0.183697 | 0.203125 | 0.000000 |
    | 25->30 | -0.000413 | [-0.003103, 0.002340] | 0.531250 | -0.091277 | [-0.276988, 0.088409] | 0.625000 | -0.106273 | 0.296875 | 0.000000 |
    | 30->35 | -0.000935 | [-0.003318, 0.001381] | 0.421875 | -0.021390 | [-0.183000, 0.145927] | 0.546875 | 0.086373 | 0.218750 | 0.000000 |
    | 35->40 | -0.000785 | [-0.002836, 0.001262] | 0.468750 | -0.074218 | [-0.215059, 0.067143] | 0.734375 | 0.066476 | 0.218750 | 0.000000 |
- **观察：**
  - `IDEA-002` 中“新增恢复成分的高频占比会随 rank 增大升高”得到支持：clean 和 adv 都从约 `0.083` 单调升到约 `0.224`。
  - 但这个趋势不是 adversarial-specific：clean/adv 曲线几乎重合，paired `adv-clean` 高频占比均值差都在约 `[-0.0010, 0.0015]` 范围内，且 adv 高频占比更高的样本比例围绕 `0.5` 波动。
  - 追加 bootstrap 后，所有 rank pair 的 `delta_hf_ratio_mean` 95% CI 都跨过 `0`；`10->15` 仍是唯一较明显的正均值，但其区间 `[-0.001262, 0.004299]` 不支持稳定正效应。
  - `delta_margin_mean` 只有 `15->20` 的 95% CI 完全低于 `0`，表示该段 adversarial 相对 clean 的 true-label margin 变化更差；但它并不伴随更高的 adversarial 高频占比均值，因此不能解释为“高频恢复导致 margin 变差”的稳定链条。
  - `corr_delta_hf_margin` 绝对值整体较小，最大约 `0.262`；高频占比差和分类边界差没有形成强线性关系。
  - `hf_up_top1_bad_rate` 基本为 `0`，说明相邻 rank 增长中“adv 高频占比上升且 top1 从对变错”的事件很少见。
  - 高频占比升高主要发生在“新增成分总量变小”的背景下：`incremental_l2` 从约 `23500` 降到约 `1410`，绝对高频能量也从约 `3.0e6` 降到约 `4.7e5`。因此这里更准确的解释是：新增恢复成分越来越少，但其中高频比例越来越高。
  - `reconstruction_gain_rel` 下降幅度有限，末端 `35->40` 仍约 `0.27`，说明当前 MSE 相对收益定义没有进入“收益很小”的区域；若要做停止准则，需要重新定义收益饱和指标或改用 absolute gain / classification-stability / robust acc proxy。
  - 当前 `hf_stop_candidate` 阈值过严，在本实验中完全不触发，不适合作为实际 rank 选择规则。
- **问题：**
  - 已实现频域增量指标：rank callback 在启用开关时临时保存每档净化张量，计算完相邻 rank 指标后从持久化 trace 中移除大张量，避免显著增大 `.pt` 输出。
  - 高频阈值 `30 Hz` 是先验诊断设置，不应直接视为最终停止阈值；如果结果对阈值敏感，需要追加 `20/25/30/35 Hz` sweep。
  - full-sweep 会强制跑到 `rank40`，计算成本高；先跑 `sample_num=64`，避免直接启动长实验。
  - 该实验只验证“高频占比是否是有信息量的诊断信号”，不直接证明用该信号停止能提升 `standard acc` 或 `robust acc`。
- **结论：**
  - `EXP-005` 部分支持 `IDEA-002`：rank 增长过程中新增恢复成分的高频占比确实上升。
  - 但当前结果不支持“高频占比是 adversarial 样本特异的扰动恢复信号”：clean 和 adv 的高频占比、相对重构收益、L2 增量都高度接近。
  - 追加 margin 与 bootstrap 分析后，仍不支持把 `10->15` 的局部正峰值解释为稳定的 adversarial-specific 高频细节恢复；它更像小样本 paired 差值波动。
  - 暂不建议把 `incremental_high_freq_ratio` 单独作为 rank-growth 停止准则；更合理的用法是把它作为诊断特征，和 absolute reconstruction gain、分类稳定性、robust acc 或 paired clean/adv 差异共同分析。
- **下一步：**
  - 若继续探索高频停止规则，应先改用更合理的候选阈值，例如基于 `incremental_high_freq_ratio >= 0.20/0.22` 与 absolute MSE gain 或 absolute `incremental_l2` 的联合条件，而不是当前 `0.5/0.02`。
  - 针对 `10->15` 的局部正峰值，追加更细 rank 网格实验，把粗粒度 `5->10->15->20` 拆成 `5,6,7,...,20`，确认峰值来自整个 `10->15` 区间还是集中在某个更窄 rank 段。
  - 追加 `high_freq_cutoff_hz=20/25/35` 的小样本敏感性分析，确认高频占比单调上升是否依赖 `30 Hz` 阈值。
  - 如果目标是 adversarial-specific 信号，应继续寻找 paired sample-level 分类边界指标，而不是只依赖 aggregate 高频占比。

### EXP-006：rank 5-20 细粒度高频增量诊断

- **日期：** 2026-06-02
- **状态：** 已完成
- **相关 idea：** `IDEA-002`
- **目的：**
  - 复查 `EXP-005` 中 `10->15` 的 paired adv-clean 高频占比局部正峰值。
  - 将 rank 序列从粗粒度 `[5,10,15,20]` 拆成细粒度 `5,6,7,...,20`，判断对抗样本“多恢复高频细节”的现象是贯穿 `10->15`，还是集中在 `10->11/11->12/.../14->15` 的某个窄区间。
  - 评估细粒度 rank 统计是否能提供更稳定的 adversarial-specific 信号。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：新增 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py --rank_growth_ranks` 参数，允许命令行覆盖 YAML 中的 `rank_growth_ranks`，避免为每个细粒度 sweep 新建配置文件。
- **配置文件：**
  - 基础配置：`configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - 计划 override：
    - `rank_growth_ranks=5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20`
    - `rank_growth_js_threshold=-1.0`
    - `rank_growth_max_mse_to_input=None`
- **命令：**
  ```bash
  SAMPLE_NUM=64
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp006_hf_fine_r5-20_n${SAMPLE_NUM}"
  LOG_FILE="logs/exp006_rank_growth_hf_fine_r5-20_n${SAMPLE_NUM}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  setsid nohup conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --rank_growth_full_sweep \
    --rank_growth_ranks 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
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
    --enable_incremental_frequency_metrics \
    --high_freq_cutoff_hz 30.0 \
    --freq_energy_floor_hz 1.0 \
    --output_dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - rank 序列：`range(5, 21)`，即 `5` 到 `20` 闭区间
  - 样本数：先用 `64`，与 `EXP-005` 对齐
  - 高频阈值：先固定 `30 Hz`，避免同时引入 cutoff 和 rank 网格两个变量
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp006_hf_fine_r5-20_n64/`
  - 运行记录：`nohup` 后台 PID `2181454`，日志 `logs/exp006_rank_growth_hf_fine_r5-20_n64_20260602_194544.log`
  - 实现验证：`python -m py_compile tensor_ring_rank_analysis/analyze_tr_rank_predictions.py` 通过；`conda run -n torch python ... --help` 能显示 `--rank_growth_ranks`；`sample_num=1, rank_growth_steps_per_rank=1, --rank_growth_ranks 5,6,7` smoke run 成功，`meta.json` 中 `rank_growth_ranks=[5,6,7]`，频域 summary 输出 `5->6` 和 `6->7`；基于 smoke 输出的 `--plot_only` 成功。
- **指标：**
  - 主指标：每个相邻 rank pair 的 `delta_incremental_high_freq_ratio_mean`、`adv_hf_ratio_higher_rate`
  - 次指标：`delta_reconstruction_gain_rel_mean`、`delta_incremental_l2_mean`、`top1_change_rate`
  - 重点区间：`10->11`、`11->12`、`12->13`、`13->14`、`14->15`
- **结果：**
  - 进程已结束，结果文件写入 `tensor_ring_rank_analysis/results_rank_growth_exp006_hf_fine_r5-20_n64/`。
  - 输出文件：
    - `rank_growth_incremental_frequency.csv`
    - `rank_growth_incremental_frequency_summary.csv`
    - `rank_growth_incremental_frequency_pair_delta.csv`
    - `rank_growth_incremental_frequency_pair_delta_summary.csv`
    - `rank_growth_history.csv`
    - `rank_growth_summary.csv`
    - `rank_growth_predictions.pt`
    - `meta.json`
    - `plots/`
  - `meta.json` 确认 `rank_growth_ranks=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]`，`rank_growth_full_sweep=True`，`sample_num=64`。
  - 细粒度 paired adv-clean 高频占比差值：

    | rank pair | delta HF ratio mean | adv HF ratio higher rate | delta rel gain mean |
    | --- | ---: | ---: | ---: |
    | 5->6 | -0.000589 | 0.546875 | -0.000483 |
    | 6->7 | -0.002634 | 0.500000 | 0.001479 |
    | 7->8 | 0.000189 | 0.500000 | 0.000769 |
    | 8->9 | 0.001372 | 0.562500 | -0.001766 |
    | 9->10 | -0.001766 | 0.468750 | 0.001397 |
    | 10->11 | -0.002113 | 0.437500 | 0.000569 |
    | 11->12 | 0.001069 | 0.468750 | -0.004038 |
    | 12->13 | -0.001983 | 0.500000 | -0.000752 |
    | 13->14 | -0.002633 | 0.484375 | -0.000141 |
    | 14->15 | -0.005575 | 0.468750 | -0.002016 |
    | 15->16 | -0.002687 | 0.390625 | 0.000954 |
    | 16->17 | -0.001405 | 0.468750 | 0.000252 |
    | 17->18 | -0.002465 | 0.437500 | 0.001430 |
    | 18->19 | 0.002659 | 0.578125 | -0.000778 |
    | 19->20 | 0.001337 | 0.500000 | -0.000117 |

  - `10->15` 重点区间中，只有 `11->12` 为正 `0.001069`，其余 `10->11/12->13/13->14/14->15` 均为负；五个细粒度 pair 的均值平均约 `-0.002247`。
  - 最大正均值出现在 `18->19`，为 `0.002659`；最大负均值出现在 `14->15`，为 `-0.005575`。
  - 所有细粒度 pair 的标准差约 `0.0157~0.0221`，远大于均值差，说明样本级波动仍然很大。
- **观察：**
  - `EXP-005` 中粗粒度 `10->15` 的 paired 正峰值没有在细粒度 `10..15` 内复现为连续正区间。
  - `10->15` 内部更像是噪声主导的局部波动：`11->12` 有小幅正值，但 `14->15` 出现更明显负值，整体不支持“rank10 到 rank15 期间 adv 持续恢复更多高频细节”。
  - 细粒度实验与粗粒度实验不是严格同一条优化路径：`rank_growth_ranks` 改成逐 rank 增长后，TN 优化会经历更多中间 rank block，因此不能把 `EXP-005` 的 `10->15` 直接数学分解为本实验的五个相邻 pair。不过作为更精细诊断，本实验没有支持该局部正峰值是稳定结构。
  - `18->19` 和 `19->20` 出现正均值，但量级仍小、方差大，暂时也不能解释为稳定 adversarial-specific 信号。
- **问题：**
  - 细粒度 rank 会显著增加 full-sweep 计算量：每个样本从 8 个 rank block 增加到 16 个 rank block；`sample_num=64` 预计耗时约为 `EXP-005` 的两倍左右。
  - 如果 `10->15` 的正峰值只由少数样本驱动，细粒度统计可能仍需要更大样本数或 bootstrap 置信区间。
  - 已实现命令行 rank override；后续若要批量 cutoff sweep，可复用 `--rank_growth_ranks`，不需要复制 YAML。
- **结论：**
  - `EXP-006` 不支持 `EXP-005` 中 `10->15` 粗粒度正峰值是一个稳定、可定位的 adversarial-specific 高频恢复区间。
  - 当前更合理的解释是：paired adv-clean 高频占比差值在细粒度 rank 上主要是小幅样本波动，尚未形成可靠的停止或诊断信号。
  - 不建议继续围绕 `10->15` 单独设计 rank-growth 停止准则；若继续研究高频指标，应转向 cutoff 敏感性、bootstrap 置信区间或与分类边界指标联合。
- **下一步：**
  - 对 paired delta 增加 bootstrap 置信区间或可视化误差棒，避免仅凭均值折线判断局部峰值。
  - 若继续做 cutoff sweep，应覆盖完整 `5..20` 区间，而不是只围绕 `10..15`。
  - 可以考虑把高频占比与分类 top1 变化、confidence delta 或 margin delta 联合分析，寻找更接近 adversarial perturbation 的信号。

### EXP-007：Optuna 离线寻优 rank-growth 早停/选 rank 规则

- **日期：** 2026-06-03
- **状态：** 已完成
- **相关 idea：** `IDEA-003`
- **目的：**
  - 基于已有 rank-growth full-sweep 轨迹，离线重放不同 sample-wise rank selection 规则，避免每个 Optuna trial 重新运行 TN 净化。
  - 验证 JS、净化前预测标签 margin 和 MSE 能否形成比当前手工阈值更严密的早停/选 rank 标准。
  - 以 robust accuracy 为主目标，同时记录 clean accuracy、selected MSE 和平均 selected rank，避免单纯追求高 rank 或过拟合小样本。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：新增 `tensor_ring_rank_analysis/optuna_rank_growth_selection.py`，只做离线 Optuna 调参，不修改 `purify.py` 和 `PTR_3d_rank_growth` 默认早停逻辑。
- **配置文件：**
  - 输入 full-sweep 目录：`tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_r4-40_step2_n64/`
  - 输入轨迹文件：`rank_growth_predictions.pt`
  - 原始 full-sweep 配置：`configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
- **命令：**
  ```bash
  SAMPLE_NUM=64
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64"
  LOG_FILE="logs/exp007_optuna_rank_selection_r4-40_step2_n64_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  setsid nohup conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/optuna_rank_growth_selection.py \
    --input_dir tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_r4-40_step2_n64 \
    --output_dir "${OUT_DIR}" \
    --study_name exp007_optuna_rank_selection_r4_40_step2_n64 \
    --n_trials 300 \
    --seed 42 \
    --tune_ratio 0.5 \
    --selection_modes threshold,score \
    --objective robust_priority \
    --clean_weight 0.20 \
    --mse_weight 0.10 \
    --rank_weight 0.02 \
    --plot_format png \
    --plot_dpi 180 \
    > "${LOG_FILE}" 2>&1 &
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 输入 rank 序列：`4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40`
  - 样本数：`64`
  - tune/holdout 划分：按 `sample_id` 固定随机划分，`tune_ratio=0.5`
  - rank selection 规则：
    - `threshold`：首个满足 `JS <= js_threshold`、`MSE <= mse_threshold`、`margin >= margin_threshold` 的相邻 rank pair，选择前一个 rank；否则选择最大 rank。
    - `score`：使用归一化特征 `alpha * JS + beta * MSE - gamma * Margin + delta * rank_norm`，选择分数最低 rank。
  - margin 定义：以净化前预测类别为参考，计算净化后该类别 softmax 概率与最强其他类别概率的差，不使用 true label 参与 rank selection。
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64/`
  - 运行记录：正式运行已完成，日志 `logs/exp007_optuna_rank_selection_r4-40_step2_n64_20260603_195715.log`，输出写入 `tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64/`。
  - 运行备注：当前执行环境首次 `setsid nohup` 子进程未保活，正式结果来自同参数前台等待完成命令；stdout/stderr 仍重定向到上述稳定日志文件。
  - raw logits：输入 `.pt` 中没有 raw logits，脚本根据 meta 中 checkpoint/ad_data 路径轻量重算，`meta.json` 中 `raw_logits_recomputed=true`。
- **指标：**
  - 主指标：holdout 上的 adversarial selected accuracy。
  - 次指标：holdout clean selected accuracy、selected MSE、平均 selected rank、tune/holdout objective gap、selected rank 分布。
  - 诊断指标：`trials.csv` 中不同 selector 与参数的 objective、`selected_rows.csv` 中逐样本 selected rank/margin/MSE/JS。
- **结果：**
  - 进程已结束，退出码 `0`。
  - 输出文件：
    - `trials.csv`
    - `best_config.json`
    - `selected_rows.csv`
    - `summary.csv`
    - `meta.json`
    - `plots/objective_history.png`
    - `plots/selected_rank_distribution.png`
  - 最优 trial：`272`，tune objective `1.0078534033349973`。
  - 最优 selector：`score`。
  - 最优参数：
    - `alpha=0.18952655220538`
    - `beta=1.9165340519262832`
    - `gamma=0.0425660367878316`
    - `delta=0.5821138968534183`
  - selected accuracy / MSE / rank 摘要：

    | split | source | count | accuracy | mean MSE | mean rank | objective |
    | --- | --- | ---: | ---: | ---: | ---: | ---: |
    | tune | clean | 32 | 0.906250 | 0.065946 | 22.937500 |  |
    | tune | adv | 32 | 0.843750 | 0.066916 | 22.875000 |  |
    | tune | all | 64 | 0.875000 | 0.066431 | 22.906250 | 1.007853 |
    | holdout | clean | 32 | 0.968750 | 0.070229 | 23.375000 |  |
    | holdout | adv | 32 | 0.906250 | 0.068233 | 23.812500 |  |
    | holdout | all | 64 | 0.937500 | 0.069231 | 23.593750 | 1.082191 |
    | all | clean | 64 | 0.937500 | 0.068088 | 23.156250 |  |
    | all | adv | 64 | 0.875000 | 0.067575 | 23.343750 |  |
    | all | all | 128 | 0.906250 | 0.067831 | 23.250000 | 1.045022 |

  - selected rank 分布：

    | rank | clean count | adv count | all count |
    | ---: | ---: | ---: | ---: |
    | 16 | 1 | 2 | 3 |
    | 18 | 9 | 8 | 17 |
    | 20 | 5 | 5 | 10 |
    | 22 | 12 | 14 | 26 |
    | 24 | 20 | 15 | 35 |
    | 26 | 10 | 8 | 18 |
    | 28 | 5 | 9 | 14 |
    | 30 | 2 | 3 | 5 |
- **观察：**
  - 在本次 `n64` tune split 上，最优规则选择了 `score` selector，而不是 `threshold` selector。
  - 最优 score 中 `beta` 权重最大，`delta` 次之，`alpha/gamma` 较小，说明本次小样本调参更偏向 MSE 保真和中等 rank 惩罚，JS 与净化前预测 margin 的直接贡献较弱。
  - holdout adversarial selected accuracy 为 `0.90625`，高于 tune adversarial selected accuracy `0.84375`；这不是过拟合证据，但由于 holdout 只有 `32` 个样本，波动可能较大。
  - clean 与 adversarial 的平均 selected rank 很接近：all split 上 clean 为 `23.15625`，adv 为 `23.34375`，当前最优规则没有明显把 adversarial 样本推到更高 rank。
  - selected rank 主要集中在 `22/24/26`，没有大量退到最低 rank，也没有退化成全部选择最大 rank。
- **问题：**
  - 本实验只使用 `64` 个样本，其中 tune/holdout 各 `32` 个样本；结果只能视为离线规则可行性和候选参数，不足以作为最终策略。
  - objective 基于 full-sweep 已保存 logits 做离线选择，没有生成真实 selected purified `.pth`，也没有通过 `purify.py` 重新评估完整 pipeline。
  - 当前没有把最优规则和既有 `js=0.02,mse=0.08`、固定 rank25/30 在同一 `r4-40 step2 n64` 输入上做统一 baseline replay。
- **结论：**
  - `EXP-007` 证明离线 Optuna rank-selection 重放流程可运行，并能从 full-sweep 轨迹中导出可解释的候选选 rank 规则。
  - 本次最优候选为 score 规则，倾向选择中高 rank、较低 MSE 的重建；在 `n64` holdout 上达到 adversarial selected accuracy `0.90625`，但该数值受小样本划分影响较大。
  - 当前不应直接把该 Optuna 规则写入 `PTR_3d_rank_growth` 在线早停逻辑；更合理的是先补 baseline replay 和更大样本验证。
- **下一步：**
  - 在同一离线脚本中加入 baseline replay：固定 rank、现有 `JS+MSE gate`、纯 MSE gate、纯 JS gate，和 Optuna best 进行同表比较。
  - 若 baseline replay 仍支持 Optuna score 规则，再规划 `sample_num=512` full-sweep 或复用已有更大 full-sweep 轨迹做正式验证。
  - 后续若要接入在线净化，应把 feature normalization 统计、score 参数和 label-free margin 定义显式写入配置，避免隐式依赖 tune split。

### EXP-008：单指标与两两组合 rank-selection ablation

- **日期：** 2026-06-03
- **状态：** 已完成（Completed）
- **相关 idea：** `IDEA-003`
- **目的：**
  - 在 `EXP-007` 基础上追加离线 ablation，判断 JS、MSE、净化前预测标签 margin 单独或两两组合时对 sample-wise rank selection 的贡献。
  - 比较 `JS-only`、`MSE-only`、`Margin-only`、`JS+MSE`、`MSE+Margin`、`JS+Margin` 与 `EXP-007` 的 `Full score` best。
  - 继续复用已有 full-sweep 轨迹，不重新运行 TN 净化。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：扩展 `tensor_ring_rank_analysis/optuna_rank_growth_selection.py`，新增 ablation selection modes、mode-wise Optuna 和 `mode_best_summary.csv`。
- **配置文件：**
  - 输入 full-sweep 目录：`tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_r4-40_step2_n64/`
  - 输入轨迹文件：`rank_growth_predictions.pt`
  - `Full score` 对照：`tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64/best_config.json`
- **命令：**
  ```bash
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp008_ablation_r4-40_step2_n64"
  LOG_FILE="logs/exp008_rank_selection_ablation_r4-40_step2_n64_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/optuna_rank_growth_selection.py \
    --input_dir tensor_ring_rank_analysis/results_rank_growth_exp005_hf_full_sweep_r4-40_step2_n64 \
    --output_dir "${OUT_DIR}" \
    --study_name exp008_rank_selection_ablation_r4_40_step2_n64 \
    --n_trials 300 \
    --seed 42 \
    --tune_ratio 0.5 \
    --selection_modes js_only,mse_only,margin_only,js_mse,mse_margin,js_margin \
    --include_exp007_full_score tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64/best_config.json \
    --objective robust_priority \
    --clean_weight 0.20 \
    --mse_weight 0.10 \
    --rank_weight 0.02 \
    --plot_format png \
    --plot_dpi 180 \
    > "${LOG_FILE}" 2>&1
  ```
- **实际运行：**
  - 运行时间：2026-06-03
  - 日志文件：`logs/exp008_rank_selection_ablation_r4-40_step2_n64_20260603_204525.log`
  - 退出码：`0`
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 输入 rank 序列：`4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40`
  - 样本数：`64`
  - tune/holdout 划分：沿用 `EXP-007` 的 `seed=42`、`tune_ratio=0.5`
  - Optuna：每个 ablation mode 单独 `300` trials；`Full score` 不重新调参，只复用 `EXP-007` best 参数。
  - ablation selection modes：
    - `js_only`：首个 `JS <= js_threshold` 的相邻 rank pair，选择前一个 rank。
    - `mse_only`：首个 `MSE <= mse_threshold` 的 rank。
    - `margin_only`：首个 `margin >= margin_threshold` 的 rank。
    - `js_mse`：首个同时满足 `JS <= js_threshold` 和前一个 rank `MSE <= mse_threshold` 的相邻 rank pair。
    - `mse_margin`：首个同时满足 `MSE <= mse_threshold` 和 `margin >= margin_threshold` 的 rank。
    - `js_margin`：首个同时满足 `JS <= js_threshold` 和前一个 rank `margin >= margin_threshold` 的相邻 rank pair。
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp008_ablation_r4-40_step2_n64/`
- **指标：**
  - 主指标：各 selector 在 holdout 上的 adversarial selected accuracy。
  - 次指标：holdout clean selected accuracy、selected MSE、平均 selected rank、tune/holdout objective gap。
  - 诊断指标：`mode_best_summary.csv` 中各 mode 的 best trial 与参数，`summary.csv` 中各 selector 的 tune/holdout/all 指标，`selected_rows.csv` 中逐样本 selected rank。
- **结果：**
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp008_ablation_r4-40_step2_n64/`
  - 输出文件：`trials.csv`、`best_config.json`、`selected_rows.csv`、`summary.csv`、`mode_best_summary.csv`、`meta.json`、`plots/objective_history_*.png`、`plots/selected_rank_distribution_global_best.png`
  - `trials.csv`：`1800` 行，即 `6` 个 ablation mode × 每个 mode `300` trials；`exp007_full_score` 只作为复用对照，不重新调参。
  - `summary.csv`：`63` 行，即 `7` 个 selector × `tune/holdout/all` × `clean/adv/all`。
  - `selected_rows.csv`：`896` 行，即 `7` 个 selector × `64` 个样本 × `clean/adv`。
  - 全局最佳 selector：`exp007_full_score`，沿用 `EXP-007` 的 score 参数；best tune objective 为 `1.0078534033349973`。
  - 各 selector 的 best 结果：

    | selector | best trial | best objective | best params | holdout adv acc | holdout clean acc | holdout MSE | holdout rank | all adv acc | all clean acc | all MSE | all rank |
    | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `js_only` | 16 | 1.000802 | `js_threshold=1.084292e-05` | 0.90625 | 0.96875 | 0.082093 | 29.46875 | 0.87500 | 0.93750 | 0.095281 | 28.75000 |
    | `mse_only` | 138 | 1.007776 | `mse_threshold=0.0684426` | 0.90625 | 0.96875 | 0.063658 | 24.75000 | 0.87500 | 0.93750 | 0.063607 | 24.15625 |
    | `margin_only` | 251 | 1.000948 | `margin_threshold=0.999980` | 0.90625 | 0.96875 | 0.034698 | 38.15625 | 0.87500 | 0.93750 | 0.054624 | 36.01562 |
    | `js_mse` | 134 | 1.007778 | `js_threshold=0.0472429, mse_threshold=0.0698950` | 0.90625 | 0.96875 | 0.064386 | 24.59375 | 0.87500 | 0.93750 | 0.064397 | 24.00000 |
    | `mse_margin` | 277 | 1.007776 | `mse_threshold=0.0684070, margin_threshold=-0.518746` | 0.90625 | 0.96875 | 0.063426 | 24.78125 | 0.87500 | 0.93750 | 0.063491 | 24.17188 |
    | `js_margin` | 91 | 1.000825 | `js_threshold=1.142011e-05, margin_threshold=0.357020` | 0.90625 | 0.96875 | 0.084132 | 29.09375 | 0.87500 | 0.93750 | 0.096448 | 28.51562 |
    | `exp007_full_score` | - | 1.007853 | `alpha=0.189527, beta=1.916534, gamma=0.0425660, delta=0.582114` | 0.90625 | 0.96875 | 0.069231 | 23.59375 | 0.87500 | 0.93750 | 0.067831 | 23.25000 |
- **观察：**
  - 在本次 `n64` 离线 replay 中，`7` 个 selector 的 holdout adversarial selected accuracy 全部为 `0.90625`，holdout clean selected accuracy 全部为 `0.96875`；主指标没有区分出单指标、两两组合和 full score。
  - objective 的差距主要来自 MSE 与 rank penalty。`exp007_full_score` 在 tune/all objective 上最高，原因是平均 rank 最低：holdout mean rank `23.59375`、all mean rank `23.25`。
  - `mse_only`、`js_mse`、`mse_margin` 与 `exp007_full_score` 非常接近；holdout objective 只差约 `7e-5` 到 `8e-5`，说明在当前样本上 MSE gate 已能复现 full score 的大部分效果。
  - `js_only` 和 `js_margin` 倾向选择更高 rank 或退到最大 rank，holdout/all MSE 高于 MSE 相关规则，objective 明显低于 full score。
  - `margin_only` 的 holdout MSE 最低，为 `0.034698`，但平均 rank 达到 `38.15625`；`selected_rows.csv` 中 `112/128` 个 clean/adv rows 选择 rank `40`，更像是高 rank 保真对照，而不是有效早停规则。
- **问题：**
  - 所有 selector 的 selected accuracy 完全相同，说明 `n64` 样本对 accuracy 指标的分辨率不足；当前 ablation 主要能比较 MSE/rank trade-off，不能证明某个指标组合带来分类准确率优势。
  - `mse_margin` 的 best margin threshold 为负值 `-0.518746`，实际约等于放松 margin 条件，主要仍由 MSE threshold 决定。
  - 本实验仍是 full-sweep logits 的离线 replay，没有重新生成真实 selected purified `.pth`，也没有接入 `purify.py` 在线早停。
- **结论：**
  - `EXP-008` 支持一个保守判断：在当前 `n64` 设置下，MSE 是最有用的单指标；JS 和 margin 对 objective 的增益不明显。
  - `EXP-007` full score 仍是全局最佳，但优势很小，更多体现在更低平均 rank，而不是更高 selected accuracy。
  - 如果后续要简化在线规则，`mse_only` 或 `js_mse` 是比 full score 更容易实现和解释的候选，但需要更大样本确认。
- **下一步：**
  - 在 `n512` 或跨 seed/fold 上复跑同样的离线 ablation，重点看 accuracy 是否仍完全打平。
  - 补一个 fixed-rank baseline replay，把固定 rank `22/24/26/30/40` 与 `mse_only`、`js_mse`、`exp007_full_score` 放到同一张表。
  - 若更大样本仍显示 MSE 规则足够接近 full score，优先规划一个只含 `mse_threshold` 的在线早停版本，避免引入复杂 score 权重。

### EXP-009：n512 rank growth full-sweep 轨迹分析（r5-40 step5）

- **日期：** 2026-06-03
- **状态：** 已完成
- **相关 idea：** `IDEA-003`
- **目的：**
  - 将 `EXP-007/EXP-008` 的 `n64` 离线调参扩展到 `sample_num=512`，提高 accuracy 指标的分辨率。
  - 使用较粗但标准的 rank 序列 `range(5, 45, 5)`，即 `5,10,15,20,25,30,35,40`，生成完整 full-sweep 轨迹。
  - 为后续 `EXP-010` Optuna rank-selection 调参提供输入，不重新修改 `purify.py` 或 baseline 在线早停逻辑。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：复用 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py --rank_growth_full_sweep --rank_growth_ranks`。
- **配置文件：**
  - 基础配置：`configs/thubenchmark/PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml`
  - checkpoint：`checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth`
  - adversarial data：`ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth`
- **命令：**
  ```bash
  SAMPLE_NUM=512
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512"
  LOG_FILE="logs/exp009_rank_growth_hf_full_sweep_r5-40_step5_n512_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  setsid nohup conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --rank_growth_full_sweep \
    --rank_growth_ranks 5,10,15,20,25,30,35,40 \
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
    --enable_incremental_frequency_metrics \
    --high_freq_cutoff_hz 30.0 \
    --freq_energy_floor_hz 1.0 \
    --plot_format png \
    --plot_dpi 180 \
    --output_dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &
  ```
- **实际运行：**
  - 启动时间：2026-06-03 21:40:03
  - 后台 pipeline PID：`84384`
  - pipeline 状态日志：`logs/exp009_010_pipeline_r5-40_step5_n512_20260603_214003.log`（记录中预期存在；当前未在 `logs/` 下观察到该文件）
  - `EXP-009` 实时日志：`logs/exp009_rank_growth_hf_full_sweep_r5-40_step5_n512_20260603_214003.log`
  - `EXP-010` 预分配日志：`logs/exp010_optuna_rank_selection_r5-40_step5_n512_20260603_214003.log`
  - 启动方式：沙箱内 `nohup` 无法保活后，改用沙箱外 `setsid nohup` 启动；`EXP-010` 会在 `EXP-009` 成功结束后自动运行。
  - 运行确认：2026-06-03 21:40 左右日志进入 `clean rank-growth`，进度已到 `5/512`，日志持续增长。
  - 完成时间：2026-06-03 23:42 左右；日志结尾显示结果已保存到输出目录。
- **关键设置：**
  - 数据集/划分：`thubenchmark`，`fold=0`，`train_only_subject_no_ea_subject_split`
  - 随机种子：`42`
  - 模型：`eegnet`
  - 攻击：`autoattack`，`eps=0.03`
  - 对抗训练策略：`madry`
  - 样本数：`512`
  - 输入 rank 序列：`5,10,15,20,25,30,35,40`
  - full-sweep：开启；禁用早停并评估每个 rank。
  - 频域增量指标：开启；沿用 `high_freq_cutoff_hz=30.0`、`freq_energy_floor_hz=1.0`。
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/`
- **指标：**
  - 主输入：`rank_growth_predictions.pt`，供 `EXP-010` 离线 Optuna replay 使用。
  - 诊断指标：`rank_growth_summary.csv`、`rank_growth_history.csv`、`rank_growth_incremental_frequency_summary.csv` 中 clean/adv 的 JS、MSE、top1 change、高频增量趋势。
  - 检查项：`meta.json` 中 `sample_num=512`、`rank_growth_full_sweep=True`、`rank_growth_ranks=[5,10,15,20,25,30,35,40]`。
- **结果：**
  - 输出文件已生成：
    - `tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/rank_growth_predictions.pt`
    - `tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/rank_growth_history.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/rank_growth_summary.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/rank_growth_incremental_frequency_summary.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/rank_growth_incremental_frequency_pair_delta_summary.csv`
  - 固定 rank full-sweep 指标：

    | rank | clean acc | adv acc | clean MSE | adv MSE |
    | ---: | ---: | ---: | ---: | ---: |
    | 5 | 0.654296875 | 0.607421875 | 0.466249215 | 0.466416900 |
    | 10 | 0.796875000 | 0.792968750 | 0.236228291 | 0.236217403 |
    | 15 | 0.855468750 | 0.824218750 | 0.154737043 | 0.154797028 |
    | 20 | 0.865234375 | 0.837890625 | 0.108968614 | 0.108971995 |
    | 25 | 0.880859375 | 0.830078125 | 0.077177534 | 0.077267562 |
    | 30 | 0.898437500 | 0.830078125 | 0.054547436 | 0.054664118 |
    | 35 | 0.916015625 | 0.835937500 | 0.039143459 | 0.039172390 |
    | 40 | 0.910156250 | 0.835937500 | 0.028405062 | 0.028476913 |

  - 高频增量占比随 rank 增大整体上升：

    | rank pair | adv hf ratio | clean hf ratio | paired adv-clean delta |
    | --- | ---: | ---: | ---: |
    | 5->10 | 0.082801 | 0.083094 | -0.000292 |
    | 10->15 | 0.158379 | 0.158367 | 0.000012 |
    | 15->20 | 0.167139 | 0.167229 | -0.000089 |
    | 20->25 | 0.173010 | 0.172429 | 0.000581 |
    | 25->30 | 0.189939 | 0.189603 | 0.000336 |
    | 30->35 | 0.209183 | 0.209443 | -0.000260 |
    | 35->40 | 0.223626 | 0.223353 | 0.000272 |
- **观察：**
  - 固定 rank 的 adversarial accuracy 在 `rank20` 达到最高 `0.837890625`，`rank35/40` 为 `0.8359375`，继续提升 rank 没有带来 robust acc 单调提升。
  - clean accuracy 在 `rank35` 达到最高 `0.916015625`，`rank40` 虽然 MSE 最低，但 clean acc 回落到 `0.91015625`。
  - clean/adv MSE 都随 rank 单调下降，且两者数值几乎重合，说明 full-sweep 的重建保真趋势本身不是 adversarial-specific。
  - 高频增量占比从约 `0.083` 升到约 `0.224`，复现了 `EXP-005` 的 rank 增长频谱趋势；但 paired `adv-clean` delta 全部接近 `0`，`adv_hf_ratio_higher_rate` 也围绕 `0.5` 波动，仍不支持高频占比作为 adversarial-specific 停止信号。
  - 相邻 rank 的 top1 change rate 随 rank 增大总体下降；adv 的 `5->10` top1 change rate 为 `0.33203125`，到 `35->40` 降到 `0.029296875`。
- **问题：**
  - 本实验是离线 full-sweep 轨迹生成，不等同于在线早停净化效果；真正的 selector 结论依赖 `EXP-010` replay。
  - 记录中预期的 pipeline 状态日志 `logs/exp009_010_pipeline_r5-40_step5_n512_20260603_214003.log` 当前未在 `logs/` 下观察到；但 EXP-009/010 各自日志和输出文件完整存在。
  - 仍只覆盖 `thubenchmark fold0 seed42 sample_num=512 eps=0.03`，不能代表跨 fold/seed 稳定性。
- **结论：**
  - `EXP-009` 成功产出 `n512, ranks=5..40 step5` 的完整 full-sweep 输入，可用于 `EXP-010` 离线 rank selection。
  - 固定 rank 结果显示 robust acc 与 clean/MSE 的最优 rank 不一致：robust 更偏 `rank20`，clean/MSE 更偏高 rank。
  - 高频增量指标可作为 rank-growth 诊断特征，但本次 `n512` 结果再次显示它不具备明显 clean/adv 区分能力。
- **下一步：**
  - 使用本实验输出的 `rank_growth_predictions.pt` 运行并总结 `EXP-010`。

### EXP-010：n512 rank-selection Optuna 调参与 ablation（r5-40 step5）

- **日期：** 2026-06-03
- **状态：** 已完成
- **相关 idea：** `IDEA-003`
- **目的：**
  - 在 `EXP-009` 的 `n512, ranks=5..40 step5` full-sweep 轨迹上重新调参，观察样本量上升后 selector 的 accuracy、MSE 与 rank trade-off 是否分化。
  - 同时评估 `threshold`、`score`、单指标规则和两两组合规则；保留 `EXP-007` best full score 作为跨 rank-grid/样本量迁移对照。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：复用 `tensor_ring_rank_analysis/optuna_rank_growth_selection.py`。
- **配置文件：**
  - 输入 full-sweep 目录：`tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/`
  - 输入轨迹文件：`rank_growth_predictions.pt`
  - 迁移对照：`tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64/best_config.json`
- **命令：**
  ```bash
  OUT_DIR="tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512"
  LOG_FILE="logs/exp010_optuna_rank_selection_r5-40_step5_n512_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p logs
  conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/optuna_rank_growth_selection.py \
    --input_dir tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512 \
    --output_dir "${OUT_DIR}" \
    --study_name exp010_optuna_rank_selection_r5_40_step5_n512 \
    --n_trials 300 \
    --seed 42 \
    --tune_ratio 0.5 \
    --selection_modes threshold,score,js_only,mse_only,margin_only,js_mse,mse_margin,js_margin \
    --include_exp007_full_score tensor_ring_rank_analysis/results_rank_growth_exp007_optuna_selection_r4-40_step2_n64/best_config.json \
    --objective robust_priority \
    --clean_weight 0.20 \
    --mse_weight 0.10 \
    --rank_weight 0.02 \
    --plot_format png \
    --plot_dpi 180 \
    > "${LOG_FILE}" 2>&1
  ```
- **实际运行：**
  - 排队时间：2026-06-03 21:40:03
  - 后台 pipeline PID：`84384`
  - pipeline 状态日志：`logs/exp009_010_pipeline_r5-40_step5_n512_20260603_214003.log`（记录中预期存在；当前未在 `logs/` 下观察到该文件）
  - `EXP-010` 日志：`logs/exp010_optuna_rank_selection_r5-40_step5_n512_20260603_214003.log`
  - 触发条件：只有 `EXP-009` 命令退出码为 `0` 时才会开始 Optuna；若 `EXP-009` 失败，本实验保持未运行。
  - 完成时间：2026-06-03 23:42 左右；日志结尾显示结果已保存到输出目录。
- **关键设置：**
  - tune/holdout 划分：`seed=42`、`tune_ratio=0.5`
  - Optuna：每个调参 selector 单独 `300` trials；共 `8` 个调参 selector，外加 `exp007_full_score` 固定迁移对照。
  - true label：只用于 objective 和 holdout evaluation，不进入 rank-selection 规则。
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/`
- **指标：**
  - 主指标：各 selector 在 holdout 上的 adversarial selected accuracy。
  - 次指标：holdout clean selected accuracy、selected MSE、平均 selected rank、tune/holdout objective gap。
  - 诊断指标：`mode_best_summary.csv`、`summary.csv`、`trials.csv`、`selected_rows.csv`。
- **结果：**
  - 输出文件已生成：
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/mode_best_summary.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/summary.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/trials.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/selected_rows.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/best_config.json`
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/fixed_rank_selector_comparison.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp010_optuna_selection_r5-40_step5_n512/plots/fixed_rank_selector_comparison.png`
  - 各 selector 的 best 结果：

    | selector | best trial | best params | holdout adv acc | holdout clean acc | holdout MSE | holdout rank | all adv acc | all clean acc | all MSE | all rank |
    | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `threshold` | 137 | `js=0.053533,mse=0.142338,margin=-0.682932` | 0.8359375 | 0.87890625 | 0.113238 | 19.25781 | 0.84765625 | 0.884765625 | 0.113463 | 19.43359 |
    | `score` | 290 | `alpha=1.394142,beta=1.905168,gamma=0.040578,delta=0.882820` | 0.8359375 | 0.88671875 | 0.109311 | 19.60938 | 0.841796875 | 0.884765625 | 0.109180 | 19.83887 |
    | `js_only` | 52 | `js=0.018455` | 0.83203125 | 0.89453125 | 0.280244 | 11.88477 | 0.841796875 | 0.888671875 | 0.283195 | 11.93848 |
    | `mse_only` | 157 | `mse=0.109223` | 0.8359375 | 0.87890625 | 0.091114 | 22.12891 | 0.83984375 | 0.876953125 | 0.090733 | 22.34863 |
    | `margin_only` | 190 | `margin=0.999986` | 0.84765625 | 0.91796875 | 0.063762 | 35.43945 | 0.8359375 | 0.91015625 | 0.064696 | 35.73242 |
    | `js_mse` | 117 | `js=0.040303,mse=0.164277` | 0.8359375 | 0.890625 | 0.130659 | 17.37305 | 0.845703125 | 0.890625 | 0.129503 | 17.67090 |
    | `mse_margin` | 204 | `mse=0.109562,margin=-0.229481` | 0.83203125 | 0.890625 | 0.089556 | 22.52930 | 0.837890625 | 0.88671875 | 0.089119 | 22.76367 |
    | `js_margin` | 228 | `js=0.018449,margin=-0.799840` | 0.83203125 | 0.89453125 | 0.278516 | 12.04102 | 0.841796875 | 0.892578125 | 0.280104 | 12.19238 |
    | `exp007_full_score` | - | `alpha=0.189527,beta=1.916534,gamma=0.042566,delta=0.582114` | 0.828125 | 0.890625 | 0.073560 | 24.97070 | 0.828125 | 0.888671875 | 0.073842 | 25.08789 |

  - 关键 rank 分布：
    - `threshold`：主要选 `rank15/20/25`，全量分布 `{10:34, 15:366, 20:359, 25:222, 30:36, 35:4, 40:3}`。
    - `js_mse`：平均 rank 最低的强候选之一，分布 `{10:79, 15:484, 20:329, 25:104, 30:23, 35:5}`。
    - `margin_only`：`1024` 个 clean/adv rows 中 `860` 个选择 `rank40`，更像高 rank 保真对照。
    - `exp007_full_score`：主要选 `rank20/25/30`，分布 `{15:21, 20:274, 25:427, 30:271, 35:30, 40:1}`。
- **观察：**
  - `n512` 相比 `EXP-008/n64` 已经能区分 selector 的 selected accuracy；不再是所有 selector 打平。
  - 按 holdout adversarial accuracy 看，`margin_only` 最高 `0.84765625`，但它的平均 rank 为 `35.44`，且大量样本退到 `rank40`，不适合作为低成本早停规则。
  - 按全量 adversarial accuracy 看，`threshold` 最高 `0.84765625`；`js_mse` 次高 `0.845703125`，但平均 rank 更低 `17.67`，比 `threshold` 的 `19.43` 更省。
  - `mse_only` 的全量 MSE 更低 `0.090733`，但全量 adversarial accuracy 只有 `0.83984375`，说明单纯 MSE 门槛在本次 `n512` 上不是 robust acc 最优。
  - `js_only` 和 `js_margin` 平均 rank 约 `12`，但 MSE 约 `0.28`，重建保真明显不足。
  - `EXP-007` 的 full score 迁移到本次 `r5-40 step5 n512` 后全量 adversarial accuracy 只有 `0.828125`，低于 `threshold/js_mse/mse_only`，不能继续视为稳健候选。
  - 与 `EXP-009` 固定 rank 对照相比，`threshold` 全量 adversarial accuracy `0.84765625` 高于固定 rank 最佳 `rank20=0.837890625`；`js_mse` 也高于固定 rank 最佳，同时平均 rank 低于 `20`。
  - 已追加固定 `rank15/20/25/30/35/40` 与 `threshold/js_mse/mse_only` 的统一对照图；该图同时展示 clean/adv accuracy、mean rank 与 mean MSE，便于直接观察 robust acc 与保真/成本 trade-off。
- **问题：**
  - `threshold` 的 best margin threshold 为负值，实际 margin 条件很弱；该 selector 更接近 JS+MSE 的宽松联合阈值。
  - `margin_only` 的 holdout adv/clean acc 最好，但依赖高 rank，不能证明 margin 是有效早停信号。
  - 本实验仍是 full-sweep logits 的离线 replay，没有重新生成真实 selected purified `.pth`，也没有接入 `purify.py` 在线早停。
  - 仍只覆盖 `thubenchmark fold0 seed42 sample_num=512 eps=0.03`，需要跨 seed/fold 或至少另一个攻击强度确认。
- **结论：**
  - `EXP-010` 改写了 `EXP-008` 的初步判断：`mse_only` 仍是有用的保真控制，但在 `n512` 上不是 robust acc 最优。
  - 当前更值得继续在线化或复验的候选是 `threshold` 和 `js_mse`；其中 `threshold` 全量 adv acc 最高，`js_mse` 在接近的 adv acc 下平均 rank 更低。
  - `EXP-007` full score 不应作为默认实现目标；其跨样本量/跨 rank-grid 迁移效果较差。
  - `margin_only` 可作为高 rank 保真/clean acc 对照，但不适合作为 adaptive rank-selection 主规则。
- **下一步：**
  - 固定 rank `15/20/25/30/35/40` 与 `threshold/js_mse/mse_only` 的统一对照图已追加完成。
  - 若要推进在线实现，优先做 `threshold` 和 `js_mse` 的显式配置开关，不改 baseline 默认行为。
  - 在线实现前建议先补一个跨 seed/fold 或不同 `eps` 的 replay，确认 `threshold/js_mse` 优势不是本次 split 偶然结果。

### EXP-011：预测熵 rank-selection Optuna 调参与 ablation（r5-40 step5 n512）

- **日期：** 2026-06-04
- **状态：** 已完成
- **相关 idea：** `IDEA-004`
- **目的：**
  - 在 rank-growth full-sweep 轨迹中补充预测熵 `H(p_r) = -sum_c p_r(c) log p_r(c)`，用于衡量不同 rank 下分类器输出不确定性。
  - 复用 `EXP-009` 的 `n512, ranks=5..40 step5` full-sweep logits，不重新运行长时间 TN 净化。
  - 评估熵单指标、熵与 JS/MSE/margin 的组合，以及 `score_entropy` 是否优于 `EXP-010` 中的 `threshold/js_mse/mse_only` 强候选。
- **代码版本 / 分支：**
  - 分支：`main`
  - commit：`e99016a`
  - working tree：新增 rank-growth entropy 统计与 Optuna 熵 selector；未修改 `purify.py` 或 baseline 在线早停逻辑。
- **输入：**
  - full-sweep 输入目录：`tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512/`
  - 输入轨迹文件：`rank_growth_predictions.pt`
  - 数据集/划分：`thubenchmark fold0 seed42`
  - 攻击：`autoattack eps=0.03`
  - 模型来源：`consistancy_rank25-30_n512_eps0p03`
- **实现变更：**
  - `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py`
    - rank-growth `history_rows` 与新生成的 `.pt` trace 记录 `entropy`。
    - `rank_growth_history.csv` 新增 `entropy`。
    - `rank_growth_summary.csv` 新增 `entropy_mean`、`entropy_std`。
    - rank-growth plots 新增 `entropy_by_rank.png` 和 clean/adv entropy trajectory heatmap。
  - `tensor_ring_rank_analysis/optuna_rank_growth_selection.py`
    - 从 `eval_records[*].logits` 动态计算 entropy，因此可直接兼容旧 `EXP-009` 的 `.pt`。
    - 新增 selector：`entropy_only`、`js_entropy`、`mse_entropy`、`entropy_margin`、`js_mse_entropy`、`score_entropy`。
    - 旧 selector 语义保持不变；输出 CSV 增加 `selected_entropy`、`mean_entropy`、`entropy_threshold`、`eta`。
- **验证：**
  - 语法检查通过：
    ```bash
    conda run -n torch --no-capture-output python -m py_compile \
      tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
      tensor_ring_rank_analysis/optuna_rank_growth_selection.py
    ```
  - rank-growth entropy smoke 通过：
    - 输出目录：`tensor_ring_rank_analysis/results_smoke_exp011_entropy_rank_growth_20260604_131248/`
    - 设置：`sample_num=1`，`rank_growth_ranks=5,10`，`rank_growth_steps_per_rank=1`，`--no_visualize`
    - 验证：`rank_growth_history.csv` 包含 `entropy`，`rank_growth_summary.csv` 包含 `entropy_mean/entropy_std`。
  - 追加 trace smoke 通过：
    - 输出目录：`tensor_ring_rank_analysis/results_smoke_exp011_entropy_rank_growth_trace_20260604_131749/`
    - 验证：`rank_growth_predictions.pt` 的 `traces[*].eval_records[*]` 和 `history_rows` 均包含 `entropy`。
  - Optuna entropy selector smoke 通过：
    - 输出目录：`tensor_ring_rank_analysis/results_smoke_exp011_entropy_optuna_20260604_131352/`
    - 设置：`max_samples=8`，每个熵 selector `n_trials=2`
    - 验证：`selected_rows.csv` 包含 `selected_entropy`，`summary.csv` 和 `fixed_rank_selector_comparison.csv` 包含 `mean_entropy`，`mode_best_summary.csv` 包含 `entropy_threshold/eta`。
- **正式命令：**
  ```bash
  conda run -n torch --no-capture-output python -u \
    tensor_ring_rank_analysis/optuna_rank_growth_selection.py \
    --input_dir tensor_ring_rank_analysis/results_rank_growth_exp009_hf_full_sweep_r5-40_step5_n512 \
    --output_dir tensor_ring_rank_analysis/results_rank_growth_exp011_entropy_optuna_selection_r5-40_step5_n512 \
    --study_name exp011_entropy_optuna_selection_r5_40_step5_n512 \
    --n_trials 300 \
    --seed 42 \
    --tune_ratio 0.5 \
    --selection_modes threshold,score,js_only,mse_only,margin_only,js_mse,mse_margin,js_margin,entropy_only,js_entropy,mse_entropy,entropy_margin,js_mse_entropy,score_entropy \
    --fixed_rank_baselines 15,20,25,30,35,40 \
    --comparison_selectors threshold,js_mse,mse_only,entropy_only,js_mse_entropy,score_entropy \
    --objective robust_priority \
    --clean_weight 0.20 \
    --mse_weight 0.10 \
    --rank_weight 0.02 \
    --plot_format png \
    --plot_dpi 180
  ```
- **实际运行：**
  - 日志：`logs/exp011_entropy_optuna_selection_r5-40_step5_n512_20260604_131428.log`
  - 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp011_entropy_optuna_selection_r5-40_step5_n512/`
  - 退出码：`0`
  - 输出文件：
    - `best_config.json`
    - `trials.csv`
    - `selected_rows.csv`
    - `summary.csv`
    - `mode_best_summary.csv`
    - `fixed_rank_selector_comparison.csv`
    - `meta.json`
    - `plots/`
- **结果：**
  - tune objective 全局最佳 selector 为 `js_mse_entropy`：
    - `js_threshold=0.059462126`
    - `mse_threshold=0.143013170`
    - `entropy_threshold=2.512311413`
    - tune objective `1.019322397`
  - 关键 selector 摘要：

    | selector | holdout adv acc | holdout clean acc | holdout MSE | holdout entropy | holdout rank | all adv acc | all clean acc | all MSE | all entropy | all rank |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `threshold` | 0.8359375 | 0.87890625 | 0.113238 | 0.449888 | 19.2578125 | 0.84765625 | 0.884765625 | 0.113463 | 0.457286 | 19.433594 |
    | `js_mse` | 0.8359375 | 0.890625 | 0.130659 | 0.456377 | 17.373047 | 0.845703125 | 0.890625 | 0.129503 | 0.454352 | 17.670898 |
    | `mse_only` | 0.8359375 | 0.87890625 | 0.091114 | 0.438933 | 22.128906 | 0.83984375 | 0.876953125 | 0.090733 | 0.441444 | 22.348633 |
    | `entropy_only` | 0.84765625 | 0.91796875 | 0.041009 | 0.339794 | 38.134766 | 0.8359375 | 0.91015625 | 0.042455 | 0.339537 | 38.178711 |
    | `js_entropy` | 0.83203125 | 0.89453125 | 0.280244 | 0.467851 | 11.884766 | 0.841796875 | 0.888671875 | 0.283484 | 0.455335 | 11.904297 |
    | `mse_entropy` | 0.81640625 | 0.859375 | 0.118553 | 0.466234 | 18.632812 | 0.83203125 | 0.86328125 | 0.119509 | 0.468028 | 18.666992 |
    | `entropy_margin` | 0.84765625 | 0.91796875 | 0.066027 | 0.339796 | 35.244141 | 0.8359375 | 0.91015625 | 0.067247 | 0.339540 | 35.463867 |
    | `js_mse_entropy` | 0.828125 | 0.875 | 0.114400 | 0.448669 | 19.150391 | 0.845703125 | 0.876953125 | 0.115024 | 0.454288 | 19.228516 |
    | `score_entropy` | 0.8359375 | 0.89453125 | 0.115518 | 0.359093 | 19.257812 | 0.845703125 | 0.890625 | 0.116645 | 0.350927 | 19.350586 |

  - 关键 rank 分布：
    - `threshold`：`{10:34, 15:366, 20:359, 25:222, 30:36, 35:4, 40:3}`
    - `js_mse`：`{10:79, 15:484, 20:329, 25:104, 30:23, 35:5}`
    - `mse_only`：`{10:2, 15:168, 20:385, 25:301, 30:152, 35:16}`
    - `entropy_only`：`{5:17, 10:22, 15:8, 20:8, 25:11, 30:7, 35:3, 40:948}`
    - `js_mse_entropy`：`{10:37, 15:377, 20:360, 25:218, 30:26, 35:1, 40:5}`
    - `score_entropy`：`{10:13, 15:425, 20:377, 25:133, 30:48, 35:23, 40:5}`
- **观察：**
  - `js_mse_entropy` 在 tune objective 上最高，但 holdout adversarial accuracy 只有 `0.828125`，低于 `threshold/js_mse/mse_only` 的 `0.8359375`；全量 adversarial accuracy 为 `0.845703125`，与 `js_mse`、`score_entropy` 持平，低于 `threshold` 的 `0.84765625`。
  - `entropy_only` 和 `entropy_margin` 的 holdout adv/clean acc 都较高，但平均 rank 分别达到 `38.13` 和 `35.24`；`entropy_only` 在 `1024` 个 clean/adv rows 中有 `948` 个选择 `rank40`，更像高 rank/低熵保真对照，不是低成本 adaptive 早停规则。
  - `score_entropy` 相比普通 `score` 降低了 selected entropy，并在 all split 上达到 `0.845703125` adv acc，但没有超过 `threshold`；其 all mean rank `19.35` 也高于 `js_mse` 的 `17.67`。
  - `js_entropy` 平均 rank 约 `11.90`，但 MSE 高到约 `0.283`，说明单靠 JS+entropy 容易过早停止，重建保真不足。
  - `mse_entropy` 的 all adv acc 为 `0.83203125`，低于 `mse_only`，说明在当前搜索空间中加入 entropy 阈值反而收紧了有效样本选择。
- **结论：**
  - `IDEA-004` 的实现和离线调参流程可运行，预测熵可作为 rank-growth 不确定性诊断字段。
  - 当前 `thubenchmark fold0 seed42 eps0.03 n512` 结果不支持把预测熵作为主 rank-selection 信号。
  - 低熵本身倾向选择高 rank，因为高 rank 常带来更低熵和更低 MSE；这会退化成高 rank 保真策略，而不是低成本自适应 rank selection。
  - 后续主候选仍应沿用 `EXP-010` 的 `threshold` 与 `js_mse`；entropy 可保留为辅助诊断或 `score_entropy` 的可选特征，但不应优先在线化。
- **剩余问题：**
  - 本实验仍是 full-sweep logits 的离线 replay，没有重新生成 selected purified `.pth`，也没有通过在线 `purify.py` 验证。
  - 只覆盖 `thubenchmark fold0 seed42 eps=0.03`；若未来继续研究 entropy，应先做跨 seed/fold 或不同 eps 的 replay，而不是直接写入 baseline。

### EXP-012：eps=0.1 rank-growth full pipeline 复验

- **日期：** 2026-06-04
- **状态：** 已完成
- **相关 idea：** `IDEA-002`、`IDEA-003`、`IDEA-004`
- **目的：**
  - 将 `EXP-009/010/011` 的 rank-growth full-sweep、paired-delta/bootstrap、Optuna rank-selection pipeline 扩展到更强攻击 `eps=0.1`。
  - 重点观察 `eps=0.1` 下 `threshold/js_mse`、entropy 相关 selector、固定 rank baseline 的 robust/clean trade-off 是否与 `eps=0.03` 一致。
  - 使用 `analyze_rank_growth_pair_delta.py` 复查高频增量与 true-label margin 的 paired 差值，用 `optuna_rank_growth_selection.py` 做离线 rank selector 调参。
- **输入与前置检查：**
  - 当前仓库没有 eps=0.1 的 `rank_growth_predictions.pt`，因此先运行 `analyze_tr_rank_predictions.py` 生成 full-sweep 输入，再运行用户指定的两个离线脚本。
  - eps=0.1 checkpoint 已存在：
    `checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.1_42_fold0_consistancy_rank25-30_n512_eps0p1_best.pth`
  - eps=0.1 adversarial data 已存在：
    `ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p1_madry_autoattack_eps0.1_seed42_fold0.pth`
  - smoke 已通过：
    - 输出目录：`tensor_ring_rank_analysis/results_smoke_exp012_eps0p1_rank_growth_20260604_133153/`
    - 设置：`sample_num=1`，`rank_growth_ranks=5,10`，`rank_growth_steps_per_rank=1`
- **关键设置：**
  - 数据集/划分：`thubenchmark fold0 seed42`
  - 攻击：`autoattack eps=0.1`
  - 模型来源：`consistancy_rank25-30_n512_eps0p1`
  - rank 序列：`5,10,15,20,25,30,35,40`
  - 样本数：`512`
  - full-sweep：开启；禁用 early stopping，完整评估所有 rank。
  - 频域增量指标：开启；`high_freq_cutoff_hz=30.0`，`freq_energy_floor_hz=1.0`
  - bootstrap：`5000` 次，`95% CI`
  - Optuna：每个 selector `300` trials，`tune_ratio=0.5`，`seed=42`
- **输出目录与日志：**
  - full-sweep 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp012_hf_full_sweep_eps0p1_r5-40_step5_n512/`
  - Optuna 输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp012_optuna_selection_eps0p1_r5-40_step5_n512/`
  - 后台 pipeline PID：`74113`
  - pipeline 日志：`logs/exp012_pipeline_eps0p1_r5-40_step5_n512_20260604_133222.log`
  - full-sweep 日志：`logs/exp012_rank_growth_hf_full_sweep_eps0p1_r5-40_step5_n512_20260604_133222.log`
  - paired-delta 日志：`logs/exp012_pair_delta_eps0p1_r5-40_step5_n512_20260604_133222.log`
  - Optuna 日志：`logs/exp012_optuna_selection_eps0p1_r5-40_step5_n512_20260604_133222.log`
  - 启动备注：沙箱内 `setsid nohup` 未保活且未生成日志；随后使用沙箱外提权启动，pipeline 日志确认已进入 full-sweep 阶段。
  - 完成时间：`2026-06-04T15:37:21+08:00`，pipeline 日志打印 `EXP-012 pipeline complete`。
- **命令：**
  ```bash
  FULL_OUT="tensor_ring_rank_analysis/results_rank_growth_exp012_hf_full_sweep_eps0p1_r5-40_step5_n512"
  OPTUNA_OUT="tensor_ring_rank_analysis/results_rank_growth_exp012_optuna_selection_eps0p1_r5-40_step5_n512"

  conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --rank_growth_full_sweep \
    --rank_growth_ranks 5,10,15,20,25,30,35,40 \
    --dataset thubenchmark \
    --model eegnet \
    --no_ea \
    --eps 0.1 \
    --fold 0 \
    --attack autoattack \
    --at_strategy madry \
    --consistency_version consistancy \
    --consistency_tag consistancy_rank25-30_n512_eps0p1 \
    --adv_model_tag consistancy_rank25-30_n512_eps0p1 \
    --config PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml \
    --checkpoint_path checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.1_42_fold0_consistancy_rank25-30_n512_eps0p1_best.pth \
    --ad_data_path ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p1_madry_autoattack_eps0.1_seed42_fold0.pth \
    --sample_num 512 \
    --enable_incremental_frequency_metrics \
    --high_freq_cutoff_hz 30.0 \
    --freq_energy_floor_hz 1.0 \
    --plot_format png \
    --plot_dpi 180 \
    --output_dir "${FULL_OUT}"

  conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_rank_growth_pair_delta.py \
    --input_dir "${FULL_OUT}" \
    --output_prefix rank_growth_pair_delta_bootstrap \
    --bootstrap_iters 5000 \
    --ci 95 \
    --seed 42 \
    --plot_format png \
    --plot_dpi 180

  conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/optuna_rank_growth_selection.py \
    --input_dir "${FULL_OUT}" \
    --output_dir "${OPTUNA_OUT}" \
    --study_name exp012_optuna_selection_eps0p1_r5_40_step5_n512 \
    --n_trials 300 \
    --seed 42 \
    --tune_ratio 0.5 \
    --selection_modes threshold,score,js_only,mse_only,margin_only,js_mse,mse_margin,js_margin,entropy_only,js_entropy,mse_entropy,entropy_margin,js_mse_entropy,score_entropy \
    --fixed_rank_baselines 15,20,25,30,35,40 \
    --comparison_selectors threshold,js_mse,mse_only,entropy_only,js_mse_entropy,score_entropy \
    --objective robust_priority \
    --clean_weight 0.20 \
    --mse_weight 0.10 \
    --rank_weight 0.02 \
    --plot_format png \
    --plot_dpi 180
  ```
- **结果：**
  - full-sweep、paired-delta/bootstrap、Optuna 三个阶段均已完成，核心产物已落盘：
    - `rank_growth_predictions.pt`
    - `rank_growth_summary.csv`
    - `rank_growth_pair_delta_bootstrap_summary.csv`
    - `summary.csv`
    - `mode_best_summary.csv`
    - `fixed_rank_selector_comparison.csv`
  - 固定 rank 对照（全量 clean/adv rows）：

    | method | clean acc | adv acc | all acc | mean MSE | mean entropy |
    |---|---:|---:|---:|---:|---:|
    | rank15 | 0.816406 | 0.796875 | 0.806641 | 0.155501 | 0.941930 |
    | rank20 | 0.843750 | 0.787109 | 0.815430 | 0.109655 | 0.887871 |
    | rank25 | 0.841797 | 0.771484 | 0.806641 | 0.077825 | 0.864005 |
    | rank30 | 0.863281 | 0.755859 | 0.809570 | 0.055167 | 0.847012 |
    | rank35 | 0.871094 | 0.724609 | 0.797852 | 0.039663 | 0.824700 |
    | rank40 | 0.888672 | 0.712891 | 0.800781 | 0.028898 | 0.807164 |

  - 主要 Optuna selector（全量 clean/adv rows）：

    | selector | clean acc | adv acc | all acc | mean rank | mean MSE | mean entropy | objective |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | threshold | 0.832031 | 0.787109 | 0.809570 | 18.242188 | 0.136675 | 0.961077 | 0.932281 |
    | score | 0.830078 | 0.798828 | 0.814453 | 15.317383 | 0.151396 | 0.929407 | 0.943809 |
    | js_mse | 0.828125 | 0.796875 | 0.812500 | 15.292969 | 0.153875 | 0.944155 | 0.941231 |
    | entropy_only | 0.871094 | 0.728516 | 0.799805 | 18.886719 | 0.242039 | 0.891775 | 0.870595 |
    | js_mse_entropy | 0.833984 | 0.796875 | 0.815430 | 15.434570 | 0.154187 | 0.940915 | 0.942291 |
    | score_entropy | 0.837891 | 0.794922 | 0.816406 | 16.074219 | 0.146392 | 0.840620 | 0.941533 |

  - Optuna 全局最优按 tune objective 是 `score_entropy`，best trial `131`，tune objective `0.945337`；但其全量 adversarial accuracy 为 `0.794922`，没有超过 `score`、`js_mse`、`js_mse_entropy` 和固定 `rank15` 的 `0.796875~0.798828` 区间。
  - paired bootstrap 显示 `25->30`、`30->35`、`35->40` 的 `delta_hf_ratio_mean` 95% CI 全为正，分别为 `[0.000510, 0.002443]`、`[0.000125, 0.001926]`、`[0.001095, 0.002804]`；同时这些 rank 段 `delta_margin_mean` 均为负，提示更高 rank 的新增高频能量与 true-label margin 下降同时出现。
  - `entropy_only` 在 `eps=0.1` 下仍不是稳健主策略：clean acc 较高（`0.871094`），但 adv acc 降到 `0.728516`，更像偏高 rank/保真侧的诊断信号。
- **结论：**
  - `eps=0.1` 下，`threshold/js_mse` 不再复现 `eps=0.03` 上的明显优势；`score`、`js_mse`、`js_mse_entropy`、固定 `rank15` 的 robust accuracy 差距很小。
  - 高 rank baseline 明显提升 clean acc、降低 MSE 和 entropy，但 robust acc 随 rank 增大下降；在线 rank-selection 不能简单追求高 rank 保真。
  - 暂不把任何 selector 写入默认在线早停逻辑；下一步应做跨 seed/fold 或不同攻击强度的离线 replay，再决定是否实现显式配置开关。

### EXP-013：eps=0.1 js_mse 在线早停计算开销复验

- **日期：** 2026-06-04
- **状态：** 已完成
- **相关 idea：** `IDEA-003`
- **目的：**
  - 最后一次复验 adaptive rank：不再追求更高 robust accuracy，而是检查 `js_mse` 在线早停是否能显著降低计算时间开销。
  - 使用 `EXP-012` 在 `eps=0.1` full-sweep 轨迹上得到的 `js_mse` 最优阈值，直接跑真实在线早停。
  - 与 `EXP-012` full-sweep 运行时间做粗略对比：`EXP-012` full-sweep 日志从 `2026-06-04 13:34:29` 到 pipeline 记录 `15:37:01`，约 `2h02m32s`，其中 clean/adv 都完整评估到 `rank40`。
- **关键设置：**
  - 数据集/划分：`thubenchmark fold0 seed42`
  - 攻击：`autoattack eps=0.1`
  - 模型来源：`consistancy_rank25-30_n512_eps0p1`
  - rank 序列：`5,10,15,20,25,30,35,40`
  - 样本数：`512`
  - 早停策略：在线 `js_mse`
  - 阈值来源：`EXP-012` 的 `js_mse` Optuna best trial
    - `rank_growth_js_threshold=0.036768744516209435`
    - `rank_growth_max_mse_to_input=0.19581095691938585`
  - 早停语义：当相邻 rank 的 top1 不变、`JS(rank_prev, rank_next) <= threshold`，且 `rank_prev` 的 `MSE <= threshold` 时，选择 `rank_prev` 并停止；这与离线 `js_mse` replay 的语义一致。
- **验证：**
  - smoke 已通过：
    - 输出目录：`tensor_ring_rank_analysis/results_smoke_exp013_js_mse_early_stop_eps0p1_20260604_155842/`
    - 设置：`sample_num=1`，`--no_visualize`
    - `meta.json` 确认 `rank_growth_full_sweep=false`，且 JS/MSE 阈值正确写入。
    - smoke 的 clean/adv 单样本均评估到 `rank20` 后选择 `rank15`。
- **输出目录与日志：**
  - 正式输出目录：`tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512/`
  - 正式运行方式：`tmux` session `exp013_js_mse_eps0p1`
  - 正式启动时间：`2026-06-04T16:02:10+08:00`
  - 正式日志：`logs/exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512_20260604_160139.log`
  - time 日志：`logs/exp013_js_mse_early_stop_time_eps0p1_r5-40_step5_n512_20260604_160139.txt`
  - 启动备注：普通后台启动和 `setsid nohup` 在当前环境下未保活；最终使用 `tmux` 启动成功。
  - 完成时间：`2026-06-04T17:11:21+08:00`，日志结尾显示 `status=0`。
- **命令：**
  ```bash
  OUT="tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512"
  LOG="logs/exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512_20260604_160139.log"
  TIME_LOG="logs/exp013_js_mse_early_stop_time_eps0p1_r5-40_step5_n512_20260604_160139.txt"

  tmux new-session -d -s exp013_js_mse_eps0p1 bash -lc '
  cd /home/yhj/pythonProject/EEGAP
  exec > "'"${LOG}"'" 2>&1
  echo "[$(date -Is)] EXP-013 js_mse early-stop start"
  /usr/bin/time -v -o "'"${TIME_LOG}"'" conda run -n torch --no-capture-output python -u \
    tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_growth \
    --rank_growth_ranks 5,10,15,20,25,30,35,40 \
    --rank_growth_js_threshold 0.036768744516209435 \
    --rank_growth_max_mse_to_input 0.19581095691938585 \
    --dataset thubenchmark \
    --model eegnet \
    --no_ea \
    --eps 0.1 \
    --fold 0 \
    --attack autoattack \
    --at_strategy madry \
    --consistency_version consistancy \
    --consistency_tag consistancy_rank25-30_n512_eps0p1 \
    --adv_model_tag consistancy_rank25-30_n512_eps0p1 \
    --config PTR3d_rank_growth_8_2048_r5-40_3d_interpolate.yaml \
    --checkpoint_path checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.1_42_fold0_consistancy_rank25-30_n512_eps0p1_best.pth \
    --ad_data_path ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p1_madry_autoattack_eps0.1_seed42_fold0.pth \
    --sample_num 512 \
    --output_dir "'"${OUT}"'" \
    --plot_format png \
    --plot_dpi 180
  STATUS=$?
  echo "[$(date -Is)] EXP-013 js_mse early-stop finished status=${STATUS}"
  exit ${STATUS}
  '
  ```
- **指标：**
  - 总墙钟时间：读取 `TIME_LOG` 的 `Elapsed (wall clock) time`。
  - 与 `EXP-012` full-sweep 约 `2h02m32s` 的时间比值。
  - `rank_growth_summary.csv` 中 clean/adv 的 `selected_count`、`evaluated_count` 和 `mean rank`。
  - selected rows 的 clean/adv accuracy、mean MSE、mean entropy，用于确认时间下降是否伴随明显性能损失。
- **结果：**
  - 任务已完成，退出状态 `0`。
  - `/usr/bin/time -v` 记录的墙钟时间为 `1:09:11`；相对 `EXP-012` full-sweep 约 `2:02:32`，耗时约为 `56.5%`，粗略加速约 `1.77x`。
  - 输出文件已生成：
    - `tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512/rank_growth_history.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512/rank_growth_summary.csv`
    - `tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512/rank_growth_predictions.pt`
    - `tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512/meta.json`
    - `tensor_ring_rank_analysis/results_rank_growth_exp013_js_mse_early_stop_eps0p1_r5-40_step5_n512/plots/`
  - selected rows 汇总：

    | source | count | acc | mean rank | mean MSE | mean entropy |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | clean | 512 | 0.837890625 | 15.791016 | 0.149130628 | 0.919332704 |
    | adv | 512 | 0.781250000 | 15.830078 | 0.150320173 | 0.960024727 |
    | all | 1024 | 0.809570312 | 15.810547 | 0.149725401 | 0.939678715 |

  - selected rank 分布：

    | rank | clean count | adv count |
    | ---: | ---: | ---: |
    | 10 | 96 | 93 |
    | 15 | 274 | 274 |
    | 20 | 116 | 121 |
    | 25 | 19 | 17 |
    | 30 | 6 | 6 |
    | 40 | 1 | 1 |

  - 平均每个样本评估 rank 数：clean `4.156250`，adv `4.164062`；相比 full-sweep 每个样本固定 `8` 个 rank block，评估块数量大约减半。
- **观察：**
  - 在线 `js_mse` 早停确实显著降低了计算量，selected rank 主要集中在 `rank10/15/20`，极少数样本评估到 `rank40`。
  - 与 `EXP-012` 离线 replay 相比，本实验全量 adv acc 为 `0.78125`，低于 `EXP-012` 中 `score=0.798828`、`js_mse=0.796875`、`js_mse_entropy=0.796875` 和固定 `rank15=0.796875`。
  - 当前在线早停语义选择的是满足稳定条件时的前一档 rank；这有利于减少计算和 rank，但在 `eps=0.1` 下会牺牲一部分 robust accuracy。
- **问题：**
  - 本实验只覆盖 `thubenchmark fold0 seed42 eps=0.1 sample_num=512`。
  - 结果来自真实在线早停轨迹，不能直接与 full-sweep 离线 selector 完全等价；二者在优化路径和停止时机上仍可能有差异。
- **结论：**
  - `js_mse` 在线早停可以作为降低计算成本的候选策略，但在当前 `eps=0.1` 设置下不是 robust accuracy 最优方案。
  - 暂不建议把该在线早停写入默认净化逻辑；若后续使用，应通过显式配置开关暴露，并把它定位为 speed/compute-oriented 选项。
- **下一步：**
  - 若继续推进在线 adaptive rank，应优先比较 `score`、`js_mse_entropy` 或更保守的 rank 选择语义，而不是只复用当前 `js_mse` 早停。
  - 在跨 seed/fold 或不同 eps 的离线 replay 完成前，不修改 baseline 默认行为。

### EXP-014：PTR_3d_rank_soft_mask 在 n512 eps=0.03 上的 rank penalty sweep

- **日期：** 2026-06-05
- **状态：** 已完成
- **相关 idea：** `IDEA-005`
- **目的：**
  - 验证 `PTR_3d_rank_soft_mask` 是否能在单次优化中学习样本级 effective rank。
  - 观察 `rank_soft_mask_weight` 对 clean/adv accuracy、MSE 与 effective rank 的影响。
  - 与 `EXP-009/010/011` 的固定 rank、`threshold`、`js_mse` 和 entropy selector 结果对齐比较。
- **代码版本 / 分支：**
  - 分支：`main`
  - working tree：包含 `IDEA-005` 相关实现。
- **配置文件：**
  - `configs/thubenchmark/PTR3d_rank_soft_mask_8_2048_r40_3d_interpolate.yaml`
- **命令：**
  ```bash
  # smoke：sample_num=1，不作为正式结果
  conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
    --analysis_mode rank_soft_mask \
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
    --config PTR3d_rank_soft_mask_8_2048_r40_3d_interpolate.yaml \
    --checkpoint_path checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
    --ad_data_path ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
    --sample_num 1 \
    --rank_soft_mask_init_rank 15 \
    --rank_soft_mask_temperature 1.0 \
    --rank_soft_mask_weight 0.003 \
    --output_dir tensor_ring_rank_analysis/results_smoke_exp014_rank_soft_mask_n1 \
    --no_visualize \
    --overwrite

  # 正式 512 样本 sweep：每个 lambda 单独后台运行并写入稳定日志
  for LAMBDA in 0.0 0.001 0.003 0.01; do
    TAG=$(printf "%s" "${LAMBDA}" | sed 's/\./p/g')
    OUT_DIR="tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda${TAG}"
    LOG_FILE="logs/exp014_rank_soft_mask_eps0p03_n512_lambda${TAG}_$(date +%Y%m%d_%H%M%S).log"
    setsid nohup conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
      --analysis_mode rank_soft_mask \
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
      --config PTR3d_rank_soft_mask_8_2048_r40_3d_interpolate.yaml \
      --checkpoint_path checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_consistancy_rank25-30_n512_eps0p03_best.pth \
      --ad_data_path ad_data/thubenchmark_eegnet_no_ea_consistancy_rank25-30_n512_eps0p03_madry_autoattack_eps0.03_seed42_fold0.pth \
      --sample_num 512 \
      --rank_soft_mask_init_rank 15 \
      --rank_soft_mask_temperature 1.0 \
      --rank_soft_mask_weight "${LAMBDA}" \
      --output_dir "${OUT_DIR}" \
      --plot_format png \
      --plot_dpi 180 \
      > "${LOG_FILE}" 2>&1 &
  done
  ```
- **关键设置：**
  - 数据集/划分：`thubenchmark fold0 seed42`
  - 攻击：`autoattack eps=0.03`
  - 模型来源：`consistancy_rank25-30_n512_eps0p03`
  - 样本数：`512`
  - soft-rank 初始值：`rank_soft_mask_init_rank=15`
  - temperature：`1.0`
  - sweep：`rank_soft_mask_weight=0.0/0.001/0.003/0.01`
  - 输出目录：`tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda*/`
- **正式任务启动记录：**
  - 启动时间：`2026-06-05 17:07:37`
  - 启动方式：`setsid nohup conda run -n torch --no-capture-output python -u ... > LOG_FILE 2>&1 &`
  - `lambda=0.0`：PID `415401`，输出目录 `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p0/`，日志 `logs/exp014_rank_soft_mask_eps0p03_n512_lambda0p0_20260605_170737.log`
  - `lambda=0.001`：PID `415402`，输出目录 `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p001/`，日志 `logs/exp014_rank_soft_mask_eps0p03_n512_lambda0p001_20260605_170737.log`
  - `lambda=0.003`：PID `415403`，输出目录 `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p003/`，日志 `logs/exp014_rank_soft_mask_eps0p03_n512_lambda0p003_20260605_170737.log`
  - `lambda=0.01`：PID `415404`，输出目录 `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p01/`，日志 `logs/exp014_rank_soft_mask_eps0p03_n512_lambda0p01_20260605_170737.log`
  - 启动检查：宿主 `ps` 确认 4 个进程运行中，日志已开始实时写入。
  - 完成检查：4 个日志均记录 `Saved rank-soft-mask analysis results`，未发现 `Traceback`、`ERROR`、`RuntimeError` 或 `CUDA out of memory`；4 个 PID 均已结束。
- **实现验证：**
  - `py_compile` 已通过：
    ```bash
    conda run -n torch --no-capture-output python -m py_compile \
      TN/rank_growth/PTR_3d_rank_soft_mask.py \
      tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
      purify.py TN/utils.py
    ```
  - `sample_num=1` smoke 已通过，输出目录：`tensor_ring_rank_analysis/results_smoke_exp014_rank_soft_mask_n1/`。
  - smoke 输出确认 `rank_soft_mask_rows.csv` 含 `effective_rank`、`rho`、`rank_cost`、`rank_soft_mask_weight`，`meta.json` 记录 `analysis_mode=rank_soft_mask`。
  - 该 smoke 只用于验证代码路径和输出格式，不作为正式实验结论。
- **指标：**
  - 主指标：clean accuracy、adversarial accuracy。
  - 次指标：MSE、confidence、entropy、effective rank、rho、rank cost、近似有效参数量、单样本训练时间。
- **结果：**
  - 正式输出：
    - `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p0/`
    - `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p001/`
    - `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p003/`
    - `tensor_ring_rank_analysis/results_rank_soft_mask_exp014_eps0p03_n512_lambda0p01/`
  - 每个输出目录均包含 `rank_soft_mask_predictions.pt`、`rank_soft_mask_rows.csv`、`rank_soft_mask_summary.csv`、`meta.json` 和 `plots/effective_rank_distribution.png`。
  - sweep summary：

    | rank penalty | clean acc | adv acc | all acc | clean MSE | adv MSE | mean effective rank | mean entropy |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `0.0` | `0.900391` | `0.832031` | `0.866211` | `0.058365` | `0.058537` | `19.536` | `0.400834` |
    | `0.001` | `0.886719` | `0.830078` | `0.858398` | `0.059127` | `0.059350` | `19.330` | `0.400110` |
    | `0.003` | `0.894531` | `0.828125` | `0.861328` | `0.060689` | `0.060897` | `18.939` | `0.402222` |
    | `0.01` | `0.896484` | `0.837891` | `0.867188` | `0.065804` | `0.065905` | `17.715` | `0.408294` |

  - 相对 `lambda=0.0`：
    - `lambda=0.001`：effective rank 降低约 `0.206`，adv acc 低 `1/512`，clean acc 低 `7/512`。
    - `lambda=0.003`：effective rank 降低约 `0.596`，adv acc 低 `2/512`，clean acc 低 `3/512`。
    - `lambda=0.01`：effective rank 降低约 `1.821`，adv acc 高 `3/512`，clean acc 低 `2/512`，但 MSE 增加约 `0.0074`。
- **观察：**
  - rank penalty 确实按预期压低 effective rank：`0.0 -> 0.001 -> 0.003 -> 0.01` 的全量 mean effective rank 为 `19.536 -> 19.330 -> 18.939 -> 17.715`。
  - accuracy 对 rank penalty 非单调。当前 sweep 中 `lambda=0.01` 的 adv acc 最高，为 `0.837891`；`lambda=0.0` 的 clean acc 最高，为 `0.900391`。
  - effective rank 的样本级分布较窄，不像真正学到了强 sample-wise rank allocation：
    - `lambda=0.0` 全量 effective rank 的 `q10/median/q90` 为 `18.887/19.560/20.164`。
    - `lambda=0.01` 全量 effective rank 的 `q10/median/q90` 为 `17.072/17.761/18.261`。
    - clean 与 adv 的 effective rank 均值几乎重合，说明当前 soft gate 没有形成明显 clean/adv 区分。
  - effective rank 与重构 MSE 强相关，相关系数约 `0.78~0.84`；但与分类正确性相关很弱且略为负，说明 gate 主要在追踪重构难度，而不是分类/鲁棒性所需 rank。
  - 与 `EXP-010/011` 对照：
    - `lambda=0.01` 的 adv acc `0.837891` 与固定 `rank20` 持平，但低于 `threshold=0.847656` 和 `js_mse=0.845703`。
    - `lambda=0.01` 的 clean acc `0.896484` 高于 `threshold=0.884766` 和 `js_mse=0.890625`，但低于固定 `rank35=0.916016` 和 `rank40=0.910156`。
    - soft-mask 的 MSE 明显低于 `threshold/js_mse` selector，但这没有转化为更高 robust accuracy。
- **问题：**
  - 初版目标 `MSE + rank cost` 不足以学习有判别意义的动态 rank；它更像一个可微的全局 rank/重构难度调节器。
  - `effective_rank` 的数值不能直接等同于 hard fixed rank，因为模型仍使用 `max_rank=40` 的完整参数容器和 soft mask，参数共享/表达方式与 hard slice 不完全一致。
  - 当前只跑了 `fold0 seed42 eps=0.03`，未做跨 seed/fold 复验；但本次主趋势已经足以说明该目标不是当前最强候选。
- **结论：**
  - `PTR_3d_rank_soft_mask` 的工程路径可行，输出完整，rank penalty 可控。
  - 但核心假设只得到弱支持：它能降低平均 effective rank，却没有学出足够清晰的样本级最小充分 rank，也没有超过 hard rank-growth selector 的 robust accuracy。
  - 当前不建议把 `MSE + rank cost` 的 soft-mask 版本作为 accuracy-oriented 主线或替代 `threshold/js_mse`；它更适合作为后续可微 rank gate 的原型和诊断工具。
- **下一步：**
  - 若继续推进 soft-mask，应优先改优化目标，而不是只调 `rank_soft_mask_weight`：
    - 引入能刻画 rate-distortion knee/边际收益的目标或约束，避免只用全局线性 rank cost。
    - 设计让 `rho` 真正 sample-adaptive 的机制，例如基于中间重构统计动态调节 penalty 或 temperature。
    - 若仍追求 robust accuracy，需要引入与分类边界或净化稳定性相关的信号；仅 MSE 不够。
  - 在论文主线选择上，当前结果更支持继续围绕 hard rank-growth selector 的 `threshold/js_mse` 和更可解释的在线选择机制推进。

### EXP-015：PTR_3d_rank_growth + JS_MSE 跨 dataset/model/seed/fold/eps 完整复验

- **日期：** 2026-06-05
- **状态：** Stopped（用户要求停止，保留已有产物）
- **相关 idea：** `IDEA-006`
- **目的：**
  - 对 `PTR_3d_rank_growth + JS_MSE` 做跨数据集、模型、随机种子、fold 和攻击强度的完整复验。
  - 与普通 AT baseline（Madry/TRADES/FBF）和 ABAT baseline 对齐比较 clean/adv accuracy。
- **关键设置：**
  - 数据集：`thubenchmark`、`seediv`
  - 模型：主方法与普通 AT baseline 使用 `eegnet`、`conformer`；ABAT 使用 `eegnet_ea_forward`、`conformer_ea_forward`
  - seeds：`42,43,45`
  - folds：`0,1,2`
  - EPS：`0.01,0.03,0.05`
  - 攻击：`autoattack`
  - 主方法训练增强：paired `consistancy`，训练集净化 rank 为 `25/30`，`sample_num=512`
  - 主方法测试净化：`PTR_3d_rank_growth`，rank 序列 `5,10,15,20,25,30,35,40`
  - JS_MSE 固定阈值来源：`EXP-010` 的 `eps=0.03` n512 `js_mse` selector
    - `rank_growth_js_threshold=0.0403031319472992`
    - `rank_growth_max_mse_to_input=0.1642765770602721`
- **实现计划：**
  - 新增 EXP-015 专用 rank-growth 配置：
    - `configs/thubenchmark/PTR3d_rank_growth_js_mse_exp015_8_2048_r5-40_3d_interpolate.yaml`
    - `configs/seediv/PTR3d_rank_growth_js_mse_exp015_8_2048_r5-40_3d_interpolate.yaml`
  - 新增矩阵编排脚本：`TN/rank_growth/run_exp015_full_test.sh`
  - 新增结果汇总脚本：`tensor_ring_rank_analysis/summarize_exp015_results.py`
  - 修复训练入口 fold 支持，避免 fold1/fold2 实验误用 fold0。
  - 扩展 ABAT EA-in-forward 路径，使其支持 Conformer。
- **计划命令：**
  ```bash
  # 语法和导入验证
  conda run -n torch --no-capture-output python -m py_compile \
    train_AT.py train_AT_consistancy.py train_AT_ea_forward.py attack.py attack_ea_forward.py \
    models/eegnet_ea_forward.py tensor_ring_rank_analysis/summarize_exp015_results.py

  # dry-run 检查矩阵与日志路径
  DRY_RUN=1 EXP015_SMOKE=1 bash TN/rank_growth/run_exp015_full_test.sh

  # smoke：小样本、1 epoch，不作为正式结果
  EXP015_SMOKE=1 bash TN/rank_growth/run_exp015_full_test.sh

  # full：smoke 通过后用 nohup 后台启动
  RUN_ID=exp015_full_YYYYMMDD_HHMMSS
  mkdir -p "logs/exp015/${RUN_ID}"
  nohup setsid bash -lc "cd /home/yhj/pythonProject/EEGAP && EXP015_RUN_ID=${RUN_ID} bash TN/rank_growth/run_exp015_full_test.sh" \
    > "logs/exp015/${RUN_ID}/controller.log" 2>&1 < /dev/null &
  ```
- **指标：**
  - 主指标：clean accuracy、adversarial accuracy。
  - 主方法额外记录：purified clean accuracy、purified adversarial accuracy、MSE、实际 sample_num。
  - 汇总文件：`logs/exp015/<run_id>/summary.csv`
- **验证：**
  - `py_compile` 已通过：
    ```bash
    conda run -n torch --no-capture-output python -m py_compile \
      train_AT.py train_AT_consistancy.py train_AT_ea_forward.py attack.py attack_ea_forward.py \
      models/eegnet_ea_forward.py tensor_ring_rank_analysis/summarize_exp015_results.py
    ```
  - Shell 语法检查已通过：
    ```bash
    bash -n TN/rank_growth/run_exp015_full_test.sh
    bash -n purify_aug_consistancy_pipeline.sh
    ```
  - Dry-run 已通过：
    - `exp015_dryrun_20260605_233552`：`EXP015_SMOKE=1` 默认 smoke 矩阵生成 `16` 个计划任务，`planned_tasks.csv` 共 `17` 行。
    - `tensor_ring_rank_analysis/summarize_exp015_results.py` 已能基于 dry-run `planned_tasks.csv` 生成 `summary.csv`。
  - 轻量 smoke 已通过：
    - run id：`exp015_smoke_light_seediv_conformer_20260605_235011`
    - 设置：`EXP015_SMOKE=1 DATASETS_CSV=seediv MODELS_CSV=conformer SMOKE_SAMPLE_NUM=1 SMOKE_TRAIN_SAMPLE_NUM=64 SMOKE_ATTACK_SAMPLE_NUM=2`
    - `planned_tasks.csv` 共 `5` 行，覆盖 clean、主方法 `main_js_mse`、Madry baseline、ABAT `conformer_ea_forward`。
    - 汇总：`logs/exp015/exp015_smoke_light_seediv_conformer_20260605_235011/summary.csv`，4 个任务均为 `success/artifact_found`。
  - 以上 smoke 只验证工程路径、日志路径、产物命名和汇总脚本，不作为正式实验结论。
- **启动记录：**
  - 2026-06-05 23:58:18 首次尝试 `nohup` 启动 `exp015_full_20260605_235818`，controller 写入启动日志后未持续运行，无训练子进程保留；该 run id 不作为正式 full 记录。
  - 2026-06-05 23:59:30 使用 `nohup + setsid` 在可访问 GPU 的环境中正式启动 full：
    ```bash
    nohup setsid bash -lc "cd /home/yhj/pythonProject/EEGAP && EXP015_RUN_ID=exp015_full_20260605_235930 bash TN/rank_growth/run_exp015_full_test.sh" \
      > logs/exp015/exp015_full_20260605_235930/controller.log 2>&1 < /dev/null &
    ```
  - run id：`exp015_full_20260605_235930`
  - controller PID：`546853`，记录文件：`logs/exp015/exp015_full_20260605_235930/controller.pid`
  - controller log：`logs/exp015/exp015_full_20260605_235930/controller.log`
  - 初始状态：进入 Phase 1 clean checkpoints，`thubenchmark/eegnet/seed42/fold0` 因已有 clean checkpoint 被跳过，当前训练 `thubenchmark/eegnet/seed42/fold1`。
  - GPU 确认：`nvidia-smi` 显示训练进程 `546883` 使用 `NVIDIA GeForce RTX 4070 Ti SUPER`，约 `2206 MiB` 显存，约 `52%` GPU utilization。
  - 2026-06-06 01:23:02 状态检查：
    - controller 仍在运行，当前训练进程为 `train_AT.py --dataset thubenchmark --model eegnet --at_strategy clean --fold 1 --seed 45`，GPU 进程 PID 为 `574947`。
    - 仍处于 Phase 1 clean checkpoints，`planned_tasks.csv` 共 `10` 行；已记录到 `thubenchmark/eegnet/seed45/fold2`。
    - 已完成：`thubenchmark/eegnet/seed42/fold1`、`seed42/fold2`、`seed43/fold0`、`seed43/fold1`、`seed43/fold2`、`seed45/fold0`；`seed42/fold0` 因已有 clean checkpoint 跳过。
    - 当前运行：`thubenchmark/eegnet/seed45/fold1`；`nvidia-smi` 显示约 `2206 MiB` 显存、约 `59%` GPU utilization。
    - 日志扫描未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`failed` 或 `Exception`。
  - 预计任务规模：Phase 1 clean `36` 个，Phase 2 主方法 `108` 个，Phase 3 普通 AT baseline `324` 个，Phase 4 ABAT `108` 个，共 `576` 个计划任务；`planned_tasks.csv` 会随调度推进逐步追加。
  - 2026-06-08 15:19 状态检查：
    - Phase 1 clean：`35/36` 个有效 artifact；`seediv/conformer/seed42/fold0` 被跳过但对应 clean checkpoint 缺失。
    - Phase 2 main_js_mse：`12/108` 个正式净化结果完成，均为 `thubenchmark/eegnet`。
    - 当前任务停在 `thubenchmark/eegnet/seed43/fold1/eps=0.01`，已完成 Stage 1 两个训练集净化文件，正在 Stage 2 `train_AT_consistancy.py`。
    - 按用户要求停止 EXP-015：对进程组 `546853` 发送 `TERM`，复查宿主进程表未见 `exp015_full_20260605_235930`、`run_exp015_full_test.sh` 或当前训练子进程残留。
  - 完成后汇总命令：
    ```bash
    conda run -n torch --no-capture-output python -u tensor_ring_rank_analysis/summarize_exp015_results.py \
      --log_root logs/exp015/exp015_full_20260605_235930
    ```
- **结果：**
  - 当前已有正式结果仅覆盖 `thubenchmark/eegnet` 的 12 个主方法组合：
    - `seed42`：fold `0,1,2` × eps `0.01,0.03,0.05`
    - `seed43`：fold `0` × eps `0.01,0.03,0.05`
  - 已有 12 个组合的均值：
    - clean accuracy：`0.935221`
    - adversarial accuracy：`0.764811`
    - purified clean accuracy：`0.899577`
    - purified adversarial accuracy：`0.834961`
    - purified adversarial accuracy 相对普通 consistancy 提升：`+0.070150`
    - purified clean accuracy 相对普通 consistancy 下降：`-0.035645`
    - MSE：`0.126506`
  - 按 eps 汇总：
    - `eps=0.01`（n=4）：adv `0.882812` -> purified adv `0.894531`，提升 `+0.011719`
    - `eps=0.03`（n=4）：adv `0.764648` -> purified adv `0.833008`，提升 `+0.068359`
    - `eps=0.05`（n=4）：adv `0.646973` -> purified adv `0.777344`，提升 `+0.130371`
  - baseline、ABAT、Conformer、SEED-IV 尚未产生正式 full 结果。
- **风险：**
  - 全量矩阵计算量很大；先 smoke 再 full，full 使用稳定日志目录跟踪。
  - JS_MSE 阈值固定来自 `thubenchmark/eegnet/fold0/seed42/eps0.03`，跨 dataset/model/eps 泛化可能不稳定；本实验正是检验该风险。
  - `conformer_ea_forward` 是本实验新增 ABAT 路径，需要 smoke 先验证模型构造、训练和 subject-aware attack。

### EXP-016：对抗训练逐 epoch PGD 鲁棒准确率评估

- **日期：** 2026-06-10
- **状态：** 已完成（工程 smoke）
- **相关 idea：** None
- **目的：**
  - 在 `train_AT.py` 每个 epoch 结束后，用当前训练配置的 PGD 攻击完整测试集并记录鲁棒准确率。
- **关键设置：**
  - 攻击参数复用 `epsilon`、`pgd_step_size`、`pgd_steps`、`pgd_random_start` 和输入裁剪参数。
  - `Robust Acc` 与原有 epoch 指标写入同一行日志。
  - 鲁棒评估前后恢复 PyTorch CPU/CUDA RNG 状态，避免评估随机起点改变后续训练随机序列。
- **命令：**
  ```bash
  conda run -n torch --no-capture-output python -m py_compile train_AT.py

  conda run -n torch --no-capture-output python -u train_AT.py \
    --dataset thubenchmark --model eegnet --at_strategy madry --fold 0 \
    --epsilon 0.01 --epochs 1 --batch_size 128 --train_sample_num 64 \
    --patience 1 --seed 42 --gpu_id 0
  ```
- **结果：**
  - 运行环境无法连接 NVIDIA 驱动，本次 smoke 使用 CPU 完成。
  - 训练集抽样 `64` 条，验证集和测试集分别为 `840` 条。
  - epoch 日志已在同一行输出：
    - `Test Acc: 0.0167`
    - `Test Loss: 3.6893`
    - `Robust Acc: 0.0000`
  - 日志：`log_train_AT/train_thubenchmark_eegnet_no_ea_madry_eps0.01_42_fold0_0.001_0.0001_128_20260610_175329.log`
  - 该结果只验证逐 epoch PGD 鲁棒评估链路，不作为模型性能结论。
- **风险：**
  - 每个 epoch 需要额外执行一次完整测试集 PGD，训练总耗时会明显增加。
  - smoke 使用脚本的标准 checkpoint 命名，生成了 clean/madry 各两个 checkpoint；若同名文件此前存在，则已被本次 1 epoch smoke 覆盖。

### EXP-017：RPCF 首轮单条件完整实验

- **日期：** 2026-06-11
- **状态：** Completed（legacy RPCF seed protocol）
- **相关 idea：** `IDEA-007`
- **目的：**
  - 验证 purification-sensitive layer selection 与 rank curriculum fine-tuning 是否能改善 AT 模型对 EEG_TNP 多 rank 净化视图的适应性。
- **关键设置：**
  - 数据集/模型：`thubenchmark / EEGNet`
  - split：`seed=42, fold=0, no-EA`
  - 初始化：Madry AT，`eps=0.03`
  - 训练集 RPCF cache：`sample_num=512`
  - ranks：`15,20,25,30,35,40`
  - sensitive layers：组合 sensitivity Top 40%
  - fine-tuning：100 epochs，batch 64，AdamW `lr=1e-4, weight_decay=1e-4`
  - checkpoint：优先最大化 validation PGD robust accuracy
  - 最终攻击：AT 与 RPCF 各自 white-box AutoAttack
- **Pipeline：**
  ```bash
  # dry-run
  DRY_RUN=1 SMOKE=1 RPCF_RUN_ID=rpcf_dryrun_20260611 \
    bash rpcf/run_rpcf_pipeline.sh

  # smoke
  SMOKE=1 RPCF_RUN_ID=rpcf_smoke_YYYYMMDD_HHMMSS \
    bash rpcf/run_rpcf_pipeline.sh

  # full，必须后台运行
  nohup setsid bash -lc \
    "cd /home/yhj/pythonProject/EEGAP && \
     RPCF_RUN_ID=rpcf_full_YYYYMMDD_HHMMSS bash rpcf/run_rpcf_pipeline.sh" \
    > logs/rpcf/rpcf_full_YYYYMMDD_HHMMSS/controller.log 2>&1 < /dev/null &
  ```
- **已执行验证：**
  - `rpcf/*.py` 的 `py_compile` 通过。
  - `bash -n rpcf/run_rpcf_pipeline.sh` 通过。
  - cache、sensitivity、fine-tuning CLI `--help` 导入通过。
  - `python -m unittest test_rpcf.py`：6 个测试全部通过。
  - 四类 TorchEEG 模型的逻辑层 hook 随机输入检查通过。
  - dry-run：`logs/rpcf/rpcf_dryrun_20260611/`，七个阶段命令和 artifact 路径均成功展开。
  - 完整 smoke 已通过：
    - run id：`rpcf_smoke_20260611_182500`
    - 设置：六个 rank、训练样本 1、fine-tuning 1 epoch、white-box attack 样本 2、测试净化样本 2。
    - Stage 1 至 Stage 7 全部完成，统一 cache、sensitivity、checkpoint、两套 attack、两套逐-rank purification 和 summary 均已生成。
    - smoke 选择 `block2/block1`，可训练参数比例 `0.066071`；该结果只验证工程链路，不作为正式层选择结论。
    - smoke 的最佳 checkpoint 保持在初始化 epoch 0；n=1/2 的准确率不作为正式实验结果。
    - 汇总：`logs/rpcf/rpcf_smoke_20260611_182500/summary.csv`
- **正式启动记录：**
  - 启动时间：2026-06-11 18:27（Asia/Shanghai）
  - run id：`rpcf_full_20260611_183000`
  - controller PID：`2692104`
  - controller log：`logs/rpcf/rpcf_full_20260611_183000/controller.log`
  - PID 文件：`logs/rpcf/rpcf_full_20260611_183000/controller.pid`
  - 完成时间：2026-06-12 11:08（Asia/Shanghai）
  - Stage 1 至 Stage 7 全部完成。
- **结果：**
  - 汇总：`logs/rpcf/rpcf_full_20260611_183000/summary.csv`
  - selected layers：`block2/block1`，可训练参数比例 `0.066071`。
  - RPCF clean / AutoAttack：`0.939286 / 0.775000`。
  - AT clean / AutoAttack：`0.938095 / 0.778571`。
  - 六 rank purified adversarial accuracy 平均值：RPCF `0.828125`，AT `0.790039`。
- **2026-06-12 seed 协议修正：**
  - 该 run 使用旧 RPCF 子集规则 `seed + fold * 1000 + 7007`，测试净化 `selection_seed=7049`。
  - 当前代码已改为与 consistancy 完全一致的 `seed + fold * 1000`；`seed42/fold0` 对应 `selection_seed=42`。
  - fine-tuning DataLoader 已移除 `seed+991` 私有 generator，改为与 `train_AT_consistancy.py` 一样使用 `seed_everything(seed)` 后的全局 shuffle。
  - `evaluate_purification.py` 已补充 `seed_everything(seed)`，与现有 `purify.py` 入口一致。
  - 因此本实验结果保留为历史结果，但不能用于与 consistancy 的严格同子集比较；seed-aligned RPCF 正式复验尚未运行，结果为 `Pending`。
  - 修正后验证：`py_compile` 通过；`python -m unittest test_rpcf.py` 共 7 项通过；dry-run 为 `logs/rpcf/rpcf_seed_aligned_dryrun_20260612/`。
- **风险：**
  - 单次 full 包含训练集和两套 white-box 测试净化，共六个 rank，预计运行时间较长。
  - 当前首轮只覆盖一个 dataset/model/seed/fold/eps 条件，结论只能作为方法可行性验证。

### EXP-018：RPCF、consistancy 与普通 EEG_TNP+AT 公平对比

- **日期：** 2026-06-12
- **状态：** 已完成（Completed）
- **相关 idea：** `IDEA-007`
- **目的：**
  - 公平比较 RPCF、consistancy、普通 EEG_TNP 净化后由 Madry AT 模型分类三条流程。
  - 三条方法从统一新训练的 seed42 Madry AT 基础模型出发，重新运行训练、white-box AutoAttack 和最终净化评估，不复用旧实验指标。
- **关键设置：**
  - 数据集/模型：`thubenchmark / EEGNet`
  - split：`seed=42, fold=0, no-EA`
  - 统一 AT：Madry，`eps=0.03`，400 epochs，patience 20。
  - consistancy：保持既有方法定义，从随机初始化执行 Madry AT + rank25/30 paired consistency；paired cache 的攻击模型使用统一 AT checkpoint。
  - RPCF：从统一 AT checkpoint 初始化；训练 cache 使用 ranks `15,20,25,30,35,40`，Top 40% sensitive layers，100 epochs。
  - 攻击：三种分类器分别运行自身 white-box AutoAttack，显式 `attack_seed=42`。
  - 最终净化：三条方法均只运行固定 rank `25,30`。
  - 公平子集：三条方法均使用 `selection_seed=seed+fold*1000=42` 得到同一 n512 测试子集；汇总要求 source indices 顺序、labels 和 clean tensors 完全一致。
- **随机协议：**
  - 新增共享 `utils/reproducibility.py`，统一 Python、NumPy、PyTorch、CUDA seed 和稳定子集抽样。
  - 标准 AT、consistancy、RPCF、攻击和净化入口均调用共享 `seed_everything(42)`。
  - AutoAttack 从每个入口显式接收 seed42，不依赖默认参数。
- **命令：**
  ```bash
  # dry-run
  DRY_RUN=1 SMOKE=1 EXP018_RUN_ID=exp018_dryrun_full_20260612 \
    bash rpcf/run_exp018.sh

  # smoke
  SMOKE=1 SKIP_EXISTING=0 \
    EXP018_RUN_ID=exp018_smoke_20260612_1240 \
    bash rpcf/run_exp018.sh

  # 正式实验必须后台运行
  nohup setsid bash -lc \
    "cd /home/yhj/pythonProject/EEGAP && \
     EXP018_RUN_ID=exp018_full_20260612_124131 bash rpcf/run_exp018.sh" \
    > logs/exp018/exp018_full_20260612_124131/controller.log 2>&1 < /dev/null &
  ```
- **Pipeline：**
  1. 新训练统一 Madry AT。
  2. 生成 consistancy rank25/30 paired train cache。
  3. 训练 consistancy。
  4. 生成 RPCF 六-rank train cache。
  5. 计算 RPCF sensitivity。
  6. RPCF fine-tuning。
  7. 三模型分别运行 AutoAttack。
  8. 三模型分别运行 rank25/30 净化。
  9. 严格校验并汇总。
- **已执行验证：**
  - `py_compile` 通过。
  - `bash -n rpcf/run_exp018.sh` 通过。
  - CLI `--help` 通过。
  - `python -m unittest test_rpcf.py`：9 项通过。
  - dry-run：`logs/exp018/exp018_dryrun_full_20260612/`。
  - 完整 GPU smoke 已通过：
    - run id：`exp018_smoke_20260612_1240`
    - 设置：AT/consistancy/RPCF 各 1 epoch，训练缓存 n1，攻击/净化 n2；RPCF 保留六个训练 rank，最终保留 rank25/30。
    - 9 个 stage 全部完成，三条方法的 source indices 均为 `[816,695]`，严格汇总成功。
    - 汇总：`logs/exp018/exp018_smoke_20260612_1240/summary.csv`。
    - smoke 指标仅验证工程链路，不作为性能结果。
- **正式启动记录：**
  - 启动时间：2026-06-12 12:41（Asia/Shanghai）
  - 完成时间：2026-06-13 05:12（Asia/Shanghai）
  - run id：`exp018_full_20260612_124131`
  - controller PID：`2927720`
  - controller log：`logs/exp018/exp018_full_20260612_124131/controller.log`
  - PID 文件：`logs/exp018/exp018_full_20260612_124131/controller.pid`
- **输出：**
  - 控制日志：`logs/exp018/<run_id>/controller.log`
  - checkpoints：`checkpoints/*exp018_full_20260612_124131*`
  - attack：`ad_data/exp018/`
  - train/eval purification：`purified_data/exp018/`
  - 公平汇总：`logs/exp018/<run_id>/summary.csv`、`summary.json`
- **结果：**
  - 完整测试集（n=840）：
    - 普通 AT：clean `0.938095`，AutoAttack `0.778571`。
    - consistancy：clean `0.935714`，AutoAttack `0.761905`。
    - RPCF：clean `0.938095`，AutoAttack `0.778571`。
  - 统一 seed42 n512 净化：
    - rank25 purified adversarial accuracy：AT `0.818359`，consistancy `0.832031`，RPCF `0.818359`。
    - rank30 purified adversarial accuracy：AT `0.816406`，consistancy `0.833984`，RPCF `0.816406`。
    - rank25 purified clean accuracy：AT/RPCF `0.929688`，consistancy `0.921875`。
    - rank30 purified clean accuracy：三者均为 `0.925781`。
  - RPCF 选择 `block2`、`block1`，可训练参数比例 `0.066071`；最佳 epoch 为 `0`。15 个 fine-tuning epoch 均未超过初始化 checkpoint，因此本次 RPCF checkpoint 与统一 AT checkpoint 等价，全部评估指标也与普通 AT 完全一致。
  - consistancy 虽然自身 white-box AutoAttack accuracy 低于 AT `1.6667` 个百分点，但净化后的 adversarial accuracy 在 rank25/30 分别高于 AT `1.3672`、`1.7578` 个百分点。
  - 汇总：`logs/exp018/exp018_full_20260612_124131/summary.csv`、`summary.json`、`full_test_attack.csv`。
- **结论：**
  - 当前 RPCF 配置没有产生有效 fine-tuning 收益，不能据此宣称优于 AT。
  - consistancy 的收益主要出现在“净化后对抗样本”场景，而不是未净化 white-box robustness。
  - 在该设置下，rank30 对 consistancy 的 clean/robust/MSE 综合表现最好；普通 AT/RPCF 的准确率则略偏向 rank25。
- **风险：**
  - 结果仍仅覆盖 `thubenchmark / EEGNet / fold0 / seed42 / eps0.03`，不能直接外推到其他 seed、fold、模型或攻击强度。
  - RPCF 的 epoch0 checkpoint 选择需要进一步区分“训练目标无效”和“validation PGD 选模标准过于保守”。

#### EXP-018 RPCF inter-layer sensitivity + consistancy KL 续跑

- **日期：** 2026-06-14
- **状态：** 已手动停止（Stopped）
- **run id：** `exp018_rpcf_interlayer_kl_20260614_2340`
- **修改：**
  - sensitivity 从绝对 feature shift 改为相邻阶段放大率：
    `C_l(r)=S_l(r)/(S_previous(r)+eps)`；首层使用 input shift 作为分母。
  - clean-purified 和 adversarial-purified 分支分别计算相对敏感度，再跨 rank 等权组合。
  - fine-tuning 使用 clean logits 作为 detached teacher；`x_adv`、多 rank `x_pur/x_adv_pur` 均使用温度 KL 对齐 clean teacher，同时保留 hard-label CE 和动态 rank 权重。
- **复用产物：**
  - 统一 AT checkpoint、seed42 n512 六-rank RPCF cache。
  - 原 EXP-018 的 AT/consistancy checkpoint 和 rank25/30 评估产物。
- **初步 sensitivity：**
  - `block2`：`1.054864`。
  - `block1`：`1.041012`。
  - `lin`：`0.696991`。
  - Top 40% 仍选择 `block2, block1`，可训练参数比例 `6.6071%`。
- **fine-tuning 状态：**
  - 运行 15 epochs 后 early stopping。
  - 最佳 checkpoint 仍为 epoch0：validation PGD robust accuracy `0.798810`，
    clean accuracy `0.925000`，validation loss `0.263624`。
  - 2026-06-14 在 Stage 3 white-box AutoAttack 期间按用户要求手动停止。
  - sensitivity、fine-tuning history 和 checkpoint 已保留；后续可从 Stage 3 重启。
  - 最终 AutoAttack 和净化指标仍为 `Pending`。
- **验证：**
  - `py_compile`、`bash -n`、CLI `--help` 和 dry-run 通过。
  - `python -m unittest test_rpcf.py`：11 项通过。
- **日志：**
  - controller：`logs/exp018/exp018_rpcf_interlayer_kl_20260614_2340/controller.log`
  - fine-tuning：`logs/exp018/exp018_rpcf_interlayer_kl_20260614_2340/finetune.log`
- **结果：**
  - Pending

#### EXP-018 RPCF 无早停续跑

- **日期：** 2026-06-14
- **状态：** 已完成（Completed）
- **run id：** `exp018_rpcf_no_early_stop_20260614_2357`
- **训练策略：**
  - 复用原 EXP-018 AT checkpoint、seed42 n512 六-rank训练 cache 和新版 inter-layer sensitivity。
  - clean logits 作为 detached teacher，使用 consistancy CE+KL 和动态 rank 权重。
  - 不使用 validation early stopping 或 best-checkpoint 回退，固定训练 100 epochs 并保存 epoch100。
  - 每轮 clean/PGD validation 指标仅用于诊断，不参与 checkpoint 选择。
- **验证：**
  - `py_compile`、`bash -n`、dry-run 通过。
  - `python -m unittest test_rpcf.py`：11 项通过。
  - 1-epoch GPU smoke 确认保存最终 epoch，`checkpoint_policy=final_epoch_no_early_stopping`。
- **日志：**
  - controller：`logs/exp018/exp018_rpcf_no_early_stop_20260614_2357/controller.log`
  - fine-tuning：`logs/exp018/exp018_rpcf_no_early_stop_20260614_2357/finetune.log`
- **结果：**
  - 完成时间：2026-06-15 02:05（Asia/Shanghai）。
  - 100 epochs 全部完成，最终 validation clean accuracy `0.933333`，
    PGD robust accuracy `0.728571`；初始值分别为 `0.925000`、`0.798810`。
  - 完整测试集（n=840）：clean accuracy `0.950000`，AutoAttack accuracy
    `0.722619`。相对 AT 分别变化 `+1.1905`、`-5.5952` 个百分点。
  - seed42 n512、rank25：purified clean `0.929688`，purified adversarial
    `0.835938`；后者高于 AT `1.7578`、高于 consistancy `0.3906` 个百分点。
  - seed42 n512、rank30：purified clean `0.937500`，purified adversarial
    `0.841797`；分别高于 AT `1.1719/2.5391` 个百分点，高于 consistancy
    `1.1719/0.7813` 个百分点。
  - 汇总：`logs/exp018/exp018_rpcf_no_early_stop_20260614_2357/summary.csv`、
    `summary.json`、`full_test_attack.csv`。
- **结论：**
  - 无早停 RPCF 确实获得了更强的 purification adaptation，rank25/30 净化后
    adversarial accuracy 均为三种方法最高，rank30 最优。
  - 代价是分类器自身 white-box robustness 明显下降；当前 RPCF 学到了净化分布，
    但没有保持原始 AT robust decision boundary。
  - 后续重点不应继续无约束增加 epochs，而应控制 purification adaptation 与原始
    robustness 的权衡，例如减少 epochs、降低 KL/CE 权重或加入参数漂移约束。

#### EXP-018 RPCF w.o. sensitivity layer selection

- **日期：** 2026-06-15
- **状态：** 已完成（Completed）
- **run id：** `exp018_rpcf_all_layers_20260615_0933`
- **目的：**
  - 检验 purification-sensitive layer selection 的独立作用。
  - 与无早停 selective RPCF 保持相同 AT 初始化、seed42 n512 六-rank cache、
    clean-teacher consistancy CE+KL、动态 rank 权重、100 epochs 和最终评估。
  - 唯一核心变化是启用 `--all_layers`，微调全模型参数且所有 BatchNorm 保持训练状态。
- **训练范围：**
  - 可训练参数：`32208/32208`，比例 `100%`。
  - sensitivity artifact 仅用于 cache/rank 元数据校验，不参与参数冻结。
- **验证：**
  - `py_compile`、`bash -n`、CLI dry-run 通过。
  - `python -m unittest test_rpcf.py`：12 项通过。
  - 完整 n2 smoke 通过：`exp018_rpcf_all_layers_smoke_20260615_0932`；
    source indices 与原 EXP-018 smoke 均为 `[816,695]`。
- **日志：**
  - controller：`logs/exp018/exp018_rpcf_all_layers_20260615_0933/controller.log`
  - fine-tuning：`logs/exp018/exp018_rpcf_all_layers_20260615_0933/finetune.log`
- **结果：**
  - 完成时间：2026-06-15 11:41（Asia/Shanghai）。
  - 100 epochs 全部完成，最终 validation clean accuracy `0.935714`，
    PGD robust accuracy `0.728571`。
  - 完整测试集（n=840）：clean accuracy `0.946429`，AutoAttack accuracy
    `0.727381`。相对 selective RPCF 分别变化 `-0.3571/+0.4762` 个百分点。
  - seed42 n512、rank25：purified clean `0.925781`，purified adversarial
    `0.837891`；相对 selective RPCF 分别变化 `-0.3906/+0.1953` 个百分点。
  - seed42 n512、rank30：purified clean `0.933594`，purified adversarial
    `0.828125`；相对 selective RPCF 均下降，分别为 `-0.3906/-1.3672`
    个百分点。
  - 汇总：`logs/exp018/exp018_rpcf_all_layers_20260615_0933/summary.csv`、
    `summary.json`、`full_test_attack.csv`。
- **结论：**
  - 全模型微调没有稳定优于 sensitive-layer selection；其收益仅表现为未净化
    AutoAttack `+0.48` 和 rank25 purified adversarial `+0.20` 个百分点。
  - selective RPCF 在 rank30 purified clean/adversarial accuracy 上均更高，
    尤其 purified adversarial accuracy 高 `1.37` 个百分点。
  - 当前单次实验支持 sensitive-layer selection 的独立价值：仅训练 `6.61%`
    参数即可取得更好的 rank30 净化性能，并减少全模型适配造成的不稳定性。

#### EXP-018 RPCF w.o. rank schedule

- **日期：** 2026-06-15
- **状态：** 已完成（Completed）
- **run id：** `exp018_rpcf_static_ranks_20260615_1155`
- **目的：**
  - 检验 low-rank-to-high-rank动态 curriculum 的独立作用。
  - 保留新版 inter-layer sensitivity 和 `block2/block1` selective fine-tuning；
    与主 RPCF 使用相同 AT 初始化、seed42 n512 六-rank cache、clean-teacher
    consistancy CE+KL、100 epochs 和最终评估。
  - 唯一核心变化是启用 `--static_rank_weights`，rank
    `15,20,25,30,35,40` 在所有 epoch 均固定为 `1/6`。
- **验证：**
  - `py_compile`、`bash -n`、dry-run 通过。
  - `python -m unittest test_rpcf.py`：12 项通过。
  - 完整 n2 smoke 通过：`exp018_rpcf_static_ranks_smoke_20260615_1154`；
    六个 rank 权重均为 `0.1666667`，source indices 为 `[816,695]`。
- **日志：**
  - controller：`logs/exp018/exp018_rpcf_static_ranks_20260615_1155/controller.log`
  - fine-tuning：`logs/exp018/exp018_rpcf_static_ranks_20260615_1155/finetune.log`
- **结果：**
  - 完成时间：2026-06-15 14:03（Asia/Shanghai）。
  - 完整测试集（n=840）：clean accuracy `0.951190`，AutoAttack accuracy
    `0.715476`。相对 dynamic-schedule RPCF 分别变化 `+0.1190/-0.7143`
    个百分点。
  - seed42 n512、rank25：purified clean `0.929688`，purified adversarial
    `0.839844`；相对 dynamic schedule 分别变化 `0.0000/+0.3906`
    个百分点。
  - seed42 n512、rank30：purified clean `0.937500`，purified adversarial
    `0.835938`；相对 dynamic schedule 分别变化 `0.0000/-0.5859`
    个百分点。
  - 汇总：`logs/exp018/exp018_rpcf_static_ranks_20260615_1155/summary.csv`、
    `summary.json`、`full_test_attack.csv`。
- **结论：**
  - 去除 schedule 后并非全面退化，而是 rank25 略有提升、rank30 和未净化
    AutoAttack 下降，说明 schedule 确实改变了模型对不同净化强度的适应分布。
  - dynamic schedule 在当前主要结果 rank30 上高 `0.59` 个百分点，同时保留
    更高的分类器自身 AutoAttack accuracy，因此继续作为 RPCF 默认配置。
  - rank25/30 purified adversarial accuracy 的两-rank均值差异很小；schedule
    的证据主要来自 rank30 和未净化鲁棒性，仍需跨 seed/fold 验证。

#### EXP-018 consistancy 六-rank增强续跑

- **日期：** 2026-06-17
- **状态：** 正式实验运行中（Running）
- **相关 idea：** `IDEA-007`
- **目的：**
  - 对齐 seed42 中 consistancy 与 RPCF 的训练 rank 分布。
  - 将 consistancy 训练增强从 rank25/30 改为与 RPCF 一致的
    rank15/20/25/30/35/40。
- **复用产物：**
  - AT：`exp018_full_20260612_124131`
  - RPCF selective：`exp018_rpcf_no_early_stop_20260614_2357`
  - RPCF all-layers：`exp018_rpcf_all_layers_20260615_0933`
  - RPCF rank-weight uniform：`exp018_rpcf_static_ranks_20260615_1155`
- **重跑内容：**
  1. 使用 seed42/fold0/n512 和同一 AT checkpoint 生成六个 paired train cache。
  2. 使用六个 paired cache 重新训练 consistancy。
  3. 对新 consistancy checkpoint 运行自身 white-box AutoAttack。
  4. 对新 attack 结果运行 rank25/30 净化评估。
  5. 复用原 AT/RPCF 结果，重新生成五方法汇总表。
- **验证：**
  - `bash -n rpcf/run_exp018_consistancy_six_rank.sh` 通过。
  - dry-run 通过：
    `EXP018_CONSISTANCY_SIX_RANK_RUN_ID=exp018_consistancy_six_rank_dryrun`。
- **正式命令：**
  ```bash
  nohup setsid bash -lc \
    "cd /home/yhj/pythonProject/EEGAP && \
     EXP018_CONSISTANCY_SIX_RANK_RUN_ID=exp018_seed42_fold0_consistancy_six_rank_YYYYMMDD_HHMM \
     bash rpcf/run_exp018_consistancy_six_rank.sh" \
    > logs/exp018/exp018_seed42_fold0_consistancy_six_rank_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-06-17 10:21（Asia/Shanghai）
  - run id：`exp018_seed42_fold0_consistancy_six_rank_20260617_1020`
  - controller PID：`180071`
  - PID 文件：
    `logs/exp018/exp018_seed42_fold0_consistancy_six_rank_20260617_1020/controller.pid`
  - controller log：
    `logs/exp018/exp018_seed42_fold0_consistancy_six_rank_20260617_1020/controller.log`
  - 启动后状态：Stage 1 rank15 paired cache 正在生成。
- **结果：**
  - Pending

### EXP-019：五方法 seed43 公平复验

- **日期：** 2026-06-15 至 2026-06-16
- **状态：** 已完成
- **相关 idea：** `IDEA-007`
- **目的：**
  - 在独立随机种子上补全五种方法的完整训练、white-box attack 和净化结果。
  - 检查 EXP-018 中 selective layer 与 dynamic rank schedule 的结论能否迁移到
    seed43。
- **方法：**
  - `Madry AT`
  - `consistancy`
  - `RPCF selective`
  - `RPCF all-layers`
  - `RPCF rank-weight uniform`
- **统一设置：**
  - dataset/model：`thubenchmark / EEGNet`
  - split：`seed=43, fold=0, no-EA`
  - attack：white-box AutoAttack，`eps=0.03`，显式 `attack_seed=43`
  - AT/consistancy：最多 400 epochs，patience 20
  - RPCF：六-rank cache `15,20,25,30,35,40`，三种变体均训练 100 epochs
  - 训练 cache：n512；最终净化：同一 seed43 n512 子集、rank25/30
- **Pipeline：**
  - `rpcf/run_exp019.sh` 共 11 个可续跑阶段。
  - 三个 RPCF 变体共享同一 AT 初始化、六-rank cache 和 sensitivity 结果。
  - 五种方法分别生成自身 white-box AutoAttack，再独立执行 rank25/30 净化。
  - 汇总严格校验实验协议、完整测试集样本数、baseline 重复行和
    `source_indices`。
- **验证：**
  - `bash -n rpcf/run_exp019.sh` 通过。
  - 完整 11-stage dry-run 通过：`exp019_seed43_fold0_dryrun`。
  - `python -m unittest test_rpcf.py`：15 项通过。
  - n2 GPU smoke 全流程通过：
    `exp019_seed43_fold0_smoke_20260615_1817`。
  - smoke 的五方法净化 `source_indices=[196,214]`，最终 JSON 标识为
    `exp019_five_method_comparison`；smoke 指标不作为正式结果。
- **正式命令：**
  ```bash
  nohup setsid bash -lc \
    "cd /home/yhj/pythonProject/EEGAP && \
     EXP019_RUN_ID=exp019_seed43_fold0_full_YYYYMMDD_HHMM \
     bash rpcf/run_exp019.sh" \
    > logs/exp019/exp019_seed43_fold0_full_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-06-15 18:21（Asia/Shanghai）
  - run id：`exp019_seed43_fold0_full_20260615_1821`
  - controller PID：`3874811`
  - PID 文件：
    `logs/exp019/exp019_seed43_fold0_full_20260615_1821/controller.pid`
  - controller log：
    `logs/exp019/exp019_seed43_fold0_full_20260615_1821/controller.log`
  - 完成时间：2026-06-16 14:20:39（Asia/Shanghai）
- **结果：**
  - 汇总产物：
    `logs/exp019/exp019_seed43_fold0_full_20260615_1821/five_methods/five_methods_table.md`
  - 长格式 CSV：
    `logs/exp019/exp019_seed43_fold0_full_20260615_1821/five_methods/five_methods_long.csv`
  - 宽格式 CSV：
    `logs/exp019/exp019_seed43_fold0_full_20260615_1821/five_methods/five_methods_wide.csv`
  - JSON：
    `logs/exp019/exp019_seed43_fold0_full_20260615_1821/five_methods/five_methods_summary.json`

| Method | Full clean | Full AutoAttack | Rank 25 clean | Rank 25 adversarial | Rank 30 clean | Rank 30 adversarial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Madry AT | 91.79% | 79.52% | 88.87% | 81.64% | 90.04% | 81.84% |
| consistancy | 92.14% | 77.50% | 88.09% | 82.23% | 88.87% | 82.62% |
| RPCF selective | 93.45% | 71.67% | 91.02% | 81.64% | 90.43% | 80.86% |
| RPCF all-layers | 93.45% | 71.55% | 89.26% | 81.25% | 90.04% | 80.86% |
| RPCF rank-weight uniform | 93.45% | 71.31% | 90.82% | 82.03% | 90.82% | 82.03% |

- **观察：**
  - seed43 上，未净化完整测试集 AutoAttack 最好的是 `Madry AT`（`79.52%`）。
  - rank25/30 净化后 adversarial accuracy 最好的是 `consistancy`
    （`82.23%/82.62%`）。
  - 三个 RPCF 变体完整 clean accuracy 均为 `93.45%`，但完整 AutoAttack
    只有 `71.31%` 至 `71.67%`，明显低于 AT 和 consistancy。
  - 与 seed42 不同，`RPCF selective` 在 rank25/30 净化后 adversarial
    accuracy 均没有超过 consistancy；`rank-weight uniform` 也没有复现
    seed42 上 rank25 的优势。
- **结论：**
  - EXP-018 seed42 中“RPCF selective/dynamic schedule 更优”的结论不能直接作为
    跨 seed 稳定结论。
  - 当前更稳妥的表述是：RPCF 能提高净化分布上的 clean 适配，但在 seed43
    中没有稳定提升净化后 adversarial accuracy，且会显著牺牲未净化
    AutoAttack。
  - consistancy 只使用 rank25/30 净化样本做数据增强，与本次 rank25/30
    评估分布更匹配；RPCF 使用六-rank 训练分布，这可能是 seed43 结果反转的
    重要因素之一。

#### EXP-019 consistancy 六-rank增强续跑

- **日期：** 2026-06-16 至 2026-06-17
- **状态：** 已完成
- **相关 idea：** `IDEA-007`
- **目的：**
  - 修正 EXP-019 中 consistancy 只使用 rank25/30 做数据增强的对比不公平问题。
  - 让 consistancy 使用与 RPCF 训练 cache 一致的六个 rank：
    `15,20,25,30,35,40`。
- **复用产物：**
  - 原 EXP-019 seed43 AT checkpoint。
  - 原 EXP-019 三个 RPCF 变体 checkpoint、attack/purification 结果和 history。
  - 原 EXP-019 AT rank25/30 净化结果。
- **重跑内容：**
  1. 为 consistancy 生成 rank15/20/25/30/35/40 六个 paired train cache，均使用
     seed43、fold0、n512 和同一 AT checkpoint。
  2. 使用六个 paired cache 重新训练 consistancy。
  3. 对新 consistancy checkpoint 运行自身 white-box AutoAttack。
  4. 对新 attack 结果运行 rank25/30 净化评估。
  5. 复用原 AT/RPCF 结果，重新生成五方法汇总表。
- **验证：**
  - `bash -n rpcf/run_exp019_consistancy_six_rank.sh` 通过。
  - dry-run 通过：
    `EXP019_CONSISTANCY_SIX_RANK_RUN_ID=exp019_consistancy_six_rank_dryrun`。
- **正式命令：**
  ```bash
  nohup setsid bash -lc \
    "cd /home/yhj/pythonProject/EEGAP && \
     EXP019_CONSISTANCY_SIX_RANK_RUN_ID=exp019_seed43_fold0_consistancy_six_rank_YYYYMMDD_HHMM \
     bash rpcf/run_exp019_consistancy_six_rank.sh" \
    > logs/exp019/exp019_seed43_fold0_consistancy_six_rank_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-06-16 14:52（Asia/Shanghai）
  - run id：`exp019_seed43_fold0_consistancy_six_rank_20260616_1451`
  - controller PID：`4128123`
  - PID 文件：
    `logs/exp019/exp019_seed43_fold0_consistancy_six_rank_20260616_1451/controller.pid`
  - controller log：
    `logs/exp019/exp019_seed43_fold0_consistancy_six_rank_20260616_1451/controller.log`
  - 完成时间：2026-06-17 00:18:34（Asia/Shanghai）
- **结果：**
  - 汇总表：
    `logs/exp019/exp019_seed43_fold0_consistancy_six_rank_20260616_1451/five_methods/five_methods_table.md`
  - 长格式 CSV：
    `logs/exp019/exp019_seed43_fold0_consistancy_six_rank_20260616_1451/five_methods/five_methods_long.csv`
  - 宽格式 CSV：
    `logs/exp019/exp019_seed43_fold0_consistancy_six_rank_20260616_1451/five_methods/five_methods_wide.csv`
  - JSON：
    `logs/exp019/exp019_seed43_fold0_consistancy_six_rank_20260616_1451/five_methods/five_methods_summary.json`

| Method | Full clean | Full AutoAttack | Rank 25 clean | Rank 25 adversarial | Rank 30 clean | Rank 30 adversarial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Madry AT | 91.79% | 79.52% | 88.87% | 81.64% | 90.04% | 81.84% |
| consistancy | 91.19% | 74.17% | 87.11% | 81.05% | 89.06% | 79.49% |
| RPCF selective | 93.45% | 71.67% | 91.02% | 81.64% | 90.43% | 80.86% |
| RPCF all-layers | 93.45% | 71.55% | 89.26% | 81.25% | 90.04% | 80.86% |
| RPCF rank-weight uniform | 93.45% | 71.31% | 90.82% | 82.03% | 90.82% | 82.03% |

- **观察：**
  - 相比原 EXP-019 rank25/30-only consistancy，six-rank consistancy 明显下降：
    full AutoAttack 从 `77.50%` 降至 `74.17%`，rank25/30 purified adversarial
    从 `82.23%/82.62%` 降至 `81.05%/79.49%`。
  - 在公平 six-rank增强口径下，rank25/30 净化后 adversarial accuracy 最好的是
    `RPCF rank-weight uniform`（`82.03%/82.03%`）。
  - `Madry AT` 仍是未净化 full AutoAttack 最强方法（`79.52%`）。
- **结论：**
  - 原 seed43 结论中 “consistancy 在净化后 adversarial 最好” 依赖于
    consistancy 只使用 rank25/30 训练增强、与评估 rank 完全匹配。
  - 当 consistancy 也使用与 RPCF 一致的六-rank增强后，其 rank25/30 净化后
    adversarial accuracy 不再领先；这说明训练 rank 分布与评估 rank 分布匹配度
    是关键混杂因素。
  - 后续若继续比较 consistancy 和 RPCF，必须显式区分 rank25/30-only 与 six-rank
    两种训练分布，不能混用后直接下结论。

#### EXP-019 seed44 五方法与 consistancy 六-rank复验

- **日期：** 2026-06-17 起
- **状态：** 已完成
- **相关 idea：** `IDEA-007`
- **目的：**
  - 在 `seed=44, fold=0, eps=0.03` 上复验 EXP-019 五方法结果。
  - 将 full five-method 中的 `consistancy` 直接改为 six-rank 训练增强口径，
    形成与 seed42/seed43 可对齐的三 seed 汇总。
- **统一设置：**
  - dataset/model：`thubenchmark / EEGNet`
  - split：`seed=44, fold=0, no-EA`
  - attack：white-box AutoAttack，`eps=0.03`
  - 训练 cache：n512；最终净化：同一 seed44 n512 子集、rank25/30
- **执行计划：**
  - 运行新版 `rpcf/run_exp019.sh`，生成 seed44 的 Madry AT、six-rank
    consistancy、RPCF selective、RPCF all-layers、RPCF rank-weight uniform
    及五方法汇总。
  - `run_exp019.sh` 中的 consistancy 训练增强 rank 已改为
    `15,20,25,30,35,40`；最终 `five_methods` 表不再需要后置替换。
- **验证：**
  - 旧版 dry-run：
    `SEED=44 DRY_RUN=1 EXP019_RUN_ID=exp019_seed44_fold0_full_dryrun bash rpcf/run_exp019.sh`
    通过。
  - 独立 six-rank rerun dry-run：
    `SEED=44 DRY_RUN=1 BASE_RUN_ID=exp019_seed44_fold0_full_dryrun EXP019_CONSISTANCY_SIX_RANK_RUN_ID=exp019_seed44_fold0_consistancy_six_rank_dryrun bash rpcf/run_exp019_consistancy_six_rank.sh`
    通过。
  - 新版 full six-rank dry-run：
    `SEED=44 DRY_RUN=1 EXP019_RUN_ID=exp019_seed44_fold0_full_sixrank_dryrun bash rpcf/run_exp019.sh`
    通过。
- **正式命令：**
  ```bash
  nohup setsid bash -lc \
    "cd /home/yhj/pythonProject/EEGAP && \
     SEED=44 EXP019_RUN_ID=exp019_seed44_fold0_full_sixrank_20260617_2114 \
     bash rpcf/run_exp019.sh" \
    > logs/exp019/exp019_seed44_fold0_full_sixrank_20260617_2114/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 旧链式 run `exp019_seed44_fold0_all_20260617_2108` 已在 stage1 停止；
    停止原因：用户要求把 full five-method 内部的 consistancy 改为 six-rank 版本。
  - 新启动时间：2026-06-17 21:14（Asia/Shanghai）
  - run id：`exp019_seed44_fold0_full_sixrank_20260617_2114`
  - controller PID：`321947`
  - controller log：
    `logs/exp019/exp019_seed44_fold0_full_sixrank_20260617_2114/controller.log`
- **结果：**
  - 完成时间：2026-06-18 22:28（Asia/Shanghai）
  - 汇总表：
    `logs/exp019/exp019_seed44_fold0_full_sixrank_20260617_2114/five_methods/five_methods_table.md`
  - JSON：
    `logs/exp019/exp019_seed44_fold0_full_sixrank_20260617_2114/five_methods/five_methods_summary.json`

### EXP-020：eps=0.05 五方法三 seed 公平复验

- **日期：** 2026-06-21 至 2026-06-24
- **状态：** 已完成
- **相关 idea：** `IDEA-008`
- **目的：**
  - 检验 EXP-019 的五方法结论在更强扰动 `eps=0.05` 下是否稳定。
- **统一设置：**
  - dataset/model：`thubenchmark / EEGNet`
  - split：seed42/43/44，fold0，no-EA
  - 方法：Madry AT、consistancy six-rank、RPCF selective、
    RPCF all-layers、RPCF rank-weight uniform
  - attack：各方法自身 white-box AutoAttack，`eps=0.05`
  - consistancy/RPCF 训练 rank：`15,20,25,30,35,40`
  - 训练 cache：n512；最终净化：同 seed n512 子集、rank25/30
  - 除 epsilon 和独立输出目录外，其余参数与 EXP-019 相同。
- **实现：**
  - 单 seed pipeline：`rpcf/run_exp020.sh`
  - 三 seed 串行调度：`rpcf/run_exp020_all_seeds.sh`
  - 输出隔离到 `logs/exp020`、`ad_data/exp020` 和
    `purified_data/exp020`，不复用 eps0.03 产物。
- **正式命令：**
  ```bash
  EXP020_RUN_TAG=20260621_0839 \
  EXP020_CHAIN_ID=exp020_eps0p05_seeds42-44_20260621_0839 \
  nohup setsid bash rpcf/run_exp020_all_seeds.sh \
    > logs/exp020/exp020_eps0p05_seeds42-44_20260621_0839/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-06-21 08:39（Asia/Shanghai）
  - chain id：`exp020_eps0p05_seeds42-44_20260621_0839`
  - chain controller PID：`8204`
  - seed42 run：`exp020_seed42_fold0_eps0p05_20260621_0839`
  - seed43 run：`exp020_seed43_fold0_eps0p05_20260621_0839`
  - seed44 run：`exp020_seed44_fold0_eps0p05_20260621_0839`
  - seed42 完成时间：2026-06-22 09:27（Asia/Shanghai）。
  - seed43 完成时间：2026-06-23 10:31（Asia/Shanghai）。
  - seed44 完成时间：2026-06-24 14:09（Asia/Shanghai）。
  - 当前状态：三个 seed均已完成。
  - chain log：
    `logs/exp020/exp020_eps0p05_seeds42-44_20260621_0839/controller.log`
- **seed43 结果：**
  - 汇总表：
    `logs/exp020/exp020_seed43_fold0_eps0p05_20260621_0839/five_methods/five_methods_table.md`
  - JSON：
    `logs/exp020/exp020_seed43_fold0_eps0p05_20260621_0839/five_methods/five_methods_summary.json`

| Method | Full clean | Full AutoAttack | Rank 25 clean | Rank 25 adversarial | Rank 30 clean | Rank 30 adversarial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Madry AT | 91.19% | 68.57% | 87.89% | 72.07% | 90.04% | 71.88% |
| consistancy | 92.62% | 63.45% | 88.09% | 75.39% | 88.28% | 74.02% |
| RPCF selective | 93.21% | 49.76% | 90.23% | 76.76% | 91.21% | 74.61% |
| RPCF all-layers | 92.26% | 50.71% | 90.04% | 76.76% | 91.41% | 72.85% |
| RPCF rank-weight uniform | 93.33% | 48.69% | 90.23% | 77.34% | 91.60% | 73.83% |

- **seed43 阶段性观察：**
  - 相比 `eps=0.03`，所有方法的未净化和净化后 adversarial accuracy 均下降，
    符合攻击强度提高后的预期。
  - Madry AT 仍保持最好的未净化 AutoAttack accuracy（`68.57%`）。
  - RPCF 在净化分布上的适配优势更加明确：rank25 下三个 RPCF 变体均达到
    `76.76%` 以上，优于 consistancy 的 `75.39%` 和 Madry AT 的 `72.07%`。
  - rank30 下 RPCF selective 最好（`74.61%`），但相对 consistancy
    （`74.02%`）优势较小；all-layers 则低于 consistancy。
  - RPCF 的 full AutoAttack 下降到约 `49%~51%`，说明其收益高度依赖净化
    前处理，原始输入空间的鲁棒性退化仍是主要代价。
  - 三 seed 总结需等待 seed44 完成；当前结论仍为阶段性结果。

### EXP-021：RPCF_AT eps0.03 三 seed 复验

- **日期：** 2026-06-23 至 2026-06-24
- **状态：** 已完成
- **相关 idea：** `IDEA-009`
- **目的：**
  - 验证完整训练集在线 Madry AT 能否缓解原 RPCF 未净化 AutoAttack accuracy
    下降，同时保留 rank25/30 净化后的性能优势。
- **统一设置：**
  - dataset/model：`thubenchmark / EEGNet`
  - split：seed42/43/44，fold0，no-EA
  - epsilon：`0.03`
  - selective layers 与原 EXP-018 sensitivity 保持一致
  - 在线 AT：完整 train split、10-step PGD、step size `0.006`、
    batch size `128`、adversarial CE
  - cache loss：n512、rank `15,20,25,30,35,40`，保留 clean、
    clean-purified、adversarial-purified CE/KL，移除固定 `x_adv` CE/KL
  - RPCF fine-tuning：100 epochs、AdamW、lr `1e-4`、weight decay `1e-4`
  - 最终评估：完整 white-box AutoAttack、同 seed n512 rank25/30 净化
- **复用产物：**
  - AT checkpoint：
    `checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp018_full_20260612_124131_at_best.pth`
  - RPCF cache：
    `purified_data/exp018/rpcf_train/exp018_full_20260612_124131_six_rank.pth`
  - sensitivity：
    `logs/exp018/exp018_rpcf_no_early_stop_20260614_2357/sensitivity.json`
  - Madry AT 与 six-rank consistancy 的已有 rank25/30 净化产物。
- **实现与验证：**
  - pipeline：`rpcf/run_exp021_rpcf_at.sh`
  - `rpcf/finetune.py` 的原模式默认行为不变；RPCF_AT 由
    `--online_madry_at` 显式启用。
  - `python -m py_compile rpcf/finetune.py` 通过。
  - `bash -n rpcf/run_exp021_rpcf_at.sh` 通过。
  - `conda run -n torch --no-capture-output python -m unittest test_rpcf.py`
    通过，共 19 项测试。
  - smoke run：
    `exp021_rpcf_at_seed42_smoke_20260623_1259`。
  - smoke 已完整通过 1 epoch RPCF_AT、2-sample AutoAttack 和
    2-sample rank25/30 净化；history 确认 `online_at_loss` 已记录，
    `cached_adv_loss_enabled=false`，固定 `adv_ce/adv_kl` 均为 0。
- **正式命令：**
  ```bash
  EXP021_RUN_ID=exp021_rpcf_at_seed42_YYYYMMDD_HHMM \
  nohup setsid bash rpcf/run_exp021_rpcf_at.sh \
    > logs/exp021/exp021_rpcf_at_seed42_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-06-23 13:04（Asia/Shanghai）
  - run id：`exp021_rpcf_at_seed42_20260623_1303`
  - controller PID：`707704`
  - 完成时间：2026-06-23 22:33（Asia/Shanghai）
  - controller log：
    `logs/exp021/exp021_rpcf_at_seed42_20260623_1303/controller.log`
  - 与 EXP-020 seed44 并发运行；启动后总 GPU 显存占用约 `9.5/16 GB`，
    仍有约 `6.4 GB` 余量。
- **结果：**
  - 公平汇总：
    `logs/exp021/exp021_rpcf_at_seed42_20260623_1303/comparison/summary.json`
  - 完整测试集（n=840）上，RPCF_AT 的 clean / AutoAttack accuracy 为
    `93.93% / 77.98%`；普通 RPCF 为 `95.00% / 72.26%`。RPCF_AT 的
    未净化鲁棒准确率提高 `5.71` 个百分点，clean accuracy 降低 `1.07`
    个百分点。
  - 同一 n512 净化子集上，净化前 RPCF_AT 的 clean / AutoAttack accuracy
    为 `94.53% / 78.91%`；普通 RPCF 为 `96.09% / 72.66%`。
  - rank25 净化后，RPCF_AT 的 clean / adversarial accuracy 为
    `93.16% / 84.38%`，相对普通 RPCF 的 `92.97% / 83.59%` 分别变化
    `+0.20 / +0.78` 个百分点。
  - rank30 净化后，RPCF_AT 的 clean / adversarial accuracy 为
    `92.38% / 83.59%`，相对普通 RPCF 的 `93.75% / 84.18%` 分别变化
    `-1.37 / -0.59` 个百分点。
  - rank25/30 平均 purified adversarial accuracy 为 `83.98%`，普通 RPCF
    为 `83.89%`，整体基本持平。在线 Madry AT 达到了恢复未净化鲁棒性的主要
    目标，但没有形成稳定的净化后增益。
- **运行备注：**
  - 首次 full AutoAttack 使用 batch size 32 时因与 EXP-020 并发发生 OOM；
    后续以 batch size 8 成功重跑并完成全部评估。
- **seed43/44 补跑：**
  - 主脚本已参数化支持 seed42/43/44，并将 full AutoAttack 默认 batch size
    调整为 `8`，降低显存峰值和并发残留导致的 OOM 风险。
  - seed43 复用 EXP-019 的 AT checkpoint、RPCF six-rank cache、sensitivity，
    以及单独续跑的 six-rank consistancy checkpoint 和 rank25/30 净化结果。
  - seed44 复用
    `exp019_seed44_fold0_full_sixrank_20260617_2114` 的对应产物。
  - 串行调度脚本：`rpcf/run_exp021_seeds43_44.sh`。
  - 启动命令：
    ```bash
    EXP021_CHAIN_ID=exp021_rpcf_at_seeds43-44_YYYYMMDD_HHMM \
    nohup setsid bash rpcf/run_exp021_seeds43_44.sh \
      > logs/exp021/exp021_rpcf_at_seeds43-44_YYYYMMDD_HHMM/controller.log \
      2>&1 < /dev/null &
    ```
  - 正式启动时间：2026-06-24 09:54（Asia/Shanghai）。
  - chain id：`exp021_rpcf_at_seeds43-44_20260624_0955`。
  - controller PID：`1138062`。
  - seed43 完成时间：2026-06-24 13:25（Asia/Shanghai）。
  - seed44 完成时间：2026-06-24 16:37（Asia/Shanghai）。
  - controller log：
    `logs/exp021/exp021_rpcf_at_seeds43-44_20260624_0955/controller.log`。
- **seed43/44 结果：**

| Seed | Method | Full clean | Full AutoAttack | Rank25 purified clean | Rank25 purified adv | Rank30 purified clean | Rank30 purified adv |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 普通 RPCF | 93.45% | 71.67% | 91.02% | 81.64% | 90.43% | 80.86% |
| 43 | RPCF_AT | 91.79% | 79.40% | 88.28% | 82.23% | 90.23% | 81.84% |
| 44 | 普通 RPCF | 93.21% | 68.81% | 91.02% | 81.45% | 91.60% | 79.69% |
| 44 | RPCF_AT | 92.62% | 78.57% | 90.62% | 83.01% | 90.43% | 83.01% |

- **三 seed 汇总（mean ± sample std）：**

| Method | Full clean | Full AutoAttack | Rank25 purified clean | Rank25 purified adv | Rank30 purified clean | Rank30 purified adv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Madry AT | 92.70±1.03% | 78.37±1.00% | 90.76±2.07% | 81.51±0.41% | 91.15±1.30% | 81.38±0.63% |
| six-rank consistancy | 92.42±1.32% | 74.40±0.52% | 89.32±2.15% | 81.71±0.60% | 90.10±0.98% | 80.66±1.70% |
| 普通 RPCF | 93.89±0.97% | 70.91±1.85% | 91.67±1.13% | 82.23±1.19% | 91.93±1.68% | 81.58±2.33% |
| RPCF_AT | 92.78±1.08% | 78.65±0.72% | 90.69±2.44% | 83.20±1.09% | 91.02±1.19% | 82.81±0.90% |

- **三 seed 结论：**
  - 相比普通 RPCF，RPCF_AT 在 seed42/43/44 的 full AutoAttack 分别提高
    `5.71/7.74/9.76` 个百分点，平均提高 `7.74±2.02` 个百分点；跨 seed
    标准差也从 `1.85` 降至 `0.72`。
  - RPCF_AT 的 full AutoAttack 三 seed 均值为 `78.65%`，与 Madry AT 的
    `78.37%` 基本持平，说明在线 AT 稳定修复了普通 RPCF 的原始输入鲁棒性退化。
  - 相比普通 RPCF，RPCF_AT 的 rank25/rank30 purified adversarial accuracy
    平均分别提高 `0.98/1.24` 个百分点；六个 seed×rank 条件中有五个提高，
    唯一下降是 seed42 rank30 的 `-0.59` 个百分点。
  - RPCF_AT 的 rank25/rank30 purified adversarial 三 seed均值为
    `83.20%/82.81%`，高于 Madry AT 的 `81.51%/81.38%`，也高于 six-rank
    consistancy 的 `81.71%/80.66%`。
  - 主要代价是 clean accuracy：相较普通 RPCF，full clean 平均下降
    `1.11±0.54` 个百分点；purified clean 在 rank25/30 平均分别下降
    `0.98/0.91` 个百分点。

### EXP-022：RPCF_AT eps0.05 三 seed 复验

- **日期：** 2026-06-25 至 2026-06-26
- **状态：** 已完成
- **相关 idea：** `IDEA-010`
- **目的：**
  - 验证 EXP-021 的 RPCF_AT 收益在更强 `eps=0.05` 下是否稳定。
  - 检查在线 Madry AT 能否修复 EXP-020 普通 RPCF selective 的 full
    AutoAttack 退化，同时保留 rank25/30 净化后优势。
- **统一设置：**
  - dataset/model：`thubenchmark / EEGNet`
  - split：seed42/43/44，fold0，no-EA
  - epsilon：`0.05`
  - RPCF_AT：selective layers、dynamic rank schedule、100 epochs
  - 在线 AT：完整训练 split、10-step PGD、step size `0.01=eps/5`、
    batch size `128`
  - cache loss：复用 n512 rank `15,20,25,30,35,40`，保留 clean、
    clean-purified、adversarial-purified CE/KL，不使用固定 `x_adv` CE/KL
  - 最终评估：自身完整 white-box AutoAttack、同 seed n512 rank25/30 净化
- **复用 EXP-020 产物：**
  - 每个 seed 的 Madry AT checkpoint。
  - 每个 seed 的 RPCF six-rank cache 与 sensitivity。
  - 每个 seed 的 Madry AT、six-rank consistancy checkpoint 和 rank25/30
    净化结果，用于最终公平汇总。
  - 不重新运行 AT 训练、cache 生成或 sensitivity 分析。
- **预期输出：**
  - 单 seed pipeline：`rpcf/run_exp022_rpcf_at.sh`
  - 三 seed 串行调度：`rpcf/run_exp022_all_seeds.sh`
  - 日志：`logs/exp022/`
  - 攻击数据：`ad_data/exp022/`
  - 净化结果：`purified_data/exp022/eval/`
- **实现与验证：**
  - `bash -n rpcf/run_exp022_rpcf_at.sh` 通过。
  - `bash -n rpcf/run_exp022_all_seeds.sh` 通过。
  - `python -m py_compile rpcf/compare_exp022.py` 通过。
  - seed42/43/44 dry-run 均通过，且所有 EXP-020 复用产物存在。
  - 四方法汇总器已使用 EXP-021 seed42 结果做协议兼容性验证，能够正确输出
    Madry AT、six-rank consistancy、普通 RPCF selective 和 RPCF_AT。
  - smoke run：`exp022_seed42_smoke_20260625_1458`。
  - smoke 已完整通过 1 epoch RPCF_AT、2-sample AutoAttack 和
    2-sample rank25/30 净化。
- **正式命令：**
  ```bash
  EXP022_RUN_TAG=YYYYMMDD_HHMM \
  EXP022_CHAIN_ID=exp022_rpcf_at_eps0p05_seeds42-44_YYYYMMDD_HHMM \
  nohup setsid bash rpcf/run_exp022_all_seeds.sh \
    > logs/exp022/exp022_rpcf_at_eps0p05_seeds42-44_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-06-25 15:00（Asia/Shanghai）。
  - chain id：`exp022_rpcf_at_eps0p05_seeds42-44_20260625_1500`。
  - controller PID：`15867`。
  - seed42 run：`exp022_seed42_fold0_eps0p05_20260625_1500`。
  - seed43 run：`exp022_seed43_fold0_eps0p05_20260625_1500`。
  - seed44 run：`exp022_seed44_fold0_eps0p05_20260625_1500`。
  - seed42 完成时间：2026-06-25 18:03（Asia/Shanghai）。
  - seed43 完成时间：2026-06-25 21:06（Asia/Shanghai）。
  - seed44 完成时间：2026-06-26 00:08（Asia/Shanghai）。
  - controller log：
    `logs/exp022/exp022_rpcf_at_eps0p05_seeds42-44_20260625_1500/controller.log`。
- **结果：**
  - 三个 seed 均成功生成 RPCF_AT checkpoint、自身 white-box AutoAttack、
    rank25/30 净化结果和四方法严格汇总。

| Seed | Method | Full clean | Full AutoAttack | Rank25 purified clean | Rank25 purified adv | Rank30 purified clean | Rank30 purified adv |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 普通 RPCF | 94.05% | 49.76% | 91.99% | 76.37% | 92.38% | 74.41% |
| 42 | RPCF_AT | 93.21% | 68.33% | 90.82% | 79.30% | 91.41% | 79.30% |
| 43 | 普通 RPCF | 93.21% | 49.76% | 90.23% | 76.76% | 91.21% | 74.61% |
| 43 | RPCF_AT | 91.55% | 67.98% | 87.50% | 77.73% | 90.04% | 77.15% |
| 44 | 普通 RPCF | 92.86% | 47.86% | 91.21% | 72.85% | 90.62% | 69.14% |
| 44 | RPCF_AT | 92.14% | 68.81% | 90.23% | 81.05% | 90.23% | 79.30% |

- **三 seed 汇总（mean ± sample std）：**

| Method | Full clean | Full AutoAttack | Rank25 purified clean | Rank25 purified adv | Rank30 purified clean | Rank30 purified adv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Madry AT | 92.34±1.13% | 68.81±0.41% | 89.52±1.44% | 71.35±0.63% | 90.56±0.49% | 70.90±1.37% |
| six-rank consistancy | 92.58±0.07% | 62.10±1.44% | 89.32±1.08% | 75.20±0.70% | 90.17±1.66% | 74.54±0.49% |
| 普通 RPCF | 93.37±0.61% | 49.13±1.10% | 91.15±0.88% | 75.33±2.15% | 91.41±0.90% | 72.72±3.10% |
| RPCF_AT | 92.30±0.84% | 68.37±0.42% | 89.52±1.77% | 79.36±1.66% | 90.56±0.74% | 78.58±1.24% |

- **结论：**
  - 相比普通 RPCF，RPCF_AT 的 full AutoAttack 在 seed42/43/44 分别提高
    `18.57/18.21/20.95` 个百分点，平均提高 `19.25±1.49` 个百分点。
  - RPCF_AT 的 full AutoAttack 为 `68.37±0.42%`，与 Madry AT 的
    `68.81±0.41%` 基本持平，说明在线 AT 在更强 `eps=0.05` 下仍稳定修复
    原始输入鲁棒性退化。
  - RPCF_AT 的 rank25/rank30 purified adversarial accuracy 为
    `79.36±1.66%/78.58±1.24%`，相比普通 RPCF 平均提高
    `4.04/5.86` 个百分点。
  - 相比 Madry AT，RPCF_AT 的 rank25/rank30 purified adversarial accuracy
    平均提高 `8.01/7.68` 个百分点；相比 six-rank consistancy 分别提高
    `4.17/4.04` 个百分点。
  - clean 代价仍存在：相较普通 RPCF，full clean 平均下降 `1.07` 个百分点，
    rank25/30 purified clean 分别下降 `1.63/0.85` 个百分点；但 RPCF_AT 的
    full clean 与 Madry AT 基本相同。

### EXP-023：EEG_TNP + RPCF_AT 的 BPDA+PGD-10 adaptive attack

- **日期：** 2026-06-26
- **状态：** 已实现，smoke 已通过；正式三 seed 结果 Pending
- **相关 idea：** `IDEA-011`
- **目的：**
  - 在 adaptive white-box 假设下评估 EXP-021 的 EEG_TNP+RPCF_AT 组合防御。
  - 检查攻击直接穿过 EEG_TNP 净化器时，rank25/30 净化后鲁棒准确率是否仍能保持优势。
- **统一设置：**
  - dataset/model：`thubenchmark / EEGNet`
  - split：seed42/43/44，fold0，no-EA
  - checkpoint：复用 EXP-021 RPCF_AT，不重新微调
  - attack：BPDA+PGD-10，`eps=0.03`，`pgd_alpha=0.006`
  - BPDA：forward 真实 EEG_TNP 净化 + RPCF_AT 推理，backward 将 EEG_TNP 视作 identity
  - ranks：rank25 和 rank30 分开攻击、分开汇总
  - 样本量：n512，抽样规则为 `seed + fold * 1000`
- **预期输出：**
  - 单 seed pipeline：`rpcf/run_exp023_bpda_pgd.sh`
  - 三 seed 串行调度：`rpcf/run_exp023_all_seeds.sh`
  - baseline PGD-10 补充：`rpcf/run_exp023_baseline_pgd10.sh`
  - 汇总器：`rpcf/compare_exp023.py`
  - 日志：`logs/exp023/`
  - adaptive attack artifact：`ad_data/exp023/`
- **实现与验证：**
  - 新增 `rpcf/evaluate_bpda_pgd.py`，通过 BPDA autograd wrapper 在 forward
    中执行真实 EEG_TNP，并在 backward 中对 EEG_TNP 使用 identity gradient。
  - 新增 `rpcf/run_exp023_bpda_pgd.sh` 和 `rpcf/run_exp023_all_seeds.sh`。
  - 新增 `rpcf/compare_exp023.py`，支持单 seed 或三 seed 汇总，并可对照 EXP-021
    非 adaptive 净化结果。
  - 新增 `rpcf/run_exp023_baseline_pgd10.sh`，用于在同一 seed/fold/n512 子集上补跑
    Madry/TRADES/FBF 面对 PGD-10 的 baseline 结果；`attack.py` 增加可选
    `--pgd_steps` 和 `--pgd_alpha`，默认不传时保持原 PGD 行为。
  ```bash
  python -m py_compile rpcf/evaluate_bpda_pgd.py rpcf/compare_exp023.py
  bash -n rpcf/run_exp023_bpda_pgd.sh
  bash -n rpcf/run_exp023_all_seeds.sh
  bash -n rpcf/run_exp023_baseline_pgd10.sh
  DRY_RUN=1 EXP023_SEED=42 bash rpcf/run_exp023_bpda_pgd.sh
  DRY_RUN=1 EXP023_BASELINE_RUN_ID=exp023_baseline_pgd10_seed42_dryrun \
    bash rpcf/run_exp023_baseline_pgd10.sh
  SMOKE=1 EXP023_SEED=42 EXP023_RUN_ID=exp023_seed42_smoke bash rpcf/run_exp023_bpda_pgd.sh
  ```
  - 以上静态检查和 dry-run 已通过。
  - smoke run：`exp023_seed42_smoke_20260626_1100`，已成功生成 rank25/30 的
    2-sample BPDA+PGD-10 artifact 和 `comparison/summary.json`。
  - smoke 过程中发现 `autograd.Function.forward` 会禁用 EEG_TNP 内部训练梯度；
    已在 BPDA forward 中用 `torch.enable_grad()` 仅恢复净化器内部优化，外层 BPDA
    backward 仍保持 identity 透传。
- **正式命令：**
  ```bash
  EXP023_CHAIN_ID=exp023_bpda_pgd10_eps0p03_seeds42-44_YYYYMMDD_HHMM \
  nohup setsid bash rpcf/run_exp023_all_seeds.sh \
    > logs/exp023/exp023_bpda_pgd10_eps0p03_seeds42-44_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录（seed42，rank25/30 并行）：**
  - 启动时间：2026-06-26 11:10（Asia/Shanghai）。
  - run id：`exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108`。
  - controller PID：`289421`。
  - 运行范围：seed42、rank25/30、n512，两个 rank 并行。
  - controller log：
    `logs/exp023/exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108/controller.log`
  - rank25 log：
    `logs/exp023/exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108/stage1_bpda_pgd_rank25.log`
  - rank30 log：
    `logs/exp023/exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108/stage1_bpda_pgd_rank30.log`
  - 目标产物：
    `ad_data/exp023/exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108_rpcf_at_bpda_pgd10_rank25.pth`
    和
    `ad_data/exp023/exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108_rpcf_at_bpda_pgd10_rank30.pth`。
- **正式启动记录（seed43/44 串行补跑）：**
  - 启动时间：2026-06-29 12:06（Asia/Shanghai）。
  - chain id：`exp023_bpda_pgd10_eps0p03_seeds43-44_20260629_1151`。
  - controller PID：`30062`。
  - 运行范围：seed43/44、rank25/30、n512；脚本按 seed 与 rank 串行运行，避免单 GPU 并发。
  - controller log：
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seeds43-44_20260629_1151/controller.log`
  - 当前阶段：seed43 rank25 `BPDA+PGD-10` 运行中。
  - 预期产物：
    `ad_data/exp023/exp023_bpda_pgd10_seed43_20260629_1151_rpcf_at_bpda_pgd10_rank25.pth`、
    `ad_data/exp023/exp023_bpda_pgd10_seed43_20260629_1151_rpcf_at_bpda_pgd10_rank30.pth`、
    `ad_data/exp023/exp023_bpda_pgd10_seed44_20260629_1151_rpcf_at_bpda_pgd10_rank25.pth`、
    `ad_data/exp023/exp023_bpda_pgd10_seed44_20260629_1151_rpcf_at_bpda_pgd10_rank30.pth`。
  - 预期汇总：
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seeds43-44_20260629_1151/comparison/summary.json`。
  - 三 seed 自动汇总 watcher PID：`30705`；等待 seed43/44 controller 结束后，
    将 seed42 既有 rank25/30 BPDA artifact 与 seed43/44 新产物合并汇总到
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seeds43-44_20260629_1151/comparison_three_seed/`。
  - watcher log：
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seeds43-44_20260629_1151/three_seed_summary_waiter.log`。
  - 2026-06-29 14:39 更新：`rpcf/run_exp023_bpda_pgd.sh` 已增加
    `PARALLEL_RANKS`，默认 `1`，后续新启动的单 seed pipeline 会并行运行
    rank25/30。当前 seed43 rank25 已在旧脚本中运行，未中断；seed44 预计会使用
    新脚本并行 rank25/30。
  - 2026-06-30 更新：该 controller 在 seed43 rank25/30 产物生成后退出，
    未进入 seed44；三 seed watcher 因缺少 seed44 artifact 报
    `FileNotFoundError`。seed43 产物已保留并纳入后续汇总。
- **seed44 单独补跑启动记录：**
  - 启动时间：2026-06-30 00:10（Asia/Shanghai）。
  - run id：`exp023_bpda_pgd10_seed44_20260630_1030`。
  - controller PID：`178638`。
  - 运行范围：seed44、rank25/30、n512；`PARALLEL_RANKS=1`，rank25/30
    并行运行。
  - controller log：
    `logs/exp023/exp023_bpda_pgd10_seed44_20260630_1030/controller.log`
  - rank logs：
    `logs/exp023/exp023_bpda_pgd10_seed44_20260630_1030/stage1_bpda_pgd_rank25.log`、
    `logs/exp023/exp023_bpda_pgd10_seed44_20260630_1030/stage1_bpda_pgd_rank30.log`。
  - 预期产物：
    `ad_data/exp023/exp023_bpda_pgd10_seed44_20260630_1030_rpcf_at_bpda_pgd10_rank25.pth`、
    `ad_data/exp023/exp023_bpda_pgd10_seed44_20260630_1030_rpcf_at_bpda_pgd10_rank30.pth`。
  - 三 seed 自动汇总 watcher PID：`179583`；等待 seed44 controller 结束后，
    将 seed42/43/44 的 rank25/30 BPDA artifact 合并汇总到
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seed44_20260630_1030/comparison_three_seed/`。
  - watcher log：
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seed44_20260630_1030/three_seed_summary_waiter.log`。
  - 结果：Pending。
- **baseline PGD-10 补充启动记录（seed42，Madry/TRADES/FBF）：**
  - 启动时间：2026-06-26 15:01（Asia/Shanghai）。
  - run id：`exp023_baseline_pgd10_seed42_after_bpda_20260626_1500`。
  - controller PID：`392807`。
  - 运行策略：等待 seed42 rank25/30 BPDA artifact 均生成后，串行运行
    `madry`、`trades`、`fbf` 的 PGD-10。
  - 参数：`eps=0.03`，`pgd_steps=10`，`pgd_alpha=0.006`，
    `attack_sample_num=512`，fold0，no-EA。
  - controller log：
    `logs/exp023/exp023_baseline_pgd10_seed42_after_bpda_20260626_1500/controller.log`
  - 命令：
    ```bash
    EXP023_BASELINE_RUN_ID=exp023_baseline_pgd10_seed42_after_bpda_20260626_1500 \
    WAIT_FOR_RUN_ID=exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108 \
    SKIP_EXISTING=1 \
    nohup setsid bash rpcf/run_exp023_baseline_pgd10.sh \
      > logs/exp023/exp023_baseline_pgd10_seed42_after_bpda_20260626_1500/controller.log \
      2>&1 < /dev/null &
    ```
  - 预期产物：
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_madry_pgd_eps0.03_seed42_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_trades_pgd_eps0.03_seed42_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_fbf_pgd_eps0.03_seed42_fold0.pth`。
- **baseline PGD-10 环境修复：**
  - 首次 baseline controller 未进入 `torch` 环境，`attack.py` 报
    `ModuleNotFoundError: No module named 'torch'` 后退出。
  - 已将 `rpcf/run_exp023_baseline_pgd10.sh` 改为使用
    `conda run -n torch --no-capture-output python -u attack.py`。
  - 修复后重启 run：
    `exp023_baseline_pgd10_seed42_after_bpda_conda_20260626_2131`，
    controller PID `490532`，已完成。
- **baseline PGD-10 seed43/44 追加排队：**
  - 请求：在 seed44 BPDA+PGD-10 完成后，补跑 seed43/44 的
    Madry/TRADES/FBF 标准 PGD-10 baseline。
  - 当前检查结果：seed43/44 的 Madry AT checkpoint 存在；seed43/44 的
    TRADES/FBF checkpoint 原本不在 `checkpoints/` 中，不能用 seed42 checkpoint
    代替，需先补训再评估。
  - 已更新 `rpcf/run_exp023_baseline_pgd10.sh`：支持
    `EXP023_BASELINE_${STRATEGY}_CHECKPOINT` 显式覆盖，并为 seed43/44 的
    Madry AT 设置 EXP-019 tagged checkpoint 默认路径。
  - 2026-06-30 01:03 的 Madry-only watcher（PID `190666`）已在尚未开始计算时
    终止，避免与后续训练/评估队列抢占 GPU。
  - 新增调度脚本：
    `rpcf/run_exp023_train_baselines_then_pgd10.sh`。
  - 新队列启动时间：2026-06-30 01:08（Asia/Shanghai）。
  - run id：`exp023_baseline_train_then_pgd10_seeds43-44_20260630_0102`。
  - controller PID：`192078`；等待 seed44 BPDA controller `178638` 退出后，
    串行执行：
    seed43 TRADES 训练、seed43 FBF 训练、seed44 TRADES 训练、seed44 FBF 训练，
    然后运行 seed43/44 的 Madry/TRADES/FBF PGD-10 baseline。
  - controller log：
    `logs/exp023/exp023_baseline_train_then_pgd10_seeds43-44_20260630_0102/controller.log`
  - 预期新训练 checkpoint：
    `checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_trades_eps0.03_43_fold0_exp023_baseline_seed43-44_20260630_0102_best.pth`、
    `checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_fbf_eps0.03_43_fold0_exp023_baseline_seed43-44_20260630_0102_best.pth`、
    `checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_trades_eps0.03_44_fold0_exp023_baseline_seed43-44_20260630_0102_best.pth`、
    `checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_fbf_eps0.03_44_fold0_exp023_baseline_seed43-44_20260630_0102_best.pth`。
  - 预期 PGD-10 产物：
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_madry_pgd_eps0.03_seed43_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_trades_pgd_eps0.03_seed43_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_fbf_pgd_eps0.03_seed43_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_madry_pgd_eps0.03_seed44_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_trades_pgd_eps0.03_seed44_fold0.pth`、
    `ad_data/thubenchmark_eegnet_no_ea_exp023_baseline_pgd10_fbf_pgd_eps0.03_seed44_fold0.pth`。
  - 完成时间：2026-06-30 09:14（Asia/Shanghai）。
  - 结果：Completed。
- **三 seed BPDA+PGD-10 adaptive attack 结果：**
  - 汇总表：
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seed44_20260630_1030/comparison_three_seed/comparison.md`
  - JSON：
    `logs/exp023/exp023_bpda_pgd10_eps0p03_seed44_20260630_1030/comparison_three_seed/summary.json`

| Rank | BPDA purified clean | BPDA purified adv | Attack MSE | Non-adaptive purified adv | Δ adv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 90.36%±2.44% | 80.14%±0.79% | 0.00083054±0.00000122 | 83.20%±1.09% | -3.06 pp |
| 30 | 91.34%±2.05% | 81.05%±1.53% | 0.00084055±0.00000186 | 82.81%±0.90% | -1.76 pp |

- **三 seed 标准 PGD-10 baseline 结果：**
  - logs：
    `log_attack/attack_thubenchmark_eegnet_no_ea_{madry,trades,fbf}_pgd_0.03_{42,43,44}_fold0_exp023_baseline_pgd10.log`

| Method | Clean accuracy | PGD-10 adv accuracy | Attack MSE | Note |
| --- | ---: | ---: | ---: | --- |
| Madry AT | 92.58%±1.60% | 79.30%±1.55% | 0.000877±0.000001 | seed42/43/44 n512 |
| TRADES | 93.10%±1.77% | 65.23%±2.64% | 0.000868±0.000002 | seed43/44 为本次补训 checkpoint |
| FBF | 94.66%±1.47% | 55.66%±3.65% | 0.000861±0.000002 | seed43/44 为本次补训 checkpoint |

- **seed42 当前结果：**
  - BPDA 汇总：
    `logs/exp023/exp023_bpda_pgd10_seed42_rank25-30_parallel_20260626_1108/comparison/comparison.md`
  - baseline PGD-10 logs：
    `log_attack/attack_thubenchmark_eegnet_no_ea_madry_pgd_0.03_42_fold0_exp023_baseline_pgd10.log`、
    `log_attack/attack_thubenchmark_eegnet_no_ea_trades_pgd_0.03_42_fold0_exp023_baseline_pgd10.log`、
    `log_attack/attack_thubenchmark_eegnet_no_ea_fbf_pgd_0.03_42_fold0_exp023_baseline_pgd10.log`。

| Method | Clean accuracy | PGD-10 / BPDA adv accuracy | Attack MSE | Note |
| --- | ---: | ---: | ---: | --- |
| EEG_TNP+RPCF_AT rank25 | 93.16% | 80.86% | 0.000829 | BPDA+PGD-10；非 adaptive purified adv 为 84.38%，下降 3.52 pp |
| EEG_TNP+RPCF_AT rank30 | 93.36% | 82.81% | 0.000838 | BPDA+PGD-10；非 adaptive purified adv 为 83.59%，下降 0.78 pp |
| Madry AT | 94.34% | 79.88% | 0.000877 | 标准 PGD-10 baseline，n512 |
| TRADES | 95.12% | 67.77% | 0.000868 | 标准 PGD-10 baseline，n512 |
| FBF | 96.09% | 56.25% | 0.000859 | 标准 PGD-10 baseline，n512 |

- **seed43 当前结果：**
  - 临时单 seed 汇总：
    `/tmp/exp023_seed43_current_summary/comparison.md`

| Method | Clean accuracy | BPDA adv accuracy | Attack MSE | Note |
| --- | ---: | ---: | ---: | --- |
| EEG_TNP+RPCF_AT rank25 | 88.67% | 79.30% | 0.000831 | BPDA+PGD-10；非 adaptive purified adv 为 82.23%，下降 2.93 pp |
| EEG_TNP+RPCF_AT rank30 | 89.26% | 80.08% | 0.000841 | BPDA+PGD-10；非 adaptive purified adv 为 81.84%，下降 1.76 pp |

- **结果状态：**
  - seed42 的 rank25/30 BPDA+PGD-10 和 Madry/TRADES/FBF PGD-10 baseline 已完成。
  - seed43 的 rank25/30 BPDA+PGD-10 已完成。
  - seed44 的 rank25/30 BPDA+PGD-10 已完成。
  - seed43/44 的 TRADES/FBF baseline checkpoint 补训与 Madry/TRADES/FBF
    PGD-10 baseline 已完成。
  - EXP-023 三 seed adaptive attack 与 baseline 对照均已完成。

### EXP-024：其他 backbone 的 RPCF_AT 与 baseline 全流程测试

- **目标：**
  - 在 `thubenchmark`、no-EA、fold0、`eps=0.03` 条件下，补全
    `tsception`、`atcnet`、`conformer` 三个 backbone 的测试。
  - 每个 backbone 覆盖 Madry AT、white-box AutoAttack、rank25/30 净化、
    RPCF_AT 微调、RPCF_AT attack/净化测试，以及 clean/TRADES/FBF baseline
    训练和攻击评估。
- **协议：**
  - seed：`42`
  - fold：`0`
  - 训练 rank：`15,20,25,30,35,40`
  - 净化评估 rank：`25,30`
  - RPCF_AT：selective layers、dynamic rank schedule、`online_madry_at`
  - attack：默认 `autoattack`
  - train cache sample：`512`
  - purification eval sample：`512`
- **实现：**
  - 单 backbone pipeline：`rpcf/run_exp024_backbone.sh`
  - 三 backbone 串行调度：`rpcf/run_exp024_all_backbones.sh`
  - 汇总器：`rpcf/compare_exp024.py`
  - 日志：`logs/exp024/`
  - 攻击 artifact：`ad_data/exp024/`
  - 训练/评估净化 artifact：`purified_data/exp024/`
- **验证：**
  ```bash
  bash -n rpcf/run_exp024_backbone.sh
  bash -n rpcf/run_exp024_all_backbones.sh
  python -m py_compile rpcf/compare_exp024.py
  DRY_RUN=1 SMOKE=1 EXP024_MODEL=conformer EXP024_RUN_ID=exp024_conformer_dryrun \
    bash rpcf/run_exp024_backbone.sh
  DRY_RUN=1 EXP024_MODELS='tsception atcnet conformer' \
    EXP024_CHAIN_ID=exp024_other_backbones_dryrun \
    bash rpcf/run_exp024_all_backbones.sh
  ```
  - 以上静态检查和 dry-run 已通过。
- **正式命令：**
  ```bash
  EXP024_RUN_TAG=YYYYMMDD_HHMM \
  EXP024_CHAIN_ID=exp024_other_backbones_seed42_YYYYMMDD_HHMM \
  nohup setsid bash rpcf/run_exp024_all_backbones.sh \
    > logs/exp024/exp024_other_backbones_seed42_YYYYMMDD_HHMM/controller.log \
    2>&1 < /dev/null &
  ```
- **正式启动记录：**
  - 启动时间：2026-07-01 12:08（Asia/Shanghai）。
  - chain id：`exp024_other_backbones_seed42_20260701_1208`。
  - chain controller PID：`617922`。
  - 当前子流程：`exp024_other_backbones_seed42_20260701_1208_tsception`。
  - 当前子流程 controller PID：`617927`。
  - 当前训练进程 PID：`617942`。
  - 当前阶段：`tsception` 的 `stage1_train_madry_at` 运行中。
  - controller log：
    `logs/exp024/exp024_other_backbones_seed42_20260701_1208/controller.log`
  - 当前阶段日志：
    `logs/exp024/exp024_other_backbones_seed42_20260701_1208_tsception/stage1_train_madry_at.log`
  - 运行范围：`tsception`、`atcnet`、`conformer` 串行；每个 backbone 跑完整
    9-stage pipeline。
  - 启动后检查：`nvidia-smi` 显示训练进程 `617942` 占用约 `15384 MiB`。
- **范围调整记录：**
  - 2026-07-01 15:08（Asia/Shanghai）：考虑三 backbone 全流程耗时过长，本轮先只跑
    `tsception`。
  - 已向外层 all-backbones controller PID `617922` 发送 `TERM`，阻止当前
    `tsception` 完成后继续进入 `atcnet` 和 `conformer`。
  - `tsception` 子流程 PID `617927` 和训练进程 PID `617942` 保持运行。
  - 调整后状态：`tsception stage1_train_madry_at` 运行中；训练日志显示已到
    epoch 72，`Val Acc=0.7107`、`Test Acc=0.6881`、`Robust Acc=0.3143`。
  - `atcnet` 和 `conformer` 本轮未启动，结果保持 `Pending`。
- **并行重启记录：**
  - 日期：2026-07-02 15:33 至 15:42（Asia/Shanghai）。
  - 本轮不修改 pipeline 代码，直接并行启动三个单-backbone run；每个 run 仍使用 `rpcf/run_exp024_backbone.sh` 的 9-stage pipeline。
  - run id / GPU 分配：
    - `exp024_parallel_seed42_20260702_153337_tsception`：GPU 0，controller PID `53543`。
    - `exp024_parallel_seed42_20260702_153337_atcnet_retry2`：GPU 1，controller PID `61669`。
    - `exp024_parallel_seed42_20260702_153337_conformer_retry2`：GPU 2，controller PID `61724`。
  - 由于本服务器 GPU 0-3 每张约 10GB，本轮通过环境变量将 `AT_BATCH_SIZE=64`、`BASELINE_BATCH_SIZE=64`、`ONLINE_AT_BATCH_SIZE=64`、`RPCF_BATCH_SIZE=32`、`ATTACK_BATCH_SIZE=16` 传入脚本，避免显存不足；未修改训练代码。
  - 首次后台启动因非交互 shell 找不到 `conda` 失败；随后通过 `source /home/yihangjie/miniconda3/etc/profile.d/conda.sh` 后重启。
  - `atcnet` 和 `conformer` 首次与 `tsception` 并行读写 TorchEEG cache 时遇到 `io_path corrupted` / `lmdb.MapResizedError`；等待 `tsception` 完成 `cached_data/thubenchmark` 构建后，使用 retry2 重新启动成功。
  - 当前日志：
    - `logs/exp024/exp024_parallel_seed42_20260702_153337_tsception/stage1_train_madry_at.log`
    - `logs/exp024/exp024_parallel_seed42_20260702_153337_atcnet_retry2/stage1_train_madry_at.log`
    - `logs/exp024/exp024_parallel_seed42_20260702_153337_conformer_retry2/stage1_train_madry_at.log`
  - 启动后检查：`nvidia-smi` 显示 GPU0/1/2 分别被三个 stage1 训练任务占用，GPU3 保留空闲。
  - 2026-07-03 10:58 状态检查：三个 backbone 的 Madry AT checkpoint 均已生成；`tsception` 仍在 stage2 `rpcf.generate_cache`，已生成 `base.pth` 和 rank15/20/25 以及 rank30 partial shard。`atcnet` 与 `conformer` 的首次 stage2 因 TN 净化内部 `cuda:0` 与外部 `cuda:1/2` 混用失败。
  - 2026-07-03 10:59：不改代码，使用 `CUDA_VISIBLE_DEVICES=1/2` 隔离物理 GPU，并在脚本内设置 `GPU_ID=0 START_STAGE=2`，分别从 stage2 续跑 `atcnet` 和 `conformer`；新 controller PID 为 `335381` 和 `335408`。
  - 2026-07-03 11:15：按用户要求暂停 EXP-024，向 `53543`、`335381`、`335408` 对应进程组发送 `TERM`；确认 `run_exp024_backbone` / `rpcf.generate_cache` 相关进程已退出。当前 partial cache 保留，可后续续跑。
  - 2026-07-03 11:17：更新 `rpcf/run_exp024_all_backbones.sh`，加入 GPU 空闲检测；默认在 `EXP024_GPU_IDS` / `GPU_IDS` 指定的 GPU 中选择显存占用不超过 `EXP024_GPU_IDLE_MAX_USED_MB=100` 的卡。子流程统一以 `CUDA_VISIBLE_DEVICES=<physical_gpu> GPU_ID=0` 启动，避免 TN 净化内部设备混用。若仅运行 `START_STAGE=2 STOP_STAGE=2` 或 `START_STAGE=6 STOP_STAGE=6`，默认每张空闲 GPU 允许 `EXP024_TN_SLOTS_PER_IDLE_GPU=2` 个 TN 净化任务；同时支持 `EXP024_RUN_ID_TSCEPTION`、`EXP024_RUN_ID_ATCNET`、`EXP024_RUN_ID_CONFORMER` 覆盖单个 backbone 的 run id，便于续跑已有 partial cache。
  - 2026-07-03 11:24：使用新调度器续跑 stage2：`EXP024_CHAIN_ID=exp024_parallel_seed42_20260702_153337_stage2_resume`，`START_STAGE=2 STOP_STAGE=2`，`EXP024_GPU_IDS=0,1,2,3`，并复用三个已有 run id。启动时 GPU1-3 忙，调度器先在空闲 GPU0 启动 `tsception` 与 `atcnet` 两个 TN 任务；随后 GPU1 释放后自动启动 `conformer`。当前 stage2 继续运行，结果仍为 Pending。
  - 2026-07-03 11:33：再次暂停 `exp024_parallel_seed42_20260702_153337_stage2_resume`，停止旧的 per-backbone stage2 进程，保留 `*.work/base.pth`、已完成 rank shard 与 partial shard。
  - 2026-07-03 11:38：为 RPCF cache 增加 rank 级并行能力：`rpcf.generate_cache` 新增 `--base_only`、`--rank_shard_only`、`--finalize_only` 三个显式模式；新增 `rpcf/run_exp024_stage2_rank_parallel.sh`，先确保 base shard，再将 `tsception`、`atcnet`、`conformer` 的 6 个 rank shard 分别作为独立任务调度，默认每张空闲 GPU 跑 2 个 rank，且 `EXP024_MAX_ACTIVE_RANK_JOBS` 默认限制总 rank 并发为 6。已通过 `python3 -m py_compile rpcf/generate_cache.py`、`bash -n rpcf/run_exp024_stage2_rank_parallel.sh` 和 `DRY_RUN=1` 检查；结果仍为 Pending。
  - 2026-07-03 11:40：使用 rank 级并行脚本正式续跑 stage2，chain id 为 `exp024_parallel_seed42_20260702_153337_stage2_rank_parallel`，外层 PID `391732`。启动命令使用 `EXP024_GPU_IDS=0,1,2,3`、`EXP024_RANK_SLOTS_PER_IDLE_GPU=2`、`EXP024_MAX_ACTIVE_RANK_JOBS=6`，并复用三个已有 run id。启动后检查显示首批 6 个 rank worker 已派发：GPU0/1/2 各 2 个；调度器等待 rank 完成后继续投放剩余 shard。
  - 2026-07-03 11:45：发现本机实际有 8 张 GPU，GPU4/5/6/7 空闲，按用户要求将 `conformer` 也提前跑上。为避免原 rank controller 后续重复派发同一 conformer rank，先对外层 controller PID `391732` 发送 `STOP`，保留已运行的 tsception/atcnet worker 继续执行；随后手动使用 `--rank_shard_only` 启动 conformer rank15/20/25/30/35/40，分配为 GPU4 两个、GPU5 两个、GPU6 两个，日志为 `logs/exp024/exp024_parallel_seed42_20260702_153337_conformer_retry2/stage2_rank*_manual.log`。确认 GPU4/5/6 分别已有两个 conformer worker；另起 watcher PID `405334`，等待 conformer 手动 rank worker 结束后自动 `CONT` 原 controller。
  - 2026-07-03 16:29：检查 stage3 自动衔接与多 GPU 调度。`rpcf/run_exp024_all_backbones.sh` 会在 `EXP024_GPU_IDS=0,1,2,3,4,5,6,7` 中按显存空闲选择物理 GPU，并以 `CUDA_VISIBLE_DEVICES=<physical> GPU_ID=0` 启动各 backbone；`rpcf/analyze_sensitivity.py`、`rpcf.finetune`、`rpcf.evaluate_attack`、`rpcf.evaluate_purification` 均使用 `--gpu_id` 构造 device，因此可配合该隔离方式运行。已执行 `DRY_RUN=1 START_STAGE=3` 验证 all-backbone 调度命令可正常展开。发现旧 stage3 watcher 未携带 10GB 显存下更稳的 batch size 设置，已替换为 PID `721414`，在自动启动 stage3+ 时传入 `AT_BATCH_SIZE=64`、`BASELINE_BATCH_SIZE=64`、`ONLINE_AT_BATCH_SIZE=64`、`RPCF_BATCH_SIZE=32`、`ATTACK_BATCH_SIZE=16`。
  - 2026-07-04 12:41：状态检查。stage2 三个 final RPCF cache 均已生成（各约 2.6GB），自动 watcher 于 2026-07-03 19:13 启动 stage3+。`atcnet` 已完成 stage3、stage4、stage5，目前运行到 `stage6_purify_madry_at`；已生成 `atcnet_retry2_rpcf_at_best.pth` 与 `finetune_rpcf_at.json`。`tsception` 与 `conformer` 已完成 stage3 sensitivity，但均在 stage4 `rpcf.finetune` 的 initial PGD 评估处 CUDA OOM 退出，需要后续用更小 `RPCF_BATCH_SIZE` / `ONLINE_AT_BATCH_SIZE` 或独占空闲 GPU 续跑 stage4。
  - 2026-07-04 12:46：按用户要求立即续跑 `tsception` 和 `conformer`，从 `START_STAGE=4` 开始，chain id 为 `exp024_parallel_seed42_20260702_153337_stage4_retry_lowmem`，外层 PID `1112791`。本轮加入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并进一步降低显存相关 batch：`RPCF_BATCH_SIZE=8`、`ONLINE_AT_BATCH_SIZE=16`、`ATTACK_BATCH_SIZE=8`、`AT_BATCH_SIZE=32`、`BASELINE_BATCH_SIZE=32`。调度器将 `tsception` 分配到物理 GPU2，将 `conformer` 分配到物理 GPU7；已确认子进程环境包含 `CUDA_VISIBLE_DEVICES=2/7` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  - 2026-07-06 10:19：复查 `tsception` stage4 后续风险。低显存重试已在 2026-07-04 18:10 保存 `tsception_rpcf_at_best.pth`，stage5 的 `madry_at` / `rpcf_at` AutoAttack 产物均已生成，stage6 的 rank25/30 净化产物也已生成。未发现类似 conformer 重复启动导致 stage5 进程被 kill 的风险；`stage5_attack_madry_at.log` 和 `stage5_attack_rpcf_at.log` 均正常输出对应 `.pth`。
  - 2026-07-06 10:19：发现 `tsception` stage9 汇总失败于 `rpcf/compare_exp024.py` 的 `KeyError: 'sample_num'`，原因是 `rpcf.evaluate_purification` 将 `sample_num` 写在 artifact `meta` 中而非每个 rank 的 `metrics` 中。已将汇总器改为 `metric.get("sample_num", meta["sample_num"])` 兼容旧/现有产物，并重跑 stage9 成功生成 `comparison.md`、`full_test_attack.csv`、`purification.csv`、`summary.json`。
  - 2026-07-06 10:29：复查 EXP-024 完成度。`tsception` stage9 已完成；`conformer` 的 attack、purification、baseline attack 产物齐全，但 stage9 旧日志因 `sample_num` schema 兼容问题失败。使用修复后的 `rpcf.compare_exp024` 重跑 `conformer` stage9，已生成 `comparison.md`、`full_test_attack.csv`、`purification.csv`、`summary.json`。
  - 2026-07-06 10:30：`atcnet_retry2` 目前完成到 stage6：`madry_at` / `rpcf_at` attack 与 rank25/30 purification 产物齐全，但 stage7/8/9 尚未完成。原因是旧脚本在 stage6 后触发 `rpcf/run_exp024_backbone.sh: line 299: --ranks: command not found` 后退出；当前脚本该断行问题已不存在，`DRY_RUN=1 START_STAGE=7 STOP_STAGE=9` 可正常展开，后续需要单独续跑 `atcnet` stage7-9。
  - 2026-07-06 10:35：按用户要求续跑 `atcnet_retry2` stage7-9，chain id 为 `exp024_atcnet_stage7_9_resume`，外层 controller PID `2471607`。使用 `nohup setsid env ... bash rpcf/run_exp024_all_backbones.sh` 启动，调度器将任务分配到物理 GPU2；当前进入 `stage7_train_baseline_clean`，训练进程已上卡，日志为 `logs/exp024/exp024_parallel_seed42_20260702_153337_atcnet_retry2/stage7_train_baseline_clean.log`，外层日志为 `logs/exp024/exp024_atcnet_stage7_9_resume.nohup.log`。
- **完成记录：**
  - 2026-07-06 21:51（Asia/Shanghai）：`atcnet_retry2` stage9 完成，三个 backbone 的 comparison 均已生成。
  - `tsception`：`logs/exp024/exp024_parallel_seed42_20260702_153337_tsception/comparison/`
  - `conformer`：`logs/exp024/exp024_parallel_seed42_20260702_153337_conformer_retry2/comparison/`
  - `atcnet`：`logs/exp024/exp024_parallel_seed42_20260702_153337_atcnet_retry2/comparison/`
- **结果汇总：**

  | Backbone | RPCF_AT trainable ratio | Full AA Madry | Full AA RPCF_AT | Δ RPCF_AT-Madry | Rank25 Madry+TNP | Rank25 RPCF_AT+TNP | Δ | Rank30 Madry+TNP | Rank30 RPCF_AT+TNP | Δ |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `tsception` | 35.10% | 38.08% | 38.20% | +0.12 | 48.63% | 53.32% | +4.69 | 47.85% | 48.44% | +0.59 |
  | `conformer` | 6.60% | 72.41% | 66.29% | -6.12 | 77.34% | 73.05% | -4.30 | 75.78% | 71.68% | -4.10 |
  | `atcnet` | 66.44% | 86.27% | 85.14% | -1.12 | 87.50% | 86.91% | -0.59 | 87.11% | 85.94% | -1.17 |

  这里的 `Madry+TNP` 指 `madry_at` checkpoint 遭遇自身 white-box AutoAttack 后再执行 rank25/30 EEG_TNP 净化；`RPCF_AT+TNP` 指 `rpcf_at` checkpoint 遭遇自身 white-box AutoAttack 后再执行同 rank 净化。
- **观察：**
  - `RPCF_AT` 并没有在跨 backbone 上稳定优于 `Madry_AT + EEG_TNP`。
  - `tsception` 是唯一正例，rank25 净化后提升明显（+4.69 个百分点），rank30 和未净化 AutoAttack 只有小幅提升。
  - `conformer` 是明确负例：未净化 AutoAttack、rank25、rank30 全部下降约 4-6 个百分点。
  - `atcnet` 本身 Madry AT 已很强，RPCF_AT 的未净化和净化后指标均略低；且本轮 40% 逻辑层规则实际微调了 66.44% 参数，参数效率并不好。
  - 因此 EXP-021/022 在 EEGNet 上得到的“RPCF_AT 提高净化后鲁棒性”不能直接外推到其他 backbone。
- **结论：**
  - EXP-024 支持把 RPCF_AT 表述为“EEGNet 上有效、跨 backbone 需要条件化”的方法，而不是稳定支配 `Madry_AT + EEG_TNP` 的通用改进。
  - 后续 EXP-025 的敏感层前缀预算曲线很关键：需要检查当前固定 40% 层数规则是否导致 conformer/atcnet 的负迁移，尤其是 atcnet 可训练参数比例过高的问题。
- **闭环检查：**
  - `DECISIONS.md`：已新增 `DEC-020`。
  - `方法进展梳理.md`：已补充 EXP-024 跨 backbone 结论。
  - `CODEMAP.md`：不需要。
  - `PROMPTS.md`：不需要。
- **结果：** 已完成
- **DeepConvNet/TCNet 扩展复跑：**
  - 日期：2026-07-07（Asia/Shanghai）。
  - 动机：按用户要求引入更权威/更多样的 EEG backbone；`deepconvnet` 参考 Braindecode `Deep4Net`/`deep4.py` 结构在本项目内自包含实现，`tcnet` 直接复用 TorchEEG `TCNet`。
  - 新增模型接入：`deepconvnet`、`tcnet` 已加入 `models/registry.py`、`models/model_args.py`、`train.py`、`train_AT.py`、`attack.py`、`purify.py` 和 `rpcf/*` 入口。
  - RPCF 逻辑层：`deepconvnet` 使用 `conv_time_spat/conv_2/conv_3/conv_4/final_layer`；`tcnet` 使用 `eegnet/tcn_blocks.0/tcn_blocks.1/classifier`。
  - 运行脚本：继续复用 `rpcf/run_exp024_all_backbones.sh` 与 `rpcf/run_exp024_backbone.sh`，仅设置 `EXP024_MODELS="deepconvnet tcnet"`。
  - 验证：
    ```bash
    python3 -m py_compile models/model_args.py models/deepconvnet.py models/registry.py rpcf/core.py train_AT.py attack.py purify.py
    conda run -n torch --no-capture-output python - <<'PY'
    import torch
    from models.registry import MODEL_CLASSES
    from models.model_args import get_model_args
    from rpcf.core import logical_layer_names
    info = {'chunk_size': 1024, 'num_electrodes': 64, 'num_classes': 4, 'sampling_rate': 250}
    for name in ['deepconvnet', 'tcnet']:
        model = MODEL_CLASSES[name](**get_model_args(name, 'thubenchmark', info))
        print(name, tuple(model(torch.randn(2, 64, 1024)).shape), logical_layer_names(name, model))
    PY
    EXP024_MODELS="deepconvnet tcnet" EXP024_GPU_IDS="4,5" EXP024_GPU_IDLE_MAX_USED_MB=99999 DRY_RUN=1 STOP_STAGE=1 \
      bash rpcf/run_exp024_all_backbones.sh
    ```
  - 首次启动：`exp024_deepconvnet_tcnet_seed42_20260707_1430`，外层 PID `2983001`，分配 GPU1/2；stage1 立即失败，原因是训练管线输入为 `[B,1,C,T]`，而新增 `deepconvnet` 与 TorchEEG `TCNet` 原生 forward 期望 `[B,C,T]`。
  - 修复：`DeepConvNet.forward` 与 `ProjectTCNet.forward` 均兼容单例维度 `[B,1,C,T] -> [B,C,T]`；已用 `[2,1,64,1500]` 假 batch 验证输出为 `[B,num_classes]`。
  - 正式 retry：`exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1`，外层 controller PID `2984943`；`deepconvnet` 子 controller PID `2984961`，物理 GPU1；`tcnet` 子 controller PID `2985037`，物理 GPU2。
  - 正式 retry 命令：
    ```bash
    nohup setsid env EXP024_RUN_TAG=20260707_1435 \
      EXP024_CHAIN_ID=exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1 \
      EXP024_MODELS="deepconvnet tcnet" EXP024_GPU_IDS="1,2" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      AT_BATCH_SIZE=64 BASELINE_BATCH_SIZE=64 RPCF_BATCH_SIZE=8 \
      RPCF_EVAL_BATCH_SIZE=2 ONLINE_AT_BATCH_SIZE=16 ATTACK_BATCH_SIZE=8 \
      bash rpcf/run_exp024_all_backbones.sh \
      > logs/exp024/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1.nohup.log \
      2>&1 < /dev/null &
    ```
  - 启动后检查：`deepconvnet` 已进入 stage1 Madry AT，训练日志到 epoch2；`tcnet` 已进入 stage1 Madry AT，训练日志到 epoch1；GPU1/2 均有对应训练进程。
  - 结果：Pending。


### EXP-025：RPCF_AT 敏感层降序累加微调预算曲线

- **状态：** Pending
- **动机：** 当前主方法已基本确定，后续重点转向补实验、消融、调参、可视化和低秩分析。本实验检查 EXP-024 固定 40% 敏感层规则是否过于静态，是否限制了不同 backbone 的 RPCF_AT 微调潜力。
- **核心设置：**
  - backbone：`eegnet`、`tsception`、`conformer`、`atcnet`。
  - 数据与协议：默认对齐 EXP-024，`thubenchmark`、`seed=42`、`fold=0`、`eps=0.03`、`AutoAttack`。
  - 层选择：读取对应 `sensitivity.json` 中的 `layers[*].score`，按分数降序排序，依次运行 `top1, top2, ..., all` 前缀预算。
  - 每个预算 checkpoint 单独执行 white-box attack，再做 rank25/rank30 净化和评测；不复用 EXP-024 的净化缓存。
- **新增入口：**
  - 运行脚本：`rpcf/run_exp025_layer_prefix.sh`
  - 汇总脚本：`rpcf/compare_exp025.py`
  - 日志结构：`logs/exp025/<run_id>/<backbone>/budget_<k>/`
  - 汇总输出：`logs/exp025/<run_id>/<backbone>/comparison/`
- **运行命令：**
  ```bash
  EXP025_RUN_ID=exp025_layer_prefix_seed42_YYYYMMDD_HHMMSS \
  EXP025_MODELS="eegnet tsception conformer atcnet" \
  CUDA_VISIBLE_DEVICES=0 GPU_ID=0 \
  nohup setsid bash rpcf/run_exp025_layer_prefix.sh \
    > logs/exp025/exp025_layer_prefix_seed42_YYYYMMDD_HHMMSS/controller.log \
    2>&1 < /dev/null &
  ```
  - 如果本地缺少 EEGNet 的 AT checkpoint、six-rank RPCF train cache 或 sensitivity artifact，需要先补生成，或显式传入：
    `EXP025_AT_CHECKPOINT_EEGNET`、`EXP025_CACHE_EEGNET`、`EXP025_SENSITIVITY_EEGNET`。
- **验证：**
  ```bash
  bash -n rpcf/run_exp025_layer_prefix.sh
  python3 -m py_compile rpcf/finetune.py rpcf/compare_exp025.py
  ```
- **启动记录：**
  - 2026-07-06 12:27（Asia/Shanghai）：按用户要求使用空闲 GPU4-7 启动 EXP-025。
  - `tsception`：run id `exp025_layer_prefix_seed42_20260706_1225_tsception`，物理 GPU4，controller PID `2574941`，日志 `logs/exp025/exp025_layer_prefix_seed42_20260706_1225_tsception/controller.log`。
  - `conformer`：run id `exp025_layer_prefix_seed42_20260706_1225_conformer`，物理 GPU5，controller PID `2574853`，日志 `logs/exp025/exp025_layer_prefix_seed42_20260706_1225_conformer/controller.log`。
  - `atcnet`：run id `exp025_layer_prefix_seed42_20260706_1225_atcnet`，物理 GPU6，controller PID `2574990`，日志 `logs/exp025/exp025_layer_prefix_seed42_20260706_1225_atcnet/controller.log`。
  - `eegnet`：run id `exp025_layer_prefix_seed42_20260706_1225_eegnet`，物理 GPU7；先用 `EXP018_RUN_ID=exp025_eegnet_prep_seed42_20260706_1225` 补 Madry AT checkpoint、six-rank RPCF train cache 和 sensitivity artifact，完成后自动进入 EEGNet EXP-025。外层启动 PID `2575612`，当前链式 shell/训练进程见 `2575614` / `2575618` / `2575686`，总日志 `logs/exp025/exp025_eegnet_prep_then_layer_prefix_seed42_20260706_1225/controller.log`。
  - 本轮显存相关参数：`RPCF_BATCH_SIZE=8`、`RPCF_EVAL_BATCH_SIZE=2`、`ONLINE_AT_BATCH_SIZE=16`、`ATTACK_BATCH_SIZE=8`，并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **结果：** Pending
- **闭环检查：**
  - `IDEAS.md`：不需要；本实验不是新方法，只是既定 RPCF_AT 的预算曲线。
  - `DECISIONS.md`：Pending；需等待预算曲线结果后再判断是否替代固定 40% 规则。
  - `方法进展梳理.md`：Pending；需等待结果后再更新论文叙事。
  - `CODEMAP.md`：已更新 EXP-025 入口。
