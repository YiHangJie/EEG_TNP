# CODEMAP.md

本文件用于帮助 Codex 和研究者快速理解仓库结构、主要工作流和安全扩展点。

更新依据：已阅读主要训练、攻击、净化、数据读取、模型参数、EA、配置、实验 pipeline 和分析脚本。没有完整检查的文件会明确标记为 TODO，不臆测职责。

## 主要入口

- `train.py`
  - 普通 EEG 分类训练入口。
  - 支持 `seediv`、`m3cv`、`bciciv2a`、`thubenchmark` 数据集，以及 `eegnet`、`tsception`、`atcnet`、`conformer`、`tcnet`、`deepconvnet` 模型。
  - 通过 `data.subject_ea.iter_subject_folds` 构造 subject split；当前脚本在 fold 循环中 `if index >= 1: break`，实际只训练第一个 fold。
  - 日志写入 `log_train/`，最佳 checkpoint 写入 `checkpoints/`。

- `train_AT.py`
  - 对抗训练入口。
  - 支持 `madry`、`fbf`、`trades`、`clean` 策略。
  - 支持 `--fold` 指定 subject split fold；默认 `fold0`。
  - 可通过 `--use_purified_aug` 和 `--purified_aug_paths` 拼接训练集净化增强样本。
  - 日志写入 `log_train_AT/`，checkpoint 通过 `utils.experiment_artifacts.build_checkpoint_path` 命名并写入 `checkpoints/`。

- `attack.py`
  - 测试集攻击与攻击前后评估入口。
  - 支持 `fgsm`、`pgd`、`cw`、`autoattack`。
  - 从 `checkpoints/` 加载模型，生成对抗样本，可保存到 `ad_data/`。
  - `--attack pgd` 支持可选 `--pgd_steps` 和 `--pgd_alpha`；默认不传时保持
    `attack/pgd.py` 原有 PGD 参数。
  - 日志写入 `log_attack/`。

- `purify.py`
  - EEG_TNP / Tensor Network 净化核心入口。
  - 读取 `configs/<dataset>/<config>.yaml`，将 EEG 样本 resize 到 TN 需要的表示，训练 TN 重建，再反变换回原始 EEG 空间。
  - 可评估 clean、adversarial、purified clean、purified adversarial 数据。
  - 净化后的测试集样本保存到 `purified_data/attacked/`，日志写入 `log_purify/`。

- `purify_train_clean.py`
  - 从训练 split 中随机抽样 clean 样本并净化，输出可用于训练增强的 `.pth`。
  - 默认输出目录是 `purified_data/train_clean/`。

- `purify_train_adv.py`
  - 从训练 split 中随机抽样 clean 样本，先用指定 checkpoint 生成训练集 adversarial 样本，再净化。
  - 默认输出目录是 `purified_data/train_ad/`。

- `purify_train_pair_consistancy.py`
  - 生成成对训练增强缓存：`x`、`x_pur`、`x_adv`、`x_adv_pur`、`labels`、`source_indices`。
  - 默认输出目录是 `purified_data/train_pair_consistancy/`。
  - 注意文件名中使用历史拼写 `consistancy`。

- `purify_train_pair_consistency2.py`
  - 生成 consistency2 版本的成对训练增强缓存。
  - 默认输出目录是 `purified_data/train_pair_consistency2/`。

- `train_AT_consistancy.py`
  - 使用 `train_pair_consistancy` 缓存进行 Madry AT + purified pair consistency 损失训练。
  - 支持 `--fold` 指定 subject split fold；paired cache meta 会校验 fold 一致性。
  - 支持 CE 和 KL 对齐损失；日志写入 `log_train_AT/`，checkpoint 写入 `checkpoints/`。

- `train_AT_consistency2.py`
  - 使用多个 consistency2 rank 缓存进行 Madry AT + 多 rank consistency 损失训练。
  - 要求至少两个 `--consistency_aug_paths`，会按 `source_indices` 对齐不同 rank。

- `rpcf/`
  - Rank-aware Purification Critical Fine-tuning 独立实现，不改变现有 AT 和 consistency baseline。
  - 训练/测试子集抽样严格复用 consistancy pipeline：`selection_seed = seed + fold * 1000`；fine-tuning shuffle 复用全局 `seed_everything(seed)`，不再添加 RPCF 专用 seed offset。
  - `generate_cache.py`：对训练子集只攻击一次，再对相同 `x/x_adv` 生成多 rank clean/adversarial purification，最终保存统一 `[N, R, ...]` cache；每个 rank 使用可续跑 work shard。
  - `analyze_sensitivity.py`：在逻辑层上计算 clean-anchor 的绝对 shift，并用 input/相邻逻辑层的 shift 比值计算层间相对敏感度，按 clean/adv-pur 等权组合分数选择 Top-K。
  - `finetune.py`：冻结非敏感层和对应 BatchNorm 统计量，使用 clean-teacher consistancy CE+KL 与动态 rank curriculum 微调 AT checkpoint；固定训练满 epochs 并保存最终 epoch，不使用 validation early stopping。显式开启 `--online_madry_at` 后，每个 epoch 会先在完整训练 split 上执行在线 PGD adversarial CE，再执行不含固定 `x_adv` 项的多-rank 净化适配损失。
  - `evaluate_attack.py`、`evaluate_purification.py`、`summarize.py`：分别负责 white-box attack、逐 rank 净化评估和 CSV/JSON 汇总。
  - `run_rpcf_pipeline.sh`：七阶段可续跑 pipeline，支持 `SMOKE`、`DRY_RUN`、`START_STAGE`、`STOP_STAGE` 和 `SKIP_EXISTING`。
  - `compare_exp018.py`、`run_exp018.sh`：执行 EXP-018 三方法 seed42 全流程重跑；统一训练 AT、训练 consistancy/RPCF、生成各自 white-box AutoAttack，并严格校验同一 n512 测试子集上的 rank25/30 净化结果。
  - `run_exp018_consistancy_six_rank.sh`：EXP-018 续跑脚本；复用 seed42 AT/RPCF 正式产物，只把 consistancy 的训练增强从 rank25/30 改为 rank15/20/25/30/35/40，随后重跑 consistancy checkpoint、white-box AutoAttack、rank25/30 净化和五方法汇总。
  - `run_exp018_rpcf_rerun.sh`：复用 EXP-018 的 AT、六-rank cache 和已有对照产物，只重跑新版 RPCF sensitivity、fine-tuning、white-box attack、rank25/30 净化与公平汇总。
  - `run_exp018_rpcf_all_layers.sh`：EXP-018 的 `RPCF w.o. sensitivity layer selection` 消融，保持数据、损失、rank curriculum 和评估不变，仅通过 `--all_layers` 微调全模型。
  - `run_exp018_rpcf_static_ranks.sh`：EXP-018 的 `RPCF w.o. rank schedule` 消融，保留 sensitive-layer selection，仅通过 `--static_rank_weights` 将六个 rank 权重固定为 `1/6`。
  - `compare_exp018_five_methods.py`：只读复用公平比较 `summary.json`，严格校验实验协议、完整测试集样本数和净化子集 `source_indices`，统一输出 Madry AT、consistancy、RPCF selective、RPCF all-layers、RPCF rank-weight uniform 五方法长表、宽表和 Markdown 表。
  - `run_exp019.sh`：EXP-019 五方法 seed43/fold0 复验 pipeline；统一生成 AT/consistancy、三个 RPCF 变体、五组 white-box AutoAttack、五组 rank25/30 净化和最终严格汇总，支持 smoke、dry-run 与断点续跑。
  - `run_exp019_consistancy_six_rank.sh`：EXP-019 续跑脚本；复用原 seed43 AT/RPCF 产物，只把 consistancy 的训练增强从 rank25/30 改为与 RPCF 一致的 rank15/20/25/30/35/40，随后重跑 consistancy checkpoint、white-box AutoAttack、rank25/30 净化和五方法汇总。
  - `run_exp020.sh`：EXP-020 的 eps0.05 五方法复验 pipeline；完整复制 EXP-019 six-rank 公平对比协议，但将所有训练、攻击和评估的 epsilon 固定为 `0.05`，产物隔离写入 `exp020`。
  - `run_exp020_all_seeds.sh`：按 seed42/43/44 串行调度 EXP-020，避免单 GPU 并发，并为每个 seed 保留独立 controller log。
  - `run_exp021_rpcf_at.sh`：按 seed42/43/44 复用 EXP-018/019 的 AT checkpoint、六-rank cache 和 sensitivity，只重跑 RPCF_AT selective 微调、自身 white-box AutoAttack、rank25/30 净化与三方法公平汇总。
  - `run_exp021_seeds43_44.sh`：串行调度 EXP-021 seed43/44，避免单 GPU 并发。
  - `run_exp022_rpcf_at.sh`：复用 EXP-020 eps0.05 的 AT checkpoint、六-rank cache、sensitivity 和已有对照净化结果，只重跑 RPCF_AT 微调、攻击与净化。
  - `run_exp022_all_seeds.sh`：串行调度 EXP-022 seed42/43/44。
  - `compare_exp022.py`：严格校验 EXP-020 与 EXP-022 的协议和净化子集，输出 Madry AT、six-rank consistancy、普通 RPCF selective、RPCF_AT 四方法汇总。
  - `evaluate_bpda_pgd.py`、`run_exp023_bpda_pgd.sh`、`run_exp023_all_seeds.sh`、`compare_exp023.py`：EXP-023 的 EEG_TNP+RPCF_AT adaptive attack；BPDA forward 真实执行 rank25/30 EEG_TNP 净化，backward 将 EEG_TNP 视作恒等变换，并用 PGD-10 生成 rank-specific 对抗样本。
  - `run_exp023_baseline_pgd10.sh`：EXP-023 baseline 补充脚本，复用
    `attack.py` 在同一 n512 子集上评估 Madry/TRADES/FBF 面对 PGD-10 的性能。
  - `run_exp023_train_baselines_then_pgd10.sh`：EXP-023 baseline 补全调度脚本；
    等待 BPDA controller 结束后，先补训缺失的 seed43/44 TRADES/FBF baseline
    checkpoint，再调用 `run_exp023_baseline_pgd10.sh` 评估 seed43/44 的
    Madry/TRADES/FBF PGD-10。
  - `run_exp024_backbone.sh`、`run_exp024_all_backbones.sh`、`compare_exp024.py`：
    EXP-024 其他 backbone 扩展测试；对 `tsception`、`atcnet`、`conformer`、`deepconvnet`、`tcnet`
    串行执行 Madry AT、RPCF cache/sensitivity、RPCF_AT 在线微调、AutoAttack、
    rank25/30 净化测试，并训练/评估 clean、TRADES、FBF baseline。
  - `run_exp025_layer_prefix.sh`、`compare_exp025.py`：EXP-025 敏感层前缀预算曲线；
    复用已有 AT checkpoint、RPCF train cache 和 sensitivity artifact，按敏感性分数降序
    依次累加可训练层，为每个前缀预算单独微调 RPCF_AT、执行 white-box attack、
    rank25/30 净化并汇总 clean/adv/purified 指标。
  - 默认 rank 为 `15,20,25,30,35,40`，正式长实验必须通过 `nohup` 后台启动。

- `train_AT_ea_forward.py`
  - EA-in-forward 特殊训练入口。
  - 使用 raw/no_ea 输入，将 subject-wise EA 放到模型 forward 中执行。
  - 当前支持 `eegnet_ea_forward`、`conformer_ea_forward` 和 `madry`。

- `attack_ea_forward.py`
  - EA-in-forward 模型的 subject-aware 攻击入口。
  - 对需要 subject id 的模型做 wrapper，使现有攻击类可以使用。

- `collect_log.py`
  - 解析 `log_purify/` 下的净化日志，导出 Excel，并可生成对比图。
  - 默认示例输出包括 `purify_summary.xlsx` 和 `plots/` 下的图。

- `tools/exp_status.py`
  - 只读汇总 `logs/*/<run_id>` 下的 PID、阶段日志、错误和最终 summary。
  - 只有与当前 run id 匹配的最终完成日志，或 controller 退出后存在最终 summary，才会标记为 `completed`；中间 comparison summary 不代表整个 run 完成。

- `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py`
  - 分析 Tensor Ring rank 与预测表现相关的实验脚本。
  - 默认 `--analysis_mode tensorly_tr` 保留旧的 TensorLy 普通 TR sweep；`--analysis_mode rank_growth` 会直接运行 `PTR_3d_rank_growth` 并记录动态 rank 轨迹。
  - `--analysis_mode rank_soft_mask` 会运行 `PTR_3d_rank_soft_mask`，记录单次 soft-rank 净化的 effective rank、MSE 和分类指标。
  - rank-growth 诊断趋势时应使用 `--rank_growth_full_sweep`，禁用早停以避免高 rank 统计缺失。
  - 会读取 checkpoint、paired cache 或 adversarial data，并写入 `tensor_ring_rank_analysis/results*`。

- `tensor_ring_rank_analysis/analyze_rank_growth_pair_delta.py`
  - 对 `rank_growth` 输出做离线 paired-delta 分析，不重新运行净化。
  - 读取 `rank_growth_predictions.pt` 和 `rank_growth_incremental_frequency.csv`，联合统计高频占比变化、true-label margin 变化、bootstrap 置信区间、相关系数和分类边界恶化率。
  - 主要用于 `EXP-005` 这类 full-sweep 结果的追加诊断，输出 CSV、meta JSON 和误差棒图到原实验目录。

- `tensor_ring_rank_analysis/summarize_exp015_results.py`
  - 汇总 `EXP-015` 矩阵 runner 的 `planned_tasks.csv` 与产物 meta，输出 `summary.csv`。
  - 主方法读取 `purified_data/attacked/` 的 JS_MSE rank-growth 净化结果；baseline 读取 `ad_data/` 的攻击结果。

- `trial_lowrank_analysis/analyze_trial_hosvd_lowrank.py`
  - 从 trial/time/channel/frequency 等视角分析 clean、adv、perturbation 的低秩谱。
  - 默认输出到 `trial_lowrank_analysis/outputs/`。

- Shell pipeline
  - `train.sh`、`train_AT.sh`、`attack.sh`：批量训练/对抗训练/攻击脚本。
  - `purify.sh`：批量遍历 `configs/<dataset>/` 下的净化配置，支持 `DRY_RUN`、`GPU_IDS`、`MAX_JOBS` 等环境变量。
  - `purify_aug_pipeline.sh`：clean/ad train purification augmentation 端到端流程。
  - `purify_aug_consistancy_pipeline.sh`：旧拼写 consistancy 的 paired augmentation + AT 流程。
  - `purify_aug_consistency2_pipeline.sh`：consistency2 paired augmentation + AT 流程。
  - `TN/rank_growth/run_exp015_full_test.sh`：`EXP-015` 跨 dataset/model/seed/fold/eps 的主方法与 baseline 矩阵 runner，日志集中写入 `logs/exp015/<run_id>/`。
  - 这些脚本可能启动长时间任务，除非用户明确要求，不要直接运行。

## 数据读取

- `data/load.py`
  - 基于 TorchEEG 数据集类加载原始 EEG 数据，并写入/读取 `cached_data/`。
  - 数据集函数：
    - `load_seediv()`
    - `load_m3cv()`
    - `load_bciciv2a()`
    - `load_thubenchmark()`
  - 返回 `(dataset, info)`，其中 `info` 至少包含：
    - `num_electrodes`
    - `chunk_size`
    - `num_classes`
    - `sampling_rate`
  - 代码中 `root_path` 从 `configs/data_roots.yaml` 读取；默认 `default_root` 为 `/data2/yihangjie/pythonProjects/data`，可通过环境变量 `EEG_TNP_DATA_ROOT` 或 `EEG_TNP_<DATASET>_ROOT` 临时覆盖。不要移动或修改这些原始数据目录。

- `data/subject_ea.py`
  - 负责 subject-wise split、train-only subject EA、raw/no_ea wrapper、EA-in-forward wrapper。
  - 关键协议 tag：
    - `train_only_subject_ea_subject_split`
    - `train_only_subject_no_ea_subject_split`
  - split CSV 默认写入 `cached_data/<dataset>_train_only_subject_ea_subject_split_seed<seed>_split/`。
  - 主要函数：
    - `prepare_subject_fold`
    - `iter_subject_folds`
    - `prepare_subject_ea_forward_fold`
  - 主要 Dataset wrapper：
    - `CachedEEGDataset`
    - `SubjectEAAlignedDataset`
    - `SubjectIndexedCachedEEGDataset`

- `data/ea_utils.py`
  - 实现 EA 矩阵计算、EEG alignment、mean/std normalization、最终样本 tensor 化。

- `utils/experiment_artifacts.py`
  - 提供路径 token、统一 collate、tensor payload 读取和 checkpoint 命名工具。
  - `eeg_classification_collate` 用于普通分类 batch。
  - `eeg_subject_classification_collate` 用于 EA-in-forward batch。
  - `load_tensor_payload` 兼容 `(data, labels)`、`(data, labels, meta)` 和 dict 格式。

- `utils/reproducibility.py`
  - 提供全仓共享的 `seed_everything()` 和 `stable_subset_indices()`。
  - EXP-018 相关 AT、consistancy、RPCF、攻击和净化入口统一使用该实现。

## 模型定义

- 分类模型主要来自 `torcheeg.models`
  - `EEGNet`
  - `TSCeption`
  - `ATCNet`
  - `Conformer`

- `models/model_args.py`
  - 根据 `model`、`dataset` 和 `info` 生成 TorchEEG 模型初始化参数。
  - 如果新增分类模型，通常需要在这里补充参数规则，并同步更新入口脚本中的 `model_dict`。

- `models/eegnet_ea_forward.py`
  - 定义通用 EA-in-forward wrapper，以及 `SubjectEAEEGNet`、`SubjectEAConformer`。
  - 在模型 forward 中根据 `subject_ids` 查 EA 矩阵，执行 `R^{-1/2} @ X`，再做归一化并交给 EEGNet 或 Conformer backbone。

- `TN/`
  - Tensor Network 净化模型与工具目录。
  - `purify.py` 当前会导入并选择：
    - `TN.PTR.PTR`
    - `TN.PTR_3d.PTR_3d`
    - `TN.PTR_3d_fs.PTR_3d_fs`
    - `TN.PTR_tfs.PTR_tfs`
    - `TN.rank_growth.PTR_3d_rank_growth`
    - `TN.rank_growth.PTR_3d_rank_soft_mask`
  - `TN/opt.py` 定义 YAML config 的默认 `Config` 和 `yaml_config_parser`。
  - `TN/utils.py` 提供 TN 参数构造和若干通用工具。
  - TODO：新增或修改 TN 架构前，应进一步阅读对应模型文件的训练接口和 shape 约定。

- `attack/`
  - 攻击实现目录。
  - 当前入口脚本使用 `FGSM`、`PGD`、`CW`、`AutoAttack`，EA-in-forward 入口还使用 APGD、APGDT、FAB、Square 等内部组件。

## 训练流程

- 普通训练
  - `train.py` 调用 `seed_everything` 固定随机源。
  - 加载 dataset 和 subject split。
  - 构造 TorchEEG 模型。
  - 使用 AdamW、ReduceLROnPlateau、CrossEntropyLoss、early stopping。
  - 每个 epoch 记录 train loss、val acc/loss、test acc/loss。
  - 保存最佳 `state_dict` 到 `checkpoints/`。

- 对抗训练
  - `train_AT.py` 在普通训练流程上增加在线 adversarial example 生成。
  - `madry` 使用 PGD adversarial examples。
  - `fbf` 使用 fast adversarial-style replay。
  - `trades` 使用 KL robust loss。
  - `clean` 走普通 CE 训练。
  - 可选把 `purified_data/train_clean` 或 `purified_data/train_ad` 的 `.pth` 作为增强样本 concat 到训练集。

- Paired consistency 训练
  - `train_AT_consistancy.py` 使用单 rank paired cache。
  - `train_AT_consistency2.py` 使用多 rank paired cache，并增加低 rank 到高 rank 的 KL 对齐。
  - 这两条路径默认仍先在原始训练集上执行 Madry AT，再在 paired cache 上执行额外一致性训练。

- EA-in-forward 训练
  - `train_AT_ea_forward.py` 使用 `prepare_subject_ea_forward_fold` 返回 raw/no_ea 数据、subject index 和 EA 矩阵。
  - 模型内部执行 EA，攻击梯度会穿过 EA 操作。

## 评估流程

- 目前没有独立统一的 `eval.py`。
- `train.py`、`train_AT.py`、`attack.py`、`purify.py`、`train_AT_consistancy.py`、`train_AT_consistency2.py` 都各自定义或复用 `evaluate`。
- 训练脚本通常在每个 epoch 记录 validation 和 test 指标，并在 early stopping 后记录最佳模型 test 指标。
- `attack.py` 会记录攻击前 test accuracy/loss、攻击后 test accuracy/loss、clean/adversarial MSE，并可抽样 512 个测试样本做额外评估。
- `purify.py` 会记录 clean/adversarial 数据净化前后 accuracy/loss，以及 mean MSE。
- `collect_log.py` 用正则解析 `log_purify/` 的日志，汇总净化实验指标。
- `test_subject_ea.py` 是可运行的 EA/split/EA-in-forward 相关单元测试与 smoke test。
- `test_rpcf.py` 覆盖 RPCF cache、feature shift、Top-K、rank curriculum、冻结 BatchNorm、checkpoint 选择规则和 consistancy 抽样 seed 对齐。
- `test_graph_loss.py` 当前未观察到有效内容，TODO：确认是否保留或补充。

## 实验配置

- `configs/data_roots.yaml`
  - 原始 EEG 数据根目录配置，`data/load.py` 会读取该文件来确定各数据集的 `root_path`。
  - `default_root` 可通过 `EEG_TNP_DATA_ROOT` 覆盖；单个数据集可通过 `EEG_TNP_SEEDIV_ROOT`、`EEG_TNP_M3CV_ROOT`、`EEG_TNP_BCICIV2A_ROOT`、`EEG_TNP_THUBENCHMARK_ROOT` 覆盖。

- `configs/<dataset>/*.yaml`
  - 净化/TN 配置文件目录。
  - 已观察到的数据集子目录：
    - `configs/bciciv2a/`
    - `configs/thubenchmark/`
    - `configs/seediv/`
  - YAML 关键字段包括：
    - `model`
    - `strategy`
    - `lr`
    - `num_iterations`
    - `max_batch_size`
    - `seed`
    - `stage`
    - `payload_position`
    - `iterations_for_upsampling`
    - `max_rank`
    - `loss_fn_str`
    - `regularization_weight`
  - 新实验应新增配置文件，不要覆盖已有 baseline 配置。

- `configs_backup/`
  - 备份配置目录。
  - 不要修改或删除，除非用户明确要求。

- Shell pipeline 环境变量
  - `purify.sh`、`purify_aug_pipeline.sh`、`purify_aug_consistancy_pipeline.sh`、`purify_aug_consistency2_pipeline.sh` 通过环境变量控制 dataset、GPU、eps、rank config、sample num、run tag、dry run 等。
  - 修改 shell pipeline 前应特别注意变量引用、失败退出、路径和日志命名。

## 输出目录

- `cached_data/`
  - TorchEEG cache 和 subject split CSV。
  - 由 `data/load.py` 和 `data/subject_ea.py` 生成/读取。

- `checkpoints/`
  - 模型 checkpoint 输出目录。
  - 训练脚本和 AT 脚本会写入 `.pth`。

- `ad_data/`
  - 攻击生成的 adversarial data `.pth`。
  - `attack.py` 和 `attack_ea_forward.py` 可写入该目录。

- `purified_data/`
  - 净化样本输出目录。
  - 已观察到子目录：
    - `purified_data/attacked/`
    - `purified_data/train_clean/`
    - `purified_data/train_ad/`
    - `purified_data/train_pair_consistancy/`
    - `purified_data/train_pair_consistency2/`

- 日志目录
  - `log_train/`
  - `log_train_AT/`
  - `log_attack/`
  - `log_purify/`
  - `log_purify_1/`
  - `logs/`
  - 根目录还存在若干 `.log` 和 `nohup.out`。

- 分析与可视化输出
  - `plots/`
  - `visualization/`
  - `purify_summary.xlsx`
  - `tensor_ring_rank_analysis/results*/`
  - `trial_lowrank_analysis/outputs/`

## 安全扩展点

- 新净化/TN 实验
  - 优先在 `configs/<dataset>/` 新增 YAML。
  - 如果需要新 TN 模型，新增 `TN/<new_model>.py`，再在 `purify.py` 的 `TN_dict` 中用新 key 显式接入。
  - 不要修改已有配置来改变 baseline。

- 新分类模型
  - 优先新增 `models/<new_model>.py` 或使用 TorchEEG 现有模型。
  - 当前统一注册表为 `models/registry.py`；新增模型后同步更新 `models/model_args.py`、RPCF 逻辑层定义和相关入口脚本。
  - 默认行为必须保持旧模型不变。

- 新训练 idea
  - 优先在现有训练脚本中加显式 flag，或新增独立训练入口。
  - 对 baseline 有影响的默认值必须保持关闭。
  - 实现后更新 `docs/EXPERIMENTS.md`，没有实际结果时写 `Pending`。

- 新攻击方法
  - 优先新增 `attack/<method>.py`，再在 `attack.py` 或对应 EA-in-forward 入口中显式注册。
  - 注意普通模型和 subject-aware 模型的接口差异。

- 新数据集
  - 新增 `data/load.py` 中的 loader 和 `info` 规则。
  - 同步更新入口脚本的 dataset choices 和 dataset dict。
  - 不要移动或重写现有 raw data/cached data。

- 新 pipeline
  - 优先新建 shell 脚本，支持 `DRY_RUN=1`。
  - 默认不要覆盖已有输出；需要覆盖时使用显式 `OVERWRITE=1` 或等价参数。

- 新结果整理
  - 优先扩展 `collect_log.py` 或新增只读分析脚本。
  - 输出到新的结果目录或带 run tag 的文件，避免覆盖旧结果。

## 不应修改的文件/目录

- 外部 raw data
  - `/data2/yihangjie/pythonProjects/data/seediv/eeg_raw_data`
  - `/data2/yihangjie/pythonProjects/data/m3cv`
  - `/data2/yihangjie/pythonProjects/data/bciciv2a`
  - `/data2/yihangjie/pythonProjects/data/THUBenchmark`

- 仓库内实验产物和缓存
  - `cached_data/`
  - `checkpoints/`
  - `ad_data/`
  - `purified_data/`
  - `log_train/`
  - `log_train_AT/`
  - `log_attack/`
  - `log_purify/`
  - `log_purify_1/`
  - `logs/`
  - `plots/`
  - `visualization/`
  - `tensor_ring_rank_analysis/results*/`
  - `trial_lowrank_analysis/outputs/`
  - `purify_summary.xlsx`
  - 根目录已有 `.log`、`.nohup.log`、`nohup.out`

- 配置和备份
  - `configs_backup/`
  - 已存在的 baseline config 文件不要覆盖；需要新实验时新增 config。

- 临时和解释器缓存
  - `__pycache__/`
  - `.pytest_cache/`
  - `.codex/`
  - `.vscode/`

## TODO

- TODO：进一步检查 notebooks：
  - `AE_visulization.ipynb`
  - `hankel_test.ipynb`
- TODO：进一步检查辅助脚本：
  - `look_perturbation_freq.py`
  - `utils/standalone_to_interpolated_grid.py`
  - `utils/analyze_thubenchmark_channel_amplitude.py`
  - `utils/visualize.py`
- TODO：进一步检查 `TN/` 各模型文件的具体张量 shape、训练接口和保存逻辑。
- TODO：如果要把 CODEMAP 作为长期维护文档，应在每次新增入口脚本、输出目录或实验 pipeline 后同步更新。
