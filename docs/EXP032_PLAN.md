# EXP-032：低秩动机、统一主表与跨范数补充实验

- 登记日期：2026-09-17。
- 状态：已实现，smoke 通过，正式实验后台运行中；正式结果 **Pending**。
- 来源：补充 EXP-031 `exp031_full_20260729_174215`，关联 IDEA-002/009/011/012。
- 范围：本轮只覆盖用户提出的第 0、1、2 项；不改变 EXP-031 的 checkpoint、日志、缓存或结论。

## 共同协议

- `thubenchmark / seediv / bciciv2a` × `eegnet / deepconvnet / tsception / atcnet / conformer / tcnet` × seed `42–46`，fold0，既有 no-EA subject split。EA-forward 保留自身 subject-aware forward。
- 每个 dataset/seed 使用 `stable_subset_indices(len(test), min(512, len(test)), seed, fold)`，即 NumPy `RandomState(seed + fold * 1000)` 无放回抽样。保存实际 source indices、标签、样本数和摘要；跨方法比较必须核对样本身份。
- 双指标为相同子集上的 standard（clean）accuracy 和 robust（adversarial）accuracy，单位 %；五 seed 报均值与样本标准差，配对差值先逐 seed 计算。全测试集旧指标仅作背景，禁止与 n512 相减。
- 已有模型及 TNP 测试 rank25/rank30 均冻结并分别报告；不在测试集挑 rank、不按攻击重调参数、不挑最好 seed。RPCF_AT 沿用全层、六 rank 静态权重 `1/6`、无 feature loss 的 EXP-031 checkpoint。
- “先攻击分类器、再净化”结果明确标为非自适应净化评估，不等同于攻击完整防御的最坏情况鲁棒性。外部可微净化另报告穿过净化器的攻击。
- 现存产物只读复用；新增产物统一使用独立 `exp032` 路径。新攻击保留完整 n512 float32 adversarial 并引用已有同源 clean；TNP 保留逐样本预测、MSE、重构 SHA256、每 rank 两个示例及可复算来源，不重复保存完整重构张量；断点保留 RNG 状态。长任务使用 `nohup`、`python -u` 和 `conda run --no-capture-output`，保留稳定日志和可续跑任务清单。

## 0. 真实 EEG 低秩 motivation

- 预先固定 EEGNet，覆盖三数据集和五 seed，使用上述固定子集的真实 EEG；主诊断为 clean-only 分类器产生的 PGD-200 扰动，复核 Madry/RPCF_AT 的现存 PGD-200。不同来源分组报告，不将模型间的重复 clean 样本当成独立观测。
- 表示复用 TNP 的电极空间插值和时间张量化；逐样本分析，不把 trial 堆叠后的低秩当成单样本低秩。
- 对照包括 `x`、`delta = x_adv - x`、`x + delta`、等范数 Gaussian 随机扰动 `eta`、`x + eta`；逐样本分别匹配 L2 与 L∞ 范数，两个对照明确区分。范数匹配在原 EEG 坐标中进行，变换后范数也记录。
- 报各展开矩阵的奇异值谱、entropy effective rank、stable rank、95% 能量秩，以及截断 HOSVD 相对 Frobenius 重构误差曲线。HOSVD mode rank 与 TNP Tensor Ring bond rank 分开命名，不能视为同一参数。
- 多样本输出逐样本 CSV、分位数/ECDF 或箱线图及按 seed 汇总；展示全部预定样本，不能只挑显著示例。零扰动和零能量样本单独计数。
- 证伪条件：若 clean 与扰动低秩分布不能区分，或随机对照同样解释观察，应如实报告；低秩重构证据本身不证明分类鲁棒性。

## 1. 统一 n512 主实验与 baseline 审计

| 方法 | 来源与新增工作 |
| --- | --- |
| Madry / TRADES / FBF / EA-forward / RPCF_AT | 复用 EXP-031 checkpoint 与已保存攻击，重新推理统一子集双指标；审计不通过的攻击单独重跑 |
| Madry+TNP / RPCF_AT+TNP | 严格核对并复用 EXP-031 rank25/30 净化张量，重新推理 |
| clean-only | 补训普通 clean CE 分类器，训练划分、随机源、优化器及验证选模规则与既有 baseline 对齐 |
| clean-only+TNP | **Deferred（2026-09-17 用户要求先不跑）**：暂缓五种攻击对应的450个TNP任务，保留原计划供后续恢复 |
| clean-only+DCAE | 根据用户提供论文的公式 (10)–(13)、表 1/2 重建；复现假设见 `docs/EXP032_BASELINES.md` |
| clean-only+MagNet-Reformer (EEG adaptation) | 外部论文的自编码器重构净化路线；独立 clean train split 训练，不使用测试标签或攻击样本选模；同一 clean-only 分类器配对 |

外部方法依据 [MagNet 论文](https://arxiv.org/abs/1705.09064)及[作者实现](https://github.com/Trevillie/MagNet)。本轮比较其 reformer 净化组件，不包含 detector/rejection；名称必须带 `Reformer` 与 `EEG adaptation`，不能声称完整复现原论文。EEG 为标准化、有正负值的信号，因此网络结构、输出激活、噪声尺度及训练设定的适配须完整记录。该方法作为有公开来源的净化对照；不把它称为当前最强外部方法。

审计至少覆盖：训练/验证划分、checkpoint 与实际 batch/epoch、early stopping、TRADES beta、FBF replay、扰动预算和实际范数、模型 eval 状态及推理前后 state_dict 变化。重点复核 EA-forward 的四攻击 clean 漂移、SEED-IV/BCI 上 TRADES/FBF 异常低值。发现异常先确定实现或协议原因，保留原数值和修正来源，不因结果差而删除 baseline。若需要调整训练超参数，只能由 train/validation 选择并作为独立变体，不能覆盖旧结果。

## 2. 跨攻击与跨范数

| 攻击 | 明确协议 |
| --- | --- |
| FGSM | L∞，epsilon=0.03，1 step，沿用原 EEG 坐标 |
| PGD-200 | L∞，epsilon=0.03，alpha=2/255，无随机起点，复现 EXP-031 |
| AutoAttack | standard L∞，epsilon=0.03，保留具体子攻击和 seed |
| CW | 旧版无半径约束 L2：steps=200、lr=0.1、c=10000、kappa=1，独立报告实际 L2 分布 |
| PGD-L2-200 | **绝对 L2 半径 1.0**，alpha=0.1，200 steps，5 random restarts；按样本选成功攻击/最大 CE 候选，记录实际范数和预算违规数 |

- L2 半径针对标准化后的每个完整 EEG 样本，不是每通道半径；同时记录维数 D 与 `epsilon/sqrt(D)`，不声称与 L∞ epsilon=0.03 等强。
- 所有攻击使用相同 checkpoint、数据子集与冻结的净化参数。不同攻击分别出表，禁止跨范数混合平均。
- 逐样本交集正确率可作为明确攻击集合下的经验 worst-case accuracy；只有保存逐样本预测且同一威胁模型时才计算。

## 验收与交付

1. 先通过语法/导入、索引对齐、范数投影和小样本完整链路检查；smoke 产物与正式结果隔离。
2. 输出实验 manifest、任务状态、冻结配置、异常审计表、逐样本预测、分攻击五 seed 主表及 motivation CSV/图片。
3. 只有当前执行范围内任务、90 个条件和五 seed 计数全部通过才能写 Completed active scope；每条件70行、合计6300行。暂缓450个clean-only+TNP任务独立标为Deferred，不能误标失败或声称原完整计划已完成；运行中、缺失和失败均保留Pending/Failed。
4. 本文件登记规划，实际启动命令、run id、日志和验证结果追加至 `docs/EXPERIMENTS.md`；实验结果影响研究判断时再更新 `docs/DECISIONS.md`。

## 存储预算

2026-09-17 检查 `/data2` 剩余约394GB，初版重复保存 attack/clean/TNP/partial 的方案不可执行。已在大规模产物写入前停止首次正式队列并保留其审计记录。精简后900份新攻击张量约112.5GiB，连同模型、TNP示例和分析表保守预留160GiB；写入攻击前保留至少32GiB空闲。旧artifact、日志和checkpoint不删除。

读取 EXP-032 attack 请用 `rpcf.exp032_common.load_payload`，它按 source indices 验证标签并恢复 `clean_reference_path`。EXP-032 TNP产物kind为 `exp032_tnp_summary`，不再承诺包含完整 `adv_pur_by_rank`；逐样本复算可由冻结checkpoint、原攻击、config、seed和输出sha256校验。

## 当前运行

- 正式：`exp032_full_20260917_2020`，原计划2,685任务保留；当前执行2,235任务，另450个clean-only+TNP任务按用户要求Deferred。TNP仅新增Madry/RPCF_AT的PGD-L2，各90任务、rank25/30。GPU0–5，结果Pending。
- 稳定日志：`logs/exp032_full_20260917_2020.controller.log`。
- 最终smoke：`exp032_smoke_20260917_v4`，34/34任务、80/80指标组合，无验收错误；新旧格式数值和TNP示例hash一致。
- 初始正式 `exp032_full_20260917_2011` 已因存储预算主动停止，保留其审计产物。

## 2026-09-17 执行范围调整：暂缓 clean-only+TNP

- 保留90审计、90clean训练、30外部净化训练、900攻击、180TNP、900外部评估、45低秩诊断；合计2,235任务。clean-only原始指标、MagNet/DCAE及其自适应评估继续执行。
- 原`manifest.json`、`planned_tasks.csv`和`source_sha256.json`保持原样。独立`execution_scope.json`、`active_tasks.csv`、`deferred_tasks.csv`记录用户调整，`scope_sources_sha256.json`冻结续跑和验收代码。
- 通过`rpcf/resume_exp032.py`沿用原run-id、路径、seed和任务命令，跳过已完成任务。启动脚本检测到执行范围后自动走该入口；当前run后续应统一使用`rpcf/run_exp032.sh`，不直接调用原`rpcf.exp032 run`入口。
- 新汇总写入`logs/exp032/exp032_full_20260917_2020/summary_without_clean_tnp/`，保留原summary；报告同时列原计划数、当前任务数、Deferred数和`full_plan_completed=false`。
- 续跑命令：
  ```bash
  RUN_ID=exp032_full_20260917_2020 DEFER_CLEAN_TNP=1 GPU_IDS=0,1,2,3,4,5 nohup setsid bash rpcf/run_exp032.sh run >> logs/exp032_full_20260917_2020.controller.log 2>&1 < /dev/null &
  ```

- 调整后已后台恢复（launcher PID2823769）；保留7个已完成审计。范围验证复用旧smoke预测通过29/29当前任务、70/70指标行、5个Deferred；正式首份新范围汇总为7/2235任务、252/6300指标行，errors为空，结果仍Pending。

## 2026-09-18 提升并行度

- 当前使用`rpcf/run_exp032_parallel.sh`独立调度入口：CPU toy最多4任务（每任务2线程），不占GPU名额；GPU0–5上TNP最多2任务/卡，其余训练、攻击、外部评估保持独占。旧`run_exp032.sh`用于原调度方案，当前运行不要重复启动。
- 当前科学任务范围仍为2,235 active + 450 Deferred，所有任务命令、seed、rank、攻击参数和存储策略与原计划一致。纯CPU toy通过`CUDA_VISIBLE_DEVICES=-1`隔离，已验证CPU插值及HOSVD统计结果与CUDA可见时一致。
- 第二个TNP要求显存空闲至少4096MiB；主机可用内存低于16GiB时暂停派发新任务。TNP并发OOM后全局降为单TNP再试，保留原日志及RNG断点。GPU6/7不使用。
- 原worker通过暂停旧controller并核对PID/start_ticks接管，继续执行。新调度器使用独立锁，读取旧worker真实退出码；所有旧worker结束后旧controller自动退出，新调度器接续原锁。原manifest、任务表、科学hash、scope和已完成结果均保留。
- 控制日志：`logs/exp032_full_20260917_2020.parallel.controller.log`；实时资源与任务映射：`logs/exp032/exp032_full_20260917_2020/parallel_runtime_state.json`；策略：同目录`parallel_policy_v2.json`。
- 启动命令（先完成脚本开头的capture步骤，随后只启动一个controller）：
  ```bash
  RUN_ID=exp032_full_20260917_2020 GPU_IDS=0,1,2,3,4,5 CPU_WORKERS=4 TNP_PER_GPU=2 nohup setsid bash rpcf/run_exp032_parallel.sh run > logs/exp032_full_20260917_2020.parallel.controller.log 2>&1 < /dev/null &
  ```

- 并行调度已上线：launcher3427819/controller3427877。保留6个在途worker，新增4个任务，当前并发10个（4 CPU + 6 GPU）；GPU1/2/3采样利用率96%/95%/94%。旧controller2823828暂停且不再派发，旧worker自然完成后自动退出。6项测试通过，正式结果仍Pending，450个clean-only+TNP仍Deferred。

## 2026-09-20 锁接管修复后的当前状态

- 9月18日14:48并行controller因锁接管未处理BlockingIOError而退出；9月20日巡检发现，已修复锁等待和冷恢复，并通过9项测试。
- 12项已完成产物经严格校验补登记，总完成329/2235；恢复status明确标记未观测退出码和估算elapsed，证据完整保留。450个clean-only+TNP仍Deferred。
- 当前有效策略为`parallel_policy_v3.json`；launcher3670107/controller3670146已获得原controller.lock，`handoff_complete=true`。
- 当前有效日志：`logs/exp032_full_20260917_2020.parallel_v3.controller.log`。GPU池仍为0–5，但1–3暂被其它实验占用显存，因此当前先用GPU0/4/5和4个CPU队列。其它实验未被停止或修改。
- 先前预计日期不再可靠，需按恢复后可用资源重新估计。结果Pending。


### 2026-09-20 GPU6/7 扩容

按用户授权，parallel v4 卡池为 0–7，避让其他实验占用的 GPU1–3，当前可用 0/4/5/6/7。保留 7 个在途任务后新增两卡任务，初始共 9 个并行；CPU toy=4、TNP 每卡上限=2。控制日志为 `logs/exp032_full_20260917_2020.parallel_v4.controller.log`，新 controller PID=3674865。原 2235 active / 450 deferred 范围、科学参数均不变，clean-only+TNP 仍不运行。原三卡耗时估计待扩容吞吐稳定后更新。接管与验证证据见 EXPERIMENTS.md 同日 v4 条目。


### 2026-09-20 GPU1–3 暂停进程共卡

用户授权后，parallel v5对GPU1–3引入绑定PID/start_ticks的共享白名单。每卡最多1个TNP/attack/external_eval任务，启动前剩余显存>=6144MiB、外部进程仍暂停且无未知context；训练继续使用普通卡。外部恢复后停止新派发，在途任务自然完成。已保留原6个worker并新增3个任务，初始9并行；新controller3801846，日志`logs/exp032_full_20260917_2020.parallel_v5.controller.log`。外部进程未被启停，科学协议及2235 active/450 deferred范围不变。启动命令、风险与14项测试证据见EXPERIMENTS.md同日v5条目。

## 2026-09-22：自适应攻击纠正为历史 PGD-10（当前执行协议）

用户指出未授权将自适应攻击加到 PGD-200。此前外部净化评估错误沿用了普通攻击构造器，导致五种攻击都再次攻击完整防御；该扩展停止，不能与历史 TNP BPDA PGD-10 直接合并比较。

- 外部净化的 adaptive 部分仅执行 **PGD-10，L∞ eps=0.03，alpha=0.006，attack batch=1，clean 起点，无随机起点、无额外重启、无 EOT、不额外裁剪 EEG 数据范围，返回最后一步**；512 样本和原 seed/fold 抽样规则保持一致。TNP 历史实现使用 BPDA；MagNet/DCAE 可微，使用净化器和分类器的真实梯度，明确标注区别。
- 原五种普通攻击及其非自适应净化评估继续保留原协议。每条件外部评估仍为 5 攻击×2 净化器共 10 个调度任务，仅原 `pgd` 槽位额外计算一次 `pgd10` adaptive 行；不再运行 adaptive FGSM/AA/CW/L2，也不重复计算五份 PGD-10。
- 当前验收改为每条件 **62 行 = 30 raw + 20 TNP + 10 external nonadaptive + 2 external adaptive PGD10**，90 条件共 **5,580 行**。2235 active / 450 clean-only+TNP Deferred 的逻辑任务数不变；此条替代前述 70 行/6300 行的旧验收口径。
- 原任务清单、状态、日志、训练与攻击产物均保留。新外部结果、任务状态和汇总隔离在 `logs/exp032/exp032_full_20260917_2020/execution_revisions/adaptive_pgd10_v1/`；`current_execution.json` 指向当前目录，根目录心跳注明 `execution_dir/status_dir`。统计进度须读取新状态目录，不能继续统计旧 `status/`。
- 迁移核验：773 个未改协议任务保留完成状态；368 个非 PGD 外部任务只提取已验证的 nonadaptive 行；90 个已完成的旧 PGD 外部任务须重新评估 PGD-10。因此新协议起点为1141/2235 completed，进度回退反映协议更正，原产物未丢失。
- 后台入口为 `rpcf/run_exp032_pgd10.sh`，正式日志为 `logs/exp032_full_20260917_2020.pgd10_v1.controller.log`；完成后自动执行 `rpcf.summarize_exp032_pgd10`。原 v5 入口及原汇总器仅用于解释旧记录，不用于当前修订续跑和验收。
- 该更正不启动另行规划的 EXP-033–036，不改变已有模型、rank 或训练设置。正式实验仍 **Pending**。
