# EXP-035：参数敏感性扩展至十个随机种子

- 用户要求：仅扩展参数敏感性，在原 seeds 42–46 基础上新增 47–51；保持科学设置不变，尽可能增加并行度。
- 当前状态：2026-09-30 10:06 正式 **Complete**；run-id `exp035_full_20260928_v1`，345/345任务、550条十seed结果通过严格验收，控制器已退出。不把 smoke、初始化模型或部分 seed 结果计为完成。
- 原结果：只读保留 `logs/exp033/exp033_full_20260923_v6/` 的 275 条 rank/loss 结果，既有 Excel 不自动改写。
- 新结果：`logs/exp035/<run_id>/`；目标新增 275 条，合并为 55 个条件 × 10 seed = 550 条。

## 科学协议

THUBenchmark / EEGNet / fold0 / S512 / no-EA subject split。沿用 `seed_everything(seed)` 的 Python、NumPy、PyTorch、CUDA 随机源与 `stable_subset_indices` 的 NumPy RandomState；子集规则 `seed + fold * 1000`。新 seed 使用自己的 split、初始化、训练缓存、分类器和对抗样本，不复用旧 seed 权重冒充新重复。

rank 为 15、20、25、30、35、40，CAF raw 单列为参考。训练仍使用六 rank 缓存。默认测试 rank25/30 沿用 EXP-031 参考链，其余四 rank 调用 EXP-033 原评估。

| 权重 | 默认 | 扫描实际值（包含默认） |
| --- | ---: | --- |
| clean_ce_weight | 1 | 0、0.5、1、2 |
| pur_ce_weight | 0.5 | 0、0.25、0.5、1 |
| adv_pur_ce_weight | 1 | 0、0.5、1、2 |
| lambda_pur | 0.2 | 0、0.1、0.2、0.4 |
| lambda_adv_pur | 0.5 | 0、0.25、0.5、1 |

每 seed 共 16 个唯一训练配置：默认 + 15 个单变量配置，每个报告 raw、TRP25、TRP30。权重 1× 共用该 seed 的默认实验，不增加独立观测次数。

所有命令从 EXP-031 的实际 completed status 取模板，改变 seed 和新产物路径；扫描训练仅另改指定权重。固定设置：

- Madry 初始化：batch128、lr0.001、weight_decay0.0001、最多400 epochs、patience20；训练 PGD10、epsilon0.03、step0.006。
- 训练缓存：同 seed 训练 split 的512例，AutoAttack epsilon0.03，实际 batch16；rank15/20/25/30/35/40，原 PTR 配置与2048迭代。
- CAF 默认及变体：从同 seed Madry 初始化，100 epochs、batch64、online/eval batch128、lr0.0001、weight_decay0.0001；在线 PGD10、全层、六rank等权，原 CE/KL 温度与其余默认参数保持不变。未启用的缓存 adversarial CE/KL 不扫描。
- 测试攻击：PGD200 / L∞ epsilon0.03 / alpha2/255 / 无随机起点。新权重变体仍使用 EXP-033 的逐样本攻击；默认参考仍使用 EXP-031 的完整 test 攻击及 S512 留存链。原 CLI 的 batch32 实际受安全表限制为16，新目录适配显式保留该有效值。
- canonical clean 净化：依照同 seed Madry AutoAttack 的 rank25/30 原评估链生成；后续 CAF 配置复用同一 clean 净化。保留原子集留存、再次排列、按 source_indices 对齐的顺序。

不改变训练轮数、batch、样本数、攻击步数、净化迭代、rank、精度、优化器、权重默认值或模型选择规则。不执行其他数据集/模型、AP、消融、结构计时等无关实验。

## 依赖与并行

新 seed47–51 的对应 checkpoint/cache 原先均不存在，必须补齐同 seed 来源。

| 任务 | 数量 |
| --- | ---: |
| 数据身份准备 | 5 |
| Madry 初始化训练 | 5 |
| 训练缓存 AutoAttack base | 5 |
| 缓存 rank RNG 规划 | 5 |
| 六 rank 缓存净化 | 30 |
| 缓存合并 | 5 |
| Madry 测试 AutoAttack / PGD | 10 |
| canonical 测试净化 | 5 |
| CAF 默认及15个权重配置训练 | 80 |
| 默认 CAF PGD / rank25/30 净化 | 10 |
| 新种子来源审计 | 5 |
| rank / loss 默认参考 | 10 |
| 四个额外 rank 净化 | 20 |
| 15个权重变体 PGD | 75 |
| 15个权重变体双rank净化 | 75 |
| 合计 | **345** |

资源策略采用原 EXP-033 并行器的 PID 归属、显存/主存预留检查，在新的独立策略文件冻结：

- GPU0–7 动态检查，仅使用无未知计算进程的卡；不停止其他任务。
- 重任务（训练、缓存 AutoAttack、测试 AutoAttack、缓存合并）独占卡，训练可使用全部空闲卡；有轻任务待执行时最多6个重任务，给净化保留资源，纯训练阶段可达8卡。
- 轻任务每卡最多3个，总worker上限24；每worker CPU线程2。最多每轮启动两个任务，提前扣除未实际分配的显存/内存预留，避免同时启动时过量派发。
- 轻/重任务分别预留2/8 GiB显存，另保留1 GiB；主存预留4/10 GiB，整机保留24 GiB；磁盘至少32 GiB。未预留足够资源时等待，不降低 batch。
- 多个 seed 与权重配置并行；各自 train→attack→TNP 依赖保持。训练只等待同 seed 初始化和训练cache，不等待无关测试任务。
- 原始训练缓存按六 rank 并行。先沿原实现生成一次 AutoAttack base 并捕获完整随机状态，然后用原 PTR 初始化函数按原串行顺序推进随机流，保存各 rank 起点；执行同一个原 `generate_cache` 的 rank-shard 接口。每次 clean/adv 净化的四次模板初始化调用保持一致，不使用 `seed+rank` 等新随机规则。
- 新的 rank partial 附完整随机状态和来源指纹，恢复到原流位置。旧无随机状态的 partial 拒绝复用。只有真实 CUDA 的逐位对照与正式2048步随机状态检查通过才启用并行。

## 隔离、断点与验收

旧实验代码、权重、日志、结果和配置保持只读。新脚本为 `rpcf/exp035*.py` 与 `rpcf/run_exp035.sh`。Madry 训练复用原训练器，checkpoint 使用独立 EXP-035 run/attempt 标签；其余新产物放在当前 run 内。

未完成训练从同一初始化与 seed 开始新的 attempt，不覆盖旧 attempt。攻击/TNP 继续沿用原 RNG 断点；前置测试净化额外保存原循环边界的 RNG，恢复时不跳过随机消耗。控制器持排他锁；supervisor 独立保存真实退出码。重启按 PID + start_ticks 校验并接管本实验工作进程，不能仅凭产物存在标完成。

验收核对 task fingerprint、文件 hash、source_indices/labels、攻击模型/输入身份、逐样本预测重算和精确条件集合。原275条也回查已完成任务并与正式CSV比对；新275条通过后才合并。正式完成要求55个条件均有且仅有 seeds42–51，均值和样本标准差采用 ddof=1。

输出 `summary/metrics_new_seeds.csv`、`metrics_long.csv`、`metrics_grouped.csv`、配对差值、任务状态、`report.json`。完成后自动生成 rank 与五权重的 SA/RA 图和源数据；不自动覆盖已有 Excel。

## 操作

```bash
# 建档与只读任务检查
RUN_ID=exp035_full_20260928_v1 bash rpcf/run_exp035.sh plan
# 正式后台启动（请先查看runtime，避免重复启动）
RUN_ID=exp035_full_20260928_v1 nohup setsid bash rpcf/run_exp035.sh run > logs/exp035_full_20260928_v1.controller.log 2>&1 < /dev/null &
# 只读汇总；有缺口时strict返回非零
RUN_ID=exp035_full_20260928_v1 bash rpcf/run_exp035.sh summary --strict
# 回归检查
/home/yihangjie/miniconda3/envs/torch/bin/python -m unittest discover -s tests -p 'test_exp035.py' -v
```

心跳：`logs/exp035/<run_id>/runtime.json`；状态和实时日志分别为 `status/`、`worker_logs/`。真实失败保留证据，检查后用同入口 `run --retry-failed`，不得改变冻结的科学参数。


## 实际验证与正式启动记录

- 真实CUDA预检：`logs/exp035_preflight_20260928_v1/verification.json`为PASS，2样本×6rank串行/拆分缓存逐位一致，rank25中断续跑一致，各rank正式2048步RNG一致；40份旧seed命令归一化一致，32份原冻结科学源码未改变（`protocol_audit.json`）。
- 全流程：`exp035_smoke_20260928_v2`，24/24任务、10/10指标、smoke=true、complete=true、errors=[]。v1仅在首任务发现CUDA首次初始化顺序错误，已修复且保留失败证据；正式冻结版本与通过的v2源码hash一致。
- 四项最小回归、全部新模块语法、shell语法通过。十seed图表链路用临时测试数据验证6张PNG+6张PDF后清除；测试数据不计实验结果。
- 正式：`exp035_full_20260928_v1` 已通过 `launch_review.json` 后以nohup+setsid实际启动，PID2537707，GPU0–7、max_workers24、light_per_gpu3。初始严格科学验收为Pending，完成后自动再次验收；不需手工再启动。
- 持续状态以`runtime.json`为准；初始`summary/report.json`仅保留旧275条，新增结果尚未完成，不把它解释为十seed结果。


## 正式完成（2026-09-30 10:06，Asia/Shanghai）

345/345任务全部completed；runtime为0 running、0 pending、failures=[]。`summary/report.json`为smoke=false、complete=true、errors=[]：新增275条，保留275条，55条件×10seed=550条。10:06:09计算完成，10:06:16报告生成，10:06:24图表产物齐备。

已复核每个seed均55条、每组n=10且complete=true；六组敏感性PNG/PDF及chart_data.csv完整。数据入口为`summary/metrics_grouped.csv`（十seed均值/样本标准差）、`metrics_long.csv`（逐seed来源）、`metrics_new_seeds.csv`（新增47–51）及配对差值；图在`figures/`。运行已经结束，无需再次启动。既有Excel仍等用户要求再整理，研究结论不由完整性验收直接推导。
