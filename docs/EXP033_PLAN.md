# EXP-033：结构对照、参数敏感性、TRP+clean 消融与可视化

- 建档日期：2026-09-23。
- 状态：实现与 smoke 验收已完成；正式实验 **Pending**。
- 范围：THU Benchmark × EEGNet × seed 42–46 × fold0。CAF = RPCF_AT，TRP = TNP。
- 独立入口：`rpcf/exp033.py`；产物统一写入 `logs/exp033/<run_id>/`，包括新 checkpoint、攻击、断点、预测、案例、图表及 manifest。
- 源实验：`exp031_full_20260729_174215`；clean-only / MagNet / DCAE 与审计来自 `exp032_full_20260917_2020`。源实验只读。

## 固定协议

先按原 `seed_everything`、`stable_subset_indices` 和 fold 数据划分取得规范 n512，再核对 indices、labels、clean、checkpoint 指纹与来源。smoke 只取该子集前2例，不重新抽样。PGD-200，L∞ ε=0.03，α=2/255，无随机起点；新 CAF checkpoint 必须生成自身攻击。报告净化后 clean/robust accuracy、五 seed 均值与样本标准差（ddof=1）、逐 seed 配对差值；指标为 0–1 比例。

正式图表不混入 smoke。缺 seed、缺产物、修改指纹或指标无法从逐样本预测复算时不能通过严格验收。测试结果不用于回选 rank 配置或主实验损失权重。

## 结构效果与效率（structure）

固定同 seed Madry classifier、同一攻击样本和公共变换：EEG 插值及时间重采样至 10×11×2048，再用同一逆变换还原。

| 方法 | 表示 / 实现 |
|---|---|
| 当前 PTR/QTR（ptr） | 原 `PTR_3d` 渐进优化和时间量化；准确率复用，计时重测 |
| 普通 TR（tr_dense） | TensorLy TR-SVD，非均匀闭环 rank，所有环连接 ≥2 |
| 普通 TT（tt_dense） | TensorLy TT-SVD，10×11×2048 |
| 时间张量化 TT（tt_time） | TensorLy TT-SVD，10×11×2×…×2，11个二进制时间维 |
| Tucker（tucker） | TensorLy HOOI，SVD 初始化，最多100轮，tol=1e-4 |
| 矩阵 SVD（svd） | PyTorch 截断 SVD，110×2048 |

不升级当前 TensorLy 0.9.0 / PyTorch 2.4.1 依赖。实现参考仓库 `tensor_ring_rank_analysis/analyze_tr_rank_predictions.py`、既有 TT-SVD 及已安装库源码。普通 TR 先将张量外部转为时间/空间/空间，原 rank `[a,b,c,a]` 转为 `[c,a,b,c]`，TensorLy mode=0，重构后逆排列，避免库内非零 mode 对非均匀闭环的重排问题。

两档表示参数预算分别为14,731和19,891，不计分类器。普通 TR 低档固定 `[3,22,2,3]`、13,432参数，记录用户确认的 −8.8% 例外；其余候选范围为目标 ±5%。枚举满足分解形状约束的 rank，去除逐维被另一候选支配的配置。时间 TT 枚举可行 uniform rank cap 并展开边界 rank。每 seed 仅用原 validation split 的32个 clean 样本，按平均相对 Frobenius 重构误差选择；平局按预算绝对偏差、rank 字典序决定。配置、分数、validation indices 及输入哈希全部保存；普通 TT/Tucker 的矩阵等价情形明确记录。

正式新增50个准确率条件，复用当前方法10个条件。60个独立计时条件：同 seed、方法、预算的规范子集前8 clean +前8 adv，预热后逐例测量。GPU 同步，独占检测，排除磁盘、首次导入和分类器推理；输出分解/完整净化时间、峰值显存、实际参数和压缩率。标准分解与渐进优化采用不同求解过程，参数匹配不代表优化开销匹配；预算横轴使用实际参数量。CPU smoke 时间不能作为 GPU 正式结果。

## 测试 rank（rank）

冻结默认 CAF；测试15、20、25、30、35、40。25/30复用已有正式测试产物，其余新增20个 rank–seed 条件；新 rank 的 clean 和 adv 均实际执行净化。训练六 rank cache 仅供微调使用，不替代测试净化。

## 损失权重（loss）

| 实际项 | 配置字段 | 扫描值 |
|---|---|---|
| clean CE | clean_ce_weight | 0、0.5、1、2 |
| purified-clean CE | pur_ce_weight | 0、0.25、0.5、1 |
| purified-adversarial CE | adv_pur_ce_weight | 0、0.5、1、2 |
| purified-clean KL | lambda_pur | 0、0.1、0.2、0.4 |
| purified-adversarial KL | lambda_adv_pur | 0、0.25、0.5、1 |

默认依次1、0.5、1、0.2、0.5；单变量扫描共16个唯一配置。复用默认五 seed，新增75次微调、75次自身攻击、75个双 rank 净化任务。复制 EXP-031 已完成的实际微调命令，仅变更目标权重与独立输出路径：100轮、在线 Madry AT、原学习率/温度、全层、六 rank 等权。未启用的缓存 adversarial CE/KL 不扫描。每个变体报告 raw、TRP25、TRP30，默认点加入各自曲线。

## 消融（ablation）

仅新增 THU × EEGNet ×五 seed 的 TRP+clean PGD-200 净化，每任务输出25/30两个 rank。复用 clean-only 模型与其攻击，校验后复用同 seed 的 clean 净化。另五组 clean-only、Madry、CAF、TRP+Madry、TRP+CAF 引用正式结果；TRP+clean 不扩展至主实验或跨攻击表。

## 可视化（visualize）

固定同一个 clean-only classifier 和配对输入，比较 TRP25/30、MagNet、DCAE。每 seed 按 source index 升序选择不重复的净化成功、净化失败、clean 损伤、方法预测分歧案例，前三类以 TRP25 为选择锚点，分歧包含四种净化输出的预测类别；空类别记录为空。案例覆盖整个 n512 选择范围，选择结果不改变净化参数。

保存 clean、adv、各方法 purified-clean/purified-adv；图中派生扰动 adv−clean、移除残差 adv−purified、剩余误差 purified−clean。输出统一坐标/色标的波形、Welch PSD、STFT 图及真实类别、预测、置信度。汇总全部512样本的重构误差、分类变化、纠正/损伤比例；不会以少数案例代替总体统计。

TRP 只常驻逐例预测、误差和重构哈希；案例在选择后恢复原 RNG 精确回放并验证重构哈希，避免保存所有大张量。AP 使用已完成的 EXP-032 checkpoint，不追加训练。

## 任务清单与隔离

科学条件数与调度任务数不同；一个任务可输出两个 rank 的结果。

| 任务类型 | 正式数量 |
|---|---:|
| 来源审计 sources | 5 |
| validation 配置选择 calibrate | 5 |
| 既有结果复用 reference | 20 |
| 新结构准确率 structure | 50 |
| 独占计时 timing | 60 |
| 净化 tnp（rank20 +loss75 +ablation5） | 100 |
| CAF 微调 train | 75 |
| 新 checkpoint 攻击 attack | 75 |
| 可视化 visualize | 5 |
| 合计 | **395** |

普通队列335、计时队列60。计时队列消费已完成的校准依赖，先完成 compute 再启动 timing；同一 run 控制器锁防止重复启动。默认 all 可顺序依赖调度全部任务。GPU 仅向未发现其他计算进程的卡派发，每卡最多一个本实验 worker；正式计时开始、逐例及结束均核对独占。

manifest 冻结共同配置、任务图与代码/config SHA256；sources 输出保存库版本、原 checkpoint/config/攻击/缓存路径及指纹、规范样本身份。新攻击为 float32，引用既有 clean，保留32GiB剩余空间。产物由单独 metadata 记录 size/mtime/SHA，汇总时验证是否缺失或改变。所有错误保留真实失败状态，不标为完成。

评估断点保存阶段、逐例进度以及 Python/NumPy/PyTorch/CUDA RNG；恢复必须在模型初始化之后进行。原微调器只保存最终 epoch，因此**未完成的训练 attempt 从原初始化和 seed 重新开始**，不承诺从中间 epoch 续训；旧 attempt、日志与产物全部保留，已完成且哈希正确的 checkpoint 可复用。这避免改变 EXP-031/032 的核心训练行为。

## 使用方式

在仓库根目录、torch 环境执行。脚本默认只建档；所有长任务通过 nohup 和实时日志独立启动。

```bash
# 建档 / 查看任务；不会启动训练
RUN_ID=exp033_full_20260923_v6 bash rpcf/run_exp033.sh dry-run

# 真实 CPU smoke：seed42、测试2例、validation2例、1个权重变体
# 1轮训练、首个缓存批次、validation前2例、PTR40步；测试PGD仍为200步；34个调度任务
RUN_ID=exp033_smoke_recheck_v6 nohup setsid bash rpcf/run_exp033.sh smoke --cpu > logs/exp033_smoke_recheck_v6.controller.log 2>&1 < /dev/null &

# 正式普通计算，需手动启动
RUN_ID=exp033_full_20260923_v6 nohup setsid bash rpcf/run_exp033.sh run --queue compute > logs/exp033_full_20260923_v6.compute.log 2>&1 < /dev/null &

# compute完成后，独立计时
RUN_ID=exp033_full_20260923_v6 nohup setsid bash rpcf/run_exp033.sh run --queue timing > logs/exp033_full_20260923_v6.timing.log 2>&1 < /dev/null &

# 可选：仅指定模块 / 卡；依赖自动纳入compute
RUN_ID=exp033_full_20260923_v6 bash rpcf/run_exp033.sh run --queue compute --groups ablation,visualize --gpu-ids 6,7

# 中断后沿用run-id；真实失败检查日志后显式追加 --retry-failed
RUN_ID=exp033_full_20260923_v6 bash rpcf/run_exp033.sh summary --strict

# 最小回归
conda run -n torch --no-capture-output python -m unittest discover -s tests -p 'test_exp033_*.py' -q
```

修改科学代码后需新 run-id，禁止绕过旧清单的冻结校验。只有 smoke 验证通过不代表正式结果已产出，正式表、五 seed 曲线与 GPU 耗时均保持 Pending，待正式任务完成后自动汇总。

## 汇总产物与验收口径

`summary/` 输出 `metrics_long.csv`（逐seed）、`metrics_grouped.csv`（均值/样本标准差）、`paired_differences.csv`、`paired_grouped.csv`、`task_states.csv`、`report.json` 和 `README.md`；结构、rank、各权重和消融图均为PNG/PDF。配对案例及图保存在 `cases/`、`figures/seed*/`。

消融配对分别报告TRP+clean/Madry/CAF相对各自raw分类器、Madry/CAF相对clean，以及同rank TRP+CAF相对TRP+Madry。正式rank曲线的TRP六点不混入raw rank0，raw单列展示。

`status=Complete` 必须同时检查 `smoke` 字段：`smoke=true` 仅证明工程链路通过。严格汇总会核对每种任务应有的方法/rank/预算/变体集合，不能因所有seed同时遗漏某一条件而误判完成。

### smoke 的独立训练包装器

`rpcf/exp033_smoke_finetune.py` 仅接受smoke清单和固定的1轮/在线2样本/1缓存批次参数，拒绝正式清单和重复参数绕过。包装器临时限制validation前两例，并以原DataLoader的首个真实批次调用原`train_epoch`；结束或异常均恢复原函数引用。`smoke_validation.json`记录原始/实际验证样本数、索引、split和实际缓存批次限制。原`max_cache_batches`只在balanced sampler中生效，因此此处显式限制。正式微调仍调用未改动的`rpcf.finetune`。

缩小验证集会改变其随机起点PGD消耗的随机数，smoke保持自身可复现，不宣称其checkpoint与完整validation训练逐位一致。正式随机逻辑保持原值。

## 实施验收完成（2026-09-23）

- 56项单元测试通过，后台脚本语法和diff格式检查通过。
- 真实CPU smoke `exp033_smoke_20260923_v5`：34/34任务完成，严格汇总`Complete`、`smoke=true`、`errors=[]`；48条逐seed指标、30条配对差值，以及结构/rank/权重/消融图与真实配对案例。
- 审计报告：`logs/exp033/exp033_smoke_20260923_v5/summary/report.json`；验收与最终报告器指纹：同run的`verification/acceptance.json`。只有reporter的旧artifact相对路径解析在计算结束后修正，计算代码和原始结果均未改写；随后重新通过严格汇总。
- 最终正式清单：`logs/exp033/exp033_full_20260923_v6/manifest.json`，395任务、状态Pending、尚未启动。早期v1–v5正式清单保留为实施记录，正式运行使用v6。上面的smoke命令使用新的`exp033_smoke_recheck_v6`，供需要时复验，不会覆盖已验收产物。
- GPU独占计时和五seed正式结果尚未运行；CPU smoke时间及两例准确率不能作为论文结果。
