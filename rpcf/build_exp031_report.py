"""从 EXP-031 严格汇总的标量 CSV 生成可审查的最终报告。

运行：python3 rpcf/build_exp031_report.py --summary-dir logs/exp031/<run-id>/summary \
    --output-dir docs/EXP031_results_final
"""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("thubenchmark", "seediv", "bciciv2a")
MODELS = ("eegnet", "deepconvnet", "tsception", "atcnet", "conformer", "tcnet")
SEEDS = (42, 43, 44, 45, 46)
ATTACKS = ("autoattack", "fgsm", "pgd", "cw")
RAW_METHODS = ("madry", "trades", "fbf", "ea_forward", "rpcf_at")
TNP_METHODS = ("madry", "rpcf_at")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    ]


def describe(values, signed=False):
    """五种子均值与样本标准差；差值输入和输出都使用百分点。"""
    if len(values) != 5:
        raise ValueError(f"need five seeds, got {len(values)}")
    mean = statistics.mean(values.values())
    std = statistics.stdev(values.values())
    return f"{mean:+.2f} ± {std:.2f}" if signed else f"{mean:.2f} ± {std:.2f}"


def build_index(rows):
    index = {}
    for row in rows:
        key = (row["dataset"], row["model"], int(row["seed"]), row["method"], row["attack"],
               row["metric"], row["rank"])
        value = float(row["value"])
        if key in index or not 0 <= value <= 1:
            raise ValueError(f"duplicate or invalid metric: {key}")
        index[key] = value
    return index


def score(index, dataset, model, method, attack, metric, rank):
    """返回五个种子的百分点，不允许缺失条件被静默聚合。"""
    return {seed: 100 * index[(dataset, model, seed, method, attack, metric, str(rank))]
            for seed in SEEDS}


def delta(left, right):
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise ValueError("paired comparison requires exactly five matching seeds")
    return {seed: left[seed] - right[seed] for seed in SEEDS}


def condition_counts(deltas):
    means = [statistics.mean(values.values()) for values in deltas]
    return sum(value > 0 for value in means), sum(value < 0 for value in means), sum(value == 0 for value in means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = args.summary_dir
    complete = json.loads((summary / "completeness.json").read_text(encoding="utf-8"))
    if not complete.get("completed") or complete.get("errors") or not complete.get("streaming_tensor_validation"):
        raise SystemExit("strict summary is incomplete; refusing to publish final report")
    if (complete.get("long_row_count"), complete.get("bpda_row_count")) != (7920, 10):
        raise SystemExit("strict summary row counts do not match the EXP-031 protocol")
    rows = read_csv(summary / "conditions_long.csv")
    bpda = read_csv(summary / "bpda.csv")
    if len(rows) != 7920 or len(bpda) != 10:
        raise SystemExit("summary CSV row count mismatch")
    index = build_index(rows)
    if len(index) != len(rows):
        raise SystemExit("duplicate metric rows")

    lines = [
        "# EXP-031 完整矩阵最终汇总", "",
        f"严格汇总 run-id：`{complete['run_id']}`；任务、训练记录、攻击与净化产物校验通过。", "",
        "## 口径", "",
        "- 三数据集 × 六 backbone × 五种子（42–46）、fold0；表格准确率均为五种子均值 ± 样本标准差，单位 %；差值单位百分点（pp）。",
        "- [F] 是完整 test split，[S] 是按预先固定索引选出的最多 n512 子集。[F] 与 [S] 不能直接相减；本报告的配对差值只在同一范围、同一数据集/backbone/seed/attack 内计算。",
        "- Madry 与 RPCF_AT 各用自身分类器生成 white-box 攻击，再分别净化；因此配对比较是方法级结果，不是同一对抗样本上的纯净化效应。",
        "- AA/FGSM/PGD 是 L∞ ε=0.03；CW 是不受该 ε 限制的 L2 攻击。不同范数不求混合平均，也不把任何跨攻击均值解释为最坏情况鲁棒性。",
        "- TNP 净化使用固定测试 rank25/30；RPCF_AT 训练为全层、在线 PGD-10、六 rank 静态 1/6、无 feature loss。",
        "- 以下为已保存产物的严格一致性审核与统计，不是重新训练或重新运行攻击；EA-forward clean 字段的异常单独列出。", "",
    ]

    # 完整测试集 raw 鲁棒性：只显示同范数 PGD-200，全部攻击值在源 CSV。
    raw_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            raw_rows.append([dataset, model] + [
                describe(score(index, dataset, model, method, "pgd", "robust_accuracy", "raw"))
                for method in RAW_METHODS
            ])
    lines += ["## 完整测试集 raw PGD-200 鲁棒准确率 [F]", "",
              "攻击分别针对各自 checkpoint；以下各列处于相同的完整 test 范围。", ""]
    lines += table(["数据集", "Backbone", "Madry", "TRADES", "FBF", "EA-forward", "RPCF_AT"], raw_rows)
    lines += ["", "## n512 净化后 PGD-200 鲁棒准确率 [S]", "",
              "Madry 与 RPCF_AT 的 TNP 结果处于相同 source indices/labels；这不是完整 test 准确率。", ""]
    tnp_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            tnp_rows.append([dataset, model] + [
                describe(score(index, dataset, model, method, "pgd", "purified_robust_accuracy", rank))
                for rank in (25, 30) for method in TNP_METHODS
            ])
    lines += table(["数据集", "Backbone", "Madry+TNP r25", "RPCF+TNP r25",
                    "Madry+TNP r30", "RPCF+TNP r30"], tnp_rows)

    lines += ["", "## 直接配对：RPCF_AT − Madry", "",
              "正数表示 RPCF_AT 更高。raw [F] 与 TNP [S] 的差值分别计算，不能跨列相减。", ""]
    pgd_delta_rows = []
    comparisons = defaultdict(list)
    for dataset in DATASETS:
        for model in MODELS:
            cells = [dataset, model]
            for attack in ATTACKS:
                raw = delta(score(index, dataset, model, "rpcf_at", attack, "robust_accuracy", "raw"),
                            score(index, dataset, model, "madry", attack, "robust_accuracy", "raw"))
                comparisons[(attack, "raw [F]")].append(raw)
                if attack == "pgd":
                    cells.append(describe(raw, signed=True))
                for rank in (25, 30):
                    purified = delta(
                        score(index, dataset, model, "rpcf_at", attack, "purified_robust_accuracy", rank),
                        score(index, dataset, model, "madry", attack, "purified_robust_accuracy", rank),
                    )
                    comparisons[(attack, f"TNP r{rank} [S]")].append(purified)
                    if attack == "pgd":
                        cells.append(describe(purified, signed=True))
            pgd_delta_rows.append(cells)
    lines += table(["数据集", "Backbone", "raw [F]", "TNP r25 [S]", "TNP r30 [S]"], pgd_delta_rows)
    lines += ["", "各攻击中，18 个数据集–backbone 条件的五种子平均配对差值方向（仅描述性计数，非显著性检验）：", ""]
    counts_rows = []
    for attack in ATTACKS:
        for scope in ("raw [F]", "TNP r25 [S]", "TNP r30 [S]"):
            positive, negative, tied = condition_counts(comparisons[(attack, scope)])
            counts_rows.append([attack, scope, positive, negative, tied])
    lines += table(["攻击", "比较", "RPCF 更高", "Madry 更高", "持平"], counts_rows)

    lines += ["", "## 同模型净化变化：TNP − 未净化 [S]", "",
              "下表先在同一 n512 子集、同一 checkpoint、同一攻击和种子内求差，再统计18个条件的方向。正数表示净化后鲁棒准确率更高。", ""]
    change_rows = []
    for attack in ("autoattack", "pgd"):
        for method in TNP_METHODS:
            for rank in (25, 30):
                changes = [delta(
                    score(index, dataset, model, method, attack, "purified_robust_accuracy", rank),
                    score(index, dataset, model, method, attack, "subset_robust_accuracy", "subset_raw"),
                ) for dataset in DATASETS for model in MODELS]
                change_rows.append([attack, method, rank, *condition_counts(changes)])
    lines += table(["攻击", "模型", "rank", "净化提高", "净化降低", "持平"], change_rows)

    bpda_groups = defaultdict(dict)
    for row in bpda:
        key = (row["dataset"], row["model"], int(row["rank"]))
        seed = int(row["seed"])
        if seed in bpda_groups[key]:
            raise ValueError(f"duplicate BPDA row: {key}/{seed}")
        bpda_groups[key][seed] = row
    if set(bpda_groups) != {("thubenchmark", "eegnet", 25), ("thubenchmark", "eegnet", 30)}:
        raise ValueError("unexpected BPDA condition set")
    bpda_table = []
    for rank in (25, 30):
        group = bpda_groups[("thubenchmark", "eegnet", rank)]
        if set(group) != set(SEEDS):
            raise ValueError("BPDA seeds incomplete")
        bpda_table.append([rank,
                           describe({seed: 100 * float(group[seed]["purified_clean_accuracy"]) for seed in SEEDS}),
                           describe({seed: 100 * float(group[seed]["bpda_purified_adv_accuracy"]) for seed in SEEDS})])
    lines += ["", "## 有限自适应攻击审计", "",
              "仅 THU/EEGNet/RPCF_AT+TNP 五种子，BPDA+PGD-10（L∞ ε=0.03、步长0.006）；没有对应 Madry BPDA 对照，且步数与上表 PGD-200 不同，不作横向强弱结论。", ""]
    lines += table(["rank", "净化 clean", "BPDA 鲁棒"], bpda_table)

    clean_spreads = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                values = [index[(dataset, model, seed, "ea_forward", attack, "standard_accuracy", "raw")]
                          for attack in ATTACKS]
                spread = 100 * (max(values) - min(values))
                if spread > 1e-8:
                    clean_spreads.append((spread, dataset, model, seed))
    clean_spreads.sort(reverse=True)
    lines += ["", "## 审计警告与解释边界", "",
              f"- EA-forward 同一 checkpoint 的四攻击记录中，{len(clean_spreads)}/90 个 dataset–backbone–seed 条件的 raw clean accuracy 不一致；最大跨度 {clean_spreads[0][0]:.2f} pp（{clean_spreads[0][1]}/{clean_spreads[0][2]}/seed{clean_spreads[0][3]}）。攻击脚本在攻击循环后才计算 clean accuracy，但目前未证实这就是数值漂移的充分原因。该异常使 EA-forward clean 数值及其跨攻击解释需要复核；本报告不修正原产物。",
              "- RPCF_AT 含额外 100 epochs 适配训练；此矩阵不能单独归因于 logit 对齐、全层微调或静态 rank 权重，也不能宣称在所有条件下普遍优于 baseline。",
              "- 攻击与净化结果的协议、索引、标签、clean 张量内容与准确率范围经过严格汇总校验；仍未独立重跑模型推理或攻击。", "",
              "## 可复核产物", "",
              f"- 严格完整性：[completeness.json](../../{summary.as_posix()}/completeness.json)",
              f"- 全指标长表：[conditions_long.csv](../../{summary.as_posix()}/conditions_long.csv)",
              f"- 五种子聚合：[five_seed_mean_std.csv](../../{summary.as_posix()}/five_seed_mean_std.csv)",
              f"- 配对差值：[rpcf_at_minus_madry_paired.csv](../../{summary.as_posix()}/rpcf_at_minus_madry_paired.csv)",
              f"- 净化变化：[tnp_minus_raw_paired.csv](../../{summary.as_posix()}/tnp_minus_raw_paired.csv)",
              f"- BPDA：[bpda.csv](../../{summary.as_posix()}/bpda.csv)", ""]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "run_id": complete["run_id"],
        "strict_summary_completed": True,
        "strict_summary_dir": str(summary),
        "conditions_sha256": hashlib.sha256((summary / "conditions_long.csv").read_bytes()).hexdigest(),
        "completeness_sha256": hashlib.sha256((summary / "completeness.json").read_bytes()).hexdigest(),
        "metric_rows": len(rows),
        "bpda_rows": len(bpda),
        "ea_clean_inconsistent_conditions": len(clean_spreads),
        "seed_aggregation": "paired within seed; five-seed mean and sample std (ddof=1)",
        "new_experiment_run": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
