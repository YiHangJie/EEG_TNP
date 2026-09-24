"""EXP-033 的逐样本验收、五 seed 汇总与配对案例图。

所有导出只写入当前 run 的 ``summary`` 目录。缺失任务保留 Pending，
准确率仅在逐样本身份和预测通过检查后参与统计，smoke 不产生正式结论。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ACCURACIES = ("standard_accuracy", "robust_accuracy")
LOSS_DEFAULTS = {
    "clean_ce_weight": 1.0,
    "pur_ce_weight": 0.5,
    "adv_pur_ce_weight": 1.0,
    "lambda_pur": 0.2,
    "lambda_adv_pur": 0.5,
}
CONDITION_FIELDS = (
    "group", "method", "variant", "rank", "budget_rank", "evaluation",
    "scan_parameter", "scan_value", "dataset", "model", "attack", "comparison",
)


def _atomic_text(path: Path, value: str) -> None:
    """生成报表允许重复，但中断不能留下半写入的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, payload) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                         else value for key, value in row.items()})
    _atomic_text(path, stream.getvalue())


def _condition(row: dict) -> dict:
    # 各 seed 在 validation 选出的实际 rank 可以不同；结构比较按目标预算聚合。
    return {key: row[key] for key in CONDITION_FIELDS
            if row.get(key) is not None and not (row.get("group") == "structure" and key == "rank")}


def _key(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _group_rows(rows: list[dict], expected_seeds: list[int]) -> tuple[list[dict], list[str]]:
    buckets = defaultdict(list)
    errors = []
    for row in rows:
        buckets[_key(_condition(row))].append(row)
    grouped = []
    excluded = set(CONDITION_FIELDS) | {"seed", "sample_num", "task_id"}
    for key, values in sorted(buckets.items()):
        counts = Counter(row["seed"] for row in values)
        duplicates = sorted(seed for seed, count in counts.items() if count != 1)
        if duplicates:
            errors.append(f"Duplicate seed rows for {key}: {duplicates}")
            # 重复 seed 不作为独立观测，也不任意选择其中一个参与均值。
            values = [row for row in values if row["seed"] not in duplicates]
        seeds = sorted(row["seed"] for row in values)
        result = json.loads(key)
        result.update(n=len(seeds), seeds=seeds,
                      missing_seeds=sorted(set(expected_seeds) - set(seeds)),
                      complete=seeds == sorted(expected_seeds) and not duplicates)
        numeric_fields = sorted(set().union(*(set(row) for row in values)) - excluded) if values else []
        for field in numeric_fields:
            numbers = [row[field] for row in values if field in row and _is_number(row[field])]
            if not numbers:
                continue
            result[field + "_n"] = len(numbers)
            result[field + "_mean"] = statistics.mean(numbers)
            result[field + "_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        grouped.append(result)
    return grouped, errors


def _paired_rows(rows: list[dict]) -> list[dict]:
    """只生成唯一可确定的同 seed 参照，不跨模型或不同预算配对。"""
    paired = []
    for row in rows:
        group = row.get("group")
        if not all(metric in row for metric in ACCURACIES):
            continue
        if group == "ablation":
            # 净化与自己的 raw 分类器配对；另外保留 CAF 与 Madry 的同 rank 对照。
            references = {
                "madry": [("clean", 0)],
                "caf": [("clean", 0)],
                "trp_clean": [("clean", 0)],
                "trp_madry": [("madry", 0)],
                "trp_caf": [("caf", 0), ("trp_madry", row.get("rank"))],
            }.get(row.get("method"), [])
            ignored = {"rank", "method", "variant", "evaluation"}
            identity = {key: value for key, value in _condition(row).items() if key not in ignored}
            for reference_method, reference_rank in references:
                candidates = [candidate for candidate in rows
                              if candidate.get("seed") == row["seed"]
                              and candidate.get("method") == reference_method
                              and candidate.get("rank") == reference_rank
                              and {key: value for key, value in _condition(candidate).items() if key not in ignored} == identity
                              and all(metric in candidate for metric in ACCURACIES)]
                if len(candidates) != 1:
                    continue
                reference = candidates[0]
                result = dict(_condition(row), seed=row["seed"], reference_method=reference_method,
                              reference_variant=reference.get("variant"), reference_rank=reference_rank,
                              comparison=f"{row['method']}_vs_{reference_method}")
                result.update({metric + "_delta": row[metric] - reference[metric] for metric in ACCURACIES})
                paired.append(result)
            continue
        if group == "structure":
            if row.get("method") == "ptr":
                continue
            ignored = {"method", "rank", "variant"}
            predicate = lambda candidate: candidate.get("method") == "ptr"
        elif group == "loss":
            if row.get("variant") == "default":
                continue
            ignored = {"variant", "scan_parameter", "scan_value"}
            predicate = lambda candidate: candidate.get("variant") == "default"
        elif group == "rank":
            if row.get("rank") == 0:
                continue
            ignored = {"rank", "method", "variant", "evaluation"}
            predicate = lambda candidate: candidate.get("rank") == 0
        elif group == "visualize":
            if row.get("method") == "raw":
                continue
            ignored = {"rank", "method", "variant", "evaluation"}
            predicate = lambda candidate: candidate.get("method") == "raw" and candidate.get("rank") == 0
        else:
            continue
        identity = {key: value for key, value in _condition(row).items() if key not in ignored}
        candidates = [candidate for candidate in rows
                      if candidate.get("seed") == row["seed"] and predicate(candidate)
                      and {key: value for key, value in _condition(candidate).items() if key not in ignored} == identity
                      and all(metric in candidate for metric in ACCURACIES)]
        if len(candidates) != 1:
            continue
        reference = candidates[0]
        result = dict(_condition(row), seed=row["seed"], reference_method=reference.get("method"),
                      reference_variant=reference.get("variant"), reference_rank=reference.get("rank"))
        result.update({metric + "_delta": row[metric] - reference[metric] for metric in ACCURACIES})
        paired.append(result)
    return paired


def _validate_identity(payload: dict, expected_n: int, task: dict, identities: dict) -> list:
    indices, labels = payload.get("source_indices"), payload.get("labels")
    if not isinstance(indices, list) or not isinstance(labels, list):
        raise ValueError("Missing source_indices/labels")
    if len(indices) != expected_n or len(labels) != expected_n:
        raise ValueError(f"Expected {expected_n} source indices and labels")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
        raise ValueError("Invalid source index")
    if len(set(indices)) != expected_n:
        raise ValueError("Duplicate source indices")
    if any(not isinstance(label, int) or isinstance(label, bool) or label < 0 for label in labels):
        raise ValueError("Invalid label")
    seed = int(task["seed"])
    identity = (indices, labels)
    if seed in identities and identities[seed] != identity:
        raise ValueError(f"Cross-method source_indices/labels mismatch for seed {seed}")
    identities[seed] = identity
    return labels


def _validate_row(row: dict, labels: list | None, task: dict, expected_n: int) -> dict:
    row = dict(row)
    row.setdefault("group", task.get("group"))
    row.setdefault("seed", int(task["seed"]))
    row["seed"] = int(row["seed"])
    if row["seed"] != int(task["seed"]) or row.get("group") != task.get("group"):
        raise ValueError("Row seed/group disagrees with task")
    if not row.get("method"):
        raise ValueError("Missing method")
    if row.get("rank") == 0 and not row.get("evaluation"):
        # 历史 raw 行未显式标记 evaluation，新攻击的等价行带 classifier_whitebox。
        row["evaluation"] = "classifier_whitebox"
    if row.get("evaluation") == "timing":
        for field in ("decomposition_seconds_mean", "total_seconds_mean", "peak_memory_bytes", "parameters"):
            if not _is_number(row.get(field)) or row[field] < 0:
                raise ValueError(f"Invalid timing field {field}")
    else:
        if labels is None or row.get("sample_num") != expected_n:
            raise ValueError("Invalid accuracy sample count")
        for metric, field in zip(ACCURACIES, ("clean_predictions", "adv_predictions")):
            prediction = row.get(field)
            if not isinstance(prediction, list) or len(prediction) != len(labels):
                raise ValueError(f"Invalid {field} count")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in prediction):
                raise ValueError(f"Invalid {field} class")
            accuracy = sum(value == label for value, label in zip(prediction, labels)) / len(labels)
            if not _is_number(row.get(metric)) or abs(row[metric] - accuracy) > 1e-12:
                raise ValueError(f"Prediction replay mismatch for {metric}: {row.get(metric)} != {accuracy}")
            row[metric] = accuracy
    for field, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite metric {field}")
    row["task_id"] = task["task_id"]
    return row


def _validate_result_rows(rows: list[dict], task: dict) -> None:
    """逐任务核对预期方法/rank 集合；整组结果缺失也不能被误判为齐全。"""
    kind, group = task.get("kind"), task.get("group")
    variant = task.get("variant", "default")

    def signature(method, rank=None, budget_rank=None, evaluation=None):
        return (method, None if group == "structure" else rank, budget_rank, variant, evaluation)

    raw = "classifier_whitebox"
    purified = "nonadaptive_purification"
    if kind in {"sources", "calibrate", "train"}:
        expected = []
    elif kind == "reference":
        if group == "structure":
            expected = [signature("ptr", budget_rank=rank, evaluation=purified) for rank in (25, 30)]
        elif group in {"rank", "loss"}:
            expected = [signature("caf", 0, evaluation=raw)]
            expected += [signature("trp_caf", rank, evaluation=purified) for rank in (25, 30)]
        elif group == "ablation":
            expected = [signature(method, 0, evaluation=raw) for method in ("clean", "madry", "caf")]
            expected += [signature(method, rank, evaluation=purified)
                         for method in ("trp_madry", "trp_caf") for rank in (25, 30)]
        else:
            raise ValueError(f"Unknown reference group: {group}")
    elif kind == "structure":
        expected = [signature(task["method"], budget_rank=task["budget_rank"], evaluation=purified)]
    elif kind == "timing":
        expected = [signature(task["method"], budget_rank=task["budget_rank"], evaluation="timing")]
    elif kind == "attack":
        expected = [signature("caf", 0, evaluation=raw)]
    elif kind == "tnp":
        method = "trp_clean" if task["method"] == "clean" else "trp_caf"
        expected = [signature(method, rank, evaluation=purified) for rank in task["ranks"]]
    elif kind == "visualize":
        expected = [signature("raw", 0, evaluation=raw)]
        expected += [signature("trp_clean", rank, evaluation=purified) for rank in (25, 30)]
        expected += [signature(method) for method in ("magnet", "dcae")]
    else:
        # 通用 evaluate 测试与未来独立扩展不强行套用当前生产任务模板。
        return
    actual = []
    for row in rows:
        actual.append((row.get("method"), None if group == "structure" else row.get("rank"),
                       row.get("budget_rank"), row.get("variant", "default"), row.get("evaluation")))
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    if expected_counts != actual_counts:
        missing = list((expected_counts - actual_counts).elements())
        unexpected = list((actual_counts - expected_counts).elements())
        raise ValueError(f"Result row signatures mismatch: missing={missing}, unexpected={unexpected}")


def _visualization_statistics(rows: list[dict], payload: dict, labels: list) -> None:
    """对全部规范样本汇总误差和分类变化，防止个例代替总体证据。"""
    visual = [row for row in rows if row.get("group") == "visualize"]
    raw_rows = [row for row in visual if row.get("method") == "raw" and row.get("rank") == 0]
    if not visual or len(raw_rows) != 1:
        return
    raw = raw_rows[0]
    diagnostics = payload.get("statistics", {})
    for row in visual:
        for prefix, field in (("clean", "clean_predictions"), ("adv", "adv_predictions")):
            before, after = raw[field], row[field]
            row[prefix + "_changed_fraction"] = sum(a != b for a, b in zip(before, after)) / len(labels)
            row[prefix + "_corrected_fraction"] = sum(a != y and b == y for a, b, y in zip(before, after, labels)) / len(labels)
            row[prefix + "_damaged_fraction"] = sum(a == y and b != y for a, b, y in zip(before, after, labels)) / len(labels)
        if row.get("method") == "raw" or not diagnostics:
            continue
        if row["method"].startswith("trp"):
            samples = diagnostics.get("trp", {}).get(str(row.get("rank")))
        else:
            samples = diagnostics.get("external", {}).get(row["method"])
        if samples is None:
            raise ValueError(f"Missing all-sample diagnostics for {row['method']}, rank {row.get('rank')}")
        if len(samples) != len(labels) or [sample.get("source_index") for sample in samples] != payload["source_indices"]:
            raise ValueError("Visualization diagnostic sample identity mismatch")
        for metric in ("clean_mse", "adv_mse_to_clean", "removed_mse"):
            values = [sample.get(metric) for sample in samples]
            if not all(_is_number(value) and value >= 0 for value in values):
                raise ValueError(f"Invalid visualization error metric {metric}")
            row[metric + "_mean"] = statistics.mean(values)


def _resolve_output_path(run_dir: Path, manifest: dict, path) -> Path:
    """从冻结的 run 位置推导仓库根，兼容已有产物与 run 相对路径。"""
    output = Path(path)
    if output.is_absolute():
        return output
    run_dir = Path(run_dir).resolve()
    declared_root = Path(manifest.get("run_dir", "."))
    local = run_dir / output
    if not declared_root.is_absolute() and declared_root != Path("."):
        if output.is_relative_to(declared_root):
            return run_dir / output.relative_to(declared_root)
        # 例如 /repo/logs/exp033/id 与清单 logs/exp033/id 对齐，得到 /repo。
        # 不以当前 cwd 推断 checkpoints 等旧产物的来源。
        relative_parts = declared_root.parts
        if run_dir.parts[-len(relative_parts):] == relative_parts:
            repository_root = run_dir.parents[len(relative_parts) - 1]
            repository = repository_root / output
            if local.exists():
                return local
            if repository.exists():
                return repository
            if output.parts and not (run_dir / output.parts[0]).exists() and (repository_root / output.parts[0]).is_dir():
                return repository
    return local


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifacts(payload: dict, run_dir: Path, manifest: dict) -> None:
    """大产物先比 size/mtime；文件有变化时才重新计算内容哈希。"""
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("Artifact manifest must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or any(field not in artifact for field in ("path", "size", "mtime_ns", "sha256")):
            raise ValueError("Invalid artifact metadata")
        path = _resolve_output_path(run_dir, manifest, artifact["path"])
        if not path.is_file():
            raise ValueError(f"Referenced artifact is missing: {path}")
        metadata = path.stat()
        unchanged = metadata.st_size == artifact["size"] and metadata.st_mtime_ns == artifact["mtime_ns"]
        if not unchanged and _file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Referenced artifact content changed: {path}")


def summarize(run_dir: Path, strict: bool = False) -> dict:
    """检查当前 run 并导出报表；strict 在任务不齐或证据错误时抛出 ValueError。"""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    tasks = json.loads((run_dir / "tasks.json").read_text())
    if isinstance(tasks, dict):
        tasks = tasks["tasks"]
    errors, states, rows, identities = [], [], [], {}
    seeds = manifest.get("seeds", [42] if manifest.get("smoke") else list(range(42, 47)))
    expected_n = 2 if manifest.get("smoke") else 512
    if manifest.get("experiment_id") not in {"EXP-033", "EXP033"}:
        errors.append("Manifest experiment_id must be EXP-033")
    if len(set(seeds)) != len(seeds) or (not manifest.get("smoke") and sorted(seeds) != list(range(42, 47))):
        errors.append("Invalid expected seed coverage")
    if not tasks:
        errors.append("Empty task graph")
    task_ids = [task["task_id"] for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        errors.append("Duplicate task ids")
    for task in tasks:
        identifier = task["task_id"]
        output = _resolve_output_path(run_dir, manifest, task["output_path"])
        status_path = run_dir / "status" / f"{identifier}.json"
        try:
            status_payload = json.loads(status_path.read_text()) if status_path.exists() else {}
            state = status_payload.get("status", "pending")
        except (ValueError, OSError) as error:
            errors.append(f"{identifier}: Invalid status file: {error}")
            state = "invalid"
        record = dict(task_id=identifier, group=task.get("group"), kind=task.get("kind"),
                      seed=task.get("seed"), status=state, output_path=str(output))
        if state != "completed":
            record["reason"] = "Task has no completed status"
            states.append(record)
            continue
        if not output.is_file():
            record.update(status="invalid", reason="Completed task output is missing")
            errors.append(f"{identifier}: Completed task output is missing: {output}")
            states.append(record)
            continue
        try:
            if int(task["seed"]) not in seeds:
                raise ValueError("Task seed is outside manifest seeds")
            if "output_sha256" in status_payload and _file_sha256(output) != status_payload["output_sha256"]:
                raise ValueError("Result JSON hash differs from completed status output_sha256")
            payload = json.loads(output.read_text())
            _validate_artifacts(payload, run_dir, manifest)
            output_rows = payload.get("rows", [])
            if not isinstance(output_rows, list):
                raise ValueError("Output rows must be a list")
            needs_identity = any(row.get("evaluation") != "timing" for row in output_rows)
            labels = _validate_identity(payload, expected_n, task, identities) if needs_identity else None
            validated = [_validate_row(row, labels, task, expected_n) for row in output_rows]
            _validate_result_rows(validated, task)
            if labels is not None:
                _visualization_statistics(validated, payload, labels)
            # 同一任务任何一行无效时整份结果不参与统计。
            rows.extend(validated)
        except (ValueError, TypeError, KeyError, OSError) as error:
            record.update(status="invalid", reason=str(error))
            errors.append(f"{identifier}: {error}")
        states.append(record)
    grouped, duplicate_errors = _group_rows(rows, seeds)
    errors.extend(duplicate_errors)
    pending = [state for state in states if state["status"] != "completed"]
    incomplete_groups = [row for row in grouped if not row["complete"]]
    paired = _paired_rows(rows)
    paired_grouped, pair_errors = _group_rows(paired, seeds)
    errors.extend(pair_errors)
    complete = not errors and not pending and not incomplete_groups
    report = dict(experiment_id="EXP-033", run_id=manifest.get("run_id", run_dir.name),
                  smoke=bool(manifest.get("smoke")), status="Complete" if complete else "Pending",
                  complete=complete, expected_seeds=seeds, expected_sample_num=expected_n,
                  task_count=len(tasks), completed_task_count=sum(s["status"] == "completed" for s in states),
                  errors=errors, pending_tasks=pending, incomplete_groups=incomplete_groups,
                  task_states=states, rows=rows, grouped=grouped,
                  paired_differences=paired, paired_grouped=paired_grouped,
                  note="Smoke 验证，不用于正式科研结论。" if manifest.get("smoke") else
                       "仅汇总当前 run 中已验收的输出；标准差为 seed 间样本标准差。")
    destination = run_dir / "summary"
    destination.mkdir(exist_ok=True)
    _write_csv(destination / "metrics_long.csv", rows)
    _write_csv(destination / "metrics_grouped.csv", grouped)
    _write_csv(destination / "paired_differences.csv", paired)
    _write_csv(destination / "paired_grouped.csv", paired_grouped)
    _write_csv(destination / "task_states.csv", states)
    report["plots"] = _plot_summary(grouped, destination, bool(manifest.get("smoke")),
                                    loss_defaults=manifest.get("loss_defaults", LOSS_DEFAULTS))
    _write_json(destination / "report.json", report)
    lines = [f"EXP-033 — {report['status']}", "", report["note"], "",
             f"已验收任务：{report['completed_task_count']} / {len(tasks)}。",
             f"未完成任务：{len(pending)}；不完整条件：{len(incomplete_groups)}；错误：{len(errors)}。",
             "", "逐 seed 明细、均值/样本标准差及配对差值见同目录 CSV。"]
    if errors:
        lines.extend(["", "错误：", *["- " + error for error in errors]])
    _atomic_text(destination / "README.md", "\n".join(lines) + "\n")
    if strict and not complete:
        raise ValueError(f"EXP033 remains Pending: {len(pending)} tasks, "
                         f"{len(incomplete_groups)} incomplete seed groups, {len(errors)} errors; "
                         f"see {destination / 'report.json'}")
    return report


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, stem: Path, plt) -> list[str]:
    paths = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix("." + extension)
        fd, temporary = tempfile.mkstemp(prefix=f".{stem.name}.", suffix="." + extension, dir=stem.parent)
        os.close(fd)
        try:
            fig.savefig(temporary, dpi=160, bbox_inches="tight", format=extension)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_summary(grouped: list[dict], destination: Path, smoke: bool,
                  loss_defaults: dict | None = None) -> list[str]:
    if not grouped:
        return []
    plt = _pyplot()
    paths = []
    suffix = " [SMOKE: synthetic/limited verification]" if smoke else ""
    if any(not row.get("complete", True) for row in grouped):
        suffix += " [Pending: incomplete seed coverage]"
    loss_defaults = LOSS_DEFAULTS if loss_defaults is None else loss_defaults
    accuracy = [row for row in grouped if "robust_accuracy_mean" in row]
    structure = [row for row in accuracy if row.get("group") == "structure" and "parameters_mean" in row]
    if structure:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
        for ax, metric in zip(axes[0], ACCURACIES):
            for method in sorted({row["method"] for row in structure}):
                values = sorted((row for row in structure if row["method"] == method), key=lambda row: row["parameters_mean"])
                ax.errorbar([row["parameters_mean"] for row in values], [row[metric + "_mean"] for row in values],
                            yerr=[row.get(metric + "_std") or 0 for row in values], marker="o", label=method)
            ax.set(xlabel="Actual representation parameters", ylabel=metric.replace("_", " "), ylim=(0, 1))
            ax.grid(alpha=.2)
            ax.legend(fontsize="small")
        fig.suptitle("Structure accuracy" + suffix)
        paths.extend(_save_figure(fig, destination / "structure_accuracy_parameters", plt))
        timings = [row for row in grouped if row.get("group") == "structure" and row.get("evaluation") == "timing"]
        joined = [(row, timing) for row in structure for timing in timings
                  if row.get("method") == timing.get("method") and row.get("budget_rank") == timing.get("budget_rank")]
        if joined:
            fig, ax = plt.subplots(figsize=(7, 4))
            for row, timing in joined:
                label = f"{row['method']} / {row.get('budget_rank', '')}"
                ax.errorbar(timing["total_seconds_mean_mean"], row["robust_accuracy_mean"],
                            xerr=timing.get("total_seconds_mean_std") or 0,
                            yerr=row.get("robust_accuracy_std") or 0, fmt="o", label=label)
            ax.set(xlabel="Purification time (seconds / sample)", ylabel="Robust accuracy", ylim=(0, 1),
                   title="Structure efficiency" + suffix)
            ax.legend(fontsize="small", bbox_to_anchor=(1.01, 1), loc="upper left")
            paths.extend(_save_figure(fig, destination / "structure_time_accuracy", plt))
    rank_rows = [row for row in accuracy if row.get("group") == "rank" and _is_number(row.get("rank"))]
    if rank_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in sorted({row["method"] for row in rank_rows}):
            values = sorted((row for row in rank_rows if row["method"] == method), key=lambda row: row["rank"])
            for metric in ACCURACIES:
                ax.errorbar([row["rank"] for row in values], [row[metric + "_mean"] for row in values],
                            yerr=[row.get(metric + "_std") or 0 for row in values], marker="o",
                            label=f"{method} / {metric.replace('_accuracy', '')}")
        ax.set(xlabel="Test rank", ylabel="Accuracy", ylim=(0, 1), title="Rank sensitivity" + suffix)
        ax.legend(fontsize="small")
        ax.grid(alpha=.2)
        paths.extend(_save_figure(fig, destination / "rank_curve", plt))
    loss_rows = [row for row in accuracy if row.get("group") == "loss"]
    scan_parameters = sorted({row["scan_parameter"] for row in loss_rows if row.get("scan_parameter")})
    for parameter in scan_parameters:
        values = [row for row in loss_rows if row.get("scan_parameter") == parameter and _is_number(row.get("scan_value"))]
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ranks = sorted({row.get("rank", 0) for row in values})
        for rank in ranks:
            selected = [row for row in values if row.get("rank", 0) == rank]
            defaults = [row for row in loss_rows if row.get("variant") == "default" and row.get("rank", 0) == rank]
            if len(defaults) == 1 and parameter in loss_defaults and not any(
                    row["scan_value"] == loss_defaults[parameter] for row in selected):
                selected.append(dict(defaults[0], scan_value=loss_defaults[parameter]))
            selected.sort(key=lambda row: row["scan_value"])
            for metric in ACCURACIES:
                ax.errorbar([row["scan_value"] for row in selected], [row[metric + "_mean"] for row in selected],
                            yerr=[row.get(metric + "_std") or 0 for row in selected], marker="o",
                            label=f"rank {rank} / {metric.replace('_accuracy', '')}")
        ax.set(xlabel=parameter, ylabel="Accuracy", ylim=(0, 1), title="Single-weight sensitivity" + suffix)
        ax.legend(fontsize="small")
        ax.grid(alpha=.2)
        safe = "".join(character if character.isalnum() or character == "_" else "_" for character in parameter)
        paths.extend(_save_figure(fig, destination / f"loss_{safe}", plt))
    ablation = [row for row in accuracy if row.get("group") == "ablation"]
    if ablation:
        fig, ax = plt.subplots(figsize=(max(8, len(ablation) * .65), 4))
        for offset, metric in zip((-.18, .18), ACCURACIES):
            ax.bar([index + offset for index in range(len(ablation))], [row[metric + "_mean"] for row in ablation],
                   width=.36, yerr=[row.get(metric + "_std") or 0 for row in ablation], label=metric.replace("_", " "))
        ax.set_xticks(range(len(ablation)), [f"{row['method']} / {row.get('rank', '')}" for row in ablation],
                      rotation=35, ha="right")
        ax.set(ylabel="Accuracy", ylim=(0, 1), title="Ablation" + suffix)
        ax.legend(fontsize="small")
        paths.extend(_save_figure(fig, destination / "ablation", plt))
    return paths


def render_cases(bundle_path, output_dir) -> list[str]:
    """将同一 clean-only 分类器上的配对案例绘成波形、PSD 和时频图。

    信号为 [1,C,T] 或 [C,T]，固定展示第 0 通道；所有方法共用坐标与色标。
    选择规则由 worker 实现，本函数保留 category/source_index 及空类别记录。
    """
    import numpy as np
    import torch
    from scipy import signal

    bundle_path, output_dir = Path(bundle_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    plt = _pyplot()
    paths, selections, seen = [], [], set()
    for case in bundle.get("case_list", []):
        source_index = int(case["source_index"])
        if source_index in seen:
            raise ValueError(f"Repeated case source_index: {source_index}")
        seen.add(source_index)
        source = case["signals"]

        def array(name):
            value = source[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float64)
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 2 or value.shape[1] < 2 or not np.isfinite(value).all():
                raise ValueError(f"Invalid case signal {name}: expected finite [1,C,T] or [C,T]")
            return value

        clean, adv = array("clean"), array("adv")
        if clean.shape != adv.shape:
            raise ValueError("Case clean/adv shape mismatch")
        fs = float(case.get("sampling_rate", 250))
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError("Invalid sampling_rate")
        methods = [name for name in ("trp25", "trp30", "magnet", "dcae") if name + "_adv" in source]
        if not methods:
            raise ValueError("Case has no purified signal")
        arrays = {name: array(name + "_adv") for name in methods}
        clean_arrays = {name: array(name + "_clean") for name in methods if name + "_clean" in source}
        if any(value.shape != clean.shape for value in list(arrays.values()) + list(clean_arrays.values())):
            raise ValueError("Case purification shape mismatch")
        series = [("clean", clean[0]), ("perturbation", (adv - clean)[0]), ("adv", adv[0])]
        for name in methods:
            series.extend([(name + " purified adv", arrays[name][0]),
                           (name + " removed", (adv - arrays[name])[0]),
                           (name + " remaining", (arrays[name] - clean)[0])])
            if name in clean_arrays:
                series.append((name + " purified clean", clean_arrays[name][0]))
        times = np.arange(clean.shape[1]) / fs
        psds, spectrograms = [], []
        for _, values in series:
            frequencies, power = signal.welch(values, fs=fs, nperseg=min(250, values.size))
            psds.append((frequencies, 10 * np.log10(np.maximum(power, 1e-20))))
            nperseg = min(128, values.size)
            f, t, z = signal.stft(values, fs=fs, nperseg=nperseg, noverlap=min(96, nperseg - 1))
            spectrograms.append((f, t, 10 * np.log10(np.maximum(np.abs(z) ** 2, 1e-20))))
        amplitude = max(float(np.max(np.abs(value))) for _, value in series) or 1.
        psd_limits = (min(float(value.min()) for _, value in psds), max(float(value.max()) for _, value in psds))
        color_limits = (min(float(value.min()) for _, _, value in spectrograms),
                        max(float(value.max()) for _, _, value in spectrograms))
        if psd_limits[0] == psd_limits[1]:
            psd_limits = (psd_limits[0] - 1, psd_limits[1] + 1)
        if color_limits[0] == color_limits[1]:
            color_limits = (color_limits[0] - 1, color_limits[1] + 1)
        fig, axes = plt.subplots(len(series), 3, figsize=(14, len(series) * 1.35), squeeze=False,
                                 constrained_layout=True)
        for index, ((name, values), (frequencies, power), (f, t, spectrum)) in enumerate(zip(series, psds, spectrograms)):
            axes[index, 0].plot(times, values, lw=.6)
            axes[index, 0].set(xlim=(times[0], times[-1]), ylim=(-amplitude, amplitude), ylabel=name)
            axes[index, 1].plot(frequencies, power, lw=.7)
            axes[index, 1].set(xlim=(0, fs / 2), ylim=psd_limits)
            mesh = axes[index, 2].pcolormesh(t, f, spectrum, shading="auto", vmin=color_limits[0], vmax=color_limits[1], cmap="viridis")
            axes[index, 2].set(xlim=(times[0], times[-1]), ylim=(0, fs / 2))
            for ax in axes[index]:
                ax.tick_params(labelsize=7)
        for ax, title in zip(axes[0], ("Waveform: channel 0", "Welch PSD (dB / Hz)", "STFT power (dB)")):
            ax.set_title(title, fontsize=10)
        for ax, label in zip(axes[-1], ("Time (seconds)", "Frequency (Hz)", "Time (seconds)")):
            ax.set_xlabel(label)
        fig.colorbar(mesh, ax=list(axes[:, 2]), label="STFT power (dB)", shrink=.7)
        predictions = case.get("predictions", {})
        descriptions = []
        for name in ("raw", *methods):
            prediction = predictions.get(name, {})
            details = []
            for condition in ("clean", "adv"):
                confidence = prediction.get(condition + "_confidence")
                details.append(f"{condition}={prediction.get(condition, '?')}"
                               + (f" ({float(confidence):.3f})" if confidence is not None else ""))
            descriptions.append(name + ": " + ", ".join(details))
        title = f"Seed {bundle.get('seed', '?')} | {case['category']} | source {source_index} | label {case['label']}"
        if bundle.get("smoke"):
            title += " | SMOKE: synthetic/limited verification"
        fig.suptitle(title + "\n" + "\n".join(descriptions), fontsize=10)
        safe_category = "".join(character if character.isalnum() or character in "_-" else "_" for character in case["category"])
        stem = output_dir / f"seed{bundle.get('seed', 'unknown')}_{safe_category}_{source_index}"
        paths.extend(_save_figure(fig, stem, plt))
        selections.append(dict(category=case["category"], source_index=source_index, label=int(case["label"]),
                               channel=0, sampling_rate=fs, predictions=predictions,
                               signals=[name for name, _ in series], figure_stem=str(stem)))
    metadata_path = output_dir / f"seed{bundle.get('seed', 'unknown')}_cases.json"
    _write_json(metadata_path, dict(seed=bundle.get("seed"), smoke=bool(bundle.get("smoke")),
                                   bundle_path=str(bundle_path.resolve()), cases=selections,
                                   empty_categories=bundle.get("empty_categories", []),
                                   statistics=bundle.get("statistics", {})))
    paths.append(str(metadata_path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bundle:
        if not args.output_dir:
            parser.error("--bundle requires --output-dir")
        print(json.dumps(render_cases(args.bundle, args.output_dir), ensure_ascii=False))
    elif args.run_dir:
        report = summarize(args.run_dir, strict=args.strict)
        print(json.dumps({key: report[key] for key in ("status", "task_count", "completed_task_count", "errors")}, ensure_ascii=False))
    else:
        parser.error("Provide --run-dir or --bundle")


if __name__ == "__main__":
    main()
