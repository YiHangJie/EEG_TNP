"""校验控制器中断后已完成的产物，补登记状态；不伪造遗失的退出码。"""

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import torch

from rpcf.exp031 import (actual_batch, actual_cache_attack_batch, actual_rpcf_batch,
                        actual_rpcf_eval_batch, checkpoint, render_command, reservation_active)
from rpcf.resume_exp032 import digest, prepare_scope


def validate_metrics(payload, task, canonical):
    labels = payload["labels"].tolist() if isinstance(payload["labels"], torch.Tensor) else payload["labels"]
    assert payload["source_indices"] == canonical["source_indices"]
    assert labels == canonical["labels"] and len(labels) == 512
    rows = payload["rows"]
    expected = ({(f"{task.method}_tnp_r{rank}", "nonadaptive_purification") for rank in (25, 30)}
                if task.kind == "tnp" else
                {(f"clean_{task.method}", evaluation) for evaluation in ("nonadaptive_purification", "adaptive_exact_gradient")})
    assert len(rows) == 2 and {(r["method"], r["evaluation"]) for r in rows} == expected
    for row in rows:
        assert (row["dataset"], row["model"], row["seed"], row["attack"]) == (task.dataset, task.model, task.seed, task.attack)
        assert row["sample_num"] == 512 and row["scope"] == "S"
        for key, metric in (("clean_predictions", "standard_accuracy"), ("adv_predictions", "robust_accuracy")):
            assert len(row[key]) == 512
            value = sum(p == y for p, y in zip(row[key], labels)) / 512
            assert abs(row[metric] - value) < 1e-12


def validate_artifact(task, run_dir):
    output = Path(task.output_path)
    assert output.is_file() and output.stat().st_size > 0
    task_log = run_dir / "tasks" / f"{task.task_id}.log"
    log = task_log.read_text()
    assert "Traceback (most recent call last)" not in log
    evidence = [output, task_log]
    command = render_command(task, run_dir)
    if task.kind in {"tnp", "external_eval"}:
        canonical = json.loads((run_dir / "audit" / f"{task.dataset}_{task.model}_seed{task.seed}.json").read_text())
        if task.kind == "tnp":
            payload = torch.load(output, map_location="cpu", mmap=True, weights_only=False)
            assert payload["ranks"] == [25, 30] and payload["meta"]["kind"] == "exp032_tnp_summary"
            assert payload["attack_path"] == command[command.index("--attack-path") + 1]
            assert "TNP rank30 512/512" in log
            for rank in (25, 30):
                diagnostics = payload["sample_diagnostics"][rank]
                assert [r["source_index"] for r in diagnostics] == payload["source_indices"]
                assert all(math.isfinite(r["mse_to_attack"]) and math.isfinite(r["mse_to_clean"])
                           and re.fullmatch(r"[0-9a-f]{64}", r["purified_sha256"]) for r in diagnostics)
                preview = payload["previews_by_rank"][rank]
                assert len(preview) == 2 and torch.isfinite(preview).all()
            metrics_file = Path(str(output) + ".metrics.json")
            metrics = json.loads(metrics_file.read_text())
            assert metrics["rows"] == payload["rows"]
            evidence.append(metrics_file)
        else:
            payload = json.loads(output.read_text())
            assert f"ADAPTIVE {task.method}/{task.attack} 512/512" in log
            assert payload["purifier_meta"]["method"] == task.method
            assert payload["purifier_meta"]["seed"] == task.seed
        validate_metrics(payload, task, canonical)
    elif task.kind == "toy":
        payload = json.loads(output.read_text())
        assert payload["complete"] and payload["sample_num"] == 512 and payload["noise_seed"] == task.seed
        assert f"TOY {task.dataset}/{task.method} 512/512" in log
        source_path = command[command.index("--attack-path") + 1]
        assert payload["source_path"] == source_path
        source = torch.load(source_path, map_location="cpu", mmap=True, weights_only=False)
        assert payload["source_indices"] == [int(i) for i in source["source_indices"][:512]]
        for suffix, expected in (("spectra", 512 * 7 * 3), ("reconstruction", 512 * 7 * 7), ("samples", 512 * 7)):
            file = output.with_suffix(f".{suffix}.csv")
            with file.open() as handle:
                rows = csv.DictReader(handle)
                count = 0
                for row in rows:
                    assert row["dataset"] == task.dataset and int(row["seed"]) == task.seed
                    assert row["source_method"] == task.method
                    count += 1
            assert count == expected, (suffix, count)
            evidence.append(file)
        for suffix in (".png", ".pdf"):
            file = output.with_suffix(suffix)
            assert file.is_file() and file.stat().st_size > 0
            evidence.append(file)
    elif task.kind == "train_standard":
        assert task.method == "clean"
        pattern = f"train_{task.dataset}_{task.model}_no_ea_clean_eps0_{task.seed}_fold0_*_{run_dir.name}_clean_*.log"
        logs = list(Path("log_train_AT").glob(pattern))
        assert len(logs) == 1, logs
        text = logs[0].read_text()
        assert "Early stopping at epoch" in text and "Test Acc:" in text
        assert re.search(r"Acc: [0-9.]+±[0-9.]+, Loss:", text)
        state = torch.load(output, map_location="cpu", mmap=True, weights_only=False)
        reference = torch.load(checkpoint("exp031_full_20260729_174215", task.dataset, task.model, task.seed, "madry"),
                               map_location="cpu", mmap=True, weights_only=False)
        assert set(state) == set(reference)
        assert all(state[k].shape == reference[k].shape and torch.isfinite(state[k]).all() for k in state)
        evidence.append(logs[0])
    else:
        raise ValueError(f"Unsupported orphan task kind: {task.kind}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="exp031_full_20260729_174215")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    args.defer_clean_tnp = False
    run_dir = Path("logs/exp032") / args.run_id
    with (run_dir / "parallel_controller.lock").open("a") as parallel_lock, (run_dir / "controller.lock").open("a") as lock:
        fcntl.flock(parallel_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks, _ = prepare_scope(args, run_dir)
        task_map = {t.task_id: t for t in tasks}
        previous = json.loads((run_dir / "parallel_runtime_state.json").read_text())
        boot_epoch = time.time() - float(Path("/proc/uptime").read_text().split()[0])
        recovered = []
        for item in previous["running"]:
            if reservation_active((item["pid"], item["start_ticks"])):
                raise RuntimeError(f"Worker still active; recovery refused: {item['task_id']}")
            task = task_map[item["task_id"]]
            status_path = run_dir / "status" / f"{task.task_id}.json"
            if status_path.exists():
                assert json.loads(status_path.read_text())["status"] == "completed"
                continue
            evidence = validate_artifact(task, run_dir)
            started = boot_epoch + item["start_ticks"] / os.sysconf("SC_CLK_TCK")
            ended = max(p.stat().st_mtime for p in evidence)
            status = {"task_id": task.task_id, "status": "completed", "returncode": None,
                      "output_exists": True, "output_path": task.output_path,
                      "physical_gpu": item["gpu"], "gpu_id": 0,
                      "elapsed_seconds": max(0, ended - started), "elapsed_seconds_estimated": True,
                      "actual_batch_size": actual_batch(run_dir, task),
                      "actual_cache_attack_batch_size": actual_cache_attack_batch(run_dir, task),
                      "actual_rpcf_batch_size": actual_rpcf_batch(run_dir, task),
                      "actual_rpcf_eval_batch_size": actual_rpcf_eval_batch(run_dir, task),
                      "command": render_command(task, run_dir),
                      "completion_basis": "validated_artifact_and_terminal_log_after_controller_failure",
                      "exit_code_observed": False, "recovered_at_epoch": time.time(),
                      "evidence_sha256": {str(p): digest(p) for p in evidence}}
            recovered.append(status)
            print(f"VALIDATED {task.task_id}", flush=True)
        if args.apply:
            archive = run_dir / "scope_amendments" / f"lock_recovery_{int(time.time())}"
            archive.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(run_dir / "parallel_runtime_state.json", archive / "runtime_before_recovery.json")
            (archive / "validated_statuses.json").write_text(json.dumps(recovered, indent=2) + "\n")
            for status in recovered:
                path = run_dir / "status" / f"{status['task_id']}.json"
                with path.open("x") as handle:
                    json.dump(status, handle, indent=2)
                    handle.write("\n")
            print(f"RECOVERED {len(recovered)} tasks; evidence={archive}", flush=True)
        else:
            print(f"DRY_RUN {len(recovered)} validated tasks; no task status changed", flush=True)


if __name__ == "__main__":
    main()
