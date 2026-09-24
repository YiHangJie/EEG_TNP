"""EXP-033 清单、单变量扫描和恢复门禁；不读取真实数据或启动 worker。"""

import copy
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rpcf import exp033
from rpcf.exp033_common import (ATTACK_PROTOCOL, GROUPS, SEEDS, WEIGHTS, file_hash,
                               loss_variants, read_json, training_arguments, write_json)


def manifest(root="isolated", smoke=False):
    return dict(run_dir=str(root), smoke=smoke, seeds=[42] if smoke else list(SEEDS),
                source_run="source", external_run="external")


def original_training_command():
    command = ["python", "-u", "-m", "rpcf.finetune", "--dataset", "thubenchmark",
               "--model", "eegnet", "--seed", "42", "--fold", "0", "--epsilon", "0.03",
               "--cache_path", "original/cache.pth", "--checkpoint_path", "original/madry.pth",
               "--epochs", "100", "--batch_size", "17", "--eval_batch_size", "63",
               "--online_at_batch_size", "29", "--lr", "0.0001",
               "--consistancy_temperature", "2.0", "--online_madry_at", "--all_layers",
               "--static_rank_weights", "--output_checkpoint", "old/checkpoint.pth",
               "--history_prefix", "old/history"]
    for key, value in WEIGHTS.items():
        command.extend(["--" + key, str(value)])
    return command


def option(command, flag):
    return command[command.index(flag) + 1]


class WorkflowTest(unittest.TestCase):
    def test_full_counts_and_topological_dependencies(self):
        tasks = exp033.build_tasks(manifest())
        self.assertEqual(len(tasks), 395)
        self.assertEqual(Counter(t["kind"] for t in tasks),
                         dict(sources=5, calibrate=5, reference=20, structure=50,
                              timing=60, tnp=100, train=75, attack=75, visualize=5))
        seen = set()
        for task in tasks:
            self.assertLessEqual(set(task["dependencies"]), seen)
            self.assertNotIn(task["task_id"], seen)
            seen.add(task["task_id"])
        self.assertEqual(Counter(t["seed"] for t in tasks), {seed: 79 for seed in SEEDS})
        self.assertEqual({t["method"] for t in tasks if t["kind"] == "structure"},
                         {"tr_dense", "tt_dense", "tt_time", "tucker", "svd"})
        self.assertTrue(all(t.get("exclusive") for t in tasks if t["kind"] == "timing"))
        self.assertEqual(len(exp033.build_tasks(manifest(smoke=True))), 34)

    def test_module_selection_contains_dependency_closure(self):
        tasks = exp033.build_tasks(manifest())
        expected = dict(structure=125, rank=30, loss=235, ablation=15, visualize=15)
        for group in GROUPS:
            with self.subTest(group=group):
                selected = exp033.select_tasks(tasks, [group])
                self.assertEqual(len(selected), expected[group])
                ids = {task["task_id"] for task in selected}
                self.assertTrue(all(set(t["dependencies"]) <= ids for t in selected))
                self.assertEqual(sum(t["kind"] == "sources" for t in selected), 5)
                if group == "visualize":
                    self.assertEqual(Counter(t["kind"] for t in selected),
                                     dict(sources=5, tnp=5, visualize=5))
        self.assertEqual(exp033.select_tasks(tasks, GROUPS), tasks)

    def test_compute_and_timing_queues_partition_tasks_without_readding_dependencies(self):
        for smoke, compute_count, timing_count in ((False, 335, 60), (True, 22, 12)):
            tasks = exp033.build_tasks(manifest(smoke=smoke))
            snapshot = copy.deepcopy(tasks)
            compute = exp033.select_queue(tasks, "compute")
            timing = exp033.select_queue(tasks, "timing")
            self.assertEqual(exp033.select_queue(tasks, "all"), tasks)
            self.assertEqual(len(compute), compute_count)
            self.assertEqual(len(timing), timing_count)
            self.assertTrue(all(t["kind"] != "timing" for t in compute))
            self.assertTrue(all(t["kind"] == "timing" for t in timing))
            compute_ids = {t["task_id"] for t in compute}
            timing_ids = {t["task_id"] for t in timing}
            self.assertFalse(compute_ids & timing_ids)
            self.assertEqual(compute_ids | timing_ids, {t["task_id"] for t in tasks})
            self.assertTrue(all(set(t["dependencies"]) <= compute_ids for t in timing))
            self.assertEqual(tasks, snapshot)
        selected = exp033.select_tasks(exp033.build_tasks(manifest()), ["structure"])
        self.assertEqual(len(exp033.select_queue(selected, "compute")), 65)
        self.assertEqual(len(exp033.select_queue(selected, "timing")), 60)
        with self.assertRaises(ValueError):
            exp033.select_queue([], "unknown")

    def test_timing_queue_reuses_completed_dependencies_and_rejects_incomplete_or_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [dict(task_id="sources_seed42", kind="sources", group="shared", dependencies=[]),
                     dict(task_id="calibrate_seed42", kind="calibrate", group="structure", dependencies=["sources_seed42"]),
                     dict(task_id="timing_seed42_svd_25", kind="timing", group="structure", dependencies=["calibrate_seed42"])]
            for task in tasks:
                task["output_path"] = str(root / "metrics" / f"{task['task_id']}.json")
            for task in tasks[:2]:
                write_json(task["output_path"], {"task": task["task_id"]})
                write_json(root / "status" / f"{task['task_id']}.json",
                           dict(status="completed", output_sha256=file_hash(task["output_path"])))

            def start_worker(command, **kwargs):
                self.assertEqual(command[-1], "timing_seed42_svd_25")
                write_json(tasks[-1]["output_path"], {"timing": "mocked"})
                return SimpleNamespace(pid=os.getpid() + 100000, poll=lambda: 0, wait=lambda: 0)

            with patch.object(exp033, "verify_plan", return_value=(dict(smoke=True), tasks)), \
                 patch.object(exp033.subprocess, "Popen", side_effect=start_worker) as spawn, \
                 patch.object(exp033.time, "sleep"):
                exp033.run(root, ["structure"], [], cpu=True, queue="timing")
                self.assertEqual(spawn.call_count, 1)
                self.assertEqual(read_json(root / "status" / "timing_seed42_svd_25.json")["status"], "completed")
                spawn.reset_mock()
                write_json(root / "status" / "calibrate_seed42.json", dict(status="failed"))
                with self.assertRaises((ValueError, RuntimeError)):
                    exp033.run(root, ["structure"], [], cpu=True, queue="timing")
                spawn.assert_not_called()
                write_json(root / "status" / "calibrate_seed42.json",
                           dict(status="completed", output_sha256="changed-hash"))
                with self.assertRaises((ValueError, RuntimeError)):
                    exp033.run(root, ["structure"], [], cpu=True, queue="timing")
                spawn.assert_not_called()

    def test_fifteen_variants_change_only_the_declared_weight(self):
        variants = loss_variants()
        self.assertEqual(len(variants), 16)
        self.assertEqual(len({item["variant"] for item in variants}), 16)
        self.assertEqual(variants[0]["weights"], WEIGHTS)
        counts = Counter()
        for variant in variants[1:]:
            changed = {key for key in WEIGHTS if variant["weights"][key] != WEIGHTS[key]}
            self.assertEqual(changed, {variant["scan_parameter"]})
            self.assertEqual(variant["weights"][variant["scan_parameter"]], variant["scan_value"])
            counts[variant["scan_parameter"]] += 1
        self.assertEqual(counts, {key: 3 for key in WEIGHTS})
        self.assertEqual(ATTACK_PROTOCOL,
                         dict(norm="Linf", eps=.03, steps=200, alpha=2 / 255, random_start=False))
        variants[0]["weights"]["clean_ce_weight"] = 99
        self.assertEqual(WEIGHTS["clean_ce_weight"], 1.)

    def test_training_clones_actual_source_command_and_preserves_settings(self):
        original = original_training_command()
        snapshot = copy.deepcopy(original)
        task = dict(seed=42, **loss_variants()[2])
        source = dict(status="completed", command=original)
        with patch("rpcf.exp033_common.source_paths", return_value={"training_status": "mock.json"}), \
             patch("rpcf.exp033_common.read_json", return_value=source):
            formal = training_arguments(manifest(), task, Path("new/attempt1"))
            smoke = training_arguments(manifest(smoke=True), task, Path("new/smoke1"))
        self.assertEqual(original, snapshot)
        for flag in ("--dataset", "--model", "--seed", "--fold", "--epsilon", "--cache_path",
                     "--checkpoint_path", "--epochs", "--batch_size", "--eval_batch_size",
                     "--online_at_batch_size", "--lr", "--consistancy_temperature"):
            self.assertEqual(option(formal, flag), option(original, flag))
        self.assertEqual(option(formal, "--output_checkpoint"), "new/attempt1/checkpoint.pth")
        self.assertEqual(option(formal, "--history_prefix"), "new/attempt1/history")
        for key, value in task["weights"].items():
            self.assertEqual(float(option(formal, "--" + key)), value)
        self.assertEqual(option(smoke, "--epochs"), "1")
        self.assertEqual(option(smoke, "--max_cache_batches"), "1")
        self.assertEqual(option(smoke, "--online_train_sample_num"), "2")
        self.assertEqual(option(smoke, "--seed"), "42")

    def test_training_rejects_missing_caf_flags_and_unresolved_command(self):
        task = dict(seed=42, **loss_variants()[1])
        for bad in (original_training_command() + ["--cache_tag", "__UNRESOLVED__"],
                    [v for v in original_training_command() if v != "--all_layers"]):
            with patch("rpcf.exp033_common.source_paths", return_value={"training_status": "mock.json"}), \
                 patch("rpcf.exp033_common.read_json", return_value=dict(status="completed", command=bad)):
                with self.assertRaises(ValueError):
                    training_arguments(manifest(), task, Path("isolated"))

    def test_run_id_and_frozen_manifests(self):
        for unsafe in ("", ".", "..", "../outside", "/absolute", "a/b", "bad name"):
            with self.subTest(run_id=unsafe), self.assertRaises(ValueError):
                exp033.create_plan(unsafe)
        with tempfile.TemporaryDirectory() as temporary:
            old_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                with patch.object(exp033, "science_hashes", return_value={"code.py": "abc"}):
                    root, first, tasks = exp033.create_plan("safe_033-test", smoke=True)
                    self.assertEqual(exp033.create_plan("safe_033-test", smoke=True)[1], first)
                    self.assertEqual(exp033.verify_plan(root), (first, tasks))
                    with self.assertRaises(ValueError):
                        exp033.create_plan("safe_033-test", smoke=False)
                    modified = read_json(root / "tasks.json")
                    modified[0]["seed"] = 99
                    write_json(root / "tasks.json", modified)
                    with self.assertRaisesRegex(ValueError, "Task manifest"):
                        exp033.verify_plan(root)
            finally:
                os.chdir(old_cwd)

    def test_cli_rejects_unsafe_run_id_before_reading_existing_plan(self):
        with patch("sys.argv", ["exp033", "dry-run", "--run-id", "../outside"]), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(exp033, "verify_plan", return_value=(dict(smoke=False), [])) as verify:
            with self.assertRaises((ValueError, SystemExit)):
                exp033.main()
            verify.assert_not_called()

    def test_resume_checks_completed_output_hash_and_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "metrics" / "sources_seed42.json"
            write_json(output, {"source": "original"})
            task = dict(task_id="sources_seed42", kind="sources", group="structure",
                        output_path=str(output), dependencies=[])
            write_json(root / "status" / "sources_seed42.json",
                       dict(status="completed", output_sha256=file_hash(output)))
            with patch.object(exp033, "verify_plan", return_value=(dict(smoke=True), [task])), \
                 patch.object(exp033.subprocess, "Popen") as spawn:
                exp033.run(root, ["structure"], [], cpu=True)
                spawn.assert_not_called()
                write_json(output, {"source": "tampered"})
                with self.assertRaisesRegex(ValueError, "Completed output missing/changed"):
                    exp033.run(root, ["structure"], [], cpu=True)
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
