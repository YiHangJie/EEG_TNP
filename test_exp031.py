"""EXP-031 协议与核心扩展的轻量回归测试。"""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

from models.eegnet_ea_forward import (
    EA_FORWARD_MODEL_CHOICES,
    SubjectEAClassifier,
    build_subject_ea_model,
)
from rpcf.core import build_model, compute_rank_weights
from rpcf.exp031 import (
    EXPECTED_FULL_COUNTS,
    Task,
    actual_cache_attack_batch,
    actual_rpcf_batch,
    build_tasks,
    lower_cache_attack_batch_after_oom,
    lower_rpcf_batch_after_oom,
    parse_reserved_gpu_processes,
    reservation_active,
    render_command,
    select_tasks,
    task_complete,
    validate_plan,
)
from rpcf.exp031_artifacts import (
    atomic_torch_save,
    compact_exp031_attack_artifact,
)
from rpcf.finetune import resolve_layer_configuration
from rpcf.generate_cache import load_shared_clean_cache
from rpcf.evaluate_purification import load_shared_clean_payload
from rpcf.summarize_exp031 import audit_task_artifacts, validate_attack


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x.flatten(1).mean(dim=1, keepdim=True) * self.weight


class Exp031Tests(unittest.TestCase):
    def test_full_plan_exact_counts_and_dependencies(self):
        tasks = build_tasks("unit_full")
        counts = validate_plan(tasks)
        for kind, expected in EXPECTED_FULL_COUNTS.items():
            self.assertEqual(counts[kind], expected)
        ids = {task.task_id for task in tasks}
        self.assertTrue(all(dep in ids for task in tasks for dep in task.dependencies))

    def test_plan_rejects_missing_dependency(self):
        tasks = build_tasks("unit_missing")
        broken = list(tasks)
        first = broken[0]
        broken[0] = type(first)(**{
            **first.__dict__, "dependencies": ("does_not_exist",),
        })
        with self.assertRaisesRegex(ValueError, "Unknown dependencies"):
            validate_plan(broken)

    def test_thu_eegnet_closure_scope_exact_counts(self):
        tasks = build_tasks("unit_scope")
        selected = select_tasks(tasks, task_scope="thu_eegnet_closure")
        counts = {
            kind: sum(task.kind == kind for task in selected)
            for kind in ("attack", "tnp", "bpda")
        }
        self.assertEqual(counts, {"attack": 100, "tnp": 40, "bpda": 10})
        self.assertEqual(len(selected), 150)
        self.assertTrue(all(task.dataset == "thubenchmark" for task in selected))
        self.assertTrue(all(task.model == "eegnet" for task in selected))
        self.assertTrue(all(task.stage >= 4 for task in selected))

    def test_reserved_gpu_process_parser_and_liveness(self):
        current_pid = __import__("os").getpid()
        stat = Path(f"/proc/{current_pid}/stat").read_text(encoding="utf-8")
        start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
        reservations = parse_reserved_gpu_processes(
            f"0:{current_pid}:{start_ticks},5:999999999:1"
        )
        self.assertTrue(reservation_active(reservations[0]))
        self.assertFalse(reservation_active(reservations[5]))
        with self.assertRaisesRegex(ValueError, "Duplicate reserved GPU"):
            parse_reserved_gpu_processes("0:1:1,0:2:2")

    def test_static_six_rank_weights_are_uniform(self):
        ranks = [15, 20, 25, 30, 35, 40]
        for epoch in (0, 50, 99):
            weights = compute_rank_weights(ranks, epoch, 100, static=True)
            self.assertTrue(torch.allclose(weights, torch.full((6,), 1 / 6)))

    def test_all_layers_does_not_require_sensitivity(self):
        info = {"chunk_size": 256, "num_electrodes": 22, "num_classes": 4}
        model = build_model("eegnet", "bciciv2a", info)
        args = Namespace(
            all_layers=True, selected_layers_override=None, sensitivity_path=None,
            model="eegnet", dataset="bciciv2a", fold=0, seed=42, epsilon=0.03,
        )
        artifact, layers = resolve_layer_configuration(
            args, {"ranks": [15, 20, 25, 30, 35, 40]}, model
        )
        self.assertEqual(artifact["kind"], "rpcf_all_layers")
        self.assertEqual(artifact["selected_param_ratio"], 1.0)
        self.assertEqual(layers, ["block1", "block2", "lin"])

    def test_subject_ea_gradient_and_subject_lookup(self):
        matrices = torch.stack([
            torch.eye(2), torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ])
        model = SubjectEAClassifier(matrices, TinyBackbone(), num_electrodes=2)
        x = torch.tensor([[[[1.0, 2.0, 4.0], [9.0, 3.0, 0.0]]]], requires_grad=True)
        aligned0 = model.apply_ea(x, torch.tensor([0]))
        aligned1 = model.apply_ea(x, torch.tensor([1]))
        self.assertFalse(torch.equal(aligned0, aligned1))
        model(x, torch.tensor([1])).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.backbone.weight.grad)
        with self.assertRaises(ValueError):
            model(x.detach(), torch.tensor([2]))

    def test_six_ea_forward_factories_and_gradients(self):
        expected = {
            "eegnet_ea_forward", "deepconvnet_ea_forward", "tsception_ea_forward",
            "atcnet_ea_forward", "conformer_ea_forward", "tcnet_ea_forward",
        }
        self.assertEqual(set(EA_FORWARD_MODEL_CHOICES), expected)
        info = {
            "chunk_size": 1750, "num_electrodes": 22,
            "num_classes": 4, "sampling_rate": 250,
        }
        matrices = torch.stack([torch.eye(22), torch.eye(22)])
        x = torch.randn(1, 1, 22, 1750, requires_grad=True)
        for name in EA_FORWARD_MODEL_CHOICES:
            model = build_subject_ea_model(name, "bciciv2a", info, matrices)
            model.eval()
            output = model(x, torch.tensor([1]))
            self.assertEqual(tuple(output.shape), (1, 4), name)
            output.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()), name)
            model.zero_grad(set_to_none=True)
            x.grad = None

    def test_summary_missing_and_batch_mismatch_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = run_dir / "checkpoint.pth"
            row = {
                "task_id": "train_standard_thubenchmark_eegnet_seed42_madry",
                "kind": "train_standard", "dataset": "thubenchmark",
                "model": "eegnet", "output_path": str(output),
            }
            errors = audit_task_artifacts([row], run_dir)
            self.assertTrue(any("missing status" in error for error in errors))
            self.assertTrue(any("missing output" in error for error in errors))

            (run_dir / "status").mkdir()
            output.write_bytes(b"checkpoint")
            (run_dir / "actual_batch_sizes.json").write_text(
                '{"thubenchmark_eegnet": 64}\n', encoding="utf-8"
            )
            (run_dir / "status" / f"{row['task_id']}.json").write_text(
                '{"status": "completed", "actual_batch_size": 128}\n', encoding="utf-8"
            )
            errors = audit_task_artifacts([row], run_dir)
            self.assertEqual(errors, [f"actual batch manifest mismatch: {row['task_id']}"])

    def test_rpcf_cache_attack_batch_oom_fallback_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = run_dir / "cache.pth"
            task = Task(
                task_id="rpcf_cache_thubenchmark_eegnet_seed46_madry",
                stage=2,
                kind="rpcf_cache",
                dataset="thubenchmark",
                model="eegnet",
                seed=46,
                method="madry",
                output_path=str(output),
                command=("python", "--attack_batch_size", "__CACHE_ATTACK_BATCH_SIZE__"),
            )
            manifest = run_dir / "actual_cache_attack_batch_sizes.json"
            manifest.write_text('{"thubenchmark_eegnet": 32}\n', encoding="utf-8")
            log_path = run_dir / "cache.log"
            log_path.write_text(
                "torch.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8"
            )

            self.assertEqual(
                lower_cache_attack_batch_after_oom(run_dir, task, log_path), 16
            )
            self.assertEqual(
                lower_cache_attack_batch_after_oom(
                    run_dir, task, log_path, used_batch=32
                ),
                16,
            )
            self.assertEqual(actual_cache_attack_batch(run_dir, task), 16)
            self.assertEqual(render_command(task, run_dir)[-1], "16")

            (run_dir / "status").mkdir()
            output.write_bytes(b"cache")
            status_path = run_dir / "status" / f"{task.task_id}.json"
            status_path.write_text(
                '{"status": "completed", "actual_cache_attack_batch_size": 32}\n',
                encoding="utf-8",
            )
            self.assertFalse(task_complete(run_dir, task))
            row = {
                "task_id": task.task_id,
                "kind": task.kind,
                "dataset": task.dataset,
                "model": task.model,
                "output_path": task.output_path,
            }
            self.assertEqual(
                audit_task_artifacts([row], run_dir),
                [f"cache attack batch manifest mismatch: {task.task_id}"],
            )

    def test_rpcf_batch_fallback_is_concurrency_safe_and_legacy_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = run_dir / "rpcf.pth"
            task = Task(
                task_id="rpcf_at_thubenchmark_deepconvnet_seed46_rpcf_at",
                stage=3, kind="rpcf_at", dataset="thubenchmark",
                model="deepconvnet", seed=46, method="rpcf_at",
                output_path=str(output),
                command=(
                    "python", "--batch_size", "__RPCF_BATCH_SIZE__",
                    "--eval_batch_size", "__RPCF_EVAL_BATCH_SIZE__",
                ),
            )
            (run_dir / "actual_batch_sizes.json").write_text(
                '{"thubenchmark_deepconvnet": 128}\n', encoding="utf-8"
            )
            rpcf_manifest = run_dir / "actual_rpcf_batch_sizes.json"
            rpcf_manifest.write_text(
                '{"thubenchmark_deepconvnet": 64}\n', encoding="utf-8"
            )
            (run_dir / "status").mkdir()
            output.write_bytes(b"checkpoint")
            status_path = run_dir / "status" / f"{task.task_id}.json"
            status_path.write_text(
                '{"status":"completed","actual_batch_size":128,'
                '"command":["python","--batch_size","64",'
                '"--eval_batch_size","128"]}\n',
                encoding="utf-8",
            )
            self.assertTrue(task_complete(run_dir, task))
            row = {
                "task_id": task.task_id, "kind": task.kind,
                "dataset": task.dataset, "model": task.model,
                "output_path": task.output_path,
            }
            self.assertEqual(audit_task_artifacts([row], run_dir), [])
            self.assertEqual(actual_rpcf_batch(run_dir, task), 64)
            self.assertEqual(
                render_command(task, run_dir)[-3:],
                ["64", "--eval_batch_size", "128"],
            )

            log_path = run_dir / "rpcf.log"
            log_path.write_text("CUDA out of memory\n", encoding="utf-8")
            self.assertEqual(
                lower_rpcf_batch_after_oom(run_dir, task, log_path, used_batch=64), 32
            )
            self.assertEqual(
                lower_rpcf_batch_after_oom(run_dir, task, log_path, used_batch=64), 32
            )
            self.assertEqual(actual_rpcf_batch(run_dir, task), 32)
            self.assertFalse(task_complete(run_dir, task))
            self.assertEqual(
                audit_task_artifacts([row], run_dir),
                [f"RPCF batch manifest mismatch: {task.task_id}"],
            )

    def test_oom_detection_ignores_previous_attempt_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            task = Task(
                task_id="rpcf_at_thubenchmark_eegnet_seed42_rpcf_at",
                stage=3, kind="rpcf_at", dataset="thubenchmark", model="eegnet",
            )
            (run_dir / "actual_rpcf_batch_sizes.json").write_text(
                '{"thubenchmark_eegnet": 64}\n', encoding="utf-8"
            )
            log_path = run_dir / "rpcf.log"
            log_path.write_text("old CUDA out of memory\n", encoding="utf-8")
            current_offset = log_path.stat().st_size
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("current ValueError\n")
            self.assertIsNone(lower_rpcf_batch_after_oom(
                run_dir, task, log_path, used_batch=64,
                log_start_offset=current_offset,
            ))
            current_offset = log_path.stat().st_size
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("current CUDA out of memory\n")
            self.assertEqual(lower_rpcf_batch_after_oom(
                run_dir, task, log_path, used_batch=64,
                log_start_offset=current_offset,
            ), 32)

    def test_compact_attack_artifact_keeps_full_test_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ad_data/exp031/unit/attack.pth"
            clean = torch.randn(840, 1, 2, 3)
            labels = torch.arange(840) % 4
            stored, audit = compact_exp031_attack_artifact(
                clean, clean + 0.01, labels, list(range(840)), output, 43, 0
            )
            payload = {
                "clean": stored[0], "adversarial": stored[1],
                "labels": stored[2], "source_indices": stored[3],
                "meta": {
                    "dataset": "thubenchmark", "model": "eegnet", "fold": 0,
                    "seed": 43, "attack": "fgsm", "eps": 0.03,
                    "selection_strategy": "full_test_split",
                    "actual_attack_batch_size": 16, "clean_accuracy": 0.5,
                    "adv_accuracy": 0.4, "attack_l2_mean": 0.01,
                    "attack_mse": 0.0001,
                    "attack_protocol": {"norm": "Linf", "eps": 0.03, "steps": 1},
                    **audit,
                },
            }
            validate_attack({
                "dataset": "thubenchmark", "model": "eegnet", "seed": 43,
                "method": "madry", "attack": "fgsm",
            }, payload)
            atomic_torch_save(payload, output)
            loaded = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(len(loaded["source_indices"]), 512)
            self.assertEqual(loaded["meta"]["evaluation_sample_num"], 840)
            self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))

    @staticmethod
    def shared_train_payload(x, labels, indices):
        ranks = [15, 20, 25, 30, 35, 40]
        return {
            "x": x, "x_adv": x.clone(),
            "x_pur_by_rank": x[:, None].repeat(1, len(ranks), 1, 1, 1),
            "x_adv_pur_by_rank": x[:, None].repeat(1, len(ranks), 1, 1, 1),
            "labels": labels, "source_indices": indices, "ranks": ranks,
            "meta": {
                "kind": "rpcf_train_cache",
                "dataset": "thubenchmark", "model": "eegnet", "fold": 0,
                "seed": 42, "eps": 0.03, "ranks": ranks, "source_split": "train",
            },
        }

    def test_shared_clean_cache_identity_checks(self):
        x = torch.randn(2, 1, 3, 8)
        labels = torch.tensor([0, 1])
        indices = [4, 9]
        ranks = [15, 20, 25, 30, 35, 40]
        args = Namespace(dataset="thubenchmark", fold=0, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pth"
            torch.save(self.shared_train_payload(x, labels, indices), path)
            shared = load_shared_clean_cache(path, args, ranks, x, labels, indices)
            self.assertEqual(shared["source_indices"], indices)
            with self.assertRaisesRegex(ValueError, "source_indices"):
                load_shared_clean_cache(path, args, ranks, x, labels, [9, 4])

    def test_shared_test_payload_identity_checks(self):
        clean = torch.randn(2, 1, 3, 8)
        labels = torch.tensor([0, 1])
        indices = [4, 9]
        ranks = [25, 30]
        payload = {
            "clean": clean, "clean_pur_by_rank": clean[:, None].repeat(1, 2, 1, 1, 1),
            "labels": labels, "source_indices": indices, "ranks": ranks,
            "meta": {"dataset": "thubenchmark", "fold": 0, "seed": 42, "eps": 0.03},
        }
        args = Namespace(dataset="thubenchmark", fold=0, seed=42, eps=0.03)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared_eval.pth"
            torch.save(payload, path)
            result = load_shared_clean_payload(path, args, ranks, clean, labels, indices)
            self.assertEqual(tuple(result.shape[:2]), (2, 2))
            with self.assertRaisesRegex(ValueError, "labels"):
                load_shared_clean_payload(path, args, ranks, clean, labels.flip(0), indices)


if __name__ == "__main__":
    unittest.main()
