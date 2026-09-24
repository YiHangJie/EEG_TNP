"""EXP-033 worker 的随机流恢复、来源拒绝及可视化案例续跑检查。"""

import contextlib
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rpcf import exp033_worker as worker
from rpcf.exp033_common import ATTACK_PROTOCOL, file_hash, load_attack, write_json
from rpcf.exp032_common import load_payload as real_load_payload
from rpcf.exp032_tnp import rng_state, restore_rng
from utils.reproducibility import seed_everything


def draws():
    return [random.random(), float(np.random.rand()), *torch.rand(3).tolist()]


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1))
        with torch.no_grad():
            self.weight.fill_(1)

    def forward(self, inputs):
        values = inputs.flatten(1).mean(1) * self.weight
        return torch.stack((values, -values), 1)


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.saved_rng = rng_state()

    def tearDown(self):
        restore_rng(self.saved_rng)

    def test_resume_restores_all_cpu_rng_after_model_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.pth"
            seed_everything(42)
            prefix = draws()
            worker.save_state(path, "same-source", {"draws": prefix})
            expected = [draws() for _ in range(3)]
            seed_everything(999)
            TinyClassifier()
            progress = worker.load_state(path, "same-source", {})
            self.assertEqual(progress, {"draws": prefix})
            self.assertEqual([draws() for _ in range(3)], expected)
            before = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, "source/config"):
                worker.load_state(path, "other-source", {})
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            default = {"nested": []}
            loaded = worker.load_state(Path(temporary) / "missing.pth", "same", default)
            loaded["nested"].append(1)
            self.assertEqual(default, {"nested": []})

    def test_capture_selects_distinct_minimal_indices_independent_of_processing_order(self):
        records = [(10, 0, 0, [1]), (2, 1, 1, [0]), (4, 1, 0, [1]),
                   (1, 0, 0, [1]), (3, 1, 1, [0]), (9, 0, 0, [1])]
        outputs = []
        for sequence in (records, list(reversed(records))):
            indices = [record[0] for record in sequence]
            ranks = {r: dict(cp=[record[1] for record in sequence], ap=[record[2] for record in sequence])
                     for r in (25, 30)}
            selected = worker.select_cases(indices, [0] * 6, [0] * 6, [1] * 6, ranks,
                                           {"external": [record[3][0] for record in sequence]})
            self.assertEqual(selected, dict(success=1, failure=2, clean_damage=3, disagreement=4))
            self.assertEqual(len(selected), len(set(selected.values())))
            outputs.append(selected)
        self.assertEqual(outputs[0], outputs[1])
        # 分歧包含 rank30，以及两个方法均错误但类别不同的样本。
        selected = worker.select_cases([5], [0], [1], [1],
                                       {25: dict(cp=[1], ap=[1]), 30: dict(cp=[1], ap=[2])}, {})
        self.assertEqual(selected, {"disagreement": 5})

    def test_timing_rejects_foreign_process_and_requires_one_gpu(self):
        with patch.object(worker.subprocess, "check_output") as query:
            worker.assert_exclusive(torch.device("cpu"))
            query.assert_not_called()
        for visible in ("", "0,1"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), self.assertRaises(RuntimeError):
                worker.assert_exclusive(torch.device("cuda:0"))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), \
             patch.object(worker.subprocess, "check_output", return_value=f"{os.getpid()}\n") as query:
            worker.assert_exclusive(torch.device("cuda:0"))
            self.assertEqual(query.call_args.args[0][2], "3")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), \
             patch.object(worker.subprocess, "check_output", return_value=f"{os.getpid()}\n{os.getpid() + 1}\n"), \
             self.assertRaisesRegex(RuntimeError, "shared"):
            worker.assert_exclusive(torch.device("cuda:0"))

    def test_frozen_sources_reject_changed_hash_and_clean_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "source.pth"
            artifact.write_bytes(b"original")
            stat = artifact.stat()
            clean = torch.zeros(2, 1, 2)
            snapshot = dict(files={"source": dict(path=str(artifact), size=stat.st_size,
                                                mtime_ns=stat.st_mtime_ns, sha256=file_hash(artifact))},
                            source_indices=[7, 3], labels=[0, 1], clean_sha256=worker.tensor_hash(clean))
            write_json(root / "metrics" / "sources_seed42.json", snapshot)
            ctx = SimpleNamespace(indices=[7, 3], labels=torch.tensor([0, 1]), clean=clean)
            worker.frozen_sources(root, {"seed": 42}, ctx)
            with self.assertRaisesRegex(ValueError, "Canonical"):
                worker.frozen_sources(root, {"seed": 42}, SimpleNamespace(**{**vars(ctx), "clean": clean + 1}))
            artifact.write_bytes(b"different source")
            with self.assertRaisesRegex(ValueError, "Frozen source"):
                worker.frozen_sources(root, {"seed": 42}, ctx)

    def test_attack_protocol_and_recorded_checkpoint_hash_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pth"
            checkpoint.write_bytes(b"weights")
            clean = torch.zeros(2, 1, 2)
            labels = torch.tensor([0, 1])
            ctx = SimpleNamespace(args=SimpleNamespace(seed=42), clean=clean, labels=labels, indices=[7, 3])
            payload = dict(clean=clean, adversarial=clean + .01, labels=labels, source_indices=[7, 3],
                           meta=dict(dataset="thubenchmark", model="eegnet", seed=42, fold=0, attack="pgd",
                                     checkpoint_path=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                                     attack_protocol=dict(ATTACK_PROTOCOL)))
            with patch("rpcf.exp032_common.load_payload", return_value=payload):
                attack, _ = load_attack("mock.pth", ctx, checkpoint)
                self.assertTrue(torch.equal(attack, clean + .01))
                payload["meta"]["attack_protocol"]["steps"] = 10
                with self.assertRaisesRegex(ValueError, "protocol"):
                    load_attack("mock.pth", ctx, checkpoint)
                payload["meta"]["attack_protocol"]["steps"] = 200
                payload["meta"]["checkpoint_sha256"] = "wrong-hash"
                with self.assertRaises(ValueError):
                    load_attack("mock.pth", ctx, checkpoint)
                payload["meta"]["checkpoint_sha256"] = file_hash(checkpoint)
                for invalid in (torch.full_like(clean, float("nan")), clean + .031):
                    payload["adversarial"] = invalid
                    with self.assertRaisesRegex(ValueError, "out-of-budget|Non-finite"):
                        load_attack("mock.pth", ctx, checkpoint)

    def test_smoke_configs_keep_positive_effective_warmup_per_stage(self):
        import yaml
        from rpcf.exp033_common import config_for_rank
        with tempfile.TemporaryDirectory() as temporary:
            for rank in (25, 30):
                original = Path(f"configs/thubenchmark/PTR3d_8_2048_rank{rank}_3d_interpolate.yaml")
                before = file_hash(original)
                generated = Path(config_for_rank(dict(smoke=True, run_dir=temporary), rank))
                values = yaml.safe_load(generated.read_text())
                self.assertEqual(values["num_iterations"], 40)
                self.assertEqual(values["iterations_for_upsampling"], [10, 20, 30])
                boundaries = [0, *values["iterations_for_upsampling"], values["num_iterations"]]
                stage_steps = [end - start for start, end in zip(boundaries, boundaries[1:])]
                self.assertEqual(len(stage_steps), values["stage"])
                # PTR_3d.train 使用每阶段步数的 10%，不读取 config.warmup_steps。
                self.assertTrue(all(steps >= 10 for steps in stage_steps))
                self.assertTrue(all(int(steps * .1) >= 1 for steps in stage_steps))
                self.assertTrue(generated.is_relative_to(Path(temporary)))
                self.assertEqual(file_hash(original), before)

    def test_reconstruct_transform_matches_original_interpolation_bitwise(self):
        from purify import interpolate, inv_interpolate
        sample = torch.randn(1, 64, 1500, generator=torch.Generator().manual_seed(42))
        args = SimpleNamespace(dataset="thubenchmark",
                               config="configs/thubenchmark/PTR3d_8_2048_rank25_3d_interpolate.yaml")
        expected_pre = interpolate(args, sample, 250)
        expected_output = inv_interpolate(args, expected_pre, sample.shape[-2:], "3d_interpolate")
        seen = []

        def identity_decomposition(tensor, spec):
            seen.append(tensor.clone())
            return tensor.clone(), dict(actual_parameters=1, actual_rank=1)

        with patch("rpcf.exp033_structures.decompose", side_effect=identity_decomposition):
            output, diagnostic = worker.reconstruct(dict(smoke=False), sample,
                                                     dict(method="svd", rank=1), torch.device("cpu"))
        self.assertEqual(len(seen), 1)
        self.assertEqual(tuple(seen[0].shape), (10, 11, 2048))
        self.assertTrue(torch.equal(seen[0], expected_pre))
        self.assertTrue(torch.equal(output, expected_output))
        self.assertEqual(output.shape, sample.shape)
        self.assertGreaterEqual(diagnostic["total_seconds"], diagnostic["decomposition_seconds"])

    def test_reconstruct_reads_all_yaml_before_timing_starts(self):
        import yaml
        sample = torch.randn(1, 64, 1500, generator=torch.Generator().manual_seed(43))
        events = []
        safe_load, counter = yaml.safe_load, worker.time.perf_counter

        def record_yaml(*args, **kwargs):
            events.append("yaml")
            return safe_load(*args, **kwargs)

        def record_clock():
            events.append("clock")
            return counter()

        with tempfile.TemporaryDirectory() as temporary, \
             patch("yaml.safe_load", side_effect=record_yaml), \
             patch.object(worker.time, "perf_counter", side_effect=record_clock), \
             patch("rpcf.exp033_structures.decompose", side_effect=lambda tensor, spec: (tensor, {})):
            worker.reconstruct(dict(smoke=True, run_dir=temporary), sample,
                               dict(method="svd", rank=1), torch.device("cpu"))
        self.assertGreaterEqual(events.count("yaml"), 2)
        self.assertIn("clock", events)
        first_clock = events.index("clock")
        self.assertTrue(all(event == "yaml" for event in events[:first_clock]))
        self.assertNotIn("yaml", events[first_clock:])

    def _tnp_fixture(self, root, capture=True, interrupt=False):
        """用随机微扰替代昂贵 TN 优化，保留真实逐样本保存/恢复和 case 保存流程。"""
        root.mkdir(parents=True, exist_ok=True)
        checkpoint, attack_file = root / "source_cp.pth", root / "source_attack.pth"
        checkpoint.write_bytes(b"checkpoint")
        attack_file.write_bytes(b"attack")
        clean = torch.ones(9, 1, 2, 3)
        ctx = SimpleNamespace(clean=clean, labels=torch.zeros(9, dtype=torch.long),
                              indices=[8, 1, 7, 2, 6, 3, 5, 4, 9],
                              args=SimpleNamespace(dataset="thubenchmark", model="eegnet", seed=42))
        paths = dict(checkpoint_clean=str(checkpoint), attack_clean=str(attack_file), canonical_clean_tnp="canonical")
        canonical = dict(source_indices=ctx.indices, labels=ctx.labels, clean=clean,
                         ranks=[25, 30], clean_pur_by_rank=torch.stack((clean, clean), 1))
        task = dict(task_id="tnp_seed42_clean", group="ablation", seed=42, method="clean", ranks=[25, 30], capture=capture)
        manifest = dict(smoke=True, run_dir=str(root))

        def purify(args, index, sample, *unused, **kwargs):
            if interrupt and args.rank == 30 and index == 17:
                raise RuntimeError("simulated interruption")
            noise = (random.random() + np.random.rand() + torch.rand(()).item()) * .001
            return -sample + noise, 0.

        def external(*args):
            torch.rand(20)  # 模拟新增外部净化器的随机初始化。
            return {"external": nn.Identity()}

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(worker, "source_paths", return_value=paths))
            stack.enter_context(patch.object(worker, "load_model", side_effect=lambda *args: TinyClassifier()))
            stack.enter_context(patch.object(worker, "load_attack", return_value=(-clean, {})))
            stack.enter_context(patch.object(worker, "frozen_sources", return_value={"files": {}}))
            stack.enter_context(patch.object(worker, "purifier_args", side_effect=lambda m, r: SimpleNamespace(rank=r, seed=42)))
            stack.enter_context(patch.object(worker, "external_models", side_effect=external))
            stack.enter_context(patch("rpcf.exp032_common.load_payload", side_effect=lambda p: canonical if str(p) == "canonical" else real_load_payload(p)))
            stack.enter_context(patch("purify.purify", side_effect=purify))
            result = worker.tnp(manifest, task, ctx, root, torch.device("cpu"))
        return result, real_load_payload(result["capture_path"]) if capture else None

    def test_rank30_resume_preserves_previously_captured_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_everything(42)
            expected, expected_capture = self._tnp_fixture(root / "continuous")
            seed_everything(42)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self._tnp_fixture(root / "resumed", interrupt=True)
            seed_everything(42)
            actual, actual_capture = self._tnp_fixture(root / "resumed")
            self.assertEqual(actual["rows"], expected["rows"])
            self.assertEqual(actual["sample_diagnostics"], expected["sample_diagnostics"])
            self.assertEqual(actual_capture["categories"], expected_capture["categories"])
            for source, signals in expected_capture["cases"].items():
                self.assertEqual(set(actual_capture["cases"][source]), set(signals))
                for key, tensor in signals.items():
                    self.assertTrue(torch.equal(actual_capture["cases"][source][key], tensor))

    def test_capture_prediction_initialization_does_not_change_purification_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_everything(42)
            plain, _ = self._tnp_fixture(root / "plain", capture=False)
            seed_everything(42)
            captured, _ = self._tnp_fixture(root / "captured", capture=True)
            self.assertEqual(plain["sample_diagnostics"], captured["sample_diagnostics"])


if __name__ == "__main__":
    unittest.main()
