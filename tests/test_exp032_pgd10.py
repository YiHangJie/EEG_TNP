"""PGD-10 与历史更新一致性、真实净化梯度及旧结果隔离检查。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from rpcf.evaluate_bpda_pgd import bpda_pgd_attack
from rpcf.exp032 import tasks_for
from rpcf.exp032_external_pgd10 import pgd10
from rpcf.revise_exp032_adaptive import revised_tasks, validated_nonadaptive


class PGD10RevisionTest(unittest.TestCase):
    def test_matches_historical_updates_without_rng_draws(self):
        model = nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., -2., 3.], [-1., 2., -3.]]))
            model.bias.zero_()
        clean = torch.tensor([[-2., .1, 2.], [-.2, 1.3, -.5]])
        labels = torch.tensor([0, 1])
        args = SimpleNamespace(batch_size=1, pgd_steps=10, pgd_alpha=.006, eps=.03)
        expected = bpda_pgd_attack(model, nn.Identity(), clean, labels, args, torch.device("cpu"))
        before = torch.get_rng_state().clone()
        inputs = []
        hook = model.register_forward_pre_hook(lambda module, values: inputs.append(values[0].detach().clone()))
        actual = pgd10(model, clean, labels, torch.device("cpu"))
        hook.remove()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(inputs), 20)
        self.assertTrue(torch.equal(inputs[0], clean[:1]))
        self.assertTrue(torch.equal(inputs[10], clean[1:]))
        self.assertLessEqual(float((actual-clean).abs().max()), .030001)
        self.assertLess(float(actual.min()), 0.)
        self.assertGreater(float(actual.max()), 1.)

    def test_differentiable_purifier_uses_exact_gradient(self):
        class Negate(nn.Module):
            def forward(self, x):
                return -x
        class Classifier(nn.Module):
            def forward(self, x):
                return torch.cat([x, -x], dim=1)
        clean = torch.tensor([[.1]])
        result = pgd10(nn.Sequential(Negate(), Classifier()), clean, torch.tensor([0]), "cpu")
        self.assertTrue(torch.allclose(result, clean + .03))

    def test_queue_changes_only_external_evaluation(self):
        original = tasks_for(SimpleNamespace(run_id="exp032_test", source_run="source", smoke=False))
        revised = revised_tasks(original, Path("isolated"))
        self.assertEqual(len(revised), 2685)
        adaptive = 0
        for old, new in zip(original, revised):
            if old.kind != "external_eval":
                self.assertEqual(old, new)
            else:
                self.assertEqual(old.dependencies, new.dependencies)
                self.assertEqual(old.task_id, new.task_id)
                self.assertNotEqual(old.output_path, new.output_path)
                self.assertIn("rpcf.exp032_external_pgd10", new.command)
                adaptive += new.attack == "pgd"
        self.assertEqual(adaptive, 180)
        self.assertEqual(sum(not(t.kind == "tnp" and t.method == "clean") for t in revised), 2235)

    def test_reuse_drops_stronger_adaptive_and_rejects_wrong_samples(self):
        task = next(t for t in tasks_for(SimpleNamespace(run_id="exp032_test", source_run="source", smoke=True))
                    if t.kind == "external_eval" and t.attack == "fgsm")
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            row = dict(dataset=task.dataset, model=task.model, seed=task.seed, fold=0,
                       method=f"clean_{task.method}", attack="fgsm", scope="S", sample_num=2,
                       evaluation="nonadaptive_purification", standard_accuracy=.5, robust_accuracy=1.,
                       clean_predictions=[0, 0], adv_predictions=[0, 1])
            payload = dict(source_indices=[2, 4], labels=[0, 1], rows=[row, dict(row, evaluation="adaptive_exact_gradient")])
            path.write_text(json.dumps(payload))
            task = replace(task, output_path=str(path))
            result = validated_nonadaptive(payload, task, payload, 2)
            self.assertEqual(result["rows"], [row])
            with self.assertRaises(ValueError):
                validated_nonadaptive(payload, task, dict(payload, source_indices=[4, 2]), 2)


if __name__ == "__main__":
    unittest.main()
