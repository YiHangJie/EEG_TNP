import unittest

import numpy as np
import torch
from torch import nn

from rpcf.compare_exp018 import (
    validate_purification_payload,
    validate_shared_subset,
)
from rpcf.compare_exp018_five_methods import (
    METHOD_SPECS,
    build_five_method_rows,
    build_wide_rows,
    render_markdown,
    validate_shared_protocol,
)
from rpcf.core import (
    ATTACK_CLASSES,
    build_attack,
    checkpoint_is_better,
    compute_interlayer_sensitivity,
    compute_rank_weights,
    configure_trainable_layers,
    normalized_feature_shift,
    select_sensitive_layers,
    set_frozen_batchnorm_eval,
    stable_subset_indices,
    validate_rpcf_cache,
)
from utils.reproducibility import seed_everything
from rpcf.finetune import kl_to_clean_teacher, rank_weighted_kl


class TinyRPCFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.block2 = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.head = nn.Linear(2, 2)

    def forward(self, x):
        return self.head(self.block2(self.block1(x)))


class RPCFCoreTest(unittest.TestCase):
    def _fair_payload(self, method="at"):
        return {
            "clean": torch.zeros(2, 1, 3, 4),
            "adversarial": torch.ones(2, 1, 3, 4),
            "clean_pur_by_rank": torch.zeros(2, 2, 1, 3, 4),
            "adv_pur_by_rank": torch.ones(2, 2, 1, 3, 4),
            "labels": torch.tensor([0, 1]),
            "source_indices": [4, 7],
            "ranks": [25, 30],
            "metrics": [],
            "meta": {
                "kind": "rpcf_purification_eval",
                "dataset": "toy",
                "model": "toy",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "sample_num": 2,
                "selection_seed": 42,
                "checkpoint_path": f"{method}.pth",
                "attack_meta": {
                    "attack": "autoattack",
                    "attack_seed": 42,
                    "checkpoint_path": f"{method}.pth",
                },
            },
        }

    def _five_method_summary(self, variant="base"):
        rpcf_meta = {
            "selected_layers": ["block2", "block1"],
            "selected_param_ratio": 0.1,
            "best_epoch": 100,
        }
        if variant == "all_layers":
            rpcf_meta.update({"all_layers": True})
        elif variant == "uniform":
            rpcf_meta.update(
                {"all_layers": False, "static_rank_weights": True}
            )
        rows = []
        full_rows = []
        for method_index, method in enumerate(
            ("at_tnp", "consistancy", "rpcf")
        ):
            full_rows.append(
                {
                    "method": method,
                    "sample_num": 4,
                    "clean_accuracy": 0.9 + method_index * 0.01,
                    "adv_accuracy": 0.7 + method_index * 0.01,
                    "attack_mse": 0.001 + method_index * 0.001,
                }
            )
            for rank in (25, 30):
                rows.append(
                    {
                        "method": method,
                        "rank": rank,
                        "sample_num": 2,
                        "clean_accuracy": 0.8 + method_index * 0.01,
                        "adv_accuracy": 0.6 + method_index * 0.01,
                        "attack_mse": 0.01 + method_index * 0.01,
                        "purified_clean_accuracy": (
                            0.85 + method_index * 0.01 + rank / 10000
                        ),
                        "purified_adv_accuracy": (
                            0.75 + method_index * 0.01 + rank / 10000
                        ),
                        "mean_clean_mse": 0.1,
                        "mean_adv_mse": 0.2,
                    }
                )
        return {
            "kind": "exp018_seed42_fair_comparison",
            "dataset": "toy",
            "model": "toy",
            "seed": 42,
            "fold": 0,
            "eps": 0.03,
            "attack": "autoattack",
            "ranks": [25, 30],
            "sample_num": 2,
            "source_indices": [4, 7],
            "rows": rows,
            "full_test_attack": full_rows,
            "rpcf": rpcf_meta,
        }

    def test_cache_validation(self):
        payload = {
            "x": torch.zeros(2, 1, 3, 4),
            "x_adv": torch.ones(2, 1, 3, 4),
            "x_pur_by_rank": torch.zeros(2, 2, 1, 3, 4),
            "x_adv_pur_by_rank": torch.ones(2, 2, 1, 3, 4),
            "labels": torch.tensor([0, 1]),
            "source_indices": [4, 7],
            "ranks": [15, 40],
            "meta": {
                "kind": "rpcf_train_cache",
                "dataset": "toy",
                "model": "toy",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "ranks": [15, 40],
            },
        }
        normalized = validate_rpcf_cache(
            payload,
            expected={"dataset": "toy", "fold": 0, "eps": 0.03},
        )
        self.assertEqual(tuple(normalized["x_pur_by_rank"].shape), (2, 2, 1, 3, 4))

        invalid = dict(payload)
        invalid["source_indices"] = [4, 4]
        with self.assertRaises(ValueError):
            validate_rpcf_cache(invalid)

    def test_clean_and_advpur_sensitivity_use_clean_anchor(self):
        clean_feature = torch.tensor([[3.0, 4.0]])
        clean_pur_feature = torch.tensor([[0.0, 4.0]])
        adv_pur_feature = torch.tensor([[6.0, 8.0]])
        clean_shift = normalized_feature_shift(clean_feature, clean_pur_feature)
        advpur_shift = normalized_feature_shift(clean_feature, adv_pur_feature)
        self.assertAlmostEqual(clean_shift.item(), 3.0 / 5.0)
        self.assertAlmostEqual(advpur_shift.item(), 5.0 / 5.0)

    def test_interlayer_sensitivity_uses_input_then_previous_layer(self):
        raw = {
            "block1": torch.tensor([2.0, 6.0]),
            "block2": torch.tensor([8.0, 3.0]),
        }
        relative = compute_interlayer_sensitivity(
            raw, ["block1", "block2"], input_values=torch.tensor([1.0, 2.0])
        )
        self.assertTrue(torch.allclose(relative["block1"], torch.tensor([2.0, 3.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(relative["block2"], torch.tensor([4.0, 0.5], dtype=torch.float64)))

    def test_rank_weighted_kl_aligns_each_rank_to_clean_teacher(self):
        teacher_logits = torch.tensor([[2.0, 0.0]])
        teacher_probs = torch.softmax(teacher_logits / 2.0, dim=1)
        students = torch.stack([teacher_logits[0], torch.tensor([0.0, 2.0])])
        weights = torch.tensor([0.75, 0.25])
        loss = rank_weighted_kl(students, teacher_probs, 2, weights, 2.0)
        expected = (
            kl_to_clean_teacher(students, teacher_probs.expand(2, -1), 2.0)
            * weights
        ).sum()
        self.assertAlmostEqual(loss.item(), expected.item())

    def test_top_40_percent_selection_is_deterministic(self):
        scores = {"b": 0.8, "a": 0.8, "c": 0.2, "d": 0.1}
        self.assertEqual(select_sensitive_layers(scores, 0.4), ["a", "b"])

    def test_subset_seed_matches_consistancy_pipeline(self):
        seed_everything(42)
        indices, selection_seed = stable_subset_indices(
            dataset_size=840,
            sample_num=512,
            seed=42,
            fold=0,
        )
        expected = np.random.RandomState(42).choice(
            840, size=512, replace=False
        ).tolist()
        self.assertEqual(selection_seed, 42)
        self.assertEqual(indices, expected)
        self.assertEqual(
            indices[:20],
            [
                695,
                816,
                30,
                599,
                96,
                244,
                558,
                352,
                464,
                543,
                447,
                254,
                250,
                65,
                668,
                215,
                39,
                192,
                86,
                493,
            ],
        )

    def test_autoattack_receives_explicit_seed(self):
        captured = {}

        class FakeAutoAttack:
            def __init__(self, model, **kwargs):
                captured.update(kwargs)

        original = ATTACK_CLASSES["autoattack"]
        ATTACK_CLASSES["autoattack"] = FakeAutoAttack
        try:
            build_attack(
                "autoattack",
                nn.Linear(2, 2),
                eps=0.03,
                info={"num_classes": 2},
                device=torch.device("cpu"),
                seed=42,
            )
        finally:
            ATTACK_CLASSES["autoattack"] = original
        self.assertEqual(captured["seed"], 42)

    def test_exp018_shared_subset_and_metadata_validation(self):
        expected = {
            "dataset": "toy",
            "model": "toy",
            "fold": 0,
            "seed": 42,
            "eps": 0.03,
            "sample_num": 2,
            "selection_seed": 42,
        }
        payloads = {
            method: validate_purification_payload(
                self._fair_payload(method), method, expected, [25, 30]
            )
            for method in ("at_tnp", "consistancy", "rpcf")
        }
        source_indices, labels, clean = validate_shared_subset(payloads)
        self.assertEqual(source_indices, [4, 7])
        self.assertTrue(torch.equal(labels, torch.tensor([0, 1])))
        self.assertEqual(tuple(clean.shape), (2, 1, 3, 4))

        invalid = self._fair_payload("rpcf")
        invalid["meta"]["attack_meta"]["attack_seed"] = 7
        with self.assertRaises(ValueError):
            validate_purification_payload(
                invalid, "rpcf", expected, [25, 30]
            )

    def test_rank_weight_curriculum(self):
        ranks = [15, 20, 25, 30, 35, 40]
        start = compute_rank_weights(ranks, epoch=0, epochs=3)
        middle = compute_rank_weights(ranks, epoch=1, epochs=3)
        end = compute_rank_weights(ranks, epoch=2, epochs=3)
        self.assertGreater(start[0].item(), start[-1].item())
        self.assertTrue(torch.allclose(middle, torch.full((6,), 1 / 6)))
        self.assertLess(end[0].item(), end[-1].item())
        static = compute_rank_weights(ranks, epoch=0, epochs=3, static=True)
        self.assertTrue(torch.allclose(static, torch.full((6,), 1 / 6)))

    def test_exp018_five_method_mapping_and_order(self):
        summaries = {
            "base": self._five_method_summary("base"),
            "selective": self._five_method_summary("selective"),
            "all_layers": self._five_method_summary("all_layers"),
            "uniform": self._five_method_summary("uniform"),
        }
        paths = {name: f"{name}.json" for name in summaries}
        validate_shared_protocol(summaries, paths)
        long_rows = build_five_method_rows(summaries, paths)
        self.assertEqual(
            [row["method_id"] for row in long_rows[::2]],
            [spec[0] for spec in METHOD_SPECS],
        )
        self.assertEqual([row["rank"] for row in long_rows[:2]], [25, 30])
        wide_rows = build_wide_rows(long_rows, [25, 30])
        self.assertEqual(len(wide_rows), 5)
        self.assertIn(
            "rank30_purified_adv_accuracy",
            wide_rows[-1],
        )
        markdown = render_markdown(wide_rows, [25, 30])
        self.assertIn("RPCF rank-weight uniform", markdown)
        self.assertIn("Full AutoAttack", markdown)

    def test_exp018_five_method_rejects_protocol_mismatch(self):
        summaries = {
            "base": self._five_method_summary("base"),
            "selective": self._five_method_summary("selective"),
            "all_layers": self._five_method_summary("all_layers"),
            "uniform": self._five_method_summary("uniform"),
        }
        summaries["uniform"]["source_indices"] = [7, 4]
        paths = {name: f"{name}.json" for name in summaries}
        with self.assertRaisesRegex(ValueError, "source_indices"):
            validate_shared_protocol(summaries, paths)

    def test_exp018_five_method_rejects_wrong_variant(self):
        summaries = {
            "base": self._five_method_summary("base"),
            "selective": self._five_method_summary("selective"),
            "all_layers": self._five_method_summary("all_layers"),
            "uniform": self._five_method_summary("uniform"),
        }
        summaries["uniform"]["rpcf"]["static_rank_weights"] = False
        paths = {name: f"{name}.json" for name in summaries}
        with self.assertRaisesRegex(ValueError, "uniform rank weights"):
            validate_shared_protocol(summaries, paths)

    def test_freeze_and_batchnorm_mode(self):
        model = TinyRPCFModel()
        stats = configure_trainable_layers(model, ["block2"])
        self.assertGreater(stats["trainable_params"], 0)
        self.assertFalse(model.block1[0].weight.requires_grad)
        self.assertTrue(model.block2[0].weight.requires_grad)
        self.assertFalse(model.head.weight.requires_grad)

        model.train()
        set_frozen_batchnorm_eval(model, selected_layers=["block2"])
        self.assertFalse(model.block1[1].training)
        self.assertTrue(model.block2[1].training)

    def test_all_layers_enables_all_parameters_and_batchnorm(self):
        model = TinyRPCFModel()
        stats = configure_trainable_layers(
            model, ["block2"], all_layers=True
        )
        self.assertEqual(stats["trainable_params"], stats["total_params"])
        self.assertEqual(stats["trainable_ratio"], 1.0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

        model.train()
        set_frozen_batchnorm_eval(
            model, selected_layers=["block2"], all_layers=True
        )
        self.assertTrue(model.block1[1].training)
        self.assertTrue(model.block2[1].training)

    def test_checkpoint_priority(self):
        best = {"robust_acc": 0.8, "clean_acc": 0.9, "val_loss": 0.5}
        self.assertTrue(
            checkpoint_is_better(
                {"robust_acc": 0.81, "clean_acc": 0.1, "val_loss": 2.0}, best
            )
        )
        self.assertTrue(
            checkpoint_is_better(
                {"robust_acc": 0.8, "clean_acc": 0.91, "val_loss": 2.0}, best
            )
        )
        self.assertTrue(
            checkpoint_is_better(
                {"robust_acc": 0.8, "clean_acc": 0.9, "val_loss": 0.4}, best
            )
        )
        self.assertFalse(
            checkpoint_is_better(
                {"robust_acc": 0.79, "clean_acc": 1.0, "val_loss": 0.1}, best
            )
        )

if __name__ == "__main__":
    unittest.main()
