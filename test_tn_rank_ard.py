import inspect
import unittest
from types import SimpleNamespace

import torch

from TN.rank_growth.PTR_3d_rank_ard import PTR_3d_rank_ard
from TN.rank_growth.PTR_3d_rank_cv import PTR_3d_rank_cv
from rpcf.analyze_exp030_auto_rank import align_auto_to_fixed


class _Logger:
    def info(self, *args, **kwargs):
        return None


def _build_model(target, **kwargs):
    defaults = {
        "stage": 2,
        "max_rank": 4,
        "device": "cpu",
        "num_iterations": 2,
        "iterations_for_upsampling": [1],
        "rank_ard_min_rank": 2,
        "rank_ard_energy": 0.8,
        "rank_ard_enable_regrow": False,
    }
    defaults.update(kwargs)
    return PTR_3d_rank_ard(target=target, **defaults)


class RankARDTest(unittest.TestCase):
    def test_paired_score_is_gauge_invariant(self):
        torch.manual_seed(1)
        left = torch.randn(3, 2, 4)
        right = torch.randn(4, 2, 5)
        scale = torch.tensor([0.25, 2.0, 4.0, 0.5])
        baseline = PTR_3d_rank_ard.paired_component_scores(left, right)
        transformed = PTR_3d_rank_ard.paired_component_scores(
            left * scale.view(1, 1, -1),
            right / scale.view(-1, 1, 1),
        )
        self.assertTrue(torch.allclose(baseline, transformed, rtol=1e-5, atol=1e-6))

    def test_component_selection_respects_energy_and_min_rank(self):
        scores = torch.tensor([10.0, 3.0, 1.0, 0.1])
        keep = PTR_3d_rank_ard.select_components(scores, min_rank=2, energy=0.9)
        self.assertEqual(keep.tolist(), [0, 1])
        keep_all = PTR_3d_rank_ard.select_components(
            torch.ones(4), min_rank=2, energy=1.0
        )
        self.assertEqual(keep_all.numel(), 4)

    def test_physical_prune_and_regrow_update_adjacent_shapes(self):
        model = _build_model(torch.randn(2, 2, 8))
        bond = model.variable_bond_indices()[0]
        old_left = model.tn[bond].shape
        old_right = model.tn[bond + 1].shape
        pruned = model._prune_bond(bond, torch.tensor([0, 2]))
        self.assertEqual(pruned, 2)
        self.assertEqual(model.tn[bond].shape[-1], 2)
        self.assertEqual(model.tn[bond + 1].shape[0], 2)
        self.assertEqual(model.tn[bond].shape[:-1], old_left[:-1])
        self.assertEqual(model.tn[bond + 1].shape[1:], old_right[1:])
        regrown = model._regrow_bond(bond, 1)
        self.assertEqual(regrown, 1)
        self.assertEqual(model.tn[bond].shape[-1], 3)
        self.assertEqual(model.tn[bond + 1].shape[0], 3)

    def test_training_is_reproducible_and_label_permutation_cannot_enter(self):
        self.assertNotIn("labels", inspect.signature(PTR_3d_rank_ard.train).parameters)
        target = torch.randn(2, 2, 8)
        ranks = []
        for ignored_labels in (torch.tensor([0, 1]), torch.tensor([1, 0])):
            del ignored_labels
            torch.manual_seed(7)
            model = _build_model(target)
            args = SimpleNamespace(
                lr=0.005,
                lr_decay_factor=0.9,
                lr_decay_factor_until_next_upsampling=0.5,
                num_iterations=2,
                iterations_for_upsampling=[1, 2],
            )
            model.train(target, args, target_index=3, logging=_Logger())
            diagnostics = model.get_rank_diagnostics()
            self.assertFalse(diagnostics["uses_labels"])
            self.assertFalse(diagnostics["uses_classifier_logits"])
            ranks.append(diagnostics["final_rank_vector"])
        self.assertEqual(ranks[0], ranks[1])

    def test_classifier_is_rejected_during_rank_inference(self):
        target = torch.randn(2, 2, 8)
        model = _build_model(target)
        args = SimpleNamespace(
            lr=0.005,
            lr_decay_factor=0.9,
            lr_decay_factor_until_next_upsampling=0.5,
            num_iterations=2,
            iterations_for_upsampling=[1, 2],
        )
        with self.assertRaisesRegex(ValueError, "cannot access a classifier"):
            model.train(target, args, clf=object(), logging=_Logger())

    def test_pilot_auto_payload_aligns_to_n512_fixed_subset(self):
        source_indices = [10, 20]
        clean = torch.stack([torch.full((1, 2, 2), value) for value in (1.0, 2.0)])
        auto = {
            "clean": clean.clone(),
            "adversarial": clean.clone() + 0.1,
            "clean_pur": clean.clone(),
            "adv_pur": clean.clone(),
            "labels": torch.tensor([0, 1]),
            "source_indices": source_indices,
            "meta": {
                "dataset": "thubenchmark",
                "model": "eegnet",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "sample_num": 2,
                "checkpoint_path": "checkpoint.pth",
                "attack_path": "attack.pth",
            },
        }
        fixed_clean = torch.stack(
            [torch.full((1, 2, 2), value) for value in (0.0, 1.0, 2.0)]
        )
        fixed = {
            "clean": fixed_clean,
            "adversarial": fixed_clean + 0.1,
            "clean_pur_by_rank": fixed_clean.unsqueeze(1),
            "adv_pur_by_rank": fixed_clean.unsqueeze(1),
            "labels": torch.tensor([2, 0, 1]),
            "source_indices": [5, 10, 20],
            "meta": {
                "dataset": "thubenchmark",
                "model": "eegnet",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "sample_num": 512,
                "checkpoint_path": "checkpoint.pth",
                "attack_path": "attack.pth",
            },
        }
        aligned_auto, aligned_fixed = align_auto_to_fixed(auto, fixed, "synthetic")
        self.assertEqual(aligned_fixed["source_indices"], source_indices)
        self.assertTrue(torch.equal(aligned_auto["labels"], aligned_fixed["labels"]))
        self.assertEqual(aligned_fixed["clean"].shape[0], 2)


class RankMaskedCVTest(unittest.TestCase):
    def test_one_se_rule_selects_lowest_eligible_rank(self):
        rows = [
            {"rank": 15, "heldout_mse": 0.120, "heldout_se": 0.010},
            {"rank": 20, "heldout_mse": 0.108, "heldout_se": 0.009},
            {"rank": 25, "heldout_mse": 0.100, "heldout_se": 0.010},
        ]
        selected, best, threshold = PTR_3d_rank_cv.select_rank_one_se(rows)
        self.assertEqual(best, 25)
        self.assertEqual(selected, 20)
        self.assertAlmostEqual(threshold, 0.110)

    def test_block_mask_is_reproducible_and_hides_complete_time_blocks(self):
        kwargs = {
            "stage": 2,
            "max_rank": 4,
            "device": "cpu",
            "num_iterations": 2,
            "iterations_for_upsampling": [1],
            "rank_ard_min_rank": 2,
            "rank_ard_mask_fraction": 0.5,
            "rank_cv_candidate_ranks": [2, 3, 4],
            "rank_cv_holdout_blocks": 4,
        }
        target = torch.randn(2, 2, 8)
        models = [PTR_3d_rank_cv(target=target, **kwargs) for _ in range(2)]
        for model in models:
            model._build_train_masks(target_index=7)
        self.assertEqual(
            models[0]._rank_cv_holdout_block_ids,
            models[1]._rank_cv_holdout_block_ids,
        )
        self.assertTrue(
            torch.equal(models[0]._train_masks[-1], models[1]._train_masks[-1])
        )
        mask = models[0]._train_masks[-1]
        for start, stop in models[0]._rank_cv_block_slices:
            self.assertFalse(mask[..., start:stop].any())

    def test_final_cv_prunes_without_classifier_or_labels(self):
        self.assertNotIn("labels", inspect.signature(PTR_3d_rank_cv.train).parameters)
        model = PTR_3d_rank_cv(
            target=torch.randn(2, 2, 8),
            stage=2,
            max_rank=4,
            device="cpu",
            num_iterations=2,
            iterations_for_upsampling=[1],
            rank_ard_min_rank=2,
            rank_ard_mask_fraction=0.5,
            rank_cv_candidate_ranks=[2, 3, 4],
            rank_cv_holdout_blocks=4,
        )
        model._build_train_masks(target_index=3)
        row = model._stage_adapt(
            len(model.interm_resos) - 1, model.end_reso, allow_regrow=False
        )
        diagnostics = model.get_rank_diagnostics()
        self.assertIn(diagnostics["selected_rank"], [2, 3, 4])
        self.assertGreater(row["pruned_count"], 0)
        self.assertFalse(diagnostics["uses_labels"])
        self.assertFalse(diagnostics["uses_classifier_logits"])


if __name__ == "__main__":
    unittest.main()
