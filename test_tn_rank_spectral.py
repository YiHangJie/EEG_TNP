import inspect
import unittest
from types import SimpleNamespace

import torch

from TN.rank_growth.PTR_3d_rank_spectral import PTR_3d_rank_spectral


class _Logger:
    def info(self, *args, **kwargs):
        return None


def _build_model(**kwargs):
    defaults = {
        "target": torch.randn(2, 2, 8),
        "stage": 2,
        "max_rank": 4,
        "device": "cpu",
        "num_iterations": 2,
        "iterations_for_upsampling": [1],
        "rank_ard_min_rank": 2,
        "rank_ard_full_refit_steps": 0,
        "rank_spectral_min_rank": 2,
        "rank_spectral_max_rank": 4,
    }
    defaults.update(kwargs)
    return PTR_3d_rank_spectral(**defaults)


class RankSpectralTest(unittest.TestCase):
    def test_unknown_noise_threshold_coefficient_at_square_matrix(self):
        value = PTR_3d_rank_spectral.optimal_hard_threshold_coefficient(1.0)
        self.assertAlmostEqual(value, 2.86, places=6)

    def test_spectral_rank_separates_large_values_from_noise_floor(self):
        singular_values = torch.tensor([10.0, 8.0, 0.1, 0.1, 0.1, 0.1])
        result = PTR_3d_rank_spectral.select_spectral_rank(
            singular_values,
            matrix_rows=10,
            matrix_cols=10,
            min_rank=1,
            max_rank=6,
        )
        self.assertEqual(result["raw_rank"], 2)
        self.assertEqual(result["selected_rank"], 2)

    def test_full_rank_two_site_rounding_preserves_joint_tensor(self):
        model = _build_model(rank_spectral_min_rank=4, rank_spectral_max_rank=4)
        bond = model.variable_bond_indices()[0]
        before, _, _ = model._merge_two_site(
            model.tn[bond].detach(), model.tn[bond + 1].detach()
        )
        row = model._round_two_site(bond, sweep_index=0)
        after, _, _ = model._merge_two_site(
            model.tn[bond].detach(), model.tn[bond + 1].detach()
        )
        self.assertEqual(row["selected_rank"], 4)
        self.assertTrue(torch.allclose(before, after, rtol=1e-4, atol=1e-6))

    def test_truncated_rounding_updates_both_adjacent_shapes(self):
        model = _build_model(rank_spectral_max_rank=2)
        bond = model.variable_bond_indices()[0]
        row = model._round_two_site(bond, sweep_index=0)
        self.assertEqual(row["selected_rank"], 2)
        self.assertEqual(model.tn[bond].shape[-1], 2)
        self.assertEqual(model.tn[bond + 1].shape[0], 2)
        self.assertGreaterEqual(row["retained_energy"], 0.0)
        self.assertLessEqual(row["retained_energy"], 1.0 + 1e-6)

    def test_training_and_diagnostics_remain_tn_only(self):
        self.assertNotIn(
            "labels", inspect.signature(PTR_3d_rank_spectral.train).parameters
        )
        model = _build_model(rank_spectral_max_rank=2)
        args = SimpleNamespace(
            lr=0.005,
            lr_decay_factor=0.9,
            lr_decay_factor_until_next_upsampling=0.5,
            num_iterations=2,
            iterations_for_upsampling=[1, 2],
        )
        model.train(model.target, args, target_index=3, logging=_Logger())
        diagnostics = model.get_rank_diagnostics()
        self.assertFalse(diagnostics["uses_labels"])
        self.assertFalse(diagnostics["uses_classifier_logits"])
        self.assertTrue(diagnostics["spectral_rounding"])
        self.assertTrue(all(rank <= 2 for rank in diagnostics["final_rank_vector"]))


if __name__ == "__main__":
    unittest.main()
