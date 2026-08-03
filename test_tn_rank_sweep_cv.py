import inspect
import unittest
from types import SimpleNamespace

import torch

from TN.rank_growth.PTR_3d_rank_sweep_cv import PTR_3d_rank_sweep_cv


def _args():
    return SimpleNamespace(
        lr=0.008,
        lr_decay_factor=0.5,
        lr_decay_patience=50,
        lr_decay_factor_until_next_upsampling=0.5,
        L2_regularization=0.0,
        regularization_weight=0.0,
        num_iterations=20,
        iterations_for_upsampling=[10],
    )


class RankSweepCVTest(unittest.TestCase):
    def test_scaled_boundaries(self):
        self.assertEqual(
            PTR_3d_rank_sweep_cv.scaled_boundaries(
                [256, 512, 1024], total_steps=2048, new_total_steps=512
            ),
            [64, 128, 256],
        )

    def test_train_interface_is_classifier_free(self):
        parameters = inspect.signature(PTR_3d_rank_sweep_cv.train).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("logits", parameters)
        self.assertIsNone(parameters["clf"].default)

    def test_classifier_is_rejected_before_rank_inference(self):
        target = torch.randn(2, 2, 8)
        model = PTR_3d_rank_sweep_cv(
            target=target,
            stage=2,
            max_rank=3,
            num_iterations=20,
            iterations_for_upsampling=[10],
            rank_sweep_candidates=[2, 3],
            rank_sweep_cv_iterations=2,
            rank_sweep_mask_fraction=0.25,
            rank_sweep_mask_blocks=2,
        )
        with self.assertRaisesRegex(ValueError, "cannot access a classifier"):
            model.train(target, _args(), clf=object())

    def test_invalid_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            PTR_3d_rank_sweep_cv(
                target=torch.randn(2, 2, 8),
                stage=2,
                max_rank=3,
                num_iterations=2,
                iterations_for_upsampling=[1],
                rank_sweep_candidates=[],
            )

    def test_tiny_end_to_end_fit(self):
        torch.manual_seed(7)
        target = torch.randn(2, 2, 8)
        model = PTR_3d_rank_sweep_cv(
            target=target,
            stage=2,
            max_rank=3,
            num_iterations=20,
            iterations_for_upsampling=[10],
            rank_sweep_candidates=[2, 3],
            rank_sweep_cv_iterations=2,
            rank_sweep_mask_fraction=0.25,
            rank_sweep_mask_blocks=2,
            rank_sweep_one_se_multiplier=1.0,
            rank_sweep_seed=11,
        )
        output, _, _ = model.train(target, _args())
        diagnostics = model.get_rank_diagnostics()

        self.assertEqual(tuple(output.shape), tuple(target.shape))
        self.assertIn(diagnostics["selected_rank"], [2, 3])
        self.assertEqual(
            [row["rank"] for row in diagnostics["candidate_validation"]], [2, 3]
        )
        self.assertNotIn("labels", diagnostics)
        self.assertNotIn("logits", diagnostics)
        self.assertNotIn("predictions", diagnostics)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
