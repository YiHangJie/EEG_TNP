import inspect
import unittest

import torch

from rpcf.build_exp030_rank_knee_payload import (
    gather_selected,
    reconstruction_mse_curves,
    select_log_mse_knee,
)


class RankKneeTest(unittest.TestCase):
    def test_known_diminishing_return_curve_selects_middle_knee(self):
        ranks = [15, 20, 25, 30, 35, 40]
        mse = torch.tensor([[1.0, 0.45, 0.25, 0.20, 0.18, 0.17]])
        result = select_log_mse_knee(mse, ranks)
        self.assertEqual(ranks[int(result["selected_indices"][0])], 25)

    def test_non_monotone_noise_uses_cumulative_min_envelope(self):
        ranks = [15, 20, 25, 30]
        mse = torch.tensor([[1.0, 0.5, 0.55, 0.2]])
        result = select_log_mse_knee(mse, ranks)
        self.assertTrue(
            torch.equal(
                result["monotone_mse"],
                torch.tensor([[1.0, 0.5, 0.5, 0.2]]),
            )
        )

    def test_selection_api_cannot_receive_labels_or_classifier(self):
        parameters = inspect.signature(select_log_mse_knee).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("classifier", parameters)
        ranks = [15, 20, 25]
        mse = torch.tensor([[1.0, 0.5, 0.2], [1.0, 0.6, 0.3]])
        baseline = select_log_mse_knee(mse, ranks)["selected_indices"]
        for ignored_labels in (torch.tensor([0, 1]), torch.tensor([1, 0])):
            del ignored_labels
            current = select_log_mse_knee(mse, ranks)["selected_indices"]
            self.assertTrue(torch.equal(baseline, current))

    def test_mse_and_gather_preserve_sample_alignment(self):
        inputs = torch.zeros(2, 1, 2)
        purified = torch.tensor(
            [
                [[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]],
                [[[4.0, 4.0]], [[5.0, 5.0]], [[6.0, 6.0]]],
            ]
        )
        mse = reconstruction_mse_curves(inputs, purified)
        self.assertEqual(mse.tolist(), [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
        selected = gather_selected(purified, torch.tensor([2, 0]))
        self.assertTrue(torch.equal(selected[0], purified[0, 2]))
        self.assertTrue(torch.equal(selected[1], purified[1, 0]))


if __name__ == "__main__":
    unittest.main()
