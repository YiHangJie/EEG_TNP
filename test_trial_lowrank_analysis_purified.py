import unittest

import pandas as pd
import torch

from trial_lowrank_analysis.analyze_purified_trial_hosvd_lowrank import (
    validate_and_align_payload,
)
from trial_lowrank_analysis.prepare_purification_input import build_attack_payload


class _Args:
    dataset = "thubenchmark"
    model = "eegnet"
    fold = 0
    seed = 42
    eps = 0.03


class PurifiedTrialLowrankTest(unittest.TestCase):
    @staticmethod
    def _bundle():
        return {
            "clean": torch.tensor([[[[1.0]]], [[[2.0]]]]),
            "adv": torch.tensor([[[[1.1]]], [[[2.2]]]]),
            "labels": torch.tensor([3, 4]),
            "metadata": {
                "original_split_index": [7, 4],
                "subject_id": ["s1", "s2"],
            },
            "args": {
                "dataset": "thubenchmark",
                "model": "eegnet",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "pgd_steps": 200,
                "pgd_alpha": 2 / 255,
            },
        }

    def test_prepare_payload_preserves_original_order(self):
        payload = build_attack_payload(
            self._bundle(),
            bundle_path="bundle.pt",
            dataset="thubenchmark",
            model="eegnet",
            fold=0,
            seed=42,
            eps=0.03,
        )
        self.assertEqual(payload["source_indices"], [7, 4])
        self.assertEqual(payload["labels"].tolist(), [3, 4])
        self.assertEqual(payload["meta"]["pgd_steps"], 200)

    def test_purification_payload_is_reordered_to_bundle_order(self):
        bundle = self._bundle()
        source = {
            "clean": bundle["clean"],
            "adv": bundle["adv"],
            "labels": bundle["labels"],
            "metadata": pd.DataFrame(bundle["metadata"]),
            "source_indices": [7, 4],
        }
        payload = {
            "clean": bundle["clean"].flip(0),
            "adversarial": bundle["adv"].flip(0),
            "clean_pur_by_rank": (bundle["clean"].flip(0) + 1).unsqueeze(1),
            "adv_pur_by_rank": (bundle["adv"].flip(0) + 1).unsqueeze(1),
            "labels": bundle["labels"].flip(0),
            "source_indices": [4, 7],
            "ranks": [15],
            "meta": {
                "dataset": "thubenchmark",
                "model": "eegnet",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "configs": ["rank15.yaml"],
            },
        }
        records = validate_and_align_payload(payload, source, _Args(), "rank15.pth")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_indices"], [7, 4])
        self.assertEqual(records[0]["labels"].tolist(), [3, 4])
        self.assertEqual(records[0]["clean_pur"][:, 0, 0, 0].tolist(), [2.0, 3.0])

    def test_rejects_label_mismatch(self):
        bundle = self._bundle()
        source = {
            "clean": bundle["clean"],
            "adv": bundle["adv"],
            "labels": bundle["labels"],
            "metadata": pd.DataFrame(bundle["metadata"]),
            "source_indices": [7, 4],
        }
        payload = {
            "clean": bundle["clean"],
            "adversarial": bundle["adv"],
            "clean_pur_by_rank": bundle["clean"].unsqueeze(1),
            "adv_pur_by_rank": bundle["adv"].unsqueeze(1),
            "labels": torch.tensor([9, 9]),
            "source_indices": [7, 4],
            "ranks": [15],
            "meta": {
                "dataset": "thubenchmark",
                "model": "eegnet",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "configs": ["rank15.yaml"],
            },
        }
        with self.assertRaisesRegex(ValueError, "labels mismatch"):
            validate_and_align_payload(payload, source, _Args(), "rank15.pth")


if __name__ == "__main__":
    unittest.main()
