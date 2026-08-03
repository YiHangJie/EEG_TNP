import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rpcf.analyze_exp027_oracle_rank import (
    align_methods,
    load_method_payloads,
    paired_bootstrap_difference,
    select_best_fixed,
    select_robust_first_oracle,
)


class Exp027OracleRankTest(unittest.TestCase):
    def test_best_fixed_rank_prefers_adv_then_clean_then_lower_rank(self):
        ranks = [15, 20, 25]
        adv = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
                [0, 0, 0],
            ],
            dtype=torch.bool,
        )
        clean = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertEqual(select_best_fixed(clean, adv, ranks), 1)

        adv[:, 2] = adv[:, 1]
        clean[:, 2] = clean[:, 1]
        self.assertEqual(select_best_fixed(clean, adv, ranks), 1)

    def test_robust_first_oracle_uses_clean_and_margin_tiebreaks(self):
        ranks = [15, 20, 25]
        adv_correct = torch.tensor(
            [[0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=torch.bool
        )
        clean_correct = torch.tensor(
            [[1, 0, 1], [1, 0, 0], [1, 1, 1]], dtype=torch.bool
        )
        adv_margin = torch.tensor(
            [[-0.5, 0.8, 0.4], [-0.2, -0.3, -0.4], [0.2, 0.7, 0.7]]
        )
        clean_margin = torch.tensor(
            [[0.5, -0.1, 0.3], [0.6, -0.2, -0.3], [0.1, 0.2, 0.2]]
        )
        selected = select_robust_first_oracle(
            clean_correct,
            adv_correct,
            clean_margin,
            adv_margin,
            ranks,
        )
        self.assertEqual(selected.tolist(), [2, 0, 1])

    def test_paired_bootstrap_is_reproducible(self):
        oracle = np.array([1, 1, 1, 0, 1], dtype=np.int64)
        fixed = np.array([1, 0, 1, 0, 0], dtype=np.int64)
        first = paired_bootstrap_difference(oracle, fixed, 200, seed=42)
        second = paired_bootstrap_difference(oracle, fixed, 200, seed=42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 0.4)
        self.assertGreaterEqual(first["ci_low"], 0.0)

    @staticmethod
    def _payload(rank, order, checkpoint="checkpoint.pth", attack="attack.pth"):
        values = {
            4: torch.full((1, 2, 2), 4.0),
            7: torch.full((1, 2, 2), 7.0),
        }
        labels_by_source = {4: 0, 7: 1}
        clean = torch.stack([values[source] for source in order])
        adversarial = clean + 0.1
        clean_pur = clean + float(rank) / 100.0
        adv_pur = adversarial + float(rank) / 100.0
        return {
            "clean": clean,
            "adversarial": adversarial,
            "clean_pur_by_rank": clean_pur.unsqueeze(1),
            "adv_pur_by_rank": adv_pur.unsqueeze(1),
            "labels": torch.tensor([labels_by_source[source] for source in order]),
            "source_indices": list(order),
            "ranks": [rank],
            "metrics": [],
            "meta": {
                "kind": "rpcf_purification_eval",
                "dataset": "thubenchmark",
                "model": "eegnet",
                "fold": 0,
                "seed": 42,
                "eps": 0.03,
                "sample_num": 2,
                "checkpoint_path": checkpoint,
                "attack_path": attack,
            },
        }

    def _save_payload(self, root, name, payload):
        path = Path(root) / name
        torch.save(payload, path)
        return str(path)

    def test_payload_merge_reorders_source_indices_and_sorts_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            rank20 = self._save_payload(
                root, "rank20.pth", self._payload(20, [4, 7])
            )
            rank15 = self._save_payload(
                root, "rank15.pth", self._payload(15, [7, 4])
            )
            merged = load_method_payloads(
                "madry_at", [rank20, rank15], expected_ranks=[15, 20]
            )
        self.assertEqual(merged["ranks"], [15, 20])
        self.assertEqual(merged["source_indices"], [4, 7])
        self.assertEqual(merged["labels"].tolist(), [0, 1])
        self.assertAlmostEqual(
            merged["clean_pur_by_rank"][0, 0, 0, 0, 0].item(), 4.15, places=5
        )
        self.assertAlmostEqual(
            merged["clean_pur_by_rank"][1, 1, 0, 0, 0].item(), 7.20, places=5
        )

    def test_payload_merge_rejects_duplicate_missing_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            rank15_a = self._save_payload(
                root, "rank15_a.pth", self._payload(15, [4, 7])
            )
            rank15_b = self._save_payload(
                root, "rank15_b.pth", self._payload(15, [4, 7])
            )
            with self.assertRaisesRegex(ValueError, "duplicate rank"):
                load_method_payloads(
                    "madry_at", [rank15_a, rank15_b], expected_ranks=[15]
                )
            with self.assertRaisesRegex(ValueError, "expected"):
                load_method_payloads(
                    "madry_at", [rank15_a], expected_ranks=[15, 20]
                )

            invalid = self._payload(20, [4, 7])
            invalid["labels"] = torch.tensor([1, 0])
            rank20 = self._save_payload(root, "rank20_bad.pth", invalid)
            with self.assertRaisesRegex(ValueError, "labels mismatch"):
                load_method_payloads(
                    "madry_at", [rank15_a, rank20], expected_ranks=[15, 20]
                )

    def test_payload_merge_rejects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            rank15 = self._save_payload(
                root, "rank15.pth", self._payload(15, [4, 7])
            )
            invalid = self._payload(20, [4, 7])
            invalid["meta"]["sample_num"] = 512
            rank20 = self._save_payload(root, "rank20_bad_meta.pth", invalid)
            with self.assertRaisesRegex(ValueError, "meta.sample_num"):
                load_method_payloads(
                    "madry_at", [rank15, rank20], expected_ranks=[15, 20]
                )

    def test_payload_merge_rejects_source_index_set_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            rank15 = self._save_payload(
                root, "rank15.pth", self._payload(15, [4, 7])
            )
            invalid = self._payload(20, [4, 7])
            invalid["source_indices"] = [4, 9]
            rank20 = self._save_payload(root, "rank20_bad_sources.pth", invalid)
            with self.assertRaisesRegex(ValueError, "source_indices set mismatch"):
                load_method_payloads(
                    "madry_at", [rank15, rank20], expected_ranks=[15, 20]
                )

    def test_cross_method_alignment_reorders_but_keeps_own_adversarial(self):
        with tempfile.TemporaryDirectory() as root:
            madry_path = self._save_payload(
                root,
                "madry.pth",
                self._payload(15, [4, 7], checkpoint="madry.pth", attack="madry_adv.pth"),
            )
            rpcf_payload = self._payload(
                15, [7, 4], checkpoint="rpcf.pth", attack="rpcf_adv.pth"
            )
            rpcf_payload["adversarial"] = rpcf_payload["adversarial"] + 1.0
            rpcf_payload["adv_pur_by_rank"] = rpcf_payload["adv_pur_by_rank"] + 1.0
            rpcf_path = self._save_payload(root, "rpcf.pth", rpcf_payload)
            methods = {
                "madry_at": load_method_payloads(
                    "madry_at", [madry_path], expected_ranks=[15]
                ),
                "rpcf_at": load_method_payloads(
                    "rpcf_at", [rpcf_path], expected_ranks=[15]
                ),
            }
            align_methods(methods)
        self.assertEqual(methods["rpcf_at"]["source_indices"], [4, 7])
        self.assertTrue(
            torch.equal(methods["madry_at"]["clean"], methods["rpcf_at"]["clean"])
        )
        self.assertFalse(
            torch.equal(
                methods["madry_at"]["adversarial"],
                methods["rpcf_at"]["adversarial"],
            )
        )


if __name__ == "__main__":
    unittest.main()
