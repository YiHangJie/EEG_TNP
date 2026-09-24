"""EXP-033 标准分解的参数公平性与张量重构检查；仅使用 CPU。"""

import unittest

import tensorly as tl
import torch

from rpcf.exp033_structures import (BUDGETS, METHODS, SHAPE, candidates, decompose,
                                   parameter_count, select_candidate)


class StructureTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fullrank_reconstruction_and_backend_restoration(self):
        shape = (2, 3, 8)
        data = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
        ranks = dict(tr_dense=[2, 4, 3, 2], tt_dense=[1, 2, 6, 1],
                     tt_time=[1, 2, 6, 4, 2, 1], tucker=[2, 3, 6], svd=6)
        before = tl.get_backend()
        for method, rank in ranks.items():
            with self.subTest(method=method):
                spec = dict(method=method, rank=rank, shape=shape,
                            parameters=parameter_count(method, rank, shape))
                result, diag = decompose(data, spec)
                self.assertEqual(tuple(result.shape), shape)
                self.assertLess(float((result - data).abs().max()), 1e-12)
                self.assertEqual(diag["actual_parameters"], spec["parameters"])
                self.assertEqual(diag["actual_rank"], rank)
                self.assertEqual(tl.get_backend(), before)

    def test_candidates_feasible_with_budget_and_nondominance(self):
        for method in METHODS:
            for budget in BUDGETS:
                specs = candidates(method, budget)
                self.assertTrue(specs, (method, budget))
                ranks = [(s["rank"],) if isinstance(s["rank"], int) else tuple(s["rank"]) for s in specs]
                for i, spec in enumerate(specs):
                    with self.subTest(method=method, budget=budget, rank=spec["rank"]):
                        self.assertEqual(spec["parameters"], parameter_count(method, spec["rank"]))
                        if not spec["budget_exception"]:
                            self.assertLessEqual(abs(spec["budget_deviation"]), .05)
                        self.assertFalse(any(j != i and all(x <= y for x, y in zip(ranks[i], rank))
                                             and ranks[i] != rank for j, rank in enumerate(ranks)))
                        if method == "tr_dense":
                            a, b, c, closing = spec["rank"]
                            self.assertEqual(a, closing)
                            self.assertGreaterEqual(min(a, b, c), 2)
                            self.assertLessEqual(a * c, min(SHAPE[-1], SHAPE[0] * SHAPE[1]))
                            self.assertLessEqual(b, min(10 * a, 11 * c))

    def test_tr_exception_actual_factors_and_no_rank_one(self):
        spec, = candidates("tr_dense", 25)
        self.assertEqual(spec["rank"], [3, 22, 2, 3])
        self.assertEqual(spec["parameters"], 13432)
        self.assertTrue(spec["budget_exception"])
        self.assertAlmostEqual(spec["budget_deviation"], 13432 / 14731 - 1)
        data = torch.randn(SHAPE, generator=torch.Generator().manual_seed(42))
        result, diag = decompose(data, spec)
        self.assertEqual(diag["factor_shapes"], [[3, 10, 22], [22, 11, 2], [2, 2048, 3]])
        self.assertEqual(diag["actual_parameters"], 13432)
        self.assertTrue(bool(torch.isfinite(result).all()))
        with self.assertRaises(ValueError):
            decompose(torch.zeros(2, 3, 8), dict(method="tr_dense", rank=[1, 2, 2, 1]))

    def test_tt_time_clipped_ranks_match_factors(self):
        spec, = candidates("tt_time", 30)
        self.assertEqual(spec["tensorized_shape"], [10, 11] + [2] * 11)
        self.assertEqual(spec["rank"][-2], 2)
        self.assertEqual(spec["rank"][1], 10)
        data = torch.randn(SHAPE, generator=torch.Generator().manual_seed(43))
        result, diag = decompose(data, spec)
        self.assertEqual(diag["actual_rank"], spec["rank"])
        self.assertEqual(diag["actual_parameters"], spec["parameters"])
        self.assertEqual(result.shape, data.shape)

    def test_selection_ties_and_candidate_copy(self):
        specs = [dict(rank=[3, 2], mean_relative_error=.5, budget_deviation=.03),
                 dict(rank=[2, 3], mean_relative_error=.5, budget_deviation=-.03),
                 dict(rank=[1, 1], mean_relative_error=.6, budget_deviation=.0)]
        self.assertIs(select_candidate(specs), specs[1])
        specs[0]["budget_deviation"] = .02
        self.assertIs(select_candidate(specs), specs[0])
        with self.assertRaises(ValueError):
            select_candidate([dict(specs[0], mean_relative_error=float("nan"))])
        changed = candidates("tr_dense", 25)[0]
        changed["rank"][0] = 999
        self.assertEqual(candidates("tr_dense", 25)[0]["rank"][0], 3)

    def test_invalid_shape_budget_and_count_rejected(self):
        with self.assertRaises(ValueError):
            candidates("tt_time", 25, (2, 3, 7))
        with self.assertRaises(ValueError):
            candidates("svd", 99)
        with self.assertRaises(ValueError):
            decompose(torch.zeros(2, 3, 8), dict(method="svd", rank=2, parameters=1))
        with self.assertRaises(ValueError):
            decompose(torch.full((2, 3, 8), float("nan")), dict(method="svd", rank=2))


if __name__ == "__main__":
    unittest.main()
