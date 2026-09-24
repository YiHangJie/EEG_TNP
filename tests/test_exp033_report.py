"""EXP-033 汇总证据和配对案例绘图的最小验收。"""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rpcf.exp033_report import LOSS_DEFAULTS, _plot_summary, _resolve_output_path, render_cases, summarize


class EXP033ReportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name)
        (self.run_dir / "status").mkdir()
        self.tasks = []
        self.manifest = dict(experiment_id="EXP-033", run_id="test", smoke=True, seeds=[42])
        self.plot_patch = patch("rpcf.exp033_report._plot_summary", return_value=[])
        self.plot_patch.start()
        self.addCleanup(self.plot_patch.stop)

    def write_graph(self):
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest))
        (self.run_dir / "tasks.json").write_text(json.dumps(self.tasks))

    def add_result(self, name="rank42", seed=42, clean=None, adv=None, **fields):
        clean = [0, 1] if clean is None else clean
        adv = [1, 1] if adv is None else adv
        row = dict(group="rank", method="caf", rank=25, seed=seed, sample_num=2,
                   standard_accuracy=sum(x == y for x, y in zip(clean, [0, 1])) / 2,
                   robust_accuracy=sum(x == y for x, y in zip(adv, [0, 1])) / 2,
                   clean_predictions=clean, adv_predictions=adv)
        row.update(fields)
        payload = dict(source_indices=[2, 7], labels=[0, 1], rows=[row])
        output_path = self.run_dir / (name + ".json")
        output_path.write_text(json.dumps(payload))
        task = dict(task_id=name, seed=seed, group=row["group"], kind="evaluate",
                    output_path=str(output_path), dependencies=[])
        self.tasks.append(task)
        (self.run_dir / "status" / (name + ".json")).write_text(json.dumps(dict(status="completed")))
        return output_path

    def test_accuracy_replayed_and_wrong_value_rejected(self):
        self.add_result(robust_accuracy=.9)
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertEqual(result["status"], "Pending")
        self.assertEqual(result["rows"], [])
        self.assertIn("Prediction replay mismatch", result["errors"][0])
        with self.assertRaisesRegex(ValueError, "Pending"):
            summarize(self.run_dir, strict=True)
        self.assertTrue((self.run_dir / "summary/report.json").exists())

    def test_completed_missing_output_and_fresh_pending_are_explicit(self):
        path = self.add_result()
        path.unlink()
        self.tasks.append(dict(task_id="fresh", seed=42, group="loss", kind="train",
                               output_path="fresh.json", dependencies=[]))
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertEqual(result["completed_task_count"], 0)
        self.assertEqual(len(result["pending_tasks"]), 2)
        self.assertEqual(result["pending_tasks"][0]["status"], "invalid")
        self.assertEqual(result["pending_tasks"][1]["status"], "pending")
        self.assertIn("missing", result["errors"][0])

    def test_seed_mean_sample_std_and_not_population_std(self):
        self.manifest["seeds"] = [42, 43]
        self.add_result(seed=42, adv=[0, 1])
        self.add_result(name="rank43", seed=43, adv=[1, 0])
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(result["status"], "Complete")
        row = result["grouped"][0]
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["robust_accuracy_mean"], .5)
        self.assertAlmostEqual(row["robust_accuracy_std"], math.sqrt(.5))

    def test_one_seed_std_none_and_missing_seed_stays_pending(self):
        self.manifest["seeds"] = [42, 43]
        self.add_result()
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertEqual(result["status"], "Pending")
        row = result["grouped"][0]
        self.assertEqual(row["missing_seeds"], [43])
        self.assertIsNone(row["robust_accuracy_std"])

    def test_duplicate_seed_cannot_inflate_sample_count(self):
        self.add_result()
        self.add_result(name="duplicate")
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertEqual(result["grouped"][0]["n"], 0)
        self.assertIn("Duplicate seed rows", result["errors"][0])

    def test_cross_method_indices_must_match_order(self):
        self.add_result()
        path = self.add_result(name="rank30", rank=30)
        payload = json.loads(path.read_text())
        payload["source_indices"] = [7, 2]
        path.write_text(json.dumps(payload))
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertEqual(len(result["rows"]), 1)
        self.assertIn("Cross-method", result["errors"][0])

    def test_structure_actual_ranks_may_vary_and_pairing_uses_budget(self):
        self.manifest["seeds"] = [42, 43]
        for seed, rank in ((42, [3, 22, 2, 3]), (43, [3, 23, 2, 3])):
            self.add_result(name=f"tr{seed}", seed=seed, group="structure", method="tr", rank=rank,
                            budget_rank=25, parameters=13432, adv=[0, 1])
            self.add_result(name=f"ptr{seed}", seed=seed, group="structure", method="ptr", rank=25,
                            budget_rank=25, parameters=14731)
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(len(result["grouped"]), 2)
        self.assertEqual(len(result["paired_differences"]), 2)
        self.assertEqual(result["paired_differences"][0]["robust_accuracy_delta"], .5)

    def test_loss_pairs_default_at_same_rank(self):
        self.add_result(name="default", group="loss", variant="default", method="caf",
                        budget_rank=None, scan_parameter=None, scan_value=None)
        self.add_result(name="zero_ce", group="loss", variant="clean_ce_0", method="caf",
                        scan_parameter="clean_ce", scan_value=0, adv=[0, 1])
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(len(result["paired_differences"]), 1)
        self.assertEqual(result["paired_differences"][0]["reference_variant"], "default")

    def test_repository_relative_output_path_works_outside_repository(self):
        self.manifest["run_dir"] = "logs/exp033/test"
        path = self.add_result()
        self.tasks[0]["output_path"] = "logs/exp033/test/" + path.name
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(result["status"], "Complete")
        self.assertEqual(result["task_states"][0]["output_path"], str(path))

    def test_repository_artifacts_resolve_from_manifest_root_independent_of_cwd(self):
        repository = self.run_dir / "repository"
        run = repository / "logs/exp033/finished"
        run.mkdir(parents=True)
        (repository / "checkpoints").mkdir()
        checkpoint = repository / "checkpoints/previous.pth"
        checkpoint.write_bytes(b"existing checkpoint")
        generated = run / "case.pth"
        generated.write_bytes(b"generated case")
        manifest = dict(run_dir="logs/exp033/finished")
        with patch("os.getcwd", return_value=str(self.run_dir / "unrelated")):
            self.assertEqual(_resolve_output_path(run, manifest, "checkpoints/previous.pth"), checkpoint)
            self.assertEqual(_resolve_output_path(run, manifest, "case.pth"), generated)
            self.assertEqual(_resolve_output_path(run, manifest, "logs/exp033/finished/case.pth"), generated)
            self.assertEqual(_resolve_output_path(run, manifest, "checkpoints/missing.pth"),
                             repository / "checkpoints/missing.pth")

    def test_historical_raw_matches_new_classifier_whitebox_loss(self):
        self.add_result(name="default", group="loss", variant="default", method="caf", rank=0)
        self.add_result(name="new", group="loss", variant="clean_ce_0", method="caf", rank=0,
                        evaluation="classifier_whitebox", scan_parameter="clean_ce_weight", scan_value=0)
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(len(result["paired_differences"]), 1)

    def test_rank_curves_pair_purification_against_raw(self):
        self.add_result(name="raw", rank=0)
        self.add_result(name="rank25", method="trp_caf", evaluation="nonadaptive_purification", budget_rank=None)
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(len(result["paired_differences"]), 1)
        self.assertEqual(result["paired_differences"][0]["reference_rank"], 0)

    def test_all_five_loss_figures_include_manifest_default_with_missing_scan_field(self):
        # 检查实际曲线对象中的四个横坐标，而不只检查候选列表。
        import matplotlib.pyplot as plt
        defaults = {name: value * 3 for name, value in LOSS_DEFAULTS.items()}
        base = dict(group="loss", method="caf", rank=0, n=1, complete=True,
                    standard_accuracy_mean=1., standard_accuracy_std=None,
                    robust_accuracy_mean=.5, robust_accuracy_std=None)
        values = [dict(base, variant="default", scan_parameter=None, scan_value=None)]
        for name, default in defaults.items():
            for multiplier in (0., .5, 2.):
                values.append(dict(base, variant=f"{name}_{multiplier}", scan_parameter=name,
                                   scan_value=default * multiplier))
        seen = {}

        def inspect(fig, stem, backend):
            parameter = str(stem.name).removeprefix("loss_")
            seen[parameter] = [line.get_xdata().tolist() for line in fig.axes[0].lines]
            backend.close(fig)
            return []

        with patch("rpcf.exp033_report._save_figure", side_effect=inspect):
            _plot_summary(values, self.run_dir, smoke=True, loss_defaults=defaults)
        self.assertEqual(set(seen), set(defaults))
        for name, lines in seen.items():
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(values == [0., defaults[name] * .5, defaults[name], defaults[name] * 2]
                                for values in lines))
        plt.close("all")

    def test_visualization_aggregates_all_samples_and_pairs_with_raw(self):
        path = self.add_result(group="visualize", method="raw", rank=0)
        payload = json.loads(path.read_text())
        raw = payload["rows"][0]
        payload["rows"].append(dict(raw, method="trp_clean", rank=25,
                                    clean_predictions=[0, 0], adv_predictions=[0, 0],
                                    standard_accuracy=.5, robust_accuracy=.5))
        payload["statistics"] = {"trp": {"25": [
            dict(source_index=2, clean_mse=1., adv_mse_to_clean=2., removed_mse=3.),
            dict(source_index=7, clean_mse=3., adv_mse_to_clean=4., removed_mse=5.)]}}
        path.write_text(json.dumps(payload))
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        row = next(row for row in result["rows"] if row["method"] == "trp_clean")
        self.assertEqual(row["clean_mse_mean"], 2.)
        self.assertEqual(row["adv_changed_fraction"], 1.)
        self.assertEqual(row["adv_corrected_fraction"], .5)
        self.assertEqual(row["adv_damaged_fraction"], .5)
        self.assertEqual(row["clean_damaged_fraction"], .5)
        self.assertEqual(result["paired_differences"][0]["reference_method"], "raw")

    def test_missing_or_changed_referenced_artifact_prevents_completion(self):
        path = self.add_result()
        payload = json.loads(path.read_text())
        artifact = self.run_dir / "attack.pth"
        artifact.write_bytes(b"original")
        stat = artifact.stat()
        payload["artifacts"] = [dict(path="attack.pth", size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                     sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())]
        path.write_text(json.dumps(payload))
        self.write_graph()
        artifact.unlink()
        result = summarize(self.run_dir)
        self.assertEqual(result["status"], "Pending")
        self.assertIn("Referenced artifact is missing", result["errors"][0])
        artifact.write_bytes(b"changed contents")
        result = summarize(self.run_dir)
        self.assertEqual(result["status"], "Pending")
        self.assertIn("Referenced artifact content changed", result["errors"][0])
        with self.assertRaisesRegex(ValueError, "Pending"):
            summarize(self.run_dir, strict=True)

    def test_artifact_stat_fast_path_and_identical_content_after_stat_change(self):
        path = self.add_result()
        payload = json.loads(path.read_text())
        artifact = self.run_dir / "attack.pth"
        artifact.write_bytes(b"original")
        stat = artifact.stat()
        payload["artifacts"] = [dict(path="attack.pth", size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                     sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())]
        path.write_text(json.dumps(payload))
        self.write_graph()
        with patch("rpcf.exp033_report._file_sha256", side_effect=AssertionError("unchanged file rehashed")):
            result = summarize(self.run_dir, strict=True)
        self.assertTrue(result["complete"])
        payload["artifacts"][0]["mtime_ns"] -= 1
        path.write_text(json.dumps(payload))
        result = summarize(self.run_dir, strict=True)
        self.assertTrue(result["complete"])

    def test_completed_status_result_hash_is_verified(self):
        path = self.add_result()
        self.write_graph()
        status = self.run_dir / "status/rank42.json"
        status.write_text(json.dumps(dict(status="completed", output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        self.assertTrue(summarize(self.run_dir, strict=True)["complete"])
        path.write_text(path.read_text() + " ")
        result = summarize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertIn("output_sha256", result["errors"][0])

    def test_empty_known_evaluation_output_cannot_complete(self):
        path = self.add_result(group="structure", method="tr_dense", budget_rank=25)
        self.tasks[0].update(kind="structure", method="tr_dense", budget_rank=25)
        payload = json.loads(path.read_text())
        payload["rows"] = []
        path.write_text(json.dumps(payload))
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertIn("Result row signatures mismatch", result["errors"][0])
        with self.assertRaisesRegex(ValueError, "Pending"):
            summarize(self.run_dir, strict=True)

    def test_known_tnp_requires_both_declared_ranks(self):
        self.add_result(group="ablation", method="trp_clean", rank=25, evaluation="nonadaptive_purification")
        self.tasks[0].update(kind="tnp", method="clean", ranks=[25, 30])
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertIn("trp_clean', 30", result["errors"][0])

    def test_exact_known_row_signatures_reject_substituted_method(self):
        self.add_result(group="structure", method="ptr", budget_rank=25, evaluation="nonadaptive_purification")
        self.tasks[0].update(kind="structure", method="tr_dense", budget_rank=25)
        self.write_graph()
        result = summarize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertIn("unexpected", result["errors"][0])

    def test_six_ablation_methods_have_explicit_paired_comparisons(self):
        path = self.add_result(group="ablation", method="clean", rank=0)
        payload = json.loads(path.read_text())
        clean = payload["rows"][0]
        payload["rows"] += [dict(clean, method=method) for method in ("madry", "caf")]
        payload["rows"] += [dict(clean, method=method, rank=rank, evaluation="nonadaptive_purification")
                            for method in ("trp_clean", "trp_madry", "trp_caf") for rank in (25, 30)]
        path.write_text(json.dumps(payload))
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        pairs = result["paired_differences"]
        self.assertEqual(len(pairs), 10)
        self.assertEqual({row["comparison"] for row in pairs}, {
            "madry_vs_clean", "caf_vs_clean", "trp_clean_vs_clean", "trp_madry_vs_madry",
            "trp_caf_vs_caf", "trp_caf_vs_trp_madry"})
        self.assertTrue(all(row["n"] == 1 for row in result["paired_grouped"]))
        self.assertTrue(all(row["reference_rank"] == row["rank"]
                            for row in pairs if row["comparison"] == "trp_caf_vs_trp_madry"))

    def test_timing_rows_do_not_require_accuracy_predictions(self):
        path = self.add_result(group="structure", method="ptr", parameters=14731)
        payload = json.loads(path.read_text())
        payload["rows"] = [dict(group="structure", method="ptr", seed=42, evaluation="timing",
                                budget_rank=25, parameters=14731, decomposition_seconds_mean=.1,
                                total_seconds_mean=.15, peak_memory_bytes=2000000)]
        path.write_text(json.dumps(payload))
        self.write_graph()
        result = summarize(self.run_dir, strict=True)
        self.assertEqual(result["grouped"][0]["total_seconds_mean_mean"], .15)

    def test_render_case_keeps_selection_metadata_and_finite_figures(self):
        clean = torch.sin(torch.linspace(0, 12, 32))[None, None, :]
        adv = clean + .02
        bundle = dict(seed=42, smoke=True, empty_categories=["clean_damaged"], statistics={"sample_num": 2},
                      case_list=[dict(category="success", source_index=2, label=0, sampling_rate=250,
                                      signals=dict(clean=clean, adv=adv, trp25_adv=clean + .005,
                                                   trp25_clean=clean),
                                      predictions=dict(raw=dict(clean=0, adv=1, clean_confidence=.9,
                                                                adv_confidence=.6),
                                                       trp25=dict(clean=0, adv=0, clean_confidence=.9,
                                                                  adv_confidence=.8)))])
        path = self.run_dir / "bundle.pt"
        torch.save(bundle, path)
        paths = render_cases(path, self.run_dir / "figures")
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(Path(path).stat().st_size > 100 for path in paths))
        metadata = json.loads(Path(paths[-1]).read_text())
        self.assertEqual(metadata["cases"][0]["source_index"], 2)
        self.assertEqual(metadata["cases"][0]["channel"], 0)
        self.assertEqual(metadata["empty_categories"], ["clean_damaged"])


if __name__ == "__main__":
    unittest.main()
