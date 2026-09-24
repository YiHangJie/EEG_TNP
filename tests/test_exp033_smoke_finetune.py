"""EXP-033 独立 smoke 验证前缀包装器；不执行训练。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rpcf
from rpcf import exp033_smoke_finetune as smoke
from rpcf.exp033_common import read_json, write_json


def options(root, epochs=1, is_smoke=True):
    write_json(root / "manifest.json", {"smoke": is_smoke})
    output = root / "training" / "variant" / "attempt001" / "checkpoint.pth"
    return ["--epochs", str(epochs), "--online_train_sample_num", "2",
            "--max_cache_batches", "1", "--output_checkpoint", str(output)]


class SmokeFineTuneTest(unittest.TestCase):
    def test_options_allow_only_smoke_manifest_and_bounded_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = options(root / "smoke")
            self.assertEqual(smoke.smoke_options(valid), Path(valid[-1]).parent)
            with self.assertRaises(ValueError):
                smoke.smoke_options(options(root / "formal", is_smoke=False))
            with self.assertRaises(ValueError):
                smoke.smoke_options(options(root / "long", epochs=100))
            for name, value in (("--online_train_sample_num", "512"), ("--max_cache_batches", "8")):
                invalid = list(valid)
                invalid[invalid.index(name) + 1] = value
                with self.assertRaises(ValueError):
                    smoke.smoke_options(invalid)
            (root / "smoke" / "manifest.json").unlink()
            with self.assertRaises(ValueError):
                smoke.smoke_options(valid)

    def test_duplicate_options_cannot_override_the_guarded_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = options(root / "smoke")
            formal = options(root / "formal", is_smoke=False)
            for invalid in (valid + ["--epochs", "100"], valid + ["--epochs=100"],
                            valid + ["--output_checkpoint", formal[-1]]):
                with self.subTest(argv=invalid), self.assertRaises(ValueError):
                    smoke.smoke_options(invalid)

    def test_wrapper_only_replaces_validation_with_prefix_and_records_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            train, validation, test = object(), list(range(6)), object()
            original = Mock(return_value=(train, validation, test, "split/path"))
            original_epoch = Mock()
            module = SimpleNamespace(prepare_subject_fold=original, train_epoch=original_epoch)

            def main():
                actual_train, limited, actual_test, split = module.prepare_subject_fold("thubenchmark", fold_id=0)
                self.assertIs(actual_train, train)
                self.assertIs(actual_test, test)
                self.assertIs(limited.dataset, validation)
                self.assertEqual(limited.indices, [0, 1])
                self.assertEqual(list(limited), [0, 1])
                self.assertEqual(split, "split/path")

            module.main = Mock(side_effect=main)
            with patch.object(rpcf, "finetune", module, create=True), patch("sys.argv", ["smoke", *args]):
                smoke.main()
            self.assertIs(module.prepare_subject_fold, original)
            self.assertIs(module.train_epoch, original_epoch)
            original.assert_called_once_with("thubenchmark", fold_id=0)
            module.main.assert_called_once_with()
            self.assertEqual(validation, list(range(6)))
            record = read_json(Path(args[-1]).parent / "smoke_validation.json")
            self.assertTrue(record["smoke"])
            self.assertEqual(record["selections"], [dict(original_sample_num=6, sample_num=2,
                                                        validation_indices=[0, 1], split_path="split/path")])

    def test_wrapper_restores_module_function_after_training_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            original = Mock(return_value=(object(), [0, 1, 2], object(), "split"))
            original_epoch = Mock()
            module = SimpleNamespace(prepare_subject_fold=original, train_epoch=original_epoch)

            def fail():
                module.prepare_subject_fold(seed=42)
                raise RuntimeError("simulated training error")

            module.main = Mock(side_effect=fail)
            with patch.object(rpcf, "finetune", module, create=True), patch("sys.argv", ["smoke", *args]):
                with self.assertRaisesRegex(RuntimeError, "simulated training"):
                    smoke.main()
            self.assertIs(module.prepare_subject_fold, original)
            self.assertIs(module.train_epoch, original_epoch)
            original.assert_called_once_with(seed=42)

    def test_cache_epoch_consumes_only_first_batch_and_preserves_arguments_and_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            original = Mock(return_value=(object(), [0, 1, 2], object(), "split"))
            model, optimizer, rank_weights = object(), object(), object()
            batches = [object(), object(), object()]
            consumed = []
            metrics = {"loss": .25, "clean_ce": .1}

            def loader():
                for index, batch in enumerate(batches):
                    consumed.append(index)
                    yield batch

            def epoch(actual_model, limited, *positional, **kwargs):
                self.assertIs(actual_model, model)
                self.assertEqual(positional, (optimizer, "cpu", rank_weights))
                self.assertEqual(kwargs, {"use_cached_adv": False})
                self.assertEqual(list(limited), batches[:1])
                return metrics

            original_epoch = Mock(side_effect=epoch)
            module = SimpleNamespace(prepare_subject_fold=original, train_epoch=original_epoch)

            def main():
                module.prepare_subject_fold()
                actual = module.train_epoch(model, loader(), optimizer, "cpu", rank_weights, use_cached_adv=False)
                self.assertIs(actual, metrics)

            module.main = Mock(side_effect=main)
            with patch.object(rpcf, "finetune", module, create=True), patch("sys.argv", ["smoke", *args]):
                smoke.main()
            self.assertEqual(consumed, [0])
            original_epoch.assert_called_once()
            self.assertIs(module.prepare_subject_fold, original)
            self.assertIs(module.train_epoch, original_epoch)
            audit = read_json(Path(args[-1]).parent / "smoke_validation.json")
            self.assertEqual(audit["cache_batch_limit"], 1)
            self.assertTrue(audit["cache_batch_limit_enforced"])
            self.assertEqual(audit["selections"][0]["validation_indices"], [0, 1])

    def test_cache_epoch_exception_restores_both_functions_without_success_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            original = Mock(return_value=(object(), [0, 1, 2], object(), "split"))
            consumed = []

            def loader():
                for index in range(3):
                    consumed.append(index)
                    yield index

            def fail(model, limited, *args, **kwargs):
                self.assertEqual(list(limited), [0])
                raise RuntimeError("simulated cache epoch error")

            original_epoch = Mock(side_effect=fail)
            module = SimpleNamespace(prepare_subject_fold=original, train_epoch=original_epoch)

            def main():
                module.prepare_subject_fold()
                module.train_epoch(object(), loader())

            module.main = Mock(side_effect=main)
            with patch.object(rpcf, "finetune", module, create=True), patch("sys.argv", ["smoke", *args]):
                with self.assertRaisesRegex(RuntimeError, "cache epoch error"):
                    smoke.main()
            self.assertEqual(consumed, [0])
            self.assertIs(module.prepare_subject_fold, original)
            self.assertIs(module.train_epoch, original_epoch)
            audit = read_json(Path(args[-1]).parent / "smoke_validation.json")
            self.assertFalse(audit.get("cache_batch_limit_enforced", False))

    def test_empty_validation_rejected_and_function_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            original = Mock(return_value=(object(), [], object(), "split"))
            original_epoch = Mock()
            module = SimpleNamespace(prepare_subject_fold=original, train_epoch=original_epoch)
            module.main = Mock(side_effect=lambda: module.prepare_subject_fold())
            with patch.object(rpcf, "finetune", module, create=True), patch("sys.argv", ["smoke", *args]):
                with self.assertRaisesRegex(ValueError, "Empty validation"):
                    smoke.main()
            self.assertIs(module.prepare_subject_fold, original)
            self.assertIs(module.train_epoch, original_epoch)


if __name__ == "__main__":
    unittest.main()
