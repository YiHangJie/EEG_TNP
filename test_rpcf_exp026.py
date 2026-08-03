import unittest

import torch
import torch.nn.functional as F

from models.model_args import get_model_args
from models.registry import MODEL_CLASSES
from rpcf.feature_alignment import (
    ClassBalancedBatchSampler,
    PenultimateFeatureAdapter,
    biased_rbf_mmd,
    class_conditional_mmd_loss,
    prototype_alignment_losses,
    stratified_cache_split_indices,
    trial_contrastive_loss,
)


class Exp026FeatureAlignmentTest(unittest.TestCase):
    def test_all_backbone_adapters_preserve_logits(self):
        info = {
            "chunk_size": 1500,
            "num_electrodes": 64,
            "num_classes": 40,
            "sampling_rate": 250,
        }
        x = torch.randn(2, 1, 64, 1500)
        for name in (
            "eegnet",
            "tsception",
            "atcnet",
            "conformer",
            "tcnet",
            "deepconvnet",
        ):
            with self.subTest(model=name):
                model = MODEL_CLASSES[name](
                    **get_model_args(name, "thubenchmark", info)
                ).eval()
                adapter = PenultimateFeatureAdapter(name, model)
                with torch.no_grad():
                    expected = model(x)
                    logits, features = adapter.forward_with_features(x)
                self.assertTrue(torch.allclose(logits, expected, atol=1e-6))
                self.assertEqual(tuple(features.shape[:1]), (2,))
                self.assertGreater(features.size(1), 0)
                self.assertTrue(
                    torch.allclose(
                        features.norm(dim=1), torch.ones(2), atol=1e-5
                    )
                )

    def test_biased_mmd_handles_equal_shifted_and_singleton_inputs(self):
        x = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
        equal = biased_rbf_mmd(x, x)
        shifted = biased_rbf_mmd(x, -x)
        singleton = biased_rbf_mmd(x[:1], -x[:1])
        self.assertAlmostEqual(equal.item(), 0.0, places=6)
        self.assertGreater(shifted.item(), equal.item())
        self.assertTrue(torch.isfinite(singleton))

    def test_cmmd_is_rank_weighted_and_clean_anchor_can_be_detached(self):
        clean_source = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            requires_grad=True,
        )
        clean = F.normalize(clean_source, dim=1).detach()
        pur = torch.stack([clean.clone(), -clean.clone()], dim=1).requires_grad_()
        labels = torch.tensor([0, 0, 1, 1])
        loss = class_conditional_mmd_loss(
            clean,
            pur,
            pur,
            labels,
            rank_weights=torch.tensor([0.75, 0.25]),
        )
        loss.backward()
        self.assertIsNone(clean_source.grad)
        self.assertIsNotNone(pur.grad)
        self.assertGreater(loss.item(), 0.0)

    def test_prototype_alignment_and_margin(self):
        clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        identical = clean.unsqueeze(1)
        alignment, margin = prototype_alignment_losses(
            clean,
            identical,
            torch.tensor([0, 1]),
            torch.tensor([1.0]),
            margin=1.0,
        )
        self.assertAlmostEqual(alignment.item(), 0.0, places=6)
        self.assertAlmostEqual(margin.item(), 0.0, places=6)
        collapsed = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])
        alignment, margin = prototype_alignment_losses(
            clean,
            collapsed,
            torch.tensor([0, 1]),
            torch.tensor([1.0]),
            margin=1.0,
        )
        self.assertGreater(alignment.item(), 0.0)
        self.assertAlmostEqual(margin.item(), 1.0, places=6)

    def test_trial_contrastive_uses_same_trial_positive_group(self):
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        adv = anchors.clone().requires_grad_()
        adv_pur = torch.stack([anchors, anchors], dim=1).requires_grad_()
        loss, skipped = trial_contrastive_loss(
            anchors.detach(), adv, adv_pur, temperature=1.0
        )
        candidates = torch.cat([adv.detach().unsqueeze(1), adv_pur.detach()], dim=1)
        logits = torch.einsum("bd,jvd->bjv", anchors, candidates)
        expected = -(
            logits[torch.arange(2), torch.arange(2)].mean(dim=1)
            - torch.logsumexp(logits.reshape(2, -1), dim=1)
        ).mean()
        self.assertFalse(skipped)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertIsNotNone(adv.grad)
        self.assertIsNotNone(adv_pur.grad)

        one_loss, one_skipped = trial_contrastive_loss(
            anchors[:1], adv[:1], adv_pur[:1]
        )
        self.assertTrue(one_skipped)
        self.assertEqual(one_loss.item(), 0.0)

    def test_class_balanced_sampler_is_deterministic_and_balanced(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        sampler_a = ClassBalancedBatchSampler(
            labels, classes_per_batch=4, samples_per_class=2, seed=42
        )
        sampler_b = ClassBalancedBatchSampler(
            labels, classes_per_batch=4, samples_per_class=2, seed=42
        )
        batches_a = list(sampler_a)
        batches_b = list(sampler_b)
        self.assertEqual(batches_a, batches_b)
        for batch in batches_a:
            counts = torch.bincount(labels[batch], minlength=4)
            self.assertEqual(sorted(counts.tolist()), [2, 2, 2, 2])
        sampler_a.set_epoch(1)
        self.assertNotEqual(batches_a, list(sampler_a))

    def test_stratified_holdout_keeps_singletons_in_training(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2])
        train, holdout = stratified_cache_split_indices(labels, 0.2, seed=7)
        self.assertIn(5, train)
        self.assertNotIn(5, holdout)
        self.assertTrue(set(train).isdisjoint(holdout))
        self.assertEqual(set(train) | set(holdout), set(range(len(labels))))


if __name__ == "__main__":
    unittest.main()
