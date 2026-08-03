"""通过独立候选 Tensor Ring masked-CV 自动选择样本级 rank。"""

import copy
import time

import torch
import torch.nn.functional as F

from TN.PTR_3d import PTR_3d
from TN.rank_growth.PTR_3d_rank_cv import PTR_3d_rank_cv


class PTR_3d_rank_sweep_cv(torch.nn.Module):
    """
    每个候选 rank 独立执行短程 masked TN 拟合，再完整重训选中的 rank。

    该实现刻意不共享候选间的 cores，避免从 over-complete Tensor Ring 截断时
    受到 gauge、component permutation 和多 bond 误差累积的影响。
    """

    def __init__(
        self,
        target,
        stage,
        max_rank=40,
        dtype="float32",
        loss_fn_str="L2",
        use_TTNF_sampling=False,
        payload=1,
        payload_position="first_core",
        regularization_type="TV",
        dimensions=2,
        regularization_weight=0.0,
        noisy_target=None,
        device="cpu",
        masked_avg_pooling=False,
        sigma_init=0,
        num_iterations=None,
        iterations_for_upsampling=None,
        rank_sweep_candidates=(15, 20, 25, 30, 35, 40),
        rank_sweep_cv_iterations=512,
        rank_sweep_mask_fraction=0.125,
        rank_sweep_mask_blocks=32,
        rank_sweep_one_se_multiplier=1.0,
        rank_sweep_seed=42,
        **unused_kwargs,
    ):
        super().__init__()
        del unused_kwargs
        self.model = "PTR_3d_rank_sweep_cv"
        self.target = target
        self.stage = int(stage)
        self.max_rank = int(max_rank)
        self.num_iterations = int(num_iterations)
        self.iterations_for_upsampling = list(iterations_for_upsampling or [])
        self.rank_sweep_candidates = sorted(
            {int(rank) for rank in rank_sweep_candidates}
        )
        self.rank_sweep_cv_iterations = int(rank_sweep_cv_iterations)
        self.rank_sweep_mask_fraction = float(rank_sweep_mask_fraction)
        self.rank_sweep_mask_blocks = int(rank_sweep_mask_blocks)
        self.rank_sweep_one_se_multiplier = float(
            rank_sweep_one_se_multiplier
        )
        self.rank_sweep_seed = int(rank_sweep_seed)
        self.base_model_kwargs = {
            "stage": self.stage,
            "dtype": dtype,
            "loss_fn_str": loss_fn_str,
            "use_TTNF_sampling": use_TTNF_sampling,
            "payload": payload,
            "payload_position": payload_position,
            "regularization_type": regularization_type,
            "dimensions": dimensions,
            "regularization_weight": regularization_weight,
            "noisy_target": noisy_target,
            "device": device,
            "masked_avg_pooling": masked_avg_pooling,
            "sigma_init": sigma_init,
        }
        self.final_model = None
        self.img = None
        self.selected_rank = None
        self.candidate_validation = []
        self.rank_trajectory = []

        if not self.rank_sweep_candidates:
            raise ValueError("rank_sweep_candidates cannot be empty.")
        if self.rank_sweep_candidates[0] < 1:
            raise ValueError("rank_sweep_candidates must be positive.")
        if self.rank_sweep_candidates[-1] > self.max_rank:
            raise ValueError("rank_sweep_candidates cannot exceed max_rank.")
        if self.rank_sweep_cv_iterations < self.stage:
            raise ValueError("rank_sweep_cv_iterations is too small for all stages.")
        if not 0.0 < self.rank_sweep_mask_fraction < 1.0:
            raise ValueError("rank_sweep_mask_fraction must be in (0, 1).")
        if self.rank_sweep_mask_blocks < 2:
            raise ValueError("rank_sweep_mask_blocks must be at least 2.")
        if self.rank_sweep_one_se_multiplier < 0:
            raise ValueError("rank_sweep_one_se_multiplier must be non-negative.")

    @staticmethod
    def scaled_boundaries(boundaries, total_steps, new_total_steps):
        """按原 coarse-to-fine 比例缩放阶段边界，并保持严格递增。"""
        total_steps = int(total_steps)
        new_total_steps = int(new_total_steps)
        if total_steps <= 0 or new_total_steps <= 1:
            raise ValueError("iteration totals must be positive.")
        raw = [
            int(round(int(boundary) / total_steps * new_total_steps))
            for boundary in boundaries
            if 0 < int(boundary) < total_steps
        ]
        scaled = []
        for boundary in raw:
            boundary = min(new_total_steps - 1, max(1, boundary))
            if scaled and boundary <= scaled[-1]:
                boundary = scaled[-1] + 1
            if boundary < new_total_steps:
                scaled.append(boundary)
        return scaled

    @staticmethod
    def _seed_all(seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _sample_seed(self, target_index):
        index = 0 if target_index is None else int(target_index)
        return self.rank_sweep_seed + index * 1009

    def _candidate_model(self, target, rank, boundaries):
        return PTR_3d_rank_cv(
            target=target,
            max_rank=int(rank),
            num_iterations=self.rank_sweep_cv_iterations,
            iterations_for_upsampling=boundaries,
            rank_ard_min_rank=int(rank),
            rank_ard_energy=1.0,
            rank_ard_sparsity_weight=0.0,
            rank_ard_enable_regrow=False,
            rank_ard_mask_fraction=self.rank_sweep_mask_fraction,
            rank_ard_mask_seed=self.rank_sweep_seed,
            rank_ard_full_refit_steps=0,
            rank_cv_candidate_ranks=[int(rank)],
            rank_cv_one_se_multiplier=0.0,
            rank_cv_holdout_blocks=self.rank_sweep_mask_blocks,
            **self.base_model_kwargs,
        )

    def _final_model(self, target, rank, boundaries):
        return PTR_3d(
            target=target,
            max_rank=int(rank),
            num_iterations=self.num_iterations,
            iterations_for_upsampling=boundaries,
            **self.base_model_kwargs,
        )

    def train(
        self,
        target,
        args,
        target_index=None,
        visualize=False,
        cln_target=None,
        visualize_dir="",
        record_loss=False,
        clf=None,
        logging=None,
    ):
        """只用各候选 TN 的 held-out reconstruction 选择 rank。"""
        del visualize, cln_target, visualize_dir, record_loss
        if clf is not None:
            raise ValueError(
                "PTR_3d_rank_sweep_cv rank inference cannot access a classifier."
            )
        if logging is None:
            class _NullLogger:
                def info(self, *args, **kwargs):
                    return None
            logging = _NullLogger()

        start_time = time.time()
        candidate_boundaries_template = self.scaled_boundaries(
            self.iterations_for_upsampling,
            self.num_iterations,
            self.rank_sweep_cv_iterations,
        )
        candidate_rows = []
        for rank in self.rank_sweep_candidates:
            # 所有候选共享同一样本种子和 holdout blocks，减少初始化方差对
            # rank 比较的干扰；不同 rank 仍各自实例化、优化独立的 TN。
            self._seed_all(self._sample_seed(target_index))
            candidate_args = copy.copy(args)
            candidate_args.num_iterations = self.rank_sweep_cv_iterations
            candidate_boundaries = list(candidate_boundaries_template)
            candidate_args.iterations_for_upsampling = candidate_boundaries
            candidate = self._candidate_model(
                target, rank, candidate_boundaries
            )
            _, elapsed, _ = candidate.train(
                target,
                candidate_args,
                target_index=target_index,
                logging=logging,
            )
            diagnostics = candidate.get_rank_diagnostics()
            validation = diagnostics["rank_cv_selection"]["candidates"][0]
            row = {
                "rank": int(rank),
                "heldout_mse": float(validation["heldout_mse"]),
                "heldout_se": float(validation["heldout_se"]),
                "heldout_block_mses": list(
                    validation["heldout_block_mses"]
                ),
                "candidate_time_sec": float(elapsed),
            }
            candidate_rows.append(row)
            logging.info("Independent masked-CV candidate finished: %s", row)
            del candidate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        selected_rank, best_rank, threshold = (
            PTR_3d_rank_cv.select_rank_one_se(
                candidate_rows,
                multiplier=self.rank_sweep_one_se_multiplier,
            )
        )
        self.selected_rank = int(selected_rank)
        self.candidate_validation = candidate_rows

        # 完整重训与 masked 候选隔离，但不让所选 rank 改变初始化种子。
        final_seed = self._sample_seed(target_index) + 1000003
        self._seed_all(final_seed)
        final_args = copy.copy(args)
        final_boundaries = list(self.iterations_for_upsampling)
        final_args.iterations_for_upsampling = final_boundaries
        self.final_model = self._final_model(
            target, selected_rank, final_boundaries
        )
        reconstruction, _, _ = self.final_model.train(
            target,
            final_args,
            target_index=target_index,
            logging=logging,
        )
        self.img = reconstruction.detach().clone()
        target_device = target.to(self.img.device)
        mse = float(F.mse_loss(self.img, target_device).item())
        variance = float(target_device.var(unbiased=False).item())
        rank_vector = [
            int(self.final_model.tn[index].shape[-1])
            for index in range(2, len(self.final_model.tn) - 1)
        ]
        self.rank_trajectory = [
            {
                "stage_index": 0,
                "resolution": int(target.shape[-1]),
                "rank_vector_before": list(self.rank_sweep_candidates),
                "rank_vector_after": rank_vector,
                "pruned_by_bond": {},
                "regrown_by_bond": {},
                "pruned_count": 0,
                "regrown_count": 0,
                "mse": mse,
                "normalized_residual": mse / max(variance, 1e-12),
                "heldout_mse": float(
                    next(
                        row["heldout_mse"]
                        for row in candidate_rows
                        if row["rank"] == selected_rank
                    )
                ),
                "candidate_validation": candidate_rows,
                "best_validation_rank": int(best_rank),
                "one_se_threshold": float(threshold),
            }
        ]
        elapsed = time.time() - start_time
        logging.info(
            "Independent masked-CV rank selected: rank=%d best=%d threshold=%.8f time=%.4f",
            selected_rank,
            best_rank,
            threshold,
            elapsed,
        )
        return self.img, elapsed, []

    def count_parameters(self):
        return 0 if self.final_model is None else self.final_model.count_parameters()

    def get_rank_diagnostics(self):
        if self.final_model is None or self.selected_rank is None:
            raise RuntimeError("rank sweep has not been trained.")
        ranks = self.rank_trajectory[-1]["rank_vector_after"]
        return {
            "method": self.model,
            "selected_rank": int(self.selected_rank),
            "final_rank_vector": ranks,
            "mean_rank": float(sum(ranks) / len(ranks)),
            "min_rank": int(min(ranks)),
            "max_rank": int(max(ranks)),
            "effective_parameters": int(self.count_parameters()),
            "stage_trajectory": self.rank_trajectory,
            "candidate_validation": self.candidate_validation,
            "rank_inference_signals": [
                "independent_masked_tensor_networks",
                "heldout_time_block_reconstruction",
                "one_standard_error_rule",
            ],
            "uses_labels": False,
            "uses_classifier_logits": False,
        }
