"""通过样本内 masked cross-validation 自动选择最终 Tensor Ring rank。"""

import math

import torch
import torch.nn.functional as F
from opt_einsum import contract

from TN.rank_growth.PTR_3d_rank_ard import PTR_3d_rank_ard


class PTR_3d_rank_cv(PTR_3d_rank_ard):
    """
    在不读取标签或分类器的前提下，用隐藏时间块的重构误差选择样本级 rank。

    优化阶段保持 over-complete Tensor Ring，最终比较若干截断结构在 held-out
    时间块上的误差，并使用 one-standard-error 规则选择最小可接受 rank。
    """

    def __init__(
        self,
        rank_cv_candidate_ranks=(15, 20, 25, 30, 35, 40),
        rank_cv_one_se_multiplier=1.0,
        rank_cv_holdout_blocks=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = "PTR_3d_rank_cv"
        self.rank_cv_candidate_ranks = sorted(
            {int(rank) for rank in rank_cv_candidate_ranks}
        )
        self.rank_cv_one_se_multiplier = float(rank_cv_one_se_multiplier)
        self.rank_cv_holdout_blocks = int(rank_cv_holdout_blocks)
        self.rank_cv_selection = None
        self._rank_cv_holdout_block_ids = []
        self._rank_cv_block_slices = []

        if not self.rank_cv_candidate_ranks:
            raise ValueError("rank_cv_candidate_ranks cannot be empty.")
        if self.rank_cv_candidate_ranks[0] < 1:
            raise ValueError("rank_cv_candidate_ranks must be positive.")
        if self.rank_cv_candidate_ranks[-1] > int(self.max_rank):
            raise ValueError(
                "rank_cv_candidate_ranks cannot exceed max_rank."
            )
        if self.rank_cv_one_se_multiplier < 0:
            raise ValueError("rank_cv_one_se_multiplier must be non-negative.")
        if self.rank_cv_holdout_blocks < 2:
            raise ValueError("rank_cv_holdout_blocks must be at least 2.")
        if self.rank_ard_mask_fraction <= 0:
            raise ValueError(
                "PTR_3d_rank_cv requires rank_ard_mask_fraction > 0."
            )

    def _build_train_masks(self, target_index):
        """
        隐藏完整的连续时间块，并在所有 coarse-to-fine target 上复用同一位置。

        连续块比独立随机点更难由局部插值直接恢复，同时避免不同分辨率阶段
        使用不一致的验证集合。
        """
        time_length = int(self.targets[-1].shape[-1])
        block_count = min(self.rank_cv_holdout_blocks, time_length)
        holdout_count = int(round(block_count * self.rank_ard_mask_fraction))
        holdout_count = min(block_count - 1, max(1, holdout_count))
        seed_offset = 0 if target_index is None else int(target_index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.rank_ard_mask_seed + seed_offset)
        chosen = torch.randperm(block_count, generator=generator)[:holdout_count]
        self._rank_cv_holdout_block_ids = sorted(int(value) for value in chosen)
        self._rank_cv_block_slices = [
            (
                block_id * time_length // block_count,
                (block_id + 1) * time_length // block_count,
            )
            for block_id in self._rank_cv_holdout_block_ids
        ]

        masks = []
        for target in self.targets:
            mask = torch.ones(target.shape, dtype=torch.bool, device=target.device)
            for start, stop in self._rank_cv_block_slices:
                mask[..., start:stop] = False
            if not mask.any():
                raise RuntimeError("Masked-CV training mask has no observed entries.")
            masks.append(mask)
        self._train_masks = masks

    def _rank_keep_indices(self, rank, reso):
        keep_by_bond = {}
        for bond_index in self._eligible_bond_indices(reso):
            scores = self.paired_component_scores(
                self.tn[bond_index],
                self.tn[bond_index + 1],
                eps=self.rank_ard_balance_eps,
            )
            keep_count = min(int(rank), int(scores.numel()))
            order = torch.argsort(scores, descending=True, stable=True)
            keep_by_bond[bond_index] = torch.sort(order[:keep_count]).values
        return keep_by_bond

    def _candidate_reconstruction(self, reso, keep_by_bond):
        cores = [core.detach() for core in self._active_cores(reso)]
        for bond_index, keep in keep_by_bond.items():
            cores[bond_index] = cores[bond_index].index_select(2, keep)
            cores[bond_index + 1] = cores[bond_index + 1].index_select(0, keep)
        output = contract(self.einsum_str, *cores, optimize="greedy")
        return output.reshape([self.C1, self.C2, self.T])

    def _heldout_block_errors(self, recon, target, mask):
        squared_error = (recon - target).square()
        errors = []
        for start, stop in self._rank_cv_block_slices:
            block_mask = (~mask)[..., start:stop]
            if block_mask.any():
                errors.append(
                    squared_error[..., start:stop][block_mask].mean()
                )
        if not errors:
            errors.append(squared_error[~mask].mean())
        return torch.stack(errors)

    @staticmethod
    def select_rank_one_se(candidate_rows, multiplier=1.0):
        """返回 one-SE 阈值内的最低 rank，tie-break 始终偏向低秩。"""
        if not candidate_rows:
            raise ValueError("candidate_rows cannot be empty.")
        rows = sorted(candidate_rows, key=lambda row: int(row["rank"]))
        best = min(
            rows,
            key=lambda row: (float(row["heldout_mse"]), int(row["rank"])),
        )
        threshold = float(best["heldout_mse"]) + float(multiplier) * float(
            best["heldout_se"]
        )
        eligible = [
            row
            for row in rows
            if float(row["heldout_mse"]) <= threshold + 1e-15
        ]
        selected = min(eligible, key=lambda row: int(row["rank"]))
        return int(selected["rank"]), int(best["rank"]), float(threshold)

    def _select_final_rank(self, reso):
        target = self.targets[self.interm_resos.index(reso)]
        mask = self._train_masks[self.interm_resos.index(reso)]
        candidate_rows = []
        keep_by_rank = {}
        with torch.no_grad():
            for rank in self.rank_cv_candidate_ranks:
                keep_by_bond = self._rank_keep_indices(rank, reso)
                recon = self._candidate_reconstruction(reso, keep_by_bond)
                errors = self._heldout_block_errors(recon, target, mask)
                heldout_se = (
                    errors.std(unbiased=True) / math.sqrt(errors.numel())
                    if errors.numel() > 1
                    else torch.zeros((), device=errors.device)
                )
                candidate_rows.append(
                    {
                        "rank": int(rank),
                        "heldout_mse": float(errors.mean().item()),
                        "heldout_se": float(heldout_se.item()),
                        "heldout_block_mses": [
                            float(value) for value in errors.cpu()
                        ],
                    }
                )
                keep_by_rank[int(rank)] = keep_by_bond

        selected_rank, best_rank, threshold = self.select_rank_one_se(
            candidate_rows, multiplier=self.rank_cv_one_se_multiplier
        )
        pruned_by_bond = {}
        for bond_index, keep in keep_by_rank[selected_rank].items():
            pruned_by_bond[str(bond_index)] = self._prune_bond(
                bond_index, keep
            )
        self._rebuild_contraction_path()
        self.rank_cv_selection = {
            "selected_rank": int(selected_rank),
            "best_validation_rank": int(best_rank),
            "one_se_threshold": float(threshold),
            "one_se_multiplier": float(self.rank_cv_one_se_multiplier),
            "holdout_block_ids": list(self._rank_cv_holdout_block_ids),
            "candidates": candidate_rows,
        }
        return pruned_by_bond

    def _stage_adapt(self, stage_index, reso, allow_regrow):
        del allow_regrow
        before = self.bond_rank_vector()
        is_final = stage_index == len(self.interm_resos) - 1
        pruned_by_bond = self._select_final_rank(reso) if is_final else {}
        recon = self.custom_contract_qtt(reso).detach()
        target = self.targets[self.interm_resos.index(reso)]
        mse = float(F.mse_loss(recon, target).item())
        variance = float(target.detach().var(unbiased=False).item())
        normalized_residual = mse / max(variance, self.rank_ard_balance_eps)
        mask = ~self._train_masks[self.interm_resos.index(reso)]
        heldout_mse = (
            float(F.mse_loss(recon[mask], target[mask]).item())
            if mask.any()
            else None
        )
        row = {
            "stage_index": int(stage_index),
            "resolution": int(reso),
            "rank_vector_before": before,
            "rank_vector_after": self.bond_rank_vector(),
            "pruned_by_bond": pruned_by_bond,
            "regrown_by_bond": {},
            "pruned_count": int(sum(pruned_by_bond.values())),
            "regrown_count": 0,
            "mse": mse,
            "normalized_residual": normalized_residual,
            "heldout_mse": heldout_mse,
            "selection_deferred": not is_final,
            "rank_cv_selection": self.rank_cv_selection if is_final else None,
        }
        self.rank_trajectory.append(row)
        return row

    def get_rank_diagnostics(self):
        diagnostics = super().get_rank_diagnostics()
        diagnostics.update(
            {
                "method": self.model,
                "selected_rank": (
                    None
                    if self.rank_cv_selection is None
                    else int(self.rank_cv_selection["selected_rank"])
                ),
                "rank_cv_selection": self.rank_cv_selection,
                "rank_inference_signals": [
                    "masked_time_block_reconstruction",
                    "candidate_core_truncation",
                    "one_standard_error_rule",
                ],
            }
        )
        return diagnostics
