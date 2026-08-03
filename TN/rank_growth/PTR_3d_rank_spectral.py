"""基于 two-site SVD spectral rounding 的 EEG_TNP 自动定秩。"""

import math

import torch
import torch.nn.functional as F

from TN.rank_growth.PTR_3d_rank_ard import PTR_3d_rank_ard


class PTR_3d_rank_spectral(PTR_3d_rank_ard):
    """
    合并相邻 Tensor Ring cores，并用未知噪声最优硬阈值确定共享 bond rank。

    与逐 component norm 裁剪不同，two-site SVD 在联合矩阵上给出局部最优的
    Frobenius 低秩近似，因此不依赖共享维上的 gauge scaling 或 component 排列。
    """

    def __init__(
        self,
        rank_spectral_min_rank=15,
        rank_spectral_max_rank=None,
        rank_spectral_sweeps=1,
        rank_spectral_eps=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = "PTR_3d_rank_spectral"
        self.rank_spectral_min_rank = int(rank_spectral_min_rank)
        self.rank_spectral_max_rank = int(
            self.max_rank
            if rank_spectral_max_rank is None
            else rank_spectral_max_rank
        )
        self.rank_spectral_sweeps = int(rank_spectral_sweeps)
        self.rank_spectral_eps = float(rank_spectral_eps)
        self.spectral_rounding = []

        if not 1 <= self.rank_spectral_min_rank <= int(self.max_rank):
            raise ValueError("rank_spectral_min_rank must be in [1, max_rank].")
        if not (
            self.rank_spectral_min_rank
            <= self.rank_spectral_max_rank
            <= int(self.max_rank)
        ):
            raise ValueError(
                "rank_spectral_max_rank must be in [min_rank, max_rank]."
            )
        if self.rank_spectral_sweeps < 1:
            raise ValueError("rank_spectral_sweeps must be positive.")
        if self.rank_spectral_eps <= 0:
            raise ValueError("rank_spectral_eps must be positive.")

    @staticmethod
    def optimal_hard_threshold_coefficient(beta):
        """Gavish–Donoho unknown-noise optimal hard threshold 近似系数。"""
        beta = float(beta)
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1].")
        return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43

    @classmethod
    def select_spectral_rank(
        cls,
        singular_values,
        matrix_rows,
        matrix_cols,
        min_rank,
        max_rank,
    ):
        """按 unknown-noise OHT 选秩，并返回完整阈值诊断。"""
        values = torch.as_tensor(singular_values).detach().float().view(-1)
        if values.numel() == 0:
            raise ValueError("singular_values cannot be empty.")
        if matrix_rows < 1 or matrix_cols < 1:
            raise ValueError("matrix dimensions must be positive.")
        limit = min(int(max_rank), values.numel())
        floor = min(max(1, int(min_rank)), limit)
        beta = min(matrix_rows, matrix_cols) / max(matrix_rows, matrix_cols)
        coefficient = cls.optimal_hard_threshold_coefficient(beta)
        median = values.median()
        threshold = coefficient * median
        raw_rank = int((values > threshold).sum().item())
        selected_rank = min(limit, max(floor, raw_rank))
        return {
            "selected_rank": selected_rank,
            "raw_rank": raw_rank,
            "beta": float(beta),
            "coefficient": float(coefficient),
            "median_singular_value": float(median.item()),
            "threshold": float(threshold.item()),
        }

    @staticmethod
    def _merge_two_site(left_core, right_core):
        if left_core.shape[-1] != right_core.shape[0]:
            raise ValueError("Adjacent Tensor Ring cores have mismatched bond rank.")
        merged = torch.einsum("aib,bjc->aijc", left_core, right_core)
        rows = int(left_core.shape[0] * left_core.shape[1])
        cols = int(right_core.shape[1] * right_core.shape[2])
        return merged.reshape(rows, cols), rows, cols

    def _round_two_site(self, bond_index, sweep_index):
        left = self.tn[bond_index].detach()
        right = self.tn[bond_index + 1].detach()
        old_rank = int(left.shape[-1])
        matrix, rows, cols = self._merge_two_site(left, right)
        with torch.no_grad():
            u, singular_values_all, vh = torch.linalg.svd(
                matrix, full_matrices=False
            )
        # two-site 矩阵的精确秩不超过原共享 bond；忽略数值 SVD 产生的额外近零值。
        usable = min(old_rank, singular_values_all.numel())
        singular_values = singular_values_all[:usable]
        selection = self.select_spectral_rank(
            singular_values,
            matrix_rows=rows,
            matrix_cols=cols,
            min_rank=self.rank_spectral_min_rank,
            max_rank=min(self.rank_spectral_max_rank, old_rank),
        )
        selected_rank = int(selection["selected_rank"])
        left_new = u[:, :selected_rank].reshape(
            left.shape[0], left.shape[1], selected_rank
        )
        right_new = (
            singular_values_all[:selected_rank, None] * vh[:selected_rank]
        ).reshape(selected_rank, right.shape[1], right.shape[2])
        self.tn[bond_index] = left_new.detach().clone()
        self.tn[bond_index + 1] = right_new.detach().clone()

        total_energy = singular_values.square().sum().clamp_min(
            self.rank_spectral_eps
        )
        retained_energy = (
            singular_values[:selected_rank].square().sum() / total_energy
        )
        row = {
            "sweep_index": int(sweep_index),
            "bond_index": int(bond_index),
            "old_rank": old_rank,
            **selection,
            "retained_energy": float(retained_energy.item()),
            "singular_values": [float(value) for value in singular_values.cpu()],
        }
        self.spectral_rounding.append(row)
        return row

    def _spectral_round(self, reso):
        before = self.bond_rank_vector()
        pruned_by_bond = {}
        rows = []
        for sweep_index in range(self.rank_spectral_sweeps):
            for bond_index in self._eligible_bond_indices(reso):
                row = self._round_two_site(bond_index, sweep_index)
                rows.append(row)
                key = str(bond_index)
                pruned_by_bond[key] = pruned_by_bond.get(key, 0) + (
                    int(row["old_rank"]) - int(row["selected_rank"])
                )
        # 最终分辨率只使用 self.tn；同步 shape 可避免后续诊断误用旧 tn_tmp。
        self.tn_tmp = [torch.zeros_like(core.detach()) for core in self.tn]
        self._rebuild_contraction_path()
        return before, pruned_by_bond, rows

    def _stage_adapt(self, stage_index, reso, allow_regrow):
        del allow_regrow
        is_final = stage_index == len(self.interm_resos) - 1
        if is_final:
            before, pruned_by_bond, spectral_rows = self._spectral_round(reso)
        else:
            before = self.bond_rank_vector()
            pruned_by_bond = {}
            spectral_rows = []

        recon = self.custom_contract_qtt(reso).detach()
        target = self.targets[self.interm_resos.index(reso)]
        mse = float(F.mse_loss(recon, target).item())
        variance = float(target.detach().var(unbiased=False).item())
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
            "normalized_residual": mse / max(
                variance, self.rank_spectral_eps
            ),
            "heldout_mse": None,
            "selection_deferred": not is_final,
            "spectral_rounding": spectral_rows,
        }
        self.rank_trajectory.append(row)
        return row

    def get_rank_diagnostics(self):
        diagnostics = super().get_rank_diagnostics()
        diagnostics.update(
            {
                "method": self.model,
                "spectral_rounding": self.spectral_rounding,
                "rank_inference_signals": [
                    "two_site_singular_values",
                    "unknown_noise_optimal_hard_threshold",
                    "local_frobenius_optimal_rounding",
                ],
            }
        )
        return diagnostics
