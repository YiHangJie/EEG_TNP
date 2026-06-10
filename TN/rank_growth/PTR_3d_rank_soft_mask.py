import math

import torch
import torch.nn.functional as F
from opt_einsum import contract

from TN.PTR_3d import PTR_3d


class PTR_3d_rank_soft_mask(PTR_3d):
    """
    PTR_3d 的可微 soft-rank 版本。

    该模型保留 max_rank 的完整 Tensor Ring 参数容器，通过一个可学习的
    连续 rho 生成 soft-prefix mask。优化目标只包含 MSE 重构误差和有效
    rank 约束，不引入分类器语义项或频域残差假设。
    """

    def __init__(
        self,
        target,
        stage,
        max_rank=256,
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
        rank_soft_mask_init_rank=15.0,
        rank_soft_mask_temperature=1.0,
        rank_soft_mask_weight=0.003,
    ):
        super().__init__(
            target=target,
            stage=stage,
            max_rank=max_rank,
            dtype=dtype,
            loss_fn_str=loss_fn_str,
            use_TTNF_sampling=use_TTNF_sampling,
            payload=payload,
            payload_position=payload_position,
            regularization_type=regularization_type,
            dimensions=dimensions,
            regularization_weight=regularization_weight,
            noisy_target=noisy_target,
            device=device,
            masked_avg_pooling=masked_avg_pooling,
            sigma_init=sigma_init,
            num_iterations=num_iterations,
            iterations_for_upsampling=iterations_for_upsampling,
        )

        self.model = "PTR_3d_rank_soft_mask"
        self.rank_soft_mask_init_rank = float(rank_soft_mask_init_rank)
        self.rank_soft_mask_temperature = float(rank_soft_mask_temperature)
        self.rank_soft_mask_weight = float(rank_soft_mask_weight)
        if self.max_rank <= 1:
            raise ValueError("PTR_3d_rank_soft_mask requires max_rank > 1.")
        if self.rank_soft_mask_temperature <= 0:
            raise ValueError("rank_soft_mask_temperature must be positive.")

        raw_rho = self._init_raw_rho(self.rank_soft_mask_init_rank)
        self.raw_rho = torch.nn.Parameter(torch.tensor(raw_rho, dtype=self.dtype, device=self.device))
        self.last_soft_mask_stats = {}

    def _init_raw_rho(self, init_rank):
        """将初始有效 rank 映射到 unconstrained raw rho。"""
        init_rank = min(max(float(init_rank), 1.0), float(self.max_rank))
        scaled = (init_rank - 1.0) / max(float(self.max_rank - 1), 1.0)
        scaled = min(max(scaled, 1e-4), 1.0 - 1e-4)
        return math.log(scaled / (1.0 - scaled))

    def set_optimizer(self, current_grid, lr):
        params = {i: t for i, t in enumerate(self.tn)}
        torch_params = torch.nn.ParameterDict({
            f"{i:03d}": torch.nn.Parameter(initial).to(self.device)
            for i, initial in params.items()
        })
        for i in range(len(self.tn)):
            torch_params[f"{i:03d}"].requires_grad = True

        self.raw_rho.requires_grad = True
        optimizer_params = list(torch_params.values()) + [self.raw_rho]
        optimizer = torch.optim.Adam([{"params": optimizer_params, "lr": lr}])
        self.tn = [p for _, p in torch_params.items()]
        return optimizer

    def _rho(self):
        return 1.0 + float(self.max_rank - 1) * torch.sigmoid(self.raw_rho)

    def _soft_rank_gates(self, dtype=None, device=None):
        dtype = dtype or self.dtype
        device = device or self.device
        rho = self._rho().to(device=device, dtype=dtype)
        positions = torch.arange(1, int(self.max_rank) + 1, dtype=dtype, device=device)
        gates = torch.sigmoid((rho - positions) / float(self.rank_soft_mask_temperature))
        effective_rank = gates.sum()
        rank_cost = effective_rank / float(self.max_rank)
        return gates, effective_rank, rho, rank_cost

    def _mask_core(self, core, core_index, gates):
        """对时间相关 bond 施加 sqrt(g) soft-prefix mask，空间 core 保持不变。"""
        if core_index < 2:
            return core

        sqrt_gates = gates.sqrt().to(dtype=core.dtype, device=core.device)
        if core_index == 2:
            return core * sqrt_gates.view(1, 1, -1)
        if core_index == len(self.tn) - 1:
            return core * sqrt_gates.view(-1, 1, 1)
        return core * sqrt_gates.view(-1, 1, 1) * sqrt_gates.view(1, 1, -1)

    def custom_contract_qtt(self, output_reso):
        contract_core_num = int(math.log2(output_reso)) + 2
        cores = self.tn[:contract_core_num] + self.tn_tmp[contract_core_num:]
        gates, effective_rank, rho, rank_cost = self._soft_rank_gates(
            dtype=cores[0].dtype,
            device=cores[0].device,
        )
        masked_cores = [
            self._mask_core(core, core_index, gates)
            for core_index, core in enumerate(cores)
        ]
        output = contract(self.einsum_str, *masked_cores, optimize=self.path)
        self.last_soft_mask_stats = {
            "rho": rho,
            "effective_rank": effective_rank,
            "rank_cost": rank_cost,
        }
        return output.reshape([self.C1, self.C2, self.T])

    def forward(self, reso):
        assert reso in self.interm_resos, f"Invalid intermediate resolution, should be one of {self.interm_resos}"

        reso_index = self.interm_resos.index(reso)
        recon = self.custom_contract_qtt(output_reso=reso)
        self.img = recon.detach().clone()
        loss_recon = F.mse_loss(recon, self.targets[reso_index])
        rank_cost = self.last_soft_mask_stats["rank_cost"]
        loss = loss_recon + self.rank_soft_mask_weight * rank_cost
        self.last_soft_mask_stats = {
            **self.last_soft_mask_stats,
            "loss_recon": loss_recon.detach(),
            "loss_rank": (self.rank_soft_mask_weight * rank_cost).detach(),
            "loss_total": loss.detach(),
        }
        return loss, float(loss_recon.detach().item())

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
        if logging is None:
            class _NullLogger:
                def info(self, *args, **kwargs):
                    return None

            logging = _NullLogger()
        return super().train(
            target=target,
            args=args,
            target_index=target_index,
            visualize=visualize,
            cln_target=cln_target,
            visualize_dir=visualize_dir,
            record_loss=record_loss,
            clf=clf,
            logging=logging,
        )

    def get_soft_mask_stats(self):
        gates, effective_rank, rho, rank_cost = self._soft_rank_gates()
        return {
            "rho": float(rho.detach().cpu().item()),
            "effective_rank": float(effective_rank.detach().cpu().item()),
            "rank_cost": float(rank_cost.detach().cpu().item()),
            "rank_soft_mask_weight": float(self.rank_soft_mask_weight),
            "rank_soft_mask_temperature": float(self.rank_soft_mask_temperature),
            "rank_soft_mask_init_rank": float(self.rank_soft_mask_init_rank),
            "gates": gates.detach().cpu(),
        }

    def count_parameters(self):
        """返回 soft effective rank 下的近似有效参数量。"""
        stats = self.get_soft_mask_stats()
        effective_rank = float(stats["effective_rank"])
        total = 0.0
        for core_index, core in enumerate(self.tn):
            if core_index < 2:
                total += float(core.numel())
            elif core_index == 2:
                total += float(core.shape[0] * core.shape[1]) * effective_rank
            elif core_index == len(self.tn) - 1:
                total += effective_rank * float(core.shape[1] * core.shape[2])
            else:
                total += effective_rank * float(core.shape[1]) * effective_rank
        return total
