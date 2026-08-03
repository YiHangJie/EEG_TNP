"""EEG_TNP 的样本级、bond-wise 自动定秩 Tensor Ring。"""

import math
import time

import torch
import torch.nn.functional as F
from opt_einsum import contract, contract_path

from TN.PTR_3d import PTR_3d


class PTR_3d_rank_ard(PTR_3d):
    """通过相邻 core 的成对贡献在分辨率阶段边界自动裁剪 Tensor Ring bond。"""

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
        rank_ard_min_rank=15,
        rank_ard_energy=0.98,
        rank_ard_sparsity_weight=0.0,
        rank_ard_balance_eps=1e-12,
        rank_ard_enable_regrow=False,
        rank_ard_regrow_step=2,
        rank_ard_regrow_residual_ratio=0.03,
        rank_ard_regrow_scale=0.05,
        rank_ard_mask_fraction=0.0,
        rank_ard_mask_seed=42,
        rank_ard_full_refit_steps=0,
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
        self.model = "PTR_3d_rank_ard"
        self.rank_ard_min_rank = int(rank_ard_min_rank)
        self.rank_ard_energy = float(rank_ard_energy)
        self.rank_ard_sparsity_weight = float(rank_ard_sparsity_weight)
        self.rank_ard_balance_eps = float(rank_ard_balance_eps)
        self.rank_ard_enable_regrow = bool(rank_ard_enable_regrow)
        self.rank_ard_regrow_step = int(rank_ard_regrow_step)
        self.rank_ard_regrow_residual_ratio = float(
            rank_ard_regrow_residual_ratio
        )
        self.rank_ard_regrow_scale = float(rank_ard_regrow_scale)
        self.rank_ard_mask_fraction = float(rank_ard_mask_fraction)
        self.rank_ard_mask_seed = int(rank_ard_mask_seed)
        self.rank_ard_full_refit_steps = int(rank_ard_full_refit_steps)
        self.rank_trajectory = []
        self._train_masks = None
        self._current_reso = self.init_reso

        if not 1 <= self.rank_ard_min_rank <= int(self.max_rank):
            raise ValueError("rank_ard_min_rank must be in [1, max_rank].")
        if not 0.0 < self.rank_ard_energy <= 1.0:
            raise ValueError("rank_ard_energy must be in (0, 1].")
        if not 0.0 <= self.rank_ard_mask_fraction < 1.0:
            raise ValueError("rank_ard_mask_fraction must be in [0, 1).")
        if self.rank_ard_balance_eps <= 0:
            raise ValueError("rank_ard_balance_eps must be positive.")

    @staticmethod
    def paired_component_scores(left_core, right_core, eps=1e-12):
        """
        返回相邻 core 每个共享 component 的 gauge-invariant 成对贡献。

        若一侧乘以 ``a``、另一侧乘以 ``1/a``，两侧 Frobenius 范数乘积不变。
        """
        if left_core.ndim != 3 or right_core.ndim != 3:
            raise ValueError("Tensor Ring cores must be three-dimensional.")
        if left_core.shape[-1] != right_core.shape[0]:
            raise ValueError(
                "Adjacent core bond mismatch: "
                f"{left_core.shape[-1]} != {right_core.shape[0]}."
            )
        left_norm = torch.linalg.vector_norm(
            left_core, ord=2, dim=(0, 1)
        ).clamp_min(eps)
        right_norm = torch.linalg.vector_norm(
            right_core, ord=2, dim=(1, 2)
        ).clamp_min(eps)
        return left_norm * right_norm

    @classmethod
    def relevance_precision(cls, left_core, right_core, eps=1e-12):
        """以成对贡献的逆量作为 ARD-like relevance precision。"""
        scores = cls.paired_component_scores(left_core, right_core, eps=eps)
        scale = scores.square().mean().clamp_min(eps)
        return scale / scores.square().clamp_min(eps)

    @staticmethod
    def select_components(scores, min_rank, energy):
        """选择累计成对能量达到阈值的最小 component 集合。"""
        scores = torch.as_tensor(scores).detach().float().view(-1)
        if scores.numel() == 0:
            raise ValueError("scores cannot be empty.")
        min_rank = min(max(1, int(min_rank)), scores.numel())
        order = torch.argsort(scores, descending=True, stable=True)
        component_energy = scores[order].square()
        total = component_energy.sum()
        if not torch.isfinite(total) or float(total) <= 0:
            keep_count = min_rank
        else:
            cumulative = torch.cumsum(component_energy, dim=0) / total
            threshold_index = int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(float(energy), device=cumulative.device),
                    right=False,
                ).item()
            )
            keep_count = max(min_rank, min(scores.numel(), threshold_index + 1))
        # 排序仅用于选择，恢复原 component 顺序可保持可复现的 contraction。
        return torch.sort(order[:keep_count]).values

    def variable_bond_indices(self):
        """时间 QTT 的可变 bond；空间 core 与环闭合 bond 保持原尺寸。"""
        return list(range(2, len(self.tn) - 1))

    def bond_rank_vector(self):
        return [int(self.tn[index].shape[-1]) for index in self.variable_bond_indices()]

    def _eligible_bond_indices(self, reso):
        contract_core_num = int(math.log2(int(reso))) + 2
        return [
            index
            for index in self.variable_bond_indices()
            if index + 1 < contract_core_num
        ]

    def _active_cores(self, output_reso):
        contract_core_num = int(math.log2(int(output_reso))) + 2
        return self.tn[:contract_core_num] + self.tn_tmp[contract_core_num:]

    def custom_contract_qtt(self, output_reso):
        output = contract(
            self.einsum_str,
            *self._active_cores(output_reso),
            optimize=self.path,
        )
        return output.reshape([self.C1, self.C2, self.T])

    def _rebuild_contraction_path(self):
        with torch.no_grad():
            cores = self._active_cores(self.end_reso)
            self.path, _ = contract_path(
                self.einsum_str, *cores, optimize="greedy"
            )

    def _balance_bond(self, bond_index):
        """对共享 bond 做 reciprocal rescaling，避免单侧范数造成虚假 relevance。"""
        left = self.tn[bond_index]
        right = self.tn[bond_index + 1]
        eps = self.rank_ard_balance_eps
        with torch.no_grad():
            left_norm = torch.linalg.vector_norm(
                left, ord=2, dim=(0, 1)
            ).clamp_min(eps)
            right_norm = torch.linalg.vector_norm(
                right, ord=2, dim=(1, 2)
            ).clamp_min(eps)
            left_scale = torch.sqrt(right_norm / left_norm)
            right_scale = torch.reciprocal(left_scale)
            left.mul_(left_scale.view(1, 1, -1))
            right.mul_(right_scale.view(-1, 1, 1))

    @staticmethod
    def _index_select_core(core, dim, indices):
        return core.detach().index_select(dim, indices.to(core.device)).clone()

    def _prune_bond(self, bond_index, keep_indices):
        old_rank = int(self.tn[bond_index].shape[-1])
        if keep_indices.numel() >= old_rank:
            return 0
        for core_list in (self.tn, self.tn_tmp):
            core_list[bond_index] = self._index_select_core(
                core_list[bond_index], 2, keep_indices
            )
            core_list[bond_index + 1] = self._index_select_core(
                core_list[bond_index + 1], 0, keep_indices
            )
        return old_rank - int(keep_indices.numel())

    @staticmethod
    def _append_core_components(core, dim, count, scale):
        shape = list(core.shape)
        shape[dim] = int(count)
        addition = torch.randn(
            shape, dtype=core.dtype, device=core.device
        ) * float(scale)
        return torch.cat([core.detach(), addition], dim=dim)

    def _regrow_bond(self, bond_index, count):
        current_rank = int(self.tn[bond_index].shape[-1])
        count = min(int(count), int(self.max_rank) - current_rank)
        if count <= 0:
            return 0
        left_scale = (
            float(self.tn[bond_index].detach().std().item())
            * self.rank_ard_regrow_scale
        )
        right_scale = (
            float(self.tn[bond_index + 1].detach().std().item())
            * self.rank_ard_regrow_scale
        )
        left_scale = max(left_scale, self.rank_ard_balance_eps)
        right_scale = max(right_scale, self.rank_ard_balance_eps)
        self.tn[bond_index] = self._append_core_components(
            self.tn[bond_index], 2, count, left_scale
        )
        self.tn[bond_index + 1] = self._append_core_components(
            self.tn[bond_index + 1], 0, count, right_scale
        )
        # tn_tmp 只负责尚未激活的 contraction 补位，新增分量使用零值避免突变。
        for core_index, dim in ((bond_index, 2), (bond_index + 1, 0)):
            core = self.tn_tmp[core_index]
            shape = list(core.shape)
            shape[dim] = count
            zeros = torch.zeros(shape, dtype=core.dtype, device=core.device)
            self.tn_tmp[core_index] = torch.cat([core.detach(), zeros], dim=dim)
        return count

    def _sparsity_penalty(self, reso):
        scores = []
        cores = self._active_cores(reso)
        for bond_index in self._eligible_bond_indices(reso):
            scores.append(
                self.paired_component_scores(
                    cores[bond_index],
                    cores[bond_index + 1],
                    eps=self.rank_ard_balance_eps,
                ).mean()
            )
        if not scores:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        return torch.stack(scores).mean()

    def _build_train_masks(self, target_index):
        if self.rank_ard_mask_fraction <= 0:
            self._train_masks = None
            return
        seed_offset = 0 if target_index is None else int(target_index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.rank_ard_mask_seed + seed_offset)
        masks = []
        for target in self.targets:
            observed = (
                torch.rand(target.shape, generator=generator)
                >= self.rank_ard_mask_fraction
            )
            if not observed.any():
                observed.view(-1)[0] = True
            masks.append(observed.to(target.device))
        self._train_masks = masks

    def forward(self, reso):
        if reso not in self.interm_resos:
            raise ValueError(f"Invalid intermediate resolution: {reso}.")
        self._current_reso = int(reso)
        reso_index = self.interm_resos.index(reso)
        recon = self.custom_contract_qtt(reso)
        self.img = recon.detach().clone()
        target = self.targets[reso_index]
        if self._train_masks is None:
            loss_recon = F.mse_loss(recon, target)
        else:
            mask = self._train_masks[reso_index]
            loss_recon = F.mse_loss(recon[mask], target[mask])
        loss = loss_recon
        if self.rank_ard_sparsity_weight > 0:
            loss = loss + self.rank_ard_sparsity_weight * self._sparsity_penalty(
                reso
            )
        return loss, float(loss_recon.detach().item())

    def _stage_adapt(self, stage_index, reso, allow_regrow):
        before = self.bond_rank_vector()
        pruned_by_bond = {}
        regrown_by_bond = {}
        score_rows = {}
        precision_rows = {}
        for bond_index in self._eligible_bond_indices(reso):
            self._balance_bond(bond_index)
            scores = self.paired_component_scores(
                self.tn[bond_index],
                self.tn[bond_index + 1],
                eps=self.rank_ard_balance_eps,
            )
            precision = self.relevance_precision(
                self.tn[bond_index],
                self.tn[bond_index + 1],
                eps=self.rank_ard_balance_eps,
            )
            keep = self.select_components(
                scores, self.rank_ard_min_rank, self.rank_ard_energy
            )
            pruned_by_bond[str(bond_index)] = self._prune_bond(
                bond_index, keep
            )
            score_rows[str(bond_index)] = [float(value) for value in scores.cpu()]
            precision_rows[str(bond_index)] = [
                float(value) for value in precision.cpu()
            ]

        self._rebuild_contraction_path()
        recon = self.custom_contract_qtt(reso).detach()
        target = self.targets[self.interm_resos.index(reso)]
        mse = float(F.mse_loss(recon, target).item())
        variance = float(target.detach().var(unbiased=False).item())
        normalized_residual = mse / max(variance, self.rank_ard_balance_eps)

        if (
            allow_regrow
            and self.rank_ard_enable_regrow
            and normalized_residual >= self.rank_ard_regrow_residual_ratio
        ):
            for bond_index in self._eligible_bond_indices(reso):
                regrown_by_bond[str(bond_index)] = self._regrow_bond(
                    bond_index, self.rank_ard_regrow_step
                )
            self._rebuild_contraction_path()

        heldout_mse = None
        if self._train_masks is not None:
            mask = ~self._train_masks[self.interm_resos.index(reso)]
            if mask.any():
                heldout_mse = float(F.mse_loss(recon[mask], target[mask]).item())
        row = {
            "stage_index": int(stage_index),
            "resolution": int(reso),
            "rank_vector_before": before,
            "rank_vector_after": self.bond_rank_vector(),
            "pruned_by_bond": pruned_by_bond,
            "regrown_by_bond": regrown_by_bond,
            "pruned_count": int(sum(pruned_by_bond.values())),
            "regrown_count": int(sum(regrown_by_bond.values())),
            "mse": mse,
            "normalized_residual": normalized_residual,
            "heldout_mse": heldout_mse,
            "component_scores": score_rows,
            "relevance_precision": precision_rows,
        }
        self.rank_trajectory.append(row)
        return row

    @staticmethod
    def _safe_scheduler(optimizer, lr, decay, steps):
        if steps <= 1:
            return None
        warmup = max(1, int(steps * 0.1))
        decay_steps = max(1, steps - warmup)
        gamma = float(decay) ** (1.0 / decay_steps)
        return warmup, torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
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
        """训练并仅依据输入重构证据确定 rank；``clf`` 必须为空。"""
        del visualize, cln_target, visualize_dir, record_loss
        if clf is not None:
            raise ValueError("PTR_3d_rank_ard rank inference cannot access a classifier.")
        if logging is None:
            class _NullLogger:
                def info(self, *args, **kwargs):
                    return None
            logging = _NullLogger()

        self._build_train_masks(target_index)
        boundaries = [0] + list(args.iterations_for_upsampling) + [
            int(args.num_iterations)
        ]
        stage_steps = [
            boundaries[index + 1] - boundaries[index]
            for index in range(len(self.interm_resos))
        ]
        lr = float(args.lr)
        optimizer = self.set_optimizer(self.init_reso, lr=lr)
        start_time = time.time()

        for stage_index, (reso, steps) in enumerate(
            zip(self.interm_resos, stage_steps)
        ):
            self.adjust_optimizer(reso)
            schedule = self._safe_scheduler(
                optimizer,
                lr,
                args.lr_decay_factor_until_next_upsampling,
                steps,
            )
            for step in range(max(0, int(steps))):
                optimizer.zero_grad()
                loss, _ = self.forward(reso)
                loss.backward()
                optimizer.step()
                if schedule is not None:
                    warmup, scheduler = schedule
                    if step < warmup:
                        factor = float(step + 1) / float(warmup)
                        for group in optimizer.param_groups:
                            group["lr"] = lr * factor
                    else:
                        scheduler.step()

            row = self._stage_adapt(
                stage_index,
                reso,
                allow_regrow=stage_index < len(self.interm_resos) - 1,
            )
            logging.info("ARD stage finished: %s", row)
            lr *= float(args.lr_decay_factor)
            if stage_index < len(self.interm_resos) - 1:
                optimizer = self.set_optimizer(reso, lr=lr)

        # 最终阶段刚完成物理裁剪，没有下一分辨率阶段替新结构恢复重构；
        # 因而所有变体都允许无标签 full-target refit。masked 变体同时在此
        # 切回完整观测，其他变体则只是对裁剪后的 cores 做结构适配。
        if self.rank_ard_full_refit_steps > 0:
            self._train_masks = None
            optimizer = self.set_optimizer(self.end_reso, lr=lr)
            self.adjust_optimizer(self.end_reso)
            for _ in range(self.rank_ard_full_refit_steps):
                optimizer.zero_grad()
                loss, _ = self.forward(self.end_reso)
                loss.backward()
                optimizer.step()
            logging.info(
                "ARD post-prune full-target refit finished: steps=%d",
                self.rank_ard_full_refit_steps,
            )

        with torch.no_grad():
            self.img = self.custom_contract_qtt(self.end_reso).detach().clone()
        elapsed = time.time() - start_time
        logging.info(
            "ARD rank selected: vector=%s mean=%.4f time=%.4f",
            self.bond_rank_vector(),
            sum(self.bond_rank_vector()) / len(self.bond_rank_vector()),
            elapsed,
        )
        return self.img, elapsed, []

    def get_rank_diagnostics(self):
        ranks = self.bond_rank_vector()
        return {
            "method": self.model,
            "final_rank_vector": ranks,
            "mean_rank": float(sum(ranks) / len(ranks)),
            "min_rank": int(min(ranks)),
            "max_rank": int(max(ranks)),
            "effective_parameters": int(self.count_parameters()),
            "stage_trajectory": self.rank_trajectory,
            "rank_inference_signals": [
                "paired_core_slice_norm",
                "reconstruction_residual",
                "optional_masked_reconstruction",
            ],
            "uses_labels": False,
            "uses_classifier_logits": False,
        }
