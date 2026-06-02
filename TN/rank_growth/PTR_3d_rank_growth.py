import math
import time

import torch
import torch.nn.functional as F
from opt_einsum import contract

from TN.PTR_3d import PTR_3d


class PTR_3d_rank_growth(PTR_3d):
    """
    基于 PTR_3d 的动态 rank-growth 版本。

    模型始终创建 max_rank=Rmax 的 Tensor Ring 参数容器，forward 时只使用
    active_rank 对应的时间 bond 切片。训练从低 rank 开始，逐档扩展 rank，
    每档结束后由 purify.py 传入的回调计算原始 EEG 空间 MSE 和分类 logits，
    再根据相邻 rank 的 JS/top1 稳定性选择最小充分 rank。
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
        rank_growth_ranks=None,
        rank_growth_steps_per_rank=512,
        rank_growth_js_threshold=0.02,
        rank_growth_max_mse_to_input=None,
        rank_growth_lr_decay_factor=1.0,
    ):
        rank_sequence = self._normalize_rank_sequence(rank_growth_ranks, max_rank)
        max_rank = max(max_rank, max(rank_sequence))

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

        self.model = "PTR_3d_rank_growth"
        self.rank_growth_ranks = rank_sequence
        self.rank_growth_steps_per_rank = int(rank_growth_steps_per_rank)
        self.rank_growth_js_threshold = float(rank_growth_js_threshold)
        self.rank_growth_max_mse_to_input = self._optional_float(rank_growth_max_mse_to_input)
        self.rank_growth_lr_decay_factor = float(rank_growth_lr_decay_factor)
        self.active_rank = self.rank_growth_ranks[0]
        self.selected_rank = None
        self.dynamic_rank_history = []

    @staticmethod
    def _normalize_rank_sequence(rank_growth_ranks, max_rank):
        """解析 YAML/环境变量传入的 rank 序列，并保证 rank 单调递增且合法。"""
        if rank_growth_ranks is None:
            ranks = list(range(5, int(max_rank) + 1, 5))
            if ranks[-1] != int(max_rank):
                ranks.append(int(max_rank))
        elif isinstance(rank_growth_ranks, str):
            ranks = [int(item.strip()) for item in rank_growth_ranks.split(",") if item.strip()]
        else:
            ranks = [int(item) for item in rank_growth_ranks]

        ranks = sorted(set(ranks))
        if not ranks or ranks[0] <= 0:
            raise ValueError(f"rank_growth_ranks must contain positive ranks, got: {rank_growth_ranks}")
        if ranks[-1] > int(max_rank):
            raise ValueError(
                f"rank_growth_ranks max rank {ranks[-1]} exceeds max_rank={max_rank}."
            )
        return ranks

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return float(value)

    def _slice_core(self, core, core_index, active_rank):
        """只切时间相关 bond，空间 core 保持完整。"""
        if core_index < 2:
            return core
        if core_index == 2:
            return core[:, :, :active_rank]
        if core_index == len(self.tn) - 1:
            return core[:active_rank, :, :]
        return core[:active_rank, :, :active_rank]

    def custom_contract_qtt(self, output_reso):
        contract_core_num = int(math.log2(output_reso)) + 2
        cores = self.tn[:contract_core_num] + self.tn_tmp[contract_core_num:]
        cores = [
            self._slice_core(core, core_index, self.active_rank)
            for core_index, core in enumerate(cores)
        ]
        output = contract(self.einsum_str, *cores, optimize=self.path)
        return output.reshape([self.C1, self.C2, self.T])

    def count_parameters(self):
        """返回当前选中 rank 的有效参数量，避免用 Rmax 总参数量误导压缩率。"""
        rank = int(self.selected_rank or self.active_rank)
        total = 0
        for core_index, core in enumerate(self.tn):
            total += self._slice_core(core, core_index, rank).numel()
        return total

    @staticmethod
    def _js_divergence_from_logits(logits_a, logits_b):
        probs_a = F.softmax(logits_a, dim=-1).clamp_min(1e-12)
        probs_b = F.softmax(logits_b, dim=-1).clamp_min(1e-12)
        probs_a = probs_a / probs_a.sum(dim=-1, keepdim=True)
        probs_b = probs_b / probs_b.sum(dim=-1, keepdim=True)
        midpoint = 0.5 * (probs_a + probs_b)
        kl_a = (probs_a * (probs_a / midpoint).log()).sum(dim=-1)
        kl_b = (probs_b * (probs_b / midpoint).log()).sum(dim=-1)
        return 0.5 * (kl_a + kl_b)

    @staticmethod
    def _set_optimizer_lr(optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _scheduled_lr(self, base_lr, decay_factor, step, total_steps):
        """每个分辨率段内做轻量 warmup + 指数衰减，避免小步数时 scheduler 除零。"""
        if total_steps <= 1:
            return base_lr
        warmup_steps = max(1, int(total_steps * 0.1))
        if step < warmup_steps:
            return base_lr * float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps + 1) / float(decay_steps)
        return base_lr * (decay_factor ** progress)

    def _resolution_steps_for_rank(self, args):
        """
        将原始 PTR 的分辨率 schedule 按比例压缩到每个 rank block。

        例如原始 2048 步的 [256, 512, 1024] 会在 512 步 rank block 中
        变成 [64, 64, 128, 256]，从而保留 coarse-to-fine 的训练节奏。
        """
        total_steps = max(1, int(self.rank_growth_steps_per_rank))
        base_total = int(getattr(args, "num_iterations", total_steps) or total_steps)
        raw_points = [
            int(value)
            for value in getattr(args, "iterations_for_upsampling", [])
            if 0 < int(value) < base_total
        ]
        raw_points = raw_points[: max(0, len(self.interm_resos) - 1)]

        if len(raw_points) == len(self.interm_resos) - 1 and base_total > 0:
            points = [0] + raw_points + [base_total]
            base_intervals = [points[i + 1] - points[i] for i in range(len(self.interm_resos))]
            scaled = [max(1, round(total_steps * interval / base_total)) for interval in base_intervals]
            delta = total_steps - sum(scaled)
            scaled[-1] += delta
            if scaled[-1] <= 0:
                scaled[-2] += scaled[-1] - 1
                scaled[-1] = 1
            return scaled

        steps = [0] * (len(self.interm_resos) - 1) + [total_steps]
        return steps

    def _evaluate_rank(self, rank, target, rank_eval_callback):
        old_rank = self.active_rank
        self.active_rank = int(rank)
        with torch.no_grad():
            recon = self.custom_contract_qtt(self.end_reso).detach().cpu()

        if rank_eval_callback is None:
            mse_to_input = F.mse_loss(recon.to(target.device), target).item()
            logits = None
        else:
            eval_result = rank_eval_callback(recon)
            mse_to_input = float(eval_result["mse_to_input"])
            logits = eval_result.get("logits")
            if logits is not None:
                logits = logits.detach().cpu()

        top1 = None
        confidence = None
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            confidence_tensor, top1_tensor = probs.max(dim=-1)
            top1 = int(top1_tensor.item())
            confidence = float(confidence_tensor.item())

        self.active_rank = old_rank
        return {
            "rank": int(rank),
            "recon": recon,
            "mse_to_input": mse_to_input,
            "logits": logits,
            "top1": top1,
            "confidence": confidence,
        }

    def _train_one_rank(self, args, optimizer, rank, rank_index):
        self.active_rank = int(rank)
        rank_lr = float(args.lr) * (self.rank_growth_lr_decay_factor ** rank_index)
        resolution_steps = self._resolution_steps_for_rank(args)
        loss_recon = None

        for reso, step_count in zip(self.interm_resos, resolution_steps):
            if step_count <= 0:
                continue
            self.adjust_optimizer(reso)
            segment_lr = rank_lr
            for step in range(step_count):
                lr = self._scheduled_lr(
                    segment_lr,
                    float(args.lr_decay_factor_until_next_upsampling),
                    step,
                    step_count,
                )
                self._set_optimizer_lr(optimizer, lr)
                optimizer.zero_grad()
                loss, loss_recon = self.forward(reso)
                loss.backward()
                optimizer.step()
            rank_lr *= float(args.lr_decay_factor)

        return float(loss_recon) if loss_recon is not None else None

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
        rank_eval_callback=None,
    ):
        del visualize, cln_target, visualize_dir, record_loss, clf
        time_start = time.time()
        optimizer = self.set_optimizer(self.init_reso, lr=args.lr)
        previous_eval = None
        selected_eval = None

        for rank_index, rank in enumerate(self.rank_growth_ranks):
            loss_recon = self._train_one_rank(args, optimizer, rank, rank_index)
            current_eval = self._evaluate_rank(rank, target, rank_eval_callback)

            js_to_prev = None
            mse_rel_delta_to_prev = None
            top1_unchanged = None
            fidelity_gate_pass = None
            rejected_by_mse_gate = False
            stable = False

            if previous_eval is not None:
                mse_prev = previous_eval["mse_to_input"]
                mse_curr = current_eval["mse_to_input"]
                mse_rel_delta_to_prev = abs(mse_prev - mse_curr) / max(abs(mse_prev), 1e-12)

                if previous_eval["logits"] is not None and current_eval["logits"] is not None:
                    js_to_prev = float(
                        self._js_divergence_from_logits(
                            previous_eval["logits"],
                            current_eval["logits"],
                        ).item()
                    )
                    top1_unchanged = previous_eval["top1"] == current_eval["top1"]
                    stable = (
                        bool(top1_unchanged)
                        and js_to_prev <= self.rank_growth_js_threshold
                    )

                if stable:
                    if self.rank_growth_max_mse_to_input is None:
                        fidelity_gate_pass = True
                    else:
                        fidelity_gate_pass = (
                            previous_eval["mse_to_input"] <= self.rank_growth_max_mse_to_input
                        )

                    if fidelity_gate_pass:
                        selected_eval = previous_eval
                    else:
                        rejected_by_mse_gate = True
                        stable = False

            row = {
                "rank": int(rank),
                "mse_to_input": current_eval["mse_to_input"],
                "js_to_prev": js_to_prev,
                "top1": current_eval["top1"],
                "confidence": current_eval["confidence"],
                "mse_rel_delta_to_prev": mse_rel_delta_to_prev,
                "top1_unchanged": top1_unchanged,
                "fidelity_gate_pass": fidelity_gate_pass,
                "rank_growth_max_mse_to_input": self.rank_growth_max_mse_to_input,
                "rejected_by_mse_gate": rejected_by_mse_gate,
                "loss": loss_recon,
                "selected": False,
            }
            self.dynamic_rank_history.append(row)

            if logging is not None:
                logging.info(f"Rank growth block finished: {row}")

            if selected_eval is not None:
                self.selected_rank = int(selected_eval["rank"])
                break
            previous_eval = current_eval

        if selected_eval is None:
            selected_eval = current_eval
            self.selected_rank = int(current_eval["rank"])

        for row in self.dynamic_rank_history:
            row["selected"] = row["rank"] == self.selected_rank

        self.active_rank = self.selected_rank
        self.img = selected_eval["recon"].detach().clone()
        elapsed = time.time() - time_start
        mse_final = F.mse_loss(self.img.to(target.device), target).item()

        if logging is not None:
            logging.info(f"Dynamic rank selected: {self.selected_rank}")
            logging.info(f"Dynamic rank history: {self.dynamic_rank_history}")
            logging.info(
                f"Training time: {elapsed}, loss: {mse_final}, "
                f"effective_params_num: {self.count_parameters()}"
            )

        return self.img, elapsed, []
