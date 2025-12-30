import string
import time
import torch
import numpy as np
from opt_einsum import contract, contract_path

from TN.tn_utils import *
from TN.BaseTNModel import BaseTNModel


class PTR_3d_fs(BaseTNModel):
    def __init__(self, target, stage, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=1, payload_position='first_core', regularization_type="TV", dimensions=2, regularization_weight=0.0, noisy_target=None, device='cpu', masked_avg_pooling=False, sigma_init=0, num_iterations=None, iterations_for_upsampling=None):
        """
        PTR variant that models a 2D channel grid with shared time-frequency cores.
        Input target is expected to be shaped as (channel_x, channel_y, frequency, time).
        """
        self.model = "PTR_3d_fs"

        self.CX = target.shape[0]
        self.CY = target.shape[1]
        self.F = target.shape[2]
        self.T = target.shape[3]
        self.stage = stage

        # Align freq/time to the nearest power-of-two to match TT cores
        ft_dim = int(2 ** np.ceil(np.log2(max(self.F, self.T))))
        if self.F != ft_dim or self.T != ft_dim:
            target = torch.nn.functional.interpolate(
                target.view(1, self.CX * self.CY, self.F, self.T),
                size=(ft_dim, ft_dim),
                mode='bilinear',
                align_corners=False
            ).view(self.CX, self.CY, ft_dim, ft_dim)
            self.F = ft_dim
            self.T = ft_dim
        self.ft_dim = ft_dim

        init_reso = self.ft_dim // (2 ** (stage - 1))
        end_reso = self.ft_dim
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, payload, payload_position, regularization_type, dimensions, noisy_target, device, masked_avg_pooling, sigma_init)

        self.ft_log2 = int(np.log2(self.ft_dim))

        self.init_reso = init_reso
        self.current_reso = init_reso
        self.end_reso = end_reso
        self.num_iterations = num_iterations
        self.iterations_for_upsampling = iterations_for_upsampling or []
        self.iterations_for_upsampling.append(self.num_iterations)

        interm_resos = [self.init_reso * 2 ** i for i in range(int(np.log2(self.end_reso / self.init_reso)))] + [end_reso]
        self.interm_resos = interm_resos
        self.factors = [int(self.end_reso / grid) for grid in self.interm_resos]

        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling

        self.regularization_weight = regularization_weight

        self.mask_rank = max_rank

        self.init_tn(self.CX, self.CY, self.ft_dim, self.stage)
        self.generate_targets_with_resos(self.target)
        self.current_reso_init_img = None
        self.adversarial_flag = False

        self.iteration = 0

        self.einsum_str = self.build_einsum_chain(len(self.tn))
        self.path, _ = contract_path(self.einsum_str, *self.tn, optimize='greedy')

    def init_tn(self, CX, CY, FT, stage):
        """
        Initializes the tensor network (TN) for the model.
        """
        self.tn = get_rr_template_qtr_eeg_3d_fs(CX, CY, FT, stage, self.max_rank, dim=self.dimensions, sigma_init=self.sigma_init, device=self.device)
        self.tn_tmp = get_rr_template_qtr_eeg_3d_fs(CX, CY, FT, stage, self.max_rank, dim=self.dimensions, sigma_init=self.sigma_init, device=self.device)

    def generate_targets_with_resos(self, img):
        """
        img: input with shape (CX, CY, F, T)
        Downsample only along frequency/time to build coarse-to-fine targets.
        """
        assert len(img.shape) == 4, "The input should have the same resolution as the end resolution"

        different_res_imgs = []
        img_ft = img.view(1, self.CX * self.CY, self.F, self.T)
        for factor in self.factors:
            low_res = torch.nn.functional.avg_pool2d(img_ft, kernel_size=(factor, factor), stride=(factor, factor))
            high_res = torch.nn.functional.interpolate(low_res, size=(self.F, self.T), mode='nearest')
            different_res_imgs.append(high_res.view(self.CX, self.CY, self.F, self.T).to(self.device))

        self.original_targets = [x.detach().clone() for x in different_res_imgs]
        self.targets = [x.detach().clone() for x in different_res_imgs]

    def build_einsum_chain(self, n):
        """
        构建链式 einsum 表达式，类似于 'abc,cde,efg,gh->abdfh'
        参数:
            n: 张量数量，最小为2
        返回:
            einsum_str: einsum 表达式
        """
        assert n >= 2, "至少需要两个张量"

        letter_gen = iter(string.ascii_letters)
        einsum_inputs = []
        output_labels = []

        l1 = next(letter_gen)
        l2 = next(letter_gen)
        l3 = next(letter_gen)
        einsum_inputs.append(l1 + l2 + l3)
        output_labels.extend([l2])

        prev = l3
        for _ in range(n - 2):
            mid = next(letter_gen)
            end = next(letter_gen)
            einsum_inputs.append(prev + mid + end)
            output_labels.append(mid)
            prev = end

        last = next(letter_gen)
        einsum_inputs.append(prev + last + l1)
        output_labels.append(last)

        einsum_str = ','.join(einsum_inputs) + '->' + ''.join(output_labels)
        return einsum_str

    def custom_contract_qtt(self, output_reso):
        contract_core_num = int(np.log2(output_reso)) + 2
        cores = self.tn[:contract_core_num] + self.tn_tmp[contract_core_num:]
        output = contract(self.einsum_str, *cores, optimize=self.path)

        factor = output.reshape([self.CX, self.CY] + [2] * (2 * self.ft_log2))
        order = [0, 1] + [2 + i * 2 for i in range(self.ft_log2)] + [2 + i * 2 + 1 for i in range(self.ft_log2)]
        factor = factor.permute(order)
        output = factor.reshape(self.CX, self.CY, self.ft_dim, self.ft_dim)
        return output

    def set_optimizer(self, current_grid, lr):
        params = {
            i: t
            for i, t in enumerate(self.tn)
        }
        torch_params = torch.nn.ParameterDict({
            f"{i:03d}": torch.nn.Parameter(initial).to(self.device)
            for i, initial in params.items()
        })
        for i in range(len(self.tn)):
            torch_params[f"{i:03d}"].requires_grad = True

        out = [{'params': torch_params.values(), 'lr': lr}]
        optimizer = torch.optim.Adam(out)
        self.tn = [p for _, p in torch_params.items()]

        return optimizer

    def adjust_optimizer(self, current_grid):
        current_exclude_ind = int(np.log2(current_grid)) + 1

        for idx, param in enumerate(self.tn):
            if idx <= current_exclude_ind:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def count_parameters(self):
        num = 0
        for _, core in enumerate(self.tn):
            num += core.numel()
        return num

    def TV_4d(self, tensor, p=2):
        """
        Total variation across spatial and time-frequency axes.
        """
        diff_x = torch.pow(abs(tensor[1:, :, :, :] - tensor[:-1, :, :, :]), p)
        diff_y = torch.pow(abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :]), p)
        diff_f = torch.pow(abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]), p)
        diff_t = torch.pow(abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]), p)
        return (diff_x.sum() + diff_y.sum() + diff_f.sum() + diff_t.sum()) / tensor.numel()

    def forward(self, reso):
        assert reso in self.interm_resos, f"Invalid intermediate resolution, should be one of {self.interm_resos}"

        reso_index = self.interm_resos.index(reso)
        recon = self.custom_contract_qtt(output_reso=reso)
        recon_ = recon.clone()
        self.img = recon.detach().clone()
        loss = self.loss_fn(recon, self.targets[reso_index])
        if self.loss_fn_str == "TV":
            loss_reg = self.TV_4d(recon_)
            loss += self.regularization_weight * loss_reg
        loss_recon = loss.item()

        return loss, loss_recon

    def train(self, target, args, target_index=None, visualize=False, cln_target=None, visualize_dir='', record_loss=False, clf=None, logging=None):
        if visualize is False:
            target_index = None
            cln_target = None
            visualize_dir = ''
        else:
            assert target_index is not None and cln_target is not None and visualize_dir != '', 'visualize is True but target_index, cln_target, visualize_dir are not provided'

        lr = args.lr
        self.lr_decay_factor = args.lr_decay_factor
        iterations = [0] + args.iterations_for_upsampling + [args.num_iterations]
        iterations = [iterations[i + 1] - iterations[i] for i in range(len(self.interm_resos))]
        recons_multi_resos = []
        time1 = time.time()
        mse_history = []
        optimizer = self.set_optimizer(self.init_reso, lr=lr)

        for i, reso in enumerate(self.interm_resos):
            self.adjust_optimizer(reso)
            lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, int(iterations[i] * 0.1))
            lr_gamma = calculate_gamma(lr, args.lr_decay_factor_until_next_upsampling, iterations[i] - int(iterations[i] * 0.1))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

            j = 0
            while j < iterations[i]:
                optimizer.zero_grad()
                loss, loss_recon = self.forward(reso)
                loss.backward()
                optimizer.step()

                if j < int(iterations[i] * 0.1):
                    lr_warmup_scheduler.step()
                else:
                    scheduler.step()

                if record_loss:
                    if not self.unify_reso:
                        tmp = torch.nn.functional.interpolate(self.img.detach().cpu().permute(2, 0, 1).unsqueeze(0), size=(self.end_reso, self.end_reso), mode='nearest').squeeze().permute(1, 2, 0)
                        mse_history.append(self.loss_fn(tmp, target).item())
                    else:
                        mse_history.append(self.loss_fn(self.img.detach().cpu(), target).item())
                j += 1
            recons_multi_resos.append(self.img)

            lr *= args.lr_decay_factor
            for _, group in enumerate(optimizer.param_groups):
                group['lr'] = lr
                if 'initial_lr' in group:
                    del group['initial_lr']
            optimizer.state.clear()

            logging.info(f"End purification for resolution {reso}*{reso}, loss={loss_recon}, iterations={iterations[i]}")
        time2 = time.time()
        logging.info(f"Training time: {time2 - time1}, loss: {torch.nn.functional.mse_loss(self.img, target).item()}, params_num: {self.count_parameters()}")

        recon = self.img
        return recon, time2 - time1, mse_history


def calculate_gamma(lr0, decay_factor, num_iters):
    lrT = lr0 * decay_factor
    gamma = (lrT / lr0) ** (1 / num_iters)
    return gamma


def linear_warmup_lr_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        return min(1.0, float(current_step + 1) / warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def core_linear_warmup_lr_scheduler(optimizer, warmup_steps, warmup_group_ids=None):
    assert warmup_group_ids is not None

    def lr_lambda(current_step):
        return min(1.0, float(current_step + 1) / (warmup_steps))

    def fix_lr_lambda(current_step):
        return min(1.0, float(current_step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[fix_lr_lambda] * (warmup_group_ids) + [lr_lambda] + [fix_lr_lambda] * (len(optimizer.param_groups) - warmup_group_ids - 1))
