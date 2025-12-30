import string
import torch
import numpy as np
from opt_einsum import contract, contract_path
import yaml

from TN.opt import Config
from TN.tn_utils import *
from TN.BaseTNModel import BaseTNModel

class PTR_3d(BaseTNModel):
    def __init__(self, target, stage, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=1, payload_position='first_core', regularization_type="TV", dimensions=2, regularization_weight = 0.0, noisy_target = None, device = 'cpu',  masked_avg_pooling = False, sigma_init=0, num_iterations=None, iterations_for_upsampling=None):
        """
        Initializes the QTTModel object.

        Parameters:
        - target: the target tensor to model.
        - init_reso: the initial side length of the tensor.
        - max_rank: maximum rank for tensor decompositions.
        - dtype: data type for computations (default 'float32').
        - loss_fn_str: loss function to be used (e.g., "L2").
        - use_TTNF_sampling: whether to use TTNF V2 sampling - See TTNF Obukhov et. al. 2023.
        - payload: additional payload dimensions.
        - payload_position: the position of the payload in the tensor network - either 'first_core' or 'grayscale': No payload (for grayscale images)
        - canonization: method for canonization in tensor network.
        - activation: activation function to be used (e.g., "None", "relu").
        - compression_alg: algorithm for tensor compression - either 'compress_all' (TT-SVD) or 'compress_tree'.
        - regularization_type: type of regularization (e.g., "TV" for total variation).
        - dimensions: the number of dimensions of the input - e.g. 2 for 2D or and 3 for 3D structures
        - regularization_weight: weight of the regularization term.
        - noisy_target: noisy version of the target tensor - for experiments with noisy or incomplete data
        - device: computation device (e.g., 'cpu', 'cuda').
        - masked_avg_pooling: whether to use masked average pooling - used for incomplete data experiments
        """
        self.model = "PTR_3d"

        self.C1 = target.shape[0]
        self.C2 = target.shape[1]
        self.T = target.shape[2]
        self.stage = stage
        init_reso = self.T // (2 ** (stage-1))
        end_reso = self.T
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, payload, payload_position, regularization_type, dimensions, noisy_target, device, masked_avg_pooling, sigma_init)
        self.shape_source = None
        self.shape_factors = None
        self.factor_target_to_source = None
        self.dim_grid_log2 = int(np.log2(self.T))

        self.init_reso = init_reso
        self.current_reso = init_reso
        self.end_reso = end_reso
        self.num_iterations = num_iterations
        self.iterations_for_upsampling = iterations_for_upsampling
        self.iterations_for_upsampling.append(self.num_iterations)
            
        interm_resos = [self.init_reso * 2**i for i in range(int(np.log2(self.end_reso/self.init_reso)))] + [end_reso]
        self.interm_resos = interm_resos
        self.factors = [int(self.end_reso/grid) for grid in self.interm_resos]

        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling

        self.regularization_weight = regularization_weight
        
        self.mask_rank = max_rank

        self.init_tn(self.C1, self.C2, self.T, self.stage)
        self.generate_targets_with_resos(self.target)
        self.current_reso_init_img = None
        self.adversarial_flag = False

        self.iteration = 0

        # self.einsum_strs = [self.build_einsum_chain(i) for i in [int(np.log2(j))+1 for j in interm_resos]]
        self.einsum_str = self.build_einsum_chain(len(self.tn))
        # print(f"Einsum string: {self.einsum_str}")
        # self.path, _ = contract_path(self.einsum_str, *self.tn, optimize='optimal')
        self.path, _ = contract_path(self.einsum_str, *self.tn, optimize='greedy')

    def init_tn(self, C1, C2, T, stage):
        """
        Initializes the tensor network (TN) for the model.
        """
        # Create the initial QTTNF
        self.tn = get_rr_template_qtr_eeg_3d(C1, C2, T, stage, self.max_rank, dim=self.dimensions, sigma_init=self.sigma_init, device=self.device)      
        self.tn_tmp = get_rr_template_qtr_eeg_3d(C1, C2, T, stage, self.max_rank, dim=self.dimensions, sigma_init=self.sigma_init, device=self.device)      
        # for i, core in enumerate(self.tn):
        #     print(f"core{i} shape:{core.shape}")

    def generate_targets_with_resos(self, img):
        '''
        img: the input, with shape (C, C, T)
        '''
        # generate different resolution images
        assert len(img.shape) == 3, "The input should have the same resolution as the end resolution"

        different_res_imgs = []
        for factor in self.factors:
            low_res_img = torch.nn.functional.avg_pool2d(img.unsqueeze(0).clone(), kernel_size=(1, factor), stride=(1, factor))
            # print(f"Generated low res img shape: {low_res_img.shape} for factor {factor}")
            high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[1:], mode='nearest').squeeze(0).to(self.device)
            different_res_imgs.append(high_res_moasic_img)
        # different_res_imgs.append(img.detach().clone().to(self.device))
        # self.interm_resos.append(self.end_reso * 2)
        
        self.original_targets = [img.detach().clone() for img in different_res_imgs]
        self.targets = [img.detach().clone() for img in different_res_imgs]

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

        # Step 1: 构建第一个张量
        l1 = next(letter_gen)  # a
        l2 = next(letter_gen)  # b
        l3 = next(letter_gen)  # c
        # l4 = next(letter_gen)  # d
        einsum_inputs.append(l1 + l2 + l3)
        output_labels.extend([l2])  # 输出取前两个

        prev = l3  # 下一张量的第一个维度

        # Step 2: 构建中间张量（直到倒数第二个）
        for i in range(n - 2):
            mid = next(letter_gen)  # 中间维度保留
            end = next(letter_gen)  # 新末尾维度
            einsum_inputs.append(prev + mid + end)
            output_labels.append(mid)  # 输出保留中间维度
            prev = end  # 下一轮开头

        # Step 3: 构建最后一个张量（三个）
        last = next(letter_gen)
        einsum_inputs.append(prev + last + l1)
        output_labels.append(last)

        # Step 4: 拼接
        einsum_str = ','.join(einsum_inputs) + '->' + ''.join(output_labels)
        return einsum_str
    
    def custom_contract_qtt(self, output_reso):
        contract_core_num = int(np.log2(output_reso)) + 2
        # reso_index = self.interm_resos.index(output_reso)
        # contract_core_num = reso_index + 3
        
        output_reso = self.end_reso
        cores = self.tn[:contract_core_num] + self.tn_tmp[contract_core_num:]
        output = contract(self.einsum_str, *cores, optimize=self.path)

        output = output.reshape([self.C1, self.C2, self.T])
        return output
    
    def set_optimizer(self, current_grid, lr):
        # params, skeleton = qtn.pack(tt)
        params = {
            i: t
            for i, t in enumerate(self.tn)
        }
        torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
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
            # if idx < current_exclude_ind:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def count_parameters(self):
        num = 0
        for i, core in enumerate(self.tn):
            num += core.numel()
        return num
    
    def TV_3d(self, tensor, p=2):
        """
        shape: h,w,c
        calculate total variation
        """
        # Compute the squared differences in the third direction
        horizontal_diff = torch.pow(abs(tensor[:, :, 1:] - tensor[:, :, :-1]), p)

        # Compute the squared differences in the second direction
        vertical_diff = torch.pow(abs(tensor[:, 1:, :] - tensor[:, :-1, :]), p)

        # Compute the squared differences in the first direction
        depth_diff = torch.pow(abs(tensor[1:, :, :] - tensor[:-1, :, :]), p)

        # Sum up the horizontal and vertical differences
        total_variation_loss = torch.sum(horizontal_diff) + torch.sum(vertical_diff) + torch.sum(depth_diff)

        return total_variation_loss / tensor.numel()
    
    def TV_eeg(self, tensor, p=2):
        """
        shape: h,w,c
        calculate total variation
        """
        # Compute the squared differences in the horizontal direction
        horizontal_diff = torch.pow(abs(tensor[:, :, 1:] - tensor[:, :, :-1]), p)

        # Sum up the horizontal and vertical differences
        total_variation_loss = torch.sum(horizontal_diff)

        return total_variation_loss / tensor.numel()
    
    def TV_CFT(self, tensor: torch.Tensor, p: int = 1, mode: str = 'ft') -> torch.Tensor:
        """
        tensor: (C, F, T) or (F, T) etc.
        mode:
        - 'ft': 对最后两个维度(F,T)做TV（推荐用于 (C,F,T)）
        - 'f' : 只对频率维做TV
        - 't' : 只对时间维做TV
        - 'cft': 对 (C,F,T) 三维都做TV（若输入就是3D）
        """
        # 统一成至少 3D: (C,F,T)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1,F,T)
        elif tensor.dim() != 3:
            raise ValueError(f"TV expects 2D or 3D tensor, got shape={tuple(tensor.shape)}")

        # (C,F,T)
        if 'f' in mode:
            # 频率方向差分：F 维
            df = torch.abs(tensor[:, 1:, :] - tensor[:, :-1, :]).pow(p)
        if 't' in mode:
            # 时间方向差分：T 维
            dt = torch.abs(tensor[:, :, 1:] - tensor[:, :, :-1]).pow(p)
        if 'c' in mode:
            # 通道方向差分：C 维
            dc = torch.abs(tensor[1:, :, :] - tensor[:-1, :, :]).pow(p) if tensor.size(0) > 1 else None

        if mode == 'ft':
            loss = df.sum() + dt.sum()
        elif mode == 'f':
            loss = df.sum()
        elif mode == 't':
            loss = dt.sum()
        elif mode == 'cft':
            loss = df.sum() + dt.sum() + (dc.sum() if dc is not None else 0.0)
        else:
            raise ValueError(f"Unknown mode={mode}, choose from ['ft','f','t','cft']")

        return loss / tensor.numel()
    
    def robust_phase_tv(self, phase):
        """
        处理相位回绕的 TV
        phase: (C, F, T) in radians
        """
        # 1. 频率方向差分 (Group Delay)
        # 计算 exp(j * phase) 的差分，或者直接用 diff 并 wrap
        diff_f = phase[:, 1:, :] - phase[:, :-1, :]
        # 将差值 wrap 到 (-pi, pi]
        diff_f = (diff_f + torch.pi) % (2 * torch.pi) - torch.pi
        
        # 2. 时间方向差分 (Instantaneous Frequency)
        diff_t = phase[:, :, 1:] - phase[:, :, :-1]
        diff_t = (diff_t + torch.pi) % (2 * torch.pi) - torch.pi

        # L1 Loss
        loss = torch.abs(diff_f).mean() + torch.abs(diff_t).mean()
        return loss

    def tf_tv_losses(
        self,
        data: torch.Tensor,
        phase_freq_start: int = 30,
        phase_mode: str = 'f',
        mag_mode: str = 'ft',
    ) -> dict:
        """
        计算 TF-TV 

        输入 data: (C, T) 
        返回 loss
        """
        if data.dim() == 3 and data.size(-1) == 1:
            data = data.squeeze(-1)  # (C,T)
        if data.dim() != 2:
            raise ValueError(f"data should be (C,T,1) or (C,T), got shape={tuple(data.shape)}")

        C, T = data.shape

        # --- STFT ---
        n_fft = C * 2 - 2
        hop_length = T // C
        pre_data = torch.stft(
            data,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            onesided=True,
            normalized=True
        )  # shape: (new_h, F, TT)

        # --- losses ---
        mag = torch.abs(pre_data)
        phase = torch.angle(pre_data)

        tv_mag = self.TV_CFT(mag, p=1, mode=mag_mode)

        # # 你原来 phase 只对高频部分算（[:, 15:, :]），并且 mode='f'
        # if phase_freq_start is not None and phase_freq_start > 0:
        #     freqs = torch.fft.rfftfreq(n_fft, d=1/250)  # (n_freq,)
        #     phase_freq_start_index = 15
        #     phase = phase[:, phase_freq_start_index:, :]

        # tv_phase = self.TV_CFT(phase, p=1, mode=phase_mode)
        tv_phase = self.robust_phase_tv(phase)

        return tv_mag + tv_phase
    
    def forward(self, reso):
        assert reso in self.interm_resos, f"Invalid intermediate resolution, should be one of {self.interm_resos}"

        # L2
        reso_index = self.interm_resos.index(reso)
        recon = self.custom_contract_qtt(output_reso=reso)
        recon_ = recon.clone()
        self.img = recon.detach().clone()
        loss = self.loss_fn(recon, self.targets[reso_index])
        if self.loss_fn_str == "TV":
            loss_reg = self.TV_3d(recon_)
            loss += self.regularization_weight * loss_reg
        elif self.loss_fn_str == "TVeeg":
            loss_reg = self.TV_eeg(recon_)
            loss += self.regularization_weight * loss_reg
        elif self.loss_fn_str == "TVft":
            loss_reg = self.tf_tv_losses(recon_.reshape(-1, self.T))
            loss += self.regularization_weight * loss_reg
        loss_recon = loss.item()

        return loss, loss_recon
    
    def train(self, target, args, target_index=None, visualize=False, cln_target=None, visualize_dir='', record_loss=False, clf=None, logging=None):
        if visualize == False:
            target_index = None
            cln_target = None
            visualize_dir = ''
        else:
            assert target_index is not None and cln_target is not None and visualize_dir != '', 'visualize is True but target_index, cln_target, visualize_dir are not provided'

        lr = args.lr
        self.lr_decay_factor = args.lr_decay_factor
        iterations = [0] + args.iterations_for_upsampling + [args.num_iterations]
        iterations = [iterations[i+1] - iterations[i] for i in range(len(self.interm_resos))]
        recons_multi_resos = []
        time1 = time.time()
        mse_history = []
        optimizer = self.set_optimizer(self.init_reso, lr=lr)
        ite = 0

        for i, reso in enumerate(self.interm_resos):
            # print(f"Start purification for resolution {reso}*{reso}")
            self.adjust_optimizer(reso)
            # Scheduler
            lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, int(iterations[i]*0.1))
            lr_gamma = calculate_gamma(lr, args.lr_decay_factor_until_next_upsampling, iterations[i]-int(iterations[i]*0.1))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

            # for t in range(len(self.tn)):
            #     print(f"Core {t} shape: {self.tn[t].shape}, requires_grad: {self.tn[t].requires_grad}")
    
            j = 0
            while j < iterations[i]:
                optimizer.zero_grad()
                loss, loss_recon = self.forward(reso)
                loss.backward()
                optimizer.step()

                if j < int(iterations[i]*0.1):
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
            for core_id, group in enumerate(optimizer.param_groups):
                group['lr'] = lr
                if 'initial_lr' in group:     # 关键！
                    del group['initial_lr']
            optimizer.state.clear()

            # print(f"End purification for resolution {reso}*{reso}, loss={loss_recon}")
            logging.info(f"End purification for resolution {reso}*{reso}, loss={loss_recon}, iterations={iterations[i]}")
        time2 = time.time()
        # print(f"Training time: {time2-time1}, loss: {torch.nn.functional.mse_loss(self.img, target).item()}, params_num: {self.count_parameters()}")
        logging.info(f"Training time: {time2-time1}, loss: {torch.nn.functional.mse_loss(self.img, target).item()}, params_num: {self.count_parameters()}")

        recon = self.img
        return recon, time2-time1, mse_history
    
    # def resize_eeg(self, args, data, sampling_rate, strategy):
    #     config_path = f'./configs/{args.dataset}/{args.config}'
    #     # init TN
    #     with open(config_path, "r", encoding="utf-8") as file:
    #         config = yaml.safe_load(file)
    #     config['device_type'] = "gpu" if torch.cuda.is_available() else "cpu"
    #     config_tmp = Config()
    #     for key, item in config.items():
    #         setattr(config_tmp, key, item)
    #     config = config_tmp

    #     data = data.permute(1, 2, 0)
    #     h, w, _ = data.shape
    #     new_h = round(2**np.ceil(np.log2(h)))
    #     eeg_t_seg = round(w / sampling_rate * 10)   # 10 segments per second
    #     k = np.ceil(w / (2 ** (config.stage - 1)) / eeg_t_seg)
    #     new_w = int(k * eeg_t_seg * (2 ** (config.stage - 1)))
    #     pre_data = torch.nn.functional.interpolate(data.permute(2, 0, 1).unsqueeze(0).cpu(), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

    #     return pre_data

    # def inv_resize_eeg(self, pre_data, original_shape):
    #     data = torch.nn.functional.interpolate(pre_data.permute(2, 0, 1).unsqueeze(0), size=original_shape, mode='bilinear', align_corners=False).squeeze(0)
    #     return data
    
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
        # return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[fix_lr_lambda]*(warmup_group_ids)+[lr_lambda]+[fix_lr_lambda]*(len(optimizer.param_groups)-warmup_group_ids-1))