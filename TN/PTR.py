import string
import mne
import pywt
import torch
import numpy as np
from opt_einsum import contract, contract_path
import yaml

from TN.opt import Config
from TN.tn_utils import *
from TN.BaseTNModel import BaseTNModel

class PTR(BaseTNModel):
    def __init__(self, target, stage, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=1, payload_position='first_core', regularization_type="TV", dimensions=2, regularization_weight = 0.0, noisy_target = None, device = 'cpu',  masked_avg_pooling = False, sigma_init=0, num_iterations=None, iterations_for_upsampling=None, ds_method="avg_pool"):
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
        self.model = "PTR"

        def is_power_of_two(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0
        # assert is_power_of_two(target.shape[0]) and is_power_of_two(target.shape[1]) and target.shape[1] >= target.shape[0], "Both height and width of the target must be powers of two."
        assert target.shape[1] >= target.shape[0], "Both height and width of the target must be powers of two."

        self.C = target.shape[0]
        self.T = target.shape[1]
        self.stage = stage
        self.ds_method = ds_method
        init_reso = min(self.C, self.T) // (2 ** (stage-1))
        end_reso = min(self.C, self.T)
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, payload, payload_position, regularization_type, dimensions, noisy_target, device, masked_avg_pooling, sigma_init)
        self.shape_source = None
        self.shape_factors = None
        self.factor_target_to_source = None
        self.dim_grid_log2 = int(np.log2(min(self.C, self.T)))

        self.init_reso = init_reso
        self.current_reso = init_reso
        self.end_reso = end_reso
        self.num_iterations = num_iterations
        self.iterations_for_upsampling = iterations_for_upsampling
        self.iterations_for_upsampling.append(self.num_iterations)
            
        interm_resos = [self.init_reso * 2**i for i in range(int(np.log2(self.end_reso/self.init_reso)))] + [end_reso]
        self.interm_resos = interm_resos
        self.factors = [int(self.end_reso/grid) for grid in self.interm_resos]

        # print(self.C)
        # print(self.T)
        # print(self.stage)
        # print(self.init_reso)
        # print(self.end_reso)
        # print(self.num_iterations)
        # print(self.iterations_for_upsampling)
        # print(self.interm_resos)
        # print(self.factors)

        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling

        self.regularization_weight = regularization_weight
        
        self.mask_rank = max_rank

        self.init_tn(self.C, self.T, self.stage)
        self.generate_targets_with_resos(self.target)
        self.current_reso_init_img = None
        self.adversarial_flag = False

        self.dim = len(self.shape_source)
        self.iteration = 0

        # self.einsum_strs = [self.build_einsum_chain(i) for i in [int(np.log2(j))+1 for j in interm_resos]]
        self.einsum_str = self.build_einsum_chain(int(np.log2(self.end_reso))+1)
        self.path, _ = contract_path(self.einsum_str, *self.tn, optimize='optimal')

    def init_tn(self, C, T, stage):
        """
        Initializes the tensor network (TN) for the model.
        """
        # Create the initial QTTNF
        self.tn, self.shape_source, self.shape_target, self.shape_factors, _, self.factor_target_to_source = get_rr_template_qtr_eeg(C, T, stage, self.max_rank, dim=self.dimensions, sigma_init=self.sigma_init, device=self.device)      
        self.tn_tmp, _, _, _, _, _  = get_rr_template_qtr_eeg(C, T, stage, self.max_rank, dim=self.dimensions, sigma_init=self.sigma_init, device=self.device)      
        # for i, core in enumerate(self.tn):
        #     print(f"core{i} shape:{core.shape}")

    def generate_targets_with_resos(self, img):
        '''
        img: the input, with shape (C, T, 1)
        '''
        # generate different resolution images
        assert len(img.shape) == 3, "The input should have the same resolution as the end resolution"

        different_res_imgs = []
        if self.ds_method == 'avg_pool':
            for factor in self.factors:
                # low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), kernel_size=(factor, factor * self.T // self.C), stride=(factor, factor * self.T // self.C))
                # low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), kernel_size=(factor, factor), stride=(factor, factor))
                low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), kernel_size=(1, factor*factor), stride=(1, factor*factor))
                # print(f"Generated low res img shape: {low_res_img.shape} for factor {factor}")
                high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
                # high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='bilinear').squeeze(0).permute(1,2,0).to(self.device)
                # high_res_moasic_img = self.fft_resample(low_res_img.squeeze(), target_len=img.shape[1]).unsqueeze(-1).to(self.device)
                # print(f"factor: {factor}, low_res_img shape: {low_res_img.shape}, high_res_moasic_img shape: {high_res_moasic_img.shape}")
                different_res_imgs.append(high_res_moasic_img)
        elif self.ds_method == 'dwt':
            for factor in self.factors:
                low_res_img = self.dwt_eeg(img, level=int(np.log2(factor)))
                # normalize the low_res_img to mean 0 and std 1, according to the last axis
                # mean = low_res_img.mean(axis=-1, keepdims=True)
                # std = low_res_img.std(axis=-1, keepdims=True)
                # low_res_img = (low_res_img - mean) / std
                high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
                different_res_imgs.append(high_res_moasic_img)
        elif self.ds_method == 'fft':
            for factor in self.factors:
                low_res_img = self.fft_resample(img.squeeze(), target_len=img.shape[1] // factor).unsqueeze(0).unsqueeze(0)
                mean = low_res_img.mean(axis=-1, keepdims=True)
                std = low_res_img.std(axis=-1, keepdims=True)
                low_res_img = (low_res_img - mean) / std
                high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
                different_res_imgs.append(high_res_moasic_img)
        elif self.ds_method == 'bandpass':
            def band_filter(data, low_pass=1, high_pass=40, sampling_rate=250):
                raw = mne.io.RawArray(data, mne.create_info(['ch1']*data.shape[0], sampling_rate, ['eeg']*data.shape[0]))
                raw.filter(l_freq=low_pass, h_freq=high_pass, picks='eeg', method='iir', verbose=False)
                return raw.get_data()
            high_pass_freq = 50
            for factor in self.factors:
                low_res_img = band_filter(img.squeeze().cpu().numpy(), low_pass=1, high_pass=high_pass_freq, sampling_rate=250)
                high_pass_freq = high_pass_freq // 2
                different_res_imgs.append(torch.from_numpy(low_res_img).unsqueeze(-1).float().to(self.device))
        elif self.ds_method == 'strided':
            for factor in self.factors:
                low_res_img = img.clone()[:, ::factor*factor, :]
                high_res_moasic_img = torch.nn.functional.interpolate(low_res_img.permute(2, 0, 1).unsqueeze(0), size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
                different_res_imgs.append(high_res_moasic_img)

        # different_res_imgs.append(img.detach().clone().to(self.device))
        # self.interm_resos.append(self.end_reso * 2)
        
        self.original_targets = [img.detach().clone() for img in different_res_imgs]
        self.targets = [img.detach().clone() for img in different_res_imgs]

    def dwt_eeg(self, x, level):
        x = x.squeeze().cpu().numpy()
        wavelet = "db4"
        
        coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level, mode="symmetric")
        cA = torch.from_numpy(coeffs[0]).float().unsqueeze(0).unsqueeze(0)

        return cA
    
    def fft_resample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        使用 FFT 对一维信号进行重采样（最后一个维度视为时间维）

        :param x: (..., N) 形式的输入信号
        :param target_len: 目标重采样长度
        :return: (..., target_len) 形式的重采样结果
        """
        orig_len = x.size(-1)
        Xf = torch.fft.rfft(x, dim=-1)  # (..., N_freq)
        num_freqs_out = target_len // 2 + 1
        num_freqs_in = Xf.size(-1)

        if target_len > orig_len:
            # zero-pad FFT (upsampling)
            pad_size = num_freqs_out - num_freqs_in
            pad = torch.zeros(*Xf.shape[:-1], pad_size, dtype=Xf.dtype, device=Xf.device)
            Xf_resampled = torch.cat([Xf, pad], dim=-1)
        else:
            # truncate FFT (downsampling)
            Xf_resampled = Xf[..., :num_freqs_out]

        # IFFT to get time domain signal
        y = torch.fft.irfft(Xf_resampled, n=target_len, dim=-1)
        return y

    def build_einsum_chain(self, n):
        """
        构建链式 einsum 表达式，类似于 'abc,cde,efg,gh->abdfh'
        参数:
            n: 张量数量，最小为2
        返回:
            einsum_str: einsum 表达式
        """
        assert n >= 2, "至少需要两个张量"

        letter_gen = iter(string.ascii_lowercase)
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
        # contract_core_num = int(np.log2(output_reso))
        contract_core_num = int(np.log2(output_reso)) + 1
        
        output_reso = self.end_reso
        # cores = self.tn[:contract_core_num+1] + self.tn_tmp[contract_core_num+1:]
        cores = self.tn[:contract_core_num] + self.tn_tmp[contract_core_num:]
        output = contract(self.einsum_str, *cores, optimize=self.path)

        shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source = get_qtt_shape_eeg(self.C, self.T, self.dim)
        # output = output.reshape(shape_factors + [self.T//self.C])
        output = output.reshape([self.T//self.C] + shape_factors)
        # factor_target_to_source = [i+1 for i in factor_target_to_source]
        # output = output.permute(factor_target_to_source + [-1])
        output = output.permute([0] +[i+1 for i in factor_target_to_source])
        output = output.reshape([self.T//self.C, self.C, self.C])
        output = output.permute(1, 0, 2)
        output = output.reshape(shape_source + [1])
        return output
    
    def set_optimizer(self, current_grid, lr):
        # params, skeleton = qtn.pack(tt)
        params = {
            i: t
            for i, t in enumerate(self.tn)
        }

        torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial).to(self.device)
            for i, initial in params.items()
        })
        start_exclude_ind = int(np.log2(self.init_reso))
        current_exclude_ind = int(np.log2(current_grid))
        end_exclude_ind = int(np.log2(self.end_reso))
        core_indices_to_exclude = [str(i+1) for i in range(start_exclude_ind, end_exclude_ind, 1)]
        # if core_indices_to_exclude length is not 0, remove the corresponding cores from the torch_params making them non-trainable
        if current_exclude_ind <= start_exclude_ind - 1:
            for i in core_indices_to_exclude:
                torch_params[i].requires_grad = False
        else:
            for i in range(end_exclude_ind):
                if i > current_exclude_ind:
                    torch_params[str(i)].requires_grad = False
                else:
                    torch_params[str(i)].requires_grad = True
        
        out = [{'params': torch_params.values(), 'lr': lr}]
        optimizer = torch.optim.Adam(out)
        self.tn = [p for _, p in torch_params.items()]

        return optimizer
    
    def adjust_optimizer(self, current_grid):
        current_exclude_ind = int(np.log2(current_grid))

        for idx, param in enumerate(self.tn):
            if idx <= current_exclude_ind:
            # if idx < current_exclude_ind:
                param.requires_grad = True
            else:
                param.requires_grad = False
            # print(f"core{idx}, shape: {param.shape}, requires_grad: {param.requires_grad}")

    def count_parameters(self):
        num = 0
        for i, core in enumerate(self.tn):
            num += core.numel()
        return num
    
    def forward(self, reso):
        assert reso in self.interm_resos, f"Invalid intermediate resolution, should be one of {self.interm_resos}"

        # L2
        reso_index = self.interm_resos.index(reso)
        recon = self.custom_contract_qtt(output_reso=reso)
        recon_ = recon.clone()
        self.img = recon.detach().clone()
        loss = self.loss_fn(recon, self.targets[reso_index])
        
        if self.loss_fn_str == "TV":
            loss += self.regularization_weight * self.TV(recon, p=2)

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
            logging.info(f"End purification for resolution {reso}*{reso}, loss={loss_recon}")
        time2 = time.time()
        # print(f"Training time: {time2-time1}, loss: {torch.nn.functional.mse_loss(self.img, target).item()}, params_num: {self.count_parameters()}")
        logging.info(f"Training time: {time2-time1}, loss: {torch.nn.functional.mse_loss(self.img, target).item()}, params_num: {self.count_parameters()}")

        recon = self.img
        return recon, time2-time1, mse_history
        
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