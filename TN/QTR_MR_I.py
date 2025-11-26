# Based on TT-NF code from https://github.com/toshas/ttnf
# Modifications and/or extensions have been made for specific purposes in this project.
import sys
import os
import cv2
import pywt
import scipy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import string
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

from opt_einsum import contract, contract_path
from tqdm import tqdm
from TN.opt import Config
from TN.tn_utils import *
from TN.utils import  SimpleSamplerNonRandom, get_model_args, save_img
# from TN.attention_mask_generator import create_cifar10_attention_generator, create_cifar100_attention_generator, create_imagenet_attention_generator
from torchvision import transforms
from TN.BaseTNModel import BaseTNModel
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class QTR_MR_I(BaseTNModel):
    def __init__(self, clf, target, init_reso, original_reso, end_reso, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position='first_core', canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions =2, regularization_weight = 0.0, noisy_target = None, device = 'cpu',  masked_avg_pooling = False, sigma_init=0, unify_reso=True, optimizer_type='part', img_TV=0, add_gaussian_noise=False, push=False, gaussian_noise_std=0.0, dataset=None, num_iterations=None, iterations_for_upsampling=None):
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
        
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, use_TTNF_sampling, payload, payload_position, canonization, activation, compression_alg, regularization_type, dimensions, noisy_target, device, masked_avg_pooling, sigma_init)
        self.model = "QTR_MR"
        self.canonization = canonization
        self.compression_alg = compression_alg
        self.regularization_type = regularization_type
        self.activation = activation

        self.shape_source = None
        self.shape_factors = None
        self.factor_target_to_source = None
        self.dim_grid_log2 = int(np.log2(init_reso))

        self.init_reso = init_reso
        self.current_reso = init_reso
        self.end_reso = end_reso
        self.original_reso = original_reso
        self.dataset = dataset
        self.num_iterations = num_iterations
        self.iterations_for_upsampling = iterations_for_upsampling
        self.iterations_for_upsampling.append(self.num_iterations)

        if self.attention_mask_generator is not None:
            target_tmp = torch.nn.functional.interpolate(target.detach().clone().permute(2,0,1).unsqueeze(0), size=(self.original_reso, self.original_reso), mode='bilinear').detach().clone()
            attention_mask, _, pure_area = self.attention_mask_generator.compute_attention_mask(target_tmp.to(self.device), return_attention_maps=True)
            attention_mask = torch.nn.functional.interpolate(attention_mask, size=(self.end_reso, self.end_reso), mode='bilinear').detach().clone()
            attention_mask = attention_mask.squeeze(0).permute(1,2,0).expand(-1, -1, 3).detach().clone()
            self.attention_mask = attention_mask.detach().clone()
            pure_area = torch.nn.functional.interpolate(pure_area, size=(self.end_reso, self.end_reso), mode='bilinear').detach().clone()
            pure_area = pure_area.squeeze(0).permute(1,2,0).expand(-1, -1, 3).detach().clone()
            self.pure_area = pure_area.detach().clone()
            
        interm_resos = [self.init_reso * 2**i for i in range(int(np.log2(self.end_reso/self.init_reso)))] + [end_reso]
        self.interm_resos = interm_resos
        self.factors = [int(self.end_reso/grid) for grid in self.interm_resos]

        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling

        self.regularization_weight = regularization_weight
        
        self.mask_rank = max_rank

        self.init_tn(end_reso, init_reso)
        self.img_TV = img_TV
        self.unify_reso = unify_reso
        assert optimizer_type in ['one', 'part']
        self.optimize_type = optimizer_type
        self.add_gaussian_noise = add_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.push = push
        self.generate_targets_with_resos(self.target, unify_reso=self.unify_reso, add_gaussian_noise=self.add_gaussian_noise)
        self.current_reso_init_img = None
        self.adversarial_flag = False

        self.dim = len(self.shape_source)
        self.iteration = 0

        self.einsum_strs = [self.build_einsum_chain(i + 1) for i in [int(np.log2(j)) for j in interm_resos]]

        if unify_reso:
            # print(self.einsum_strs[-1])
            self.path, _ = contract_path(self.einsum_strs[-1], *self.tn, optimize='optimal')
        else:
            self.path = []
            for i in range(len(self.einsum_strs)):
                contract_core_num = int(np.log2(self.interm_resos[i]))
                cores = self.tn[:contract_core_num-1] + [torch.mean(self.tn[contract_core_num-1], dim=-1) if len(self.tn[contract_core_num-1].shape)==3 else self.tn[contract_core_num-1]]
                path, _ = contract_path(self.einsum_strs[i], *cores, optimize='optimal')
                self.path.append(path)

    def init_tn(self, end_reso, init_reso):
        """
        Initializes the tensor network (TN) for the model.
        """
        # Create the initial QTTNF
        self.tn, self.shape_source, self.shape_target, self.shape_factors, _, self.factor_target_to_source = get_rr_template_qtr(end_reso, init_reso, self.max_rank, dim=self.dimensions, payload_dim=self.payload, payload_position=self.payload_position, compression_alg=self.compression_alg, canonization=self.canonization, sigma_init=self.sigma_init, device=self.device)      
        self.tn_tmp, _, _, _, _, _ = get_rr_template_qtr(end_reso, init_reso, self.max_rank, dim=self.dimensions, payload_dim=self.payload, payload_position=self.payload_position, compression_alg=self.compression_alg, canonization=self.canonization, sigma_init=self.sigma_init, device=self.device)
        # print(self.tn)
        # print("Initialized tn,", self.tn)
        for i, core in enumerate(self.tn):
            print(f"core{i} shape:{core.shape}")

    def generate_targets_with_resos(self, img, unify_reso=True, add_gaussian_noise=False):
        # generate different resolution images
        assert img.shape[0] == self.end_reso and len(img.shape) == 3, "The input image should have the same resolution as the end resolution"

        different_res_imgs = []
        for factor in self.factors:
            if unify_reso:
                low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), factor, factor)
                high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
                different_res_imgs.append(high_res_moasic_img)
            else:
                low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), factor, factor).squeeze(0).permute(1,2,0).to(self.device)
                different_res_imgs.append(low_res_img)
        
        # if add_gaussian_noise:
        #     # # self.original_targets = different_res_imgs.copy()
        #     # self.original_targets = [img.detach().clone() for img in different_res_imgs]

        #     # # self.smoothed_targets = [self.median_filter(img.permute(2,0,1).unsqueeze(0), 3).squeeze(0).permute(1,2,0).to(self.device) for img in self.original_targets]
        #     # # gauss = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))
        #     # # self.smoothed_targets = [gauss(img.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0).to(self.device) for img in self.original_targets]
        #     # # gaussian_noises = [torch.randn_like(img)*0.01*2**(i) for i, img in enumerate(different_res_imgs)]

        #     # # gaussian_noises = [torch.randn(reso, reso, 3)*0.02*1.5**(i) for i, reso in enumerate(self.interm_resos)]
        #     # stds = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
        #     # # stds = [0, 0.005, 0.008, 0.01]
        #     # gaussian_noises = [torch.randn(256, 256, 3)*stds[i] for i, reso in enumerate(self.interm_resos)]
            
        #     # gaussian_noises = [torch.randn(reso, reso, 3)*0.008*1.5**(i) for i, reso in enumerate(self.interm_resos)]
        #     # gaussian_noises = [torch.nn.functional.interpolate(noise.permute(2, 0, 1).unsqueeze(0), size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device) for noise in gaussian_noises]

        #     # # gaussian_noise = torch.randn(self.end_reso, self.end_reso, 3)*0.05
        #     # # gaussian_noises = []
        #     # # for i, reso in enumerate(self.interm_resos):
        #     # #     low_reso_noise = torch.nn.functional.interpolate(gaussian_noise.permute(2,0,1).unsqueeze(0), size=reso, mode='nearest')
        #     # #     # low_reso_noise = torch.nn.functional.avg_pool2d(gaussian_noise.permute(2,0,1).unsqueeze(0), self.factors[i], self.factors[i])
        #     # #     end_reso_noise = torch.nn.functional.interpolate(low_reso_noise, size=self.end_reso, mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
        #     # #     gaussian_noises.append(end_reso_noise.to(self.device))

        #     # # gaussian_noises = [torch.sign(torch.randn_like(image))*0.01*(i+1) for i, image in enumerate(different_res_imgs)]
        #     # for i in gaussian_noises:
        #     #     print(f"gaussian noise shape: {i.shape}, mean: {i.mean()}, std: {i.std()}, mse: {torch.nn.functional.mse_loss(i, torch.zeros_like(i)).item()}")
        #     # # different_res_imgs = [torch.clamp(img+gaussian_noises[i], 0, 1) for i, img in enumerate(different_res_imgs)]
        #     # different_res_imgs = [img+gaussian_noises[i] for i, img in enumerate(self.original_targets)]
        #     # self.targets = [img.detach().clone() for img in different_res_imgs]

        #     self.original_targets = [img.detach().clone() for img in different_res_imgs]

        #     if self.loss_fn_str == "maskTV":
        #         # img = img.clone() + self.pure_area.detach().cpu() * torch.randn_like(img) * self.gaussian_noise_std
        #         img = img.clone()
        #     else:
        #         img = img.clone() + torch.randn_like(img)*self.gaussian_noise_std

        #     different_res_imgs = []
        #     for factor in self.factors:
        #         if unify_reso:
        #             low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), factor, factor)
        #             high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
        #             different_res_imgs.append(high_res_moasic_img)
        #         else:
        #             low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0), factor, factor).squeeze(0).permute(1,2,0).to(self.device)
        #             different_res_imgs.append(low_res_img)
        #     self.targets = [img.detach().clone() for img in different_res_imgs]
        #     # self.targets[-1] = self.targets[-1] + torch.randn_like(self.targets[-1]) * self.gaussian_noise_std
        #     # print(f"add gaussian noise with std: {self.gaussian_noise_std}")
        # else:
        #     self.original_targets = [img.detach().clone() for img in different_res_imgs]
        #     self.targets = [img.detach().clone() for img in different_res_imgs]

        if self.add_gaussian_noise==1:
            self.original_targets = [img.detach().clone() for img in different_res_imgs]
            self.targets = [img.detach().clone() for img in different_res_imgs]
            stds = torch.linspace(0, self.gaussian_noise_std, steps=self.num_iterations)
            self.gaussian_targets = []
            H, W, C = img.shape
            gaussian_num = self.iterations_for_upsampling[-1]
            gaussians = torch.randn(gaussian_num, self.end_reso, self.end_reso, C).to(self.device) * self.gaussian_noise_std
            cumulated_gaussians = torch.cumsum(gaussians, dim=0)
            for i, std in enumerate(stds):
                target_i = np.where(np.array(self.iterations_for_upsampling) > i)[0]
                target_i = target_i[0] if len(target_i) > 0 else -1
                if i + 1 <= gaussian_num:
                    self.gaussian_targets.append(self.targets[target_i].clone() + cumulated_gaussians[i])
                else:
                    self.gaussian_targets.append(self.targets[target_i].clone() + cumulated_gaussians[-1])
        elif self.add_gaussian_noise==2:
            self.original_targets = [img.detach().clone() for img in different_res_imgs]
            self.targets = [img.detach().clone() for img in different_res_imgs]
            H, W, C = img.shape
            gaussian_num = self.iterations_for_upsampling[-1]
            gaussians = torch.randn(gaussian_num, self.end_reso, self.end_reso, C).to(self.device) * self.gaussian_noise_std
            self.gaussian_targets = []
            for i in range(self.num_iterations):
                target_i = np.where(np.array(self.iterations_for_upsampling) > i)[0]
                target_i = target_i[0] if len(target_i) > 0 else -1
                self.gaussian_targets.append(self.targets[target_i].clone() + gaussians[i])
        elif self.add_gaussian_noise==4:
            img = img.clone() + torch.randn_like(img) * self.gaussian_noise_std
            different_res_imgs = []
            for factor in self.factors:
                if unify_reso:
                    low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), factor, factor)
                    high_res_moasic_img = torch.nn.functional.interpolate(low_res_img, size=img.shape[:2], mode='nearest').squeeze(0).permute(1,2,0).to(self.device)
                    different_res_imgs.append(high_res_moasic_img)
                else:
                    low_res_img = torch.nn.functional.avg_pool2d(img.permute(2,0,1).unsqueeze(0).clone(), factor, factor).squeeze(0).permute(1,2,0).to(self.device)
                    different_res_imgs.append(low_res_img)
            self.original_targets = [img.detach().clone() for img in different_res_imgs]
            self.targets = [img.detach().clone() for img in different_res_imgs]
            H, W, C = img.shape
            gaussian_num = self.iterations_for_upsampling[-1]
            self.gaussian_targets = []
            for i in range(self.num_iterations):
                target_i = np.where(np.array(self.iterations_for_upsampling) > i)[0]
                target_i = target_i[0] if len(target_i) > 0 else -1
                if i + 1 <= gaussian_num:
                    self.gaussian_targets.append(self.targets[target_i].clone())
                else:
                    self.gaussian_targets.append(self.targets[target_i].clone())
        elif self.add_gaussian_noise==0:
            self.original_targets = [img.detach().clone() for img in different_res_imgs]
            self.targets = [img.detach().clone() for img in different_res_imgs]

        if self.loss_fn_str in ['Lowrank']:
            # start_weight = 1e-5
            # end_weight = 7e-4
            start_weight = 1e-5
            end_weight = 1e-5
            # start_weight = 1e-4
            # end_weight = 1e-9
            grow_factor = (end_weight/start_weight) ** (1/(len(self.interm_resos)-1))
            # self.lowrank_weight = np.linspace(start_weight, end_weight, len(self.interm_resos))
            self.lowrank_weight = [start_weight * grow_factor**i for i in range(len(self.interm_resos))]
            print(f"lowrank_weight:{self.lowrank_weight}")

        self.edge_mask = None
        if self.loss_fn_str in ['Robust', 'TV', 'Detect']:
            self.edge_mask = self.get_edge_mask(img).detach().clone()
            self.smoothed_targets = self.original_targets.copy()
        
        if self.loss_fn_str in ['Detect']:
            self.perturbations = []
        

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
        payload = self.payload
        contract_core_num = int(np.log2(output_reso))
        if self.unify_reso == False:
            # einsum_str = self.build_einsum_chain(contract_core_num)
            einsum_str = self.einsum_strs[self.interm_resos.index(output_reso)]
            cores = self.tn[:contract_core_num-1] + [torch.mean(self.tn[contract_core_num-1], dim=-1) if len(self.tn[contract_core_num-1].shape)==3 else self.tn[contract_core_num-1]]
            output = contract(einsum_str, *cores, optimize=self.path[self.interm_resos.index(output_reso)])
        elif self.unify_reso == True:
            output_reso = self.end_reso
            # einsum_str = self.build_einsum_chain(int(np.log2(self.end_reso)))
            einsum_str = self.einsum_strs[self.interm_resos.index(output_reso)]
            cores = self.tn[:contract_core_num+1] + self.tn_tmp[contract_core_num+1:]
            output = contract(einsum_str, *cores, optimize=self.path)
        # output = torch.einsum(einsum_str, *cores)

        shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source = get_qtt_shape(output_reso, dim=2)
        output = output.reshape([payload] + shape_factors)
        factor_target_to_source = [i+1 for i in factor_target_to_source]
        output = output.permute(factor_target_to_source + [0])
        output = output.reshape(shape_source + [payload])
        # print(output.shape)
        return output
    
    def add_gaussian_to_TN(self, reso, scale: float = 0.01):
        """
        Add Gaussian noise to the trainable cores up to the given resolution.

        Args:
            reso (int): Current grid resolution (side length). Noise is added to
                        cores whose index ≤ log2(reso).
            scale (float): Additional scaling factor for the noise strength.
        """
        with torch.no_grad():  # avoid contaminating gradients
            upper_idx = int(np.log2(reso))
            for idx, core in enumerate(self.tn):
                if idx > upper_idx:
                    break
                noise = torch.randn_like(core) * scale
                self.tn[idx].data += noise.data
    
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
                if self.optimize_type == 'part':
                    if i > current_exclude_ind:
                        torch_params[str(i)].requires_grad = False
                    else:
                        torch_params[str(i)].requires_grad = True
                elif self.optimize_type == 'one':
                    if i != current_exclude_ind:
                        torch_params[str(i)].requires_grad = False
                    else:
                        torch_params[str(i)].requires_grad = True
        # for k, v in torch_params.items():
        #     print(k, v.shape, v.requires_grad)
        
        out = [{'params': torch_params.values(), 'lr': lr}]
        optimizer = torch.optim.Adam(out)
        self.tn = [p for _, p in torch_params.items()]

        return optimizer

    # def set_optimizer(self, current_grid, lr):
    #     # params, skeleton = qtn.pack(tt)
    #     params = {
    #         i: t
    #         for i, t in enumerate(self.tn)
    #     }

    #     torch_params = torch.nn.ParameterDict({
    #         # torch requires strings as keys
    #         str(i): torch.nn.Parameter(initial).to(self.device)
    #         for i, initial in params.items()
    #     })
    #     start_exclude_ind = int(np.log2(self.init_reso))
    #     current_exclude_ind = int(np.log2(current_grid))
    #     end_exclude_ind = int(np.log2(self.end_reso))
    #     core_indices_to_exclude = [str(i+1) for i in range(start_exclude_ind, end_exclude_ind, 1)]
    #     # if core_indices_to_exclude length is not 0, remove the corresponding cores from the torch_params making them non-trainable
    #     if current_exclude_ind <= start_exclude_ind - 1:
    #         for i in core_indices_to_exclude:
    #             torch_params[i].requires_grad = False
    #     else:
    #         for i in range(end_exclude_ind):
    #             if self.optimize_type == 'part':
    #                 if i > current_exclude_ind:
    #                     torch_params[str(i)].requires_grad = False
    #                 else:
    #                     torch_params[str(i)].requires_grad = True
    #             elif self.optimize_type == 'one':
    #                 if i != current_exclude_ind:
    #                     torch_params[str(i)].requires_grad = False
    #                 else:
    #                     torch_params[str(i)].requires_grad = True
    #     # for k, v in torch_params.items():
    #     #     print(k, v.shape, v.requires_grad)

    #     # 每个核是一个 param group
    #     param_groups = []
    #     for name, p in torch_params.items():
    #         param_groups.append({'params': [p], 'lr': lr})
        
    #     optimizer = torch.optim.Adam(param_groups)
    #     self.tn = [p for _, p in torch_params.items()]

    #     return optimizer
    
    def adjust_optimizer(self, current_grid):
        start_exclude_ind = int(np.log2(self.init_reso))
        current_exclude_ind = int(np.log2(current_grid))
        end_exclude_ind = int(np.log2(self.end_reso))

        if self.optimize_type == 'part':
            for idx, param in enumerate(self.tn):
                if idx <= current_exclude_ind:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.optimize_type == 'one':
            if current_grid > self.init_reso:
                for idx, param in enumerate(self.tn):
                    if idx in [current_exclude_ind, current_exclude_ind-1, current_exclude_ind-2, current_exclude_ind-3]:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif current_grid == self.init_reso:
                for idx, param in enumerate(self.tn):
                    if idx <= current_exclude_ind:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if current_grid == self.end_reso:
                for idx, param in enumerate(self.tn):
                    param.requires_grad = True

    def compute_nuclear_norm(self, tn=None):
        if tn is None:
            tn = self.tn.copy()
        nuclear_norm = 0
        # weights = [8, 8, 4, 4, 2, 2, 1, 1, 0.1]
        # weights = [25.6, 12.8, 6.4, 3.2, 1.6, .8, .4, .2, .1]
        # weights = [1, 1, 1, 1, 1, 1, 0, 0, 0]
        # weights = [0.1, 0.1, 0.8, 1.6, 3.2, 6.4, 6.4, 6.4, 6.4]
        # weights = [0, 2, 4, 8, 16, 32, 32, 32, 32]
        # weights = [0, 0, 4, 8, 16, 32, 32, 32, 32]
        weights = [0, 0, 0, 0, 1, 2, 4, 8, 16]
        for i, t in enumerate(tn):
            if t.requires_grad == False:
                continue
            if len(t.shape) == 3:
                t = t.permute(0, 2, 1)
                t = t.reshape(t.shape[0]*t.shape[1], t.shape[2])
            # nuclear_norm += torch.norm(t, 'nuc')
            # nuclear_norm += np.sqrt(1/max(t.shape[0], t.shape[1])) * torch.norm(t, 'nuc')
            nuclear_norm += weights[i] * torch.norm(t, 'nuc')
        return nuclear_norm
    
    def median_filter(self, x: torch.Tensor, kernel_size: int = 3, padding: bool = True) -> torch.Tensor:
        """
        对 4D 张量或 3D 张量做中值滤波。
        输入 x 的形状可以是 (B, C, H, W) 或 (C, H, W)，输出保持同样的 shape。

        Args:
            x: 输入图像张量，取值范围任意浮点数或整数。
            kernel_size: 滤波窗口大小（须为奇数）。
            padding: 是否在边界处做 zero padding，使输出尺寸与输入相同。

        Returns:
            滤波后的张量，shape 同 x。
        """
        # 如果是 3D，先增加 batch 维度
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)

        B, C, H, W = x.shape
        if padding:
            pad = kernel_size // 2
            x = F.pad(x, (pad, pad, pad, pad), mode='reflect')  # 边界反射

        # 做 unfold，提取每个像素邻域
        patches = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        # patches: (B, C, H, W, k, k)
        patches = patches.contiguous().view(B, C, H, W, -1)
        # 计算中值
        median = patches.median(dim=-1)[0]  # shape (B, C, H, W)

        return median if is_batched else median.squeeze(0)
    
    def orthogonal_loss(self, x, x_rec):
        # x, x_rec: [B, C, H, W]
        x_flat     = x.reshape(x.shape[0], -1)        # [B, N]
        xrec_flat  = x_rec.reshape(x_rec.shape[0], -1)
        inner_prod = torch.sum(x_flat * xrec_flat, dim=1)  # [B]
        return torch.mean(inner_prod**2)
    
    def compute_total_variation_loss(self):
        """
        Computes the total variation loss for the tensor network.

        Parameters:
        - tn: the tensor network with a number of cores len(tn.tensors)

        Returns:
        The total variation loss.
        """
        total_variation_loss = 0
        num_compressed_params = 0
        for tensor in self.tn:
            num_compressed_params += tensor.numel()

            # Ensure tensor has three dimensions
            tensor = tensor.data
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)

            # Compute the squared differences in the horizontal direction
            horizontal_diff = torch.pow(tensor[:, :, 1:] - tensor[:, :, :-1], 1)

            # Compute the squared differences in the vertical direction
            vertical_diff = torch.pow(tensor[:, 1:, :] - tensor[:, :-1, :], 1)

            # Sum up the horizontal and vertical differences
            total_variation_loss += torch.sum(horizontal_diff) + torch.sum(vertical_diff)

        # Normalize by the number of paramters
        return total_variation_loss / num_compressed_params
    
    def contrast_loss(self, img):
        # 定义 Laplacian 核
        lap = torch.tensor([[0., -1.,  0.],
                            [-1., 4., -1.],
                            [0., -1.,  0.]], device=self.device)
        lap = lap.view(1,1,3,3)   # shape (1,1,3,3)

        # 如果是 RGB，就对每个通道都算，再求平均
        # recon: (H,W,C) -> (1,C,H,W)
        r = img.permute(2,0,1).unsqueeze(1)     # (C,1,H,W)
        r = r.view(-1,1, r.shape[-2], r.shape[-1])  # 合并 batch： (C,1,H,W)

        # apply laplacian
        hpf = F.conv2d(r, lap, padding=1).abs()   # (C,1,H,W)
        high_freq = hpf.view(img.shape[2], img.shape[0], img.shape[1])  
        # -> (C,H,W)

        # 平均到一个标量
        hf_mean = high_freq.mean()

        # 对比度损失：希望 high_freq 大，就最小化它的负值
        contrast_loss = - hf_mean

        return contrast_loss

    def cov_regularizer(self, lam=1e-4):
        covs = []
        for core in self.tn:
            u = core.permute(1, 0, 2).reshape(1, -1)          # flatten
            g = F.layer_norm(u, u.shape[1:])            # optional非线性
            cov = (g @ g.T) / g.size(1)                 # Σ_i
            covs.append(cov)
        loss = 0.
        for i in range(len(covs)-1):
            I = torch.eye(covs[i].shape[0], device=self.device)
            loss += (covs[i] - I).pow(2).mean()
        return lam * loss

    def low_frequency_penalty(self, x, radius_ratio=0.1):
        """
        计算图像张量的低频能量惩罚 (L2 norm)
        
        Args:
            x: [B, C, H, W] 输入图像张量 (0~1 或 -1~1)
            radius_ratio: 低频掩码的半径比例 (相对于 min(H, W))
                        比如 0.1 表示频谱中心 10% 区域是低频

        Returns:
            penalty: 标量张量，低频能量
        """
        H, W, C = x.shape

        # --- 1. 频域变换 ---
        # torch.fft.fftn 支持多通道
        freq = torch.fft.fftn(x, dim=(-3, -2))
        freq = torch.fft.fftshift(freq, dim=(-3, -2))

        mag = torch.abs(freq)
        phase = torch.angle(freq)

        # --- 2. 构建低频掩码 ---
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        cy, cx = H // 2, W // 2
        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).to(x.device)
        radius = radius_ratio * min(H, W)
        mask1 = (dist <= radius).float()  # [H, W]
        mask2 = (dist >= 0.05).float()  # [H, W]
        mask = mask1 * mask2

        # 扩展到 [H, W, C]
        mask = mask.unsqueeze(2)

        # --- 3. 低频能量 ---
        lowfreq_mag = mag * mask
        lowfreq_phase = phase * mask
        loss = self.TV(lowfreq_mag)

        return loss
    
    def count_parameters(self):
        num = 0
        for i, core in enumerate(self.tn):
            num += core.numel()
        return num
    
    # ---- 可选：低频投影工具（FFT 半径掩膜）----
    def lowfreq_proj(self, x, r0=0.12):
        # x: (B,C,H,W), r0 in (0, 0.5]
        X = torch.fft.fft2(x, norm="ortho")
        B,C,H,W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-.5,.5,H, device=x.device),
            torch.linspace(-.5,.5,W, device=x.device),
            indexing='ij'
        )
        mask = (yy**2 + xx**2).sqrt() <= r0
        X = X * mask[None,None]
        return torch.fft.ifft2(X, norm="ortho").real
    
    def forward(self, reso, aux=None, alpha=None, ite=None):
        assert reso in self.interm_resos, f"Invalid intermediate resolution, should be one of {self.interm_resos}"

        # L2
        reso_index = self.interm_resos.index(reso)
        recon = self.custom_contract_qtt(output_reso=reso)
        recon_ = recon.clone()
        self.img = recon.detach().clone()
        if aux is None:
            loss = self.loss_fn(recon, self.targets[reso_index])
        else:
            loss = self.loss_fn(recon, aux)
        loss_recon = loss.item()

        if self.loss_fn_str == "Lowrank":
            nuclear_norm = self.compute_nuclear_norm(tn=self.tn.copy())
            # loss = loss + 0.00005 * nuclear_norm
            # loss = loss + self.lowrank_weight[reso_index] * nuclear_norm + 0.03 * self.compute_total_variation_loss()
            loss = loss + self.lowrank_weight[reso_index] * nuclear_norm

        if self.loss_fn_str == "add_gaussian":
            recon = recon + self.gaussian_targets[ite]-self.targets[reso_index]
            loss = self.loss_fn(recon, self.targets[reso_index])

        if reso_index >= 1 and self.loss_fn_str in ["TV"]:
            # loss = torch.nn.functional.l1_loss(recon, self.targets[reso_index])
            # nuclear_norm = self.compute_nuclear_norm(tn=self.tn.copy())
            # loss = loss + 1e-5 * nuclear_norm
            # loss = loss + 0.03 * self.TV(recon.clone(), p=1) + 0.5 * self.loss_fn(recon.clone()*self.edge_mask, self.targets[reso_index]*self.edge_mask)
            # loss = loss + 0.03 * self.TV(recon.clone(), p=1) + 0.01 * self.contrast_loss(recon.clone())
            loss = loss + self.regularization_weight * self.TV(recon.clone(), p=1)
            # loss = loss + self.regularization_weight * self.TV(recon.clone(), p=1) + self.regularization_weight * self.loss_fn(recon.clone()*self.edge_mask, self.targets[reso_index]*self.edge_mask)
            # loss = loss + self.compute_total_variation_loss()

        if reso_index >= 1 and self.loss_fn_str in ["maskTV"]:
            # if not self.add_gaussian_noise:
            #     loss = loss + self.regularization_weight * self.TV(self.attention_mask * recon.clone(), p=1) + self.regularization_weight / 4 * self.TV((1-self.attention_mask) * recon.clone(), p=1)
            # elif self.add_gaussian_noise:
            #     loss = loss + self.regularization_weight * self.TV((1-self.attention_mask) * recon.clone(), p=1)
            # img_tmp = torch.nn.functional.interpolate(self.img.clone().permute(2, 0, 1).unsqueeze(0), size=(self.original_reso, self.original_reso), mode='bilinear').detach()
            # attention_mask, attention_maps, pure_area = self.attention_mask_generator.compute_attention_mask(img_tmp.to(self.device), return_attention_maps=True)
            # attention_mask = torch.nn.functional.interpolate(pure_area, size=(self.end_reso, self.end_reso), mode='bilinear').detach().clone()
            # attention_mask = attention_mask.squeeze(0).permute(1,2,0).expand(-1, -1, 3).detach().clone()
            # self.attention_mask = attention_mask.detach().clone()

            loss = loss + self.regularization_weight * self.TV(self.attention_mask * recon.clone(), p=1)
            # loss = loss + self.regularization_weight / 10000 * self.low_frequency_penalty(recon.clone())

        if reso_index >= 0 and self.loss_fn_str in ["strideTV"]:
            # loss = loss + self.regularization_weight * self.strided_TV(self.attention_mask * recon.clone(), p=1, stride=self.end_reso//self.current_reso)
            # reso_tmp = round(2 ** np.ceil(np.log2(min(self.current_reso, self.original_reso))))
            loss = loss + self.regularization_weight * self.strided_TV(self.attention_mask * recon.clone(), p=1, stride=self.end_reso//self.init_reso, scale_num=round(np.log2(self.end_reso//self.init_reso))+1)

        if reso_index >= 1 and ite >= 100 and self.loss_fn_str in ["Align"]:
            loss = loss + self.regularization_weight * self.TV(self.attention_mask * recon.clone(), p=1)
            if reso_index == len(self.interm_resos)-1:
                resid = self.targets[reso_index] - recon.clone()
                self.attention_mask_generator.classifier.eval()
                x = recon.detach().clone().permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
                logits = self.attention_mask_generator.classifier(x)
                
                # y = torch.argmax(logits, dim=1)
                # adv_loss = F.cross_entropy(logits, y)
                
                # # 负边际：鼓励朝“最危险”的方向（top-2 间距缩小）
                # top2 = logits.topk(2, dim=1).indices
                # z1 = logits.gather(1, top2[:, :1])
                # z2 = logits.gather(1, top2[:, 1:2])
                # adv_loss = -(z1 - z2).mean()

                # 计算 KL 散度用于无监督攻击
                prob = F.softmax(logits, dim=1)
                even_dist = torch.ones_like(prob) / prob.size(1)
                # kl_loss = F.kl_div(torch.log(prob), even_dist, reduction='batchmean')
                kl_loss = F.kl_div(torch.log(even_dist), prob, reduction='batchmean')
                adv_loss = -kl_loss

                grad = torch.autograd.grad(adv_loss, x, retain_graph=False, create_graph=False)[0]
                # grad = self.lowfreq_proj(grad, ).sign()
                # resid = self.lowfreq_proj(resid.clone().permute(2,0,1).unsqueeze(0), ).squeeze(0).permute(1,2,0).clone()
                grad = grad.squeeze(0).permute(1,2,0).detach().clone()

                cos_sim = F.cosine_similarity(grad.flatten(), resid.flatten(), dim=0)
                print(cos_sim.item())
                loss = loss - .3 * cos_sim

        if reso_index >= 1 and self.loss_fn_str in ["CSR"]:
            residual = self.custom_contract_qtt(output_reso=self.interm_resos[reso_index-1]).detach() - recon.clone()
            # use wavelet or dct to extract high frequency component
            Yl, Yh_list = self.dwt(residual.permute(2,0,1).unsqueeze(0))
            Yh = Yh_list[0]
            B, K, C, H, W = Yh.shape
            Yh = Yh.reshape(B, K*C, H, W)

            # loss += loss + 1 * Yh.abs().mean() + self.regularization_weight * self.TV(self.attention_mask * recon.clone(), p=1)
            print(f"loss: {loss.item()}, Yh.abs().mean(): {Yh.abs().mean().item()}")
            loss += loss + 0.03 * Yl.abs().mean()

        if reso_index >= 1 and self.loss_fn_str in ["cov"]:
            cov_loss = self.cov_regularizer(lam=1e5)
            # print(f"loss: {loss.item()}, cov_regularizer: {cov_loss.item()}")
            loss = loss + cov_loss

        if reso_index == len(self.interm_resos)-1 and self.loss_fn_str == "adv":
            # optimize the perturbation
            img = recon.clone()
            perturb_lr = 0.1
            ssim = SSIM(data_range=1.0).to(self.device)
            adv_loss = ssim(img.permute(2, 0, 1).unsqueeze(0), self.targets[reso_index].permute(2, 0, 1).unsqueeze(0))
            grad = torch.autograd.grad(adv_loss, [img])[0]
            img = img + perturb_lr * torch.sign(grad.detach())
            img = torch.clamp(img, 0.0, 1.0)
            loss = self.loss_fn(img, self.targets[reso_index])

            # recon = self.custom_contract_qtt(output_reso=self.interm_resos[-1])
            loss += 1*self.loss_fn(recon.clone(), self.current_reso_init_img)

        if reso_index >= 1 and self.loss_fn_str == "Progressive" and alpha != 1.0:
            recon_prev = self.custom_contract_qtt(output_reso=self.interm_resos[reso_index-1]).detach().clone()
            # loss = alpha * loss + (1 - alpha) * self.loss_fn(recon_, self.targets[reso_index-1])
            loss = self.loss_fn(alpha * recon + (1 - alpha) * recon_prev, self.targets[reso_index])
            # loss = alpha * loss + (1 - alpha) * self.loss_fn(recon_prev, self.targets[reso_index-1])

        if reso_index >= 1 and self.loss_fn_str in ["Smooth", "Detect"] and alpha != 1.0:
            # recon_prev = self.custom_contract_qtt(output_reso=self.interm_resos[reso_index-1]).detach().clone()
            # recon_smooth = alpha * recon + (1 - alpha) * recon_prev
            target_smooth = alpha * self.targets[reso_index] + (1 - alpha) * self.targets[reso_index-1]
            loss = self.loss_fn(recon.clone(), target_smooth)

        if reso_index >= 1 and self.loss_fn_str == "Robust":
            # ssim = SSIM(data_range=1.0).to(self.device)
            recon_prev = torch.nn.functional.avg_pool2d(recon.clone().permute(2,0,1).unsqueeze(0), 2, 2)
            recon_prev = torch.nn.functional.interpolate(recon_prev, size=recon.shape[:2], mode='nearest').squeeze(0).permute(1,2,0)
            recon_after = torch.nn.functional.interpolate(recon.clone().permute(2,0,1).unsqueeze(0), size=(recon.shape[0]*2, recon.shape[1]*2), mode='bilinear')
            recon_after = torch.nn.functional.interpolate(recon_after, size=recon.shape[:2], mode='nearest').squeeze(0).permute(1,2,0)

            X_d_1 = self.last_reso_final_img.clone()
            if reso_index == len(self.interm_resos) - 1:
                X_d_2 =  torch.nn.functional.interpolate(self.original_targets[reso_index].clone().permute(2,0,1).unsqueeze(0), size=(recon.shape[0]*2, recon.shape[1]*2), mode='bilinear')
                X_d_2 =  torch.nn.functional.interpolate(X_d_2, size=self.original_targets[reso_index].shape[:2], mode='nearest').squeeze(0).permute(1,2,0)
                # loss = 0.5 * self.loss_fn(recon_prev, X_d_1) + 0.5 * self.loss_fn(recon_after, X_d_2) + 0.5 * self.loss_fn(recon_, self.smoothed_targets[reso_index]) + 0.5 * self.loss_fn(recon_.clone()*self.edge_mask, self.original_targets[reso_index].clone()*self.edge_mask)
                # loss = 0.5 * self.loss_fn(recon_prev, X_d_1) + 0.5 * self.loss_fn(recon_after, X_d_2)
                loss = 0.5*self.loss_fn(recon_prev, X_d_1) + 0.5*self.loss_fn(recon_after, X_d_2) + 0.5 * self.loss_fn(recon_, self.smoothed_targets[reso_index])
                # loss = 0.5*self.loss_fn(recon_prev, X_d_1) + 0.5*self.loss_fn(recon_after, X_d_2) + 0.5 * self.loss_fn(recon_.clone()*self.edge_mask, self.original_targets[reso_index].clone()*self.edge_mask)
            else:
                X_d_2 = self.targets[reso_index+1].clone()
                # loss = 0.5*self.loss_fn(recon_prev, X_d_1) + 0.5*self.loss_fn(recon_after, X_d_2) + 0.5 * self.loss_fn(recon_, self.smoothed_targets[reso_index]) + 0.5 * self.loss_fn(recon_.clone()*self.edge_mask, self.original_targets[reso_index].clone()*self.edge_mask)
                # loss = 0.5 * self.loss_fn(recon_prev, X_d_1) + 0.5 * self.loss_fn(recon_after, X_d_2)
                loss = 0.5*self.loss_fn(recon_prev, X_d_1) + 0.5*self.loss_fn(recon_after, X_d_2) + 0.5 * self.loss_fn(recon_, self.smoothed_targets[reso_index])
                # loss = 0.5*self.loss_fn(recon_prev, X_d_1) + 0.5*self.loss_fn(recon_after, X_d_2) + 0.5 * self.loss_fn(recon_.clone()*self.edge_mask, self.original_targets[reso_index].clone()*self.edge_mask)

            # loss += 0.001 * self.contrast_loss(recon.clone())

        if reso_index >= 1 and self.loss_fn_str == "orth":
            loss = loss + 1e-9 * self.orthogonal_loss((recon.clone()-self.targets[reso_index]).permute(2,0,1).unsqueeze(0), self.targets[reso_index-1].permute(2,0,1).unsqueeze(0))

        if self.loss_fn_str == "every":
            recons = [self.custom_contract_qtt(output_reso=self.interm_resos[i]) for i in range(reso_index+1)]
            loss = 0
            for j, recon in enumerate(recons):
                loss += self.loss_fn(recon, self.targets[j])
            
            # loss = loss / len(recons)
            # loss += 1e-5 * self.compute_nuclear_norm(tn=self.tn.copy())
            # loss += 0.01 * self.TV(recon.clone(), p=1)
            # pass
            
        return loss, loss_recon
    
    def train(self, target, args, target_index=None, visualize=False, cln_target=None, visualize_dir='', record_loss=False, clf=None):
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
        aux = None
        recons_multi_resos = []
        time1 = time.time()
        mse_history = []
        optimizer = self.set_optimizer(self.init_reso, lr=lr)
        ite = 0

        # logits_history = []
        # logits_softmax_history = []
        # logits_log_softmax_history = []
        # KL_history = []
        # JS_history = []
        # Wasserstein_history = []
        # CosSim_history = []
        # adversarial_flags = []
        # adversarial_step = 0
        if self.attention_mask_generator is not None:
            target_tmp = torch.nn.functional.interpolate(target.detach().clone().permute(2,0,1).unsqueeze(0), size=(self.original_reso, self.original_reso), mode='bilinear').detach().clone()
            attention_mask, attention_maps, pure_area = self.attention_mask_generator.compute_attention_mask(target_tmp.to(self.device), return_attention_maps=True)
            attention_mask = torch.nn.functional.interpolate(pure_area, size=(self.end_reso, self.end_reso), mode='bilinear').detach().clone()
            attention_mask = attention_mask.squeeze(0).permute(1,2,0).expand(-1, -1, 3).detach().clone()
            self.attention_mask = attention_mask.detach().clone()
            if visualize:
                self.attention_mask_generator.visualize_attention(target_tmp.to(self.device), os.path.join(visualize_dir, f'attention_mask_sample{target_index}.png'))

        for i, reso in enumerate(self.interm_resos):
            # print(f"Start purification for resolution {reso}*{reso}")
            self.adjust_optimizer(reso)
            # Scheduler

            lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, int(iterations[i]*0.1))
            lr_gamma = calculate_gamma(lr, args.lr_decay_factor_until_next_upsampling, iterations[i]-int(iterations[i]*0.1))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

            if self.img_TV:
                # auxiliary variable for TV
                if reso == self.interm_resos[-2]:
                    noisy_target_downsampled = F.interpolate(target.permute(2,0,1).unsqueeze(0),
                                                        size=(int(args.end_reso/2),int(args.end_reso/2)),
                                                        mode='bilinear')[0].permute(1,2,0)
                    if self.unify_reso:
                        noisy_target_downsampled = F.interpolate(target.permute(2,0,1).unsqueeze(0),
                                                        size=(int(args.end_reso),int(args.end_reso)),
                                                        mode='nearest')[0].permute(1,2,0)

                    noisy_target_downsampled = noisy_target_downsampled.to(self.device)
                    aux = noisy_target_downsampled.detach().to(self.device)
                    aux = nn.Parameter(aux)
                    Sparse = nn.Parameter(torch.zeros_like(noisy_target_downsampled)) # sparse term
                    optimizer.add_param_group({'params': aux})  # Optional: set a different learning rate

                    for group in optimizer.param_groups:
                        for param in group['params']:
                            print(param.shape)
                if reso == self.interm_resos[-1]:
                    aux = None
            if self.loss_fn_str == "adv" and reso == self.interm_resos[-1]:
                self.current_reso_init_img = torch.nn.functional.interpolate(self.custom_contract_qtt(output_reso=int(reso/2)).permute(2, 0, 1).unsqueeze(0), size=(reso, reso), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).detach().clone()

            if self.loss_fn_str in ["Robust", "Detect"]:
            # if self.loss_fn_str in ["Robust"]:
                if aux is not None:
                    self.smoothed_targets[i] = aux.detach().clone()
                    # self.targets[i] = self.original_targets[i].detach().clone()
                if reso == self.interm_resos[-1]:
                    aux = None
                    optimizer.param_groups.pop(1)
                else:
                    noisy_target_downsampled = self.original_targets[i+1].detach().clone()
                    aux = nn.Parameter(self.original_targets[i+1].detach().clone())
                    aux.requires_grad = True
                    # 删除 param_groups 中 aux 参数
                    if len(optimizer.param_groups) == 2:
                        optimizer.param_groups.pop(1)
                    optimizer.add_param_group({'params': aux, 'lr': 0.1})  # Optional: set a different learning rate
                    # for group in optimizer.param_groups:
                    #     for param in group['params']:
                    #         print(param.shape)

                # noisy_target_downsampled = self.targets[i].detach().clone()
                # aux = nn.Parameter(self.targets[i].detach().clone())
                # aux.requires_grad = True
                # # 删除 param_groups 中 aux 参数
                # if len(optimizer.param_groups) == 2:
                #     optimizer.param_groups.pop(1)
                # optimizer.add_param_group({'params': aux})  # Optional: set a different learning rate
                # # for group in optimizer.param_groups:
                # #     for param in group['params']:
                # #         print(param.shape)
    
            # for j in range(iterations[i]):
            j = 0
            while j < iterations[i]:
                optimizer.zero_grad()
                if self.loss_fn_str == "Progressive":
                    loss, loss_recon = self.forward(reso, aux=aux, alpha=min((j//10*10)/iterations[i]*10, 1))
                # elif self.loss_fn_str == "Smooth":
                #     if iterations[i] <= 50:
                #         loss, loss_recon = self.forward(reso, aux=aux, alpha=min(j/iterations[i], 1))
                #     else:
                #         loss, loss_recon = self.forward(reso, aux=aux, alpha=min(j/50, 1))
                elif self.loss_fn_str in ["Smooth", "Detect"]:
                    # alpha = min(alpha_min + (alpha_max - alpha_min) * np.log(j+1) / np.log((iterations[i]+1)), 1)
                    # alpha = alpha_min * (alpha_max/alpha_min) ** ((j+1)*2/iterations[i])
                    # alpha = min(alpha, 1)
                    alpha=min((j//10)*10/iterations[i]*10+0.5, 1)
                    loss, loss_recon = self.forward(reso, aux=None, alpha=alpha)
                elif self.loss_fn_str == "Robust":
                    loss, loss_recon = self.forward(reso, aux=None)   
                elif self.add_gaussian_noise != 0:
                    loss, loss_recon = self.forward(reso, aux=self.gaussian_targets[ite], ite=max(ite-1, 0))
                else:
                    loss, loss_recon = self.forward(reso, aux=aux, ite=j)
                if (self.img_TV and (reso == self.interm_resos[-2])) or (self.loss_fn_str in ["Robust"] and reso != self.interm_resos[-1]):
                # if (self.img_TV and (reso == self.interm_resos[-2])) or (self.loss_fn_str in ["Robust"] and reso != self.interm_resos[-1]):
                    # loss = loss + 0.5*torch.norm(aux-noisy_target_downsampled, p='fro') + 0.5 * self.TV(aux, p=1)
                    loss = loss + 0.5 * self.TV(aux, p=1) + 0.0005 * torch.norm(aux-noisy_target_downsampled, p='fro')
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

                if visualize and self.loss_fn_str == "Smooth" and ite % 1 == 0:
                    # visualize the intermediate results, only image
                    plt.imshow(np.clip(self.img.detach().cpu().numpy(), 0, 1))
                    # plt.title("Intermediate result at step {} of resolution {}x{}".format(ite, reso, reso))
                    save_path = f"{visualize_dir}/Sample{target_index}/step{str(ite).zfill(4)}_reso{reso}.png"
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    plt.axis('off')
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()

                    alpha = min(j/iterations[i]*2, 1)
                    if i == 0:
                        target_show = self.targets[0]
                    else:
                        target_show = alpha * self.targets[i] + (1 - alpha) * self.targets[i-1]
                    plt.imshow(np.clip(target_show.detach().cpu().numpy(), 0, 1))
                    save_path = f"{visualize_dir}/Sample{target_index}_targets/step{str(ite).zfill(4)}_reso{reso}.png"
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    plt.axis('off')
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()

                # if clf != None and self.loss_fn_str in ["Robust", "Detect"]:
                #     logits = clf(F.interpolate(torch.clamp(self.img.detach(), 0, 1).permute(2, 0, 1).unsqueeze(0), size=self.original_reso, mode='bilinear', align_corners=False))
                #     logits_history.append(logits.detach().cpu().numpy())
                #     logits_softmax = torch.nn.functional.softmax(logits, dim=1)
                #     logits_softmax_history.append(logits_softmax.detach().cpu().numpy())
                #     logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
                #     logits_log_softmax_history.append(logits_log_softmax.detach().cpu().numpy())

                #     # if ite == 0 or ite % 5 != 0:
                #     if ite == 0:
                #         KL_history.append([0])
                #         JS_history.append([0])
                #         Wasserstein_history.append([0])
                #         CosSim_history.append([0])
                #     else:
                #         # kl = torch.nn.functional.kl_div(logits_log_softmax.detach().cpu(), torch.mean(torch.from_numpy(np.stack(logits_log_softmax_history[-6:-1], axis=0)), dim=0), reduction='mean', log_target=True).unsqueeze(0)
                #         kl = torch.nn.functional.kl_div(logits_log_softmax.detach().cpu(), torch.from_numpy(logits_log_softmax_history[-2]), reduction='batchmean', log_target=True).unsqueeze(0)
                #         KL_history.append(kl.detach().cpu().numpy())
                #         # wd = torch.mean(torch.abs(logits_softmax.detach().cpu() - torch.from_numpy(logits_softmax_history[-2]))).unsqueeze(0)
                #         wd = wasserstein_distance([1]*logits_softmax.shape[1], [1]*logits_softmax.shape[1], logits_softmax.detach().cpu().numpy().squeeze(), logits_softmax_history[-2].squeeze())
                #         wd = np.array([wd])
                #         Wasserstein_history.append(wd)
                #         js = jensenshannon(logits_softmax.detach().cpu().numpy().squeeze(), logits_softmax_history[-2].squeeze(), base=2)
                #         js = np.array([js**2])
                #         JS_history.append(js)
                #         # CosSim = torch.nn.functional.cosine_similarity(logits_log_softmax.detach().cpu(), torch.mean(torch.from_numpy(np.stack(logits_log_softmax_history[-6:-1], axis=0)), dim=0))
                #         CosSim = torch.nn.functional.cosine_similarity(logits_log_softmax.detach().cpu(), torch.from_numpy(logits_log_softmax_history[-2]))
                #         CosSim_history.append(CosSim.detach().cpu().numpy())

                #     if i >= len(self.interm_resos) - 3:
                #         if kl >= np.mean(np.array(KL_history[-100:])) + 2*np.std(np.array(KL_history[-100:])):
                #         # if CosSim <= np.mean(np.array(CosSim_history[64:])) - 1*np.std(np.array(CosSim_history[64:])):
                #             self.adversarial_flag = True
                #             # self.add_gaussian_to_TN()
                #             adversarial_step = ite
                #             j -= 1
                #             ite -= 1
                #         # elif adversarial_step != 0 and ite - adversarial_step <= 10:
                #         #     self.adversarial_flag = True
                #         else:
                #             self.adversarial_flag = False
                #     adversarial_flags.append(self.adversarial_flag)
                ite += 1
                j += 1

            
            if self.loss_fn_str == "Robust":
                self.last_reso_final_img = self.img.detach().clone()
                self.current_reso_init_img = torch.nn.functional.interpolate(self.img.detach().clone().permute(2, 0, 1).unsqueeze(0), size=(self.img.shape[0]*2, self.img.shape[0]*2), mode='bilinear', align_corners=False)
                self.current_reso_init_img = torch.nn.functional.interpolate(self.current_reso_init_img, size=(self.img.shape[0], self.img.shape[1]), mode='nearest').squeeze().permute(1, 2, 0).detach().clone()

            recons_multi_resos.append(self.img)
            
            lr *= args.lr_decay_factor
            # for core_id, group in enumerate(optimizer.param_groups):
            #     print(f"Original lr for resolution {reso}*{reso}, core{core_id} is {optimizer.param_groups[core_id]['lr']}")
            # if self.loss_fn_str == "init":
            #     for core_id, group in enumerate(optimizer.param_groups):
            #         if core_id <= int(np.log2(reso)):
            #             # group['lr'] *= args.lr_decay_factor
            #             # optimizer.param_groups[core_id]['lr'] = optimizer.param_groups[core_id]['lr'] * args.lr_decay_factor
            #             group['lr'] *= args.lr_decay_factor
            #             if 'initial_lr' in group:     # 关键！
            #                 del group['initial_lr']
            #         # print(f"Learning rate for resolution {reso}*{reso}, core{core_id} is {optimizer.param_groups[core_id]['lr']}")
            # else:
            #     for core_id, group in enumerate(optimizer.param_groups):
            #         group['lr'] = lr
            #         if 'initial_lr' in group:     # 关键！
            #             del group['initial_lr']
            #         # print(f"Learning rate for resolution {reso}*{reso}, core{core_id} is {optimizer.param_groups[core_id]['lr']}")

            for core_id, group in enumerate(optimizer.param_groups):
                group['lr'] = lr
                if 'initial_lr' in group:     # 关键！
                    del group['initial_lr']
            optimizer.state.clear()

            print(f"End purification for resolution {reso}*{reso}, loss={loss_recon}")
        time2 = time.time()
        print(f"Training time: {time2-time1}, loss: {torch.nn.functional.mse_loss(self.img.detach().cpu(), target).item()}, params_num: {self.count_parameters()}")

        recon = torch.clamp(self.img, 0, 1)
        
        if self.push:
            recon = torch.clamp(recon + 1*(recon-target.to(self.device)), 0, 1)
            recons_multi_resos[-1] = recon

        if self.loss_fn_str == "every":
            recons_multi_resos = [self.custom_contract_qtt(output_reso=self.interm_resos[i]).detach().clone() for i in range(len(self.interm_resos))]

        if visualize:
            gaussian_imgs = self.targets
            self.targets = self.original_targets
            cln_targets = []
            stride_window = 1
            for i in range(len(recons_multi_resos)):
                cln_target_ = torch.nn.functional.avg_pool2d(cln_target.unsqueeze(0), self.factors[::-1][i], self.factors[::-1][i]).squeeze().permute(1, 2, 0)
                if self.unify_reso:
                    cln_target_ = torch.nn.functional.interpolate(cln_target_.permute(2, 0, 1).unsqueeze(0), size=(self.end_reso, self.end_reso), mode='nearest').squeeze().permute(1, 2, 0)
                cln_targets.append(cln_target_)
                stride_window *= 2
            cln_targets = cln_targets[::-1]
            # fig, axes = plt.subplots(8 if self.add_gaussian_noise or self.loss_fn_str == "Detect" else 6, len(recons_multi_resos), figsize=(5 * len(recons_multi_resos), 48 if self.add_gaussian_noise else 36))
            fig, axes = plt.subplots(1, len(recons_multi_resos), figsize=(5 * len(recons_multi_resos), 6))
            # 调整水平和垂直间距
            fig.subplots_adjust(
                left=None,   # 默认不变
                right=None,  # 默认不变
                bottom=None, # 默认不变
                top=None,    # 默认不变
                wspace=0.2,  # 子图之间的宽度空白，默认 0.2
                hspace=0.4   # 子图之间的高度空白，默认 0.2
            )
            for k, r in enumerate(recons_multi_resos):
                if len(recons_multi_resos) == 1:
                    axes[0].imshow(r.detach().cpu().numpy())
                    axes[0].title.set_text(f"{self.init_reso*2**k}x{self.init_reso*2**k} Rec.")
                    axes[1].imshow(self.targets[k].detach().cpu().numpy())
                    axes[1].title.set_text(f"{r.shape[0]}x{r.shape[0]} targets")
                    axes[2].imshow(cln_targets[k].detach().cpu().numpy())
                    axes[2].title.set_text(f"{r.shape[0]}x{r.shape[0]} clean targets")
                    axes[3].hist(r.detach().cpu().numpy().flatten()-self.targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    axes[3].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} reconstruction error, \n mse:{torch.nn.functional.mse_loss(r.detach().cpu(), self.targets[k].detach().cpu()).item()}")
                    axes[4].hist(self.targets[k].detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    axes[4].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} true perturbation, \n mse:{torch.nn.functional.mse_loss(self.targets[k].detach().cpu(), cln_targets[k].detach().cpu()).item()}")
                    axes[5].hist(r.detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    axes[5].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} error between \n recon and clean, mse:{torch.nn.functional.mse_loss(r.detach().cpu(), cln_targets[k].detach().cpu()).item()}")
                else:
                    # axes[0, k].imshow(r.detach().cpu().numpy())
                    # axes[0, k].title.set_text(f"{self.init_reso*2**k}x{self.init_reso*2**k} Rec.")
                    # axes[0, k].axis('off')
                    axes[k].imshow(r.detach().cpu().numpy())
                    axes[k].title.set_text(f"{self.init_reso*2**k}x{self.init_reso*2**k} Rec.")
                    axes[k].axis('off')
                    # axes[1, k].imshow(self.targets[k].detach().cpu().numpy())
                    # axes[1, k].title.set_text(f"{r.shape[0]}x{r.shape[0]} targets")
                    # axes[2, k].imshow(cln_targets[k].detach().cpu().numpy())
                    # axes[2, k].title.set_text(f"{r.shape[0]}x{r.shape[0]} clean targets")
                    # axes[3, k].hist(r.detach().cpu().numpy().flatten()-self.targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    # axes[3, k].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} reconstruction error, \n mse:{torch.nn.functional.mse_loss(r.detach().cpu(), self.targets[k].detach().cpu()).item()}")
                    # axes[4, k].hist(self.targets[k].detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    # axes[4, k].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} true perturbation, \n std:{(self.targets[k].detach().cpu()-cln_targets[k].detach().cpu()).std().item()}, \n mse:{torch.nn.functional.mse_loss(self.targets[k].detach().cpu(), cln_targets[k].detach().cpu()).item()}")
                    # axes[5, k].hist(r.detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    # axes[5, k].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} error between \n recon and clean, \n mse:{torch.nn.functional.mse_loss(r.detach().cpu(), cln_targets[k].detach().cpu()).item()}")
                    # if self.add_gaussian_noise or self.loss_fn_str == "Detect":
                    #     axes[6, k].imshow(gaussian_imgs[k].detach().cpu().numpy())
                    #     axes[6, k].title.set_text(f"{gaussian_imgs[k].shape[0]}x{gaussian_imgs[k].shape[0]} gaussian noised targets")
                    #     axes[7, k].hist(gaussian_imgs[k].detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                    #     axes[7, k].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} difference between \n gaussian noised and clean, \n mse:{torch.nn.functional.mse_loss(gaussian_imgs[k].detach().cpu(), cln_targets[k].detach().cpu()).item()}, \n kurtosis:{scipy.stats.kurtosis(gaussian_imgs[k].detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten())}")
            mse = torch.nn.functional.mse_loss(r, self.targets[i])
            recon_error = recons_multi_resos[-1].detach().cpu().numpy().flatten() - self.targets[-1].detach().cpu().numpy().flatten()
            perturbation = cln_targets[-1].detach().cpu().numpy().flatten() - self.targets[-1].detach().cpu().numpy().flatten()
            cos_theta = np.dot(recon_error, perturbation) / (np.linalg.norm(recon_error) * np.linalg.norm(perturbation))
            theta = np.arccos(cos_theta)
            # fig.suptitle(f"Sample{target_index}, loss={mse.item()}, cos_theta={cos_theta}, theta={theta}, degree={np.degrees(theta)}")
            plt.subplots_adjust(wspace=0.02, hspace=0.02)  
            plt.savefig(f"{visualize_dir}/Sample{target_index}.png", bbox_inches='tight')
            plt.close()

            # visualize reconstructed image
            plt.imshow(recons_multi_resos[-1].detach().cpu().numpy())
            plt.axis('off')              # 去掉坐标轴
            plt.xticks([])               # 去掉 x 轴刻度
            plt.yticks([])               # 去掉 y 轴刻度
            plt.savefig(f"{visualize_dir}/Sample{target_index}_recon.png", bbox_inches='tight')
            plt.close()

            # visualize clean target
            plt.imshow(cln_targets[-1].detach().cpu().numpy())
            plt.axis('off')              # 去掉坐标轴
            plt.xticks([])               # 去掉 x 轴刻度
            plt.yticks([])               # 去掉 y 轴刻度
            plt.savefig(f"{visualize_dir}/Sample{target_index}_cln_target.png", bbox_inches='tight')
            plt.close()

            # visualize target
            plt.imshow(self.targets[-1].detach().cpu().numpy())
            plt.axis('off')              # 去掉坐标轴
            plt.xticks([])               # 去掉 x 轴刻度
            plt.yticks([])               # 去掉 y 轴刻度
            plt.savefig(f"{visualize_dir}/Sample{target_index}_target.png", bbox_inches='tight')
            plt.close()

            delta = self.targets[-1].detach().cpu().numpy().flatten() - cln_targets[-1].detach().cpu().numpy().flatten()
            r = self.targets[-1].detach().cpu().numpy().flatten()-recons_multi_resos[-1].detach().cpu().numpy().flatten()
            ssim = SSIM(data_range=1.0).to(self.device)
            recon_ssim = ssim(recons_multi_resos[-1].detach().cpu().permute(2, 0, 1).unsqueeze(0), cln_targets[-1].detach().cpu().permute(2, 0, 1).unsqueeze(0)).item()

            # 避免除零
            eps = 1e-12

            # 方向余弦 (CosAlign)
            cos_align = np.dot(r, delta) / (np.linalg.norm(r) * np.linalg.norm(delta) + eps)

            # 扰动归因比例 (PAR)
            PAR = np.dot(r, delta) / (np.linalg.norm(delta)**2 + eps)

            # 残余扰动比 (RPR)
            RPR = np.linalg.norm(delta - r) / (np.linalg.norm(delta) + eps)

            DQI = 2*recon_ssim*PAR / (recon_ssim + PAR + eps)

            # output txt file
            with open(f"{visualize_dir}/Sample{target_index}_result.txt", "w") as f:
                f.write(f"MSE: {mse.item()}\nCosine Alignment: {cos_align}\nPerturbation Attribution Ratio (PAR): {PAR}\nResidual Perturbation Ratio (RPR): {RPR}\nRecon SSIM: {recon_ssim}\nDQI: {DQI}")

            # visualize true perturbation
            plt.imshow(self.targets[-1].detach().cpu().numpy()-cln_targets[-1].detach().cpu().numpy()+0.5)
            plt.axis('off')              # 去掉坐标轴
            plt.xticks([])               # 去掉 x 轴刻度
            plt.yticks([])               # 去掉 y 轴刻度
            plt.savefig(f"{visualize_dir}/Sample{target_index}_perturbation.png", bbox_inches='tight')
            plt.close()

            # visualize reconstruction error
            plt.imshow(self.targets[-1].detach().cpu().numpy()-recons_multi_resos[-1].detach().cpu().numpy()+0.5)
            plt.axis('off')              # 去掉坐标轴
            plt.xticks([])               # 去掉 x 轴刻度
            plt.yticks([])               # 去掉 y 轴刻度
            plt.savefig(f"{visualize_dir}/Sample{target_index}_recon_error.png", bbox_inches='tight')
            plt.close()

            if self.edge_mask is not None:
                plt.imshow(self.edge_mask.detach().cpu().numpy())
                plt.title("Edge mask")
                plt.savefig(f"{visualize_dir}/Sample{target_index}_edge_mask.png", bbox_inches='tight')
                plt.close()
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

if __name__ == '__main__':
    import yaml
    import os
    import shutil
    if os.path.exists("./visualization"):
        shutil.rmtree("./visualization")
        os.makedirs("./visualization")
    # tensor purifier 的config
    with open("configs/QTTMR_test.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in [9])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device_type"] = "gpu" if torch.cuda.is_available() else "cpu"

    config_tmp = Config()
    for key, item in config.items():
        setattr(config_tmp, key, item)
    config = config_tmp

    # load attacked data
    ad_data_path = f"Ad_data/imgs_AutoAttack_cifar10_wrn2810_{"32"}_42.npy"
    ad_data = np.load(ad_data_path)
    ad_data = torch.tensor(ad_data).float()[0:1]
    img = torch.nn.functional.interpolate(ad_data, size=config.end_reso, mode='nearest').squeeze(0).permute(1,2,0).to(device)

    # # load a 512*512 image, using PIL
    # from PIL import Image

    # # load the image
    # img = Image.open('8.png')
    # img = np.array(img)
    # img = img.astype(np.float32) / 255.0
    # img = torch.from_numpy(img).float()
    # img = torch.nn.functional.interpolate(img.permute(2,0,1).unsqueeze(0), size=config.end_reso, mode='nearest').squeeze(0).permute(1,2,0).to(device)

    model_args = get_model_args(config, img, noisy_target=None, device_type=config.device_type)

    model = QTR_MR_I(**model_args)

    lr = config.lr
    iterations = [0] + config.iterations_for_upsampling + [config.num_iterations]
    iterations = [iterations[i+1] - iterations[i] for i in range(len(model.interm_resos))]
    time1 = time.time()
    for i, reso in enumerate(model.interm_resos):
        if i == len(model.interm_resos) -1:
            break
        print(f"Start purification for resolution {reso}*{reso}, target resolution {model.targets[i].shape[0]}*{model.targets[i].shape[1]}")
        optimizer = model.set_optimizer(reso, lr=lr)
        for j in range(iterations[i]):
            optimizer.zero_grad()
            loss, _ = model(reso)
            
            # visualization
            if j in [iterations[i]-1]:
                recons = []
                for r in model.interm_resos:
                    recons.append(model.custom_contract_qtt(output_reso=r))
                fig, axes = plt.subplots(2, len(model.targets), figsize=(4 * len(model.targets), 8))
                for k, r in enumerate(model.interm_resos):
                    axes[0, k].imshow(recons[k].detach().cpu().numpy())
                    axes[0, k].title.set_text(f"{r}x{r} reconstruction")
                    axes[1, k].imshow(model.targets[k].detach().cpu().numpy())
                    axes[1, k].title.set_text(f"{r}x{r} targets")
                mse = torch.nn.functional.mse_loss(recons[i], model.targets[i])
                fig.suptitle(f"Iteration {j}, resolution {reso}*{reso}, loss={mse.item()}")
                plt.savefig(f"./visualization/reso{reso}_iteration{j}.png")
                plt.close()

            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                print(f"grid: {reso}*{reso}, iteration{j}: loss={loss.item()}")
        lr *= 0.9

    time2 = time.time()
    print(f"Total time: {time2-time1}s")