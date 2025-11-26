# Based on TT-NF code from https://github.com/toshas/ttnf
# Modifications and/or extensions have been made for specific purposes in this project.

import quimb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from TN.tn_utils import *
from TN.utils import *
from TN.BaseTNModel import BaseTNModel
from TN.tn_utils import *
from TN.save_and_plot_utils import *
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class QTT(BaseTNModel):
    def __init__(self, target, init_reso, end_reso, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position='first_core', canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions =2, regularization_weight = 0.0, noisy_target = None, device = 'cpu',  masked_avg_pooling = False, sigma_init=0, img_TV=0, add_gaussian_noise=None, gaussian_noise_std=0):
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
        self.model = "QTT"
        self.canonization = canonization
        self.compression_alg = compression_alg
        self.regularization_type = regularization_type
        self.activation = activation

        self.shape_source = None
        self.shape_factors = None
        self.factor_target_to_source = None
        self.dim_grid_log2 = int(np.log2(init_reso))

        self.init_reso = init_reso
        self.end_reso = end_reso
        self.img_TV = img_TV
        self.add_gaussian = add_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        if self.add_gaussian is not None:
            self.taregt = self.target + torch.randn_like(self.target) * self.gaussian_noise_std

        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling

        self.regularization_weight = regularization_weight
        
        self.mask_rank = max_rank

        self.init_tn()
        
        if self.mask_rank < max_rank:
            self.create_rank_upsampling_mask()
        else:
            self.masked_components = None

        self.dim = len(self.shape_source)
        self.iteration = 0

        # extract the raw arrays and a skeleton of the TN
        self.set_torch_params()

        self.set_compression_variables()

        # get equation for contraction
        self.contraction_expression, self.path_info, self.symbol_info = self.get_contraction_expression()

        
    def get_contraction_expression(self):
        """
        Computes the contraction expression for the tensor network using opt_einsum.

        Returns:
        A tuple containing the contraction expression, path information, and symbol information.
        """
        
        output_inds = self.tn.outer_inds()
        if self.payload_position != 'grayscale':
            output_inds = [ind for ind in output_inds if ind != 'payload'] # remove string "payload" from output inds 
            output_inds = ['payload'] + output_inds # add to front of output inds
        backend = 'torch'
        optimze = 'dp'
        path_info =self.tn.contract( output_inds=output_inds, get='path-info', backend=backend, optimize=optimze)
        self.update_contraction_costs(path_info)
        symbol_info =self.tn.contract( output_inds=output_inds, get='symbol-map', backend = backend, optimize=optimze)
        contraction_expression = self.tn.contract(output_inds=output_inds, get='expression', backend=backend, optimize=optimze)
        
        return contraction_expression, path_info, symbol_info

    def init_tn(self):
        """
        Initializes the tensor network (TN) for the model.
        """
        # Create the initial QTTNF
        self.tn, self.shape_source, self.shape_target, self.shape_factors, _, self.factor_target_to_source = get_qtt_TTNF(self.current_reso, self.max_rank, dim=self.dimensions, payload_dim=self.payload, payload_position=self.payload_position, compression_alg=self.compression_alg, canonization=self.canonization, sigma_init=self.sigma_init)      
        print(self.tn.tensors)
        # print("Initialized tn,", self.tn)
    
    def create_rank_upsampling_mask(self):
        # take self.tn and create a mask of ones for each core 
        masked_components = []
        for i,c in enumerate(self.tn.tensors):
            mask = torch.ones_like(c.data)
            # all cores have form (r1, n, r2) except the last one which is (r1, n) and the first one which is (Payload n, r2)
            # set to zeros where exceeding the mask rank
            if i == 0:
                if len(mask.shape) == 2:
                    mask[:,self.mask_rank:] = 0
                else:
                    mask[ :, :,self.mask_rank:] = 0
            elif i == len(self.tn.tensors)-1:
                mask[ self.mask_rank:, :] = 0
            else:
                mask[self.mask_rank:, :, self.mask_rank:] = 0
        
            masked_components.append(mask)
            
        self.masked_components = masked_components
    
    def set_torch_params(self, core_indices_to_exclude = []):
        """
        Retrieves the parameters of the tensor network as a PyTorch ParameterDict.

        Parameters:
        - core_indices_to_exclude: indices of cores to exclude from training (making them non-trainable).

        Returns:
        A PyTorch ParameterDict containing the parameters.
        """
        
        params, self.skeleton = qtn.pack(self.tn)

        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        # if core_indices_to_exclude length is not 0, remove the corresponding cores from the torch_params making them non-trainable
        if len(core_indices_to_exclude) != 0:
            for i in core_indices_to_exclude:
                self.torch_params[str(i)].requires_grad = False

        self.tn = self.reconstruct() # Very important!
        
        return self.torch_params

    def get_optparam_groups(self, lr_init = 0.02):
        """
        Groups the optimization parameters.

        Parameters:
        - lr_init: initial learning rate for the optimization.

        Returns:
        A list of dictionaries containing parameters and their learning rates.
        """
        
        out = []
        out += [
            {'params': self.torch_params.values(), 'lr': lr_init},
        ]
        return out

    def reconstruct(self):
        """
        Reconstructs the tensor network as a quimb tensor network.

        Returns:
        The reconstructed tensor network.
        """
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        tn = qtn.unpack(params, self.skeleton)

        return tn

    def get_image_reconstruction(self, tn=None):
        """
        Reconstructs the image from the tensor network.

        Parameters:
        - tn: the tensor network to reconstruct the image from (if None, uses the current model's network).

        Returns:
        The reconstructed image.
        """
        if tn is None:
            tn = self.tn
        data = qtt_to_tensor(tn, self.shape_source, self.shape_factors, self.factor_target_to_source,
                            inds=self.inds, payload_position=self.payload_position, grayscale= self.grayscale,
                            expression=self.contraction_expression, payload=self.payload, masked_components = self.masked_components)
                            # expression=None)
        return data
    
    def set_compression_variables(self):
        """
        Sets the compression variables for the tensor network.
        """
        
        if self.dtype == torch.float32:
            self.dtype_sz_bytes = 4
        elif self.dtype == torch.float64:
            self.dtype_sz_bytes = 8

        self.num_uncompressed_params = np.prod(self.shape_source)
        if self.payload != 0:
            self.num_uncompressed_params = self.num_uncompressed_params * self.payload
        self.num_compressed_params = sum(p.numel() for p in self.torch_params.values())
        
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3) 
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params
    
    def update_contraction_costs(self, contraction_info = None):
        """
        Updates the cost of contraction variable for the tensor network.

        Parameters:
        - contraction_info: the contraction information to use (if None, computes it from the current network).
        """
        if contraction_info is None:
            contraction_info = self.tn.contraction_info()
        self.flops = contraction_info.opt_cost
        self.largest_intermediate = contraction_info.largest_intermediate
        self.flops_per_iter = self.flops_per_iter + [ self.flops]
        self.largest_intermediate_per_iter = self.largest_intermediate_per_iter + [ self.largest_intermediate]
    
    def l1_regularization(self, tn):
        """
        Computes the L1 regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The L1 regularization term.
        """
        total = 0
        for i in range(len(tn.tensors)):
            total += torch.mean(torch.abs(tn.tensors[i].data))
            
        return total 

    def l2_regularization(self, tn):
        """
        Computes the L2 regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The L2 regularization term.
        """
        total = 0
        for i in range(len(tn.tensors)):
            total += torch.mean(torch.pow(tn.tensors[i].data, 2))
            
        return total 
        

    def compute_total_variation_loss(self, xx):
        """
        Computes the total variation loss for the tensor network.

        Parameters:
        - tn: the tensor network with a number of cores len(tn.tensors)

        Returns:
        The total variation loss.
        """
        total_variation_loss = 0
        for tensor in tn.tensors:
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
        return total_variation_loss /self.num_compressed_params

    def apply_activation(self, x):
        """
        Applies the specified activation function to the input tensor.

        Parameters:
        - x: the input tensor.

        Returns:
        The tensor after applying the activation function.
        """
        if self.activation == "relu":
            return torch.nn.functional.relu(x)
        elif self.activation == "softplus":
            return torch.nn.functional.softplus(x)
        else:
            return x
        
    @torch.no_grad()
    def batched_qtt(self, compute_reconstruction=False, target = None):
        """
        Performs batched tensor network contraction for large grids that are prohibitively large to contract at once.

        Parameters:
        - compute_reconstruction: whether to compute and return the reconstruction.
        - target: the target tensor to model (if None, uses the model's target).

        Returns:
        The PSNR value and the reconstructed object (if compute_reconstruction is True).
        """

        self.downsampled_target = self.downsampled_target.cpu()

        batch_size = 512**2
        grid_size = [self.current_reso for _ in range(self.dimensions)]
        num_batches = int(np.prod(grid_size) / batch_size)

        # Create a SimpleSamplerNonRandom with the appropriate dimensions and grid size
        sampler = SimpleSamplerNonRandom(self.dimensions, batch_size, max_value=grid_size[0] - 1)

        count = 0
        acc_loss = 0

        if target is None:
            target = self.target

        if target.shape[0] == 1:
            target = target.squeeze(0)
            if self.downsampled_target.shape[-1] == 1:
                target = target.unsqueeze(-1)

        if compute_reconstruction:
            reconstructed_object = torch.zeros(target.shape)
        else :
            reconstructed_object = None
        

        for _ in tqdm(range(num_batches), desc="Processing Batches in Batched QTT"):
            batch_indices, _ = sampler.next_batch()
            values_recon = self.get_reconstructed_values(batch_indices.to(self.device))
            values_target = self.get_values_for_coords(target, batch_indices)

            loss = self.loss_fn(values_recon, values_target.to(self.device))
            acc_loss += loss
            count += 1

            if compute_reconstruction:
                values_recon = values_recon.cpu()
                if batch_indices.shape[1] == 2:
                    reconstructed_object[batch_indices[:, 0], batch_indices[:, 1]] = values_recon
                elif batch_indices.shape[1] == 3:
                    reconstructed_object[batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2]] = values_recon
                else:
                    raise ValueError("Invalid number of dimensions in batch_indices")


        psnr_val = -10. * torch.log(acc_loss.cpu() / num_batches) / torch.log(torch.Tensor([10.]))

        return psnr_val, reconstructed_object


    def get_reconstructed_values(self, x):
        """
        Gets the reconstructed values from the tensor network for given coordinates.

        Parameters:
        - x: the coordinates to get the reconstructed values for.

        Returns:
        The reconstructed values and the regularization term.
        """
        if self.use_TTNF_sampling:
            input_ = get_core_tensors(self.tn)
            if self.dimensions == 2:
                coords = coord_tensor_to_coord_qtt2d(x, len(input_), chunk=True)
            elif self.dimensions == 3:
                coords = coord_tensor_to_coord_qtt3d(x, len(input_), chunk=True)
            elif self.dimensions > 3 or self.dimensions < 2: 
                raise ValueError("Only 2D and 3D supported")

            values_recon = sample_intcoord_tt_v2(input_, coords, last_core_is_payload=False, checks=True)        
            #values_recon = sample_intcoord_tt_v2(input_, coords, last_core_is_payload=False, reverse = True, checks=True)        
        else: 
            image = self.get_image_reconstruction()
            values_recon = self.get_values_for_coords(image, x)
            self.image = image
        
        if self.activation != "None":
            values_recon = self.apply_activation(values_recon)
        
        reg_term = self.compute_regularization_term(self.tn)
        # reg_term = self.regularization_weight * self.TV(image.clone(), p=1)
        
        return values_recon, reg_term
    
    def compute_regularization_term(self, tn=None):
        """
        Computes the regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The regularization term.
        """
        if self.regularization_type == "TV":
            reg_term = self.compute_total_variation_loss(self.tn)
        elif self.regularization_type == "L1":
            reg_term = self.l1_regularization(self.tn)
        elif self.regularization_type == "L2":
            reg_term = self.l2_regularization(self.tn)
        else: 
            reg_term = 0
        
        return reg_term
    
    def count_parameters(self):
        param_numbers = 0
        for i, tensor in enumerate(self.tn.tensors):
            params = tensor.data.numel()
            param_numbers += params
        return param_numbers

    def forward(self, x, iteration = None, aux=None):
        """
        The forward pass for the model.

        Parameters:
        - x: the input tensor.
        - x_norm: normalized x values between -1 and 1 

        Returns:
        The loss for the given input.
        """
        values_recon, reg_term = self.get_reconstructed_values(x) 
        if self.loss_fn_str == "L2":
            values_target = self.get_values_for_coords(self.downsampled_target, x)
            loss =  self.loss_fn(values_recon, values_target) 
            return loss, reg_term

        elif (self.loss_fn_str == "adv" and self.current_reso != self.init_reso and self.adv_reso == None) or (self.loss_fn_str == "adv" and self.adv_reso != None and self.current_reso >= self.adv_reso):
            img = self.image.clone()
            img_loss = self.image.clone()

            # optimize the perturbation
            perturb_lr = 0.1
            ssim = SSIM(data_range=1.0).to(self.device)
            # for _ in range(steps):
            adv_loss = ssim(img.permute(2, 0, 1).unsqueeze(0), self.downsampled_target.permute(2, 0, 1).unsqueeze(0))
            # adv_loss = self.loss_fn(img.permute(2, 0, 1).unsqueeze(0), self.downsampled_target.permute(2, 0, 1).unsqueeze(0))
            grad = torch.autograd.grad(adv_loss, [img])[0]
            img = img + perturb_lr * torch.sign(grad.detach())
            img = torch.clamp(img, 0.0, 1.0)
            loss = self.loss_fn(img, self.downsampled_target) + 5*self.loss_fn(img_loss, self.current_reso_init_img)

        elif (self.loss_fn_str == "adv" and self.current_reso == self.init_reso and self.adv_reso == None) or (self.loss_fn_str == "adv" and self.adv_reso != None and self.current_reso < self.adv_reso):
            if aux is None:
                loss = self.loss_fn(self.image.clone(), self.downsampled_target)
            else:
                values_target = self.get_values_for_coords(aux, x)
                loss =  self.loss_fn(values_recon, values_target)
        
        return loss, reg_term
    
    def train(self, target, args, target_index=None, visualize=False, cln_target=None, visualize_dir='', record_loss=False):
        if visualize == False:
            target_index = None
            cln_target = None
            visualize_dir = ''
        else:
            assert target_index is not None and cln_target is not None and visualize_dir != '', 'visualize is True but target_index, cln_target, visualize_dir are not provided'
            
        noisy_target = None
        # print("Target shape", target.shape)

        # Noise Experiments
        if args.noise_std > 0 and args.noise_type != "None":
            if noisy_target is None:
                noisy_target = make_noisy_target(target, args)
            else:
                noisy_target = make_noisy_target(noisy_target, args)
        else:
            noisy_target = None

        # Adjust learning rate based on noise/incoplete data
        if noisy_target is not None and args.factor_reduce_lr_based_on_noise != 0:
            lr = args.lr
            if args.subset_to_train_on < 1.0:
                lr = lr * args.factor_reduce_lr_based_on_noise ** (1-args.subset_to_train_on) # lower subset_to_train_on requires lower lr
            elif args.noise_type == "gaussian" or args.noise_type == "laplace":
                lr = lr * args.factor_reduce_lr_based_on_noise ** args.noise_std # more noise requires lower lr
            args.lr = lr
            print("New Learning Rate: ", args.lr)

        grid_size = [args.init_reso for i in range(args.dimensions)]
        iterations = args.num_iterations
        iterations_for_upsampling, iterations_until_next_upsampling = calculate_iterations_until_next_upsampling(args, iterations)

        # print("### New grid size {}, Compression Factor {}, Model Size {}, Device {}".format(grid_size, model.compression_factor, model.sz_compressed_gb, args.device_type))

        if args.subset_to_train_on == 1.0:
            sampler = SimpleSamplerImplicit(args.dimensions, batch_size=min(args.max_batch_size, int(self.current_reso**args.dimensions)), max_value=self.current_reso-1)
        
        # optimizer = torch.optim.Adam(self.get_optparam_groups(lr_init=args.lr), weight_decay=0.1)
        optimizer = torch.optim.Adam(self.get_optparam_groups(lr_init=args.lr))

        # Get iterations until next upsampling and use it to determine warmup steps
        iterations_lr_warmup = iterations
        
        # Scheduler
        lr_gamma = calculate_gamma(args.lr, args.lr_decay_factor_until_next_upsampling, iterations_lr_warmup)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        warmup_steps = args.warmup_steps
        lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps)


        # Save data while training
        best_recon = None
        best_loss = 1e10
        losses = []
        mse_history = []
        validation_losses = []
        figsize=(16,8)
        psnr_val = -1
        saved_images = []
        saved_images_iterations = []
        save_times = []
        time_start = time.time()
        psnr_arr = []
        compression_rates = []
        aux = None
        
        if args.device_type == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        loop_obj = tqdm(range(iterations),disable= not args.use_tqdm)
        time_start = time.time()
        
        # check if paramter rank_upsampling_rank_range and rank_upsampling_iteration_range are set
        # check if contained in args
        
        if len(args.rank_upsampling_rank_range) > 0 and len(args.rank_upsampling_iteration_range) > 0:
            new_max_ranks, new_rank_upsample_iterations = setup_rank_upsampling(self, args.rank_upsampling_rank_range, args.rank_upsampling_iteration_range)
        else:
            new_max_ranks = []
            new_rank_upsample_iterations = []
            
        # print("New max ranks", new_max_ranks)
        # print("New rank upsample iterations", new_rank_upsample_iterations)
        
        recons_multi_resos = []
        downsampled_targets = []
        for ite in loop_obj:
            if ite in new_rank_upsample_iterations:
                idx = new_rank_upsample_iterations.index(ite)
                self.mask_rank = new_max_ranks[idx]
                self.create_rank_upsampling_mask()
            
            optimizer.zero_grad()
            
            batch_indices, batch_indicies_norm = sampler.next_batch()

            loss, reg_term = self.forward(batch_indices.to(device), ite, aux=aux)
            
            reg_term = reg_term * self.regularization_weight 
            
            total_loss = loss + reg_term

            # with torch.autograd.detect_anomaly():
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

            if record_loss:
                mse_history.append(self.loss_fn(self.image.detach().cpu(), target).item())
            
            if ite <= warmup_steps:
                lr_warmup_scheduler.step()
            else:
                scheduler.step()
        
        time_end = time.time()
        print("Training time: " + str(time_end - time_start), f"Losses:{np.array(losses)}")

        ### PSNR and reconstruction of object ###
        if self.model == "QTT" and len(self.shape_factors) > 25: # PyTorch cannot permute more than 25 dimensions tensors - have to use batched reconstruction
            if args.noise_std > 0 and args.noise_type is not None or args.subset_to_train_on < 1.0:
                psnr_val, best_recon = self.batched_qtt(compute_reconstruction= args.compute_reconstruction, target = target) #best_recon might be None if
            else:
                psnr_val, best_recon = self.batched_qtt(compute_reconstruction= args.compute_reconstruction) #best_recon might be None if
        else:
            best_recon = self.get_image_reconstruction()
            # psnr_val = psnr(self.target, best_recon.detach().cpu())
        best_recon = torch.clamp(best_recon, 0, 1)

        recons_multi_resos.append(best_recon)
        downsampled_targets.append(self.downsampled_target)
        if visualize:
            cln_targets = []
            stride_window = 1
            for i in range(len(recons_multi_resos)):
                cln_target_ = torch.nn.functional.avg_pool2d(cln_target.unsqueeze(0), stride_window, stride_window).squeeze().permute(1, 2, 0)
                cln_targets.append(cln_target_)
                stride_window *= 2
            cln_targets = cln_targets[::-1]
            fig, axes = plt.subplots(6, len(downsampled_targets), figsize=(5 * len(downsampled_targets), 30))
            for k, r in enumerate(recons_multi_resos):
                axes[0].imshow(r.detach().cpu().numpy())
                axes[0].title.set_text(f"{r.shape[0]}x{r.shape[0]} reconstruction")
                axes[1].imshow(downsampled_targets[k].detach().cpu().numpy())
                axes[1].title.set_text(f"{r.shape[0]}x{r.shape[0]} targets")
                axes[2].imshow(cln_targets[k].detach().cpu().numpy())
                axes[2].title.set_text(f"{r.shape[0]}x{r.shape[0]} clean targets")
                axes[3].hist(r.detach().cpu().numpy().flatten()-downsampled_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                axes[3].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} reconstruction error, \n mse:{torch.nn.functional.mse_loss(r.detach().cpu(), downsampled_targets[k].detach().cpu()).item()}")
                axes[4].hist(downsampled_targets[k].detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                axes[4].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} true perturbation, \n mse:{torch.nn.functional.mse_loss(downsampled_targets[k].detach().cpu(), cln_targets[k].detach().cpu()).item()}")
                axes[5].hist(r.detach().cpu().numpy().flatten()-cln_targets[k].detach().cpu().numpy().flatten(), bins=256, alpha=0.5)
                axes[5].title.set_text(f"Histogram of {r.shape[0]}x{r.shape[0]} error between \n recon and clean, mse:{torch.nn.functional.mse_loss(r.detach().cpu(), cln_targets[k].detach().cpu()).item()}")
            mse = torch.nn.functional.mse_loss(r, downsampled_targets[i])
            fig.suptitle(f"Sample{target_index} Iteration {ite}, loss={mse.item()}")
            plt.savefig(f"{visualize_dir}/Sample{target_index} iteration{ite}.png", bbox_inches='tight')
            plt.close()
        return best_recon, time_end - time_start, mse_history
    
def upsample_dim(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, compression_rates, iteration, iteration_index, iterations_until_next_upsampling=1000, sampled_indices_all=None, new_max_rank=None):
    """
    Function that handles both upsample common and upsample dim functionalities.

    Args:
        args: Various arguments needed for the process.
        model: The model being used.
        figsize: Figure size for any plots or images.
        saved_images: A list to store saved images.
        saved_images_iterations: Iterations at which images are saved.
        save_times: A list to store the times at which data is saved.
        time_start: The start time of the process.
        psnr_arr: An array to store PSNR values.
        compression_rates: A list to store compression rates.
        iteration: The current iteration of the process.
        iteration_index: The index of the current iteration.
        iterations_until_next_upsampling: Iterations until the next upsample. Defaults to 1000. # used for lr warmup and scheduler
        sampled_indices_all: All sampled indices. Defaults to None meaning use all. # used for subset sampling
        new_max_rank: The new maximum rank, applicable for rank upsample. Defaults to None.
    
    Returns:
        A tuple containing the sampler, optimizer, scheduler, best_loss, lr_warmup_scheduler, and percentage_of_sampled_indices.
    """
    model.upsample(iteration)

    optimizer, new_lr = setup_optimizer(args, model, iteration_index)
    scheduler, best_loss, lr_warmup_scheduler = setup_scheduler(optimizer, args, iterations_until_next_upsampling)
    sampler, percentage_of_sampled_indices = setup_sampler(args, sampled_indices_all, model)  # Not needed for rank upsample

    print_info(model, new_lr, model.dimensions, None)

    if not should_skip_saving(model, args):
        save_data_while_training(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, iteration)

    compression_rates.append(model.compression_factor)

    return sampler, optimizer, scheduler, best_loss, lr_warmup_scheduler, percentage_of_sampled_indices
    
@torch.no_grad()
def setup_optimizer(args, model, iteration_index):
    new_lr = args.lr_decay_factor ** (iteration_index + 1) * args.lr

    if "mlp" in args.model.lower():
        new_lr_mlp = args.lr_decay_factor ** (iteration_index + 1) * args.lr_init_mlp
        optimizer = torch.optim.Adam(model.get_optparam_groups(lr_init=new_lr, lr_init_mlp=new_lr_mlp))
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
        optimizer = torch.optim.Adam(model.get_optparam_groups(lr_init=new_lr))

    return optimizer, new_lr

def setup_scheduler(optimizer, args, iterations_until_next_upsampling):
    lr_gamma = calculate_gamma(args.lr, args.lr_decay_factor_until_next_upsampling, iterations_until_next_upsampling)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    best_loss = 1e10  # reset best loss

    warmup_steps = args.warmup_steps
    lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps)

    return scheduler, best_loss, lr_warmup_scheduler

def setup_sampler(args, sampled_indices_all, model):
    dimensions = model.dimensions

    if args.subset_to_train_on == 1.0:
        sampler = SimpleSamplerImplicit(dimensions, batch_size=min(args.max_batch_size, int(model.current_reso**dimensions)), max_value=model.current_reso-1)
        procentage_of_sampled_indices = None
    else:
        sampler, procentage_of_sampled_indices = get_subset_sampler(args, sampled_indices_all, model, masked_avg_pooling=args.masked_avg_pooling)

    return sampler, procentage_of_sampled_indices
                
def get_subset_sampler(args, sampled_indices_all, model, default_val_for_non_sampled = 0.0, masked_avg_pooling = False):
    if args.payload > 1:
        grid_shape = model.target.shape[:-1]
    else:
        grid_shape = model.target.shape

    sampled_indices_grid = torch.zeros(grid_shape) + default_val_for_non_sampled

    if args.dimensions == 2:
        sampled_indices_grid[sampled_indices_all[:,0], sampled_indices_all[:,1]] = 1
    elif args.dimensions == 3:
        sampled_indices_grid[sampled_indices_all[:,0], sampled_indices_all[:,1], sampled_indices_all[:,2]] = 1
    else:
        raise NotImplementedError("Dimensions not implemented")

    # do average pooling using a factor of model.current_reso to get tiles allowed to be trained on
    factor = int(model.target.shape[0]/model.current_reso) 
    sampled_indices_grid = downsample_with_avg_pooling(sampled_indices_grid, factor, args.dimensions, grayscale = True, device = None, masked=masked_avg_pooling)

    # All where sampled_indices_grid is greater equal to default_val_for_non_sampled
    sampled_indices = torch.nonzero(sampled_indices_grid != default_val_for_non_sampled).squeeze() # 
    
    procentage_of_sampled_indices = len(sampled_indices)/len(sampled_indices_grid.view(-1))
    print("Using This Procentage of all indices in downsampled target", procentage_of_sampled_indices) # get sampled_indices proportion to total number of indices

    sampler = SimpleSamplerSubset(args.dimensions, batch_size=min(args.max_batch_size, int(model.current_reso**args.dimensions)), max_value=model.current_reso-1, indices = sampled_indices)
    return sampler, procentage_of_sampled_indices

def print_info(model, new_lr, dimensions, new_max_rank):
    if new_max_rank is not None:
        print(f"### New rank {new_max_rank}, lr {new_lr}, new_compression_factor {model.compression_factor}, model_size {model.sz_compressed_gb}")
    else:
        grid_size = [model.current_reso for _ in range(dimensions)]
        print(f"### New grid_size {grid_size}, lr {new_lr}, new_compression_factor {model.compression_factor}, model_size {model.sz_compressed_gb}")