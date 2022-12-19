# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

@author: jamyl
"""

from math import sqrt
import time
import numpy as np
from numba import cuda
import torch as th
import torch.nn.functional as F

from .linalg import get_eighen_elmts_2x2
from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, getTime
from .utils_image import compute_grey_images


def estimate_kernels(img, options, params):
    """ Returns the kernels covariance matrix for the "img" frame, sampled at the
    center of every bayer quad (or at the center of every grey pixel in grey
    mode).

    Parameters
    ----------
    img : numpy Array[imshape_y, imshape_x]
        raw image
    options : dict
        options
    params : dict
        ['mode'] : {"bayer", "grey"}
            Wether the burst is raw or grey
        params['tuning'] : dict
            parameters driving the kernel shape

    Returns
    -------
    covs : device Array[imshape_y/2, imshape_x/2]
        covarince matrices samples at the center of each bayer quad).

    """
    imshape_y, imshape_x = img.shape
    
    bayer_mode = params['mode']=='bayer'
    VERBOSE = options['verbose']
    
    k_detail = params['tuning']['k_detail']
    k_denoise = params['tuning']['k_denoise']
    D_th = params['tuning']['D_th']
    D_tr = params['tuning']['D_tr']
    k_stretch = params['tuning']['k_stretch']
    k_shrink = params['tuning']['k_shrink']
    
    if VERBOSE>2:
        cuda.synchronize()
        t1 = time.perf_counter()
    
    if bayer_mode : 
        
        img_grey = compute_grey_images(img, method="decimating")
        
        if VERBOSE>2:
            cuda.synchronize()
            t1 = getTime(t1, "- Decimated Image")
    else :
        img_grey = img # no need to copy now, they will be copied to gpu later.
        
    grey_imshape_y, grey_imshape_x = grey_imshape = img_grey.shape
    
    # computing grads
    # gradsx = np.empty((grey_imshape_y, grey_imshape_x-1), DEFAULT_NUMPY_FLOAT_TYPE)
    # gradsx = img_grey[:,1:] - img_grey[:,:-1]

    
    # gradsy = np.empty((grey_imshape_y-1, grey_imshape_x), DEFAULT_NUMPY_FLOAT_TYPE)
    # gradsy = img_grey[1:,:] - img_grey[:-1,:]
    
    # full_grads = np.empty((grey_imshape_y-1, grey_imshape_x-1, 2), DEFAULT_NUMPY_FLOAT_TYPE) # x, y
    # full_grads[:, :, 0] = (gradsx[:-1, :] + gradsx[1:, :])/4
    # full_grads[:, :, 1] = (gradsy[:, :-1] + gradsy[:, 1:])/4
    
    # cuda_full_grads = cuda.to_device(full_grads)
    
    th_grey_img = th.as_tensor(img_grey, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    
    grad_kernel1 = np.array([[[[-0.5, 0.5]]],
                              
                              [[[ 0.5, 0.5]]]])
    grad_kernel1 = th.as_tensor(grad_kernel1, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    
    grad_kernel2 = np.array([[[[0.5], 
                                [0.5]]],
                              
                              [[[-0.5], 
                                [0.5]]]])
    grad_kernel2 = th.as_tensor(grad_kernel2, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")


    tmp = F.conv2d(th_grey_img, grad_kernel1)
    th_full_grad = F.conv2d(tmp, grad_kernel2, groups=2)
    # The default padding mode reduces the shape of grey_img of 1 pixel in each
    # direction, as expected
    
    cuda_full_grads = cuda.as_cuda_array(th_full_grad.squeeze().transpose(0,1).transpose(1, 2))
    # shape [y, x, 2]
    
    # grad_kernel1 = th.Tensor([[-0.25, 0.25], [-0.25, 0.25]], dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    # grad_kernel2 = th.Tensor([[-0.25, -0.25], [0.25, 0.25]], dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    
    if VERBOSE>2:
        cuda.synchronize()
        t1 = getTime(t1, "- Gradients computed")
        
    covs = cuda.device_array(grey_imshape + (2,2), DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(grey_imshape_x/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(grey_imshape_y/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_estimate_kernel[blockspergrid, threadsperblock](cuda_full_grads,
                                    k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink,
                                    covs)  
    if VERBOSE>2:
        cuda.synchronize()
        t1 = getTime(t1, "- Covariances estimated")
    
    return covs

# TODO recode this, thread separation is not good
@cuda.jit
def cuda_estimate_kernel(full_grads,
                         k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink,
                         covs):
    pixel_idx, pixel_idy = cuda.grid(2)
    
    imshape_y, imshape_x, _, _ = covs.shape
    

    
    if (0 <= pixel_idy < imshape_y and 0 <= pixel_idx < imshape_x) :
        structure_tensor = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
        structure_tensor[0, 0] = 0
        structure_tensor[0, 1] = 0
        structure_tensor[1, 0] = 0
        structure_tensor[1, 1] = 0
        
    
        for i in range(0, 2):
            for j in range(0, 2):
                x = pixel_idx - 1 + j
                y = pixel_idy - 1 + i
                
                if (0 <= y < full_grads.shape[0] and
                    0 <= x < full_grads.shape[1]):
                    
                    full_grad_x = full_grads[y, x, 0]
                    full_grad_y = full_grads[y, x, 1]
    
                    structure_tensor[0, 0] += full_grad_x * full_grad_x
                    structure_tensor[1, 0] += full_grad_x * full_grad_y
                    structure_tensor[0, 1] += full_grad_x * full_grad_y
                    structure_tensor[1, 1] += full_grad_y * full_grad_y
        
        l = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e1 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e2 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        rho = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

        get_eighen_elmts_2x2(structure_tensor, l, e1, e2)

        compute_rho(l[0], l[1], rho, k_detail, k_denoise, D_th, D_tr, k_stretch,
        k_shrink)

        rho_1_sq = rho[0]*rho[0]
        rho_2_sq = rho[1]*rho[1]
        
        covs[pixel_idy, pixel_idx, 0, 0] = rho_1_sq*e1[0]*e1[0] + rho_2_sq*e2[0]*e2[0]
        covs[pixel_idy, pixel_idx, 0, 1] = rho_1_sq*e1[0]*e1[1] + rho_2_sq*e2[0]*e2[1] 
        covs[pixel_idy, pixel_idx, 1, 0] = rho_1_sq*e1[0]*e1[1] + rho_2_sq*e2[0]*e2[1]
        covs[pixel_idy, pixel_idx, 1, 1] = rho_1_sq*e1[1]*e1[1] + rho_2_sq*e2[1]*e2[1]

    
@cuda.jit(device=True)
def compute_rho(l1, l2, rho, k_detail, k_denoise, D_th, D_tr, k_stretch,
                          k_shrink):
    """
    Computes rho_1 and rho_2 based on lambda1, lambda2 and the constants.

    Parameters
    ----------
    l1 : float
        lambda1 (dominant eighen value)
    l2 : float
        lambda2
    rho : Array[2]
        empty vector where rho_1 and rho_2 will be stored
    k_detail : TYPE
        DESCRIPTION.
    k_denoise : TYPE
        DESCRIPTION.
    D_th : TYPE
        DESCRIPTION.
    D_tr : TYPE
        DESCRIPTION.
    k_stretch : TYPE
        DESCRIPTION.
    k_shrink : TYPE
        DESCRIPTION.


    """
    A = 1+sqrt((l1 - l2)/(l1 + l2))

    D = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1)
    

    # This is a very agressive way of driving anisotropy, but it works well so far.
    if A > 1.9:
        nu_1 = 1/k_shrink
        nu_2 = k_stretch
    else:
        nu_1 = 1
        nu_2 = 1
    # nu_1 = 1/(k_shrink*(A - 1) + (2 - A))
    # nu_2 = (k_stretch*(A -1) + (2 - A))
    
    rho[0] = k_detail*((1-D)*nu_1 + D*k_detail*k_denoise)
    rho[1] = k_detail*((1-D)*nu_2 + D*k_detail*k_denoise)


