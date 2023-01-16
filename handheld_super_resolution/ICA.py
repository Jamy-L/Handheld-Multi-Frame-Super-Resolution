# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

This script contains all the operations corresponding to the function
"ICA" called in Alg. 2: Registration.

@author: jamyl
"""

import time
import math

import numpy as np
from numba import cuda
from scipy.ndimage._filters import _gaussian_kernel1d
import torch
import torch.nn.functional as F

from .linalg import bilinear_interpolation
from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS
from .linalg import solve_2x2
    
def init_ICA(ref_img, options, params):
    """
    Initializes the ICa algorithm by computing the gradients of the reference
    image, and the hessian matrix.

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
    options : dict
        verbose options.
    params : dict
        parameters.

    Returns
    -------
    cuda_gradx : Device Array
        horizontal gradients of the reference image
    cuda_grady : Device Array
        vertical gradients of the reference image
    hessian : Device Array
        hessian matrix defined for each patch of the reference image.

    """
    current_time, verbose_3 = time.perf_counter(),  options['verbose'] >= 3

    sigma_blur = params['tuning']['sigma blur']
    tile_size = params['tuning']['tileSize']
    
    imsize_y, imsize_x = ref_img.shape
    
    # image is padded during BM, we need to consider that to count patches
    
    n_patch_y = math.ceil(imsize_y/ tile_size)
    n_patch_x = math.ceil(imsize_x/ tile_size)

    # Estimating gradients with Prewitt kernels
    kernely = np.array([[-1],
                        [0],
                        [1]])
    
    kernelx = np.array([[-1,0,1]])
    
    # translating ref_img numba pointer to pytorch
    # the type needs to be explicitely specified. Filters need to be casted to float to perform convolution
    # on float image
    th_ref_img = torch.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    th_kernely = torch.as_tensor(kernely, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    th_kernelx = torch.as_tensor(kernelx, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    
    # adding 2 dummy dims for batch, channel, to use torch convolve
    if sigma_blur != 0:
        # This is the default kernel of scipy gaussian_filter1d
        # Note that pytorch Convolve is actually a correlation, hence the ::-1 flip.
        # copy to avoid negative stride (not supported by torch)
        gaussian_kernel = _gaussian_kernel1d(sigma=sigma_blur, order=0, radius=int(4*sigma_blur+0.5))[::-1].copy()
        th_gaussian_kernel = torch.as_tensor(gaussian_kernel, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
        
        
        # 2 times gaussian 1d is faster than gaussian 2d
        temp = F.conv2d(th_ref_img, th_gaussian_kernel[:, None], padding='same') # convolve y
        temp = F.conv2d(temp, th_gaussian_kernel[None, :], padding='same') # convolve x
        
        
        th_gradx = F.conv2d(temp, th_kernelx, padding='same').squeeze() # 1 batch, 1 channel
        th_grady = F.conv2d(temp, th_kernely, padding='same').squeeze()
        
    else:
        th_gradx = F.conv2d(th_ref_img, th_kernelx, padding='same').squeeze() # 1 batch, 1 channel
        th_grady = F.conv2d(th_ref_img, th_kernely, padding='same').squeeze()
        
    
    # swapping grads back to numba
    cuda_gradx = cuda.as_cuda_array(th_gradx)
    cuda_grady = cuda.as_cuda_array(th_grady)
    
    if verbose_3:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Gradients estimated')
    
    hessian = cuda.device_array((n_patch_y, n_patch_x, 2, 2), DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    
    blockspergrid_x = math.ceil(n_patch_x/threadsperblock[1])
    blockspergrid_y = math.ceil(n_patch_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    compute_hessian[blockspergrid, threadsperblock](cuda_gradx, cuda_grady,
                                                    tile_size, hessian)
    
    if verbose_3:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Hessian estimated')
    
    return cuda_gradx, cuda_grady, hessian
    
@cuda.jit
def compute_hessian(gradx, grady, tile_size, hessian):
    imshape = gradx.shape
    patch_idx, patch_idy = cuda.grid(2)
    
    patch_pos_idx = tile_size * patch_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    patch_pos_idy = tile_size * patch_idy
    
    local_hessian = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    local_hessian[0, 0] = 0
    local_hessian[0, 1] = 0
    local_hessian[1, 0] = 0
    local_hessian[1, 1] = 0
    
    for i in range(tile_size):
        for j in range(tile_size):
            pixel_global_idy = patch_pos_idy + i
            pixel_global_idx = patch_pos_idx + j
            
            inbound = (0 <= pixel_global_idy < imshape[0] and 
                       0 <= pixel_global_idx < imshape[1])
    
            if inbound : 
                local_gradx = gradx[pixel_global_idy, pixel_global_idx]
                local_grady = grady[pixel_global_idy, pixel_global_idx]
                
                local_hessian[0, 0] += local_gradx*local_gradx
                local_hessian[0, 1] += local_gradx*local_grady
                local_hessian[1, 0] += local_gradx*local_grady
                local_hessian[1, 1] += local_grady*local_grady
                
    hessian[patch_idy, patch_idx, 0, 0] = local_hessian[0, 0]
    hessian[patch_idy, patch_idx, 0, 1] = local_hessian[0, 1]
    hessian[patch_idy, patch_idx, 1, 0] = local_hessian[1, 0]
    hessian[patch_idy, patch_idx, 1, 1] = local_hessian[1, 1]


def ICA_optical_flow(cuda_im_grey, cuda_ref_grey,
                     cuda_gradx, cuda_grady,
                     hessian, cuda_pre_alignment,
                     options, params, debug = False):
    """ Computes optical flow between the ref_img and all images of comp_imgs 
    based on the ICA method http://www.ipol.im/pub/art/2016/153/
    The optical flow follows a translation per patch model, such that :
    ref_img(X) ~= comp_img(X + flow(X))
    

    Parameters
    ----------
    cuda_img_grey : device Array[imsize_y, imsize_x]
        Image to align on grey level G_n
    cuda_ref_grey : device Array[imsize_y, imsize_x]
        Reference image on grey level G_1
    cuda_gradx : device array[imsize_y, imsize_x]
        Horizontal gradient of the reference image
    cuda_grady : device array[imsize_y, imsize_x]
        Vertical gradient of the reference image
    hessian : device_array[n_tiles_y, n_tiles_x, 2, 2]
        Hessian matrix of the reference image
    cuda_pre_alignment : device Array[n_tiles_y, n_tiles_x, 2]
        optical flow for each tile of each image, outputed by bloc matching : V_n
        pre_alignment[0] must be the horizontal flow oriented towards the right if positive.
        pre_alignment[1] must be the vertical flow oriented towards the bottom if positive.
    options : dict
        options
    params : dict
        ['tuning']['kanadeIter'] : int
            Number of iterations.
        params['tuning']['tileSize'] : int
            Size of the tiles.
        params["mode"] : {"bayer", "grey"}
            Mode of the pipeline : whether the original burst is grey or raw
            
        params['tuning']['sigma blur'] : float
            If non zero, applies a gaussian blur before computing gradients.
            
    debug : bool, optional
        If True, this function returns a list containing the flow at each iteration.
        The default is False.

    Returns
    -------
    cuda_alignment : device_array[n_tiles_y, n_tiles_x, 2]
        Updated alignment vectors V_n(p) for each tile of the image

    """
    if debug : 
        debug_list = []

    n_iter = params['tuning']['kanadeIter']
        
    cuda_alignment = cuda_pre_alignment
    
    for iter_index in range(n_iter):
        ICA_optical_flow_iteration(
            cuda_ref_grey, cuda_gradx, cuda_grady, cuda_im_grey, cuda_alignment, hessian,
            options, params, iter_index)
        
        if debug :
            debug_list.append(cuda_alignment.copy_to_host())
        
    if debug:
        return debug_list
    return cuda_alignment

    
def ICA_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, hessian, options, params,
                               iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img : Array [imsize_y, imsize_x]
        Ref image (grey)
    gradx : Array [imsize_y, imsize_x]
        Horizontal gradient of the ref image
    grady : Array [imsize_y, imsize_x]
        Vertical gradient of the ref image
    comp_img : Array[imsize_y, imsize_x]
        The image to rearrange and compare to the reference (grey images)
    alignment : Array[n_tiles_y, n_tiles_x, 2]
        The inial alignment of the tiles
    options : Dict
        Options to pass
    params : Dict
        parameters
    iter_index : int
        The iteration index (for printing evolution when verbose >2,
                             and for clearing memory)

    """
    verbose_3 = options['verbose'] >= 3
    tile_size = params['tuning']['tileSize']

    n_patch_y, n_patch_x, _ = alignment.shape
    
    if verbose_3 :
        cuda.synchronize()
        current_time = time.perf_counter()
        print(" -- Lucas-Kanade iteration {}".format(iter_index))
    
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    
    blockspergrid_x = math.ceil(n_patch_x/threadsperblock[1])
    blockspergrid_y = math.ceil(n_patch_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    ICA_get_new_flow[blockspergrid, threadsperblock](
        ref_img, comp_img,
        gradsx, gradsy,
        alignment, hessian, tile_size)   

    if verbose_3:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')



@cuda.jit
def ICA_get_new_flow(ref_img, comp_img, gradx, grady, alignment, hessian, tile_size):
    """
    The update relies on solving AX = B, a 2 by 2 system.
    A is precomputed, but B is evaluated each time. 

    """
    imsize_y, imsize_x = comp_img.shape
    n_patchs_y, n_patchs_x, _ = alignment.shape
    patch_idx, patch_idy = cuda.grid(2)
    
    if not(0 <= patch_idy < n_patchs_y and
           0 <= patch_idx < n_patchs_x):
        return
    
    patch_pos_x = tile_size * patch_idx
    patch_pos_y = tile_size * patch_idy
    
    A = cuda.local.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    A[0, 0] = hessian[patch_idy, patch_idx, 0, 0]
    A[0, 1] = hessian[patch_idy, patch_idx, 0, 1]
    A[1, 0] = hessian[patch_idy, patch_idx, 1, 0]
    A[1, 1] = hessian[patch_idy, patch_idx, 1, 1]
    
    B = cuda.local.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    B[0] = 0
    B[1] = 0
    
    local_alignment = cuda.local.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    local_alignment[0] = alignment[patch_idy, patch_idx, 0]
    local_alignment[1] = alignment[patch_idy, patch_idx, 1]
    
    buffer_val = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x
                
    for i in range(tile_size):
        for j in range(tile_size):
            pixel_global_idx = patch_pos_x + j # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
            pixel_global_idy = patch_pos_y + i
            
            inbound = (0 <= pixel_global_idx < imsize_x and 
                       0 <= pixel_global_idy < imsize_y)
            
            if inbound :
                local_gradx = gradx[pixel_global_idy, pixel_global_idx]
                local_grady = grady[pixel_global_idy, pixel_global_idx]

                # Warp I with W(x; p) to compute I(W(x; p))
                new_idx = local_alignment[0] + pixel_global_idx
                new_idy = local_alignment[1] + pixel_global_idy 
 
            inbound &= (0 <= new_idx < imsize_x - 1 and
                        0 <= new_idy < imsize_y - 1) # -1 for bicubic interpolation
    
            if inbound:
                # bicubic interpolation
                normalised_pos_x, floor_x = math.modf(new_idx) # https://www.rollpie.com/post/252
                normalised_pos_y, floor_y = math.modf(new_idy) # separating floor and floating part
                floor_x = int(floor_x)
                floor_y = int(floor_y)
                
                ceil_x = floor_x + 1
                ceil_y = floor_y + 1
                pos[0] = normalised_pos_y
                pos[1] = normalised_pos_x
                
                buffer_val[0, 0] = comp_img[floor_y, floor_x]
                buffer_val[0, 1] = comp_img[floor_y, ceil_x]
                buffer_val[1, 0] = comp_img[ceil_y, floor_x]
                buffer_val[1, 1] = comp_img[ceil_y, ceil_x]
                
                comp_val = bilinear_interpolation(buffer_val, pos)
                
                gradt = comp_val - ref_img[pixel_global_idy, pixel_global_idx]
                
                
                
                B[0] += -local_gradx*gradt
                B[1] += -local_grady*gradt
        
    
    alignment_step = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    if abs(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]) > 1e-5: # system is solvable 
        solve_2x2(A, B, alignment_step)
        
        alignment[patch_idy, patch_idx, 0] = local_alignment[0] + alignment_step[0]
        alignment[patch_idy, patch_idx, 1] = local_alignment[1] + alignment_step[1]
