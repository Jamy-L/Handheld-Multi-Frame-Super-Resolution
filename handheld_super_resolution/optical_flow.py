# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

@author: jamyl
"""

import time

from math import ceil, modf, exp
import numpy as np
from numba import cuda
from scipy.ndimage._filters import _gaussian_kernel1d
import torch.nn.functional as F
import torch

from .linalg import bilinear_interpolation
from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE
from .linalg import solve_2x2
    
def init_ICA(ref_img, options, params):
    current_time, verbose, verbose_2 = time.perf_counter(), options['verbose'] > 1, options['verbose'] > 2

    sigma_blur = params['tuning']['sigma blur']
    tile_size = params['tuning']['tileSize']
    
    imsize_y, imsize_x = ref_img.shape

    n_patch_y = int(ceil(imsize_y/(tile_size//2))) + 1
    n_patch_x = int(ceil(imsize_x/(tile_size//2))) + 1

    # Estimating gradients with separated Prewitt kernels
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
        th_gaussian_kernel = torch.as_tensor(gaussian_kernel, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
        
        
        # 2 times gaussian 1d is faster than gaussian 2d
        # TODO I have not checked if the gaussian blur was exactly the same as outputed byt gaussian_filter1d
        
        temp = F.conv2d(th_ref_img, th_gaussian_kernel[None, None, :, None]) # convolve y
        temp = F.conv2d(temp, th_gaussian_kernel[None, None, None, :]) # convolve x
        
        
        th_gradx = F.conv2d(temp, th_kernelx)[0, 0] # 1 batch, 1 channel
        th_grady = F.conv2d(temp, th_kernely)[0, 0]
        
        
        # Old implementation (equivalent on CPU)
        # temp = gaussian_filter1d(ref_img, sigma=sigma_blur, axis=-1)
        # temp = gaussian_filter1d(temp, sigma=sigma_blur, axis=-2)
                
        
        # gradx = cv2.filter2D(temp, -1, kernelx)
        # grady = cv2.filter2D(temp, -1, kernely)
    else:
        th_gradx = F.conv2d(th_ref_img, th_kernelx)[0, 0] # 1 batch, 1 channel
        th_grady = F.conv2d(th_ref_img, th_kernely)[0, 0]
        
        # gradx = cv2.filter2D(ref_img, -1, kernelx)
        # grady = cv2.filter2D(ref_img, -1, kernely)
    
    # swapping grads back to numba
    gradx = cuda.as_cuda_array(th_gradx)
    grady = cuda.as_cuda_array(th_grady)
    
    if verbose_2:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Gradients estimated')
    
    cuda_gradx = cuda.to_device(gradx)
    cuda_grady = cuda.to_device(grady)
    
    if verbose_2: 
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Arrays moved to GPU')
        
    hessian = cuda.device_array((n_patch_y, n_patch_x, 2, 2), DEFAULT_NUMPY_FLOAT_TYPE)
    compute_hessian[(n_patch_x, n_patch_y), (tile_size, tile_size)](
        cuda_gradx, cuda_grady, tile_size, hessian)
    
    if verbose_2:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Hessian estimated')
    
    return cuda_gradx, cuda_grady, hessian
    
@cuda.jit
def compute_hessian(gradx, grady, tile_size, hessian):
    imshape = gradx.shape
    patch_idx, patch_idy = cuda.blockIdx.x, cuda.blockIdx.y
    pixel_local_idx = cuda.threadIdx.x # Position relative to the patch
    pixel_local_idy = cuda.threadIdx.y
    
    pixel_global_idx = tile_size//2 * (patch_idx - 1) + pixel_local_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    pixel_global_idy = tile_size//2 * (patch_idy - 1) + pixel_local_idy
    
    inbound = (0 <= pixel_global_idy < imshape[0] and 0 <= pixel_global_idx < imshape[1])
    
    
    if pixel_local_idx < 2 and pixel_local_idy < 2:
        hessian[patch_idy, patch_idx, pixel_local_idy, pixel_local_idx] = 0 # zero init
    cuda.syncthreads()
    
    # TODO gaussian windowing ?
    if inbound : 
        local_gradx = gradx[pixel_global_idy, pixel_global_idx]
        local_grady = grady[pixel_global_idy, pixel_global_idx]
        
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 0, 0), local_gradx*local_gradx)
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 0, 1), local_gradx*local_grady)
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 1, 0), local_gradx*local_grady)
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 1, 1), local_grady*local_grady)


def ICA_optical_flow(cuda_im_grey, cuda_ref_grey, cuda_gradx, cuda_grady, hessian, cuda_pre_alignment, options, params, debug = False):
    """ Computes optical flow between the ref_img and all images of comp_imgs 
    based on the ICA method (http://www.ipol.im/pub/art/2016/153/).
    The optical flow follows a translation per patch model, such that :
    ref_img(X) ~= comp_img(X + flow(X))
    

    Parameters
    ----------
    ref_img : device Array[imsize_y, imsize_x]
        reference image on grey level
    comp_img : device Array[imsize_y, imsize_x]
        image to align on grey level
    pre_alignment : device Array[n_tiles_y, n_tiles_x, 2]
        optical flow for each tile of each image, outputed by bloc matching.
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
        alignment vector for each tile of each image following the same convention
        as "pre_alignment"

    """
    if debug : 
        debug_list = []
        
    verbose, verbose_2 = options['verbose'] > 1, options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']
    
    n_patch_y, n_patch_x, _ = cuda_pre_alignment.shape

    if verbose:
        cuda.synchronize()
        t1 = time.perf_counter()
        print("\nEstimating Lucas-Kanade's optical flow")
        
    cuda_alignment = cuda_pre_alignment

    if verbose_2 : 
        cuda.synchronize()
        getTime(
            t1, ' -- Arrays moved to GPU')
    

    for iter_index in range(n_iter):
        ICA_optical_flow_iteration(
            cuda_ref_grey, cuda_gradx, cuda_grady, cuda_im_grey, cuda_alignment, hessian,
            options, params, iter_index)
        
        if debug :
            debug_list.append(cuda_alignment.copy_to_host())
    
    if verbose:
        cuda.synchronize()
        getTime(t1, 'Flows estimated (Total)')
        print('\n')
        
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

    Returns
    -------
    new_alignment
        Array[n_images, n_tiles_y, n_tiles_x, 2]
            The adjusted tile alignment

    """
    verbose_2 = options['verbose'] > 2
    tile_size = params['tuning']['tileSize']

    imsize_y, imsize_x = comp_img.shape
    n_patch_y, n_patch_x, _ = alignment.shape
    
    if verbose_2 :
        cuda.synchronize()
        current_time = time.perf_counter()
        print(" -- Lucas-Kanade iteration {}".format(iter_index))
    
    ICA_get_new_flow[[(n_patch_y, n_patch_x), (tile_size, tile_size)]
        ](ref_img, comp_img, gradsx, gradsy, alignment, hessian, tile_size)   

    if verbose_2:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')



@cuda.jit
def ICA_get_new_flow(ref_img, comp_img, gradx, grady, alignment, hessian, tile_size):
    imsize_y, imsize_x = comp_img.shape
    
    patch_idy, patch_idx = cuda.blockIdx.x, cuda.blockIdx.y
    pixel_local_idx = cuda.threadIdx.x # Position relative to the patch
    pixel_local_idy = cuda.threadIdx.y
    
    pixel_global_idx = tile_size//2 * (patch_idx - 1) + pixel_local_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    pixel_global_idy = tile_size//2 * (patch_idy - 1) + pixel_local_idy
    
    ATA = cuda.shared.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    ATB = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    
    # parallel init
    if cuda.threadIdx.x <= 1 and cuda.threadIdx.y <=1 : # copy hessian to threads
        ATA[cuda.threadIdx.y, cuda.threadIdx.x] = hessian[patch_idy, patch_idx,
                                                          cuda.threadIdx.y, cuda.threadIdx.x]
    if cuda.threadIdx.y == 2 and cuda.threadIdx.x <= 1 :
            ATB[cuda.threadIdx.x] = 0
  
    
    inbound = (0 <= pixel_global_idx < imsize_x) and (0 <= pixel_global_idy < imsize_y)

    if inbound :
        # Warp I with W(x; p) to compute I(W(x; p))
        new_idx = alignment[patch_idy, patch_idx, 0] + pixel_global_idx 

        
        new_idy = alignment[patch_idy, patch_idx, 1] + pixel_global_idy 
 
    
    inbound = inbound and (0 <= new_idx < imsize_x -1) and (0 <= new_idy < imsize_y - 1) # -1 for bicubic interpolation
    
    if inbound:
        # bicubic interpolation
        buffer_val = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
        pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x
        normalised_pos_x, floor_x = modf(new_idx) # https://www.rollpie.com/post/252
        normalised_pos_y, floor_y = modf(new_idy) # separating floor and floating part
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
        gradt = comp_val- ref_img[pixel_global_idy, pixel_global_idx]
               
        # exponentially decreasing window, of size tile_size_lk
        r_square = ((pixel_local_idx - tile_size/2)**2 +
                    (pixel_local_idy - tile_size/2)**2)
        sigma_square = (tile_size/2)**2
        w = exp(-r_square/(3*sigma_square)) # TODO this is not improving LK so much... 3 seems to be a good coef though.
        
        local_gradx = gradx[pixel_global_idy, pixel_global_idx]
        local_grady = grady[pixel_global_idy, pixel_global_idx]
        
        cuda.atomic.add(ATB, 0, -local_gradx*gradt*w)
        cuda.atomic.add(ATB, 1, -local_grady*gradt*w)

    alignment_step = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    cuda.syncthreads()
        
    if cuda.threadIdx.x <= 1 and cuda.threadIdx.y == 0:
        if abs(ATA[0, 0]*ATA[1, 1] - ATA[0, 1]*ATA[1, 0])> 1e-5: # system is solvable 
            # No racing condition, one thread for one id
            solve_2x2(ATA, ATB, alignment_step)
            alignment[patch_idy, patch_idx, cuda.threadIdx.x] += alignment_step[cuda.threadIdx.x]

    cuda.syncthreads()
