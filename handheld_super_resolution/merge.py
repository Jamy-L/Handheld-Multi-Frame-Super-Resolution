# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""


import time
import math

import numpy as np
from numba import uint8, cuda

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS
from .linalg import quad_mat_prod, invert_2x2, interpolate_cov

def init_merge(ref_img, kernels, options, params):
    VERBOSE = options['verbose']
    SCALE = params['scale']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    bayer_mode = params['mode'] == 'bayer'
    act = params['kernel'] == 'act'

    if VERBOSE > 1:
        cuda.synchronize()
        print('Beginning merge process')
        current_time = time.perf_counter()
        

    native_im_size = ref_img.shape
    # casting to integer to account for floating scale
    output_size = (round(SCALE*native_im_size[0]), round(SCALE*native_im_size[1]))

    num = cuda.device_array(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE)
    den = cuda.device_array(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE)


    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    
    blockspergrid_x = int(np.ceil(output_size[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(output_size[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    accumulate_ref[blockspergrid, threadsperblock](
        ref_img, kernels, bayer_mode, act, SCALE, CFA_pattern,
        num, den)
    
    
    if VERBOSE > 1:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' - Ref frame merged')
    
    return num, den
    
@cuda.jit
def accumulate_ref(ref_img, covs, bayer_mode, act, scale, CFA_pattern,
                   num, den):
    """



    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    covs : device array[grey_imsize_y, grey_imsize_x, 2, 2]
        covariance matrices sampled at the center of each grey pixel.
    bayer_mode : bool
        Whether the burst is raw or grey
    act : bool
        Whether ACT kernels should be used, or handhled's kernels.
    scale : float
        scaling factor
    CFA_pattern : Array[2, 2]
        CFA pattern of the burst
    output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
        The empty output image

    Returns
    -------
    None.

    """

    output_pixel_idx, output_pixel_idy = cuda.grid(2)
    output_size_y, output_size_x, _ = num.shape
    
    if not (0 <= output_pixel_idx < output_size_x and
            0 <= output_pixel_idy < output_size_y):
        return
    
    
    if bayer_mode:
        n_channels = 3
        acc = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        n_channels = 1
        acc = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    # Copying CFA locally. We will read that 9 times, so it's worth it
    # TODO threads could cooperate to read that
    local_CFA = cuda.local.array((2,2), uint8)
    for i in range(2):
        for j in range(2):
            local_CFA[i,j] = uint8(CFA_pattern[i,j])
    
    
    coarse_ref_sub_pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # y, x
    coarse_ref_sub_pos[0] = output_pixel_idy / scale          
    coarse_ref_sub_pos[1] = output_pixel_idx / scale
    
    for chan in range(n_channels):
        acc[chan] = 0
        val[chan] = 0

    
    # computing kernel
    # TODO this is rather slow and could probably be sped up
    if not act:
        interpolated_cov = cuda.local.array((2, 2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cov_i = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        
        # fetching the 4 closest covs
        close_covs = cuda.local.array((2, 2, 2 ,2), DEFAULT_CUDA_FLOAT_TYPE)
        grey_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
        if bayer_mode:
            grey_pos[0] = (coarse_ref_sub_pos[0]-0.5)/2 # grey grid is offseted and twice more sparse
            grey_pos[1] = (coarse_ref_sub_pos[1]-0.5)/2
            
        else:
            grey_pos[0] = coarse_ref_sub_pos[0] # grey grid is exactly the coarse grid
            grey_pos[1] = coarse_ref_sub_pos[1]
    
        for i in range(0, 2): # TODO Check undesirable effects on the imagse side
            for j in range(0, 2):    
                close_covs[0, 0, i, j] = covs[ 
                                                int(math.floor(grey_pos[0])),
                                                int(math.floor(grey_pos[1])),
                                                i, j]
                close_covs[0, 1, i, j] = covs[
                                                int(math.floor(grey_pos[0])),
                                                int(math.ceil(grey_pos[1])),
                                                i, j]
                close_covs[1, 0, i, j] = covs[
                                                int(math.ceil(grey_pos[0])),
                                                int(math.floor(grey_pos[1])),
                                                i, j]
                close_covs[1, 1, i, j] = covs[
                                                int(math.ceil(grey_pos[0])),
                                                int(math.ceil(grey_pos[1])),
                                                i, j]

        # interpolating covs
        interpolate_cov(close_covs, grey_pos, interpolated_cov)
        
        if abs(interpolated_cov[0, 0]*interpolated_cov[1, 1] - interpolated_cov[0, 1]*interpolated_cov[1, 0]) > 1e-6: # checking if cov is invertible
            invert_2x2(interpolated_cov, cov_i)
        else: # if not invertible, identity matrix
            cov_i[0, 0] = 1
            cov_i[0, 1] = 0
            cov_i[1, 0] = 0
            cov_i[1, 1] = 1
                    
        
    center_x = round(coarse_ref_sub_pos[1])
    center_y = round(coarse_ref_sub_pos[0])
    for i in range(-1, 2):
        for j in range(-1, 2):
            pixel_idx = center_x + j
            pixel_idy = center_y + i
            
            # in bound condition
            if (0 <= pixel_idx < output_size_x and
                0 <= pixel_idy < output_size_y):
            
                # checking if pixel is r, g or b
                if bayer_mode : 
                    channel = local_CFA[pixel_idy%2, pixel_idx%2]
                else:
                    channel = 0
                    
                # By fetching the value now, we can compute the kernel weight 
                # while it is called from global memory
                c = ref_img[pixel_idy, pixel_idx]
            
                # computing distance
                dist_x = pixel_idx - coarse_ref_sub_pos[1]
                dist_y = pixel_idy - coarse_ref_sub_pos[0]
            
                ### Computing w
                if act : 
                    y = max(0, 2*(dist_x*dist_x + dist_y*dist_y))
                else:
                    y = max(0, quad_mat_prod(cov_i, dist_x, dist_y))
                    # y can be slightly negative because of numerical precision.
                    # I clamp it to not explode the error with exp
                if bayer_mode : 
                    w = math.exp(-0.5*y)
                else:
                    w = math.exp(-0.5*4*y) # original kernel constants are designed for bayer distances, not greys, Hence x4
                ############
                val[channel] += c*w
                acc[channel] += w
                    

    for chan in range(n_channels):
        num[output_pixel_idy, output_pixel_idx, chan] = val[chan]
        den[output_pixel_idy, output_pixel_idx, chan] = acc[chan]
          
    
def merge(comp_img, alignments, covs, r, num, den,
          options, params):
    """
    Merges all the images, based on the alignments previously estimated.
    The size of the merge_result is adjustable with params['scale']


    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    comp_imgs : Array [n_images, imsize_y, imsize_x]
        The compared images
    alignments : device Array[n_images, n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment (patchwise)
    covs : device array[n_images+1, imsize_y/2, imsize_x/2, 2, 2]
        covariance matrices sampled at the center of each bayer quad.
    r : Device_Array[n_images, imsize_y/2, imsize_x/2, 3]
            Robustness of the moving images
    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    merge_result : Array[scale * imsize_y, scale * imsize_x, 3]
        merged images

    """
    VERBOSE = options['verbose']
    SCALE = params['scale']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    bayer_mode = params['mode'] == 'bayer'
    act = params['kernel'] == 'act'
    TILE_SIZE = params['tuning']['tileSize']
    N_TILES_Y, N_TILES_X, _ = alignments.shape

    if VERBOSE > 1:
        print('\nBeginning merge process')
        cuda.synchronize()
        current_time = time.perf_counter()

    native_im_size = comp_img.shape
    # casting to integer to account for floating scale
    output_size = (round(SCALE*native_im_size[0]), round(SCALE*native_im_size[1]))


    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = int(np.ceil(output_size[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(output_size[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
                    
    accumulate[blockspergrid, threadsperblock](
        comp_img, alignments, covs, r,
        bayer_mode, act, SCALE, TILE_SIZE, CFA_pattern,
        num, den)

    if VERBOSE > 1:
        cuda.synchronize()
        getTime(current_time, ' - Image merged on GPU side')


@cuda.jit
def accumulate(comp_img, alignments, covs, r,
               bayer_mode, act, scale, tile_size, CFA_pattern,
               num, den):
    """
    Cuda kernel, each block represents an output pixel. Each block contains
    a 3 by 3 neighborhood for each moving image. A single threads takes
    care of one of these pixels, for all the moving images.



    Parameters
    ----------
    comp_imgs : Array[imsize_y, imsize_x]
        The compared image
    alignements : Array[n_tiles_y, n_tiles_x, 2]
        The alignemnt vectors for each tile of the image
    covs : device array[imsize_y/2, imsize_x/2, 2, 2]
        covariance matrices sampled at the center of each bayer quad.
    r : Device_Array[imsize_y/2, imsize_x/2, 3]
            Robustness of the moving images
    bayer_mode : bool
        Whether the burst is raw or grey
    act : bool
        Whether ACT kernels should be used, or handhled's kernels.
    scale : float
        scaling factor
    tile_size : int
        tile size used for alignment (on the raw scale !)
    CFA_pattern : device Array[2, 2]
        CFA pattern of the burst
    output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
        The empty output image

    Returns
    -------
    None.

    """

    output_pixel_idx, output_pixel_idy = cuda.grid(2)

    
    output_size_y, output_size_x, _ = num.shape
    input_size_y, input_size_x = comp_img.shape
    
    if not (0 <= output_pixel_idx < output_size_x and
            0 <= output_pixel_idy < output_size_y):
        return
    
    if bayer_mode:
        n_channels = 3
        acc = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        n_channels = 1
        acc = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    # Copying CFA locally. We will read that 9 times, so it's worth it
    # TODO threads could cooperate to read that
    local_CFA = cuda.local.array((2,2), uint8)
    for i in range(2):
        for j in range(2):
            local_CFA[i,j] = uint8(CFA_pattern[i,j])

    
    coarse_ref_sub_pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # y, x
    
    coarse_ref_sub_pos[0] = output_pixel_idy / scale          
    coarse_ref_sub_pos[1] = output_pixel_idx / scale

    # fetch of the flow, as early as possible
    local_optical_flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    patch_idy = round(coarse_ref_sub_pos[0]//(tile_size//2)) + 1 # +1 because padding was made in block matching. The first patch is out of bounds
    patch_idx = round(coarse_ref_sub_pos[1]//(tile_size//2)) + 1
    local_optical_flow[0] = alignments[patch_idy, patch_idx, 0]
    local_optical_flow[1] = alignments[patch_idy, patch_idx, 1]
    
    for chan in range(n_channels):
        acc[chan] = 0
        val[chan] = 0
    
    patch_center_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x


    # fetching robustness
    # The robustness of the center of the patch is picked through neirest neigbhoor interpolation

    if bayer_mode : 
        local_r = r[round((coarse_ref_sub_pos[0] - 0.5)/2),
                    round((coarse_ref_sub_pos[1] - 0.5)/2)]

    else:
        local_r = r[int(coarse_ref_sub_pos[0]),
                    int(coarse_ref_sub_pos[1])]
        
    patch_center_pos[1] = coarse_ref_sub_pos[1] + local_optical_flow[0]
    patch_center_pos[0] = coarse_ref_sub_pos[0] + local_optical_flow[1]

    
    # updating inbound condition
    if not (0 <= patch_center_pos[1] < input_size_x and
            0 <= patch_center_pos[0] < input_size_y):
        return
    
    # computing kernel
    if not act:
        interpolated_cov = cuda.local.array((2, 2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cov_i = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        # fetching the 4 closest covs
        close_covs = cuda.local.array((2, 2, 2 ,2), DEFAULT_CUDA_FLOAT_TYPE)
        grey_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
        
        if bayer_mode :
            grey_pos[0] = (patch_center_pos[0]-0.5)/2 # grey grid is offseted and twice more sparse
            grey_pos[1] = (patch_center_pos[1]-0.5)/2
            
        else:
            grey_pos[0] = patch_center_pos[0] # grey grid is exactly the coarse grid
            grey_pos[1] = patch_center_pos[1]
        
        for i in range(0, 2):
            for j in range(0, 2):# TODO sides can get negative grey indexes. It leads to weird covs.
                close_covs[0, 0, i, j] = covs[int(math.floor(grey_pos[0])),
                                              int(math.floor(grey_pos[1])),
                                              i, j]
                close_covs[0, 1, i, j] = covs[int(math.floor(grey_pos[0])),
                                              int(math.ceil(grey_pos[1])),
                                              i, j]
                close_covs[1, 0, i, j] = covs[int(math.ceil(grey_pos[0])),
                                              int(math.floor(grey_pos[1])),
                                              i, j]
                close_covs[1, 1, i, j] = covs[int(math.ceil(grey_pos[0])),
                                              int(math.ceil(grey_pos[1])),
                                              i, j]

        # interpolating covs at the desired spot
        interpolate_cov(close_covs, grey_pos, interpolated_cov)

        if abs(interpolated_cov[0, 0]*interpolated_cov[1, 1] - interpolated_cov[0, 1]*interpolated_cov[1, 0]) > 1e-6: # checking if cov is invertible
            invert_2x2(interpolated_cov, cov_i)
        else:
            cov_i[0, 0] = 1
            cov_i[0, 1] = 0
            cov_i[1, 0] = 0
            cov_i[1, 1] = 1
    
    
    center_x = round(patch_center_pos[1])
    center_y = round(patch_center_pos[0])
    for i in range(-1, 2):
        for j in range(-1, 2):
            pixel_idx = center_x + j
            pixel_idy = center_y + i
            
            # in bound condition
            if (0 <= pixel_idx < output_size_x and
                0 <= pixel_idy < output_size_y):
            
                # checking if pixel is r, g or b
                if bayer_mode : 
                    channel = local_CFA[pixel_idy%2, pixel_idx%2]
                else:
                    channel = 0
                    
                # By fetching the value now, we can compute the kernel weight 
                # while it is called from global memory
                c = comp_img[pixel_idy, pixel_idx]
            
                # computing distance
                dist_x = pixel_idx - patch_center_pos[1]
                dist_y = pixel_idy - patch_center_pos[0]
            
                ### Computing w
                if act : 
                    y = max(0, 2*(dist_x*dist_x + dist_y*dist_y))
                else:
                    y = max(0, quad_mat_prod(cov_i, dist_x, dist_y))
                    # y can be slightly negative because of numerical precision.
                    # I clamp it to not explode the error with exp
                if bayer_mode : 
                    w = math.exp(-0.5*y)
                else:
                    w = math.exp(-0.5*4*y) # original kernel constants are designed for bayer distances, not greys, Hence x4
                ############
                    
                val[channel] += c*w*local_r
                acc[channel] += w*local_r
        
    for chan in range(n_channels):
        num[output_pixel_idy, output_pixel_idx, chan] += val[chan] 
        den[output_pixel_idy, output_pixel_idx, chan] += acc[chan]
    
    
    
    