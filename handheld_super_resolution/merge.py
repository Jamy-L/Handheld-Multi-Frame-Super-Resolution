# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

This script contains : 
    - The implementation of Alg. 4, the conventionnal accumulation
    - The implementation of Alg. 11, where the reference image is merged


@author: jamyl
"""


import math

from numba import uint8, cuda

from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS
from .utils_image import denoise_power_merge, denoise_range_merge
from .linalg import quad_mat_prod, invert_2x2, interpolate_cov

def merge_ref(ref_img, kernels, num, den, cfa_pattern, config, acc_rob=None):
    """
    Implementation of Alg. 11: AccumulationReference
    Accumulates the reference frame into num and den, while considering
    (if enabled) the accumulated robustness mask to enforce single frame SR if
    necessary.

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
    kernels : device Array[imshape_y//2, imshape_x//2, 2, 2]
        Covariance Matrices Omega_1
    num : device Array[s*imshape_y, s*imshape_x, c]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x, c]
        Denominator of the accumulator
    config : OmegaConf object
        parameters.
    acc_rob : [imshape_y, imshape_x], optional
        accumulated robustness mask. The default is None.

    Returns
    -------
    None.

    """
    scale = config.scale

    bayer_mode = config.mode == 'bayer'
    iso_kernel = config.merging.kernel == 'iso'

    robustness_denoise = config.accumulated_robustness_denoiser.enabled
    # numba is strict on types and dimension : let's use a consistent object
    # for acc_rob even when it is not used.
    if robustness_denoise:
        rad_max = config.accumulated_robustness_denoiser.merge.rad_max
        max_multiplier = config.accumulated_robustness_denoiser.merge.max_multiplier
        max_frame_count = config.accumulated_robustness_denoiser.merge.max_frame_count
    else:
        acc_rob = cuda.device_array((1,1), DEFAULT_NUMPY_FLOAT_TYPE)
        rad_max = 0
        max_multiplier = 0.
        max_frame_count = 0
    
    
    output_shape_y, output_shape_x, _ = num.shape

    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    
    blockspergrid_x = math.ceil(output_shape_x/threadsperblock[1])
    blockspergrid_y = math.ceil(output_shape_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    accumulate_ref[blockspergrid, threadsperblock](
        ref_img, kernels, bayer_mode, iso_kernel, scale, cfa_pattern,
        num, den, acc_rob, robustness_denoise, max_frame_count, rad_max, max_multiplier)
    
    
@cuda.jit
def accumulate_ref(ref_img, covs, bayer_mode, iso_kernel, scale, CFA_pattern,
                   num, den, acc_rob,
                   robustness_denoise, max_frame_count, rad_max, max_multiplier):
    
    output_pixel_idx, output_pixel_idy = cuda.grid(2)
    output_size_y, output_size_x, _ = num.shape
    input_size_y, input_size_x = ref_img.shape
    
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
    # this is rather slow and could probably be sped up
    if not iso_kernel:
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
    
    
        # clipping the coordinates to stay in bound
        floor_x = int(max(math.floor(grey_pos[1]), 0))
        floor_y = int(max(math.floor(grey_pos[0]), 0))
        
        ceil_x = min(floor_x + 1, covs.shape[1]-1)
        ceil_y = min(floor_y + 1, covs.shape[0]-1)
        for i in range(0, 2):
            for j in range(0, 2):
                close_covs[0, 0, i, j] = covs[floor_y, floor_x,
                                              i, j]
                close_covs[0, 1, i, j] = covs[floor_y, ceil_x,
                                              i, j]
                close_covs[1, 0, i, j] = covs[ceil_y, floor_x,
                                              i, j]
                close_covs[1, 1, i, j] = covs[ceil_y, ceil_x,
                                              i, j]

        # interpolating covs
        interpolate_cov(close_covs, grey_pos, interpolated_cov)
        
        invert_2x2(interpolated_cov, cov_i)

            
    
    # fetching acc robustness if required
    # Acc robustness is known for each raw pixel. An implicit interpolation done
    # from LR to HR using nearest neighbor. 
    if robustness_denoise : 
        local_acc_r = acc_rob[min(round(coarse_ref_sub_pos[0]), acc_rob.shape[0]-1),
                              min(round(coarse_ref_sub_pos[1]), acc_rob.shape[1]-1)]
        
        additional_denoise_power = denoise_power_merge(local_acc_r, max_multiplier, max_frame_count)
        rad = denoise_range_merge(local_acc_r, rad_max, max_frame_count)
        
    else:
        additional_denoise_power = 1
        rad = 1     

        
    center_x = round(coarse_ref_sub_pos[1])
    center_y = round(coarse_ref_sub_pos[0])
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            pixel_idx = center_x + j
            pixel_idy = center_y + i
            
            # in bound condition
            if (0 <= pixel_idx < input_size_x and
                0 <= pixel_idy < input_size_y):
            
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
                if iso_kernel : 
                    y = max(0, 2*(dist_x*dist_x + dist_y*dist_y))
                else:
                    y = max(0, quad_mat_prod(cov_i, dist_x, dist_y))
                    # y can be slightly negative because of numerical precision.
                    # I clamp it to not explode the error with exp
                    
                
                # this is equivalent to multiplying the covariance,
                # but at the cost of one scalar operation (instead of 4)
                y/= additional_denoise_power
                
                w = math.exp(-0.5*y)
                ############
                
                val[channel] += c*w
                acc[channel] += w
    
    if robustness_denoise and local_acc_r < max_frame_count:
        # Overwritting values to enforce single frame
        # demosaicing        
        for chan in range(n_channels):
            num[output_pixel_idy, output_pixel_idx, chan] = val[chan]
            den[output_pixel_idy, output_pixel_idx, chan] = acc[chan]
        
    else:
        for chan in range(n_channels):
            num[output_pixel_idy, output_pixel_idx, chan] += val[chan]
            den[output_pixel_idy, output_pixel_idx, chan] += acc[chan]
          
    
def merge(comp_img, alignments, covs, r, num, den, cfa_pattern, config):
    """
    Implementation of Alg. 4: Accumulation
    Accumulates comp_img (J_n, n>1) into num and den, based on the alignment
    V_n, the covariance matrices Omega_n and the robustness mask estimated before.


    Parameters
    ----------
    comp_imgs : device Array [imsize_y, imsize_x]
        The non-reference image to merge (J_n)
    alignments : device Array[n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment V_n(p)
    covs : device array[imsize_y//2, imsize_x//2, 2, 2]
        covariance matrices Omega_n
    r : Device_Array[imsize_y, imsize_x]
        Robustness mask r_n
    num : device Array[s*imshape_y, s*imshape_x, c]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x, c]
        Denominator of the accumulator
        
    config : OmegaConf object
        parameters.

    Returns
    -------
    None

    """
    scale = config.scale

    bayer_mode = config.mode == 'bayer'
    iso_kernel = config.merging.kernel == 'iso'
    tile_size = config.block_matching.tuning.tile_size

    native_im_size = comp_img.shape
    # casting to integer to account for floating scale
    output_size = (round(scale*native_im_size[0]), round(scale*native_im_size[1]))


    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(output_size[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(output_size[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
                    
    accumulate[blockspergrid, threadsperblock](
        comp_img, alignments, covs, r,
        bayer_mode, iso_kernel, scale, tile_size, cfa_pattern,
        num, den)



@cuda.jit
def accumulate(comp_img, alignments, covs, r,
               bayer_mode, iso_kernel, scale, tile_size, CFA_pattern,
               num, den):
    hr_j, hr_i = cuda.grid(2)

    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = comp_img.shape

    if not (0 <= hr_j < hr_w and
            0 <= hr_i < hr_h):
        return
    
    if bayer_mode:
        n_channels = 3
        acc = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        n_channels = 1
        acc = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    l_cfa = cuda.local.array((2,2), uint8)
    l_cfa[0,0] = uint8(CFA_pattern[0,0])
    l_cfa[0,1] = uint8(CFA_pattern[0,1])
    l_cfa[1,0] = uint8(CFA_pattern[1,0])
    l_cfa[1,1] = uint8(CFA_pattern[1,1])


    lr_x = (hr_j + 0.5) / scale
    lr_y = (hr_i + 0.5) / scale

    px = int(lr_x//tile_size)
    py = int(lr_y//tile_size)
    flowx = alignments[py, px, 0]
    flowy = alignments[py, px, 1]

    for chan in range(n_channels):
        acc[chan] = 0
        val[chan] = 0
    

    # fetching robustness
    # The robustness coefficient is known for every raw pixel, and implicitely
    # interpolated to HR using nearest neighboor interpolations.
    i_r = min(int(lr_y), lr_h-1)
    j_r = min(int(lr_x), lr_w-1)
    local_r = r[i_r, j_r]

    lr_mov_x = lr_x + flowx
    lr_mov_y = lr_y + flowy

    # updating inbound condition
    if not (0 <= lr_mov_x < lr_w and
            0 <= lr_mov_y < lr_h):
        return
    
    # computing kernel
    if not iso_kernel:
        if bayer_mode :
            kmap_j = lr_mov_x/2 - 0.5 # grey grid is offseted and twice more sparse
            kmap_i = lr_mov_y/2 - 0.5
        else:
            kmap_j = lr_mov_x - 0.5 # grey grid is exactly the coarse grid
            kmap_i = lr_mov_y - 0.5

        ## clipping bilinear interpolation of the covariance matrix
        frac_x, _ = math.modf(kmap_j)
        frac_y, _ = math.modf(kmap_i)

        floor_x = max(int(kmap_j), 0)
        floor_y = max(int(kmap_i), 0)
        ceil_x = min(floor_x + 1, covs.shape[1]-1)
        ceil_y = min(floor_y + 1, covs.shape[0]-1)

        tr_cov_xx = covs[floor_y, floor_x, 0, 0]
        tr_cov_xy = covs[floor_y, floor_x, 0, 1]
        tr_cov_yy = covs[floor_y, floor_x, 1, 1]
        tl_cov_xx = covs[floor_y, ceil_x, 0, 0]
        tl_cov_xy = covs[floor_y, ceil_x, 0, 1]
        tl_cov_yy = covs[floor_y, ceil_x, 1, 1]
        br_cov_xx = covs[ceil_y, floor_x, 0, 0]
        br_cov_xy = covs[ceil_y, floor_x, 0, 1]
        br_cov_yy = covs[ceil_y, floor_x, 1, 1]
        bl_cov_xx = covs[ceil_y, ceil_x, 0, 0]
        bl_cov_xy = covs[ceil_y, ceil_x, 0, 1]
        bl_cov_yy = covs[ceil_y, ceil_x, 1, 1]

        lerp_top_xx = tr_cov_xx + frac_x * (tl_cov_xx - tr_cov_xx)
        lerp_top_xy = tr_cov_xy + frac_x * (tl_cov_xy - tr_cov_xy)
        lerp_top_yy = tr_cov_yy + frac_x * (tl_cov_yy - tr_cov_yy)
        lerp_bot_xx = br_cov_xx + frac_x * (bl_cov_xx - br_cov_xx)
        lerp_bot_xy = br_cov_xy + frac_x * (bl_cov_xy - br_cov_xy)
        lerp_bot_yy = br_cov_yy + frac_x * (bl_cov_yy - br_cov_yy)

        interp_cov_xx = lerp_top_xx + frac_y * (lerp_bot_xx - lerp_top_xx)
        interp_cov_xy = lerp_top_xy + frac_y * (lerp_bot_xy - lerp_top_xy)
        interp_cov_yy = lerp_top_yy + frac_y * (lerp_bot_yy - lerp_top_yy)
        # inverting
        det = interp_cov_xx * interp_cov_yy - interp_cov_xy * interp_cov_xy # Invertible by design
        inv_det = 1.0 / det

        cov_i_xx =  inv_det * interp_cov_yy
        cov_i_xy = -inv_det * interp_cov_xy
        cov_i_yy =  inv_det * interp_cov_xx

    center_j = int(lr_mov_x)
    center_i = int(lr_mov_y)
    lr_mov_j = lr_mov_x - 0.5
    lr_mov_i = lr_mov_y - 0.5
    for di in range(-1, 2):
        for dj in range(-1, 2):
    
            j = center_j + dj
            i = center_i + di

            if not (0 <= j < lr_w and
                    0 <= i < lr_h):
                continue

            channel = l_cfa[i%2, j%2] if bayer_mode else 0
            c = comp_img[i, j]
        
            # computing distance
            dist_x = j - lr_mov_j
            dist_y = i - lr_mov_i

            ### Computing w
            if iso_kernel: 
                z = 2 * (dist_x*dist_x + dist_y*dist_y)
            else:
                z = cov_i_xx * dist_x * dist_x + 2 * cov_i_xy * dist_x * dist_y + cov_i_yy * dist_y * dist_y
                # z can be slightly negative because of numerical precision.
                # I clamp it to not explode the error with exp
            z = max(0, z)

            w = math.exp(-0.5*z)
            ############
                
            val[channel] += w * local_r * c
            acc[channel] += w * local_r
        
    for chan in range(n_channels):
        num[hr_i, hr_j, chan] += val[chan] 
        den[hr_i, hr_j, chan] += acc[chan]