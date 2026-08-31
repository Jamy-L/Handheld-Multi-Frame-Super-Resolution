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
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from typing import Union

from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS
from .config import Config

    
def merge(comp_img: DeviceNDArray, alignments: DeviceNDArray, covs: DeviceNDArray, r: DeviceNDArray,
          num: DeviceNDArray, den: DeviceNDArray, config: Config):
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
        
    config : Config
        parameters.

    Returns
    -------
    None

    """
    scale = config.scale

    bayer_mode = config.mode == 'bayer'
    iso_kernel = config.merging.kernel == 'iso'
    tile_size = config.alignment.tile_size

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
        bayer_mode, iso_kernel, scale, tile_size,
        num, den)



@cuda.jit
def accumulate(comp_img, alignments, covs, r,
               bayer_mode, iso_kernel, scale, tile_size,
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

            if bayer_mode:
                # rggb harcoded. so i,j even -> 0; both odd -> 2, else 1
                channel = i%2 + j%2
            else:
                channel = 0

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
