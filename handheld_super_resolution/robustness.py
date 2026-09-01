# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

This script contains : 
    - The implementation of Algorithm 6: ComputeRobustness
    - The implementation of Algorithm 7: ComputeGuideImage
    - The implementation of Algorithm 8: ComputeLocalStatistics
    - The implementation of Algorithm 9: ComputeLocalMin


@author: jamyl
"""
import time
import math

import numpy as np
from numba import cuda, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import torch
from typing import Optional, Tuple

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE,DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS, clamp, timer
from .utils_image import dogson_biquadratic_kernel, dogson_quadratic_kernel
from .config import Config
from .debug_writer import DebugWriter


def init_robustness(ref_img: DeviceNDArray, config: Config):
    """
    Initialiazes the robustness etimation procedure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Raw reference image J_1
    config : Config
        parameters. 

    Returns
    -------
    local_means : device Array[imshape_y, imshape_x, channels]
        local means of the reference image.
        
    local_stds : device Array[imshape_y, imshape_x, channels]
        local standard deviations of the reference image.

    """
    verbose_3 = config.verbose >= 3
    
    compute_guide_image_ = timer(compute_guide_image, verbose_3, " - Decimating images to RGB", ' - Image decimated')
    compute_local_stats_ = timer(compute_local_stats, verbose_3, end_s=' - Local stats estimated')
    warp_stats_ = timer(warp_stats, verbose_3, ' - Local stats warped upscaled')
    
    imshape_y, imshape_x = ref_img.shape

    bayer_mode = (config.mode == 'bayer')

    # Computing guide image
    if bayer_mode:
        guide_ref_img = compute_guide_image_(ref_img)
    else:
        # Numba friendly code to add 1 channel
        guide_ref_img = ref_img.reshape((1, imshape_y, imshape_x)) 

    local_means, local_stds = compute_local_stats_(guide_ref_img)
    
    return local_means, local_stds
    
    
def compute_robustness(comp_img: DeviceNDArray, ref_local_means: DeviceNDArray, ref_local_var: DeviceNDArray,
                       flows: DeviceNDArray,
                       noise_model: Tuple[DeviceNDArray, DeviceNDArray], config: Config,
                       debug_writer: Optional[DebugWriter] = None) -> DeviceNDArray:
    """
    this is the implementation of Algorithm 6: ComputeRobustness
    Returns the robustnesses of the compared image J_n (n>1), based on the
    provided flow V_n(p) and the local statistics of the reference frame.

    Parameters
    ----------
    comp_img : device Array[imsize_y, imsize_x]
        Compared raw image J_n (n>1).
    ref_local_means : device Array[imsize_y, imsize_x, c]
        Local means of the reference image
    ref_local_stds : device Array[imsize_y, imsize_x, c]
        Local standard deviations of the reference image
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image V_n(p)
    config : Config
        parameters.
    debug_writer : DebugWriter, optional
        Streams intermediate guide images to disk when provided.

    Returns
    -------
    r : device Array[imsize_y, imsize_x]
        Locally minimized Robustness map, sampled at the center of
        every bayer quad
    """
    current_time, verbose_3 = time.perf_counter(), config.verbose >= 3
    
    compute_guide_image_ = timer(compute_guide_image, verbose_3, " - Decimating images to RGB", ' - Image decimated')
    compute_local_stats_ = timer(compute_local_stats, verbose_3, end_s=' - Local stats estimated')
    warp_stats_ = timer(warp_stats, verbose_3, end_s=' - Local stats warped and upscaled')
    compute_d_sigma_ = timer(compute_d_sigma, verbose_3, end_s=' - Estimated color distances')
    compute_s_ = timer(compute_s, verbose_3, end_s=' - Flow irregularities registered')
    robustness_threshold_ = timer(robustness_threshold, verbose_3, end_s=' - Robustness Estimated')
    local_min_ = timer(local_min, verbose_3, end_s=' - Robustness locally minimized')
    
    imshape_y, imshape_x = comp_img.shape

    bayer_mode = (config.mode == 'bayer')

    tile_size = config.alignment.tile_size
    assert isinstance(tile_size, int), f"Got invalide tile size {tile_size}"
    t = config.robustness.t
    s1 = config.robustness.s1
    s2 = config.robustness.s2
    Mt = config.robustness.Mt
          
    cuda_std_curve, cuda_diff_curve = noise_model
        
    # Computing guide image
    if bayer_mode:
        guide_img = compute_guide_image_(comp_img)
    else:
        guide_img = comp_img.reshape((1, imshape_y, imshape_x)) # Adding 1 channel
        

    # Computing local stats (before applying optical flow)
    comp_local_means, _ = compute_local_stats_(guide_img)

    if debug_writer is not None:
        frame = np.moveaxis(comp_local_means.copy_to_host(), 0, -1)
        debug_writer.write_rgb("rgb_guides", frame)
    
    # Upscale and warp local means
    comp_local_means = warp_stats_(comp_local_means, tile_size, flows)

    if debug_writer is not None:
        frame = np.moveaxis(comp_local_means.copy_to_host(), 0, -1)
        debug_writer.write_rgb("rgb_guides_aligned", frame)
    
    # computing d_sq and sigma_sq (noise correction on the fly)
    d_sq, sigma_sq = compute_d_sigma_(ref_local_means, comp_local_means,
                                      ref_local_var, cuda_std_curve, cuda_diff_curve,
                                      config.robustness.noise_correction)

    # applying flow discontinuity penalty
    S = compute_s_(flows, Mt, s1, s2)
    R = robustness_threshold_(d_sq, sigma_sq, S, t, tile_size, bayer_mode)
    r = local_min_(R)
    return r


def compute_guide_image(raw_img: DeviceNDArray):
    """
    This is the implementation of Algorithm 7: ComputeGuideImage
    Return the guide image G associated with the raw frame J

    Parameters
    ----------
    raw_img : device Array[imshape_y, imshape_x]
        Raw frame J_n.

    Returns
    -------
    guide_img : device Array[3, imshape_y//2, imshape_x//2]
        guide image.

    """
    imshape_y, imshape_x = raw_img.shape
    guide_imshape_y, guide_imshape_x = imshape_y//2, imshape_x//2
    guide_img = cuda.device_array((3, guide_imshape_y, guide_imshape_x), DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(guide_imshape_x/threadsperblock[1])
    blockspergrid_y = math.ceil(guide_imshape_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
            
    cuda_compute_guide_image[blockspergrid, threadsperblock](raw_img, guide_img)
    
    return guide_img
    
@cuda.jit
def cuda_compute_guide_image(raw_img, guide_img):
    tx, ty = cuda.grid(2)
    _, h, w = guide_img.shape
    
    if not (0 <= ty < h and
            0 <= tx < w):
        return

    guide_img[0, ty, tx] = math.sqrt(max(raw_img[2*ty, 2*tx], 0))
    guide_img[1, ty, tx] = math.sqrt(max(0.5*(raw_img[2*ty, 2*tx+1] + raw_img[2*ty+1, 2*tx]), 0))
    guide_img[2, ty, tx] = math.sqrt(max(raw_img[2*ty+1, 2*tx+1], 0))

def compute_local_stats(guide_img: DeviceNDArray):
    """
    Implementation of Algorithm 8: ComputeLocalStatistics
    Computes the mean color and variance associated for each 3 by 3 patches of
    the guide image G_n.

    Parameters
    ----------
    guide_img : device Array[channels, guide_imshape_y, guide_imshape_x]
        Guide image G_n. 
        
    Returns
    -------
    ref_local_means : device Array[guide_imshape_y, guide_imshape_x, channels]
        Array that contains the local mean for every position of the guide image.
    ref_local_stds : device Array[guide_imshape_y, guide_imshape_x, channels]
        Array that contains the local variance sigma² for every position of the guide image.


    """
    n_channels, *guide_imshape = guide_img.shape
    if n_channels == 1:
        mean = cuda.device_array((1, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
        var = cuda.device_array((1, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
    elif n_channels == 3:
        mean = cuda.device_array((3, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
        var = cuda.device_array((3, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
    else: 
        raise ValueError("Incoherent number of channel : {}".format(n_channels))
    
    threadsperblock = (1, DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape[1]/threadsperblock[2])
    blockspergrid_y = math.ceil(guide_imshape[0]/threadsperblock[1])
    blockspergrid = (n_channels, blockspergrid_x, blockspergrid_y)
    
    cuda_compute_local_stats[blockspergrid, threadsperblock](guide_img, mean, var)
    
    return mean, var
    
    
@cuda.jit
def cuda_compute_local_stats(guide_img, mean, var):
    _, guide_imshape_y, guide_imshape_x = guide_img.shape
    
    channel, idx, idy = cuda.grid(3)
    if not(0 <= idy < guide_imshape_y and
           0 <= idx < guide_imshape_x):
        return

    mean_ = 0
    var_ = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            y = clamp(idy + i, 0, guide_imshape_y-1)
            x = clamp(idx + j, 0, guide_imshape_x-1)

            color = guide_img[channel, y, x]
            mean_ += color
            var_ += color * color

    # normalizing
    mean_ /= 9
    mean[channel, idy, idx] = mean_
    var[channel, idy, idx] = var_ / 9 - mean_ * mean_

def warp_stats(local_stats: DeviceNDArray, tile_size: int, flow: DeviceNDArray):
    """
    Upscales and warps a map of local statistics using Dogson's biquadratic approximation 

    Parameters
    ----------
    local_stats : device array [guide_imshape_y, guide_imshape_x, n_c]
        A map of ONE local stat (can have 1 or 3 channels)
    tile_size : Integer
        If required, flow tile size.
    flow : Device Array [ty, tx, 2], optional
        If required, the optical flow. The default is None.

    Returns
    -------
    upscaled_stats : Device Array[raw_imshape_y, raw_imshape_y, c]
        Upscaled and warped local stats

    """
    n_channels, *guide_imshape = local_stats.shape
    bayer_mode = (n_channels == 3)
    
    warped_stats = cuda.device_array((n_channels, 
                                        guide_imshape[0],
                                        guide_imshape[1]),
                                        DEFAULT_NUMPY_FLOAT_TYPE)

    _, ny, nx = warped_stats.shape
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(nx/threadsperblock[1])
    blockspergrid_y = math.ceil(ny/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_warp_dogson[blockspergrid, threadsperblock](local_stats,
                                                        flow, tile_size,
                                                        warped_stats)
    return warped_stats
    
    
@cuda.jit
def cuda_warp_dogson(source, flow, tile_size, warped):
    tile_size = tile_size // 2
    n_channels, ny, nx = source.shape
    
    x, y = cuda.grid(2)
    
    if not (0 <= y < ny and
            0 <= x < nx):
        return
    
    # Flow is defined on the raw image basis
    patch_idy = int(y//tile_size)
    patch_idx = int(x//tile_size)
    
    flow_x = flow[patch_idy, patch_idx, 0]
    flow_y = flow[patch_idy, patch_idx, 1]
        
        
    # Jumping from ref guide to mov guide  
    y_mov = y + flow_y * 0.5
    x_mov = x + flow_x * 0.5
    
    # Out of bounds
    if not (0 <= y_mov < ny and
            0 <= x_mov < nx):
        for c in range(n_channels):
            warped[c, y, x] = 1/0 # infinity will imply R = 0
        return
    
    center_y = round(y_mov)
    center_x = round(x_mov)
    
    # init buffer
    w_acc = 0
    buffer = cuda.local.array(3, DEFAULT_CUDA_FLOAT_TYPE)
    for c in range(n_channels):
        buffer[c] = 0
    
    for i in range(-1, 2):
        y_ = int(clamp(center_y + i, 0, ny-1))
        dy = y_ - y_mov
        wy = dogson_quadratic_kernel(dy)
        for j in range(-1, 2):
            x_ = int(clamp(center_x + j, 0, nx-1))
            dx = x_ - x_mov

            w = wy * dogson_quadratic_kernel(dx)

            for c in range(n_channels):
                buffer[c] += source[c, y_, x_] * w
            w_acc += w
    
    # Normalise and write output
    for c in range(n_channels):
        warped[c, y, x] = buffer[c]/w_acc
            

def compute_d_sigma(means_r: DeviceNDArray, means_m: DeviceNDArray, var_m: DeviceNDArray, std_curve: DeviceNDArray, diff_curve: DeviceNDArray, do_noise_correction: bool):
    """
    Computes the color distance between the two frames. They must be warped.

    Parameters
    ----------
    means_1 : device array [ny, nx, c]
        local mean of frame 1.
    means_2 : device array [ny, nx, c]
        local mean of frame 1.

    Returns
    -------
    diff : device array [ny, nx, c]
        channel wise absolute difference

    """
    assert means_r.shape == means_m.shape
    nc, ny, nx = shape = means_r.shape
    d_sq = cuda.device_array((ny, nx), DEFAULT_NUMPY_FLOAT_TYPE)
    sigma_sq = cuda.device_array((ny, nx), DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(nx/threadsperblock[1])
    blockspergrid_y = math.ceil(ny/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_compute_d_sigma[blockspergrid, threadsperblock](means_r, means_m, var_m, std_curve, diff_curve, d_sq, sigma_sq, do_noise_correction)
    
    return d_sq, sigma_sq

@cuda.jit
def cuda_compute_d_sigma(means_r, means_m, var_m, std_curve, diff_curve, d_sq, sigma_sq, do_noise_correction):
    x, y = cuda.grid(2)
    nc, ny, nx = means_r.shape
    
    if not (0 <= y < ny and
            0 <= x < nx):
        return

    d_sq_ = 0
    sigma_sq_ = 0
    for c in range(nc):
        error = means_r[c, y, x] - means_m[c, y, x]
        d_sq_ += error * error
        sigma_sq_ += var_m[c, y, x]


    if do_noise_correction:
        brightness = 0
        for c in range(nc):
            brightness += means_r[c, y, x]
        brightness /= nc
        brightness = clamp(brightness, 0, 1)
        id_noise = round(1000 * brightness) # id on the noise curve

        d_t =  diff_curve[id_noise]
        sigma_t = std_curve[id_noise]
        sigma_sq_ = max(sigma_sq_, sigma_t*sigma_t)

        shrink = d_sq_/(d_sq_ + d_t*d_t)
        d_sq_ *= shrink * shrink

    d_sq[y, x] = d_sq_
    sigma_sq[y, x] = sigma_sq_

                     
def compute_s(flows: DeviceNDArray, M_th: float, s1: float, s2: float):
    """ Computes s at every position based on flow irregularities
    

    Parameters
    ----------
    flows : device Array[n_tiles_y, n_tiles_x, 2]
        Patch wise optical flow
    M_th : float
        Threshold for M.
    s1 : float
        DESCRIPTION.
    s2 : float
        DESCRIPTION.

    Returns
    -------
    S : device Array[n_patchs_y, n_patchs_x]
        Map where s1 or s2 will be written at each position.

    """
    n_patch_y, n_patch_x, _ = flows.shape
    S = cuda.device_array((n_patch_y, n_patch_x), DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(n_patch_x/threadsperblock[1])
    blockspergrid_y = math.ceil(n_patch_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_compute_s[blockspergrid, threadsperblock](flows, M_th, s1, s2, S)
    
    return S
    
@cuda.jit
def cuda_compute_s(flows, M_th, s1, s2, S):
    patch_idx, patch_idy = cuda.grid(2)
    
    n_patch_y, n_patch_x, _ = flows.shape
    
    if not (0 <= patch_idy < n_patch_y and
            0 <= patch_idx < n_patch_x):
        return
    
    mini = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    maxi = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    mini[0] = +1/0
    mini[1] = +1/0
    maxi[0] = -1/0
    maxi[1] = -1/0
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            y = patch_idy + i
            x = patch_idx + j
    
            inbound = (0 <= x < n_patch_x and
                       0 <= y < n_patch_y)

            if inbound:
                flow[0] = flows[y, x, 0]
                flow[1] = flows[y, x, 1]
                
                #local max search
                maxi[0] = max(maxi[0], flow[0])
                maxi[1] = max(maxi[1], flow[1])
                #local min search
                mini[0] = min(mini[0], flow[0])
                mini[1] = min(mini[1], flow[1])
        
    diff_0 = maxi[0] - mini[0]
    diff_1 = maxi[1] - mini[1]
    if diff_0*diff_0 + diff_1*diff_1 > M_th*M_th:
        S[patch_idy, patch_idx] = s1
    else:
        S[patch_idy, patch_idx] = s2

def robustness_threshold(d_sq: DeviceNDArray, sigma_sq: DeviceNDArray, S: DeviceNDArray, t: float, tile_size: int, bayer_mode: bool):
    imshape = d_sq.shape 
    R = cuda.device_array(imshape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(R.shape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(R.shape[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_robustness_threshold[blockspergrid, threadsperblock](d_sq, sigma_sq, S, t, tile_size, bayer_mode, R)
    
    return R
    
@cuda.jit    
def cuda_robustness_threshold(d_sq, sigma_sq, S, t, tile_size, bayer_mode, R):
    idx, idy = cuda.grid(2)
    tile_size = tile_size//2

    if not (0 <= idy < R.shape[0] and
            0 <= idx < R.shape[1]):
        return
        
    patch_idy = int(idy//tile_size)
    patch_idx = int(idx//tile_size)
        
        
    R[idy, idx] = clamp(S[patch_idy, patch_idx] * math.exp(-d_sq[idy, idx]/sigma_sq[idy, idx]) - t,
                        0, 1)

def local_min(R: DeviceNDArray):
    """
    Implementation of Algorithm 9: ComputeLocalMin
    For each pixel of R, the minimum in a 5 by 5 window is estimated
    and stored in r.

    Parameters
    ----------
    R : Array[guide_imshape_y, guide_imshape_x]
        Robustness map for every image

    Returns
    -------
    r : Array[guide_imshape_y, guide_imshape_x]
        locally minimised version of R

    """
    r = cuda.device_array(R.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(R.shape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(R.shape[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_compute_local_min[blockspergrid, threadsperblock](R, r)
    
    return r
    
@cuda.jit
def cuda_compute_local_min(R, r):
    guide_imshape_y, guide_imshape_x = R.shape
    
    idx, idy = cuda.grid(2)
    if not(0 <= idy < guide_imshape_y and
           0 <= idx < guide_imshape_x):
        return

    mini = +1/0
    
    #local min search
    for i in range(-2, 3):
        y = clamp(idy + i, 0, guide_imshape_y-1)
        for j in range(-2, 3):
            x = clamp(idx + j, 0, guide_imshape_x-1)
            mini = min(mini, R[y, x])
    
    r[idy, idx] = mini
