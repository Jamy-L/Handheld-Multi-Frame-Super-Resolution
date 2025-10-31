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

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE,DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS, clamp, timer
from .utils_image import dogson_biquadratic_kernel, dogson_quadratic_kernel

def init_robustness(ref_img, cfa_pattern, white_balance, config):
    """
    Initialiazes the robustness etimation procedure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Raw reference image J_1
    cfa_pattern : device Array[2,2]
        Bayer pattern
    white_balance : device Array[3]
        White balance gains
    config : OmegaConf object
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
    upscale_warp_stats_ = timer(upscale_warp_stats, verbose_3, ' - Local stats warped upscaled')
    
    imshape_y, imshape_x = ref_img.shape

    bayer_mode = config.mode=='bayer'
    r_on = config.robustness.enabled

    if r_on :         
        # Computing guide image

        if bayer_mode:
            guide_ref_img = compute_guide_image_(ref_img, cfa_pattern, white_balance)
        else:
            # Numba friendly code to add 1 channel
            guide_ref_img = ref_img.reshape((1, imshape_y, imshape_x)) 
            
        local_means, local_stds = compute_local_stats_(guide_ref_img)
        
        # Upscale stats to raw coarse scale
        local_means = upscale_warp_stats(local_means)
        local_stds = upscale_warp_stats_(local_stds)
        
        return local_means, local_stds
    else:
        return None, None
    
    
def compute_robustness(comp_img, ref_local_means, ref_local_stds, flows, cfa_pattern, white_balance, noise_model, config):
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
    cfa_pattern : device Array[2,2]
        Bayer pattern
    white_balance : device Array[3]
        White balance gains
    config : OmegaConf object
        parameters.

    Returns
    -------
    r : device Array[imsize_y, imsize_x]
        Locally minimized Robustness map, sampled at the center of
        every bayer quad
    """
    current_time, verbose_3 = time.perf_counter(), config.verbose >= 3
    
    compute_guide_image_ = timer(compute_guide_image, verbose_3, " - Decimating images to RGB", ' - Image decimated')
    compute_local_stats_ = timer(compute_local_stats, verbose_3, end_s=' - Local stats estimated')
    upscale_warp_stats_ = timer(upscale_warp_stats, verbose_3, end_s=' - Local stats warped and upscaled')
    compute_dist_ = timer(compute_dist, verbose_3, end_s=' - Estimated color distances')
    apply_noise_model_ = timer(apply_noise_model, verbose_3, end_s=' - Applied noise model')
    compute_s_ = timer(compute_s, verbose_3, end_s=' - Flow irregularities registered')
    robustness_threshold_ = timer(robustness_threshold, verbose_3, end_s=' - Robustness Estimated')
    local_min_ = timer(local_min, verbose_3, end_s=' - Robustness locally minimized')
    
    imshape_y, imshape_x = comp_img.shape

    bayer_mode = config.mode=='bayer'
    r_on = config.robustness.enabled

    tile_size = config.block_matching.tuning.tile_size
    t = config.robustness.tuning.t
    s1 = config.robustness.tuning.s1
    s2 = config.robustness.tuning.s2
    Mt = config.robustness.tuning.Mt

    n_patch_y, n_patch_x, _ = flows.shape
    
    if bayer_mode:
        guide_imshape = imshape_y//2, imshape_x//2
    else:
        guide_imshape = imshape_y, imshape_x
          
    if r_on : 
        r = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
        cuda_std_curve, cuda_diff_curve = noise_model
            
        # Computing guide image
        if bayer_mode:
            guide_img = compute_guide_image_(comp_img, cfa_pattern, white_balance)
        else:
            guide_img = comp_img.reshape((imshape_y, imshape_x, 1)) # Adding 1 channel
            

        # Computing local stats (before applying optical flow)
        comp_local_means, _ = compute_local_stats_(guide_img)
        
        # Upscale and warp local means
        comp_local_means = upscale_warp_stats_(comp_local_means, 
                                               tile_size, flows)
        
        # computing d
        d_p = compute_dist_(ref_local_means, comp_local_means)
        
        
        # leveraging the noise model
        d_sq, sigma_sq = apply_noise_model_(d_p, ref_local_means, ref_local_stds,
                                            cuda_std_curve, cuda_diff_curve)
        
        # applying flow discontinuity penalty
        S = compute_s_(flows, Mt, s1, s2)
        R = robustness_threshold_(d_sq, sigma_sq, S, t, tile_size, bayer_mode)
        r = local_min_(R)
        
    else: 
        temp = np.ones(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        r = cuda.to_device(temp)
    return r


def compute_guide_image(raw_img, cfa_pattern, white_balance):
    """
    This is the implementation of Algorithm 7: ComputeGuideImage
    Return the guide image G associated with the raw frame J

    Parameters
    ----------
    raw_img : device Array[imshape_y, imshape_x]
        Raw frame J_n.
    cfa_pattern : device Array[2, 2]
        Bayer pattern
    white_balance : device Array[3]
        White balance gains

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
            
    cuda_compute_guide_image[blockspergrid, threadsperblock](raw_img, guide_img, cfa_pattern, white_balance)
    
    return guide_img
    
@cuda.jit
def cuda_compute_guide_image(raw_img, guide_img, CFA, wb):
    tx, ty = cuda.grid(2)
    _, h, w = guide_img.shape
    
    if not (0 <= ty < h and
            0 <= tx < w):
        return
        
    g = 0
    
    for i in range(2):
        for j in range(2):
            c = uint8(CFA[i, j])
            x = raw_img[2*ty + i, 2*tx + j] / wb[c] # Undo whitebalance
            
            if c == 1: # green
                g += x
            else:
                guide_img[c, ty, tx] = x
    guide_img[1, ty, tx] = g/2

def compute_local_stats(guide_img):
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
        local_means = cuda.device_array((1, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE) # mu
        local_stds = cuda.device_array((1, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE) # sigma
    elif n_channels == 3:
        local_means = cuda.device_array((3, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE) # mu for rgb
        local_stds = cuda.device_array((3, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE) # sigma for rgb
    else: 
        raise ValueError("Incoherent number of channel : {}".format(n_channels))
    
    threadsperblock = (1, DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape[1]/threadsperblock[2])
    blockspergrid_y = math.ceil(guide_imshape[0]/threadsperblock[1])
    blockspergrid = (n_channels, blockspergrid_x, blockspergrid_y)
    
    cuda_compute_local_stats[blockspergrid, threadsperblock](guide_img, local_means, local_stds)
    
    return local_means, local_stds
    
    
@cuda.jit
def cuda_compute_local_stats(guide_img, local_means, local_stds):
    _, guide_imshape_y, guide_imshape_x = guide_img.shape
    
    channel, idx, idy = cuda.grid(3)
    if not(0 <= idy < guide_imshape_y and
           0 <= idx < guide_imshape_x):
        return

    local_stats_ = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_stats_[0] = 0
    local_stats_[1] = 0

    for i in range(-1, 2):
        for j in range(-1, 2):
            y = clamp(idy + i, 0, guide_imshape_y-1)
            x = clamp(idx + j, 0, guide_imshape_x-1)

            value = guide_img[channel, y, x]
            local_stats_[0] += value
            local_stats_[1] += value*value


    # normalizing
    channel_mean = local_stats_[0]/9
    local_means[channel, idy, idx] = channel_mean
    local_stds[channel, idy, idx] = local_stats_[1]/9 - channel_mean*channel_mean

def upscale_warp_stats(local_stats, tile_size=None, flow=None):
    """
    Upscales and warps a map of local statistics using Dogson's biquadratic approximation 

    Parameters
    ----------
    local_stats : device array [guide_imshape_y, guide_imshape_x, n_c]
        A map of ONE local stat (can have 1 or 3 channels)
    tile_size : Integer, optional
        If required, flow tile size. The default is None.
    flow : Device Array [ty, tx, 2], optional
        If required, the optical flow. The default is None.

    Returns
    -------
    upscaled_stats : Device Array[raw_imshape_y, raw_imshape_y, c]
        Upscaled and warped local stats

    """
    n_channels, *guide_imshape = local_stats.shape
    bayer_mode = (n_channels == 3)
    
    if flow is None:
        flow = cuda.device_array((1, 1, 1), DEFAULT_NUMPY_FLOAT_TYPE) # just because is numba is picky on types and shapes
        is_ref = True
    else:
        is_ref= False
    
    if tile_size is None:
        tile_size = 0 # For numba's compiler
        
    
    if bayer_mode:
        upscaled_stats = cuda.device_array((n_channels,         
                                            guide_imshape[0]*2,
                                            guide_imshape[1]*2,
                                            ),
                                            DEFAULT_NUMPY_FLOAT_TYPE)

        upscale = 2
        
    else:
        upscaled_stats = cuda.device_array((n_channels, 
                                            guide_imshape[0],
                                            guide_imshape[1]),
                                           DEFAULT_NUMPY_FLOAT_TYPE)

        upscale = 1
    
    _, HR_ny, HR_nx = upscaled_stats.shape
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(HR_nx/threadsperblock[1])
    blockspergrid_y = math.ceil(HR_ny/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_uspcale_dogson[blockspergrid, threadsperblock](local_stats, upscale,
                                                        is_ref, flow, tile_size,
                                                        upscaled_stats)
    return upscaled_stats
    
    
@cuda.jit
def cuda_uspcale_dogson(LR, s, is_ref, flow, tile_size, HR):
    n_channels, LR_ny, LR_nx = LR.shape
    _, HR_ny, HR_nx = HR.shape
    
    x, y = cuda.grid(2)
    
    if not (0 <= y < HR_ny and
            0 <= x < HR_nx):
        return
    
    if is_ref:
        flow_x = 0
        flow_y = 0
    else:
        # Flow is defined on the raw image basis
        patch_idy = int(y//tile_size)
        patch_idx = int(x//tile_size)
        
        flow_x = flow[patch_idy, patch_idx, 0]
        flow_y = flow[patch_idy, patch_idx, 1]
        
        
    # Jumping from raw to guide    
    LR_y = (y + flow_y + 0.5)/s - 0.5
    LR_x = (x + flow_x + 0.5)/s - 0.5
    
    # Out of bounds
    if not (0 <= LR_y < LR_ny and
            0 <= LR_x < LR_nx):
        for c in range(n_channels):
            HR[c, y, x] = 1/0 # infinity will imply R = 0
        return
    
    center_y = round(LR_y)
    center_x = round(LR_x)
    
    # init buffer
    w_acc = 0
    buffer = cuda.local.array(3, DEFAULT_CUDA_FLOAT_TYPE)
    for c in range(n_channels):
        buffer[c] = 0
    
    for i in range(-1, 2):
        y_ = int(clamp(center_y + i, 0, LR_ny-1))
        dy = y_ - LR_y
        wy = dogson_quadratic_kernel(dy)
        for j in range(-1, 2):
            x_ = int(clamp(center_x + j, 0, LR_nx-1))
            dx = x_ - LR_x

            w = wy * dogson_quadratic_kernel(dx)

            for c in range(n_channels):
                buffer[c] += LR[c, y_, x_] * w
            w_acc += w
    
    # Normalise and write output
    for c in range(n_channels):
        HR[c, y, x] = buffer[c]/w_acc
            

def compute_dist(means_1, means_2):
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
    assert means_1.shape == means_2.shape
    nc, ny, nx = shape = means_1.shape
    diff = cuda.device_array(shape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (1, DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(nx/threadsperblock[2])
    blockspergrid_y = math.ceil(ny/threadsperblock[1])
    blockspergrid = (nc, blockspergrid_x, blockspergrid_y)
    
    cuda_compute_dist[blockspergrid, threadsperblock](means_1, means_2, diff)
    
    return diff
    

@cuda.jit
def cuda_compute_dist(means_1, means_2, diff):
    c, x, y = cuda.grid(3)
    nc, ny, nx = diff.shape
    
    if not (0 <= y < ny and
            0 <= x < nx and
            0 <= c < nc):
        return

    diff[c, y, x] = abs(means_1[c, y, x] - means_2[c, y, x])


def apply_noise_model(d_p, ref_local_means, ref_local_stds, std_curve, diff_curve):
    """
    Applying noise model to update d^2 and sigma^2

    Parameters
    ----------
    d_p : device Array[imshape_y, imshape_x, n_channels]
        Color distance between ref and compared image for each channel
    ref_local_means : device Array[imshape_y, imshape_x, channels]
        Local means of the ref image (required for fetching sigmas nois model)
    ref_local_stds : device Array[imshape_y, imshape_x, channels]
        Local variances (sigma²) of the ref image
    std_curve : device Array
        Noise model for sigma
    diff_curve : device Array
        Moise model for d

    Returns
    -------
    d_sq : device Array[imshape_y, imshape_x]
        updated version of the squarred distance
    sigma_sq : device Array[imshape_y, imshape_x]
        Array that will contained the noise-corrected sigma² value

    """
    _, *imshape = ref_local_means.shape     
    sigma_sq = cuda.device_array(imshape, DEFAULT_NUMPY_FLOAT_TYPE)
    d_sq = cuda.device_array(sigma_sq.shape, DEFAULT_NUMPY_FLOAT_TYPE)
        
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(imshape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(imshape[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_apply_noise_model[blockspergrid, threadsperblock](d_p, ref_local_means, ref_local_stds,
                                                           std_curve, diff_curve,
                                                           d_sq, sigma_sq)
    return d_sq, sigma_sq

@cuda.jit
def cuda_apply_noise_model(d_p, ref_local_means, ref_local_stds,
                           std_curve, diff_curve,
                           d_sq, sigma_sq):
    idx, idy = cuda.grid(2)
    nc, ny, nx = ref_local_means.shape
    
    if not(0 <= idy < ny and
           0 <= idx < nx):
        return
    
    d_sq_ = 0
    sigma_sq_ = 0
    for channel in range(nc):
        brightness = ref_local_means[channel, idy, idx]
        id_noise = round(1000 *brightness) # id on the noise curve
        d_t =  diff_curve[id_noise]
        sigma_t = std_curve[id_noise]

        sigma_p_sq = ref_local_stds[channel, idy, idx]
        sigma_sq_ += max(sigma_p_sq, sigma_t*sigma_t)
        
        d_p_ = d_p[channel, idy, idx]
        d_p_sq = d_p_ * d_p_
        shrink = d_p_sq/(d_p_sq + d_t*d_t)
        d_sq_ += d_p_sq * shrink * shrink
        
        
    sigma_sq[idy, idx] = sigma_sq_
    d_sq[idy, idx] = d_sq_    
    
                     
def compute_s(flows, M_th, s1, s2):
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

def robustness_threshold(d_sq, sigma_sq, S, t, tile_size, bayer_mode):
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

    if not (0 <= idy < R.shape[0] and
            0 <= idx < R.shape[1]):
        return
        
    patch_idy = int(idy//tile_size)
    patch_idx = int(idx//tile_size)
        
        
    R[idy, idx] = clamp(S[patch_idy, patch_idx] * math.exp(-d_sq[idy, idx]/sigma_sq[idy, idx]) - t,
                        0, 1)

def local_min(R):
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