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

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE,DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS, clamp

def init_robustness(ref_img, options, params):
    """
    Initialiazes the robustness etimation procesdure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Raw reference image J_1
    options : dict
        options.
    params : dict
        parameters.

    Returns
    -------
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        local statistics of the reference image. guide_imshape = imshape in grey mode,
        guide_imshape = imshape//2 in bayer mode.
        
        ref_local_stats[:, :, 0, c] is the mean value of color channel c mu(c)
        ref_local_stats[:, :, 1, c] is the variance (sigma^2) of color channel cs

    """
    imshape_y, imshape_x = ref_img.shape
    
    bayer_mode = params['mode']=='bayer'
    verbose_3 = options['verbose'] >= 3
    r_on = params['on']
    
    # TODO we may move the CFA to GPU as soon as it is read (in process() )
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    

    if r_on :         
        if verbose_3:
            print(" - Decimating images to RGB")
            current_time = time.perf_counter()

        # Computing guide image

        if bayer_mode:
            guide_ref_img = compute_guide_image(ref_img, CFA_pattern)
        else:
            # Numba friendly code to add 1 channel
            guide_ref_img = ref_img.reshape((imshape_y, imshape_x, 1)) 

        
        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Image decimated')
            
        ref_local_stats = compute_local_stats(guide_ref_img)
        
        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(current_time, ' - Local stats estimated')
            
        return ref_local_stats
    else:
        return None
    
    
def compute_robustness(comp_img, ref_local_stats, flows, options, params):
    """
    this is the implementation of Algorithm 6: ComputeRobustness
    Returns the robustnesses of the compared image J_n (n>1), based on the
    provided flow V_n(p) and the local statistics of the reference frame.

    Parameters
    ----------
    comp_img : device Array[imsize_y, imsize_x]
        Compared raw image J_n (n>1).
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Local stats of the reference image
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image V_n(p)
    options : dict
        options
    params : dict
        parameters

    Returns
    -------
    r : device Array[guide_imshape_y, guide_imshape_x]
        Locally minimized Robustness map, sampled at the center of
        every bayer quad
    """
    
    imshape_y, imshape_x = comp_img.shape

    bayer_mode = params['mode']=='bayer'
    current_time, verbose_3 = time.perf_counter(), options['verbose'] >= 3
    r_on = params['on']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    
    tile_size = params['tuning']["tileSize"]
    t = params['tuning']["t"]
    s1 = params['tuning']["s1"]
    s2 = params['tuning']["s2"]
    Mt = params['tuning']["Mt"]
    
    n_patch_y, n_patch_x, _ = flows.shape
    
    if bayer_mode:
        guide_imshape = imshape_y//2, imshape_x//2
    else:
        guide_imshape = imshape_y, imshape_x
          
    if r_on : 
        r = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
        # moving noise model to GPU
        cuda_std_curve = cuda.to_device(params['std_curve'])
        cuda_diff_curve = cuda.to_device(params['diff_curve'])
            
        if verbose_3:
            cuda.synchronize()
            current_time = getTime(current_time, ' - Moved noise model to GPU')

            
        # Computing guide image
        if bayer_mode:
            guide_img = compute_guide_image(comp_img, CFA_pattern)
            comp_local_stats = cuda.device_array(guide_imshape+(2, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
        else:
            guide_img = comp_img.reshape((imshape_y, imshape_x, 1)) # Adding 1 channel
            comp_local_stats = cuda.device_array(guide_imshape + (2, 1), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
            

        # Computing local stats (before applying optical flow)
        # 2 channels for mu, sigma
        
        if verbose_3:
            cuda.synchronize()
            current_time = getTime(current_time, ' - Image decimated to rgb')
            
        comp_local_stats = compute_local_stats(guide_img)
        
        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Local stats estimated')
        
        # computing d
        d_p = compute_patch_dist(ref_local_stats, comp_local_stats,
                                 flows, tile_size)
        
        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Estimated color distances')
        
        # leveraging the noise model
        d_sq, sigma_sq = apply_noise_model(d_p, ref_local_stats,
                                           cuda_std_curve, cuda_diff_curve)
        
        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Applied noise model')
        
        # applying flow discontinuity penalty
        S = cuda.device_array((n_patch_y, n_patch_x), DEFAULT_NUMPY_FLOAT_TYPE)
        S = compute_s(flows, Mt, s1, s2)
        
        
        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Flow irregularities registered')
        
        R = robustness_threshold(d_sq, sigma_sq, S, t, tile_size, bayer_mode)
        
        if verbose_3:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Robustness Estimated')

        r = local_min(R)
        
        if verbose_3:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Robustness locally minimized')
    else: 
        # TODO maybe it would be faster to initalize r on gpu
        # and write a cuda kernel to fill it with 1. The algorithm
        # is meant to run with r_on anyways
        temp = np.ones(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        r = cuda.to_device(temp)
    return r

def compute_guide_image(raw_img, CFA):
    """
    This is the implementation of Algorithm 7: ComputeGuideImage
    Return the guide image G associated with the raw frame J

    Parameters
    ----------
    raw_img : device Array[imshape_y, imshape_x]
        Raw frame J_n.
    CFA : device Array[2, 2]
        Bayer pattern

    Returns
    -------
    guide_img : device Array[imshape_y//2, imshape_x//2, 3]
        guide image.

    """
    imshape_y, imshape_x = raw_img.shape
    guide_imshape_y, guide_imshape_x = imshape_y//2, imshape_x//2
    guide_img = cuda.device_array((guide_imshape_y, guide_imshape_x, 3), DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape_x/threadsperblock[1])
    blockspergrid_y = math.ceil(guide_imshape_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
            
    cuda_compute_guide_image[blockspergrid, threadsperblock](raw_img, guide_img, CFA)
    
    return guide_img
    
@cuda.jit
def cuda_compute_guide_image(raw_img, guide_img, CFA):
    tx, ty = cuda.grid(2)
    
    if not (0 <= ty < guide_img.shape[0] and
            0 <= tx < guide_img.shape[1]):
        return
        
    g = 0
    
    for i in range(2):
        for j in range(2):
            c = uint8(CFA[i, j])
            
            if c == 1: # green
                g +=  raw_img[2*ty + i, 2*tx + j]
            else:
                guide_img[ty, tx, c] = raw_img[2*ty + i, 2*tx + j]
            
    guide_img[ty, tx, 1] = g/2

def compute_local_stats(guide_img):
    """
    Implementation of Algorithm 8: ComputeLocalStatistics
    Computes the mean color and variance associated for each 3 by 3 patches of
    the guide image G_n.

    Parameters
    ----------
    guide_img : device Array[guide_imshape_y, guide_imshape_x, channels]
        Guide image G_n. 
        
    Returns
    -------
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Array that contains mu and sigma² for every position of the guide image.


    """
    *guide_imshape, n_channels = guide_img.shape
    if n_channels == 1:
        local_stats = cuda.device_array(guide_imshape + [2, 1], DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
    elif n_channels == 3:
        local_stats = cuda.device_array(guide_imshape + [2, 3], DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
    else: 
        raise ValueError("Incoherent number of channel : {}".format(n_channels))
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(guide_imshape[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y, n_channels)
    
    cuda_compute_local_stats[blockspergrid, threadsperblock](guide_img, local_stats)
    
    return local_stats
    
    
@cuda.jit
def cuda_compute_local_stats(guide_img, local_stats):
    guide_imshape_y, guide_imshape_x, _ = guide_img.shape
    
    idx, idy, channel = cuda.grid(3)
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

            value = guide_img[y, x, channel]
            local_stats_[0] += value
            local_stats_[1] += value*value


    # normalizing
    channel_mean = local_stats_[0]/9
    local_stats[idy, idx, 0, channel] = channel_mean
    local_stats[idy, idx, 1, channel] = local_stats_[1]/9 - channel_mean*channel_mean
        
        
def compute_patch_dist(ref_local_stats, comp_local_stats, flows, tile_size):
    """
    Computes the map of d based on both maps of color mean

    Parameters
    ----------
    ref_local_stats : Device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        mu, sigma² map for guide ref image G_1
    comp_local_stats : Device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        mu, sigma map for guide compared image G_n (n>1)
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image V_n
    tile_size : int
        tile size used for optical flow (T)

    Returns
    -------
    d_p : Device Array[guide_imshape_y, guide_imshape_x, channels]
        Array that contains the distances 

    """
    *guide_imshape, _, n_channels = ref_local_stats.shape

    d_p = cuda.device_array(guide_imshape + [n_channels], DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(guide_imshape[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y, n_channels)
    
    cuda_compute_patch_dist[blockspergrid, threadsperblock](
        ref_local_stats, comp_local_stats, flows, tile_size, d_p)

    return d_p

@cuda.jit
def cuda_compute_patch_dist(ref_local_stats, comp_local_stats, flow, tile_size, dist):
    idx, idy, channel = cuda.grid(3)
    guide_imshape_y, guide_imshape_x, _, n_channels = ref_local_stats.shape
    
    if not (0 <= idy < guide_imshape_y and
            0 <= idx < guide_imshape_x):
        return

    
    ## Fetching flow
    local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) 
    if n_channels == 1:
        patch_idy = int(idy//tile_size) # guide scale is actually coarse scale
        patch_idx = int(idx//tile_size)
        # guide image is coarse image : the flow stays the same
        local_flow[0] = flow[patch_idy, patch_idx, 0]
        local_flow[1] = flow[patch_idy, patch_idx, 1]
        
    else:
        patch_idy = int((2*idy + 0.5)//tile_size) # guide scale is 2 times sparser than coarse
        patch_idx = int((2*idx + 0.5)//tile_size)
        
        # guide image is 2x smaller than coarse image : the flow must be divided by 2
        local_flow[0] = flow[patch_idy, patch_idx, 0]/2
        local_flow[1] = flow[patch_idy, patch_idx, 1]/2
    
    new_idx = round(idx + local_flow[0])
    new_idy = round(idy + local_flow[1])

    
    inbound = (0 <= new_idx < guide_imshape_x and
               0 <= new_idy < guide_imshape_y)
    
    if inbound :
        dif = abs(ref_local_stats[idy, idx, 0, channel] - comp_local_stats[new_idy, new_idx, 0, channel])
        dist[idy, idx, channel] = dif
        
    else:
        dist[idy, idx, channel] = +1/0 # + infinite distance will induce R = 0

def apply_noise_model(d_p, ref_local_stats, std_curve, diff_curve):
    """
    Applying noise model to update d^2 and sigma^2

    Parameters
    ----------
    d_p : device Array[guide_imshape_y, guide_imshape_x, n_channels]
        Color distance between ref and compared image for each channel
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Local statistics of the ref image (required for fetching sigmas)
    std_curve : device Array
        Noise model for sigma
    diff_curve : device Array
        Moise model for d

    Returns
    -------
    d_sq : device Array[guide_imshape_y, guide_imshape_x]
        updated version of the squarred distance
    sigma_sq : device Array[guide_imshape_y, guide_imshape_x]
        Array that will contained the noise-corrected sigma² value

    """
    *guide_imshape, _, n_channels = ref_local_stats.shape     
    sigma_sq = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
    d_sq = cuda.device_array(sigma_sq.shape, DEFAULT_NUMPY_FLOAT_TYPE)
        
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(guide_imshape[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_apply_noise_model[blockspergrid, threadsperblock](d_p, ref_local_stats,
                                                           std_curve, diff_curve,
                                                           d_sq, sigma_sq)
    return d_sq, sigma_sq

@cuda.jit
def cuda_apply_noise_model(d_p, ref_local_stats,
                           std_curve, diff_curve,
                           d_sq, sigma_sq):
    idx, idy = cuda.grid(2)
    
    if not(0 <= idy < ref_local_stats.shape[0] and
           0 <= idx < ref_local_stats.shape[1]):
        return
    n_channels = ref_local_stats.shape[-1]
    
    d_sq_ = 0
    sigma_sq_ = 0
    for channel in range(n_channels):
        brightness = ref_local_stats[idy, idx, 0, channel]
        id_noise = round(1000 *brightness) # id on the noise curve
        d_t =  diff_curve[id_noise]
        sigma_t = std_curve[id_noise]
        
        sigma_p_sq = ref_local_stats[idy, idx, 1, channel]
        sigma_sq_ += max(sigma_p_sq, sigma_t*sigma_t)
        
        d_p_ = d_p[idy, idx, channel]
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
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
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
    guide_imshape = d_sq.shape 
    R = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
    
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
        
    if bayer_mode : 
        patch_idy = int((2*idy+0.5)//tile_size)
        patch_idx = int((2*idx+0.5)//tile_size)
    else:
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
        