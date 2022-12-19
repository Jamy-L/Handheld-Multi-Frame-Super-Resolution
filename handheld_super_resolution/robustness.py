# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

@author: jamyl
"""
import time
from math import exp

import numpy as np
from numba import cuda, uint8

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_THREADS, clamp

def init_robustness(ref_img,options, params):
    """
    Computes the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        raw reference image
    options : dict
        options.
    params : dict
        parameters.

    Returns
    -------
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        local statistics of the reference image. guide_imshape = imshape in grey mode,
        guide_imshape = imshape/2 in bayer mode.
        
        ref_local_stats[:, :, 0, c] is the mean value of color channel c
        ref_local_stats[:, :, 1, c] is the variance (sigma^2) of color channel c

    """
    imshape = imshape_y, imshape_x = ref_img.shape
    
    bayer_mode = params['mode']=='bayer'
    VERBOSE = options['verbose']
    r_on = params['on']
    
    # TODO we may move the CFA to GPU as soon as it is read (in process() )
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    

    if r_on :         
        if VERBOSE > 1:
            cuda.synchronize()
            current_time = time.perf_counter()
            print("Estimating ref image local stats")
            if VERBOSE > 2:
                print(" - Decimating images to RGB")
                
        if params["mode"]=='bayer':
            guide_imshape = int(imshape_y/2), int(imshape_x/2)
        else:
            guide_imshape = imshape

                
        # Computing guide image

        if bayer_mode:
            guide_ref_img = compute_guide_image(ref_img, guide_imshape, CFA_pattern)
        else:
            guide_ref_img = ref_img[:, :, None] # Adding 1 channel

    
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Image decimated')
            
        ref_local_stats = compute_local_stats(guide_ref_img)
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Local stats estimated')
        return ref_local_stats
    
    
def compute_robustness(comp_img, ref_local_stats, flows, options, params):
    """
    Returns the robustnesses of the compared image, based on the provided flow.

    Parameters
    ----------
    comp_img : device Array[imsize_y, imsize_x]
        Compared raw image.
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Local stats of the reference image
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image
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
    VERBOSE = options['verbose']
    r_on = params['on']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    
    tile_size = params['tuning']["tileSize"]
    t = params['tuning']["t"]
    s1 = params['tuning']["s1"]
    s2 = params['tuning']["s2"]
    Mt = params['tuning']["Mt"]
    
    n_patch_y, n_patch_x, _ = flows.shape
    
    if bayer_mode:
        guide_imshape = int(imshape_y/2), int(imshape_x/2)
    else:
        guide_imshape = imshape_y, imshape_x
          
    if r_on : 
        r = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
        if VERBOSE > 1:
            current_time = time.perf_counter()
            print("Estimating Robustness")
        
        # moving noise model to GPU
        cuda_std_curve = cuda.to_device(params['std_curve'])
        cuda_diff_curve = cuda.to_device(params['diff_curve'])
            
        if VERBOSE > 2:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Moved noise model to GPU')

            
        # Computing guide image
        if bayer_mode:
            guide_img = compute_guide_image(comp_img, guide_imshape, CFA_pattern)
            comp_local_stats = cuda.device_array(guide_imshape+(2, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
        else:
            guide_img = comp_img[:, :, None] # addign 1 channel
            comp_local_stats = cuda.device_array(guide_imshape + (2, 1), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
            

        # Computing local stats (before applying optical flow)
        # 2 channels for mu, sigma
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Image decimated to rgb')
            
        comp_local_stats = compute_local_stats(guide_img)
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Local stats estimated')
        
        # computing d
        d = compute_patch_dist(ref_local_stats, comp_local_stats,
                               flows, tile_size)
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Estimated color distances')
        
        # leveraging the noise model
        d, sigma = apply_noise_model(d, ref_local_stats,
                                     cuda_std_curve, cuda_diff_curve)
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Applied noise model')
        
        # applying flow discontinuity penalty
        S = cuda.device_array((n_patch_y, n_patch_x), DEFAULT_NUMPY_FLOAT_TYPE)
        compute_s[S.shape, (3, 3)](flows, Mt, s1, s2, S)
        
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Flow irregularities registered')
        
        R = robustness_treshold(d, sigma, S, t)
        
        if VERBOSE > 2:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Robustness Estimated')
        
        r = local_min(R)
        
        if VERBOSE > 2:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Robustness locally minimized')
    else: 
        temp = np.ones(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        r = cuda.to_device(temp)
    return r

def compute_guide_image(raw_img, guide_imshape, CFA):
    guide_img = cuda.device_array(guide_imshape + (3,), DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
            
    cuda_compute_guide_image[blockspergrid, threadsperblock](raw_img, guide_img, CFA)
    
    return guide_img
    
@cuda.jit
def cuda_compute_guide_image(raw_img, guide_img, CFA):
    tx, ty = cuda.grid(2)
    
    if (0 <= ty < guide_img.shape[0] and
        0 <= tx < guide_img.shape[1]):
        
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
    Computes the mean color and variance associated for each 3 by 3 patches

    Parameters
    ----------
    guide_img : device Array[guide_imshape_y, guide_imshape_x, channels]
        ref rgb image.
        
    Returns
    -------
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Array that contains mu and sigma² for the ref img


    """
    *guide_imshape, n_channels = guide_img.shape
    if n_channels == 1:
        local_stats = cuda.device_array(guide_imshape + [2, 1], DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
    elif n_channels == 3:
        local_stats = cuda.device_array(guide_imshape + [2, 3], DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
    else: 
        raise ValueError("Incoherent number of channel : {}".format(n_channels))
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1) # maximum, we may take less
    blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, n_channels)
    
    cuda_compute_local_stats[blockspergrid, threadsperblock](guide_img, local_stats)
    
    return local_stats
    
    
@cuda.jit
def cuda_compute_local_stats(guide_img, local_stats):
    guide_imshape_y, guide_imshape_x, n_channels = guide_img.shape
    
    idx, idy, channel = cuda.grid(3)
    if not(0 <= idy < guide_imshape_y and
           0 <= idx < guide_imshape_x):
        return

    local_stats_ = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_stats_[0] = 0; local_stats_[1] = 0

    for i in range(-3, 4):
        for j in range(-3, 4):
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
    Computes the map of d^2 based on both maps of color mean

    Parameters
    ----------
    ref_local_stats : Device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        mu, sigma map for guide ref image
    comp_local_stats : Device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        mu, sigma map for guide compared image
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image
    tile_size : int
        tile size used for optical flow

    Returns
    -------
    dist : Device Array[guide_imshape_y, guide_imshape_x, 2]
        Array that contains the distances

    """
    *guide_imshape, _, _ = ref_local_stats.shape

    dist = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_compute_patch_dist[blockspergrid, threadsperblock](
        ref_local_stats, comp_local_stats, flows, tile_size, dist)

    return dist

@cuda.jit
def cuda_compute_patch_dist(ref_local_stats, comp_local_stats, flow, tile_size, dist):
    idx, idy = cuda.grid(2)
    guide_imshape_y, guide_imshape_x, _, n_channels = ref_local_stats.shape
    
    if (0 <= idy < guide_imshape_y and
        0 <= idx < guide_imshape_x):

        d = 0
        
        ## Fetching flow
        local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) 
        if n_channels == 1:
            patch_idy = round(idy//(tile_size//2)) # guide scale is actually coarse scale
            patch_idx = round(idx//(tile_size//2))
            # guide image is coarse image : the flow stays the same
            local_flow[0] = flow[patch_idy, patch_idx, 0]
            local_flow[1] = flow[patch_idy, patch_idx, 1]
            
        else:
            patch_idy = round(2*idy//(tile_size//2)) # guide scale is 2 times sparser than coarse
            patch_idx = round(2*idx//(tile_size//2))
            
            # guide image is 2x smaller than coarse image : the flow must be divided by 2
            local_flow[0] = flow[patch_idy, patch_idx, 0]/2
            local_flow[1] = flow[patch_idy, patch_idx, 1]/2
        
        new_idx = round(idx + local_flow[0])
        new_idy = round(idy + local_flow[1])
    
        
        inbound = (0 <= new_idx < guide_imshape_x) and (0 <= new_idy < guide_imshape_y)
        
        if inbound :
            for channel in range(n_channels):
                dif = ref_local_stats[idy, idx, 0, channel] - comp_local_stats[new_idy, new_idx, 0, channel]
                d += dif * dif
            dist[idy, idx] = d
            
        else:
            dist[idy, idx] = +1/0 # + infinite distance will induce R = 0

def apply_noise_model(d, ref_local_stats, std_curve, diff_curve):
    """
    Applying noise model to update d^2 and sigma^2

    Parameters
    ----------
    d : device Array[guide_imshape_y, guide_imshape_x]
        squarred color distance between ref and compared image
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Local statistics of the ref image (required for fetching sigmas)
    std_curve : device Array
        Noise model for sigma
    diff_curve : device Array
        Moise model for d

    Returns
    -------
    d : device Array[guide_imshape_y, guide_imshape_x]
        updated version of the distancde (no copy is done, the original d is modified)
    sigma : device Array[guide_imshape_y, guide_imshape_x]
        Array that will contained the noise-corrected sigma value

    """
    *guide_imshape, _, _ = ref_local_stats.shape     
    sigma = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_apply_noise_model[blockspergrid, threadsperblock](d, sigma,
                                                           ref_local_stats,
                                                           std_curve, diff_curve)
    return d, sigma

@cuda.jit
def cuda_apply_noise_model(d, sigma, ref_local_stats, std_curve, diff_curve):
    idx, idy = cuda.grid(2)
    if (0 <= idy < ref_local_stats.shape[0] and
        0 <= idx < ref_local_stats.shape[1]):
        
        # sigma squarred is the sum of the 3 colour sigma squared
        sigma_ms = (ref_local_stats[idy, idx, 1, 0] +
                    ref_local_stats[idy, idx, 1, 1] +
                    ref_local_stats[idy, idx, 1, 2])
    
        brightness = (ref_local_stats[idy, idx, 0, 0] +
                      ref_local_stats[idy, idx, 0, 1] +
                      ref_local_stats[idy, idx, 0, 2])/3
        
        id_noise = round(1000 *brightness) # id on the noise curve
        d_md =  diff_curve[id_noise]
        sigma_md = std_curve[id_noise]
        
        # Wiener shrinkage
        # the formula is slightly different than in the article, because
        # d squared is updated instead of d.
        d[idy, idx] = d[idy, idx]**2/(d[idy, idx] + d_md**2)
        sigma[idy, idx] = max(sigma_ms, sigma_md)
    
@cuda.jit
def compute_s(flows, M_th, s1, s2, S):
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
    S : device Array[guide_imshape_y, guide_imshape_x]
        Map where s1 or s2 will be written at each position.

    Returns
    -------
    None.

    """
    tx, ty = cuda.threadIdx.x - 1, cuda.threadIdx.y - 1 # 3 by 3 neigbhorhood
    patch_idy, patch_idx = cuda.blockIdx.x, cuda.blockIdx.y
    
    n_patch_y, n_patch_x, _ = flows.shape
    
    
    mini = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    maxi = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    mini[0] = +1/0
    mini[1] = +1/0
    maxi[0] = -1/0
    maxi[1] = -1/0
    
    y = patch_idy + ty
    x = patch_idx + tx
    
    inbound = (0 <= x < n_patch_x and 0 <= y < n_patch_y)

    if inbound:
        flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #local array, each threads manipulates a different flow
        flow[0] = flows[y, x, 0]
        flow[1] = flows[y, x, 1]
        
        #local max search
        cuda.atomic.max(maxi, 0, flow[0])
        cuda.atomic.max(maxi, 1, flow[1])
        #local min search
        cuda.atomic.min(mini, 0, flow[0])
        cuda.atomic.min(mini, 1, flow[1])
        
    cuda.syncthreads()    
    if tx == 0 and ty == 0:
        if (maxi[0] - mini[0])**2 + (maxi[1] - mini[1])**2 > M_th**2:
            S[patch_idy, patch_idx] = s1
        else:
            S[patch_idy, patch_idx] = s2

def robustness_treshold(d, sigma, S, t):
    guide_imshape = d.shape 
    R = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = int(np.ceil(R.shape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(R.shape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_robustness_treshold[blockspergrid, threadsperblock](d, sigma, S, t, R)
    
    return R
    
@cuda.jit    
def cuda_robustness_treshold(d, sigma, S, t, R):
    idx, idy = cuda.grid(2)
    n_patchs_y, n_patchs_x = S.shape
    guide_imshape_y, guide_imshape_x = d.shape
    
    if (0 <= idy < R.shape[0] and
        0 <= idx < R.shape[1]):
    
        # fetching patch id (for S). I use a cross product to avoid
        # adding unnecessary arguments to the function. Note that it works with 1 or 3 channels
        patch_idy = round(idy * n_patchs_y / guide_imshape_y)
        patch_idx = round(idx * n_patchs_x / guide_imshape_x)
    
        
        R[idy, idx] = clamp(S[patch_idy, patch_idx]*exp(-d[idy, idx]/sigma[idy, idx]) - t,
                            0, 1)

def local_min(R):
    """
    For each pixel of R, the minimum in a 5 by 5 window is estimated in parallel
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
    blockspergrid_x = int(np.ceil(R.shape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(R.shape[0]/threadsperblock[0]))
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
    for i in range(-5, 6):
        y = clamp(idy + i, 0, guide_imshape_y)
        for j in range(-5, 6):
            x = clamp(idx + j, 0, guide_imshape_x)
            mini = min(mini, R[y, x])
    


    r[idy, idx] = mini
        