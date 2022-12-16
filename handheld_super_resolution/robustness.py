# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

@author: jamyl
"""
import time
from math import exp

import numpy as np
from numba import cuda, uint8

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, clamp

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
            guide_ref_img = cuda.device_array(guide_imshape + (3,), DEFAULT_NUMPY_FLOAT_TYPE)
            compute_guide_image[guide_imshape, (2, 2)](ref_img, guide_ref_img, CFA_pattern)
            
            ref_local_stats = cuda.device_array(guide_imshape + (2, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
        else:
            guide_ref_img = ref_img[:, :, None] # Adding 1 channel
            
            ref_local_stats = cuda.device_array(guide_imshape + (2, 1), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
    
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Image decimated')
            
        compute_local_stats[guide_imshape, (3, 3)](
            guide_ref_img,
            ref_local_stats)
        
    
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
    imshape = imshape_y, imshape_x = comp_img.shape

    
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
    
    # moving noise model to GPU
    cuda_std_curve = cuda.to_device(params['std_curve'])
    cuda_diff_curve = cuda.to_device(params['diff_curve'])
    
    if bayer_mode:
        guide_imshape = int(imshape_y/2), int(imshape_x/2)
        n_channels = 3
    else:
        guide_imshape = imshape_y, imshape_x
        n_channels = 1
          
    if r_on : 
        r = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        R = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
        if VERBOSE > 1:
            current_time = time.perf_counter()
            print("Estimating Robustness")
            
        # Computing guide image
        if bayer_mode:
            guide_img = cuda.device_array(guide_imshape + (3,), DEFAULT_NUMPY_FLOAT_TYPE)
            
            threadsperblock = (16, 16) # maximum, we may take less
            blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
            blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            compute_guide_image[blockspergrid, threadsperblock](comp_img, guide_img, CFA_pattern)
            
            
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
            
        compute_local_stats[guide_imshape, (3, 3)](guide_img,
                                                 comp_local_stats)
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Local stats estimated')
        
        # computing d
        threadsperblock = (16, 16, 1) # maximum, we may take less
        blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
        blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
        blockspergrid_z = n_channels
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        
        d = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        cuda_compute_patch_dist[blockspergrid, threadsperblock](
            ref_local_stats, comp_local_stats, flows, tile_size, d)
        
        
        if VERBOSE > 2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Estimated color distances')
        
        # leveraging the noise model
        sigma = cuda.device_array(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
        threadsperblock = (16, 16) # maximum, we may take less
        blockspergrid_x = int(np.ceil(guide_imshape[1]/threadsperblock[1]))
        blockspergrid_y = int(np.ceil(guide_imshape[0]/threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
            
        cuda_apply_noise_model[blockspergrid, threadsperblock](
            d, sigma, ref_local_stats, cuda_std_curve, cuda_diff_curve)
        
        
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
        
        threadsperblock = (16, 16) # maximum, we may take less
        blockspergrid_x = int(np.ceil(R.shape[1]/threadsperblock[1]))
        blockspergrid_y = int(np.ceil(R.shape[0]/threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        cuda_compute_robustness[blockspergrid, threadsperblock](d, sigma, S, t, R)
        
        if VERBOSE > 2:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Robustness Estimated')
        
        compute_local_min[guide_imshape, (5, 5)](R, r)
        
        if VERBOSE > 2:
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Robustness locally minimized')
    else: 
        temp = np.ones(guide_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        r = cuda.to_device(temp)
    return r

@cuda.jit
def compute_guide_image(raw_img, guide_img, CFA):
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
    
    
@cuda.jit
def compute_local_stats(guide_img, local_stats):
    """
    Computes the mean color and variance associated for each 3 by 3 patches

    Parameters
    ----------
    guide_img : device Array[guide_imshape_y, guide_imshape_x, channels]
        ref rgb image.

    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        empty array that will contain mu and sigma² for the ref img


    """
    guide_imshape_y, guide_imshape_x, channels = guide_img.shape
    
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    idy, idx = cuda.blockIdx.x, cuda.blockIdx.y

    # single threaded zeros init
    if tx == 0 and ty ==0:
        for chan in range(channels):
            local_stats[idy, idx, 0, chan] = 0
            local_stats[idy, idx, 1, chan] = 0


    cuda.syncthreads()
    thread_idy = clamp(idy + ty -1, 0, guide_imshape_y-1)
    thread_idx = clamp(idx + tx -1, 0, guide_imshape_x-1)

    for chan in range(channels):
        thread_value = guide_img[thread_idy, thread_idx, chan]
        cuda.atomic.add(local_stats, (idy, idx, 0, chan), thread_value)
        cuda.atomic.add(local_stats, (idy, idx, 1, chan), thread_value**2)
    cuda.syncthreads()
    if ty == 0 and tx <= channels:
        # normalizing
        local_stats[idy, idx, 0, tx] /= 9 # one thread for each color channel = no racing condition
        local_stats[idy, idx, 1, tx] = local_stats[idy, idx, 1, tx]/9 -  local_stats[idy, idx, 0, tx]**2
        
@cuda.jit
def cuda_compute_patch_dist(ref_local_stats, comp_local_stats, flow, tile_size, dist):
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
    dist : Device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Empty array that will contain the distances

    Returns
    -------
    None.

    """
    idy, idx, channel = cuda.grid(3)
    
    if (0 <= idy < dist.shape[0] and
        0 <= idx < dist.shape[1]):

        guide_imshape_y, guide_imshape_x, _, n_channels = ref_local_stats.shape
    
        
        d = cuda.shared.array(3, DEFAULT_CUDA_FLOAT_TYPE) # we may use only 1 coeff
        d[channel] = 0
        
        ## Fetching flow
        local_flow = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE) # local array would work too, but shared memory is faster
        if n_channels == 1:
            patch_idy = round(idy//(tile_size//2)) # guide scale is actually coarse scale
            patch_idx = round(idx//(tile_size//2))
        else:
            patch_idy = round(2*idy//(tile_size//2)) # guide scale is 2 times sparser than coarse
            patch_idx = round(2*idx//(tile_size//2))
            
        local_flow[0] = flow[patch_idy, patch_idx, 0]
        local_flow[1] = flow[patch_idy, patch_idx, 1]
    
        
        new_idx = round(idx + local_flow[0])
        new_idy = round(idy + local_flow[1])
    
        
        inbound = (0 <= new_idx < guide_imshape_x) and (0 <= new_idy < guide_imshape_y)
        
        if inbound : 
            d[channel] = ref_local_stats[idy, idx, 0, channel] - comp_local_stats[new_idy, new_idx, 0, channel]
        else:
            d[channel] = +1/0 # + infinite distance will induce R = 0
        
        if n_channels == 1:
            dist[idy, idx] = d[0]*d[0]
        else:
            cuda.syncthreads()
            dist[idy, idx] = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
        
@cuda.jit
def cuda_apply_noise_model(d, sigma, ref_local_stats, std_curve, diff_curve):
    """
    Applying noise model to update d^2 and sigma^2

    Parameters
    ----------
    d : device Array[guide_imshape_y, guide_imshape_x]
        squarred color distance between ref and compared image
    sigma : device Array[guide_imshape_y, guide_imshape_x]
        Empty array that will contained the noise-corrected sigma value
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Local statistics of the ref image (required for fetching sigmas)
    std_curve : device Array
        Noise model for sigma
    diff_curve : device Array
        Moise model for d

    Returns
    -------
    None.

    """
    idy, idx = cuda.grid(2)
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

@cuda.jit    
def cuda_compute_robustness(d, sigma, S, t, R):
    idy, idx = cuda.grid(2)
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
    
@cuda.jit
def compute_local_min(R, r):
    """
    For each pixel of R, the minimum in a 5 by 5 window is estimated in parallel
    and stored in r.

    Parameters
    ----------
    R : Array[guide_imshape_y, guide_imshape_x]
        Robustness map for every image
    r : Array[guide_imshape_y, guide_imshape_x]
        locally minimised version of R

    Returns
    -------
    None.

    """
    guide_imshape_y, guide_imshape_x = R.shape
    
    
    
    pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x - 2, cuda.threadIdx.y - 2

    mini = cuda.shared.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    if tx == 0 and ty == 0:
        mini[0] = 1/0
    cuda.syncthreads()
    
    #local min search
    if 0 <= pixel_idx + tx < guide_imshape_x and 0 <= pixel_idy + ty < guide_imshape_y : #inbound
        cuda.atomic.min(mini, 0, R[pixel_idy + ty, pixel_idx + tx])
    
    cuda.syncthreads()
    if tx==0 and ty==0 :
        r[pixel_idy, pixel_idx] = mini[0]
        