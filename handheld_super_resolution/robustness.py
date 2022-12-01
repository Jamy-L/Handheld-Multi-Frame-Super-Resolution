# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

@author: jamyl
"""
from time import time
from math import isnan, sqrt, exp

import numpy as np
from numba import uint8, uint16, float32, float64, jit, njit, cuda, int32

from .optical_flow import get_closest_flow
from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, clamp

def init_robustness(ref_img,options, params):
    imsize = imshape_y, imshape_x = ref_img.shape
    
    bayer_mode = params['mode']=='bayer'
    VERBOSE = options['verbose']
    r_on = params['on']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    

    if r_on :         
        if VERBOSE > 1:
            current_time = time()
            print("Estimating ref image local stats")
            if VERBOSE > 2:
                print("- Decimating images to RGB")
                
        if params["mode"]=='bayer':
            rgb_imshape_y, rgb_imshape_x = int(imshape_y/2), int(imshape_x/2)
        else:
            rgb_imshape_y, rgb_imshape_x = imshape_y, imshape_x

        rgb_imshape = (rgb_imshape_y, rgb_imshape_x)
                
        # decimating images to RGB
        # TODO this takes too long.
        if params["mode"]=='bayer':
            cuda_ref_rgb_img = cuda.device_array(rgb_imshape + (3,), DEFAULT_NUMPY_FLOAT_TYPE)
            decimate_to_rgb[rgb_imshape, (2, 2)](ref_img, cuda_ref_rgb_img, CFA_pattern)
            
            ref_local_stats = cuda.device_array(rgb_imshape + (2, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
        else:
            ref_rgb_img = ref_img[None].transpose((1,2,0))
            cuda_ref_rgb_img = cuda.to_device(ref_rgb_img)
            
            ref_local_stats = cuda.device_array(rgb_imshape + (2, 1), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
    
        # Computing local stats (before applying optical flow)
        # 2 channels for mu, sigma
        
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Image decimated')
            
        compute_local_stats[rgb_imshape, (3, 3)](
            cuda_ref_rgb_img,
            ref_local_stats)
        
        cuda.synchronize()
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Local stats estimated')
        return ref_local_stats
    
    
def compute_robustness(comp_img, ref_local_stats, flows, options, params):
    """
    Returns the robustnesses of all compared images, based on the provided flow.

    Parameters
    ----------
    ref_img : numpy Array[imsize_y, imsize_x]
        ref raw image.
    comp_imgs : Array[n_images, imsize_y, imsize_x]
        Compared raw images.
    flows : device Array[n_images, n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows
    options : dict
        options to pass
    params : dict
        parameters

    Returns
    -------
    R : device Array[n_images, imsize_y/2, imsize_x/2]
        Robustness map for every compared image, sampled at the center of
        every bayer quad (only used for debugging)
    r : device Array[n_images, imsize_y/2, imsize_x/2]
        locally minimized Robustness map for every compared image, sampled at the center of
        every bayer quad
    """
    imsize = imshape_y, imshape_x = comp_img.shape

    
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
    
    if params["mode"]=='bayer':
        rgb_imshape_y, rgb_imshape_x = int(imshape_y/2), int(imshape_x/2)
        n_channels = 3
    else:
        rgb_imshape_y, rgb_imshape_x = imshape_y, imshape_x
        n_channels = 1

    rgb_imshape = (rgb_imshape_y, rgb_imshape_x)
          
    if r_on : 
        r = cuda.device_array((rgb_imshape_y, rgb_imshape_x), DEFAULT_NUMPY_FLOAT_TYPE)
        R = cuda.device_array((rgb_imshape_y, rgb_imshape_x), DEFAULT_NUMPY_FLOAT_TYPE)
        
        if VERBOSE > 1:
            current_time = time()
            print("Estimating Robustness")
        # decimating images to RGB
        # TODO this takes too long.
        if params["mode"]=='bayer':
            cuda_comp_rgb_img = cuda.device_array(rgb_imshape + (3,), DEFAULT_NUMPY_FLOAT_TYPE)
            decimate_to_rgb[rgb_imshape, (2, 2)](comp_img, cuda_comp_rgb_img, CFA_pattern)
            
            
            comp_local_stats = cuda.device_array(rgb_imshape+(2, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
        else:
            comp_rgb_img = comp_img[None].transpose((1,2,3,0)) # adding dimension 1 for single channel
            cuda_comp_rgb_img = cuda.to_device(comp_rgb_img)
            comp_local_stats = cuda.device_array(rgb_imshape + (2, 1), dtype=DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
            


        # Computing local stats (before applying optical flow)
        # 2 channels for mu, sigma
        
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Image decimated to rgb')
            
        compute_local_stats[rgb_imshape, (3, 3)](cuda_comp_rgb_img,
                                                 comp_local_stats)
        
        cuda.synchronize()
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Local stats estimated')
        
        # computing d
        # TODO this is taking too long
        d = cuda.device_array(rgb_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        cuda_compute_patch_dist[rgb_imshape, (n_channels)](
            ref_local_stats, comp_local_stats, flows, tile_size, d)
        
        cuda.synchronize()
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Estimated color distances')
        
        # leveraging the noise model
        sigma = cuda.device_array(rgb_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        cuda_apply_noise_model[rgb_imshape, (1)](
            d, sigma, ref_local_stats, cuda_std_curve, cuda_diff_curve)
        
        cuda.synchronize()
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Applied noise model')
        
        # applying flow discontinuity penalty
        S = cuda.device_array((n_patch_y, n_patch_x), DEFAULT_NUMPY_FLOAT_TYPE)
        compute_s[S.shape, (3, 3)](flows, Mt, s1, s2, S)
        
        cuda.synchronize()
        if VERBOSE > 2 :
            current_time = getTime(
                current_time, ' - Flow irregularities registered')
        
        
        cuda_compute_robustness[R.shape, (1)](d, sigma, S, t, R)
        
        cuda.synchronize()
        if VERBOSE > 2:
            current_time = getTime(
                current_time, ' - Robustness Estimated')
        
        compute_local_min[rgb_imshape, (5, 5)](R, r)
        cuda.synchronize()
        if VERBOSE > 2:
            current_time = getTime(
                current_time, ' - Robustness locally minimized')
    else: 
        temp = np.ones(rgb_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        r = cuda.to_device(temp)
        R = cuda.to_device(temp)
    return S, r

@cuda.jit
def decimate_to_rgb(raw_img, rgb_img, CFA):
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    idy, idx = cuda.blockIdx.x, cuda.blockIdx.y
    
    if tx==0 and ty==0:
        rgb_img[idy, idx, 1]=0 # 0 init for green
    cuda.syncthreads()    
    
    channel = CFA[ty, tx]
    if channel == 1: # green
        cuda.atomic.add(rgb_img, (idy, idx, 1), raw_img[idy*2 + ty, idx*2 + tx])
    else:
        rgb_img[idy, idx, channel]= raw_img[idy*2 + ty, idx*2 + tx]
        
    cuda.syncthreads()
    if tx==0 and ty==0:
        rgb_img[idy, idx, 1]/=2
    
    
@cuda.jit
def compute_local_stats(rgb_img, local_stats):
    """
    Computes the mean color and variance associated for each 3 by 3 patches

    Parameters
    ----------
    ref_rgb_img : shared Array[imsize_y/2, imsize_x/2, 3]
        ref rgb image.
    comp_rgb_imgs : shared Array[n_images, imsize_y/2, imsize_x/2, 3]
        compared images. 
    ref_local_stats : shared Array[n_images, imsize_y/2, imsize_x/2, 2, 3]
        empty array that will contain mu and sigma² for the ref image
    comp_local_stats : shared Array[n_images, imsize_y/2, imsize_x/2, 3]
        empty array that will contain mu for the compared image


    """
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    idy, idx = cuda.blockIdx.x, cuda.blockIdx.y
    channels = rgb_img.shape[2] # 1 or 3
    # single threaded zeros init
    if tx == 0 and ty ==0:
        for chan in range(channels):
            local_stats[idy, idx, 0, chan] = 0
            local_stats[idy, idx, 1, chan] = 0


    cuda.syncthreads()
    thread_idy = clamp(idy + ty -1, 0, rgb_img.shape[0]-1)
    thread_idx = clamp(idx + tx -1, 0, rgb_img.shape[1]-1)

    for chan in range(channels):
        thread_value = rgb_img[thread_idy, thread_idx, chan]
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
    Computes the map of d**2 based on the map of µ

    Parameters
    ----------
    ref_local_stats : Device Array[tgb_imshape_y, rgb_imshape_x, 2, 3]
        mu, sigma map for rgb (or grey) for reference image
    comp_local_stats : Device Array[tgb_imshape_y, rgb_imshape_x, 2, 3]
        mu, sigma map for rgb (or grey) for compared image
    flow : TYPE
        DESCRIPTION.
    tile_size : TYPE
        DESCRIPTION.
    dist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    idy, idx = cuda.blockIdx.x, cuda.blockIdx.y
    channel = cuda.threadIdx.x
    imsize = ref_local_stats.shape[:-1]
    channels = cuda.blockDim.x # number of channels 
    
    d = cuda.shared.array(3, DEFAULT_CUDA_FLOAT_TYPE) # we may use only 1 coeff
    d[channel] = 0
    
    ## Fetching flow
    local_flow = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE) # local array would work too, but shared memory is faster
    if channels == 1:
        patch_idy = round(idy//(tile_size//2)) # "rgb" scale is actually coarse scale
        patch_idx = round(idx//(tile_size//2))
    else:
        patch_idy = round(2*idy//(tile_size//2)) # rgb scale is 2 times sparser
        patch_idx = round(2*idx//(tile_size//2))
    local_flow[0] = flow[patch_idy, patch_idx, 0]
    local_flow[1] = flow[patch_idy, patch_idx, 1]
    # get_closest_flow(idx, idy, flow, tile_size, imsize, local_flow)
    # tile_size is defined in term of grey pixels. In bayer case, the grey pixel density is the same as the rgb decimated density : 4 times less than bayer
    # In the grey case, all densities are equal.
    
    new_idx = round(idx + local_flow[0])
    new_idy = round(idy + local_flow[1])
    # TODO rgb values to be compared are not proprely known. Nearest neighboor interpolation
    # is a possibility, but bilinear may be better
    
    inbound = (0 <= new_idx < imsize[1]) and (0 <= new_idy < imsize[0])
    if inbound : 
        d[channel] = ref_local_stats[idy, idx, 0, channel] - comp_local_stats[new_idy, new_idx, 0, channel]
    else:
        d[channel] = +1/0 # + infinite distance will induce R = 0
    
    if channels == 1:
        dist[idy, idx] = d[0]*d[0]
    else:
        cuda.syncthreads()
        dist[idy, idx] = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
        
@cuda.jit
def cuda_apply_noise_model(d, sigma, ref_local_stats, std_curve, diff_curve):
    idy, idx = cuda.blockIdx.x, cuda.blockIdx.y
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
    # I am manipulating d squared instead of d.
    d[idy, idx] = d[idy, idx]**2/(d[idy, idx] + d_md**2)
    sigma[idy, idx] = max(sigma_ms, sigma_md)
    
@cuda.jit
def compute_s(flows, M_th, s1, s2, S):
    """ Computes s at avery position based on flow irregularities
    

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
    S : device Array[n_images, rgb_imshape_y, rgb_imshape_x]
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
def compute_local_min(R, r):
    """
    For each pixel of R, the minimum in a 5 by 5 window is estimated in parallel
    and stored in r.

    Parameters
    ----------
    R : Array[n_images, imsize_y/2, imsize_x/2]
        Robustness map for every image
    r : Array[n_images, imsize_y/2, imsize_x/2]
        locally minimised version of R

    Returns
    -------
    None.

    """
    rgb_imshape_y, rgb_imshape_x = R.shape
    
    
    
    pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x - 2, cuda.threadIdx.y - 2

    mini = cuda.shared.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    if tx == 0 and ty == 0:
        mini[0] = 1/0
    cuda.syncthreads()
    
    #local min search
    if 0 <= pixel_idx + tx < rgb_imshape_x and 0 <= pixel_idy + ty < rgb_imshape_y : #inbound
        cuda.atomic.min(mini, 0, R[pixel_idy + ty, pixel_idx + tx])
    
    cuda.syncthreads()
    if tx==0 and ty==0 :
        r[pixel_idy, pixel_idx] = mini[0]

@cuda.jit    
def cuda_compute_robustness(d, sigma, S, t, R):
    idy, idx = cuda.blockIdx.x, cuda.blockIdx.y
    n_patch_y, n_patch_x = S.shape
    rgb_imshape_y, rgb_imshaep_x = d.shape
    
    # fetching patch id (for S). I use a cross product to avoid
    # adding unnecessary arguments to the function. Not that it works with 1 or 3 channels
    patch_idy = round(idy * n_patch_y / rgb_imshape_y)
    patch_idx = round(idy * n_patch_y / rgb_imshape_y)

    
    R[idy, idx] = clamp(S[patch_idy, patch_idx]*exp(-d[idy, idx]/sigma[idy, idx]) - t,
                        0, 1)
        