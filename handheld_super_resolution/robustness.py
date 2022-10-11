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

def compute_robustness(ref_img, comp_imgs, flows, options, params):
    """
    Returns the robustnesses of all compared images, based on the provided flow.

    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        ref image.
    comp_imgs : Array[n_images, imsize_y, imsize_x]
        Compared images.
    flows : Array[n_images, n_patchs_y, n_patchs_y, 2]
        optical flows
    options : dict
        options to pass
    params : dict
        parameters

    Returns
    -------
    R : Array[n_images, imsize_y/2, imsize_x/2, 3]
        Robustness map for every image, for the r, g and b channels
    """
    n_images, imshape_y, imshape_x = comp_imgs.shape
    imsize = (imshape_y, imshape_x)
    
    bayer_mode = params['mode']=='bayer'
    VERBOSE = options['verbose']
    
    CFA_pattern = params['exif']['CFA Pattern']
    
    tile_size = params['tuning']["tileSize"]
    t = params['tuning']["t"]
    s1 = params['tuning']["s1"]
    s2 = params['tuning']["s2"]
    Mt = params['tuning']["Mt"]
    
    # moving noise model to GPU
    cuda_std_curve = cuda.to_device(params['std_curve'])
    cuda_diff_curve = cuda.to_device(params['diff_curve'])
    
    if params["mode"]=='bayer':
        rgb_imshape_y, rgb_imshape_x = int(imshape_y/2), int(imshape_x/2)
        
        r = cuda.device_array((n_images, rgb_imshape_y, rgb_imshape_x, 3))
        R = cuda.device_array((n_images, rgb_imshape_y, rgb_imshape_x, 3))
    else:
        rgb_imshape_y, rgb_imshape_x = imshape_y, imshape_x

        
        r = cuda.device_array((n_images, rgb_imshape_y, rgb_imshape_x, 1))
        R = cuda.device_array((n_images, rgb_imshape_y, rgb_imshape_x, 1))
    rgb_imshape = (rgb_imshape_y, rgb_imshape_x)
    
    
    @cuda.jit(device=True)
    def get_channel(patch_pixel_idx, patch_pixel_idy):
        """
        Return 0, 1 or 2 depending if the coordinates point a red, green or
        blue pixel on the Bayer frame

        Parameters
        ----------
        patch_pixel_idx : unsigned int
            horizontal coordinates
        patch_pixel_idy : unigned int
            vertical coordinates

        Returns
        -------
        int

        """
        return uint8(CFA_pattern[patch_pixel_idy%2, patch_pixel_idx%2])
    
    
    
    @cuda.jit(device=True)
    def compute_guide_patchs(ref_img, comp_imgs, flows, bayer_mode, guide_patch_ref, guide_patch_comp):
        """
        Computes the guide patch (the position is ruled by the cuda block ids)

        Parameters
        ----------
        ref_img : shared Array[imsize_y, imsize_x]
            ref image.
        comp_imgs : shrred Array[n_images, imsize_y, imsize_x]
            compared images.
        flows : shared Array[n_images, n_patchs_y, n_patchs_y, 2]
            optical flows
        guide_patch_ref : Shared Array[3, 3]
            empty array which will contain the guide for ref img
        guide_patch_comp : shared Array[3, 3]
            empty array which will contain the guide for comp img


        """
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x -1, cuda.threadIdx.y -1
        if bayer_mode : 
            top_left_ref_y = pixel_idy*2 +2*ty # top left bayer pixel of the grey cell
            top_left_ref_x = pixel_idx*2 +2*tx
            
            flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
            # coordinates are given in bayer referential, so tile size is doubled
            get_closest_flow(top_left_ref_x, top_left_ref_y, flows[image_index], tile_size*2, imsize, flow)
            if (0 <= top_left_ref_y < imshape_y -1) and (0 <= top_left_ref_x < imshape_x -1): # ref inbounds
                # We need to init because green is accumulating
                guide_patch_ref[ty + 1, tx + 1, 0] = 0 
                guide_patch_ref[ty + 1, tx + 1, 1] = 0
                guide_patch_ref[ty + 1, tx + 1, 2] = 0
                
                guide_patch_comp[ty + 1, tx + 1, 0] = 0 
                guide_patch_comp[ty + 1, tx + 1, 1] = 0
                guide_patch_comp[ty + 1, tx + 1, 2] = 0
            
            
                # ref
                for i in range(2):
                    for j in range(2):
                        channel = get_channel(j, i)
                        # This accumulation is single-threaded. No need to use cuda.atomic.add because there is no racing condition
                        guide_patch_ref[ty + 1, tx + 1, channel] += ref_img[top_left_ref_y + i ,
                                                                            top_left_ref_x + j]
                        
                guide_patch_ref[ty + 1, tx + 1, 1]/=2 # averaging the green contribution
                
                
                # Moving. We divide flow by 2 because rgb image is twice smaller then bayer
                top_left_m_x = round(top_left_ref_x + flow[0]/2)
                top_left_m_y = round(top_left_ref_y + flow[1]/2)
    
                for i in range(2):
                    for j in range(2):
                        channel = get_channel(top_left_m_x + j, top_left_m_y + i)
                        if (0 <= top_left_m_y + i < imshape_y) and (0 <= top_left_m_x + j < imshape_x):
                            guide_patch_comp[ty + 1, tx + 1, channel] += comp_imgs[image_index, top_left_m_y + i, top_left_m_x + j]
                        else:
                            guide_patch_comp[ty + 1, tx + 1, channel] = 0/0 #Nan
    
                guide_patch_comp[ty + 1, tx + 1, 1] /= 2 # Averaging greens 
                
                
            else: #out of bounds
                guide_patch_comp[ty + 1, tx + 1, 0] = 0/0 #Nan
                guide_patch_comp[ty + 1, tx + 1, 1] = 0/0 #Nan
                guide_patch_comp[ty + 1, tx + 1, 2] = 0/0 #Nan
                
                guide_patch_comp[ty + 1, tx + 1, 0] = 0/0 #Nan
                guide_patch_comp[ty + 1, tx + 1, 1] = 0/0 #Nan
                guide_patch_comp[ty + 1, tx + 1, 2] = 0/0 #Nan
        else:
            # grey mode, we simply extract a 3x3 neighborhood
            ref_y = pixel_idy + ty # coordinates of the corresponding grey pixel
            ref_x = pixel_idx + tx
            
            flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
            # coordinates are given in rgb referential (and not bayer), so tile size is not doubled
            get_closest_flow(ref_x, ref_y, flows[image_index], tile_size, rgb_imshape, flow)
            # Moving. We do not divide flow by 2 because image was already grey
            top_left_m_x = round(ref_x + flow[0])
            top_left_m_y = round(ref_y + flow[1])
            
            guide_patch_ref[ty + 1, tx + 1, 0] = ref_img[ref_y, ref_x]
            guide_patch_comp[ty + 1, tx + 1, 0] = comp_imgs[image_index, top_left_m_y, top_left_m_x]
            
        

    @cuda.jit(device=True)
    def compute_local_stats(guide_patch_ref, guide_patch_comp,
                            local_stats_ref, local_stats_comp):
        """
        Computes the distance and variance associated with the 2 patches

        Parameters
        ----------
        ref_img : shared Array[imsize_y, imsize_x]
            ref image.
        comp_imgs : shrred Array[n_images, imsize_y, imsize_x]
            compared images.
        flows : shared Array[n_images, n_patchs_y, n_patchs_y, 2]
            optical flows
        local_stats_ref : shared Array[2]
            empty array that will contain mu and sigma for the ref image
        local_stats_comp : shared Array[2]
            empty Array that will contain mu and sigma for the compared image


        """
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        for chan in range(guide_patch_ref.shape[2]): # might be 3 for bayer and 1 for grey images
            if not(isnan(guide_patch_ref[ty, tx, chan])):
                cuda.atomic.add(local_stats_ref, (0, chan), guide_patch_ref[ty, tx, chan])
                cuda.atomic.add(local_stats_ref, (1, chan), guide_patch_ref[ty, tx, chan]**2)
                
            if not(isnan(guide_patch_comp[ty, tx, chan])):
                cuda.atomic.add(local_stats_comp, (0, chan), guide_patch_comp[ty, tx, chan])
                cuda.atomic.add(local_stats_comp, (1, chan), guide_patch_comp[ty, tx, chan]**2)
        
        # TODO when everybody is Nan (because patchs are out of bound) stats
        # are staying at 0. Maybe it's not that bad because we are capping with the 
        # model's values later
    
    @cuda.jit(device=True)
    def compute_m(flows, mini, maxi, bayer_mode, M):
        """
        Computes Mx and My based on the flows 

        Parameters
        ----------
        flows : shared Array[n_images, n_patchs_y, n_patchs_x, 6]
            optical flows
        mini : shared Array[2]
            empty shared array used for parallel computation of min 
        maxi : shared Array[2]
            empty shared array used for parallel computation of max 
        M : shared Array[2]
            empty array that will contain Mx and My.


        """
        tx, ty = cuda.threadIdx.x - 1, cuda.threadIdx.y - 1
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        y = pixel_idy + ty
        x = pixel_idx + tx
        if bayer_mode : 
            inbound = (0 <= x < imshape_x/2 and 0 <= y < imshape_y/2) # grey imsg twice smaller than bayer
        else:
            inbound = (0 <= x < imshape_x and 0 <= y < imshape_y)
        
        if inbound:
            flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #local array, each threads manipulates a different flow
            get_closest_flow(x, y, flows[image_index], tile_size, imsize, flow)# x and y are on grey scale and tile_size is expressed in grey pixels
            
            #local max search
            cuda.atomic.max(maxi, 0, flow[0])
            cuda.atomic.max(maxi, 1, flow[1])
            #local min search
            cuda.atomic.min(mini, 0, flow[0])
            cuda.atomic.min(mini, 1, flow[1])
            
        cuda.syncthreads()    
        if tx == 0 and ty == 0:
            M[0] = maxi[0] - mini[0]
            M[1] = maxi[1] - mini[1]
    
    
    @cuda.jit
    def cuda_compute_robustness(ref_img, comp_imgs, flows, cuda_diff_curve, cuda_std_curve, bayer_mode, R):
        """
        Computes robustness in parralel. Each bloc, computes one coefficient of R,
        and is made of 9 threads.

        Parameters
        ----------
        ref_img : shared Array[imsize_y, imsize_x]
            ref image.
        comp_imgs : shared Array[n_images, imsize_y, imsize_x]
            Compared images.
        flows : shared Array[n_images, n_patchs_y, n_patchs_y, 2]
            optical flows
        cuda_diff_curve : Array[10001]
            Expected difference based on the noise model, for 1001 different
            brightness measurements.
        cuda_std_curve : Array[10001]
            Expected standart deviation based on the noise model, for 1001 different
            brightness measurements.
        R : Array[n_images, imsize_y/2, imsize_x/2, 3]
            Robustness map for every image, for the r, g and b channels
        

        """
        if bayer_mode : 
            guide_patch_ref = cuda.shared.array((3, 3, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
            guide_patch_comp = cuda.shared.array((3, 3, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        else:
            guide_patch_ref = cuda.shared.array((3, 3, 1), dtype=DEFAULT_CUDA_FLOAT_TYPE)
            guide_patch_comp = cuda.shared.array((3, 3, 1), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        
        # For each block, the ref guide patch and the matching compared guide image are computed
        compute_guide_patchs(ref_img, comp_imgs, flows, bayer_mode, guide_patch_ref, guide_patch_comp)
        
        local_stats_ref = cuda.shared.array((2, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE) #mu, sigma for rgb
        local_stats_comp = cuda.shared.array((2,3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        maxi = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #Max Vx, Vy
        mini = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # Min Vx, Vy
        M = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #Mx, My
        # multithreaded inits
        if ty < 2:
            local_stats_ref[ty, tx] = 0
            local_stats_comp[ty, tx] = 0
            
        if tx == 0 and ty==2: # single threaded section
            
            maxi[0] = -np.inf 
            maxi[1] = -np.inf  
            mini[0] = np.inf
            mini[1] = np.inf
            
            
            
        cuda.syncthreads()

        
        compute_local_stats(guide_patch_ref, guide_patch_comp,
                            local_stats_ref, local_stats_comp)
        cuda.syncthreads()
        
        dp = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        sigma = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        if ty ==0 and bayer_mode:
            # normalizing
            local_stats_ref[0, tx] /= 9 # one thread for each color channel = no racing condition
            local_stats_ref[1, tx] = sqrt(local_stats_ref[1, tx]/9 -  local_stats_ref[0, tx]**2)
            
            local_stats_comp[0, tx] /= 9
            local_stats_comp[1, tx] = sqrt(local_stats_comp[1, tx]/9 -  local_stats_comp[0, tx]**2)
            
            # mapping the brightness from [0, 1] to the related index on the noise model curve
            id_noise = round(1000 *local_stats_comp[0, tx])
            
            # fetching noise model values
            dt = cuda_diff_curve[id_noise]  
            sigma_t = cuda_std_curve[id_noise]
            
            dp[tx] = abs(local_stats_ref[0, tx] - local_stats_comp[0, tx])
            # noise correction
            sigma[tx] = max(sigma_t, local_stats_comp[1, tx])
            dp[tx] = dp[tx]*(dp[tx]**2/(dp[tx]**2 + dt**2))
            
        elif ty==0 and tx==0 and not bayer_mode:
            # normalizing
            local_stats_ref[0, 0] /= 9
            local_stats_ref[1, 0] = sqrt(local_stats_ref[1, 0]/9 -  local_stats_ref[0, 0]**2)
            
            local_stats_comp[0, 0] /= 9
            local_stats_comp[1, 0] = sqrt(local_stats_comp[1, 0]/9 -  local_stats_comp[0, 0]**2)
            
            # mapping the brightness from [0, 1] to the related index on the noise model curve
            id_noise = round(1000 *local_stats_comp[0, 0])
            
            # fetching noise model values
            dt = cuda_diff_curve[id_noise]  
            sigma_t = cuda_std_curve[id_noise]
            
            dp[0] = abs(local_stats_ref[0, 0] - local_stats_comp[0, 0])
            # noise correction
            sigma[0] = max(sigma_t, local_stats_comp[1, 0])
            dp[0] = dp[0]*(dp[0]**2/(dp[0]**2 + dt**2))
        

        compute_m(flows, mini, maxi, bayer_mode, M)
        cuda.syncthreads()
        
        if ty == 0 and bayer_mode:
            if sqrt(M[0]**2 + M[1]**2) > Mt:
                R[image_index, pixel_idy, pixel_idx, tx] = clamp(s1*exp(-dp[tx]**2/sigma[tx]**2) - t, 0, 1)
            else:
                R[image_index, pixel_idy, pixel_idx, tx] = clamp(s2*exp(-dp[tx]**2/sigma[tx]**2) - t, 0, 1)
        
        elif ty == 0 and tx == 0 and not bayer_mode :
            if sqrt(M[0]**2 + M[1]**2) > Mt:
                R[image_index, pixel_idy, pixel_idx, 0] = clamp(s1*exp(-dp[0]**2/sigma[0]**2) - t, 0, 1)
            else:
                R[image_index, pixel_idy, pixel_idx, 0] = clamp(s1*exp(-dp[0]**2/sigma[0]**2) - t, 0, 1)
        
    @cuda.jit
    def compute_local_min(R, bayer_mode, r):
        """
        For each pixel of R, the minimum in a 5 by 5 window is estimated in parallel
        and stored in r.

        Parameters
        ----------
        R : Array[n_images, imsize_y/2, imsize_x/2, 3]
            Robustness map for every image, for the r, g and b channels
        r : Array[n_images, imsize_y/2, imsize_x/2, 3]
            locally minimised version of R

        Returns
        -------
        None.

        """
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x - 2, cuda.threadIdx.y - 2
        if bayer_mode : 
            mini = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        else : 
            mini = cuda.shared.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        if tx == 0 and ty >= 0 and bayer_mode:
            mini[ty] = 1/0 # + inf
        elif not bayer_mode and tx == 0 and ty == 0:
            mini[0] = 1/0
            
        cuda.syncthreads()
        
        #local min search
        if 0 <= pixel_idx + tx < rgb_imshape_x and 0<= pixel_idy + ty < rgb_imshape_y : #inbound
            cuda.atomic.min(mini, 0, R[image_index, pixel_idy + ty, pixel_idx + tx, 0])
            if bayer_mode : 
                cuda.atomic.min(mini, 1, R[image_index, pixel_idy + ty, pixel_idx + tx, 1])
                cuda.atomic.min(mini, 2, R[image_index, pixel_idy + ty, pixel_idx + tx, 2])
        
        cuda.syncthreads()
        if tx == 0 and ty >= 0 and bayer_mode:
            r[image_index, pixel_idy, pixel_idx, ty] = mini[ty]
        elif tx==0 and ty==0 and not bayer_mode:
            r[image_index, pixel_idy, pixel_idx, 0] = mini[0]
        
        
        
        
    if VERBOSE > 1:
        current_time = time()
        print("Estimating Robustness")
        
    cuda_compute_robustness[(n_images, rgb_imshape_y, rgb_imshape_x), (3, 3)
        ](ref_img, comp_imgs, flows,cuda_diff_curve, cuda_std_curve, bayer_mode, R)
    
    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Robustness Estimated')
    
    compute_local_min[(n_images, rgb_imshape_y, rgb_imshape_x), (5, 5)](R, bayer_mode, r)
    
    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Robustness locally minimized')
    return R, r

@cuda.jit(device=True)
def fetch_robustness(pos_x, pos_y,image_index, R, channel):

    downscaled_posx = round(pos_x//2)
    downscaled_posy = round(pos_y//2)
    
    # TODO Neirest neighboor is made here. Maybe bilinear interpolation is better ?
    return max(0, R[image_index, downscaled_posy, downscaled_posx, channel])

        
        
        
