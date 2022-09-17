# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow, get_closest_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime

from linalg import quad_mat_prod

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
from time import time
from math import isnan, sqrt, exp

DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

def compute_robustness(ref_img, comp_imgs, flows, options, params):
    n_images, imshape_y, imshape_x = comp_imgs.shape
    imsize = ref_img.shape
    
    VERBOSE = options['verbose']
    
    CFA_pattern = params['exif']['CFA Pattern']
    
    tile_size = params['tuning']["tileSizes"]
    t = params['tuning']["t"]
    s1 = params['tuning']["s1"]
    s2 = params['tuning']["s2"]
    Mt = params['tuning']["Mt"]
    sigma_t = params['tuning']["sigma_t"]
    dt = params['tuning']["dt"]
    
    
    
    rgb_imshape_y, rgb_imshape_x = int(imshape_y/2), int(imshape_x/2)
    
    r = cuda.device_array((n_images, rgb_imshape_y, rgb_imshape_x, 3))
    R = cuda.device_array((n_images, rgb_imshape_y, rgb_imshape_x, 3))
    
    
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
    def compute_guide_patchs(ref_img, comp_imgs, flows, guide_patch_ref, guide_patch_comp):
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x -1, cuda.threadIdx.y -1
        
        top_left_ref_y = pixel_idy*2 +2*ty
        top_left_ref_x = pixel_idx*2 +2*tx
        
        flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        get_closest_flow(top_left_ref_x, top_left_ref_y, flows[image_index], tile_size, imsize, flow)
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
            
            
            # Moving. We divide flow by 2 because rgb image is twice smaller
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
            
        

    @cuda.jit(device=True)
    def compute_local_stats(guide_patch_ref, guide_patch_comp,
                            local_stats_ref, local_stats_comp):
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        for chan in range(3):
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
    def compute_m(flows, mini, maxi, M):
        tx, ty = cuda.threadIdx.x - 1, cuda.threadIdx.y - 1
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        y = pixel_idy*2 + ty
        x = pixel_idx*2 + tx
        
        if 0 <= x < imshape_x and 0 <= y < imshape_y: #inbounds
            flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #local array, each threads manipulates a different flow
            get_closest_flow(x, y, flows[image_index], tile_size, imsize, flow)
            
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
    def cuda_compute_robustness(ref_img, comp_imgs, flows, R):
        guide_patch_ref = cuda.shared.array((3, 3, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        guide_patch_comp = cuda.shared.array((3, 3, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        
        # For each block, the ref guide patch and the matching compared guide image are computed
        compute_guide_patchs(ref_img, comp_imgs, flows, guide_patch_ref, guide_patch_comp)
        
        local_stats_ref = cuda.shared.array((2, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE) #mu, sigma for rgb
        local_stats_comp = cuda.shared.array((2,3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        maxi = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #Max Vx, Vy
        mini = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # Min Vx, Vy
        M = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #Mx, My
        # multithreade inits
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
        if ty ==0:
            # normalizing
            local_stats_ref[0, tx] /= 9 # one thread for one color channel
            local_stats_ref[1, tx] = sqrt(local_stats_ref[1, tx]/9 -  local_stats_ref[0, tx]**2)
            
            
            local_stats_comp[0, tx] /= 9
            local_stats_comp[1, tx] = sqrt(local_stats_comp[1, tx]/9 -  local_stats_comp[0, tx]**2)
            
            dp[tx] = abs(local_stats_ref[0, tx] - local_stats_comp[0, tx])
            # noise correction
            sigma[tx] = max(sigma_t, local_stats_comp[1, tx])
            dp[tx] = dp[tx] # *(dp[tx]**2/(dp[tx]**2 + dt**2))
        
        

        compute_m(flows, mini, maxi, M)
        cuda.syncthreads()
        
        if ty == 0:
            if sqrt(M[0]**2 + M[1]**2) > Mt:
                R[image_index, pixel_idy, pixel_idx, tx] = s1*exp(-dp[tx]**2/sigma[tx]**2) - t
            else:
                R[image_index, pixel_idy, pixel_idx, tx] = s2*exp(-dp[tx]**2/sigma[tx]**2) - t
        
    @cuda.jit
    def compute_local_min(R, r):
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x - 2, cuda.threadIdx.y - 2
        mini = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        if tx == 0 and ty >= 0 :
            mini[ty] = 1/0 # + inf
            
        cuda.syncthreads()
        #local min search
        if 0 <= pixel_idx + tx < rgb_imshape_x and 0<= pixel_idy + ty < imshape_y : #inbound
            cuda.atomic.min(mini, 0, R[image_index, pixel_idy, pixel_idx, 0])
            cuda.atomic.min(mini, 1, R[image_index, pixel_idy, pixel_idx, 1])
            cuda.atomic.min(mini, 2, R[image_index, pixel_idy, pixel_idx, 2])
        
        cuda.syncthreads()
        if tx == 0 and ty >= 0 :
            r[image_index, pixel_idy, pixel_idx, ty] = mini[ty]
        
    if VERBOSE > 2:
        current_time = time()
        print("Estimating Robustness")
        
    cuda_compute_robustness[(n_images, rgb_imshape_y, rgb_imshape_x), (3, 3)
        ](ref_img, comp_imgs, flows,R)
    
    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Robustness Estimated')
    
    compute_local_min[(n_images, rgb_imshape_y, rgb_imshape_x), (5, 5)](R,r)
    
    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Robustness locally minimized')
    
    return R, r

@cuda.jit(device=True)
def fetch_robustness(pos_x, pos_y,image_index, R, channel):
    downscaled_posx = int(pos_x//2)
    downscaled_posy = int(pos_y//2)
    
    # TODO Neirest neighboor is made here. Maybe bilinear interpolation is better ?
    return max(0, R[image_index, downscaled_posy, downscaled_posx, channel])

        
        
        