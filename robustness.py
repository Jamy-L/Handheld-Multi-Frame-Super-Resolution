# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow, get_closest_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_kernel_cov
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
    def compute_guide_patchs(ref_img, comp_imgs, flows, guide_patch_ref, guide_patch_comp):
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x -1, cuda.threadIdx.y -1
        
        top_left_ref_y = pixel_idy*2 +2*ty
        top_left_ref_x = pixel_idx*2 +2*tx
        
        flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        get_closest_flow(top_left_ref_x, top_left_ref_y, flows[image_index], tile_size, imsize, flow)
        
        if (0 <= top_left_ref_y < imshape_y -1) and (0 <= top_left_ref_x < imshape_x -1): # ref inbounds
            # ref
            guide_patch_ref[ty + 1, tx + 1, 0] = ref_img[top_left_ref_y + 1 ,
                                                         top_left_ref_x + 1] #r
            guide_patch_ref[ty + 1, tx + 1, 1] = (ref_img[top_left_ref_y, top_left_ref_x + 1] +
                                                  ref_img[top_left_ref_y + 1, top_left_ref_x])/2#g
            guide_patch_ref[ty + 1, tx + 1, 2] = ref_img[top_left_ref_y,
                                                         top_left_ref_x] #b   
            
            
            # Moving. We divide flow by 2 because rgb image is twice smaller
            top_left_m_x = round(top_left_ref_x + flow[0]/2)
            top_left_m_x = top_left_m_x - top_left_m_x%2
            # This transformation insures that we are indeed arriving on the right color channel (Bayer top left is generally red or blue)
            
            top_left_m_y = round(top_left_ref_y + flow[1]/2)
            top_left_m_y = top_left_m_y - top_left_m_y%2

            
            if (0 <= top_left_m_y < imshape_y) and (0 <= top_left_m_x < imshape_x):
                guide_patch_comp[ty + 1, tx + 1, 2] = comp_imgs[image_index, top_left_m_y, top_left_m_x] # b
            else:
                guide_patch_comp[ty + 1, tx + 1, 2] = 0/0 #Nan
                
            
            if (0 <= top_left_m_y + 1< imshape_y) and (0 <= top_left_m_x + 1< imshape_x):
                guide_patch_comp[ty + 1, tx + 1, 0] = comp_imgs[image_index, top_left_m_y + 1, top_left_m_x + 1] # r
            else:
                guide_patch_comp[ty + 1, tx + 1, 0] = 0/0 #Nan
                
            if (0 <= top_left_m_y < imshape_y) and (0 <= top_left_m_x + 1< imshape_x):
                g1 = comp_imgs[image_index, top_left_m_y, top_left_m_x + 1] # g1
            else:
                g1 = 0/0 #Nan
            
            if (0 <= top_left_m_y + 1< imshape_y) and (0 <= top_left_m_x < imshape_x):
                g2 = comp_imgs[image_index, top_left_m_y + 1, top_left_m_x] # g2
            else:
                g2 = 0 / 0  # Nan

            if isnan(g1) and isnan(g2):
                guide_patch_comp[ty + 1, tx + 1, 1] = 0/0
            elif isnan(g1):
                guide_patch_comp[ty + 1, tx + 1, 1] = g2
            elif isnan(g2):
                guide_patch_comp[ty + 1, tx + 1, 1] = g1
            else:
                guide_patch_comp[ty + 1, tx + 1, 1] = (g1 + g2)/2
            
            
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
        top_left_y = pixel_idy*2 +2*ty
        top_left_x = pixel_idx*2 +2*tx
        
        if 0 <= top_left_x < imshape_x and 0 <= top_left_y < imshape_y: #inbounds
            flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) #local array, each threads manipulates a different flow
            get_closest_flow(top_left_x, top_left_y, flows[image_index], tile_size, imsize, flow)
            
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
        
        compute_guide_patchs(ref_img, comp_imgs, flows, guide_patch_ref, guide_patch_comp)
        local_stats_ref = cuda.shared.array((2, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE) #mu, sigma for rgb
        local_stats_comp = cuda.shared.array((2,3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        # init 0 
        if ty < 2:
            local_stats_ref[ty, tx] = 0
            local_stats_comp[ty, tx] = 0
            
        cuda.syncthreads()
        
        maxi = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) 
        mini = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        M = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
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
            dp[tx] = dp[tx]*(dp[tx]**2/(dp[tx]**2 + dt**2))
            
            if tx == 0: # single threaded section
                # init, we take one flow vector from the patch
                flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
                get_closest_flow(pixel_idx*2, pixel_idy*2, flows[image_index], tile_size, imsize, flow)
                
                maxi[0] = flow[0]
                maxi[1] = flow[1]
                mini[0] = flow[2]
                mini[1] = flow[3]
            
        cuda.syncthreads()
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
    
    return r

@cuda.jit(device=True)
def fetch_robustness(pos_x, pos_y,image_index, R, local_R):
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    downscaled_posx = int(pos_x//2)
    downscaled_posy = int(pos_y//2)
    
    if tx == 0 and ty >=0: # TODO Neirest neighboor is made here. Maybe bilinear interpolation is better ?
        local_R[ty] = max(0, R[image_index, downscaled_posy, downscaled_posx, ty])
    
    
    
# %% test code, to remove in the final version

# raw_ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N000.dng'
#                         )
# ref_img = raw_ref_img.raw_image.copy()


# comp_images = rawpy.imread(
#     'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
# for i in range(2, 10):
#     comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
#                                                       ).raw_image.copy()[None], axis=0)

# pre_alignment = np.load(
#     'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')

# n_images, n_patch_y, n_patch_x, _ = pre_alignment.shape
# tile_size1 = 32
# native_im_size = ref_img.shape

# params = {'tuning': {'tileSizes': 32}, 'scale': 1}
# params['tuning']['kanadeIter'] = 3
# t = 0.12
# s1 = 12
# s2 = 2
# Mt = 0.8

# sigma_t = 2
# dt = 15




# t1 = time()
# final_alignments = lucas_kanade_optical_flow(
#     ref_img, comp_images, pre_alignment, {"verbose": 3}, params)

# output = compute_robustness(ref_img, comp_images, final_alignments, sigma_t, dt, Mt, s1, s2, t, 32)
# print('\nTotal ellapsed time : ', time() - t1)